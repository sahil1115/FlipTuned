"""Backend auto-selection (SPEC §5.2) and install suggestions, as pure functions of a probe.

No hardware access happens here: callers pass an :class:`EnvironmentProbe`, so every machine
profile can be tested with fakes.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from fliptuned.config.models import Backend, Device, Precision
from fliptuned.errors import NotAvailableError
from fliptuned.platform.detect import MachineInfo, MlxInfo, TorchInfo

MIN_NVIDIA_DRIVER_MAJOR = 580  # CUDA 13.x minor-version compatibility (decisions.md §3.4)
MIN_COMPUTE_CAPABILITY = 7.5  # cu130 PyTorch and bitsandbytes CUDA 13 wheels: Turing+

CMD_CPU = "uv sync --extra cpu --extra web"
CMD_CUDA = "uv sync --extra cuda --extra qlora --extra web"
CMD_MLX = "uv sync --extra mlx --extra cpu --extra web"


class EnvironmentProbe(BaseModel):
    machine: MachineInfo
    torch: TorchInfo
    mlx: MlxInfo
    bitsandbytes_version: str | None = None


class BackendOption(BaseModel):
    backend: Backend
    device: str  # "apple_gpu" | "cuda" | "cpu"
    available: bool
    reason: str


class BackendSelection(BaseModel):
    backend: Backend
    device: str
    precision: str
    reason: str


class BackendReport(BaseModel):
    selected: BackendSelection | None
    options: list[BackendOption]
    warnings: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    install_command: str | None


# --------------------------------------------------------------------------- install suggestion


def _driver_major(version: str | None) -> int | None:
    try:
        return int((version or "").split(".")[0])
    except ValueError:
        return None


def _compute_capability(value: str | None) -> float | None:
    try:
        return float(value) if value else None
    except ValueError:
        return None


def suggest_install(machine: MachineInfo) -> tuple[str | None, list[str]]:
    """The ``uv sync`` command for this machine, plus notes explaining it."""
    notes: list[str] = []
    if machine.os_name == "macos":
        if machine.arch != "arm64":
            notes.append("Intel Macs have no supported PyTorch build; FlipTuned can't train here.")
            return None, notes
        if machine.emulated:
            notes.append(
                "Python is running under Rosetta (x86_64). Install an arm64 Python, e.g. "
                "`uv python install 3.12` from an arm64 terminal, then recreate the environment."
            )
        return CMD_MLX, notes

    if machine.nvidia_gpus:
        gpu = machine.nvidia_gpus[0]
        driver = _driver_major(gpu.driver_version)
        cc = _compute_capability(gpu.compute_capability)
        if cc is not None and cc < MIN_COMPUTE_CAPABILITY:
            notes.append(
                f"{gpu.name} (compute capability {gpu.compute_capability}) is older than Turing; "
                "the CUDA 13 build doesn't support it. Using CPU. See docs/platforms.md for the cu126 path."
            )
            return CMD_CPU, notes
        if driver is not None and driver < MIN_NVIDIA_DRIVER_MAJOR:
            notes.append(
                f"NVIDIA driver {gpu.driver_version} is too old for the CUDA 13 build "
                f"(needs {MIN_NVIDIA_DRIVER_MAJOR}+). Update the driver, then run `{CMD_CUDA}`."
            )
            return CMD_CPU, notes
        return CMD_CUDA, notes

    if machine.nvidia_smi_found:
        notes.append("nvidia-smi is installed but reported no GPU; using CPU.")
    return CMD_CPU, notes


# --------------------------------------------------------------------------- selection


def _cuda_precision(probe: EnvironmentProbe) -> tuple[str, str]:
    gpu = probe.machine.nvidia_gpus[0] if probe.machine.nvidia_gpus else None
    cc = _compute_capability(gpu.compute_capability) if gpu else None
    qlora_ok = (
        probe.bitsandbytes_version is not None
        and probe.machine.os_name != "macos"
        and (cc is None or cc >= MIN_COMPUTE_CAPABILITY)
    )
    if qlora_ok:
        return Precision.QLORA_4BIT.value, "bitsandbytes is installed, so 4-bit QLoRA is used"
    if probe.torch.bf16_supported:
        return Precision.BF16.value, "the GPU supports bf16"
    return Precision.FP16.value, "the GPU doesn't support bf16, so fp16 is used"


def list_options(probe: EnvironmentProbe) -> list[BackendOption]:
    m, t, x = probe.machine, probe.torch, probe.mlx
    options = []

    if m.os_name != "macos" or m.arch != "arm64":
        mlx_reason = "MLX needs a Mac with Apple Silicon"
        mlx_ok = False
    elif m.emulated:
        mlx_reason, mlx_ok = "Python runs under Rosetta; MLX needs native arm64 Python", False
    elif not x.installed:
        mlx_reason, mlx_ok = f"mlx-lm is not installed (run `{CMD_MLX}`)", False
    elif x.metal_available is False:
        mlx_reason, mlx_ok = "mlx is installed but Metal is not available", False
    else:
        mlx_reason, mlx_ok = f"mlx-lm {x.mlx_lm_version} on Apple Silicon", True
    options.append(
        BackendOption(backend=Backend.MLX, device="apple_gpu", available=mlx_ok, reason=mlx_reason)
    )

    if t.cuda_available:
        cuda_ok, cuda_reason = (
            True,
            f"PyTorch {t.version} with CUDA {t.cuda_runtime} on {t.device_name}",
        )
    elif m.nvidia_gpus and t.installed and t.build != "cuda":
        cuda_ok, cuda_reason = False, f"NVIDIA GPU found but PyTorch is CPU-only (run `{CMD_CUDA}`)"
    elif m.nvidia_gpus and not t.installed:
        cuda_ok, cuda_reason = (
            False,
            f"NVIDIA GPU found but PyTorch is not installed (run `{CMD_CUDA}`)",
        )
    elif m.nvidia_gpus:
        cuda_ok, cuda_reason = (
            False,
            "PyTorch has CUDA support but can't use the GPU (check the driver)",
        )
    else:
        cuda_ok, cuda_reason = False, "no NVIDIA GPU found"
    options.append(
        BackendOption(backend=Backend.HF_PEFT, device="cuda", available=cuda_ok, reason=cuda_reason)
    )

    if t.installed and not t.error:
        cpu_ok, cpu_reason = True, f"PyTorch {t.version} on CPU (slow; small models only)"
    elif t.installed:
        cpu_ok, cpu_reason = False, f"PyTorch is installed but failed to load: {t.error}"
    else:
        cmd, _ = suggest_install(m)
        cpu_ok, cpu_reason = False, f"PyTorch is not installed (run `{cmd or CMD_CPU}`)"
    options.append(
        BackendOption(backend=Backend.HF_PEFT, device="cpu", available=cpu_ok, reason=cpu_reason)
    )
    return options


def select_backend(
    probe: EnvironmentProbe,
    backend: Backend = Backend.AUTO,
    device: Device = Device.AUTO,
    precision: Precision = Precision.AUTO,
) -> BackendSelection:
    """Apply SPEC §5.2. Raises :class:`NotAvailableError` for an explicit choice this machine lacks."""
    options = {(o.backend, o.device): o for o in list_options(probe)}
    mlx = options[(Backend.MLX, "apple_gpu")]
    cuda = options[(Backend.HF_PEFT, "cuda")]
    cpu = options[(Backend.HF_PEFT, "cpu")]
    install_cmd, _ = suggest_install(probe.machine)

    def unavailable(option: BackendOption, what: str) -> NotAvailableError:
        return NotAvailableError(
            f"{what} isn't available on this machine: {option.reason}.",
            hint=f"Run `{install_cmd}` or choose 'auto'." if install_cmd else "Choose 'auto'.",
        )

    if backend is Backend.MLX:
        if not mlx.available:
            raise unavailable(mlx, "The MLX backend")
        return BackendSelection(
            backend=Backend.MLX,
            device="apple_gpu",
            precision="model (4-bit → QLoRA)",
            reason="MLX was requested",
        )
    if device is Device.CUDA:
        if not cuda.available:
            raise unavailable(cuda, "NVIDIA CUDA")
        return _cuda_selection(probe, precision, "CUDA was requested")
    if device is Device.CPU or (backend is Backend.HF_PEFT and not cuda.available):
        if not cpu.available:
            raise unavailable(cpu, "CPU training")
        return _cpu_selection(
            precision, "CPU was requested" if device is Device.CPU else "no NVIDIA GPU is available"
        )
    if backend is Backend.HF_PEFT:
        return _cuda_selection(
            probe, precision, "Transformers + PEFT was requested; an NVIDIA GPU is available"
        )

    # auto
    if mlx.available:
        return BackendSelection(
            backend=Backend.MLX,
            device="apple_gpu",
            precision="model (4-bit → QLoRA)",
            reason="Apple Silicon with MLX installed",
        )
    if cuda.available:
        return _cuda_selection(probe, precision, "an NVIDIA GPU with CUDA PyTorch is available")
    if cpu.available:
        return _cpu_selection(precision, "no GPU backend is available")
    raise NotAvailableError(
        "No training backend is available on this machine.",
        hint=f"Run `{install_cmd}`."
        if install_cmd
        else "See docs/platforms.md for supported machines.",
    )


def _cuda_selection(probe: EnvironmentProbe, precision: Precision, reason: str) -> BackendSelection:
    if precision is Precision.AUTO:
        value, why = _cuda_precision(probe)
        return BackendSelection(
            backend=Backend.HF_PEFT, device="cuda", precision=value, reason=f"{reason}; {why}"
        )
    if precision is Precision.QLORA_4BIT and probe.bitsandbytes_version is None:
        raise NotAvailableError(
            "4-bit QLoRA needs bitsandbytes, which isn't installed.", hint=f"Run `{CMD_CUDA}`."
        )
    if precision is Precision.BF16 and probe.torch.bf16_supported is False:
        raise NotAvailableError("This GPU doesn't support bf16.", hint="Use 'fp16' or 'auto'.")
    return BackendSelection(
        backend=Backend.HF_PEFT, device="cuda", precision=precision.value, reason=reason
    )


def _cpu_selection(precision: Precision, reason: str) -> BackendSelection:
    # CPU bf16 isn't verified on any machine yet (decisions.md C10), so CPU uses fp32.
    if precision not in (Precision.AUTO, Precision.FP32):
        raise NotAvailableError(
            f"Precision '{precision.value}' isn't supported for CPU training.",
            hint="Use 'fp32' or 'auto'.",
        )
    return BackendSelection(
        backend=Backend.HF_PEFT, device="cpu", precision=Precision.FP32.value, reason=reason
    )


def build_report(
    probe: EnvironmentProbe,
    backend: Backend = Backend.AUTO,
    device: Device = Device.AUTO,
    precision: Precision = Precision.AUTO,
) -> BackendReport:
    install_command, notes = suggest_install(probe.machine)
    options = list_options(probe)
    warnings = list(notes)
    suggestions: list[str] = []
    try:
        selected = select_backend(probe, backend, device, precision)
    except NotAvailableError as exc:
        selected = None
        warnings.append(exc.message)
        if exc.hint:
            suggestions.append(exc.hint)

    if selected and selected.device == "cpu":
        warnings.append(
            "Training will run on the CPU. That is slow and limited to small models "
            "(about 1B parameters by default)."
        )
    for option in options:
        if (
            not option.available
            and "(run `" in option.reason
            and (
                option.device == "cuda"
                and probe.machine.nvidia_gpus
                or option.backend is Backend.MLX
                and probe.machine.os_name == "macos"
            )
        ):
            suggestions.append(option.reason)
    if (
        selected
        and selected.device == "cuda"
        and probe.bitsandbytes_version is None
        and probe.machine.os_name != "macos"
    ):
        suggestions.append(f"Install bitsandbytes for 4-bit QLoRA (lower memory): `{CMD_CUDA}`")
    return BackendReport(
        selected=selected,
        options=options,
        warnings=warnings,
        suggestions=suggestions,
        install_command=install_command,
    )
