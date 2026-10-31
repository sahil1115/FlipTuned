"""OS, hardware and ML-package detection.

Never imports torch or mlx unless they are installed, and never fails because a probe fails:
every probe degrades to "unknown" with a note.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

import psutil
from pydantic import BaseModel, Field

OsName = Literal["windows", "macos", "linux"]
GIB = 1024**3  # sizes are reported in binary GB, as the OS shows them

TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "trl",
    "accelerate",
    "bitsandbytes",
    "mlx",
    "mlx-lm",
)


class GpuInfo(BaseModel):
    name: str
    memory_total_gb: float | None = None
    memory_free_gb: float | None = None
    driver_version: str | None = None
    compute_capability: str | None = None  # e.g. "8.6"


class MachineInfo(BaseModel):
    os_name: OsName
    os_version: str
    arch: str  # normalized: x86_64 | arm64 | other raw value
    python_version: str
    python_arch: str
    emulated: bool = Field(description="Python runs under Rosetta or x64-on-ARM emulation.")
    cpu_name: str
    cpu_count_logical: int
    ram_gb: float
    nvidia_smi_found: bool = False
    nvidia_gpus: list[GpuInfo] = Field(default_factory=list)
    apple_chip: str | None = None
    unified_memory_gb: float | None = None
    windows_long_paths_enabled: bool | None = None
    notes: list[str] = Field(default_factory=list)


class TorchInfo(BaseModel):
    installed: bool
    version: str | None = None
    build: Literal["cpu", "cuda", "rocm", "unknown"] | None = None
    cuda_available: bool = False
    cuda_runtime: str | None = None
    device_name: str | None = None
    bf16_supported: bool | None = None
    error: str | None = None


class MlxInfo(BaseModel):
    installed: bool
    mlx_version: str | None = None
    mlx_lm_version: str | None = None
    metal_available: bool | None = None
    error: str | None = None


def current_os() -> OsName:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def normalize_arch(machine: str) -> str:
    value = machine.lower()
    if value in ("amd64", "x86_64", "x64"):
        return "x86_64"
    if value in ("arm64", "aarch64", "armv8", "armv8l"):
        return "arm64"
    return value or "unknown"


def package_versions(names: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str | None]:
    """Installed versions from package metadata (no imports)."""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def detect_machine() -> MachineInfo:
    os_name = current_os()
    notes: list[str] = []
    native_arch = normalize_arch(platform.machine())
    python_arch = native_arch
    emulated = False
    apple_chip = None
    unified = None
    long_paths = None
    ram_gb = round(psutil.virtual_memory().total / GIB, 1)

    if os_name == "macos":
        emulated = _sysctl("sysctl.proc_translated") == "1"
        if emulated:
            native_arch = "arm64"
        apple_chip = _sysctl("machdep.cpu.brand_string") if native_arch == "arm64" else None
        if native_arch == "arm64":
            unified = ram_gb
        os_version = f"macOS {platform.mac_ver()[0] or platform.release()}"
        cpu_name = apple_chip or _sysctl("machdep.cpu.brand_string") or platform.processor()
    elif os_name == "windows":
        identifier = os.environ.get("PROCESSOR_IDENTIFIER", "")
        if python_arch == "x86_64" and "ARM" in identifier.upper():
            emulated, native_arch = True, "arm64"
        os_version, long_paths = _windows_version_and_long_paths()
        cpu_name = _windows_cpu_name() or identifier or platform.processor()
    else:
        os_version = _linux_pretty_name() or f"Linux {platform.release()}"
        cpu_name = _linux_cpu_name() or platform.processor() or "unknown"

    smi_found, gpus, smi_note = _nvidia_smi()
    if smi_note:
        notes.append(smi_note)

    return MachineInfo(
        os_name=os_name,
        os_version=os_version,
        arch=native_arch,
        python_version=platform.python_version(),
        python_arch=python_arch,
        emulated=emulated,
        cpu_name=(cpu_name or "unknown").strip(),
        cpu_count_logical=psutil.cpu_count(logical=True) or 1,
        ram_gb=ram_gb,
        nvidia_smi_found=smi_found,
        nvidia_gpus=gpus,
        apple_chip=apple_chip,
        unified_memory_gb=unified,
        windows_long_paths_enabled=long_paths,
        notes=notes,
    )


def probe_torch() -> TorchInfo:
    version = package_versions(("torch",))["torch"]
    if version is None or importlib.util.find_spec("torch") is None:
        return TorchInfo(installed=False)
    info = TorchInfo(installed=True, version=version)
    local = version.partition("+")[2]
    if local.startswith("cu"):
        info.build = "cuda"
    elif local.startswith("rocm"):
        info.build = "rocm"
    elif local == "cpu":
        info.build = "cpu"
    try:
        import torch  # noqa: PLC0415 - optional, imported only when installed

        info.cuda_runtime = getattr(torch.version, "cuda", None)
        if info.build is None:
            info.build = "cuda" if info.cuda_runtime else "unknown"
        info.cuda_available = bool(torch.cuda.is_available())
        if info.cuda_available:
            info.device_name = torch.cuda.get_device_name(0)
            info.bf16_supported = bool(torch.cuda.is_bf16_supported())
    except Exception as exc:  # noqa: BLE001 - a broken install must not crash doctor
        info.error = f"{type(exc).__name__}: {exc}"[:300]
    return info


def probe_mlx(machine: MachineInfo) -> MlxInfo:
    versions = package_versions(("mlx", "mlx-lm"))
    info = MlxInfo(
        installed=versions["mlx-lm"] is not None,
        mlx_version=versions["mlx"],
        mlx_lm_version=versions["mlx-lm"],
    )
    if not info.installed or machine.os_name != "macos" or machine.python_arch != "arm64":
        return info
    try:
        import mlx.core as mx  # noqa: PLC0415 - optional, macOS arm64 only

        info.metal_available = bool(mx.metal.is_available())
    except Exception as exc:  # noqa: BLE001
        info.error = f"{type(exc).__name__}: {exc}"[:300]
    return info


def hf_cache_dir(env: dict[str, str] | None = None) -> Path:
    """Hugging Face hub cache location, computed the way huggingface_hub does (no import)."""
    env = dict(os.environ) if env is None else env
    if env.get("HF_HUB_CACHE"):
        return Path(env["HF_HUB_CACHE"]).expanduser()
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"]).expanduser() / "hub"
    base = (
        Path(env["XDG_CACHE_HOME"]).expanduser()
        if env.get("XDG_CACHE_HOME")
        else Path.home() / ".cache"
    )
    return base / "huggingface" / "hub"


def free_disk_gb(path: Path) -> tuple[Path, float | None]:
    """Free space on the drive holding ``path``, measured at the drive/root, not inside ``path``."""
    anchor = Path(path.anchor) if path.anchor else Path("/")
    try:
        return anchor, round(shutil.disk_usage(anchor).free / GIB, 1)
    except OSError:
        return anchor, None


# --------------------------------------------------------------------------- helpers


def _run(argv: list[str], timeout: float = 10.0) -> str | None:
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _sysctl(key: str) -> str | None:
    out = _run(["sysctl", "-n", key])
    return out.strip() if out else None


def _nvidia_smi() -> tuple[bool, list[GpuInfo], str | None]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return False, [], None
    out = _run(
        [
            exe,
            "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if out is None:
        return True, [], "nvidia-smi was found but did not report any GPU"
    return True, parse_nvidia_smi(out), None


def parse_nvidia_smi(text: str) -> list[GpuInfo]:
    gpus = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5 or not parts[0]:
            continue
        gpus.append(
            GpuInfo(
                name=parts[0],
                memory_total_gb=_mib_to_gb(parts[1]),
                memory_free_gb=_mib_to_gb(parts[2]),
                driver_version=parts[3] or None,
                compute_capability=parts[4] or None,
            )
        )
    return gpus


def _mib_to_gb(value: str) -> float | None:
    try:
        return round(float(value) / 1024, 1)
    except ValueError:
        return None


def _windows_version_and_long_paths() -> tuple[str, bool | None]:
    release = platform.release()
    build = platform.version()
    display = ""
    long_paths = None
    try:
        import winreg  # noqa: PLC0415 - Windows only

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion"
        ) as key:
            display = str(winreg.QueryValueEx(key, "DisplayVersion")[0])
            product = str(winreg.QueryValueEx(key, "ProductName")[0])
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem"
        ) as key:
            long_paths = bool(winreg.QueryValueEx(key, "LongPathsEnabled")[0])
    except OSError:
        product = ""
    # Windows 11 still reports ProductName "Windows 10"; the build number tells them apart.
    try:
        build_number = int(build.split(".")[-1])
    except ValueError:
        build_number = 0
    name = f"Windows {'11' if build_number >= 22000 else release}"
    if product and "Windows" not in name:
        name = product
    suffix = f" {display}" if display else ""
    return f"{name}{suffix} ({build})", long_paths


def _windows_cpu_name() -> str | None:
    try:
        import winreg  # noqa: PLC0415 - Windows only

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        ) as key:
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0])
    except OSError:
        return None


def _linux_pretty_name() -> str | None:
    try:
        return platform.freedesktop_os_release().get("PRETTY_NAME")
    except OSError:
        return None


def _linux_cpu_name() -> str | None:
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.lower().startswith(("model name", "hardware")):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None
