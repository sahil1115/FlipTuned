"""``doctor``: environment report and backend recommendation (SPEC §6, §5.2).

Never opens, creates or lists the app home or any data directory. Free disk is measured at
the drive root of the app home, not inside it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from fliptuned import __version__
from fliptuned.config.app_home import resolve_app_home
from fliptuned.config.loader import load_config
from fliptuned.config.models import Backend, Device, Precision, TrainConfig
from fliptuned.diagnostics.backend_select import BackendReport, EnvironmentProbe, build_report
from fliptuned.platform import detect

MIN_FREE_DISK_GB = 40.0


class DoctorRequest(BaseModel):
    home: str | None = Field(default=None, description="App home override (--home).")
    train_config: str | None = Field(
        default=None,
        description="Training config whose backend/device/precision should be checked.",
    )


class DiskInfo(BaseModel):
    label: str
    path: str
    drive: str
    free_gb: float | None


class DoctorReport(BaseModel):
    fliptuned_version: str
    machine: detect.MachineInfo
    packages: dict[str, str | None]
    torch: detect.TorchInfo
    mlx: detect.MlxInfo
    app_home: str
    app_home_source: str
    hf_cache: str
    disks: list[DiskInfo]
    requested: dict[str, str]
    backend: BackendReport
    warnings: list[str]


def default_probe() -> EnvironmentProbe:
    machine = detect.detect_machine()
    packages = detect.package_versions()
    return EnvironmentProbe(
        machine=machine,
        torch=detect.probe_torch(),
        mlx=detect.probe_mlx(machine),
        bitsandbytes_version=packages.get("bitsandbytes"),
    )


def run_doctor(
    request: DoctorRequest,
    *,
    probe_fn: Callable[[], EnvironmentProbe] | None = None,
    packages_fn: Callable[[], dict[str, str | None]] | None = None,
    disk_fn: Callable[[Path], tuple[Path, float | None]] | None = None,
) -> DoctorReport:
    probe_fn = probe_fn or default_probe
    packages_fn = packages_fn or detect.package_versions
    disk_fn = disk_fn or detect.free_disk_gb
    resolved = resolve_app_home(request.home)
    backend, device, precision = Backend.AUTO, Device.AUTO, Precision.AUTO
    if request.train_config:
        config = load_config(Path(request.train_config), TrainConfig)
        backend, device, precision = config.backend, config.device, config.precision

    probe = probe_fn()
    report = build_report(probe, backend, device, precision)
    hf_cache = detect.hf_cache_dir()

    warnings: list[str] = []
    disks = []
    for label, path in (("App home", resolved.path), ("Hugging Face cache", hf_cache)):
        drive, free = disk_fn(path)
        disks.append(DiskInfo(label=label, path=str(path), drive=str(drive), free_gb=free))
        if free is not None and free < MIN_FREE_DISK_GB:
            warnings.append(
                f"{label} drive {drive} has {free} GB free; real fine-tuning needs about {MIN_FREE_DISK_GB:.0f} GB."
            )
    if probe.machine.emulated:
        warnings.append(
            "Python is running under emulation; performance and backend support are limited."
        )
    if probe.machine.windows_long_paths_enabled is False:
        warnings.append(
            "Windows long paths are disabled; deep model-cache paths can fail. "
            "See docs/platforms.md to enable them (needs admin)."
        )

    return DoctorReport(
        fliptuned_version=__version__,
        machine=probe.machine,
        packages=packages_fn(),
        torch=probe.torch,
        mlx=probe.mlx,
        app_home=str(resolved.path),
        app_home_source=resolved.source.value,
        hf_cache=str(hf_cache),
        disks=disks,
        requested={"backend": backend.value, "device": device.value, "precision": precision.value},
        backend=report,
        warnings=warnings + report.warnings,
    )
