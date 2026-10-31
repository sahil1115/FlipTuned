"""Faked machine profiles for backend-selection and doctor tests."""

from __future__ import annotations

from fliptuned.diagnostics.backend_select import EnvironmentProbe
from fliptuned.platform.detect import GpuInfo, MachineInfo, MlxInfo, TorchInfo


def _machine(**overrides) -> MachineInfo:
    base = dict(
        os_name="linux",
        os_version="Ubuntu 24.04 LTS",
        arch="x86_64",
        python_version="3.12.11",
        python_arch="x86_64",
        emulated=False,
        cpu_name="Fake CPU",
        cpu_count_logical=8,
        ram_gb=32.0,
    )
    base.update(overrides)
    return MachineInfo(**base)


def mac_m1(*, mlx_installed: bool = True, emulated: bool = False) -> EnvironmentProbe:
    return EnvironmentProbe(
        machine=_machine(
            os_name="macos",
            os_version="macOS 15.3",
            arch="arm64",
            python_arch="x86_64" if emulated else "arm64",
            emulated=emulated,
            cpu_name="Apple M1",
            apple_chip="Apple M1",
            unified_memory_gb=16.0,
            ram_gb=16.0,
        ),
        torch=TorchInfo(installed=True, version="2.14.1", build="unknown"),
        mlx=MlxInfo(
            installed=mlx_installed,
            mlx_version="0.32.4" if mlx_installed else None,
            mlx_lm_version="0.32.2" if mlx_installed else None,
            metal_available=True if mlx_installed else None,
        ),
    )


def mac_intel() -> EnvironmentProbe:
    return EnvironmentProbe(
        machine=_machine(os_name="macos", os_version="macOS 13.6", cpu_name="Intel Core i7"),
        torch=TorchInfo(installed=False),
        mlx=MlxInfo(installed=False),
    )


def windows_nvidia(
    *,
    torch_build: str = "cuda",
    bitsandbytes: bool = True,
    driver: str = "581.15",
    compute: str = "8.9",
    bf16: bool = True,
) -> EnvironmentProbe:
    cuda = torch_build == "cuda"
    return EnvironmentProbe(
        machine=_machine(
            os_name="windows",
            os_version="Windows 11 23H2 (10.0.22631)",
            nvidia_smi_found=True,
            nvidia_gpus=[
                GpuInfo(
                    name="NVIDIA GeForce RTX 4070",
                    memory_total_gb=12.0,
                    memory_free_gb=11.2,
                    driver_version=driver,
                    compute_capability=compute,
                )
            ],
            windows_long_paths_enabled=True,
        ),
        torch=TorchInfo(
            installed=True,
            version="2.14.1+cu130" if cuda else "2.14.1+cpu",
            build=torch_build,
            cuda_available=cuda,
            cuda_runtime="13.0" if cuda else None,
            device_name="NVIDIA GeForce RTX 4070" if cuda else None,
            bf16_supported=bf16 if cuda else None,
        ),
        mlx=MlxInfo(installed=False),
        bitsandbytes_version="0.50.2" if bitsandbytes else None,
    )


def linux_cpu(*, torch_installed: bool = True) -> EnvironmentProbe:
    return EnvironmentProbe(
        machine=_machine(),
        torch=TorchInfo(installed=True, version="2.14.1+cpu", build="cpu")
        if torch_installed
        else TorchInfo(installed=False),
        mlx=MlxInfo(installed=False),
    )
