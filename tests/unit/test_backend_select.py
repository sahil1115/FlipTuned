from __future__ import annotations

import pytest

from fliptuned.config.models import Backend, Device, Precision
from fliptuned.diagnostics.backend_select import (
    CMD_CPU,
    CMD_CUDA,
    CMD_MLX,
    build_report,
    select_backend,
    suggest_install,
)
from fliptuned.errors import NotAvailableError

from .profiles import linux_cpu, mac_intel, mac_m1, windows_nvidia


def test_mac_m1_auto_selects_mlx() -> None:
    probe = mac_m1()
    sel = select_backend(probe)
    assert (sel.backend, sel.device) == (Backend.MLX, "apple_gpu")
    assert suggest_install(probe.machine)[0] == CMD_MLX


def test_mac_without_mlx_falls_back_to_cpu_and_suggests_mlx() -> None:
    report = build_report(mac_m1(mlx_installed=False))
    assert report.selected is not None and report.selected.device == "cpu"
    assert any(CMD_MLX in s for s in report.suggestions)


def test_mac_under_rosetta_disables_mlx_and_warns() -> None:
    report = build_report(mac_m1(emulated=True))
    assert report.selected is not None and report.selected.backend is Backend.HF_PEFT
    assert any("Rosetta" in w for w in report.warnings)


def test_intel_mac_has_no_install_command() -> None:
    command, notes = suggest_install(mac_intel().machine)
    assert command is None and any("Intel" in n for n in notes)
    assert build_report(mac_intel()).selected is None


def test_windows_nvidia_auto_selects_cuda_qlora() -> None:
    probe = windows_nvidia()
    sel = select_backend(probe)
    assert (sel.backend, sel.device, sel.precision) == (Backend.HF_PEFT, "cuda", "qlora_4bit")
    assert suggest_install(probe.machine)[0] == CMD_CUDA


def test_windows_nvidia_without_bitsandbytes_uses_bf16_and_suggests_qlora() -> None:
    report = build_report(windows_nvidia(bitsandbytes=False))
    assert report.selected is not None and report.selected.precision == "bf16"
    assert any("bitsandbytes" in s for s in report.suggestions)


def test_gpu_without_bf16_uses_fp16() -> None:
    sel = select_backend(windows_nvidia(bitsandbytes=False, bf16=False, compute="7.5"))
    assert sel.precision == "fp16"


def test_nvidia_with_cpu_only_torch_suggests_cuda_extra() -> None:
    report = build_report(windows_nvidia(torch_build="cpu"))
    assert report.selected is not None and report.selected.device == "cpu"
    assert any("PyTorch is CPU-only" in s and CMD_CUDA in s for s in report.suggestions)


def test_old_driver_and_old_gpu_fall_back_to_cpu_install() -> None:
    command, notes = suggest_install(windows_nvidia(driver="552.22").machine)
    assert command == CMD_CPU and any("580" in n for n in notes)
    command, notes = suggest_install(windows_nvidia(compute="6.1").machine)
    assert command == CMD_CPU and any("Turing" in n for n in notes)


def test_linux_cpu_only_selects_cpu_fp32_with_warning() -> None:
    probe = linux_cpu()
    report = build_report(probe)
    assert report.selected is not None
    assert (report.selected.device, report.selected.precision) == ("cpu", "fp32")
    assert any("CPU" in w for w in report.warnings)
    assert report.install_command == CMD_CPU


def test_no_torch_means_no_backend() -> None:
    report = build_report(linux_cpu(torch_installed=False))
    assert report.selected is None
    assert any(CMD_CPU in s for s in report.suggestions)


def test_explicit_unavailable_choices_fail_with_install_hint() -> None:
    with pytest.raises(NotAvailableError) as mlx_err:
        select_backend(linux_cpu(), backend=Backend.MLX)
    assert "Apple Silicon" in mlx_err.value.message and CMD_CPU in (mlx_err.value.hint or "")
    with pytest.raises(NotAvailableError) as cuda_err:
        select_backend(linux_cpu(), device=Device.CUDA)
    assert "NVIDIA" in cuda_err.value.message


def test_explicit_choices_are_honoured_when_available() -> None:
    sel = select_backend(windows_nvidia(), device=Device.CPU)
    assert (sel.device, sel.precision) == ("cpu", "fp32")
    sel = select_backend(windows_nvidia(), device=Device.CUDA, precision=Precision.FP16)
    assert sel.precision == "fp16"
    sel = select_backend(windows_nvidia(), backend=Backend.HF_PEFT)
    assert sel.device == "cuda"


def test_explicit_unsupported_precisions_fail() -> None:
    with pytest.raises(NotAvailableError):
        select_backend(linux_cpu(), device=Device.CPU, precision=Precision.BF16)
    with pytest.raises(NotAvailableError):
        select_backend(windows_nvidia(bitsandbytes=False), precision=Precision.QLORA_4BIT)
    with pytest.raises(NotAvailableError):
        select_backend(windows_nvidia(bf16=False), device=Device.CUDA, precision=Precision.BF16)
