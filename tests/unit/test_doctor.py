from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from fliptuned.cli import app
from fliptuned.services import doctor as doctor_service
from fliptuned.services.doctor import DoctorRequest, run_doctor

from .profiles import linux_cpu, mac_m1, windows_nvidia

REPO_ROOT = Path(__file__).resolve().parents[2]


def _fake_disk(path: Path) -> tuple[Path, float | None]:
    return Path(path.anchor or "/"), 100.0


def _populated_home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    (home / "runs" / "run1").mkdir(parents=True)
    (home / "fliptuned.db").write_bytes(b"not really a db")
    (home / "runs" / "run1" / "results.jsonl").write_text('{"q": "secret"}\n', encoding="utf-8")
    return home


def _snapshot(root: Path) -> dict[str, int]:
    return {str(p.relative_to(root)): p.stat().st_mtime_ns for p in root.rglob("*")}


def test_access_recorder_catches_reads_and_listings(tmp_path: Path, file_access) -> None:
    home = _populated_home(tmp_path)
    with file_access.recording():
        (home / "runs" / "run1" / "results.jsonl").read_text(encoding="utf-8")
        list(home.iterdir())  # os.listdir on 3.12
        list(home.rglob("*"))  # os.scandir
    events = {event for event, _ in file_access.under(home)}
    assert {"open", "os.listdir", "os.scandir"} <= events


def test_doctor_never_opens_or_changes_the_app_home(tmp_path: Path, file_access) -> None:
    home = _populated_home(tmp_path)
    before = _snapshot(home)
    with file_access.recording():
        report = run_doctor(DoctorRequest(home=str(home)))
    assert report.app_home == str(home)
    assert file_access.under(home) == []
    assert _snapshot(home) == before


def test_doctor_cli_never_opens_the_app_home(tmp_path: Path, file_access) -> None:
    home = _populated_home(tmp_path)
    with file_access.recording():
        result = CliRunner().invoke(app, ["doctor", "--home", str(home), "--json"])
    assert result.exit_code == 0, result.output
    assert file_access.under(home) == []


def test_doctor_does_not_create_missing_home(tmp_path: Path) -> None:
    home = tmp_path / "not-yet"
    run_doctor(DoctorRequest(home=str(home)))
    assert not home.exists()


def test_doctor_reports_faked_profiles() -> None:
    expected = {
        "mac": (mac_m1, "mlx", "apple_gpu"),
        "windows": (windows_nvidia, "hf_peft", "cuda"),
        "linux": (linux_cpu, "hf_peft", "cpu"),
    }
    for _, (profile, backend, device) in expected.items():
        report = run_doctor(DoctorRequest(), probe_fn=profile, packages_fn=dict, disk_fn=_fake_disk)
        assert report.backend.selected is not None
        assert (report.backend.selected.backend.value, report.backend.selected.device) == (
            backend,
            device,
        )


def test_doctor_checks_explicit_backend_from_train_config(tmp_path: Path) -> None:
    config = tmp_path / "train.yaml"
    config.write_text(
        "kind: train\nname: t\nmodel: {catalog_id: tiny-qwen2}\nbackend: mlx\n", encoding="utf-8"
    )
    report = run_doctor(
        DoctorRequest(train_config=str(config)),
        probe_fn=linux_cpu,
        packages_fn=dict,
        disk_fn=_fake_disk,
    )
    assert report.requested["backend"] == "mlx"
    assert report.backend.selected is None
    assert any("MLX" in w for w in report.warnings)


def test_doctor_cli_human_and_json_output(monkeypatch) -> None:
    monkeypatch.setattr(doctor_service, "default_probe", windows_nvidia)
    runner = CliRunner()
    human = runner.invoke(app, ["doctor"])
    assert human.exit_code == 0, human.output
    assert "Will use       hf_peft on cuda (qlora_4bit)" in human.output
    assert "RTX 4070" in human.output
    as_json = runner.invoke(app, ["doctor", "--json"])
    data = json.loads(as_json.output)
    assert data["backend"]["selected"]["precision"] == "qlora_4bit"


def test_suggest_install_prints_only_the_command(monkeypatch) -> None:
    monkeypatch.setattr(doctor_service, "default_probe", mac_m1)
    result = CliRunner().invoke(app, ["doctor", "--suggest-install"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "uv sync --extra mlx --extra cpu --extra web"


def test_doctor_exits_nonzero_without_backend(monkeypatch) -> None:
    monkeypatch.setattr(doctor_service, "default_probe", lambda: linux_cpu(torch_installed=False))
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 5
    assert "no usable backend" in result.output
