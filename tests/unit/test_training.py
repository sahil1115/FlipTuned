"""Training without a real model: presets, step math, best checkpoint, early stop, generated
configs, preflight, config registry, runs and resume rules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from fliptuned.cli import app
from fliptuned.config.models import Backend, TrainConfig
from fliptuned.diagnostics.backend_select import BackendSelection
from fliptuned.errors import ConfigError, FlipTunedError, NotAvailableError
from fliptuned.models.catalog import get_entry
from fliptuned.services import configs as configs_service
from fliptuned.services import preflight as preflight_service
from fliptuned.services import training as training_service
from fliptuned.services.datasets import BuildDatasetRequest, build_dataset
from fliptuned.training import backends as registry
from fliptuned.training.backends.base import ADAPTER_MARKER, TrainSpec
from fliptuned.training.backends.hf_peft_backend import (
    latest_checkpoint,
    lora_config_kwargs,
    sft_config_kwargs,
)
from fliptuned.training.backends.mlx_backend import mlx_lora_config
from fliptuned.training.plan import BestTracker, plan_steps, resolve_training, select_best
from tests.helpers.pipeline import REPO, SAMPLE_DATASET, ingest_samples, use_fake_tokenizer
from tests.unit import profiles

SAMPLE_TRAIN = REPO / "configs" / "train.sample.yaml"
runner = CliRunner()
CPU = BackendSelection(backend=Backend.HF_PEFT, device="cpu", precision="fp32", reason="t")
MLX = BackendSelection(backend=Backend.MLX, device="apple_gpu", precision="model", reason="t")


def _config(**overrides) -> TrainConfig:
    data = yaml.safe_load(SAMPLE_TRAIN.read_text(encoding="utf-8"))
    for key, value in overrides.items():
        if isinstance(value, dict):
            data[key] = {**data.get(key, {}), **value}
        else:
            data[key] = value
    return TrainConfig.model_validate(data)


def _minimal(**overrides) -> TrainConfig:
    """A config that sets only what it must, so presets show through."""
    return TrainConfig.model_validate(
        {"name": "m", "model": {"catalog_id": "tiny-qwen2"}, **overrides}
    )


# --------------------------------------------------------------------------- presets & plan


def test_presets_fill_only_what_the_user_left_out() -> None:
    quick = resolve_training(_minimal(preset="quick"), Backend.HF_PEFT)
    assert (quick.max_epochs, quick.patience, quick.eval_every_steps) == (1, 2, 10)
    standard = resolve_training(_minimal(), Backend.HF_PEFT)
    assert standard.max_epochs == 3 and standard.learning_rate == 1.0e-4
    mlx = resolve_training(_minimal(preset="thorough"), Backend.MLX)
    assert mlx.mlx.num_layers == 16 and mlx.max_epochs == 5
    mine = resolve_training(
        _minimal(preset="thorough", training={"max_epochs": 2, "mlx": {"num_layers": 2}}),
        Backend.MLX,
    )
    assert mine.max_epochs == 2 and mine.mlx.num_layers == 2 and mine.patience == 5


def test_step_plans_per_backend() -> None:
    training = _config(training={"batch_size": 2, "grad_accumulation_steps": 4}).training
    hf = plan_steps(Backend.HF_PEFT, 117, training)
    assert (hf.updates_per_epoch, hf.total_updates, hf.iterations) == (15, 45, None)
    mlx = plan_steps(Backend.MLX, 117, training)
    # C7: iters = floor(3 × floor(117 / 2)) = 174; C6: eval every 50 updates × 4 = 200 iterations
    assert (mlx.iterations, mlx.total_updates, mlx.steps_per_eval) == (174, 43, 200)
    fractional = _config(training={"max_epochs": 1.5}).training
    assert plan_steps(Backend.MLX, 10, fractional).iterations == 15
    with pytest.raises(ConfigError, match="fewer than batch_size"):
        plan_steps(Backend.HF_PEFT, 1, _config(training={"batch_size": 2}).training)


def test_best_checkpoint_and_early_stop_from_a_fake_loss_log() -> None:
    log = [(50, 2.0), (100, 1.5), (150, 1.6), (200, 1.4), (250, 1.45), (300, 1.41), (350, 1.5)]
    assert select_best(log) == (200, 1.4)
    assert select_best([(10, 1.0), (20, 1.0)]) == (10, 1.0)
    assert select_best([]) is None

    tracker = BestTracker(patience=3)
    decisions = [tracker.update(step, loss) for step, loss in log]
    assert [best for best, _ in decisions] == [True, True, False, True, False, False, False]
    assert [stop for _, stop in decisions] == [False] * 6 + [True]
    assert (tracker.best_step, tracker.best_loss) == (200, 1.4)
    strict = BestTracker(patience=1, min_delta=0.2)
    assert strict.update(1, 2.0) == (True, False) and strict.update(2, 1.9) == (False, True)


# --------------------------------------------------------------------------- generated configs


def test_hf_training_args(tmp_path: Path) -> None:
    training = resolve_training(_config(), Backend.HF_PEFT)
    args = sft_config_kwargs(training, CPU, tmp_path)
    assert args["report_to"] == "none" and args["completion_only_loss"] is True
    assert args["eval_steps"] == args["save_steps"] == training.eval_every_steps
    assert args["load_best_model_at_end"] and args["metric_for_best_model"] == "eval_loss"
    assert args["greater_is_better"] is False and args["use_cpu"] is True
    assert args["bf16"] is False and args["fp16"] is False  # C10: explicit, never TRL's default
    assert args["max_length"] == training.max_seq_length
    cuda_q = BackendSelection(
        backend=Backend.HF_PEFT, device="cuda", precision="qlora_4bit", reason="t"
    )
    q = sft_config_kwargs(training, cuda_q, tmp_path, bf16_supported=True)
    assert q["bf16"] and not q["fp16"] and not q["use_cpu"]
    lora = lora_config_kwargs(_config(), training)
    assert lora["lora_alpha"] == 16 and lora["r"] == 8 and lora["target_modules"] == "all-linear"


def test_mlx_config_has_no_report_to_and_uses_alpha_over_rank(tmp_path: Path) -> None:
    config = _config(training={"grad_accumulation_steps": 4, "eval_every_steps": 5})
    training = resolve_training(config, Backend.MLX)
    spec = TrainSpec(
        config=config,
        training=training,
        entry=get_entry("qwen2.5-coder-0.5b"),
        selection=MLX,
        model_dir=tmp_path / "model",
        dataset_dir=tmp_path / "data",
        run_dir=tmp_path / "run",
        system_prompt="s",
    )
    plan = plan_steps(Backend.MLX, 117, training)
    cfg = mlx_lora_config(spec, plan, iters=plan.iterations, resume_file=None)
    assert "report_to" not in cfg and "project_name" not in cfg
    assert not [k for k in cfg if k.startswith("report")]
    assert cfg["lora_parameters"] == {"rank": 8, "dropout": 0.05, "scale": 2.0}
    assert cfg["steps_per_eval"] == cfg["save_every"] == 20
    assert cfg["mask_prompt"] is True and cfg["val_batches"] == -1
    assert cfg["iters"] == plan.iterations == 3 * 117


# --------------------------------------------------------------------------- preflight


@pytest.fixture
def built_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    use_fake_tokenizer(monkeypatch)
    home = tmp_path / "home"
    ingest_samples(home)
    build_dataset(BuildDatasetRequest(config_path=str(SAMPLE_DATASET), home=str(home)))
    monkeypatch.setattr(preflight_service, "load_tokenizer", lambda entry, **kw: _Tok())
    monkeypatch.setattr(preflight_service, "is_cached", lambda source: True)
    return home


class _Tok:
    chat_template = "x"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict, **kw):
        roles = {"system": 1, "user": 2, "assistant": 3}
        ids = [x for m in messages for x in (roles[m["role"]], len(m["content"]))]
        return ids + ([3] if add_generation_prompt else [])


def _preflight(home: Path, config: TrainConfig, probe=profiles.linux_cpu, **kw):
    return preflight_service.run_preflight(config, home=str(home), probe_fn=probe, **kw)


def _status(result) -> dict[str, str]:
    return {c.name: c.status.value for c in result.checks}


def test_preflight_passes_with_warnings_on_cpu(built_home: Path) -> None:
    result = _preflight(built_home, _config())
    status = _status(result)
    assert result.ok and result.context is not None
    assert status["backend"] == status["memory"] == status["template"] == status["steps"] == "pass"
    assert status["model"] == "warn" and status["cpu_time"] == "warn"
    assert result.plan["total_updates"] == 84  # 109 examples: ceil(109 / 4) = 28 per epoch × 3
    assert all(c.fix is None for c in result.checks if c.status.value == "pass")


def test_preflight_fails_an_oversized_config_with_guidance(built_home: Path) -> None:
    small = profiles.linux_cpu().model_copy(
        update={"machine": profiles.linux_cpu().machine.model_copy(update={"ram_gb": 6.0})}
    )
    result = _preflight(
        built_home, _config(model={"catalog_id": "qwen3-1.7b"}), probe=lambda: small
    )
    failed = {c.name: c for c in result.checks if c.status.value == "fail"}
    assert not result.ok and result.context is None
    assert "won't fit" in failed["memory"].message and "estimate" in failed["memory"].message
    assert "smaller model" in failed["memory"].fix
    assert "CPU training limit" in failed["cpu_model_size"].message
    assert failed["cpu_model_size"].fix


def test_preflight_cpu_size_limit_follows_config(built_home: Path) -> None:
    config = _config(
        model={"catalog_id": "qwen3-1.7b"}, training={"hf_peft": {"cpu_max_params_b": 3.0}}
    )
    assert "cpu_model_size" not in _status(_preflight(built_home, config))


def test_preflight_other_failures(built_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    status = _status(_preflight(built_home, _config(system_prompt="Something else.")))
    assert status["system_prompt"] == "fail"
    status = _status(_preflight(built_home, _config(backend="mlx")))
    assert status["backend"] == "fail"
    monkeypatch.setattr(preflight_service, "is_cached", lambda source: False)
    assert _status(_preflight(built_home, _config()))["model_files"] == "fail"
    assert _status(_preflight(built_home, _config(), allow_download=True))["model_files"] == "warn"
    late = _config(training={"eval_every_steps": 500})
    assert _status(_preflight(built_home, late))["validation"] == "warn"


def test_preflight_without_a_dataset(tmp_path: Path) -> None:
    home = tmp_path / "home"
    ingest_samples(home, synthetic=False)
    result = _preflight(home, _config())
    assert not result.ok and _status(result)["dataset"] == "fail"


def test_dry_run_cli_and_refused_start(built_home: Path) -> None:
    sample = str(SAMPLE_TRAIN)
    dry = runner.invoke(app, ["train", "--config", sample, "--dry-run", "--home", str(built_home)])
    assert dry.exit_code == 0 and "ready to train" in dry.output and "[warn] cpu_time" in dry.output
    bad = built_home.parent / "bad.yaml"
    bad.write_text(
        SAMPLE_TRAIN.read_text(encoding="utf-8").replace("backend: auto", "backend: mlx"),
        encoding="utf-8",
    )
    refused = runner.invoke(app, ["train", "--config", str(bad), "--home", str(built_home)])
    assert (
        refused.exit_code == 5
        and "Preflight failed" in refused.output
        and "backend" in refused.output
    )
    assert not list((built_home / "runs").iterdir())


# --------------------------------------------------------------------------- config registry


def test_config_versions_and_duplicate(tmp_path: Path) -> None:
    home = str(tmp_path / "home")
    first = configs_service.save_config(home, _config())
    again = configs_service.save_config(home, _config())
    changed = configs_service.save_config(home, _config(training={"max_epochs": 2}))
    assert (first.version, again.version, changed.version) == (1, 1, 2)
    assert first.created and not again.created and changed.created
    copy = configs_service.duplicate_config(
        configs_service.DuplicateConfigRequest(home=home, name="fixtures-tiny", new_name="copy")
    )
    assert copy.version == 1
    listed = configs_service.list_configs(configs_service.ListConfigsRequest(home=home))
    assert [(c.name, c.version, c.versions) for c in listed] == [
        ("copy", 1, 1),
        ("fixtures-tiny", 2, 2),
    ]
    detail = configs_service.show_config(configs_service.ShowConfigRequest(home=home, name="copy"))
    assert detail.config["training"]["max_epochs"] == 2 and detail.parent_id == changed.id
    v1 = configs_service.show_config(
        configs_service.ShowConfigRequest(home=home, name="fixtures-tiny", version=1)
    )
    assert v1.config["training"]["max_epochs"] == 3
    with pytest.raises(FlipTunedError, match="already exists"):
        configs_service.duplicate_config(
            configs_service.DuplicateConfigRequest(home=home, name="copy", new_name="fixtures-tiny")
        )
    with pytest.raises(NotAvailableError):
        configs_service.show_config(configs_service.ShowConfigRequest(home=home, name="nope"))

    shown = runner.invoke(app, ["configs", "show", "copy", "--home", home])
    assert shown.exit_code == 0 and "name: copy" in shown.output
    assert "fixtures-tiny" in runner.invoke(app, ["configs", "list", "--home", home]).output
    dup = runner.invoke(app, ["configs", "duplicate", "copy", "copy2", "--home", home])
    assert dup.exit_code == 0 and "copy2 v1" in dup.output


def test_saved_configs_keep_the_preset_in_charge(tmp_path: Path) -> None:
    """Only fields the user set are stored, so a reloaded 'quick' config still trains as quick."""
    home = str(tmp_path / "home")
    configs_service.save_config(home, _minimal(preset="quick", training={"patience": 7}))
    stored = configs_service.show_config(
        configs_service.ShowConfigRequest(home=home, name="m")
    ).config
    assert stored == {
        "name": "m",
        "model": {"catalog_id": "tiny-qwen2"},
        "preset": "quick",
        "training": {"patience": 7},
    }
    effective = resolve_training(TrainConfig.model_validate(stored), Backend.HF_PEFT)
    assert (effective.max_epochs, effective.patience, effective.eval_every_steps) == (1, 7, 10)
    configs_service.duplicate_config(
        configs_service.DuplicateConfigRequest(home=home, name="m", new_name="m2")
    )
    copy = configs_service.show_config(configs_service.ShowConfigRequest(home=home, name="m2"))
    assert copy.config == {**stored, "name": "m2"}


@pytest.mark.parametrize(
    ("probe", "auto", "options"),
    [
        (profiles.mac_m1(), ("mlx", "apple_gpu"), {"apple_gpu": ["auto"], "cpu": ["auto", "fp32"]}),
        (
            profiles.windows_nvidia(),
            ("hf_peft", "cuda"),
            {"cuda": ["auto", "qlora_4bit", "bf16", "fp16"], "cpu": ["auto", "fp32"]},
        ),
        (
            profiles.windows_nvidia(bitsandbytes=False, bf16=False),
            ("hf_peft", "cuda"),
            {"cuda": ["auto", "fp16"], "cpu": ["auto", "fp32"]},
        ),
        (profiles.linux_cpu(), ("hf_peft", "cpu"), {"cpu": ["auto", "fp32"]}),
    ],
)
def test_training_choices_list_only_what_this_machine_has(probe, auto, options) -> None:
    choices = configs_service.training_choices(probe_fn=lambda: probe)
    assert choices.auto is not None
    assert (choices.auto.backend.value, choices.auto.device) == auto
    available = {o.device: o.precisions for o in choices.options if o.available}
    assert available == options
    assert all(o.precisions == [] and o.reason for o in choices.options if not o.available)


def test_training_choices_follow_the_model() -> None:
    """The tiny model has no MLX version, so 'auto' on a Mac picks CPU for it."""
    mac = profiles.mac_m1()
    choices = configs_service.training_choices("tiny-qwen2", probe_fn=lambda: mac)
    assert choices.auto is not None and choices.auto.device == "cpu"
    missing = configs_service.training_choices("nope", probe_fn=lambda: mac)
    assert missing.auto is None and missing.auto_error


# --------------------------------------------------------------------------- runs, adapters, resume


def test_adapter_from_another_backend_is_refused(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / ADAPTER_MARKER).write_text(json.dumps({"backend": "mlx"}), encoding="utf-8")
    with pytest.raises(NotAvailableError, match="trained with the mlx backend"):
        registry.load_model(get_entry("tiny-qwen2"), CPU, adapter_path=adapter)


def test_latest_checkpoint_picks_the_highest_step(tmp_path: Path) -> None:
    for step in (5, 40, 10):
        (tmp_path / "checkpoints" / f"checkpoint-{step}").mkdir(parents=True)
    (tmp_path / "checkpoints" / "checkpoint-x").mkdir()
    assert latest_checkpoint(tmp_path).name == "checkpoint-40"
    assert latest_checkpoint(tmp_path / "none") is None


def test_start_creates_run_and_resume_rules(
    built_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(preflight_service, "default_probe", profiles.linux_cpu)
    home = str(built_home)
    start = training_service.start_training(
        training_service.TrainRequest(config_path=str(SAMPLE_TRAIN), home=home), detach=False
    )
    run = training_service.get_run(home, start.run_id)
    assert run.status == "queued" and start.job.holds_model_lock and start.config_version == 1
    meta = json.loads((Path(run.run_dir) / "metadata.json").read_text(encoding="utf-8"))
    assert meta["lora"]["effective_scale"] == 2.0 and meta["seeds"]["training"] == 42
    assert meta["model"]["revision"] == get_entry("tiny-qwen2").sources["hf_peft"].revision
    assert meta["dataset"]["fingerprint"] and meta["attempts"] == []

    with pytest.raises(FlipTunedError, match="only interrupted, failed or cancelled"):
        training_service.resume_run(
            training_service.ResumeRequest(home=home, run_id=start.run_id), detach=False
        )
    # The runner ending the job (e.g. a forced cancel) carries over to the run.
    from fliptuned.services import jobs as jobs_service

    jobs_service.cancel_job(
        jobs_service.CancelJobRequest(home=home, job_id=start.job.id, timeout=0)
    )
    assert training_service.get_run(home, start.run_id).status == "cancelled"
    with pytest.raises(FlipTunedError, match="no checkpoint"):
        training_service.resume_run(
            training_service.ResumeRequest(home=home, run_id=start.run_id), detach=False
        )
    with pytest.raises(NotAvailableError):
        training_service.resume_run(
            training_service.ResumeRequest(home=home, run_id="run_x"), detach=False
        )
