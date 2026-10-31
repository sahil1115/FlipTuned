"""Backends, catalog compatibility, template checks, parity and baseline (no downloads here;
real-model tests carry the tiny / slow / mlx / cuda markers)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fliptuned.cli import app
from fliptuned.config.models import (
    DEFAULT_SYSTEM_PROMPT,
    Backend,
    DatasetConfig,
    Device,
    TrainConfig,
)
from fliptuned.diagnostics.backend_select import BackendSelection
from fliptuned.errors import FlipTunedError, NotAvailableError
from fliptuned.inference.prompting import chat_messages
from fliptuned.models import hub
from fliptuned.models.catalog import get_entry, load_catalog
from fliptuned.models.compat import compatibility, select_for_entry
from fliptuned.services import evaluation
from fliptuned.services.datasets import BuildDatasetRequest, build_dataset
from fliptuned.services.evaluation import BaselineRequest, run_baseline
from fliptuned.services.models import ModelListRequest, list_models
from fliptuned.training import backends as registry
from fliptuned.training.backends.base import GenerationConfig, GenerationResult, LoadedModel
from fliptuned.training.templates import prefix_test, template_checksum
from tests.helpers.pipeline import SAMPLE_DATASET, ingest_samples, use_fake_tokenizer
from tests.unit import profiles

SRC = Path(__file__).resolve().parents[2] / "src" / "fliptuned"
runner = CliRunner()
CPU = BackendSelection(backend=Backend.HF_PEFT, device="cpu", precision="fp32", reason="test")


# --------------------------------------------------------------------------- registry & selection


def test_mlx_backend_is_refused_off_apple_silicon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "mlx_supported_here", lambda: False)
    with pytest.raises(NotAvailableError, match="Apple Silicon"):
        registry.get_backend(Backend.MLX)
    monkeypatch.setattr(registry, "mlx_supported_here", lambda: True)
    assert registry.get_backend(Backend.MLX).name is Backend.MLX  # module imports without mlx
    assert registry.get_backend(Backend.HF_PEFT).artifact_format() == "peft_lora_safetensors"


def test_capabilities_per_machine() -> None:
    hf = registry.get_backend(Backend.HF_PEFT)
    nvidia = hf.capabilities(profiles.windows_nvidia())
    assert nvidia.available and nvidia.devices == ["cuda", "cpu"] and nvidia.qlora
    assert nvidia.precisions[0] == "qlora_4bit"
    cpu = hf.capabilities(profiles.linux_cpu())
    assert cpu.devices == ["cpu"] and cpu.precisions == ["fp32"] and not cpu.qlora
    none = hf.capabilities(profiles.linux_cpu(torch_installed=False))
    assert not none.available and any("not installed" in r for r in none.reasons)


def test_unavailable_backend_errors_include_install_hints() -> None:
    entry = get_entry("qwen2.5-coder-0.5b")
    with pytest.raises(NotAvailableError) as info:
        select_for_entry(profiles.linux_cpu(), entry, backend=Backend.MLX)
    assert "uv sync" in (info.value.hint or "")
    with pytest.raises(NotAvailableError) as info:
        select_for_entry(profiles.linux_cpu(), entry, device=Device.CUDA)
    assert "uv sync" in (info.value.hint or "")


def test_auto_skips_backends_the_model_lacks() -> None:
    tiny = get_entry("tiny-qwen2")
    selection = select_for_entry(profiles.mac_m1(), tiny)
    assert selection.backend is Backend.HF_PEFT and "no mlx version" in selection.reason
    assert select_for_entry(profiles.mac_m1(), get_entry("qwen3-0.6b")).backend is Backend.MLX
    with pytest.raises(NotAvailableError, match="no mlx version"):
        hub.model_source(tiny, "mlx")


# --------------------------------------------------------------------------- catalog


def test_catalog_entries_are_complete() -> None:
    for entry in load_catalog().models:
        for name, source in entry.sources.items():
            assert source.download_gb > 0, entry.id
            keys = [k for k in entry.est_train_memory_gb if k.split(":")[0] == name]
            assert keys, f"{entry.id} has no memory estimate for {name}"
        assert entry.trainable, entry.id


def _with_ram(probe, ram_gb: float):
    return probe.model_copy(update={"machine": probe.machine.model_copy(update={"ram_gb": ram_gb})})


def test_compatibility_labels_per_memory_pool() -> None:
    entry = get_entry("qwen3-1.7b")  # est. cpu 10 GB, cuda qlora 3.5 GB, mlx 3.5 GB
    for ram, label in ((32, "fits"), (16, "tight"), (12, "won't fit")):
        probe = _with_ram(profiles.linux_cpu(), ram)
        compat = compatibility(entry, select_for_entry(probe, entry), probe)
        assert compat.label == label and compat.pool_gb == ram - 4 and compat.is_estimate

    nvidia = profiles.windows_nvidia()  # 11.2 GB free VRAM
    compat = compatibility(entry, select_for_entry(nvidia, entry), nvidia)
    assert (compat.label, compat.estimate_gb, compat.pool_gb, compat.pool) == (
        "fits",
        3.5,
        11.2,
        "free VRAM",
    )
    mac = profiles.mac_m1()  # 16 GB unified - 4 GB headroom
    compat = compatibility(entry, select_for_entry(mac, entry), mac)
    assert compat.pool_gb == 12 and compat.estimate_gb == 3.5 and compat.label == "fits"


def test_list_models_hides_test_models_and_reads_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    source = get_entry("qwen3-0.6b").sources["hf_peft"]
    snap = tmp_path / f"models--{source.repo.replace('/', '--')}" / "snapshots" / source.revision
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"")

    result = list_models(ModelListRequest(), probe_fn=profiles.linux_cpu)
    ids = [m.id for m in result.models]
    assert "tiny-qwen2" not in ids and "qwen3-0.6b" in ids
    by_id = {m.id: m for m in result.models}
    assert by_id["qwen3-0.6b"].cached and not by_id["qwen3-1.7b"].cached
    assert by_id["qwen3-0.6b"].backends_here == ["hf_peft"]
    assert any("CPU training limit" in n for n in by_id["qwen3-1.7b"].notes)
    assert "estimates" in result.estimate_note
    every = list_models(ModelListRequest(include_test_only=True), probe_fn=profiles.mac_m1)
    tiny = next(m for m in every.models if m.id == "tiny-qwen2")
    assert tiny.selection and tiny.selection.backend is Backend.HF_PEFT


def test_cli_models_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fliptuned.services.models.default_probe", profiles.linux_cpu)
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0, result.output
    assert "qwen3-1.7b" in result.output and "tiny-qwen2" not in result.output
    assert "estimates" in result.output


# --------------------------------------------------------------------------- offline


def test_missing_model_offline_is_actionable_and_makes_no_network_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "empty"))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    real = huggingface_hub.snapshot_download
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs.get("local_files_only"))
        return real(*args, **kwargs)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", spy)
    with pytest.raises(NotAvailableError) as info:
        registry.load_model(get_entry("qwen2.5-coder-0.5b"), CPU)
    assert "--allow-download" in (info.value.hint or "") and "1 GB" in (info.value.hint or "")
    assert calls == [True]


def test_incomplete_snapshot_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(tmp_path))
    with pytest.raises(NotAvailableError, match="incomplete"):
        hub.model_files(get_entry("tiny-qwen2"), "hf_peft", allow_download=False)


# --------------------------------------------------------------------------- templates


class _FakeTokenizer:
    """Renders messages as role-tagged words; ``broken`` adds a marker near the start only when
    no generation prompt is requested, which breaks the prefix property."""

    chat_template = "{{ fake }}"

    def __init__(self, broken: bool = False):
        self.broken = broken

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict, **kw):
        ids = []
        for m in messages:
            ids += [hash(m["role"]) % 97, *[len(w) for w in m["content"].split()], 0]
        if add_generation_prompt:
            ids += [hash("assistant") % 97]
        elif self.broken:
            ids.insert(1, 999)
        return ids


def test_prefix_test_detects_broken_templates() -> None:
    assert prefix_test(_FakeTokenizer(), {}).passed
    broken = prefix_test(_FakeTokenizer(broken=True), {})
    assert not broken.passed and broken.failures == [0, 1, 2]


def test_template_checksum_covers_kwargs() -> None:
    tok = _FakeTokenizer()
    assert template_checksum(tok, {}) != template_checksum(tok, {"enable_thinking": False})
    assert template_checksum(tok, {"a": 1, "b": 2}) == template_checksum(tok, {"b": 2, "a": 1})


# --------------------------------------------------------------------------- parity & boundary


def test_system_prompt_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Inference rebuilds every test conversation byte-identically from the dataset's system prompt."""
    assert TrainConfig.model_fields["system_prompt"].default == DEFAULT_SYSTEM_PROMPT
    assert DatasetConfig.model_fields["system_prompt"].default == DEFAULT_SYSTEM_PROMPT
    use_fake_tokenizer(monkeypatch)
    home = tmp_path / "home"
    ingest_samples(home)
    built = build_dataset(BuildDatasetRequest(config_path=str(SAMPLE_DATASET), home=str(home)))
    manifest = json.loads((Path(built.path) / "manifest.json").read_text(encoding="utf-8"))
    for name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        for line in (Path(built.path) / name).read_text(encoding="utf-8").split("\n"):
            if line:
                messages = json.loads(line)["messages"]
                rebuilt = chat_messages(manifest["system_prompt"], [messages[1]])
                assert json.dumps(rebuilt) == json.dumps(messages[:-1])


def test_chat_messages_requires_a_final_user_turn() -> None:
    with pytest.raises(ValueError):
        chat_messages("s", [{"role": "assistant", "content": "x"}])


def test_inference_package_imports_nothing_that_reads_data() -> None:
    """SPEC §3.1 boundary, static part: inference/ never imports data, dataset or retrieval code."""
    forbidden_prefixes = (
        "fliptuned.dataset",
        "fliptuned.records",
        "fliptuned.ingestion",
        "fliptuned.services",
        "fliptuned.registry",
        "langchain",
        "llama_index",
        "faiss",
        "chromadb",
        "sentence_transformers",
        "rank_bm25",
        "sklearn",
        "openai",
        "anthropic",
        "requests",
        "httpx",
    )
    for path in (SRC / "inference").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            bad = [n for n in names if n.startswith(forbidden_prefixes)]
            assert bad == [], f"{path.name} imports {bad}"


# --------------------------------------------------------------------------- baseline (fake model)


class _FakeBackend:
    name = Backend.HF_PEFT

    def __init__(self) -> None:
        self.seen: list[list[dict]] = []

    def generate(self, loaded, messages, config: GenerationConfig) -> GenerationResult:
        self.seen.append(messages)
        return GenerationResult(
            text=f"answer {len(self.seen)}",
            prompt_tokens=10,
            completion_tokens=3,
            finish_reason="stop",
            seconds=0.01,
        )

    def artifact_format(self) -> str:
        return "peft_lora_safetensors"

    def versions(self) -> dict:
        return {"torch": "fake"}


@pytest.fixture
def dataset_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    use_fake_tokenizer(monkeypatch)
    home = tmp_path / "home"
    ingest_samples(home)
    build_dataset(BuildDatasetRequest(config_path=str(SAMPLE_DATASET), home=str(home)))
    return home


def _fake_load(monkeypatch: pytest.MonkeyPatch) -> _FakeBackend:
    fake = _FakeBackend()

    def load(entry, selection, *, allow_download=False, adapter_path=None):
        loaded = LoadedModel(
            backend=selection.backend,
            entry=entry,
            selection=selection,
            model_dir=Path("."),
            adapter_path=None,
            model=None,
            tokenizer=None,
            template_checksum="c" * 64,
        )
        return fake, loaded

    monkeypatch.setattr(evaluation, "load_model", load)
    monkeypatch.setattr(evaluation, "default_probe", profiles.linux_cpu)
    return fake


def test_baseline_writes_results_metadata_and_registry(
    dataset_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake_load(monkeypatch)
    seen_progress = []
    result = run_baseline(
        BaselineRequest(home=str(dataset_home), dataset="fixtures"),
        progress=lambda done, total: seen_progress.append((done, total)),
    )
    assert result.items == 14 and result.finish_reasons == {"stop": 14}
    assert (result.backend, result.device, result.precision) == ("hf_peft", "cpu", "fp32")
    assert seen_progress[-1] == (14, 14)
    folder = Path(result.path)
    rows = [json.loads(line) for line in (folder / "results.jsonl").read_text("utf-8").splitlines()]
    assert [r["line"] for r in rows] == list(range(1, 15))
    assert rows[0]["output"] == "answer 1" and rows[0]["item_id"].startswith("qa_")
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert meta["model"]["revision"] == get_entry("tiny-qwen2").sources["hf_peft"].revision
    assert meta["model"]["template_checksum"] == "c" * 64
    assert meta["generation"] == {"max_new_tokens": 512, "decoding": "greedy"}
    assert {"backend", "device", "precision", "packages", "system_prompt_sha256"} <= set(meta)
    # Every prompt is system + question only: no answer, no other record text.
    assert all(len(m) == 2 and m[0]["role"] == "system" for m in fake.seen)

    with pytest.raises(FlipTunedError, match="already exists"):
        run_baseline(BaselineRequest(home=str(dataset_home), dataset="fixtures"))
    again = run_baseline(
        BaselineRequest(home=str(dataset_home), dataset="fixtures", overwrite=True)
    )
    assert again.rebuilt and again.baseline_id == result.baseline_id
    limited = run_baseline(BaselineRequest(home=str(dataset_home), dataset="fixtures", limit=3))
    assert limited.items == 3 and limited.baseline_id != result.baseline_id


def test_baseline_with_another_model_warns(
    dataset_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_load(monkeypatch)
    result = run_baseline(
        BaselineRequest(home=str(dataset_home), dataset="fixtures", model="qwen3-0.6b", limit=1)
    )
    assert result.model == "qwen3-0.6b" and "measured with 'tiny-qwen2'" in result.warnings[0]


def test_cli_evaluate_needs_baseline_only(
    dataset_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_load(monkeypatch)
    refused = runner.invoke(app, ["evaluate", "--dataset", "fixtures", "--home", str(dataset_home)])
    assert refused.exit_code == 2 and "--baseline-only" in refused.output
    ok = runner.invoke(
        app,
        [
            "evaluate",
            "--baseline-only",
            "--dataset",
            "fixtures",
            "--limit",
            "2",
            "--home",
            str(dataset_home),
        ],
    )
    assert ok.exit_code == 0, ok.output
    assert "Items          2" in ok.output and "answer 1" not in ok.output
