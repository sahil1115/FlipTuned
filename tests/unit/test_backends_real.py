"""Real-model backend tests. None run by default:

- ``tiny``: downloads the tiny test model (a few MB) and runs it on CPU (CI on all OSes);
- ``slow``: downloads every catalog model's tokenizer and runs the prefix test;
- ``mlx``: Apple Silicon only (smoke model, about 0.3 GB);
- ``cuda``: NVIDIA only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fliptuned.config.models import Backend
from fliptuned.diagnostics.backend_select import BackendSelection
from fliptuned.models.catalog import get_entry, load_catalog
from fliptuned.models.tokenizer import load_tokenizer
from fliptuned.services.datasets import BuildDatasetRequest, build_dataset
from fliptuned.services.evaluation import BaselineRequest, run_baseline
from fliptuned.training import backends as registry
from fliptuned.training.backends.base import GenerationConfig
from fliptuned.training.templates import SAMPLE_CONVERSATIONS, prefix_test, prompt_ids
from tests.helpers.pipeline import SAMPLE_DATASET, ingest_samples

CPU = BackendSelection(backend=Backend.HF_PEFT, device="cpu", precision="fp32", reason="test")
MESSAGES = SAMPLE_CONVERSATIONS[1]


@pytest.fixture(scope="module")
def tiny_cpu():
    return registry.load_model(get_entry("tiny-qwen2"), CPU, allow_download=True)


@pytest.mark.tiny
def test_tiny_load_and_generate_on_cpu(tiny_cpu) -> None:
    backend, loaded = tiny_cpu
    config = GenerationConfig(max_new_tokens=16)
    first = backend.generate(loaded, MESSAGES[:-1], config)
    second = backend.generate(loaded, MESSAGES[:-1], config)
    assert first.text == second.text  # greedy decoding is repeatable
    assert first.completion_tokens <= 16 and first.prompt_tokens > 0
    assert "".join(backend.stream(loaded, MESSAGES[:-1], config)) == first.text
    assert len(loaded.template_checksum) == 64


@pytest.mark.tiny
def test_tiny_prefix_and_mask_tests(tiny_cpu, tmp_path: Path) -> None:
    backend, loaded = tiny_cpu
    assert prefix_test(loaded.tokenizer, loaded.template_kwargs).passed
    for messages in SAMPLE_CONVERSATIONS:
        mask = backend.mask_check(loaded, messages, tmp_path / "trl")
        assert mask.passed, mask
        assert mask.masked_prompt_tokens == mask.prompt_tokens > 0
        assert mask.labelled_completion_tokens == mask.completion_tokens > 0


@pytest.mark.tiny
def test_tiny_baseline_end_to_end(tmp_path: Path) -> None:
    home = tmp_path / "home"
    ingest_samples(home)
    build_dataset(
        BuildDatasetRequest(config_path=str(SAMPLE_DATASET), home=str(home), allow_download=True)
    )
    result = run_baseline(
        BaselineRequest(home=str(home), dataset="fixtures", limit=2, allow_download=True)
    )
    assert result.items == 2 and (result.backend, result.device) == ("hf_peft", "cpu")
    assert (Path(result.path) / "results.jsonl").exists()


@pytest.mark.slow
@pytest.mark.parametrize("model_id", [e.id for e in load_catalog().models])
def test_catalog_prefix_test_results_hold(model_id: str) -> None:
    entry = get_entry(model_id)
    tokenizer = load_tokenizer(entry, allow_download=True)
    result = prefix_test(tokenizer, entry.chat_template_kwargs)
    assert ("pass" if result.passed else "fail") == entry.prefix_test.result, result


@pytest.mark.mlx
def test_mlx_load_generate_and_mask_offset(tmp_path: Path) -> None:
    from mlx_lm.tuner.datasets import ChatDataset

    entry = get_entry("qwen2.5-coder-0.5b")  # no template kwargs, so ChatDataset applies as-is
    selection = BackendSelection(
        backend=Backend.MLX, device="apple_gpu", precision="model (4-bit → QLoRA)", reason="test"
    )
    backend, loaded = registry.load_model(entry, selection, allow_download=True)
    config = GenerationConfig(max_new_tokens=16)
    first = backend.generate(loaded, MESSAGES[:-1], config)
    assert first.text and first.text == backend.generate(loaded, MESSAGES[:-1], config).text
    assert "".join(backend.stream(loaded, MESSAGES[:-1], config)) == first.text
    # mlx-lm's own dataset code masks exactly the prompt that our prefix test validates.
    tokens, offset = ChatDataset([], loaded.tokenizer, mask_prompt=True).process(
        {"messages": MESSAGES}
    )
    assert offset == len(prompt_ids(loaded.tokenizer, MESSAGES[:-1], {})) < len(tokens)


@pytest.mark.cuda
def test_cuda_load_and_generate() -> None:
    selection = BackendSelection(
        backend=Backend.HF_PEFT, device="cuda", precision="bf16", reason="t"
    )
    backend, loaded = registry.load_model(get_entry("tiny-qwen2"), selection, allow_download=True)
    result = backend.generate(loaded, MESSAGES[:-1], GenerationConfig(max_new_tokens=8))
    assert result.completion_tokens <= 8
