"""Export a trained run (SPEC §16).

Every export writes a folder (``<run_dir>/exports/<export_id>/``, or under the config's
``out_dir``) with ``adapter/`` (the best adapter), the generated backend config, the run's
``run_metadata.json`` and ``export.json``: the base model reference (repo, revision, license),
backend, artifact format, chat-template kwargs and checksum, and the exact system prompt the
adapter was trained with (prompts must use it unchanged).

With ``merge`` a ``merge_export`` job (model lock) adds ``merged/``, a standalone model with the
adapter merged in, through the run's backend. It then runs a 5-prompt before/after check: the
first five general-regression prompts (synthetic), greedy, on base + adapter ("before") and on the
merged model ("after"), and records in ``merge_check.json`` which answers came out identical.
Exports are recorded in the registry.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from fliptuned.config.app_home import AppPaths
from fliptuned.config.loader import load_config
from fliptuned.config.models import ExportConfig
from fliptuned.errors import FlipTunedError
from fliptuned.evaluation.regression import load_regression_set
from fliptuned.inference.prompting import chat_messages
from fliptuned.jobs.state import JobType
from fliptuned.jobs.worker import JobCancelled, JobContext
from fliptuned.models.catalog import get_entry
from fliptuned.models.hub import model_files
from fliptuned.platform.paths import remove_tree
from fliptuned.registry import repositories
from fliptuned.registry.db import connect, utc_now
from fliptuned.services.jobs import JobInfo, _submit
from fliptuned.services.review import open_registry
from fliptuned.services.runs import METADATA_FILE, RunArtifacts, load_run
from fliptuned.training.backends import get_backend, load_model
from fliptuned.training.backends.base import GenerationConfig

EXPORT_FILE = "export.json"
CHECK_FILE = "merge_check.json"
CHECK_PROMPTS = 5
CHECK_MAX_NEW_TOKENS = 64

HOW_TO_USE = {
    "hf_peft": (
        "Load the base model (repo at revision) with Transformers, then "
        "peft.PeftModel.from_pretrained(model, 'adapter'). Build prompts with the model's chat "
        "template (with chat_template_kwargs) and exactly this system_prompt."
    ),
    "mlx": (
        "mlx_lm.load(<base model folder>, adapter_path='adapter'). Build prompts with the model's "
        "chat template (with chat_template_kwargs) and exactly this system_prompt."
    ),
}


class ExportRequest(BaseModel):
    home: str | None = None
    run_id: str = Field(description="The training run to export (it must have finished).")
    config_path: str | None = Field(default=None, description="Export config (YAML).")
    merge: bool | None = Field(
        default=None,
        description="Also build a merged standalone model. Empty = the config's value.",
    )
    allow_download: bool = Field(default=False, description="Download model files if missing.")


class CheckItem(BaseModel):
    prompt_id: str
    identical: bool
    before_tokens: int
    after_tokens: int
    first_difference: int | None = Field(description="Character where the answers first differ.")


class MergeCheck(BaseModel):
    prompts: int
    identical: int
    max_new_tokens: int
    items: list[CheckItem]


class ExportResult(BaseModel):
    export_id: str
    run_id: str
    kind: Literal["adapter", "merged"]
    path: str
    files: list[str]
    merged_path: str | None = None
    merged_precision: str | None = None
    check: MergeCheck | None = None
    notes: list[str] = Field(default_factory=list)


class ExportStart(BaseModel):
    export_id: str
    path: str
    merge: bool
    result: ExportResult | None = Field(description="Ready now when no merge was asked for.")
    job: JobInfo | None = Field(description="The merge job, when a merge was asked for.")


def new_export_id() -> str:
    return f"ex_{datetime.now(UTC):%Y%m%d%H%M%S}_{secrets.token_hex(3)}"


def _write_json(path: Path, data: Any) -> None:
    path.write_bytes((json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def _export_manifest(run: RunArtifacts, export_id: str, kind: str, files: list[str]) -> dict:
    selection = run.selection
    entry = get_entry(run.catalog_id)
    source = entry.sources[selection.backend.value]
    model = run.meta.get("model", {})
    return {
        "export_id": export_id,
        "run_id": run.run_id,
        "created_at": utc_now(),
        "kind": kind,
        "base_model": {
            "catalog_id": entry.id,
            "repo": source.repo,
            "revision": source.revision,
            "license": entry.license,
            "model_card_url": entry.model_card_url,
        },
        "backend": selection.backend.value,
        "device": selection.device,
        "precision": selection.precision,
        "artifact_format": get_backend(selection.backend).artifact_format(),
        "chat_template_kwargs": entry.chat_template_kwargs,
        "template_checksum": model.get("template_checksum"),
        "system_prompt": run.system_prompt,
        "system_prompt_sha256": hashlib.sha256(run.system_prompt.encode("utf-8")).hexdigest(),
        "files": files,
        "how_to_use": HOW_TO_USE[selection.backend.value],
        "license_note": f"The base model's license ({entry.license}) applies to this export.",
        "merged": None,
    }


def export_run(request: ExportRequest, *, detach: bool) -> ExportStart:
    """Write the adapter export now; with merge, submit the merge job (refused while another job
    holds the model)."""
    paths = open_registry(request.home)
    run = load_run(paths, request.run_id)
    assert run.adapter is not None
    get_backend(run.selection.backend)  # fails clearly off-platform
    config = (
        load_config(Path(request.config_path).expanduser().resolve(), ExportConfig)
        if request.config_path
        else ExportConfig()
    )
    merge = config.merge if request.merge is None else request.merge
    export_id = new_export_id()
    root = (
        Path(config.out_dir).expanduser().resolve() if config.out_dir else run.run_dir / "exports"
    )
    out = root / (export_id if not config.out_dir else f"{run.run_id}-{export_id}")
    out.mkdir(parents=True)

    shutil.copytree(run.adapter, out / "adapter")
    files = ["adapter/"]
    backend_config = run.meta.get("backend_config_file")
    if backend_config and (run.run_dir / backend_config).exists():
        shutil.copyfile(run.run_dir / backend_config, out / backend_config)
        files.append(backend_config)
    shutil.copyfile(run.run_dir / METADATA_FILE, out / "run_metadata.json")
    files += ["run_metadata.json", EXPORT_FILE]
    kind = "merged" if merge else "adapter"
    manifest = _export_manifest(run, export_id, kind, files)
    _write_json(out / EXPORT_FILE, manifest)
    with connect(paths.db) as conn:
        repositories.insert_export(
            conn,
            export_id=export_id,
            run_id=run.run_id,
            kind=kind,
            status="queued" if merge else "succeeded",
            path=str(out),
            metadata={"files": files},
        )
    if not merge:
        result = ExportResult(
            export_id=export_id, run_id=run.run_id, kind="adapter", path=str(out), files=files
        )
        return ExportStart(export_id=export_id, path=str(out), merge=False, result=result, job=None)
    params = {"export_id": export_id, "allow_download": request.allow_download}
    try:
        job = _submit(request.home, JobType.MERGE_EXPORT, params, detach, run_id=run.run_id)
    except FlipTunedError:
        with connect(paths.db) as conn:
            repositories.update_export(conn, export_id, status="failed")
        raise
    return ExportStart(export_id=export_id, path=str(out), merge=True, result=None, job=job)


def _first_difference(a: str, b: str) -> int | None:
    if a == b:
        return None
    return next(
        (i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b))
    )


def execute_merge(ctx: JobContext) -> dict[str, Any]:
    """Handler of the ``merge_export`` job."""
    paths: AppPaths = ctx.paths
    export_id = ctx.params["export_id"]
    allow_download = bool(ctx.params.get("allow_download"))
    with connect(paths.db) as conn:
        row = repositories.get_export(conn, export_id)
        repositories.update_export(conn, export_id, status="running")
    out = Path(row["path"])
    status = "failed"
    try:
        run = load_run(paths, row["run_id"])
        assert run.adapter is not None
        selection = run.selection
        entry = get_entry(run.catalog_id)
        backend = get_backend(selection.backend)
        model_dir = model_files(entry, selection.backend.value, allow_download=allow_download)
        merged_dir = out / "merged"
        if merged_dir.exists():
            remove_tree(merged_dir)
        ctx.progress(0, CHECK_PROMPTS * 2 + 1, "merging the adapter into the model")
        merged = backend.merge(entry, selection, model_dir, out / "adapter", merged_dir)
        ctx.progress(1, CHECK_PROMPTS * 2 + 1, "merged")

        prompts = load_regression_set().items[:CHECK_PROMPTS]
        generation = GenerationConfig(max_new_tokens=CHECK_MAX_NEW_TOKENS)
        answers: dict[str, list] = {}
        step = 1
        for label, load in (
            ("before", lambda: load_model(entry, selection, adapter_path=run.adapter)),
            (
                "after",
                lambda: (
                    backend,
                    backend.load_for_inference(
                        entry,
                        selection.model_copy(update={"precision": merged.precision}),
                        merged_dir,
                    ),
                ),
            ),
        ):
            model_backend, loaded = load()
            try:
                answers[label] = []
                for item in prompts:
                    ctx.check_cancel()
                    messages = chat_messages(
                        run.system_prompt, [{"role": "user", "content": item.prompt}]
                    )
                    answers[label].append(model_backend.generate(loaded, messages, generation))
                    step += 1
                    ctx.progress(step, CHECK_PROMPTS * 2 + 1, f"{label} merge: prompt {step - 1}")
            finally:
                model_backend.unload(loaded)
        items = [
            CheckItem(
                prompt_id=item.id,
                identical=before.text == after.text,
                before_tokens=before.completion_tokens,
                after_tokens=after.completion_tokens,
                first_difference=_first_difference(before.text, after.text),
            )
            for item, before, after in zip(
                prompts, answers["before"], answers["after"], strict=True
            )
        ]
        check = MergeCheck(
            prompts=len(items),
            identical=sum(i.identical for i in items),
            max_new_tokens=CHECK_MAX_NEW_TOKENS,
            items=items,
        )
        _write_json(out / CHECK_FILE, check.model_dump(mode="json"))
        manifest = json.loads((out / EXPORT_FILE).read_text(encoding="utf-8"))
        manifest["files"] = [*manifest["files"], "merged/", CHECK_FILE]
        manifest["merged"] = {
            "path": "merged/",
            "precision": merged.precision,
            "notes": merged.notes,
            "check": {"prompts": check.prompts, "identical": check.identical},
        }
        _write_json(out / EXPORT_FILE, manifest)
        status = "succeeded"
    except (JobCancelled, KeyboardInterrupt):
        status = "cancelled"
        raise
    finally:
        with connect(paths.db) as conn:
            repositories.update_export(conn, export_id, status=status)
    notes = list(merged.notes)
    if check.identical < check.prompts:
        notes.append(
            f"{check.prompts - check.identical} of {check.prompts} check answers differ between "
            "the adapter and the merged model; see merge_check.json."
        )
    return ExportResult(
        export_id=export_id,
        run_id=run.run_id,
        kind="merged",
        path=str(out),
        files=manifest["files"],
        merged_path=str(merged_dir),
        merged_precision=merged.precision,
        check=check,
        notes=notes,
    ).model_dump(mode="json")


class ExportInfo(BaseModel):
    id: str
    run_id: str
    kind: str
    status: str
    path: str
    created_at: str


def list_exports(home: str | None, run_id: str | None = None) -> list[ExportInfo]:
    paths = open_registry(home)
    with connect(paths.db) as conn:
        rows = repositories.list_exports(conn, run_id)
    return [
        ExportInfo(
            id=r["id"],
            run_id=r["run_id"],
            kind=r["kind"],
            status=r["status"],
            path=r["path"],
            created_at=r["created_at"],
        )
        for r in rows
    ]
