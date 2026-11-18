"""``build-dataset`` and ``validate-dataset`` (SPEC §10; synchronous here, a job from Phase 5).

A build writes ``<app_home>/datasets/<dataset_id>/``: ``train/valid/test.jsonl`` (canonical
messages), ``index.jsonl`` (IDs and metadata per line), ``exclusions.json``, ``duplicates.json``
and ``manifest.json``. The dataset ID derives from the name and a fingerprint of every input, so
the same inputs always give the same ID and byte-identical files.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from fliptuned.config.app_home import AppPaths
from fliptuned.config.loader import load_config
from fliptuned.config.models import DatasetConfig
from fliptuned.dataset.builder import (
    BuildWarning,
    DatasetPlan,
    TokenStats,
    canonical_messages,
    plan_dataset,
    token_stats,
)
from fliptuned.dataset.manifest import (
    DUPLICATES_FILE,
    EXCLUSIONS_FILE,
    INDEX_FILE,
    MANIFEST_FILE,
    PARTITION_FILES,
    CountInfo,
    DatasetManifest,
    FileInfo,
    GroupInfo,
    IndexRow,
    SplitInfo,
)
from fliptuned.dataset.split import DEFAULT_TOLERANCE, PARTITIONS
from fliptuned.diagnostics.checks import Check, CheckStatus
from fliptuned.errors import FlipTunedError, NotAvailableError
from fliptuned.models.catalog import get_entry
from fliptuned.models.tokenizer import load_token_counter
from fliptuned.platform.paths import remove_tree, replace_dir
from fliptuned.records.hashing import normalize_for_dedup, sha256_hex
from fliptuned.records.schema import Record
from fliptuned.registry import repositories
from fliptuned.registry.db import connect
from fliptuned.services.ingest import RECORDS_FILE, source_dir
from fliptuned.services.review import load_reviewed, open_registry, resolve_source_ids
from fliptuned.utils.logging import get_logger

log = get_logger(__name__)

BUILDER_VERSION = "2"


class BuildDatasetRequest(BaseModel):
    config_path: str = Field(description="Dataset config file.")
    home: str | None = None
    allow_download: bool = Field(
        default=False, description="Download the model's tokenizer files if they aren't cached."
    )
    overwrite: bool = Field(default=False, description="Rebuild a dataset that already exists.")


class BuildDatasetResult(BaseModel):
    dataset_id: str
    name: str
    path: str
    fingerprint: str
    manifest_hash: str
    rebuilt: bool
    counts: CountInfo
    split: SplitInfo
    groups: GroupInfo
    tokens: dict[str, TokenStats | None]
    warnings: list[BuildWarning]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def _json(data) -> bytes:
    return (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def fingerprint(config: DatasetConfig, tokenizer: dict, items: list[tuple[str, Record]]) -> str:
    """Hash of everything that shapes the output (record order does not matter)."""
    records = sorted(sha256_hex(source_id, record.model_dump_json()) for source_id, record in items)
    payload = {
        "builder_version": BUILDER_VERSION,
        "config": config.model_dump(mode="json", exclude={"name", "kind"}),
        "tokenizer": tokenizer,
        "records": records,
    }
    return _sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))


def build_dataset(request: BuildDatasetRequest) -> BuildDatasetResult:
    config_path = Path(request.config_path).expanduser().resolve()
    config = load_config(config_path, DatasetConfig)
    entry = get_entry(config.model_catalog_id)
    paths = open_registry(request.home)
    with connect(paths.db) as conn:
        source_ids = resolve_source_ids(conn, paths, config.source_ids)
        reviewed = load_reviewed(conn, paths, source_ids)
    items = [(r.source_id, r.record) for r in reviewed]

    counter = load_token_counter(entry, allow_download=request.allow_download)
    fingerprint_ = fingerprint(config, counter.identity, items)
    dataset_id = f"ds_{sha256_hex(config.name, fingerprint_)[:12]}"
    final_dir = paths.datasets / dataset_id
    with connect(paths.db) as conn:
        existing = repositories.get_dataset(conn, dataset_id)
    if (existing is not None or final_dir.exists()) and not request.overwrite:
        raise FlipTunedError(
            f"Dataset '{config.name}' was already built from exactly these inputs ({dataset_id}).",
            hint="Use it as it is, or re-run with --overwrite to rebuild it.",
        )

    plan = plan_dataset(items, config, counter)
    manifest = _write(
        paths, final_dir, dataset_id, fingerprint_, config, counter.identity, plan, source_ids
    )
    manifest_bytes = (final_dir / MANIFEST_FILE).read_bytes()
    manifest_hash = _sha256_bytes(manifest_bytes)
    with connect(paths.db) as conn:
        repositories.upsert_dataset(
            conn,
            dataset_id=dataset_id,
            name=config.name,
            status="ready",
            path=str(final_dir),
            config=config.model_dump(mode="json"),
            manifest_hash=manifest_hash,
            fingerprint=fingerprint_,
            stats={
                "counts": manifest.counts.model_dump(mode="json"),
                "warnings": [w.code for w in manifest.warnings],
            },
        )
    log.info("built dataset %s: %s", dataset_id, manifest.counts.partitions)
    return BuildDatasetResult(
        dataset_id=dataset_id,
        name=config.name,
        path=str(final_dir),
        fingerprint=fingerprint_,
        manifest_hash=manifest_hash,
        rebuilt=existing is not None,
        counts=manifest.counts,
        split=manifest.split,
        groups=manifest.groups,
        tokens=manifest.tokens,
        warnings=manifest.warnings,
    )


def _write(
    paths: AppPaths,
    final_dir: Path,
    dataset_id: str,
    fingerprint_: str,
    config: DatasetConfig,
    tokenizer: dict,
    plan: DatasetPlan,
    source_ids: list[str],
) -> DatasetManifest:
    files: dict[str, bytes] = {}
    index: list[dict] = []
    for partition in PARTITIONS:
        examples = plan.partitions[partition]
        files[PARTITION_FILES[partition]] = _jsonl(
            [{"messages": canonical_messages(config.system_prompt, e.record)} for e in examples]
        )
        for line, e in enumerate(examples, start=1):
            assert e.group is not None
            index.append(
                IndexRow(
                    partition=partition,
                    line=line,
                    source_id=e.source_id,
                    record_id=e.record.id,
                    category=e.record.category,
                    group_key=e.group.key,
                    group_source=e.group.source.value,
                    tokens=e.tokens,
                    truncated=e.truncated,
                ).model_dump(mode="json")
            )
    files[INDEX_FILE] = _jsonl(index)
    files[EXCLUSIONS_FILE] = _json(
        [
            {"source_id": x.source_id, "record_id": x.record_id, "reason": x.reason}
            for x in plan.exclusions
        ]
    )
    files[DUPLICATES_FILE] = _json(
        [
            {
                "source_id": d.source_id,
                "record_id": d.record_id,
                "kept_source_id": d.kept_source_id,
                "kept_record_id": d.kept_record_id,
            }
            for d in plan.duplicates
        ]
    )

    included = [e for p in PARTITIONS for e in plan.partitions[p]]
    group_sizes = Counter(e.group.key for e in included if e.group)
    largest = max(group_sizes.values(), default=0)
    categories = {
        p: dict(sorted(Counter(e.record.category or "(none)" for e in plan.partitions[p]).items()))
        for p in PARTITIONS
    }
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        name=config.name,
        fingerprint=fingerprint_,
        builder_version=BUILDER_VERSION,
        model=tokenizer,
        system_prompt=config.system_prompt,
        system_prompt_sha256=_sha256_bytes(config.system_prompt.encode("utf-8")),
        seed=config.seed,
        max_seq_length=config.max_seq_length,
        truncate_long_records=config.truncate_long_records,
        split=SplitInfo(
            target=config.split.model_dump(),
            actual={p: round(plan.split.shares[p], 4) for p in PARTITIONS},
            tolerance_pts=DEFAULT_TOLERANCE * 100,
            groups=plan.split.group_counts,
        ),
        counts=CountInfo(
            input_records=plan.input_records,
            included=len(included),
            partitions={p: len(plan.partitions[p]) for p in PARTITIONS},
            excluded=plan.excluded_by_reason(),
            duplicates=len(plan.duplicates),
            truncated=sum(1 for e in included if e.truncated),
        ),
        groups=GroupInfo(
            total=len(group_sizes),
            by_source=dict(
                sorted(Counter(e.group.source.value for e in included if e.group).items())
            ),
            largest_size=largest,
            largest_share=round(largest / len(included), 4) if included else 0.0,
        ),
        tokens={p: token_stats([e.tokens for e in plan.partitions[p]]) for p in PARTITIONS},
        categories=categories,
        warnings=plan.warnings,
        sources={
            s: _sha256_bytes((source_dir(paths, s) / RECORDS_FILE).read_bytes()) for s in source_ids
        },
        config=config.model_dump(mode="json"),
        files={
            name: FileInfo(
                sha256=_sha256_bytes(data),
                lines=data.count(b"\n") if name.endswith(".jsonl") else None,
            )
            for name, data in files.items()
        },
    )

    paths.ensure()
    tmp_dir = paths.datasets / f".building-{dataset_id}"
    if tmp_dir.exists():
        remove_tree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    for name, data in files.items():
        (tmp_dir / name).write_bytes(data)
    (tmp_dir / MANIFEST_FILE).write_bytes(_json(manifest.model_dump(mode="json")))
    replace_dir(tmp_dir, final_dir)
    return manifest


# --------------------------------------------------------------------------- locate / read


def locate_dataset(paths: AppPaths, id_or_name: str) -> tuple[str, Path, str | None]:
    """(dataset_id, folder, registered manifest hash or None)."""
    with connect(paths.db) as conn:
        row = repositories.find_dataset(conn, id_or_name.strip())
    if row is not None:
        return row["id"], Path(row["path"]), row["manifest_hash"]
    folder = paths.datasets / id_or_name.strip()
    if id_or_name.startswith("ds_") and (folder / MANIFEST_FILE).exists():
        return folder.name, folder, None
    raise NotAvailableError(
        f"No dataset called '{id_or_name}'.",
        hint="Build one with `fliptuned build-dataset --config <file>`.",
    )


def read_manifest(folder: Path) -> DatasetManifest:
    try:
        return DatasetManifest.model_validate_json((folder / MANIFEST_FILE).read_bytes())
    except FileNotFoundError:
        raise NotAvailableError(
            f"Dataset folder {folder.name} has no manifest.", hint="Rebuild the dataset."
        ) from None
    except ValidationError:
        raise FlipTunedError(
            f"The manifest of {folder.name} is not valid.", hint="Rebuild the dataset."
        ) from None


def read_index(folder: Path) -> list[IndexRow]:
    with open(folder / INDEX_FILE, encoding="utf-8") as handle:
        return [IndexRow.model_validate_json(line) for line in handle if line.strip()]


# --------------------------------------------------------------------------- validate


class ValidateDatasetRequest(BaseModel):
    home: str | None = None
    dataset: str = Field(description="Dataset ID or name.")


class ValidateDatasetResult(BaseModel):
    dataset_id: str
    name: str
    path: str
    ok: bool = Field(description="True when no check failed (warnings allowed).")
    checks: list[Check]


_ROLES = ["system", "user", "assistant"]


def validate_dataset(request: ValidateDatasetRequest) -> ValidateDatasetResult:
    paths = open_registry(request.home)
    dataset_id, folder, registered_hash = locate_dataset(paths, request.dataset)
    checks: list[Check] = []

    def add(name: str, ok: bool, message: str, fix: str | None = None, *, warn: bool = False):
        status = CheckStatus.PASS if ok else CheckStatus.WARN if warn else CheckStatus.FAIL
        checks.append(Check(name=name, status=status, message=message, fix=None if ok else fix))

    rebuild = "Rebuild the dataset with `fliptuned build-dataset --config <file> --overwrite`."
    manifest = read_manifest(folder)
    manifest_hash = _sha256_bytes((folder / MANIFEST_FILE).read_bytes())
    if registered_hash is None:
        add("registry", False, "The dataset folder is not in the registry.", rebuild, warn=True)
    else:
        add(
            "registry",
            registered_hash == manifest_hash,
            "Manifest matches the registry."
            if registered_hash == manifest_hash
            else "The manifest changed after the build.",
            rebuild,
        )

    bad_files = []
    for name, info in manifest.files.items():
        path = folder / name
        if not path.exists() or _sha256_bytes(path.read_bytes()) != info.sha256:
            bad_files.append(name)
    add(
        "files",
        not bad_files,
        "All files match their recorded checksums."
        if not bad_files
        else f"Missing or changed: {', '.join(bad_files)}.",
        rebuild,
    )

    format_problems: list[str] = []
    lines_per_partition: dict[str, int] = {}
    questions: dict[str, set[str]] = defaultdict(set)
    for partition, filename in PARTITION_FILES.items():
        count = 0
        path = folder / filename
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                count += 1
                problem = _line_problem(line, manifest.system_prompt)
                if problem:
                    format_problems.append(f"{filename} line {number}: {problem}")
                else:
                    user = json.loads(line)["messages"][1]["content"]
                    questions[sha256_hex(normalize_for_dedup(user))].add(partition)
        lines_per_partition[partition] = count
    add(
        "format",
        not format_problems,
        "Every line is a system/user/assistant conversation with the dataset's system prompt."
        if not format_problems
        else f"{len(format_problems)} bad line(s): " + "; ".join(format_problems[:10]),
        rebuild,
    )
    add(
        "system_prompt",
        manifest.system_prompt_sha256 == _sha256_bytes(manifest.system_prompt.encode("utf-8")),
        "System prompt matches its checksum.",
        rebuild,
    )

    index = read_index(folder) if (folder / INDEX_FILE).exists() else []
    index_counts = Counter(r.partition for r in index)
    aligned = all(
        index_counts.get(p, 0) == lines_per_partition.get(p, -1) == manifest.counts.partitions[p]
        for p in PARTITIONS
    ) and all(
        [r.line for r in index if r.partition == p] == list(range(1, index_counts[p] + 1))
        for p in PARTITIONS
    )
    add(
        "counts",
        aligned,
        "Partition files, index and manifest agree: "
        + ", ".join(f"{p} {manifest.counts.partitions[p]}" for p in PARTITIONS)
        + ".",
        rebuild,
    )

    c = manifest.counts
    reconciled = c.included + sum(c.excluded.values()) == c.input_records and c.included == sum(
        c.partitions.values()
    )
    add(
        "reconcile",
        reconciled,
        f"{c.input_records} input records = {c.included} included + "
        f"{sum(c.excluded.values())} excluded.",
        rebuild,
    )

    group_parts: dict[str, set[str]] = defaultdict(set)
    for r in index:
        group_parts[r.group_key].add(r.partition)
    spanning = sum(1 for parts in group_parts.values() if len(parts) > 1)
    add(
        "group_leakage",
        spanning == 0,
        "No group spans two partitions."
        if spanning == 0
        else f"{spanning} group(s) span partitions.",
        rebuild,
    )
    refs = Counter((r.source_id, r.record_id) for r in index)
    repeated = sum(1 for n in refs.values() if n > 1)
    add(
        "record_overlap",
        repeated == 0,
        "Every record appears once." if repeated == 0 else f"{repeated} record(s) appear twice.",
        rebuild,
    )

    leaking = sum(1 for parts in questions.values() if len(parts) > 1)
    add(
        "cross_partition_duplicates",
        leaking == 0,
        "No identical normalized question appears in two partitions."
        if leaking == 0
        else f"{leaking} normalized question(s) appear in more than one partition.",
        "Put related records in one group (group or thread column), then rebuild.",
        warn=True,
    )

    thresholds = manifest.config.get("thresholds", {})
    min_train = thresholds.get("min_train_examples_warn", 200)
    min_test = thresholds.get("min_test_examples_conclusive", 50)
    add(
        "train_size",
        c.partitions["train"] >= min_train,
        f"{c.partitions['train']} training examples (warning below {min_train}).",
        "Approve more records; with few examples expect style-level gains only.",
        warn=True,
    )
    add(
        "test_size",
        c.partitions["test"] >= min_test,
        f"{c.partitions['test']} test examples (verdict is 'inconclusive' below {min_test}).",
        "Approve more records so the test set can support a verdict.",
        warn=True,
    )

    return ValidateDatasetResult(
        dataset_id=dataset_id,
        name=manifest.name,
        path=str(folder),
        ok=all(ch.status is not CheckStatus.FAIL for ch in checks),
        checks=checks,
    )


def _line_problem(line: str, system_prompt: str) -> str | None:
    """Why a partition line is malformed (never echoes content)."""
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return "not valid JSON"
    messages = data.get("messages") if isinstance(data, dict) else None
    if not isinstance(messages, list) or len(messages) != 3:
        return "expected exactly 3 messages"
    if [m.get("role") if isinstance(m, dict) else None for m in messages] != _ROLES:
        return "roles must be system, user, assistant"
    contents = [m.get("content") for m in messages]
    if not all(isinstance(text, str) for text in contents):
        return "message content must be text"
    if contents[0] != system_prompt:
        return "system prompt differs from the manifest"
    if not contents[1].strip() or not contents[2].strip():
        return "empty question or answer"
    return None


class DatasetInfo(BaseModel):
    id: str
    name: str
    status: str
    created_at: str
    updated_at: str
    stats: dict[str, Any] | None


def list_datasets(home: str | None) -> list[DatasetInfo]:
    from fliptuned.config.app_home import AppPaths, resolve_app_home
    from fliptuned.registry.db import migrate

    paths = AppPaths(resolve_app_home(home).path)
    paths.ensure()
    migrate(paths.db)
    with connect(paths.db) as conn:
        rows = repositories.list_datasets(conn)
    return [
        DatasetInfo(
            id=r["id"],
            name=r["name"],
            status=r["status"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            stats=json.loads(r["stats_json"]) if r["stats_json"] else None,
        )
        for r in rows
    ]
