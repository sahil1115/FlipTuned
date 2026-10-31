"""Command-line interface: thin adapters over ``fliptuned.services`` (no business logic here)."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import Annotated, TypeVar

import typer

from fliptuned.services import audit as audit_service
from fliptuned.services import chat as chat_service
from fliptuned.services import configs as configs_service
from fliptuned.services import datasets as datasets_service
from fliptuned.services import doctor as doctor_service
from fliptuned.services import evaluation as evaluation_service
from fliptuned.services import export as export_service
from fliptuned.services import ingest as ingest_service
from fliptuned.services import jobs as jobs_service
from fliptuned.services import models as models_service
from fliptuned.services import review as review_service
from fliptuned.services import runtime
from fliptuned.services import scan as scan_service
from fliptuned.services import status as status_service
from fliptuned.services import training as training_service
from fliptuned.services.errors import ExitCode, FlipTunedError

app = typer.Typer(
    name="fliptuned",
    help="FlipTuned: fine-tune a small local model on approved Q&A. Everything stays on this machine.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)

HomeOption = Annotated[
    str | None,
    typer.Option(
        "--home",
        help="App home folder (overrides FLIPTUNED_HOME; default ~/.fliptuned).",
        show_default=False,
    ),
]
JsonOption = Annotated[bool, typer.Option("--json", help="Print the result as JSON.")]
VerboseOption = Annotated[bool, typer.Option("--verbose", "-v", help="Show more log output.")]

T = TypeVar("T")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"fliptuned {runtime.version()}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Show the version."
        ),
    ] = False,
) -> None:
    """FlipTuned command-line interface."""


def _call[T](fn: Callable[[], T], verbose: bool = False) -> T:
    runtime.init_entry_point(verbose)
    try:
        return fn()
    except FlipTunedError as exc:
        _print_error(exc)
        raise typer.Exit(int(exc.exit_code)) from None
    except KeyboardInterrupt:
        typer.echo("Interrupted.", err=True)
        raise typer.Exit(int(ExitCode.INTERRUPTED)) from None


def _print_error(exc: FlipTunedError) -> None:
    typer.echo(f"Error: {exc.message}", err=True)
    if exc.hint:
        typer.echo(f"What to do: {exc.hint}", err=True)
    if exc.details:
        typer.echo(f"Details:\n{exc.details}", err=True)


def _print_json(model) -> None:
    typer.echo(json.dumps(model.model_dump(mode="json"), indent=2, ensure_ascii=False))


DetachOption = Annotated[
    bool,
    typer.Option("--detach", help="Run in the background and print the job ID instead of waiting."),
]


def _print_progress(event: dict) -> None:
    if event.get("type") == "train":
        typer.echo(f"  {_train_line(event)}", err=True)
        return
    if event.get("type") != "progress" or not event.get("message"):
        return  # per-step training progress is summarised by the train lines
    typer.echo(f"  {event['done']}/{event['total']} {event['message']}", err=True)


def _train_line(event: dict) -> str:
    parts = [f"step {event.get('step')}/{event.get('max_steps')}"]
    if event.get("loss") is not None:
        parts.append(f"loss {event['loss']:.4f}")
    if event.get("eval_loss") is not None:
        parts.append(f"validation loss {event['eval_loss']:.4f}")
    if event.get("eta_s") is not None:
        parts.append(f"ETA {int(event['eta_s'] // 60)} min {int(event['eta_s'] % 60)} s")
    if event.get("memory_gb"):
        parts.append(f"memory {event['memory_gb']:.2f} GB")
    return "  ".join(parts)


def _run_job(start, model, home: str | None, *, detach: bool, as_json: bool, verbose: bool):
    """Start a job; wait for it in this process (default) or leave it running (--detach).

    Waiting returns the typed result; a failed job exits with the failing step's exit code.
    """
    job = _call(start, verbose)
    if detach:
        if as_json:
            _print_json(job)
        else:
            typer.echo(f"Job           {job.id} ({job.type}) started in the background")
            typer.echo(f"Follow it     fliptuned jobs follow {job.id}")
        raise typer.Exit()
    typer.echo(f"Job {job.id} ({job.type})", err=True)
    _call(lambda: jobs_service.run_inline(home, job.id, _print_progress), verbose)
    return _call(lambda: jobs_service.job_result(home, job.id, model), verbose)


# --------------------------------------------------------------------------- doctor


@app.command()
def doctor(
    home: HomeOption = None,
    config: Annotated[
        str | None,
        typer.Option(
            "--config",
            help="Training config file: also check its backend/device/precision choices on this machine.",
            show_default=False,
        ),
    ] = None,
    suggest_install: Annotated[
        bool,
        typer.Option(
            "--suggest-install", help="Print only the `uv sync` command for this machine."
        ),
    ] = False,
    as_json: JsonOption = False,
    verbose: VerboseOption = False,
) -> None:
    """Check this machine: OS, hardware, installed ML packages and which backend will be used.

    Never opens your data or app-home folders.
    """
    report = _call(
        lambda: doctor_service.run_doctor(
            doctor_service.DoctorRequest(home=home, train_config=config)
        ),
        verbose,
    )
    if suggest_install:
        command = report.backend.install_command
        if command is None:
            typer.echo("No supported install for this machine.", err=True)
            for note in report.backend.warnings:
                typer.echo(f"  {note}", err=True)
            raise typer.Exit(int(ExitCode.NOT_AVAILABLE))
        typer.echo(command)
        typer.echo("(add `--extra dev` for development and tests)", err=True)
        return
    if as_json:
        _print_json(report)
    else:
        _render_doctor(report)
    if report.backend.selected is None:
        raise typer.Exit(int(ExitCode.CHECK_FAILED))


def _render_doctor(r: doctor_service.DoctorReport) -> None:
    m = r.machine
    out = typer.echo
    out(f"FlipTuned {r.fliptuned_version} — environment check (no data folders opened)\n")
    out("System")
    out(f"  OS             {m.os_version}")
    out(f"  CPU            {m.cpu_name} ({m.arch}, {m.cpu_count_logical} logical cores)")
    emulation = "emulated!" if m.emulated else "native"
    out(f"  Python         {m.python_version} ({m.python_arch}, {emulation})")
    out(f"  RAM            {m.ram_gb} GB")
    out("Accelerator")
    if m.nvidia_gpus:
        for gpu in m.nvidia_gpus:
            out(
                f"  NVIDIA         {gpu.name}, {gpu.memory_total_gb} GB VRAM "
                f"({gpu.memory_free_gb} GB free), driver {gpu.driver_version}, "
                f"compute {gpu.compute_capability}"
            )
    elif m.apple_chip:
        out(f"  Apple          {m.apple_chip}, {m.unified_memory_gb} GB unified memory")
    else:
        out("  none           no NVIDIA GPU or Apple Silicon found")
    for note in m.notes:
        out(f"  note           {note}")
    out("Packages")
    for name, version in r.packages.items():
        out(f"  {name:<14} {version or 'not installed'}")
    if r.torch.installed:
        cuda = f"CUDA {r.torch.cuda_runtime}" if r.torch.cuda_runtime else "no CUDA"
        out(
            f"  torch build    {r.torch.build} ({cuda}, GPU usable: {'yes' if r.torch.cuda_available else 'no'})"
        )
    out("Paths")
    out(f"  App home       {r.app_home} (from {r.app_home_source}; not opened)")
    out(f"  HF cache       {r.hf_cache}")
    for disk in r.disks:
        free = f"{disk.free_gb} GB free" if disk.free_gb is not None else "free space unknown"
        out(f"  Disk           {disk.label}: {free} on {disk.drive}")
    out("Backend")
    if any(v != "auto" for v in r.requested.values()):
        out(
            f"  Requested      backend={r.requested['backend']} device={r.requested['device']} "
            f"precision={r.requested['precision']}"
        )
    sel = r.backend.selected
    if sel:
        out(f"  Will use       {sel.backend.value} on {sel.device} ({sel.precision})")
        out(f"  Why            {sel.reason}")
    else:
        out("  Will use       nothing — no usable backend")
    for option in r.backend.options:
        mark = "yes" if option.available else "no "
        out(f"  [{mark}]          {option.backend.value}/{option.device}: {option.reason}")
    if r.warnings:
        out("Warnings")
        for warning in r.warnings:
            out(f"  ! {warning}")
    if r.backend.suggestions:
        out("Suggestions")
        for suggestion in dict.fromkeys(r.backend.suggestions):
            out(f"  → {suggestion}")
    if r.backend.install_command:
        out("Install command for this machine")
        out(f"  {r.backend.install_command}")


# --------------------------------------------------------------------------- status


@app.command()
def status(
    home: HomeOption = None, as_json: JsonOption = False, verbose: VerboseOption = False
) -> None:
    """Show what's in the app home: datasets, runs, jobs and disk use (creates it if missing)."""
    result = _call(
        lambda: status_service.get_status(status_service.StatusRequest(home=home)), verbose
    )
    if as_json:
        _print_json(result)
        return
    out = typer.echo
    out(f"App home       {result.app_home} (from {result.app_home_source})")
    out(f"Registry       {result.db_path} (schema v{result.schema_version})")
    if result.migrated_from is not None:
        out(
            f"Migrated       from v{result.migrated_from}; backup: {result.backup_path or 'none (new database)'}"
        )
    out("Counts")
    for table, count in result.counts.items():
        out(f"  {table:<14} {count}")
    out("Active jobs")
    if result.active_jobs:
        for job in result.active_jobs:
            out(f"  {job.id}  {job.type}  {job.status}  since {job.created_at}")
    else:
        out("  none")
    out("Disk use (MB)")
    for name, size in result.disk_use_mb.items():
        out(f"  {name:<14} {size}")


# --------------------------------------------------------------------------- scan


@app.command()
def scan(
    folder: Annotated[str, typer.Argument(help="Folder to inspect.", show_default=False)],
    name: Annotated[
        str | None,
        typer.Option(
            "--name", help="Data source name (default: the folder name).", show_default=False
        ),
    ] = None,
    home: HomeOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Inspect a folder of Q&A files without reading their content. Prints counts only.

    File and column names are shown in the web UI, not here.
    """
    result = _call(
        lambda: scan_service.scan_folder(
            scan_service.ScanRequest(path=folder, name=name, home=home)
        ),
        verbose,
    )
    counts = scan_service.scan_counts(result)
    out = typer.echo
    out(f"Data source    {result.data_source_id}")
    out("Files")
    for key, value in counts.items():
        out(f"  {key.replace('_', ' '):<30} {value}")
    out("By type        count      bytes")
    for ext, stats in result.by_type.items():
        out(f"  {ext:<12} {stats.count:>5} {stats.bytes:>10}")


# --------------------------------------------------------------------------- ingest


@app.command()
def ingest(
    config: Annotated[
        str, typer.Option("--config", help="Ingest config file (YAML).", show_default=False)
    ],
    home: HomeOption = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace records from an earlier ingest of this source."),
    ] = False,
    detach: DetachOption = False,
    as_json: JsonOption = False,
    verbose: VerboseOption = False,
) -> None:
    """Read Q&A files into normalized records, with quality and secret checks (runs as a job)."""
    request = ingest_service.IngestRequest(config_path=config, home=home, overwrite=overwrite)
    result = _run_job(
        lambda: jobs_service.start_ingest(request, detach=detach),
        ingest_service.IngestResult,
        home,
        detach=detach,
        as_json=as_json,
        verbose=verbose,
    )
    if as_json:
        _print_json(result.counts)
        return
    out = typer.echo
    c = result.counts
    out(f"Data source    {result.name} ({result.data_source_id})")
    out(f"Records        {c.records_total}  (excluded from training: {c.excluded})")
    _section("By type", c.by_type)
    _section("By status", c.by_status)
    _section("Quality flags", c.quality_flags)
    out("Files")
    out(f"  {'read':<28} {c.files_read}")
    out(f"  {'with errors':<28} {c.files_with_errors}")
    out(f"  {'unsupported':<28} {c.unsupported_files}")
    out(f"  {'skipped (hidden/lock)':<28} {c.skipped_files}")
    out(f"  {'blank rows skipped':<28} {c.blank_rows_skipped}")
    out(f"  {'rows not ingested':<28} {c.rows_not_ingested}")
    for file in result.outcome.files:
        for error in file.errors:
            out(f"  ! {file.source_file}: {error}")
        if file.invalid_encoding:
            out(
                f"  ! {file.source_file}: not valid in the configured encoding (records quarantined)"
            )
    out(f"Privacy findings ({result.outcome.privacy_policy.value} policy): {c.privacy_findings}")
    for f in result.outcome.findings:
        out(f"  - {f.source_file} ({f.location}, {f.field}): {f.type} {f.redacted}")
    out(f"Records file   {result.records_path}")


# --------------------------------------------------------------------------- data-audit


@app.command("data-audit")
def data_audit(
    source: Annotated[
        str | None,
        typer.Option("--source", help="Only this data source (name or ID).", show_default=False),
    ] = None,
    processed: Annotated[
        str | None,
        typer.Option(
            "--processed",
            help="Audit a built dataset (ID or name) instead of the ingested records.",
            show_default=False,
        ),
    ] = None,
    home: HomeOption = None,
    as_json: JsonOption = False,
    verbose: VerboseOption = False,
) -> None:
    """Statistics about ingested records or a built dataset: counts, lengths, flags, never content."""
    if processed is not None:
        _processed_audit(processed, home, as_json, verbose)
        return
    result = _call(
        lambda: audit_service.run_audit(audit_service.AuditRequest(home=home, source=source)),
        verbose,
    )
    if as_json:
        _print_json(result)
        return
    out = typer.echo
    out(f"Sources        {len(result.sources)}")
    out(f"Records        {result.records_total}")
    for title, data in (
        ("By type", result.by_type),
        ("By format", result.by_format),
        ("By origin", result.by_origin),
        ("By status", result.by_status),
        ("By category (qa pairs)", result.by_category),
    ):
        _section(title, data)
    g = result.groups
    out("Groups (qa pairs)")
    out(f"  {'in groups of 2+':<28} {g.grouped_records}")
    out(f"  {'distinct groups':<28} {g.distinct_groups}")
    out(f"  {'largest group':<28} {g.largest_group_size} ({g.largest_group_share:.1%} of qa pairs)")
    for key, value in g.by_group_source.items():
        out(f"  {'source: ' + key:<28} {value}")
    out(f"Lengths (qa pairs)           {'p50':>5} {'p90':>5} {'p99':>5} {'max':>5}")
    for label, stats in (
        ("question chars", result.question_chars),
        ("answer chars", result.answer_chars),
        ("question words", result.question_words),
        ("answer words", result.answer_words),
    ):
        if stats:
            out(f"  {label:<26} {stats.p50:>5} {stats.p90:>5} {stats.p99:>5} {stats.max:>5}")
    out(f"  note: {result.length_note}")
    _section("Quality flags", result.quality_flags)
    _section("Privacy findings by type", result.privacy_findings_by_type)
    out(f"  {'records with findings':<28} {result.records_with_privacy_findings}")
    _section("Exclusions", result.exclusions)


def _processed_audit(dataset: str, home: str | None, as_json: bool, verbose: bool) -> None:
    result = _call(
        lambda: audit_service.run_processed_audit(
            audit_service.ProcessedAuditRequest(home=home, dataset=dataset)
        ),
        verbose,
    )
    if as_json:
        _print_json(result)
        return
    out = typer.echo
    out(f"Dataset        {result.name} ({result.dataset_id})")
    out(f"Tokenizer      {result.model} (max {result.max_seq_length} tokens)")
    out(f"Records        {result.input_records} in, {result.included} included")
    out(f"Partitions     {'examples':>8} {'share':>7} {'target':>7} {'groups':>7} {'largest':>8}")
    for name, p in result.partitions.items():
        out(
            f"  {name:<12} {p.examples:>8} {result.split_actual[name]:>7.1%} "
            f"{result.split_target[name]:>7.0%} {p.groups:>7} {p.largest_group_size:>8}"
        )
    out(f"Tokens         {'p50':>6} {'p90':>6} {'p99':>6} {'max':>6} {'truncated':>10}")
    for name, p in result.partitions.items():
        t = p.tokens
        cells = [t.p50, t.p90, t.p99, t.max] if t else ["-"] * 4
        out(f"  {name:<12} " + " ".join(f"{c:>6}" for c in cells) + f" {p.truncated:>10}")
    out(
        f"Largest group  {result.largest_group_size} examples "
        f"({result.largest_group_share:.1%} of included)"
    )
    for name, p in result.partitions.items():
        _section(f"By category: {name}", p.by_category)
    for name, p in result.partitions.items():
        _section(f"By group source: {name}", p.by_group_source)
    _section("Excluded", result.excluded)
    _warnings(result.warnings)


def _warnings(warnings) -> None:
    typer.echo("Warnings")
    if not warnings:
        typer.echo("  none")
    for w in warnings:
        typer.echo(f"  ! {w.message}")


# --------------------------------------------------------------------------- review

review_app = typer.Typer(
    help="Approve or reject records through a CSV sheet (the web UI uses the same service).",
    no_args_is_help=True,
)
app.add_typer(review_app, name="review")

SourceFilterOption = Annotated[
    list[str] | None,
    typer.Option(
        "--source", help="Only this data source (name or ID); repeatable.", show_default=False
    ),
]


@review_app.command("export")
def review_export(
    out: Annotated[str, typer.Option("--out", help="CSV file to write.", show_default=False)],
    source: SourceFilterOption = None,
    status: Annotated[
        list[review_service.RecordStatus] | None,
        typer.Option("--status", help="Only this status; repeatable.", show_default=False),
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace the CSV file if it exists.")
    ] = False,
    home: HomeOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Write records to a CSV sheet. Fill in 'decision' (approved/rejected/pending_review), then import it.

    Secrets found by the privacy scan are redacted in the sheet.
    """
    result = _call(
        lambda: review_service.export_csv(
            review_service.ExportRequest(
                home=home,
                out=out,
                overwrite=overwrite,
                filter=review_service.ReviewFilter(sources=source or [], statuses=status or []),
            )
        ),
        verbose,
    )
    typer.echo(f"Wrote          {result.rows} record(s) to {result.path}")
    _section("By current status", result.by_status)


@review_app.command("import")
def review_import(
    file: Annotated[
        str, typer.Argument(help="CSV sheet from `review export`.", show_default=False)
    ],
    home: HomeOption = None,
    as_json: JsonOption = False,
    verbose: VerboseOption = False,
) -> None:
    """Apply the decisions in a review CSV sheet. Only 'decision' and 'reason' are read."""
    result = _call(
        lambda: review_service.import_csv(review_service.ImportRequest(home=home, path=file)),
        verbose,
    )
    if as_json:
        _print_json(result)
        return
    out = typer.echo
    out(f"Rows           {result.rows}")
    _section("Applied", result.applied)
    out(f"  {'unchanged':<28} {result.unchanged}")
    out(f"  {'no decision':<28} {result.blank}")
    _section("Skipped", result.skipped)
    for problem in result.problems[:20]:
        out(f"  ! row {problem.row}: {problem.reason}")
    if len(result.problems) > 20:
        out(f"  ... and {len(result.problems) - 20} more (use --json)")


# --------------------------------------------------------------------------- datasets


@app.command("build-dataset")
def build_dataset(
    config: Annotated[
        str, typer.Option("--config", help="Dataset config file (YAML).", show_default=False)
    ],
    allow_download: Annotated[
        bool,
        typer.Option(
            "--allow-download", help="Download the model's tokenizer files if they aren't cached."
        ),
    ] = False,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Rebuild a dataset that already exists.")
    ] = False,
    home: HomeOption = None,
    detach: DetachOption = False,
    as_json: JsonOption = False,
    verbose: VerboseOption = False,
) -> None:
    """Build train/valid/test files from approved records: dedup, group, split, length check
    (runs as a job)."""
    request = datasets_service.BuildDatasetRequest(
        config_path=config, home=home, allow_download=allow_download, overwrite=overwrite
    )
    result = _run_job(
        lambda: jobs_service.start_build_dataset(request, detach=detach),
        datasets_service.BuildDatasetResult,
        home,
        detach=detach,
        as_json=as_json,
        verbose=verbose,
    )
    if as_json:
        _print_json(result)
        return
    out = typer.echo
    c = result.counts
    rebuilt = " (rebuilt)" if result.rebuilt else ""
    out(f"Dataset        {result.name} ({result.dataset_id}){rebuilt}")
    out(
        f"Records        {c.input_records} in, {c.included} included, "
        f"{sum(c.excluded.values())} excluded"
    )
    out(f"Partitions     {'examples':>8} {'share':>7} {'target':>7} {'groups':>7}")
    for name, count in c.partitions.items():
        out(
            f"  {name:<12} {count:>8} {result.split.actual[name]:>7.1%} "
            f"{result.split.target[name]:>7.0%} {result.split.groups[name]:>7}"
        )
    _section("Excluded", c.excluded)
    g = result.groups
    out(f"Groups         {g.total} (largest {g.largest_size}, {g.largest_share:.1%})")
    _warnings(result.warnings)
    out(f"Folder         {result.path}")


@app.command("validate-dataset")
def validate_dataset(
    dataset: Annotated[str, typer.Argument(help="Dataset ID or name.", show_default=False)],
    home: HomeOption = None,
    as_json: JsonOption = False,
    verbose: VerboseOption = False,
) -> None:
    """Check a built dataset: checksums, format, counts, group leakage and size warnings."""
    result = _call(
        lambda: datasets_service.validate_dataset(
            datasets_service.ValidateDatasetRequest(home=home, dataset=dataset)
        ),
        verbose,
    )
    if as_json:
        _print_json(result)
    else:
        typer.echo(f"Dataset        {result.name} ({result.dataset_id})")
        for check in result.checks:
            typer.echo(f"  [{check.status.value}] {check.name}: {check.message}")
            if check.fix:
                typer.echo(f"         fix: {check.fix}")
        typer.echo("Result         " + ("valid" if result.ok else "INVALID"))
    if not result.ok:
        raise typer.Exit(int(ExitCode.CHECK_FAILED))


# --------------------------------------------------------------------------- models

models_app = typer.Typer(help="The curated model catalog.", no_args_is_help=True)
app.add_typer(models_app, name="models")


@models_app.command("list")
def models_list(
    include_test_only: Annotated[
        bool, typer.Option("--all", help="Also list tiny test-only models.")
    ] = False,
    home: HomeOption = None,
    as_json: JsonOption = False,
    verbose: VerboseOption = False,
) -> None:
    """List catalog models with what this machine can do with them (memory figures are estimates).

    Reads only the local Hugging Face cache; never downloads.
    """
    result = _call(
        lambda: models_service.list_models(
            models_service.ModelListRequest(include_test_only=include_test_only)
        ),
        verbose,
    )
    if as_json:
        _print_json(result)
        return
    out = typer.echo
    out(
        f"{'Model':<20} {'Params':>7} {'Backend here':<20} {'Memory (est.)':<22} "
        f"{'Cached':<7} {'Download':>9}  Trainable"
    )
    for m in result.models:
        sel = m.selection
        backend = f"{sel.backend.value}/{sel.device}" if sel else "none"
        c = m.compatibility
        memory = (
            f"{c.label} ({c.estimate_gb:g}/{c.pool_gb:.0f} GB)"
            if c and c.estimate_gb is not None and c.pool_gb is not None
            else (c.label if c else "-")
        )
        download = f"{m.download_gb:g} GB" if m.download_gb is not None else "-"
        out(
            f"{m.id:<20} {m.params_b:>6g}B {backend:<20} {memory:<22} "
            f"{'yes' if m.cached else 'no':<7} {download:>9}  {'yes' if m.trainable else 'no'}"
        )
        for note in m.notes:
            out(f"  ! {note}")
    out(f"Note: {result.estimate_note}")


# --------------------------------------------------------------------------- evaluate


def _usage_error(message: str, hint: str | None = None) -> None:
    typer.echo(f"Error: {message}", err=True)
    if hint:
        typer.echo(f"What to do: {hint}", err=True)
    raise typer.Exit(int(ExitCode.USAGE))


@app.command()
def evaluate(
    run: Annotated[
        str | None,
        typer.Option(
            "--run",
            help="Training run to evaluate against its base model (blind review + verdict).",
            show_default=False,
        ),
    ] = None,
    score: Annotated[
        str | None,
        typer.Option(
            "--score",
            help="Record the judgements (A, B or tie) from a filled-in review_sheet.csv of this "
            "run's evaluation, then compute the verdict.",
            show_default=False,
        ),
    ] = None,
    baseline_only: Annotated[
        bool,
        typer.Option(
            "--baseline-only",
            help="Only generate base-model answers for a dataset's test set (with --dataset).",
        ),
    ] = False,
    dataset: Annotated[
        str | None,
        typer.Option("--dataset", help="Dataset ID or name (--baseline-only).", show_default=False),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="Catalog model ID (--baseline-only; default: the model the dataset was built "
            "for).",
            show_default=False,
        ),
    ] = None,
    backend: Annotated[
        str, typer.Option("--backend", help="auto, mlx or hf_peft (--baseline-only).")
    ] = "auto",
    device: Annotated[
        str, typer.Option("--device", help="auto, cuda or cpu (--baseline-only).")
    ] = "auto",
    precision: Annotated[
        str,
        typer.Option("--precision", help="auto, qlora_4bit, bf16, fp16 or fp32 (--baseline-only)."),
    ] = "auto",
    config: Annotated[
        str | None,
        typer.Option(
            "--config",
            help="Evaluate config (YAML): acceptance criteria and generation settings.",
            show_default=False,
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            min=1,
            help="Only the first N test items (a quick look; a verdict needs the full set).",
            show_default=False,
        ),
    ] = None,
    allow_download: Annotated[
        bool, typer.Option("--allow-download", help="Download the model files if missing.")
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Regenerate an existing baseline (--baseline-only)."),
    ] = False,
    detach: DetachOption = False,
    home: HomeOption = None,
    as_json: JsonOption = False,
    verbose: VerboseOption = False,
) -> None:
    """Compare a trained run with its base model (blind review and verdict), or make a baseline."""
    if baseline_only:
        if run or score or detach:
            _usage_error("--baseline-only can't be combined with --run, --score or --detach.")
        if dataset is None:
            _usage_error("--dataset is required with --baseline-only.")
        _baseline(
            evaluation_service.BaselineRequest(
                home=home,
                dataset=dataset,
                model=model,
                backend=backend,
                device=device,
                precision=precision,
                config_path=config,
                limit=limit,
                allow_download=allow_download,
                overwrite=overwrite,
            ),
            as_json=as_json,
            verbose=verbose,
        )
        return
    if run is None:
        _usage_error(
            "--run is required.",
            hint="Pass the training run to evaluate (`fliptuned jobs list` shows run IDs), or use "
            "--baseline-only --dataset <id>.",
        )
    if dataset or model or overwrite or (backend, device, precision) != ("auto",) * 3:
        _usage_error(
            "--dataset, --model, --backend, --device, --precision and --overwrite only apply "
            "with --baseline-only.",
            hint="A run is evaluated with its own dataset, model and backend.",
        )
    if score is not None:
        if detach or limit or allow_download or config:
            _usage_error("--score only reads the review sheet; drop the other options.")
        _score_sheet(home, run, score, as_json=as_json, verbose=verbose)
        return
    request = evaluation_service.EvaluateRequest(
        home=home, run_id=run, config_path=config, limit=limit, allow_download=allow_download
    )
    start = _call(lambda: evaluation_service.start_evaluation(request, detach=detach), verbose)
    if detach:
        if as_json:
            _print_json(start)
        else:
            typer.echo(f"Evaluation    {start.evaluation_id}")
            typer.echo(f"Job           {start.job.id} (evaluate) started in the background")
            typer.echo(f"Follow it     fliptuned jobs follow {start.job.id}")
        raise typer.Exit()
    typer.echo(f"Evaluation {start.evaluation_id}, job {start.job.id} (evaluate)", err=True)
    _call(lambda: jobs_service.run_inline(home, start.job.id, _print_progress), verbose)
    outcome = _call(
        lambda: jobs_service.job_result(home, start.job.id, evaluation_service.EvaluationOutcome),
        verbose,
    )
    if as_json:
        _print_json(outcome)
        return
    out = typer.echo
    out(f"Evaluation     {outcome.evaluation_id} of run {outcome.run_id}")
    reused = "reused" if outcome.baseline_reused else "generated and saved"
    out(f"Base answers   baseline {outcome.baseline_id} ({reused})")
    out(f"Test items     {outcome.test_items}")
    out(f"Regression set {outcome.regression_items} prompts")
    _render_score(outcome.score)
    out(f"Review sheet   {outcome.review_sheet}")
    out(
        "Next: review the pairs blind. Fill the 'judgement' column with A, B or tie, then run "
        f"`fliptuned evaluate --run {outcome.run_id} --score <sheet>`."
    )


def _baseline(request, *, as_json: bool, verbose: bool) -> None:
    def progress(done: int, total: int) -> None:
        typer.echo(f"  item {done}/{total}", err=True)

    result = _call(
        lambda: evaluation_service.run_baseline(request, progress=None if as_json else progress),
        verbose,
    )
    if as_json:
        _print_json(result)
        return
    out = typer.echo
    rebuilt = " (regenerated)" if result.rebuilt else ""
    out(f"Baseline       {result.baseline_id}{rebuilt}")
    out(f"Dataset        {result.dataset_id} (test split)")
    out(f"Model          {result.model} on {result.backend}/{result.device} ({result.precision})")
    out(f"Items          {result.items}")
    _section("Finish reasons", result.finish_reasons)
    out(f"Tokens         {result.completion_tokens} generated in {result.seconds:g} s")
    for warning in result.warnings:
        out(f"  ! {warning}")
    out(f"Folder         {result.path}")


def _score_sheet(home: str | None, run: str, sheet: str, *, as_json: bool, verbose: bool) -> None:
    imported = _call(
        lambda: evaluation_service.import_review_sheet(
            evaluation_service.SheetImportRequest(home=home, path=sheet, run_id=run)
        ),
        verbose,
    )
    result = _call(
        lambda: evaluation_service.score_evaluation(
            evaluation_service.ScoreRequest(home=home, evaluation_id=imported.evaluation_id)
        ),
        verbose,
    )
    if as_json:
        payload = {
            "import": imported.model_dump(mode="json"),
            "score": result.model_dump(mode="json"),
        }
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    out = typer.echo
    recorded = ", ".join(f"{k} {v}" for k, v in imported.recorded.items()) or "none"
    out(f"Review sheet   {imported.rows} rows: recorded {recorded}; {imported.blank} blank")
    if imported.skipped:
        _section("Skipped rows", imported.skipped)
        for problem in imported.problems[:20]:
            out(f"  row {problem.row}: {problem.reason}")
    out(f"Evaluation     {result.evaluation_id} of run {result.run_id}")
    _render_score(result)


def _render_score(r) -> None:
    out = typer.echo
    out(f"Review         {r.judged} of {r.test_items} pairs reviewed ({r.review_coverage:.0%})")
    out(
        f"Pairs          fine-tuned wins {r.wins}, base wins {r.losses}, ties {r.ties}; "
        f"sign test p = {r.sign_test_p:.4g}"
    )
    out(
        f"Checks         test set: base {r.deterministic_pass_base:.1f}%, "
        f"fine-tuned {r.deterministic_pass_tuned:.1f}% of items pass every check"
    )
    if r.regression_pass_base is not None:
        out(
            f"Regression set base {r.regression_pass_base:.1f}%, "
            f"fine-tuned {r.regression_pass_tuned:.1f}%"
        )
    out(f"Verdict        {r.verdict.upper()}")
    for reason in r.reasons:
        out(f"  - {reason}")
    out("Criteria")
    for c in r.criteria:
        state = "n/a" if not c.applicable else ("met" if c.met else "NOT MET")
        gives = f" (gives {c.stage})" if c.applicable and not c.met else ""
        out(f"  [{state}] {c.description}: {c.display_value}{gives}")
    if r.categories:
        out("Categories     items / reviewed / fine-tuned wins / base wins / ties")
        for cat in r.categories:
            out(
                f"  {cat.category:<20} {cat.items:>4} {cat.judged:>4} {cat.wins:>4} "
                f"{cat.losses:>4} {cat.ties:>4}"
            )
    if r.regressions:
        out(f"Regressions    {r.regressions} item(s) lost a check the base answer passed")
    if r.prior_exposure_note:
        out(f"  ! {r.prior_exposure_note}")
    out(f"Report         {r.report_path}")


# --------------------------------------------------------------------------- train


def _render_checks(result) -> None:
    for check in result.checks:
        typer.echo(f"  [{check.status.value}] {check.name}: {check.message}")
        if check.fix:
            typer.echo(f"         fix: {check.fix}")


@app.command()
def train(
    config: Annotated[
        str, typer.Option("--config", help="Training config file (YAML).", show_default=False)
    ],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Only run the preflight checks.")
    ] = False,
    allow_download: Annotated[
        bool, typer.Option("--allow-download", help="Download the model files if missing.")
    ] = False,
    detach: DetachOption = False,
    home: HomeOption = None,
    as_json: JsonOption = False,
    verbose: VerboseOption = False,
) -> None:
    """Fine-tune the model with LoRA on the dataset (runs as a job; checks preflight first)."""
    request = training_service.TrainRequest(
        config_path=config, home=home, allow_download=allow_download
    )
    if dry_run:
        result = _call(lambda: training_service.preflight(request), verbose)
        if as_json:
            _print_json(result)
        else:
            typer.echo(f"Preflight for model {result.model} on dataset {result.dataset_id}")
            _render_checks(result)
            typer.echo("Result         " + ("ready to train" if result.ok else "BLOCKED"))
        if not result.ok:
            raise typer.Exit(int(ExitCode.CHECK_FAILED))
        return
    start = _call(lambda: training_service.start_training(request, detach=detach), verbose)
    if not as_json:
        warnings = [c for c in start.preflight.checks if c.status.value == "warn"]
        for check in warnings:
            typer.echo(f"  ! {check.message}", err=True)
    _finish_training(start, home, detach=detach, as_json=as_json, verbose=verbose)


def _finish_training(start, home: str | None, *, detach: bool, as_json: bool, verbose: bool):
    job = start.job
    if detach:
        if as_json:
            _print_json(start)
        else:
            typer.echo(f"Run           {start.run_id}")
            typer.echo(f"Job           {job.id} (train) started in the background")
            typer.echo(f"Follow it     fliptuned jobs follow {job.id}")
        raise typer.Exit()
    typer.echo(f"Run {start.run_id}, job {job.id} (train)", err=True)
    _call(lambda: jobs_service.run_inline(home, job.id, _print_progress), verbose)
    outcome = _call(
        lambda: jobs_service.job_result(home, job.id, training_service.TrainOutcome), verbose
    )
    if as_json:
        _print_json(outcome)
        return
    r = outcome.result
    out = typer.echo
    out(f"Run            {outcome.run_id}: {outcome.status}")
    out(
        f"Updates        {r.steps_done} of at most {r.total_steps}"
        + (" (stopped early)" if r.stopped_early else "")
    )
    if r.best_eval_loss is not None:
        out(f"Best           validation loss {r.best_eval_loss:.4f} at update {r.best_step}")
    else:
        out("Best           no validation ran; the last weights were kept")
    if r.peak_memory_gb is not None:
        out(f"Peak memory    {r.peak_memory_gb:.2f} GB")
    if r.resume_note:
        out(f"Resume         {r.resume_note}")
    out(f"Adapter        {r.adapter_path}")
    out(f"Metadata       {outcome.metadata_path}")
    out("Note: falling loss alone doesn't show the model got better; evaluate it next.")


# --------------------------------------------------------------------------- chat


def _print_reply(session, turns: list[dict[str, str]]) -> str:
    pieces = []
    for piece in session.reply(turns):
        pieces.append(piece)
        typer.echo(piece, nl=False)
        sys.stdout.flush()
    typer.echo("")
    return "".join(pieces)


@app.command()
def chat(
    run: Annotated[
        str, typer.Option("--run", help="Training run to chat with.", show_default=False)
    ],
    base: Annotated[
        bool, typer.Option("--base", help="Chat with the base model instead (no adapter).")
    ] = False,
    prompt: Annotated[
        str | None,
        typer.Option(
            "--prompt", help="Ask one question, print the answer and exit.", show_default=False
        ),
    ] = None,
    allow_download: Annotated[
        bool, typer.Option("--allow-download", help="Download the model files if missing.")
    ] = False,
    home: HomeOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Chat with a run's model. Model weights only: no document retrieval, and nothing is logged."""
    request = chat_service.ChatRequest(
        home=home, run_id=run, base=base, allow_download=allow_download
    )
    session = _call(lambda: chat_service.start_chat(request, detach=False), verbose)
    info = session.info
    try:
        which = "base model (no adapter)" if info.base else "fine-tuned model"
        typer.echo(
            f"Chat with run {info.run_id}: {info.model}, {which}, on {info.backend}/{info.device}.",
            err=True,
        )
        typer.echo("Model weights only: no document retrieval. Nothing is logged.", err=True)
        typer.echo("Loading the model...", err=True)
        _call(session.wait_ready, verbose)
        if prompt is not None:
            _call(lambda: _print_reply(session, [{"role": "user", "content": prompt}]), verbose)
            return
        minutes = info.idle_timeout_s / 60
        typer.echo(
            f"Type /reset to start over, /exit to quit. The model is released after "
            f"{minutes:g} min without a message.",
            err=True,
        )
        turns: list[dict[str, str]] = []
        while True:
            try:
                line = input("you> ")
            except EOFError:
                break
            if line.strip() in ("/exit", "/quit"):
                break
            if line.strip() == "/reset":
                turns = []
                typer.echo("(conversation cleared)", err=True)
                continue
            if not line.strip():
                continue
            turns.append({"role": "user", "content": line})
            typer.echo("model> ", nl=False)
            answer = _call(lambda turns=turns: _print_reply(session, turns), verbose)
            turns.append({"role": "assistant", "content": answer})
    finally:
        session.close()


# --------------------------------------------------------------------------- export


@app.command()
def export(
    run: Annotated[str, typer.Option("--run", help="Training run to export.", show_default=False)],
    merge: Annotated[
        bool,
        typer.Option(
            "--merge",
            help="Also build a standalone model with the adapter merged in (runs as a job; "
            "uses more disk).",
        ),
    ] = False,
    config: Annotated[
        str | None,
        typer.Option("--config", help="Export config (YAML).", show_default=False),
    ] = None,
    allow_download: Annotated[
        bool, typer.Option("--allow-download", help="Download the model files if missing.")
    ] = False,
    detach: DetachOption = False,
    home: HomeOption = None,
    as_json: JsonOption = False,
    verbose: VerboseOption = False,
) -> None:
    """Export a run's adapter with its metadata; optionally a merged standalone model."""
    request = export_service.ExportRequest(
        home=home,
        run_id=run,
        config_path=config,
        merge=True if merge else None,
        allow_download=allow_download,
    )
    start = _call(lambda: export_service.export_run(request, detach=detach), verbose)
    if start.job is None:
        result = start.result
    elif detach:
        if as_json:
            _print_json(start)
        else:
            typer.echo(f"Export        {start.export_id} ({start.path})")
            typer.echo(f"Job           {start.job.id} (merge_export) started in the background")
            typer.echo(f"Follow it     fliptuned jobs follow {start.job.id}")
        raise typer.Exit()
    else:
        typer.echo(f"Export {start.export_id}, job {start.job.id} (merge_export)", err=True)
        _call(lambda: jobs_service.run_inline(home, start.job.id, _print_progress), verbose)
        result = _call(
            lambda: jobs_service.job_result(home, start.job.id, export_service.ExportResult),
            verbose,
        )
    if as_json:
        _print_json(result)
        return
    out = typer.echo
    out(f"Export         {result.export_id} of run {result.run_id} ({result.kind})")
    out(f"Folder         {result.path}")
    out(f"Files          {', '.join(result.files)}")
    if result.check is not None:
        out(f"Merged model   {result.merged_path} ({result.merged_precision})")
        out(
            f"Merge check    {result.check.identical} of {result.check.prompts} answers identical "
            "before and after merging (greedy)"
        )
    for note in result.notes:
        out(f"  ! {note}")


# --------------------------------------------------------------------------- serve


@app.command()
def serve(
    port: Annotated[
        int | None,
        typer.Option(
            "--port",
            min=1,
            max=65535,
            help="Port to use (or the next free one). Default: the Settings port (8765).",
        ),
    ] = None,
    no_browser: Annotated[
        bool, typer.Option("--no-browser", help="Don't open the browser.")
    ] = False,
    home: HomeOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Start the local web app on 127.0.0.1 (only this computer can reach it)."""
    # The one adapter-to-adapter import: the CLI starts the API server (no business logic).
    from fliptuned.api.server import prepare, run

    plan = _call(lambda: prepare(home, port), verbose)
    if plan.reconciled:
        typer.echo(f"Marked {len(plan.reconciled)} job(s) whose process was gone as interrupted.")
    typer.echo(f"FlipTuned is running at http://127.0.0.1:{plan.port}/")
    typer.echo(f"Open this link (it includes this session's access token):\n  {plan.url}")
    typer.echo("Press Ctrl+C to stop.")
    _call(lambda: run(plan, home, open_browser=not no_browser), verbose)


# --------------------------------------------------------------------------- configs

configs_app = typer.Typer(help="Named, versioned training configs.", no_args_is_help=True)
app.add_typer(configs_app, name="configs")


@configs_app.command("list")
def configs_list(
    home: HomeOption = None, as_json: JsonOption = False, verbose: VerboseOption = False
) -> None:
    """Saved training configs (latest version of each). `train` saves its config automatically."""
    rows = _call(
        lambda: configs_service.list_configs(configs_service.ListConfigsRequest(home=home)),
        verbose,
    )
    if as_json:
        typer.echo(json.dumps([r.model_dump(mode="json") for r in rows], indent=2))
        return
    if not rows:
        typer.echo("No saved training configs yet.")
    for r in rows:
        typer.echo(
            f"{r.name:<24} v{r.version} ({r.versions} version(s))  model {r.model}  "
            f"backend {r.backend}  preset {r.preset}  {r.created_at}"
        )


@configs_app.command("show")
def configs_show(
    name: Annotated[str, typer.Argument(help="Config name.", show_default=False)],
    version: Annotated[
        int | None, typer.Option("--version", min=1, help="A specific version (default: latest).")
    ] = None,
    home: HomeOption = None,
    as_json: JsonOption = False,
    verbose: VerboseOption = False,
) -> None:
    """Print a saved training config as YAML."""
    detail = _call(
        lambda: configs_service.show_config(
            configs_service.ShowConfigRequest(home=home, name=name, version=version)
        ),
        verbose,
    )
    if as_json:
        _print_json(detail)
        return
    typer.echo(f"# {detail.name} v{detail.version} ({detail.created_at})")
    typer.echo(detail.yaml.rstrip())


@configs_app.command("duplicate")
def configs_duplicate(
    name: Annotated[str, typer.Argument(help="Config to copy.", show_default=False)],
    new_name: Annotated[str, typer.Argument(help="Name of the copy.", show_default=False)],
    home: HomeOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Copy the latest version of a config under a new name (as its version 1)."""
    saved = _call(
        lambda: configs_service.duplicate_config(
            configs_service.DuplicateConfigRequest(home=home, name=name, new_name=new_name)
        ),
        verbose,
    )
    typer.echo(f"Saved {saved.name} v{saved.version}.")


# --------------------------------------------------------------------------- jobs

jobs_app = typer.Typer(help="Background jobs: list, follow, cancel.", no_args_is_help=True)
app.add_typer(jobs_app, name="jobs")


@jobs_app.command("list")
def jobs_list(
    status: Annotated[
        list[jobs_service.JobStatus] | None,
        typer.Option("--status", help="Only this status; repeatable.", show_default=False),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, help="How many jobs to show.")] = 20,
    home: HomeOption = None,
    as_json: JsonOption = False,
    verbose: VerboseOption = False,
) -> None:
    """Recent jobs, newest first. Jobs whose process is gone are marked interrupted."""
    result = _call(
        lambda: jobs_service.list_jobs(
            jobs_service.ListJobsRequest(home=home, statuses=status or [], limit=limit)
        ),
        verbose,
    )
    if as_json:
        _print_json(result)
        return
    out = typer.echo
    if not result.jobs:
        out("No jobs.")
    for job in result.jobs:
        progress = f"{job.progress:.0%}" if job.progress is not None else "-"
        mode = "background" if job.detached else "foreground"
        out(
            f"{job.id}  {job.type:<14} {job.status:<12} {progress:>5}  {mode:<10}  {job.created_at}"
        )
        if job.error_summary:
            out(f"    {job.error_summary}")
    out(f"Model lock     {result.model_lock_holder or 'free'}")
    for job_id in result.reconciled:
        out(f"  ! {job_id}: its process was gone, marked interrupted")


def _render_event(event: dict) -> None:
    kind = event.get("type")
    stamp = event.get("time", "")
    if kind == "progress":
        message = f" {event['message']}" if event.get("message") else ""
        typer.echo(f"{stamp}  progress {event['done']}/{event['total']}{message}")
    elif kind == "status":
        forced = " (force-stopped)" if event.get("forced") else ""
        typer.echo(f"{stamp}  status   {event['status']}{forced}")
        error = event.get("error") or {}
        if error.get("message"):
            typer.echo(f"Error: {error['message']}", err=True)
        if error.get("hint"):
            typer.echo(f"What to do: {error['hint']}", err=True)
    elif kind == "train":
        typer.echo(f"{stamp}  train    {_train_line(event)}")
    elif kind == "log":
        typer.echo(f"{stamp}  {event.get('level', 'info'):<8} {event.get('message', '')}")


@jobs_app.command("follow")
def jobs_follow(
    job_id: Annotated[str, typer.Argument(help="Job ID.", show_default=False)],
    home: HomeOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Show a job's progress until it finishes (Ctrl+C stops following, not the job)."""
    final = _call(lambda: _follow(home, job_id), verbose)
    if final != "succeeded":
        error = jobs_service.job_error(home, job_id) if final == "failed" else None
        raise typer.Exit(int(error["exit_code"]) if error else int(ExitCode.ERROR))


def _follow(home: str | None, job_id: str) -> str:
    final = "unknown"
    for event in jobs_service.follow_job(home, job_id):
        _render_event(event)
        if event.get("type") == "status":
            final = event["status"]
    return final


@jobs_app.command("cancel")
def jobs_cancel(
    job_id: Annotated[str, typer.Argument(help="Job ID.", show_default=False)],
    timeout: Annotated[
        float, typer.Option("--timeout", min=0, help="Seconds to wait before force-stopping.")
    ] = 10.0,
    home: HomeOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Ask a job to stop; force-stop it if it hasn't stopped within --timeout seconds."""
    result = _call(
        lambda: jobs_service.cancel_job(
            jobs_service.CancelJobRequest(home=home, job_id=job_id, timeout=timeout)
        ),
        verbose,
    )
    if result.already_finished:
        typer.echo(f"Job {job_id} had already finished ({result.status}).")
    else:
        how = "force-stopped" if result.forced else "stopped"
        typer.echo(f"Job {job_id} {how}; status {result.status}.")


@jobs_app.command("resume")
def jobs_resume(
    run_id: Annotated[str, typer.Argument(help="Run ID.", show_default=False)],
    allow_download: Annotated[
        bool, typer.Option("--allow-download", help="Download the model files if missing.")
    ] = False,
    detach: DetachOption = False,
    home: HomeOption = None,
    as_json: JsonOption = False,
    verbose: VerboseOption = False,
) -> None:
    """Continue an interrupted, failed or cancelled training run from its last checkpoint."""
    start = _call(
        lambda: training_service.resume_run(
            training_service.ResumeRequest(home=home, run_id=run_id, allow_download=allow_download),
            detach=detach,
        ),
        verbose,
    )
    _finish_training(start, home, detach=detach, as_json=as_json, verbose=verbose)


@jobs_app.command("selftest")
def jobs_selftest(
    seconds: Annotated[float, typer.Option("--seconds", min=0.1, help="How long it runs.")] = 10.0,
    steps: Annotated[int, typer.Option("--steps", min=1, help="Progress steps to report.")] = 10,
    model_lock: Annotated[
        bool, typer.Option("--model-lock", help="Hold the global model lock while running.")
    ] = False,
    detach: DetachOption = False,
    home: HomeOption = None,
    as_json: JsonOption = False,
    verbose: VerboseOption = False,
) -> None:
    """Run a harmless diagnostic job to check background jobs, progress and cancel on this OS."""
    result = _run_job(
        lambda: jobs_service.start_selftest(
            jobs_service.SelftestRequest(
                home=home, seconds=seconds, steps=steps, hold_model_lock=model_lock
            ),
            detach=detach,
        ),
        jobs_service.SelftestResult,
        home,
        detach=detach,
        as_json=as_json,
        verbose=verbose,
    )
    if as_json:
        _print_json(result)
    else:
        typer.echo(f"Selftest passed: {result.steps} steps in about {result.seconds:g} s.")


def _section(title: str, data: dict[str, int]) -> None:
    typer.echo(title)
    if not data:
        typer.echo("  none")
    for key, value in data.items():
        typer.echo(f"  {key:<28} {value}")


def main() -> None:
    try:
        app()
    except BrokenPipeError:  # e.g. `fliptuned doctor | head`
        sys.exit(0)


if __name__ == "__main__":
    main()
