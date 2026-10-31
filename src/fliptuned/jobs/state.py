"""Job types, statuses and allowed transitions (SPEC §13.3)."""

from __future__ import annotations

from enum import StrEnum


class JobType(StrEnum):
    INGEST = "ingest"
    BUILD_DATASET = "build_dataset"
    MODEL_DOWNLOAD = "model_download"
    TRAIN = "train"
    EVALUATE = "evaluate"
    MERGE_EXPORT = "merge_export"
    CHAT_SESSION = "chat_session"
    SELFTEST = "selftest"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


# Job types that load a model and therefore need the global model lock.
MODEL_LOCK_TYPES = frozenset(
    {JobType.TRAIN, JobType.EVALUATE, JobType.MERGE_EXPORT, JobType.CHAT_SESSION}
)
# Job types that keep the machine awake while they run.
KEEP_AWAKE_TYPES = frozenset({JobType.TRAIN, JobType.EVALUATE})

ACTIVE = (JobStatus.QUEUED, JobStatus.RUNNING)
TERMINAL = frozenset(
    {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.INTERRUPTED}
)

TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.QUEUED: frozenset({JobStatus.RUNNING, JobStatus.CANCELLED, JobStatus.FAILED}),
    JobStatus.RUNNING: frozenset(
        {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.INTERRUPTED}
    ),
    # Terminal states are final; resuming a training run starts a new job (Phase 6).
    JobStatus.SUCCEEDED: frozenset(),
    JobStatus.FAILED: frozenset(),
    JobStatus.CANCELLED: frozenset(),
    JobStatus.INTERRUPTED: frozenset(),
}


def allowed_from(target: JobStatus) -> tuple[str, ...]:
    """Statuses a job may be in for a move to ``target``."""
    return tuple(s.value for s, targets in TRANSITIONS.items() if target in targets)


def can_transition(current: JobStatus, target: JobStatus) -> bool:
    return target in TRANSITIONS[current]
