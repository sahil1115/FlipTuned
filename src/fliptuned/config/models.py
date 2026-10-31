"""Validated configuration models.

Field descriptions are written for non-experts: the web UI shows them next to each field
(via JSON Schema), so they explain *what the setting does and when to change it*.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful technical assistant. Follow the team's answer conventions, "
    "do not invent organization-specific facts, and state uncertainty when you are not sure."
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


# --------------------------------------------------------------------------- enums


class Backend(StrEnum):
    AUTO = "auto"
    MLX = "mlx"
    HF_PEFT = "hf_peft"


class Device(StrEnum):
    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


class Precision(StrEnum):
    AUTO = "auto"
    QLORA_4BIT = "qlora_4bit"
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"


class SourceFormat(StrEnum):
    AUTO = "auto"
    CSV = "csv"
    XLSX = "xlsx"
    MARKDOWN = "md"
    TXT = "txt"


class RecordStatus(StrEnum):
    APPROVED = "approved"
    PENDING_REVIEW = "pending_review"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"


class PrivacyPolicy(StrEnum):
    WARN = "warn"
    EXCLUDE = "exclude"
    STOP = "stop"


# --------------------------------------------------------------------------- ingest


class ColumnMapping(StrictModel):
    question: str = Field(description="Name of the column that holds the question text.")
    answer: str = Field(description="Name of the column that holds the approved answer text.")
    id: str | None = Field(
        default=None,
        description="Optional column with your own record IDs. Leave empty to generate stable IDs.",
    )
    tags: str | None = Field(default=None, description="Optional column with comma-separated tags.")
    category: str | None = Field(
        default=None,
        description="Optional column with a category, used to report results per category.",
    )
    status: str | None = Field(
        default=None,
        description=(
            "Optional column with a review status (approved, pending_review, rejected). "
            "Without it, every row gets the source's default status."
        ),
    )
    group: str | None = Field(
        default=None,
        description=(
            "Optional column that groups related rows. Rows in one group always land in the "
            "same split, so near-copies can't leak from training into the test set."
        ),
    )
    thread: str | None = Field(
        default=None,
        description="Optional thread or ticket column, used for grouping when no group column exists.",
    )


class QADelimiters(StrictModel):
    question_prefix: str = Field(
        default="Q:", description="Text that starts a question line in Markdown/TXT files."
    )
    answer_prefix: str = Field(
        default="A:", description="Text that starts the answer in Markdown/TXT files."
    )


class SourceSpec(StrictModel):
    path: str = Field(
        description="File or folder to read, relative to this config file or absolute."
    )
    format: SourceFormat = Field(
        default=SourceFormat.AUTO,
        description="File format. 'auto' decides from the file extension.",
    )
    columns: ColumnMapping | None = Field(
        default=None, description="Column mapping for CSV/XLSX files. Required for those formats."
    )
    sheet: str | None = Field(
        default=None, description="XLSX only: the sheet to read. Empty means every sheet."
    )
    delimiters: QADelimiters = Field(
        default_factory=QADelimiters,
        description="Markdown/TXT only: how questions and answers are marked.",
    )
    default_status: RecordStatus = Field(
        default=RecordStatus.PENDING_REVIEW,
        description=(
            "Status given to records without a status column. 'pending_review' means you approve "
            "them before they can train; use 'approved' only for already-vetted data."
        ),
    )
    group_by_source_file: bool = Field(
        default=False,
        description=(
            "Treat the whole file as one group. Only turn this on if every record in the file is "
            "about the same topic; otherwise it can make the split impossible."
        ),
    )
    encoding: str = Field(
        default="utf-8",
        description=(
            "Text encoding of CSV/Markdown/TXT files. Keep 'utf-8' unless the scan reports an "
            "encoding problem (older Windows exports are often 'cp1252')."
        ),
    )


class QualityThresholds(StrictModel):
    min_question_chars: int = Field(
        default=8,
        ge=0,
        description="Questions shorter than this are flagged as suspiciously short.",
    )
    min_answer_chars: int = Field(
        default=3, ge=0, description="Answers shorter than this are flagged as suspiciously short."
    )
    max_question_chars: int = Field(
        default=4000,
        ge=1,
        description="Questions longer than this are flagged as suspiciously long.",
    )
    max_answer_chars: int = Field(
        default=20000,
        ge=1,
        description="Answers longer than this are flagged as suspiciously long.",
    )


class PrivacyConfig(StrictModel):
    policy: PrivacyPolicy = Field(
        default=PrivacyPolicy.EXCLUDE,
        description=(
            "What to do when a record looks like it contains a secret (API key, password, token). "
            "'warn' keeps it, 'exclude' leaves it out of training, 'stop' halts ingestion."
        ),
    )
    extra_patterns: list[str] = Field(
        default_factory=list,
        description="Additional regular expressions to treat as secrets, e.g. internal token formats.",
    )


class IngestConfig(StrictModel):
    kind: Literal["ingest"] = Field(
        default="ingest", description="Config type. Must be 'ingest' for this file."
    )
    name: str = Field(description="A short name for this data source.")
    sources: Annotated[list[SourceSpec], Field(min_length=1)] = Field(
        description="Files or folders to read."
    )
    privacy: PrivacyConfig = Field(
        default_factory=PrivacyConfig, description="Secret scanning before records are stored."
    )
    quality: QualityThresholds = Field(
        default_factory=QualityThresholds, description="Limits for the automatic quality flags."
    )
    debug_log_content: bool = Field(
        default=False,
        description="Write record text to debug logs. Off by default to keep content out of logs.",
    )

    @model_validator(mode="after")
    def _tabular_sources_need_columns(self) -> IngestConfig:
        for spec in self.sources:
            if spec.format in (SourceFormat.CSV, SourceFormat.XLSX) and spec.columns is None:
                raise ValueError(
                    f"source '{spec.path}' is {spec.format.value} but has no 'columns' mapping"
                )
        return self


# --------------------------------------------------------------------------- dataset


class SplitRatios(StrictModel):
    train: float = Field(default=0.8, gt=0, lt=1, description="Share of groups used for training.")
    valid: float = Field(
        default=0.1,
        gt=0,
        lt=1,
        description="Share used to pick the best checkpoint and stop early. Never trained on.",
    )
    test: float = Field(
        default=0.1,
        gt=0,
        lt=1,
        description="Share held back to judge the final model. Never used for training or tuning.",
    )

    @model_validator(mode="after")
    def _sum_to_one(self) -> SplitRatios:
        total = self.train + self.valid + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split ratios must add up to 1.0 (got {total:.3f})")
        return self


class DatasetThresholds(StrictModel):
    min_train_examples_warn: int = Field(
        default=200,
        ge=1,
        description="Below this many training examples, expect only style-level improvements.",
    )
    min_test_examples_conclusive: int = Field(
        default=50,
        ge=1,
        description="Below this many test examples, the evaluation verdict is 'inconclusive'.",
    )


class DatasetConfig(StrictModel):
    kind: Literal["dataset"] = Field(
        default="dataset", description="Config type. Must be 'dataset' for this file."
    )
    name: str = Field(description="A short name for this dataset build.")
    source_ids: list[str] = Field(
        default_factory=list,
        description="Data sources to include. Empty means every ingested source.",
    )
    model_catalog_id: str = Field(
        description="Model whose tokenizer measures sequence length. Use the model you will train."
    )
    system_prompt: str = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        min_length=1,
        description="Instruction placed before every question. Chat uses exactly the same text.",
    )
    split: SplitRatios = Field(default_factory=SplitRatios, description="How groups are split.")
    seed: int = Field(default=42, description="Random seed so the same input gives the same split.")
    max_seq_length: int = Field(
        default=1024,
        ge=64,
        le=32768,
        description="Longest example in tokens. Longer examples are set aside (quarantined).",
    )
    truncate_long_records: bool = Field(
        default=False,
        description="Cut long examples instead of setting them aside. Truncated answers can teach bad habits.",
    )
    thresholds: DatasetThresholds = Field(
        default_factory=DatasetThresholds, description="Size warnings."
    )


# --------------------------------------------------------------------------- train


class ModelRef(StrictModel):
    catalog_id: str = Field(description="Which model from the catalog to fine-tune.")


class MlxTrainingOptions(StrictModel):
    num_layers: int = Field(
        default=8,
        ge=-1,
        description="How many of the model's last layers get LoRA adapters (Apple Silicon). -1 means all.",
    )


class HfPeftTrainingOptions(StrictModel):
    target_modules: Literal["auto"] | list[str] = Field(
        default="auto",
        description="Which layers get LoRA adapters (NVIDIA/CPU). 'auto' picks them for the model architecture.",
    )
    cpu_max_params_b: float = Field(
        default=1.0,
        gt=0,
        description="Largest model, in billions of parameters, allowed to train on CPU. CPU training is very slow.",
    )


class TrainingOptions(StrictModel):
    max_epochs: float = Field(
        default=3,
        gt=0,
        le=100,
        description="Most passes over the training data. Training may stop earlier if validation stops improving.",
    )
    patience: int = Field(
        default=3,
        ge=1,
        description="Stop after this many validation checks in a row without improvement.",
    )
    batch_size: int = Field(
        default=1, ge=1, description="Examples processed together. Higher uses more memory."
    )
    grad_accumulation_steps: int = Field(
        default=4,
        ge=1,
        description="Batches combined before each update. Raises the effective batch size without extra memory.",
    )
    learning_rate: float = Field(
        default=1.0e-4,
        gt=0,
        lt=1,
        description="How big each update step is. Too high can make training unstable.",
    )
    max_seq_length: int = Field(
        default=1024,
        ge=64,
        le=32768,
        description="Longest example in tokens. Must match the dataset build.",
    )
    eval_every_steps: int = Field(
        default=50,
        ge=1,
        description="Check validation loss (and save a checkpoint) every this many weight updates.",
    )
    gradient_checkpointing: bool = Field(
        default=True, description="Save memory at the cost of roughly 20% slower training."
    )
    seed: int = Field(default=42, description="Random seed for reproducible training.")
    mlx: MlxTrainingOptions = Field(
        default_factory=MlxTrainingOptions, description="Settings used only on Apple Silicon (MLX)."
    )
    hf_peft: HfPeftTrainingOptions = Field(
        default_factory=HfPeftTrainingOptions,
        description="Settings used only on NVIDIA GPUs and CPU (Transformers + PEFT).",
    )


class LoraOptions(StrictModel):
    rank: int = Field(
        default=8,
        ge=1,
        le=256,
        description="Size of the adapter. Bigger can learn more but uses more memory.",
    )
    alpha: float = Field(
        default=16,
        gt=0,
        description=(
            "Strength of the adapter. The effective scale is alpha / rank on every backend "
            "(MLX receives scale = alpha / rank)."
        ),
    )
    dropout: float = Field(
        default=0.05,
        ge=0,
        lt=1,
        description="Randomly ignores part of the adapter while training to reduce overfitting.",
    )


_BACKEND_DEVICES = {
    Backend.MLX: {Device.AUTO},
    Backend.HF_PEFT: {Device.AUTO, Device.CUDA, Device.CPU},
    Backend.AUTO: {Device.AUTO, Device.CUDA, Device.CPU},
}


class TrainConfig(StrictModel):
    kind: Literal["train"] = Field(
        default="train", description="Config type. Must be 'train' for this file."
    )
    name: str = Field(description="A short name for this training configuration.")
    dataset_id: str | None = Field(
        default=None, description="Dataset build to train on. Empty means the most recent one."
    )
    model: ModelRef = Field(description="The base model.")
    backend: Backend = Field(
        default=Backend.AUTO,
        description="Training engine. 'auto' picks MLX on Apple Silicon, otherwise Transformers + PEFT.",
    )
    device: Device = Field(
        default=Device.AUTO,
        description="Hardware for Transformers + PEFT. 'auto' uses an NVIDIA GPU if present, else the CPU.",
    )
    precision: Precision = Field(
        default=Precision.AUTO,
        description="Number format. 'auto' picks 4-bit QLoRA or bf16/fp16 on NVIDIA, fp32 on CPU.",
    )
    preset: Literal["quick", "standard", "thorough"] = Field(
        default="standard",
        description=(
            "Starting point for the training settings. 'quick' checks that everything works, "
            "'standard' is a sensible default, 'thorough' trains longer and more carefully. "
            "Settings you write under 'training' override the preset."
        ),
    )
    training: TrainingOptions = Field(default_factory=TrainingOptions, description="Training loop.")
    lora: LoraOptions = Field(default_factory=LoraOptions, description="LoRA adapter shape.")
    system_prompt: str = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        min_length=1,
        description="Must be identical to the dataset's system prompt; chat uses it too.",
    )

    @model_validator(mode="after")
    def _compatible_choices(self) -> TrainConfig:
        if self.device not in _BACKEND_DEVICES[self.backend]:
            raise ValueError(
                f"device '{self.device.value}' can't be used with backend '{self.backend.value}'"
            )
        if self.backend is Backend.MLX and self.precision not in (Precision.AUTO,):
            raise ValueError(
                "MLX picks precision from the model files (4-bit models train as QLoRA); "
                "leave precision on 'auto'"
            )
        if self.device is Device.CPU and self.precision in (Precision.QLORA_4BIT, Precision.FP16):
            raise ValueError(f"precision '{self.precision.value}' is not supported on CPU")
        return self


# --------------------------------------------------------------------------- evaluate


class AcceptanceCriteria(StrictModel):
    min_test_items: int = Field(
        default=50, ge=1, description="Fewer test items than this gives an 'inconclusive' verdict."
    )
    min_review_coverage: float = Field(
        default=1.0,
        gt=0,
        le=1,
        description="Share of test pairs that must be reviewed (1.0 = all).",
    )
    min_non_tied_pairs: int = Field(
        default=20,
        ge=1,
        description="Reviews that picked a winner (not a tie) needed for a verdict.",
    )
    sign_test_alpha: float = Field(
        default=0.05,
        gt=0,
        lt=1,
        description="Significance level of the sign test. Lower demands stronger evidence.",
    )
    max_category_net_loss_share: float = Field(
        default=0.20,
        ge=0,
        le=1,
        description="Reject if, in any category with 5+ items, (losses - wins) / items exceeds this.",
    )
    max_deterministic_drop_pts: float = Field(
        default=2.0,
        ge=0,
        description="Reject if automatic checks pass this many points less often.",
    )
    max_regression_set_drop_pts: float = Field(
        default=5.0,
        ge=0,
        description="Reject if the general coding prompts pass this many points less often.",
    )


class GenerationOptions(StrictModel):
    max_new_tokens: int = Field(
        default=512, ge=1, le=8192, description="Longest answer the model may write, in tokens."
    )


class EvaluateConfig(StrictModel):
    kind: Literal["evaluate"] = Field(
        default="evaluate", description="Config type. Must be 'evaluate' for this file."
    )
    acceptance: AcceptanceCriteria = Field(
        default_factory=AcceptanceCriteria, description="Rules for the accept/reject verdict."
    )
    generation: GenerationOptions = Field(
        default_factory=GenerationOptions,
        description="Answer generation (always greedy, so results are repeatable).",
    )
    include_regression_set: bool = Field(
        default=True,
        description="Also run the bundled general coding prompts to catch regressions.",
    )
    review_seed: int = Field(
        default=1234, description="Seed for the hidden A/B order in blind review."
    )


# --------------------------------------------------------------------------- export


class ExportConfig(StrictModel):
    kind: Literal["export"] = Field(
        default="export", description="Config type. Must be 'export' for this file."
    )
    merge: bool = Field(
        default=False,
        description="Also build a standalone model with the adapter merged in. Uses more disk.",
    )
    out_dir: str | None = Field(
        default=None, description="Where to write the export. Empty means inside the run folder."
    )


CONFIG_MODELS: dict[str, type[StrictModel]] = {
    "ingest": IngestConfig,
    "dataset": DatasetConfig,
    "train": TrainConfig,
    "evaluate": EvaluateConfig,
    "export": ExportConfig,
}
