// The training config form: beginner fields, advanced overrides (rendered from the JSON Schema),
// validation, and the config sent to the server.
//
// Training settings start from the preset. Only fields the user overrides are sent, so a saved
// config keeps following its preset for everything else (the server stores only set fields).
import { formatValue, getPath, parseField, type SchemaField, setPath } from "./schema";

export type Presets = Record<string, Record<string, Record<string, unknown>>>;

export interface TrainingForm {
  name: string;
  datasetId: string; // "" = the most recent dataset
  model: string;
  preset: string;
  hardware: string; // "auto" or "<backend>:<device>"
  precision: string;
  overrides: Record<string, string>; // field path → what the user typed
}

export const NEW_FORM: TrainingForm = {
  name: "",
  datasetId: "",
  model: "",
  preset: "standard",
  hardware: "auto",
  precision: "auto",
  overrides: {},
};

/** The one training field the beginner view shows. */
export const EPOCHS = "training.max_epochs";

/** Fields with their own controls, not rendered as generic schema fields. */
const OWN_CONTROLS = new Set(["name", "dataset_id", "model.catalog_id", "backend", "device", "precision", "preset"]);

/** Clearer labels than the schema's generated titles; the descriptions come from the schema. */
export const LABELS: Record<string, string> = {
  "training.max_epochs": "Max epochs",
  "training.patience": "Patience",
  "training.batch_size": "Batch size",
  "training.grad_accumulation_steps": "Gradient accumulation steps",
  "training.learning_rate": "Learning rate",
  "training.max_seq_length": "Max sequence length (tokens)",
  "training.eval_every_steps": "Validate every N updates",
  "training.gradient_checkpointing": "Gradient checkpointing",
  "training.seed": "Seed",
  "training.mlx.num_layers": "LoRA layers",
  "training.hf_peft.target_modules": "Target modules",
  "training.hf_peft.cpu_max_params_b": "CPU limit (billions of parameters)",
  "lora.rank": "Rank",
  "lora.alpha": "Alpha",
  "lora.dropout": "Dropout",
  system_prompt: "System prompt",
};

export function label(field: SchemaField): string {
  return LABELS[field.path] ?? field.title;
}

/** Fields of the Advanced view, in schema order. */
export function advancedFields(fields: SchemaField[]): SchemaField[] {
  return fields.filter((f) => !OWN_CONTROLS.has(f.path) && f.path !== EPOCHS);
}

export function hardwareParts(hardware: string): { backend: string; device: string } {
  if (hardware === "auto") return { backend: "auto", device: "auto" };
  const [backend = "auto", device = "auto"] = hardware.split(":");
  return { backend, device: backend === "mlx" ? "auto" : device };
}

export function hardwareOf(backend: string, device: string): string {
  if (backend === "auto" && device === "auto") return "auto";
  if (backend === "mlx") return "mlx:apple_gpu";
  return `hf_peft:${device}`;
}

/** The value a training field has when the user doesn't override it: the preset's, else the default. */
export function presetValue(field: SchemaField, presets: Presets | undefined, backend: string, preset: string): unknown {
  if (field.path.startsWith("training.")) {
    const fromPreset = getPath(presets?.[backend]?.[preset], field.path.slice("training.".length));
    if (fromPreset !== undefined) return fromPreset;
  }
  return field.defaultValue;
}

export type TrainingErrors = Record<string, string>;

export function validateTrainingForm(form: TrainingForm, fields: SchemaField[]): TrainingErrors {
  const errors: TrainingErrors = {};
  if (!form.name.trim()) errors.name = "Give the config a name.";
  else if (form.name.trim().length > 100) errors.name = "Keep the name under 100 characters.";
  if (!form.model) errors.model = "Choose the model to fine-tune.";
  for (const field of fields) {
    const raw = form.overrides[field.path];
    if (raw === undefined) continue;
    const { error } = parseField(field, raw);
    if (error) errors[field.path] = error;
  }
  return errors;
}

/** The config to save: the beginner choices plus only the overridden fields. */
export function trainConfig(form: TrainingForm, fields: SchemaField[]): Record<string, unknown> {
  const { backend, device } = hardwareParts(form.hardware);
  const config: Record<string, unknown> = {
    name: form.name.trim(),
    model: { catalog_id: form.model },
    preset: form.preset,
  };
  if (form.datasetId) config.dataset_id = form.datasetId;
  if (backend !== "auto") config.backend = backend;
  if (device !== "auto") config.device = device;
  if (form.precision !== "auto") config.precision = form.precision;
  for (const field of fields) {
    const raw = form.overrides[field.path];
    if (raw === undefined) continue;
    const { value, error } = parseField(field, raw);
    if (!error) setPath(config, field.path, value);
  }
  return config;
}

/** A saved config back into the form (to edit it as a new version). */
export function formFromConfig(config: Record<string, unknown>, fields: SchemaField[]): TrainingForm {
  const overrides: Record<string, string> = {};
  for (const field of fields) {
    if (OWN_CONTROLS.has(field.path)) continue;
    const value = getPath(config, field.path);
    if (value !== undefined) overrides[field.path] = formatValue(value);
  }
  return {
    name: String(config.name ?? ""),
    datasetId: typeof config.dataset_id === "string" ? config.dataset_id : "",
    model: String(getPath(config, "model.catalog_id") ?? ""),
    preset: String(config.preset ?? "standard"),
    hardware: hardwareOf(String(config.backend ?? "auto"), String(config.device ?? "auto")),
    precision: String(config.precision ?? "auto"),
    overrides,
  };
}
