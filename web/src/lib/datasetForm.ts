// The dataset build form: values, validation and the config sent to the server.

export interface DatasetForm {
  name: string;
  model: string;
  sources: string[];
  train: number;
  valid: number;
  test: number;
  seed: number;
  maxSeqLength: number;
  allowDownload: boolean;
  overwrite: boolean;
}

export const DEFAULT_FORM: DatasetForm = {
  name: "",
  model: "",
  sources: [],
  train: 0.8,
  valid: 0.1,
  test: 0.1,
  seed: 42,
  maxSeqLength: 1024,
  allowDownload: false,
  overwrite: false,
};

export type FormErrors = Partial<Record<keyof DatasetForm | "split", string>>;

export function validateDatasetForm(form: DatasetForm): FormErrors {
  const errors: FormErrors = {};
  if (!form.name.trim()) errors.name = "Give the dataset a name.";
  else if (form.name.trim().length > 100) errors.name = "Keep the name under 100 characters.";
  if (!form.model) errors.model = "Choose the model you'll train (its tokenizer measures lengths).";
  for (const part of ["train", "valid", "test"] as const) {
    const value = form[part];
    if (!Number.isFinite(value) || value <= 0 || value >= 1) errors[part] = "Use a share between 0 and 1.";
  }
  if (!errors.train && !errors.valid && !errors.test && Math.abs(form.train + form.valid + form.test - 1) > 1e-6) {
    errors.split = "Train, valid and test must add up to 1.";
  }
  if (!Number.isInteger(form.seed)) errors.seed = "The seed is a whole number.";
  if (!Number.isInteger(form.maxSeqLength) || form.maxSeqLength < 64 || form.maxSeqLength > 32768) {
    errors.maxSeqLength = "Use a whole number of tokens between 64 and 32768.";
  }
  return errors;
}

export function datasetConfig(form: DatasetForm): Record<string, unknown> {
  return {
    name: form.name.trim(),
    model_catalog_id: form.model,
    source_ids: form.sources,
    split: { train: form.train, valid: form.valid, test: form.test },
    seed: form.seed,
    max_seq_length: form.maxSeqLength,
  };
}
