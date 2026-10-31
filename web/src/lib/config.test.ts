import { describe, expect, it } from "vitest";

import trainSchema from "../test/train-schema.json";
import { routeHref, routeParams, routePath } from "./route";
import { type JsonSchema, parseField, schemaFields, schemaGroups } from "./schema";
import {
  advancedFields,
  formFromConfig,
  NEW_FORM,
  presetValue,
  type Presets,
  trainConfig,
  validateTrainingForm,
} from "./trainingForm";

const schema = trainSchema as JsonSchema;
const fields = schemaFields(schema);
const field = (path: string) => fields.find((f) => f.path === path)!;
const PRESETS: Presets = {
  hf_peft: { quick: { max_epochs: 1, eval_every_steps: 10 }, standard: {}, thorough: { max_epochs: 5 } },
  mlx: { quick: { max_epochs: 1, mlx: { num_layers: 4 } }, standard: {}, thorough: {} },
};

describe("schema fields", () => {
  it("reads every editable field with its description, default and limits from the server's schema", () => {
    const paths = fields.map((f) => f.path);
    expect(paths).not.toContain("kind");
    expect(paths).toEqual(
      expect.arrayContaining(["name", "model.catalog_id", "training.max_epochs", "training.mlx.num_layers", "lora.rank", "system_prompt"]),
    );
    expect(field("training.max_epochs")).toMatchObject({ kind: "number", defaultValue: 3, exclusiveMinimum: 0, maximum: 100 });
    expect(field("training.max_epochs").description).toContain("passes over the training data");
    expect(field("training.hf_peft.target_modules").kind).toBe("list");
    expect(field("dataset_id")).toMatchObject({ kind: "string", nullable: true });
    expect(field("backend")).toMatchObject({ kind: "enum", options: ["auto", "mlx", "hf_peft"] });
    expect(schemaGroups(schema).map((g) => g.path)).toEqual(["model", "training", "training.mlx", "training.hf_peft", "lora"]);
  });

  it("validates input against the schema's limits", () => {
    expect(parseField(field("training.learning_rate"), "1e-4")).toEqual({ value: 0.0001 });
    expect(parseField(field("training.learning_rate"), "2").error).toBe("Use a value above 0 and below 1.");
    expect(parseField(field("training.learning_rate"), "fast").error).toContain("Use a number");
    expect(parseField(field("training.batch_size"), "1.5").error).toBe("Use a whole number.");
    expect(parseField(field("training.batch_size"), "0").error).toBe("Use a value at least 1.");
    expect(parseField(field("training.max_seq_length"), "40000").error).toBe("Use a value at least 64 and at most 32768.");
    expect(parseField(field("training.gradient_checkpointing"), "false")).toEqual({ value: false });
    expect(parseField(field("training.hf_peft.target_modules"), "q_proj, v_proj")).toEqual({ value: ["q_proj", "v_proj"] });
    expect(parseField(field("training.hf_peft.target_modules"), "q proj").error).toContain("separated by commas");
    expect(parseField(field("system_prompt"), "").error).toBe("This can't be empty.");
  });
});

describe("training form", () => {
  it("shows the preset's value for fields the user leaves alone", () => {
    expect(presetValue(field("training.max_epochs"), PRESETS, "hf_peft", "quick")).toBe(1);
    expect(presetValue(field("training.max_epochs"), PRESETS, "hf_peft", "standard")).toBe(3);
    expect(presetValue(field("training.mlx.num_layers"), PRESETS, "mlx", "quick")).toBe(4);
    expect(presetValue(field("lora.rank"), PRESETS, "hf_peft", "quick")).toBe(8);
  });

  it("sends only the overridden fields, so the preset stays in charge of the rest", () => {
    const form = {
      ...NEW_FORM,
      name: " quick-cpu ",
      model: "tiny-qwen2",
      preset: "quick",
      hardware: "hf_peft:cpu",
      precision: "fp32",
      overrides: { "training.max_epochs": "2", "lora.rank": "16", "training.hf_peft.target_modules": "auto" },
    };
    expect(validateTrainingForm(form, fields)).toEqual({});
    expect(trainConfig(form, fields)).toEqual({
      name: "quick-cpu",
      model: { catalog_id: "tiny-qwen2" },
      preset: "quick",
      backend: "hf_peft",
      device: "cpu",
      precision: "fp32",
      training: { max_epochs: 2, hf_peft: { target_modules: "auto" } },
      lora: { rank: 16 },
    });
    expect(trainConfig({ ...NEW_FORM, name: "a", model: "m" }, fields)).toEqual({
      name: "a",
      model: { catalog_id: "m" },
      preset: "standard",
    });
  });

  it("names every problem before anything is sent", () => {
    const errors = validateTrainingForm({ ...NEW_FORM, overrides: { "training.learning_rate": "2" } }, fields);
    expect(errors).toEqual({
      name: "Give the config a name.",
      model: "Choose the model to fine-tune.",
      "training.learning_rate": "Use a value above 0 and below 1.",
    });
  });

  it("loads a saved config back for editing", () => {
    const saved = {
      name: "c",
      model: { catalog_id: "tiny-qwen2" },
      dataset_id: "ds_1",
      preset: "thorough",
      backend: "mlx",
      training: { patience: 7, hf_peft: { target_modules: ["q_proj", "v_proj"] } },
    };
    const form = formFromConfig(saved, fields);
    expect(form).toEqual({
      name: "c",
      datasetId: "ds_1",
      model: "tiny-qwen2",
      preset: "thorough",
      hardware: "mlx:apple_gpu",
      precision: "auto",
      overrides: { "training.patience": "7", "training.hf_peft.target_modules": "q_proj, v_proj" },
    });
    expect(trainConfig(form, fields)).toEqual(saved);
  });

  it("keeps the beginner fields out of the advanced list", () => {
    const advanced = advancedFields(fields).map((f) => f.path);
    expect(advanced).not.toContain("training.max_epochs");
    expect(advanced).not.toContain("backend");
    expect(advanced).toContain("training.learning_rate");
  });
});

describe("routes", () => {
  it("carries parameters in the hash", () => {
    const href = routeHref("/preflight", { config: "a b", version: 2, empty: "" });
    expect(href).toBe("#/preflight?config=a+b&version=2");
    expect(routePath(href)).toBe("/preflight");
    expect(routeParams(href).get("config")).toBe("a b");
    expect(routePath("")).toBe("/");
  });
});
