// Training config: beginner view (model, preset, epochs, the auto-selected backend explained) and
// a collapsed advanced view rendered from the config's JSON Schema. Saving a changed config makes
// a new version; old versions stay, so every run points at the exact config it used.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api, type Schemas, unwrap } from "../api/client";
import { useReportError } from "../App";
import { Badge, Button, Card, ErrorPanel, Field, inputClass, Spinner } from "../components/ui";
import { useDatasets, useSettings } from "../lib/queries";
import { routeHref } from "../lib/route";
import { formatValue, type JsonSchema, type SchemaField, schemaFields, schemaGroups } from "../lib/schema";
import {
  advancedFields,
  EPOCHS,
  formFromConfig,
  hardwareParts,
  label,
  NEW_FORM,
  presetValue,
  type Presets,
  trainConfig,
  type TrainingForm,
  validateTrainingForm,
} from "../lib/trainingForm";

const HARDWARE_LABELS: Record<string, string> = {
  "mlx:apple_gpu": "MLX on the Apple Silicon GPU",
  "hf_peft:cuda": "Transformers + PEFT on the NVIDIA GPU (CUDA)",
  "hf_peft:cpu": "Transformers + PEFT on the CPU",
  "hf_peft:auto": "Transformers + PEFT (device: auto)",
};

export function useTrainSchema() {
  return useQuery({
    queryKey: ["schema", "train"],
    queryFn: () =>
      unwrap(api.GET("/api/configs/schema/{kind}", { params: { path: { kind: "train" } } })) as Promise<JsonSchema>,
    staleTime: Infinity,
  });
}

export function useConfigs() {
  return useQuery({ queryKey: ["configs"], queryFn: () => unwrap(api.GET("/api/configs")) });
}

export function TrainingConfig() {
  const schema = useTrainSchema();
  const [editing, setEditing] = useState<{ form: TrainingForm; title: string; key: number }>({
    form: NEW_FORM,
    title: "New config",
    key: 0,
  });
  const fields = schema.data ? schemaFields(schema.data) : null;
  const report = useReportError();

  const edit = async (name: string) => {
    try {
      const detail = await unwrap(api.GET("/api/configs/{name}", { params: { path: { name } } }));
      setEditing((e) => ({
        form: formFromConfig(detail.config, fields ?? []),
        title: `Edit ${detail.name} (v${detail.version})`,
        key: e.key + 1,
      }));
    } catch (error) {
      report(error);
    }
  };

  return (
    <div className="space-y-4">
      <SavedConfigs onEdit={edit} />
      {schema.error ? (
        <ErrorPanel error={schema.error} />
      ) : !schema.data || !fields ? (
        <Spinner />
      ) : (
        <ConfigForm
          key={editing.key}
          title={editing.title}
          initial={editing.form}
          schema={schema.data}
          fields={fields}
          onNew={() => setEditing((e) => ({ form: NEW_FORM, title: "New config", key: e.key + 1 }))}
        />
      )}
    </div>
  );
}

// --------------------------------------------------------------------------- saved configs

function SavedConfigs({ onEdit }: { onEdit: (name: string) => void }) {
  const configs = useConfigs();
  const [duplicating, setDuplicating] = useState<string | null>(null);
  const [done, setDone] = useState<string | null>(null);
  return (
    <Card title="Saved configs">
      {configs.error ? (
        <ErrorPanel error={configs.error} />
      ) : !configs.data ? (
        <Spinner />
      ) : configs.data.length === 0 ? (
        <p className="text-sm">No saved configs yet. Fill in the form below and save it.</p>
      ) : (
        <div className="space-y-3 text-sm">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-700">
                <th scope="col">Name</th>
                <th scope="col">Version</th>
                <th scope="col">Model</th>
                <th scope="col">Preset</th>
                <th scope="col">Backend</th>
                <th scope="col">
                  <span className="sr-only">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {configs.data.map((c) => (
                <tr key={c.name} className="border-b border-slate-100 align-top dark:border-slate-800">
                  <th scope="row" className="py-1 font-medium">
                    {c.name}
                  </th>
                  <td>
                    v{c.version}
                    {c.versions > 1 ? ` (${c.versions} versions)` : ""}
                  </td>
                  <td>{c.model}</td>
                  <td>{c.preset}</td>
                  <td>{c.backend}</td>
                  <td className="space-x-1 py-1 text-right">
                    <Button variant="secondary" onClick={() => onEdit(c.name)} aria-label={`Edit ${c.name}`}>
                      Edit
                    </Button>
                    <Button
                      variant="secondary"
                      onClick={() => setDuplicating(c.name)}
                      aria-label={`Duplicate ${c.name}`}
                    >
                      Duplicate
                    </Button>
                    <a
                      className="inline-block rounded-md px-3 py-1.5 font-medium text-sky-800 hover:bg-sky-100 dark:text-sky-300 dark:hover:bg-slate-800"
                      href={routeHref("/preflight", { config: c.name })}
                      aria-label={`Preflight ${c.name}`}
                    >
                      Preflight
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {duplicating && (
            <DuplicateForm
              key={duplicating}
              source={duplicating}
              onDone={(text) => {
                setDuplicating(null);
                setDone(text);
              }}
              onCancel={() => setDuplicating(null)}
            />
          )}
          {done && <p role="status">{done}</p>}
        </div>
      )}
    </Card>
  );
}

function DuplicateForm({
  source,
  onDone,
  onCancel,
}: {
  source: string;
  onDone: (message: string) => void;
  onCancel: () => void;
}) {
  const [name, setName] = useState(`${source}-copy`);
  const [submitted, setSubmitted] = useState(false);
  const client = useQueryClient();
  const report = useReportError();
  const error = !name.trim() ? "Give the copy a name." : null;
  const duplicate = useMutation({
    mutationFn: () =>
      unwrap(api.POST("/api/configs/{name}/duplicate", { params: { path: { name: source } }, body: { new_name: name.trim() } })),
    onSuccess: (saved) => {
      void client.invalidateQueries({ queryKey: ["configs"] });
      onDone(`Duplicated ${source} as ${saved.name} v${saved.version}.`);
    },
    onError: report,
  });
  return (
    <form
      className="flex flex-wrap items-end gap-2"
      noValidate
      onSubmit={(e) => {
        e.preventDefault();
        setSubmitted(true);
        if (!error) duplicate.mutate();
      }}
    >
      <Field label={`Name of the copy of ${source}`} htmlFor="dup-name" error={submitted ? error : null}>
        <input id="dup-name" className={inputClass} value={name} onChange={(e) => setName(e.target.value)} />
      </Field>
      <Button type="submit" disabled={duplicate.isPending}>
        Make the copy
      </Button>
      <Button variant="ghost" onClick={onCancel}>
        Cancel
      </Button>
    </form>
  );
}

// --------------------------------------------------------------------------- the form

function ConfigForm({
  title,
  initial,
  schema,
  fields,
  onNew,
}: {
  title: string;
  initial: TrainingForm;
  schema: JsonSchema;
  fields: SchemaField[];
  onNew: () => void;
}) {
  const settings = useSettings();
  const datasets = useDatasets();
  const models = useQuery({
    queryKey: ["models", settings.data?.dev_mode],
    enabled: settings.data !== undefined,
    queryFn: () =>
      unwrap(api.GET("/api/models", { params: { query: { include_test_only: settings.data?.dev_mode ?? false } } })),
  });
  const [form, setForm] = useState<TrainingForm>(initial);
  const choices = useQuery({
    queryKey: ["training-choices", form.model],
    queryFn: () => unwrap(api.GET("/api/training-choices", { params: { query: { model: form.model || undefined } } })),
  });
  const [submitted, setSubmitted] = useState(false);
  const [saved, setSaved] = useState<Schemas["SavedConfig"] | null>(null);
  const client = useQueryClient();
  const report = useReportError();

  const errors = validateTrainingForm(form, fields);
  const shown = (path: string) => (submitted || form.overrides[path] !== undefined ? errors[path] : undefined);
  const set = <K extends keyof TrainingForm>(key: K, value: TrainingForm[K]) => setForm((f) => ({ ...f, [key]: value }));
  const override = (path: string, raw: string | undefined) =>
    setForm((f) => {
      const overrides = { ...f.overrides };
      if (raw === undefined) delete overrides[path];
      else overrides[path] = raw;
      return { ...f, overrides };
    });

  const { backend } = hardwareParts(form.hardware);
  const presetBackend = backend !== "auto" ? backend : (choices.data?.auto?.backend ?? "hf_peft");
  const presets = choices.data?.presets as Presets | undefined;
  const effective = (field: SchemaField) => presetValue(field, presets, presetBackend, form.preset);
  const trainable = (models.data?.models ?? []).filter((m) => m.trainable);
  const epochs = fields.find((f) => f.path === EPOCHS);
  const presetField = schema.properties?.preset;

  const save = useMutation({
    mutationFn: () => unwrap(api.POST("/api/configs", { body: { config: trainConfig(form, fields) } })),
    onSuccess: (result) => {
      setSaved(result);
      void client.invalidateQueries({ queryKey: ["configs"] });
    },
    onError: report,
  });

  const submit = (then?: (s: Schemas["SavedConfig"]) => void) => {
    setSubmitted(true);
    if (Object.keys(errors).length) return;
    save.mutate(undefined, { onSuccess: (result) => then?.(result) });
  };

  return (
    <Card
      title={title}
      actions={
        <Button variant="secondary" onClick={onNew}>
          New config
        </Button>
      }
    >
      <form
        className="space-y-4"
        noValidate
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <div className="grid gap-3 md:grid-cols-3">
          <Field label="Name" htmlFor="tc-name" error={submitted ? errors.name : undefined} hint="Saving a changed config under the same name makes a new version.">
            <input id="tc-name" className={inputClass} value={form.name} onChange={(e) => set("name", e.target.value)} />
          </Field>
          <Field label="Model" htmlFor="tc-model" error={submitted ? errors.model : undefined} hint="Only models whose chat-template test passed.">
            <select id="tc-model" className={inputClass} value={form.model} onChange={(e) => set("model", e.target.value)}>
              <option value="">— choose —</option>
              {trainable.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.display_name} ({m.params_b}B){m.compatibility ? ` — ${m.compatibility.label} (estimate)` : ""}
                </option>
              ))}
              {form.model && !trainable.some((m) => m.id === form.model) && <option value={form.model}>{form.model}</option>}
            </select>
          </Field>
          <Field label="Dataset" htmlFor="tc-dataset" hint={schema.properties?.dataset_id?.description}>
            <select id="tc-dataset" className={inputClass} value={form.datasetId} onChange={(e) => set("datasetId", e.target.value)}>
              <option value="">The most recent dataset</option>
              {datasets.data?.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} ({d.id})
                </option>
              ))}
              {form.datasetId && !datasets.data?.some((d) => d.id === form.datasetId) && (
                <option value={form.datasetId}>{form.datasetId}</option>
              )}
            </select>
          </Field>
          <Field label="Preset" htmlFor="tc-preset" hint={presetField?.description}>
            <select id="tc-preset" className={inputClass} value={form.preset} onChange={(e) => set("preset", e.target.value)}>
              {(presetField?.enum ?? ["quick", "standard", "thorough"]).map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </Field>
          {epochs && (
            <SchemaInput field={epochs} raw={form.overrides[EPOCHS]} effective={effective(epochs)} error={shown(EPOCHS)} onChange={override} />
          )}
        </div>
        <BackendLine choices={choices.data} error={choices.error} hardware={form.hardware} precision={form.precision} />

        <details className="rounded-md border border-slate-200 p-3 dark:border-slate-700">
          <summary className="cursor-pointer text-sm font-medium">Advanced settings</summary>
          <div className="mt-3 space-y-4">
            <HardwareFields form={form} choices={choices.data} set={set} schema={schema} />
            {schemaGroups(schema)
              .filter((g) => g.path !== "model")
              .concat([{ path: "", description: "Prompt" }])
              .map((group) => {
                const inGroup = advancedFields(fields).filter((f) => f.group === group.path);
                if (!inGroup.length) return null;
                return (
                  <fieldset key={group.path} className="grid gap-3 md:grid-cols-3">
                    <legend className="mb-1 text-sm font-semibold">{group.description}</legend>
                    {inGroup.map((field) => (
                      <SchemaInput
                        key={field.path}
                        field={field}
                        raw={form.overrides[field.path]}
                        effective={effective(field)}
                        error={shown(field.path)}
                        onChange={override}
                      />
                    ))}
                  </fieldset>
                );
              })}
          </div>
        </details>

        <div className="flex flex-wrap items-center gap-2">
          <Button type="submit" disabled={save.isPending}>
            Save
          </Button>
          <Button
            variant="secondary"
            disabled={save.isPending}
            onClick={() =>
              submit((s) => {
                window.location.hash = routeHref("/preflight", { config: s.name, version: s.version });
              })
            }
          >
            Save and preflight
          </Button>
          {submitted && Object.keys(errors).length > 0 && (
            <p role="alert" className="text-sm text-rose-700 dark:text-rose-300">
              Fix the highlighted fields first.
            </p>
          )}
          {saved && (
            <p role="status" className="text-sm">
              {saved.created ? `Saved ${saved.name} v${saved.version}.` : `No changes: ${saved.name} v${saved.version} is already saved.`}
            </p>
          )}
        </div>
      </form>
    </Card>
  );
}

function BackendLine({
  choices,
  error,
  hardware,
  precision,
}: {
  choices: Schemas["TrainingChoices"] | undefined;
  error: unknown;
  hardware: string;
  precision: string;
}) {
  if (error) return <ErrorPanel error={error} />;
  if (!choices) return <Spinner label="Checking this machine…" />;
  if (hardware !== "auto") {
    return (
      <p className="text-sm">
        <span className="font-medium">Backend:</span> {HARDWARE_LABELS[hardware] ?? hardware}
        {precision !== "auto" ? `, ${precision}` : ""} (set under Advanced settings).
      </p>
    );
  }
  if (!choices.auto) {
    return (
      <p className="text-sm text-amber-800 dark:text-amber-300">
        <span className="font-medium">Backend:</span> Auto finds no backend for this model here: {choices.auto_error}
      </p>
    );
  }
  return (
    <p className="text-sm">
      <span className="font-medium">Backend:</span> Auto → {choices.auto.backend} on {choices.auto.device} ({choices.auto.precision}).{" "}
      <span className="text-slate-600 dark:text-slate-400">Why: {choices.auto.reason}.</span>
    </p>
  );
}

function HardwareFields({
  form,
  choices,
  set,
  schema,
}: {
  form: TrainingForm;
  choices: Schemas["TrainingChoices"] | undefined;
  set: <K extends keyof TrainingForm>(key: K, value: TrainingForm[K]) => void;
  schema: JsonSchema;
}) {
  const available = (choices?.options ?? []).filter((o) => o.available);
  const unavailable = (choices?.options ?? []).filter((o) => !o.available);
  const values = available.map((o) => `${o.backend}:${o.device}`);
  if (form.hardware !== "auto" && !values.includes(form.hardware)) values.push(form.hardware); // a loaded config
  const current = form.hardware === "auto" ? available.find((o) => o.device === choices?.auto?.device) : available.find((o) => `${o.backend}:${o.device}` === form.hardware);
  const precisions = current?.precisions.length ? current.precisions : ["auto"];
  if (!precisions.includes(form.precision)) precisions.push(form.precision);
  return (
    <fieldset className="grid gap-3 md:grid-cols-3">
      <legend className="mb-1 text-sm font-semibold">Backend and hardware (only what this machine has)</legend>
      <Field label="Backend and device" htmlFor="tc-hardware" hint={schema.properties?.backend?.description}>
        <select
          id="tc-hardware"
          className={inputClass}
          value={form.hardware}
          onChange={(e) => {
            set("hardware", e.target.value);
            set("precision", "auto");
          }}
        >
          <option value="auto">Auto (recommended)</option>
          {values.map((v) => (
            <option key={v} value={v}>
              {HARDWARE_LABELS[v] ?? v}
            </option>
          ))}
        </select>
      </Field>
      <Field label="Precision" htmlFor="tc-precision" hint={schema.properties?.precision?.description}>
        <select id="tc-precision" className={inputClass} value={form.precision} onChange={(e) => set("precision", e.target.value)}>
          {precisions.map((p) => (
            <option key={p} value={p}>
              {p === "auto" ? "Auto" : p}
            </option>
          ))}
        </select>
      </Field>
      {unavailable.length > 0 && (
        <div className="text-xs text-slate-600 dark:text-slate-400">
          <p className="font-medium">Not available here:</p>
          <ul aria-label="Not available here">
            {unavailable.map((o) => (
              <li key={`${o.backend}:${o.device}`}>
                {HARDWARE_LABELS[`${o.backend}:${o.device}`] ?? `${o.backend} on ${o.device}`}: {o.reason}
              </li>
            ))}
          </ul>
        </div>
      )}
    </fieldset>
  );
}

/** One schema field: empty means "use the preset/default value", which shows as the placeholder. */
function SchemaInput({
  field,
  raw,
  effective,
  error,
  onChange,
}: {
  field: SchemaField;
  raw: string | undefined;
  effective: unknown;
  error: string | undefined;
  onChange: (path: string, raw: string | undefined) => void;
}) {
  const id = `tc-${field.path.replaceAll(".", "-")}`;
  const usual = formatValue(effective);
  const hint =
    field.path === "system_prompt"
      ? field.description
      : `${field.description} ${field.path.startsWith("training.") ? "Preset value" : "Default"}: ${usual || "none"}.`;
  const reset = raw !== undefined && (
    <Button variant="ghost" className="px-1 py-0 text-xs" onClick={() => onChange(field.path, undefined)} aria-label={`Reset ${label(field)}`}>
      Reset
    </Button>
  );
  let input;
  if (field.kind === "boolean" || field.kind === "enum") {
    const options = field.kind === "boolean" ? ["true", "false"] : field.options;
    input = (
      <select id={id} className={inputClass} value={raw ?? ""} onChange={(e) => onChange(field.path, e.target.value || undefined)}>
        <option value="">Preset value ({usual === "true" ? "yes" : usual === "false" ? "no" : usual})</option>
        {options.map((o) => (
          <option key={o} value={o}>
            {o === "true" ? "yes" : o === "false" ? "no" : o}
          </option>
        ))}
      </select>
    );
  } else if (field.path === "system_prompt") {
    input = (
      <textarea
        id={id}
        rows={3}
        className={inputClass}
        value={raw ?? usual}
        onChange={(e) => onChange(field.path, e.target.value === usual ? undefined : e.target.value)}
      />
    );
  } else {
    input = (
      <input
        id={id}
        className={inputClass}
        inputMode={field.kind === "integer" ? "numeric" : field.kind === "number" ? "decimal" : undefined}
        placeholder={usual}
        value={raw ?? ""}
        aria-invalid={error ? true : undefined}
        onChange={(e) => onChange(field.path, e.target.value === "" ? undefined : e.target.value)}
      />
    );
  }
  return (
    <div className={field.path === "system_prompt" ? "md:col-span-3" : undefined}>
      <Field label={label(field)} htmlFor={id} hint={hint} error={error}>
        {input}
      </Field>
      <div className="flex items-center gap-1">
        {raw !== undefined && <Badge tone="warn">changed</Badge>}
        {reset}
      </div>
    </div>
  );
}
