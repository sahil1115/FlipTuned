// Dataset: build grouped train/valid/test splits from approved records, then read the statistics
// and warnings (counts only, never record text).
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { api, type Schemas, unwrap } from "../api/client";
import { useReportError } from "../App";
import { JobProgress } from "../components/JobProgress";
import { Badge, Button, Card, CountList, ErrorPanel, Field, inputClass, Spinner, Stat, statusTone } from "../components/ui";
import { DEFAULT_FORM, type DatasetForm, datasetConfig, validateDatasetForm } from "../lib/datasetForm";
import { useDatasets, useLatestJob, useSettings, useSources } from "../lib/queries";

export function Dataset() {
  const datasets = useDatasets();
  const [selected, setSelected] = useState<string | null>(null);
  const shown = selected ?? datasets.data?.[0]?.id ?? null;
  return (
    <div className="space-y-4">
      <BuildForm onBuilt={setSelected} />
      <Card title="Datasets">
        {datasets.error ? (
          <ErrorPanel error={datasets.error} />
        ) : !datasets.data ? (
          <Spinner />
        ) : datasets.data.length === 0 ? (
          <p className="text-sm">No datasets yet.</p>
        ) : (
          <ul className="flex flex-wrap gap-2" aria-label="Datasets">
            {datasets.data.map((d) => (
              <li key={d.id}>
                <Button variant={d.id === shown ? "primary" : "secondary"} onClick={() => setSelected(d.id)} aria-pressed={d.id === shown}>
                  {d.name} <span className="text-xs opacity-80">{d.id}</span>
                </Button>
              </li>
            ))}
          </ul>
        )}
      </Card>
      {shown && <DatasetStats key={shown} datasetId={shown} />}
    </div>
  );
}

function BuildForm({ onBuilt }: { onBuilt: (id: string) => void }) {
  const settings = useSettings();
  const sources = useSources();
  const models = useQuery({
    queryKey: ["models", settings.data?.dev_mode],
    enabled: settings.data !== undefined,
    queryFn: () =>
      unwrap(api.GET("/api/models", { params: { query: { include_test_only: settings.data?.dev_mode ?? false } } })),
  });
  const [form, setForm] = useState<DatasetForm>(DEFAULT_FORM);
  const [submitted, setSubmitted] = useState(false);
  const latest = useLatestJob("build_dataset");
  const [jobId, setJobId] = useState<string | null>(null);
  const report = useReportError();
  const client = useQueryClient();
  const errors = validateDatasetForm(form);
  const shownErrors = submitted ? errors : {};
  const trainable = (models.data?.models ?? []).filter((m) => m.trainable);
  const set = <K extends keyof DatasetForm>(field: K, value: DatasetForm[K]) => setForm((f) => ({ ...f, [field]: value }));

  const build = useMutation({
    mutationFn: () =>
      unwrap(
        api.POST("/api/datasets", {
          body: { config: datasetConfig(form), allow_download: form.allowDownload, overwrite: form.overwrite },
        }),
      ),
    onSuccess: (job) => {
      setJobId(job.id);
      client.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: report,
  });
  const watching = jobId ?? latest?.id ?? null;

  return (
    <Card title="Build a dataset">
      <form
        className="space-y-4"
        noValidate
        onSubmit={(e) => {
          e.preventDefault();
          setSubmitted(true);
          if (!Object.keys(errors).length) build.mutate();
        }}
      >
        <div className="grid gap-3 md:grid-cols-3">
          <Field label="Name" htmlFor="ds-name" error={shownErrors.name}>
            <input id="ds-name" className={inputClass} value={form.name} onChange={(e) => set("name", e.target.value)} />
          </Field>
          <Field
            label="Model"
            htmlFor="ds-model"
            error={shownErrors.model}
            hint="Its tokenizer measures sequence lengths. Use the model you'll train."
          >
            <select id="ds-model" className={inputClass} value={form.model} onChange={(e) => set("model", e.target.value)}>
              <option value="">— choose —</option>
              {trainable.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.display_name} ({m.params_b}B)
                </option>
              ))}
            </select>
          </Field>
          <Field label="Data sources" htmlFor="ds-sources" hint="None selected means every ingested source.">
            <select
              id="ds-sources"
              multiple
              className={`${inputClass} h-20`}
              value={form.sources}
              onChange={(e) => set("sources", Array.from(e.target.selectedOptions, (o) => o.value))}
            >
              {sources.data?.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
          </Field>
        </div>
        <fieldset className="grid gap-3 md:grid-cols-5">
          <legend className="mb-1 text-sm font-medium">Split (shares of groups)</legend>
          {(["train", "valid", "test"] as const).map((part) => (
            <Field key={part} label={part} htmlFor={`ds-${part}`} error={shownErrors[part]}>
              <input
                id={`ds-${part}`}
                type="number"
                step="0.05"
                min="0"
                max="1"
                className={inputClass}
                value={form[part]}
                onChange={(e) => set(part, Number(e.target.value))}
              />
            </Field>
          ))}
          <Field label="Seed" htmlFor="ds-seed" error={shownErrors.seed} hint="Same seed, same split.">
            <input id="ds-seed" type="number" className={inputClass} value={form.seed} onChange={(e) => set("seed", Number(e.target.value))} />
          </Field>
          <Field
            label="Max sequence length"
            htmlFor="ds-seq"
            error={shownErrors.maxSeqLength}
            hint="Longer records are quarantined."
          >
            <input
              id="ds-seq"
              type="number"
              className={inputClass}
              value={form.maxSeqLength}
              onChange={(e) => set("maxSeqLength", Number(e.target.value))}
            />
          </Field>
        </fieldset>
        {shownErrors.split && (
          <p role="alert" className="text-sm text-rose-700 dark:text-rose-300">
            {shownErrors.split}
          </p>
        )}
        <div className="flex flex-wrap gap-4 text-sm">
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={form.allowDownload} onChange={(e) => set("allowDownload", e.target.checked)} />
            Download the tokenizer if it's missing (a few MB from Hugging Face)
          </label>
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={form.overwrite} onChange={(e) => set("overwrite", e.target.checked)} />
            Rebuild if this exact dataset exists
          </label>
        </div>
        {models.error ? <ErrorPanel error={models.error} /> : null}
        <Button type="submit" disabled={build.isPending}>
          Build dataset
        </Button>
      </form>
      {watching && (
        <div className="mt-4">
          <JobProgress jobId={watching} label="Build">
            {(state) => {
              const result = state.outcome?.result as BuildResult | null;
              return result ? <BuildSummary result={result} onBuilt={onBuilt} /> : null;
            }}
          </JobProgress>
        </div>
      )}
    </Card>
  );
}

interface BuildResult {
  dataset_id: string;
  name: string;
  rebuilt: boolean;
  warnings: Schemas["BuildWarning"][];
}

function BuildSummary({ result, onBuilt }: { result: BuildResult; onBuilt: (id: string) => void }) {
  const client = useQueryClient();
  useEffect(() => {
    void client.invalidateQueries({ queryKey: ["datasets"] });
  }, [client]);
  return (
    <div className="space-y-1 text-sm">
      <p>
        Built <strong>{result.name}</strong> ({result.dataset_id}){result.rebuilt ? ", rebuilt" : ""}.{" "}
        <Button variant="ghost" onClick={() => onBuilt(result.dataset_id)}>
          Show its statistics
        </Button>
      </p>
      <Warnings warnings={result.warnings} />
    </div>
  );
}

function Warnings({ warnings }: { warnings: Schemas["BuildWarning"][] }) {
  if (!warnings.length) return null;
  return (
    <ul className="space-y-1" aria-label="Warnings">
      {warnings.map((w) => (
        <li key={w.code + w.message} className="text-amber-800 dark:text-amber-300">
          <Badge tone="warn">{w.code}</Badge> {w.message}
        </li>
      ))}
    </ul>
  );
}

function DatasetStats({ datasetId }: { datasetId: string }) {
  const audit = useQuery({
    queryKey: ["dataset-audit", datasetId],
    queryFn: () => unwrap(api.GET("/api/datasets/{dataset}/audit", { params: { path: { dataset: datasetId } } })),
  });
  const report = useReportError();
  const validate = useMutation({
    mutationFn: () => unwrap(api.POST("/api/datasets/{dataset}/validate", { params: { path: { dataset: datasetId } } })),
    onError: report,
  });
  if (audit.error) return <ErrorPanel error={audit.error} />;
  if (!audit.data) return <Spinner />;
  const a = audit.data;
  return (
    <Card
      title={`Statistics: ${a.name}`}
      actions={
        <Button variant="secondary" onClick={() => validate.mutate()} disabled={validate.isPending}>
          Check files
        </Button>
      }
    >
      <div className="space-y-4 text-sm">
        <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="Records considered" value={a.input_records} />
          <Stat label="Included" value={a.included} />
          <Stat label="Duplicates removed" value={a.duplicates} />
          <Stat label="Largest group" value={`${a.largest_group_size} (${Math.round(a.largest_group_share * 100)}%)`} />
        </dl>
        <p>
          Model {a.model} · max {a.max_seq_length} tokens
        </p>
        <table className="w-full text-left">
          <caption className="text-left font-medium">Partitions</caption>
          <thead>
            <tr className="border-b border-slate-200 dark:border-slate-700">
              <th scope="col">Split</th>
              <th scope="col">Target</th>
              <th scope="col">Actual</th>
              <th scope="col">Examples</th>
              <th scope="col">Groups</th>
              <th scope="col">Tokens p50 / p90 / max</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(a.partitions).map(([name, p]) => (
              <tr key={name} className="border-b border-slate-100 dark:border-slate-800">
                <th scope="row">{name}</th>
                <td>{Math.round((a.split_target[name] ?? 0) * 100)}%</td>
                <td>{Math.round((a.split_actual[name] ?? 0) * 100)}%</td>
                <td>{p.examples}</td>
                <td>{p.groups}</td>
                <td>{p.tokens ? `${p.tokens.p50} / ${p.tokens.p90} / ${p.tokens.max}` : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div>
          Excluded: <CountList counts={a.excluded} />
        </div>
        <Warnings warnings={a.warnings} />
        {validate.data && (
          <div>
            <p className="font-medium">File check: {validate.data.ok ? "all good" : "problems found"}</p>
            <ul>
              {validate.data.checks.map((c) => (
                <li key={c.name}>
                  <Badge tone={statusTone(c.status)}>{c.status}</Badge> {c.name}: {c.message}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </Card>
  );
}
