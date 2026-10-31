// Models: the curated catalog with what this machine can do with each model. Memory figures are
// catalog estimates, never measurements. Downloads run as jobs with progress.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { api, type Schemas, unwrap } from "../api/client";
import { useReportError } from "../App";
import { JobProgress } from "../components/JobProgress";
import { Badge, Button, Card, ErrorPanel, Spinner } from "../components/ui";
import { useLatestJob, useSettings } from "../lib/queries";

const FIT_TONE = { fits: "good", tight: "warn", "won't fit": "bad", unknown: "neutral" } as const;

export function useModels() {
  const settings = useSettings();
  return useQuery({
    queryKey: ["models", settings.data?.dev_mode],
    enabled: settings.data !== undefined,
    queryFn: () =>
      unwrap(api.GET("/api/models", { params: { query: { include_test_only: settings.data?.dev_mode ?? false } } })),
  });
}

export function Models() {
  const models = useModels();
  const latest = useLatestJob("model_download");
  const [started, setStarted] = useState<Record<string, string>>({}); // model id → job id
  if (models.error) return <ErrorPanel error={models.error} />;
  if (!models.data) return <Spinner label="Reading the catalog…" />;
  const startedJobs = new Set(Object.values(started));
  return (
    <div className="space-y-4">
      <p className="text-sm text-slate-600 dark:text-slate-400">{models.data.estimate_note}</p>
      {latest && !startedJobs.has(latest.id) && latest.status !== "succeeded" && (
        <Card title="Latest download">
          <JobProgress jobId={latest.id} label="Download">
            {(state) => (state.status === "succeeded" ? <Downloaded /> : null)}
          </JobProgress>
        </Card>
      )}
      <ul className="grid gap-4 md:grid-cols-2" aria-label="Models">
        {models.data.models.map((model) => (
          <li key={model.id}>
            <ModelCard model={model} jobId={started[model.id]} onStarted={(id) => setStarted((s) => ({ ...s, [model.id]: id }))} />
          </li>
        ))}
      </ul>
    </div>
  );
}

function Downloaded() {
  const client = useQueryClient();
  useEffect(() => {
    void client.invalidateQueries({ queryKey: ["models"] });
  }, [client]);
  return <p className="text-sm">Downloaded.</p>;
}

function ModelCard({
  model,
  jobId,
  onStarted,
}: {
  model: Schemas["ModelSummary"];
  jobId: string | undefined;
  onStarted: (jobId: string) => void;
}) {
  const report = useReportError();
  const client = useQueryClient();
  const download = useMutation({
    mutationFn: () =>
      unwrap(api.POST("/api/models/{model_id}/download", { params: { path: { model_id: model.id } }, body: {} })),
    onSuccess: (job) => {
      onStarted(job.id);
      void client.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: report,
  });
  const compat = model.compatibility;
  return (
    <article
      aria-label={model.display_name}
      className="h-full space-y-3 rounded-lg border border-slate-200 bg-white p-4 text-sm shadow-sm dark:border-slate-700 dark:bg-slate-900"
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h2 className="text-base font-semibold">{model.display_name}</h2>
          <p className="text-xs text-slate-600 dark:text-slate-400">{model.id}</p>
        </div>
        <div className="flex flex-wrap gap-1">
          {model.test_only && <Badge tone="warn">test only</Badge>}
          {model.trainable ? <Badge tone="good">template test passed</Badge> : <Badge tone="bad">template test failed</Badge>}
        </div>
      </div>
      <dl className="grid grid-cols-2 gap-2">
        <div>
          <dt className="text-xs text-slate-600 dark:text-slate-400">Parameters</dt>
          <dd>{model.params_b}B</dd>
        </div>
        <div>
          <dt className="text-xs text-slate-600 dark:text-slate-400">License</dt>
          <dd>{model.license}</dd>
        </div>
        <div>
          <dt className="text-xs text-slate-600 dark:text-slate-400">Backends here</dt>
          <dd>{model.backends_here.length ? model.backends_here.join(", ") : "none"}</dd>
        </div>
        <div>
          <dt className="text-xs text-slate-600 dark:text-slate-400">Download</dt>
          <dd>
            {model.download_gb !== null && model.download_gb !== undefined ? `${model.download_gb} GB` : "—"}{" "}
            {model.cached ? <Badge tone="good">downloaded</Badge> : <Badge>not downloaded</Badge>}
          </dd>
        </div>
        <div className="col-span-2">
          <dt className="text-xs text-slate-600 dark:text-slate-400">Memory on this machine (estimate)</dt>
          <dd>
            {compat ? (
              <>
                <Badge tone={FIT_TONE[compat.label]}>{compat.label}</Badge>{" "}
                {compat.estimate_gb !== null && compat.estimate_gb !== undefined && compat.pool_gb !== null && compat.pool_gb !== undefined
                  ? `about ${compat.estimate_gb} GB needed of ${compat.pool_gb.toFixed(1)} GB (${compat.pool})`
                  : "no estimate for this backend"}
                {model.selection ? ` · ${model.selection.backend} on ${model.selection.device} (${model.selection.precision})` : ""}
              </>
            ) : (
              "no backend for it here"
            )}
          </dd>
        </div>
      </dl>
      {model.notes.length > 0 && (
        <ul className="list-disc pl-5 text-amber-800 dark:text-amber-300">
          {model.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      )}
      <p className="text-xs text-slate-600 dark:text-slate-400">
        Model card: <span className="break-all">{model.model_card_url}</span>
      </p>
      {!model.cached && model.selection && !jobId && (
        <Button onClick={() => download.mutate()} disabled={download.isPending}>
          Download{model.download_gb ? ` (${model.download_gb} GB)` : ""}
        </Button>
      )}
      {jobId && (
        <JobProgress jobId={jobId} label="Download">
          {(state) => (state.status === "succeeded" ? <Downloaded /> : null)}
        </JobProgress>
      )}
    </article>
  );
}
