// Preflight: every check before training as pass / warn / fail with its fix. Start stays disabled
// while any check fails (the server refuses too).
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api, type Schemas, unwrap } from "../api/client";
import { useReportError } from "../App";
import { Badge, Button, Card, ErrorPanel, Field, inputClass, Spinner, Stat, statusTone } from "../components/ui";
import { monitorHref, useJobs } from "../lib/queries";
import { routeHref, routeParams } from "../lib/route";
import { useConfigs } from "./TrainingConfig";

export function Preflight() {
  const configs = useConfigs();
  const params = routeParams();
  const [name, setName] = useState(params.get("config") ?? "");
  const [version, setVersion] = useState(params.get("version") ?? "");
  const [allowDownload, setAllowDownload] = useState(false);
  const chosen = name || configs.data?.[0]?.name || "";
  const summary = configs.data?.find((c) => c.name === chosen);

  if (configs.error) return <ErrorPanel error={configs.error} />;
  if (!configs.data) return <Spinner />;
  if (!configs.data.length) {
    return (
      <Card title="Preflight">
        <p className="text-sm">
          No saved training config yet. Make one under{" "}
          <a className="text-sky-800 underline dark:text-sky-300" href="#/config">
            Training config
          </a>
          .
        </p>
      </Card>
    );
  }
  return (
    <div className="space-y-4">
      <ModelLockNotice />
      <Card title="Config to check">
        <div className="grid gap-3 md:grid-cols-3">
          <Field label="Training config" htmlFor="pf-config">
            <select
              id="pf-config"
              className={inputClass}
              value={chosen}
              onChange={(e) => {
                setName(e.target.value);
                setVersion("");
              }}
            >
              {configs.data.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Version" htmlFor="pf-version" hint="Each saved change is a new version.">
            <select id="pf-version" className={inputClass} value={version} onChange={(e) => setVersion(e.target.value)}>
              <option value="">Latest{summary ? ` (v${summary.version})` : ""}</option>
              {Array.from({ length: summary?.versions ?? 0 }, (_, i) => i + 1)
                .reverse()
                .map((v) => (
                  <option key={v} value={v}>
                    v{v}
                  </option>
                ))}
            </select>
          </Field>
          <label className="flex items-center gap-2 self-end text-sm">
            <input type="checkbox" checked={allowDownload} onChange={(e) => setAllowDownload(e.target.checked)} />
            Download model files if they're missing (from Hugging Face)
          </label>
        </div>
      </Card>
      {chosen && (
        <Checks key={`${chosen}:${version}:${allowDownload}`} name={chosen} version={version ? Number(version) : undefined} allowDownload={allowDownload} />
      )}
    </div>
  );
}

function Checks({ name, version, allowDownload }: { name: string; version: number | undefined; allowDownload: boolean }) {
  const body = { saved_name: name, saved_version: version ?? null, allow_download: allowDownload };
  const preflight = useQuery({
    queryKey: ["preflight", name, version, allowDownload],
    queryFn: () => unwrap(api.POST("/api/preflight", { body })),
    staleTime: 0,
  });
  const jobs = useJobs(50);
  const client = useQueryClient();
  const report = useReportError();
  const start = useMutation({
    mutationFn: () => unwrap(api.POST("/api/runs", { body })),
    onSuccess: (started) => {
      void client.invalidateQueries({ queryKey: ["jobs"] });
      void client.invalidateQueries({ queryKey: ["runs"] });
      window.location.hash = monitorHref(started.run_id);
    },
    onError: report,
  });
  const holderId = jobs.data?.model_lock_holder ?? null;

  if (preflight.error) return <ErrorPanel error={preflight.error} />;
  if (!preflight.data) return <Spinner label="Running the checks (this loads the tokenizer)…" />;
  const result = preflight.data;
  const failed = result.checks.filter((c) => c.status === "fail").length;
  const warned = result.checks.filter((c) => c.status === "warn").length;
  const canStart = result.ok && !preflight.isFetching && !start.isPending && !start.isSuccess && !holderId;
  return (
    <Card
      title={`Checks: ${name}${version ? ` v${version}` : ""}`}
      actions={
        <Button variant="secondary" onClick={() => void preflight.refetch()} disabled={preflight.isFetching}>
          {preflight.isFetching ? "Checking…" : "Run again"}
        </Button>
      }
    >
      <div className="space-y-4 text-sm">
        <p role="status" className={failed ? "font-medium text-rose-800 dark:text-rose-300" : "font-medium text-emerald-800 dark:text-emerald-300"}>
          {failed
            ? `${failed} check${failed === 1 ? "" : "s"} failed. Fix ${failed === 1 ? "it" : "them"} before training.`
            : `Ready to train${warned ? ` (${warned} warning${warned === 1 ? "" : "s"})` : ""}.`}
        </p>
        <ul className="space-y-2" aria-label="Preflight checks">
          {result.checks.map((check, index) => (
            <li key={`${check.name}-${index}`} className="rounded-md border border-slate-200 p-2 dark:border-slate-700">
              <div className="flex flex-wrap items-center gap-2">
                <Badge tone={statusTone(check.status)}>{check.status}</Badge>
                <span className="font-medium">{check.name}</span>
              </div>
              <p className="mt-1">{check.message}</p>
              {check.fix && <p className="mt-1 text-slate-700 dark:text-slate-300">What to do: {check.fix}</p>}
            </li>
          ))}
        </ul>
        <Summary result={result} />
        <div className="flex flex-wrap items-center gap-3">
          <Button onClick={() => start.mutate()} disabled={!canStart} aria-describedby="pf-start-note">
            Start training
          </Button>
          <span id="pf-start-note" className="text-slate-600 dark:text-slate-400">
            {failed
              ? "Start is disabled while any check fails."
              : holderId
                ? "Start is disabled while another job is using the model."
                : "Training runs as a background job; the run monitor opens when it starts."}
          </span>
          <a className="text-sky-800 underline dark:text-sky-300" href={routeHref("/config")}>
            Change the config
          </a>
        </div>
      </div>
    </Card>
  );
}

/** Names the job holding the model lock, if any: training can't start until it finishes. */
function ModelLockNotice() {
  const jobs = useJobs(50);
  const holderId = jobs.data?.model_lock_holder ?? null;
  if (!holderId) return null;
  const holder = jobs.data?.jobs.find((j) => j.id === holderId);
  return (
    <p
      role="status"
      className="rounded-md border border-amber-300 bg-amber-50 p-2 text-sm text-amber-950 dark:border-amber-800 dark:bg-amber-950/50 dark:text-amber-100"
    >
      Another job is using the model: {holderId}
      {holder ? ` (${holder.type}, ${holder.status})` : ""}. One model job runs at a time; start training when it has
      finished.{" "}
      {holder?.run_id && (
        <a className="text-sky-800 underline dark:text-sky-300" href={monitorHref(holder.run_id)}>
          Watch it
        </a>
      )}
    </p>
  );
}

function Summary({ result }: { result: Schemas["PreflightResult"] }) {
  const plan = result.plan as Record<string, unknown> | null;
  const training = result.effective_training as Record<string, unknown> | null;
  return (
    <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      <Stat label="Model" value={result.model} />
      <Stat label="Dataset" value={result.dataset_id ?? "none"} />
      <Stat
        label="Backend"
        value={result.selection ? `${result.selection.backend} on ${result.selection.device} (${result.selection.precision})` : "none"}
      />
      {plan && <Stat label="Optimizer updates (at most)" value={String(plan.total_updates ?? "—")} />}
      {training && (
        <>
          <Stat label="Max epochs" value={String(training.max_epochs)} />
          <Stat label="Learning rate" value={String(training.learning_rate)} />
          <Stat label="Validate every" value={`${String(training.eval_every_steps)} updates`} />
          <Stat label="Max sequence length" value={String(training.max_seq_length)} />
        </>
      )}
    </dl>
  );
}
