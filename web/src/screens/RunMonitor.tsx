// Run monitor: a training run's status, step and ETA, loss chart with the best checkpoint, live log,
// warnings, memory, Cancel and Resume. The state comes from the server (registry + events files),
// then follows the job's event stream, so reloading the page or restarting `serve` shows the same run.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

import { api, type Schemas, unwrap } from "../api/client";
import { type Connection, followJob } from "../api/sse";
import { useReportError } from "../App";
import { jobError } from "../components/JobProgress";
import { LossChart } from "../components/LossChart";
import { Badge, Button, Card, ErrorPanel, Field, inputClass, Spinner, Stat, statusTone } from "../components/ui";
import { applyEvent, etaText, fmt, type Monitor, STATUS_HELP } from "../lib/monitor";
import { isFinal } from "../lib/useJob";
import { monitorHref, useRuns } from "../lib/queries";
import { routeHref, routeParams } from "../lib/route";

const LINK = "text-sky-800 underline dark:text-sky-300";

export function RunMonitor() {
  const runs = useRuns();
  const asked = routeParams().get("run");
  if (runs.error) return <ErrorPanel error={runs.error} />;
  if (!runs.data) return <Spinner />;
  if (!runs.data.length && !asked) {
    return (
      <Card title="No training runs yet">
        <p className="text-sm">
          Start one from{" "}
          <a className={LINK} href="#/preflight">
            Preflight
          </a>{" "}
          once a training config passes its checks.
        </p>
      </Card>
    );
  }
  const active = runs.data.find((r) => r.status === "running" || r.status === "queued");
  const runId = asked ?? active?.id ?? runs.data[0]?.id ?? "";
  return (
    <div className="space-y-4">
      <Field label="Run" htmlFor="mon-run" hint="Newest first. The page follows the run you pick, also after a reload.">
        <select
          id="mon-run"
          className={`${inputClass} md:w-auto`}
          value={runId}
          onChange={(e) => {
            window.location.hash = monitorHref(e.target.value);
          }}
        >
          {!runs.data.some((r) => r.id === runId) && <option value={runId}>{runId}</option>}
          {runs.data.map((r) => (
            <option key={r.id} value={r.id}>
              {r.id} · {r.model_id} · {r.status}
            </option>
          ))}
        </select>
      </Field>
      <Monitor key={runId} runId={runId} />
    </div>
  );
}

function useNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const timer = setInterval(() => setNow(Date.now()), 1_000);
    return () => clearInterval(timer);
  }, [active]);
  return now;
}

/** Live events applied on top of one snapshot (`base`); a new snapshot starts over. */
interface Live {
  base: Monitor;
  state: Monitor;
  connection: Connection | null;
  done: boolean;
  error: unknown;
}

function Monitor({ runId }: { runId: string }) {
  const client = useQueryClient();
  const report = useReportError();
  const snapshot = useQuery({
    queryKey: ["monitor", runId],
    queryFn: () => unwrap(api.GET("/api/runs/{run_id}/monitor", { params: { path: { run_id: runId } } })),
    // While a job is active its event stream drives the page (and refetches at the end);
    // otherwise poll, e.g. for a resume started from the CLI or the model lock being freed.
    refetchInterval: (query) => {
      const job = query.state.data?.job;
      return job && !isFinal(job.status) ? false : 5_000;
    },
  });
  const [tracked, setTracked] = useState<Live | null>(null);
  const data = snapshot.data;

  useEffect(() => {
    const job = data?.job;
    if (!data || !job || isFinal(job.status)) return;
    const controller = new AbortController();
    const update = (change: (live: Live) => Live) =>
      setTracked((prev) =>
        change(prev && prev.base === data ? prev : { base: data, state: data, connection: null, done: false, error: null }),
      );
    followJob(job.id, {
      from: data.offset,
      afterSeq: data.last_seq,
      signal: controller.signal,
      onConnection: (connection) => update((live) => ({ ...live, connection })),
      onEvent: (event) => update((live) => ({ ...live, state: applyEvent(live.state, event, job.id) })),
    })
      .then(() => {
        update((live) => ({ ...live, done: true, connection: null }));
        // The final status is in: fetch the result, error and what can happen next.
        void client.invalidateQueries({ queryKey: ["monitor", runId] });
        void client.invalidateQueries({ queryKey: ["runs"] });
        void client.invalidateQueries({ queryKey: ["jobs"] });
      })
      .catch((error: unknown) => {
        if ((error as Error).name !== "AbortError") update((live) => ({ ...live, error, connection: null }));
      });
    return () => controller.abort();
  }, [data, client, runId]);

  const current = tracked && tracked.base === data ? tracked : null;
  const live = current?.state ?? data ?? null;
  const connection = current?.connection ?? null;
  const streamError = current?.error ?? null;

  const cancel = useMutation({
    mutationFn: (jobId: string) => unwrap(api.POST("/api/jobs/{job_id}/cancel", { params: { path: { job_id: jobId } }, body: { timeout: 60 } })),
    onSuccess: () => void client.invalidateQueries({ queryKey: ["monitor", runId] }),
    onError: report,
  });
  const resume = useMutation({
    mutationFn: () => unwrap(api.POST("/api/runs/{run_id}/resume", { params: { path: { run_id: runId } }, body: { allow_download: false } })),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["monitor", runId] });
      void client.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: report,
  });

  const status = live?.job?.status ?? live?.run.status ?? "queued";
  const now = useNow(status === "running");
  if (snapshot.error && !live) return <ErrorPanel error={snapshot.error} />;
  if (!live) return <Spinner label="Loading the run…" />;

  const percent = live.step != null && live.max_steps ? Math.round((live.step / live.max_steps) * 100) : null;
  const eta = status === "running" ? etaText(live, now) : null;
  const memory = live.memory;
  return (
    <div className="space-y-4">
      <Card
        title={`Run ${live.run.id}`}
        actions={
          <div className="flex flex-wrap items-center gap-2">
            {connection && (
              <span role="status" className="text-xs text-slate-600 dark:text-slate-400">
                {connection === "live" ? "● Live" : connection === "reconnecting" ? "Reconnecting…" : "Connecting…"}
              </span>
            )}
            <Button
              variant="danger"
              onClick={() => live.job && cancel.mutate(live.job.id)}
              disabled={!live.can_cancel || cancel.isPending}
            >
              {cancel.isPending ? "Cancelling…" : "Cancel"}
            </Button>
            <Button onClick={() => resume.mutate()} disabled={!live.can_resume || resume.isPending}>
              {resume.isPending ? "Resuming…" : "Resume"}
            </Button>
          </div>
        }
      >
        <div className="space-y-3 text-sm">
          <div className="flex flex-wrap items-center gap-2">
            <span>Status</span>
            <Badge tone={statusTone(status)}>{status}</Badge>
            <span className="text-slate-600 dark:text-slate-400">
              {live.run.model_id} · {live.run.backend} on {live.run.device} ({live.run.precision}) · dataset {live.run.dataset_id ?? "—"}
              {live.config_name ? ` · config ${live.config_name} v${live.config_version}` : ""}
            </span>
          </div>
          <p aria-live="polite">{STATUS_HELP[status] ?? status}</p>
          {cancel.isPending && <p role="status">Stopping at the next step and saving a checkpoint…</p>}
          {live.lock_holder && (
            <p role="status" className="rounded-md border border-amber-300 bg-amber-50 p-2 text-amber-950 dark:border-amber-800 dark:bg-amber-950/50 dark:text-amber-100">
              {live.lock_holder.message} One model job runs at a time.{" "}
              {live.lock_holder.run_id && (
                <a className={LINK} href={monitorHref(live.lock_holder.run_id)}>
                  Watch it
                </a>
              )}
            </p>
          )}
          {!live.can_resume && live.resume_note && !live.lock_holder && (
            <p className="text-slate-700 dark:text-slate-300">{live.resume_note}</p>
          )}
          {streamError ? <ErrorPanel error={streamError} /> : null}
          {live.error ? <ErrorPanel error={jobError(live.error)} /> : null}
          {status === "running" || status === "queued" ? (
            <progress className="h-2 w-full" max={100} value={percent ?? undefined} aria-label="Training progress" />
          ) : null}
          <dl className="grid grid-cols-2 gap-3 sm:grid-cols-5">
            <Stat label="Step" value={live.step != null ? `${live.step}${live.max_steps ? ` / ${live.max_steps}` : ""}` : "—"} />
            <Stat label="Epoch" value={live.epoch != null ? live.epoch.toFixed(2) : "—"} />
            <Stat label="Time left" value={eta ?? "—"} />
            <Stat
              label="Best checkpoint"
              value={live.best ? `step ${live.best.step}` : "—"}
            />
            <Stat
              label={`Peak memory (${memory.pool})`}
              value={memory.peak_gb != null ? `${memory.peak_gb} GB${memory.total_gb ? ` of ${memory.total_gb} GB` : ""}` : "—"}
            />
          </dl>
        </div>
      </Card>

      <Card title="Loss">
        <LossChart points={live.points} best={live.best} />
      </Card>

      {live.warnings.length > 0 && (
        <Card title="Warnings">
          <ul className="list-disc space-y-1 pl-5 text-sm">
            {live.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </Card>
      )}

      {live.result && status === "succeeded" && <ResultCard result={live.result} runId={live.run.id} />}

      <Card title="Log">
        <EventLog lines={live.log} />
      </Card>

      <Card title="Attempts">
        <ul className="space-y-1 text-sm" aria-label="Attempts">
          {live.attempts.map((job, index) => (
            <li key={job.id} className="flex flex-wrap items-center gap-2">
              <span>{index === 0 ? "Start" : `Resume ${index}`}</span>
              <Badge tone={statusTone(job.id === live.job?.id ? status : job.status)}>{job.id === live.job?.id ? status : job.status}</Badge>
              <span className="text-xs text-slate-600 dark:text-slate-400">
                job {job.id} · {when(job.created_at)}
                {job.finished_at ? ` → ${when(job.finished_at)}` : ""}
              </span>
            </li>
          ))}
        </ul>
      </Card>
    </div>
  );
}

function ResultCard({ result, runId }: { result: Record<string, unknown>; runId: string }) {
  return (
    <Card title="Result">
      <div className="space-y-2 text-sm">
        <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="Best step" value={String(result.best_step ?? "—")} />
          <Stat label="Best validation loss" value={typeof result.best_eval_loss === "number" ? fmt(result.best_eval_loss) : "—"} />
          <Stat label="Updates done" value={`${String(result.steps_done ?? "—")} / ${String(result.total_steps ?? "—")}`} />
          <Stat label="Stopped early" value={result.stopped_early ? "yes" : "no"} />
        </dl>
        {typeof result.adapter_path === "string" && <p>Adapter: {result.adapter_path}</p>}
        {typeof result.resume_note === "string" && <p>Resume: {result.resume_note}</p>}
        <p className="text-slate-600 dark:text-slate-400">
          Next:{" "}
          <a className={LINK} href={routeHref("/evaluation", { run: runId })}>
            evaluate the run
          </a>{" "}
          against the base model to see whether fine-tuning helped.
        </p>
      </div>
    </Card>
  );
}

/** A date and time in this computer's local time. */
function when(time: string): string {
  const parsed = new Date(time);
  return Number.isNaN(parsed.getTime()) ? time : parsed.toLocaleString();
}

/** An event's time in this computer's local time. */
function clock(time: string): string {
  const parsed = new Date(time);
  return Number.isNaN(parsed.getTime()) ? time : parsed.toLocaleTimeString();
}

function EventLog({ lines }: { lines: Schemas["LogLine"][] }) {
  const box = useRef<HTMLOListElement>(null);
  const pinned = useRef(true);
  useEffect(() => {
    const el = box.current;
    if (el && pinned.current) el.scrollTop = el.scrollHeight;
  }, [lines]);
  if (!lines.length) return <p className="text-sm text-slate-600 dark:text-slate-400">No events yet.</p>;
  return (
    <ol
      ref={box}
      role="log"
      aria-label="Event log"
      tabIndex={0}
      onScroll={(e) => {
        const el = e.currentTarget;
        pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
      }}
      className="max-h-72 space-y-0.5 overflow-y-auto rounded-md bg-slate-50 p-2 font-mono text-xs dark:bg-slate-950"
    >
      {lines.map((line) => (
        <li
          key={`${line.job_id}-${line.seq}-${line.message}`}
          className={
            line.level === "error"
              ? "text-rose-800 dark:text-rose-300"
              : line.level === "warning"
                ? "text-amber-800 dark:text-amber-300"
                : undefined
          }
        >
          <span className="text-slate-500">{clock(line.time)}</span> {line.message}
        </li>
      ))}
    </ol>
  );
}
