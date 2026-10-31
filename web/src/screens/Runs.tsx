// Runs: history with filters, compare two runs, open artifacts, export or merge, archive, and a
// permanent delete that needs the run ID typed back.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api, rawFetch, type Schemas, unwrap } from "../api/client";
import { useReportError } from "../App";
import { JobProgress } from "../components/JobProgress";
import { Badge, Button, Card, ErrorPanel, Field, inputClass, Spinner, statusTone } from "../components/ui";
import { fmt } from "../lib/monitor";
import { monitorHref, type HistoryFilter, useHistory } from "../lib/queries";
import { routeHref, routeParams } from "../lib/route";

type Run = Schemas["RunSummary"];

const LINK = "text-sky-800 underline dark:text-sky-300";
const STATUSES = ["queued", "running", "succeeded", "failed", "cancelled", "interrupted"];
const VERDICT_TONE = { accepted: "good", rejected: "bad", inconclusive: "warn" } as const;

function verdictBadge(run: Run) {
  if (!run.verdict) return <span className="text-slate-500">—</span>;
  return <Badge tone={VERDICT_TONE[run.verdict.verdict as keyof typeof VERDICT_TONE] ?? "neutral"}>{run.verdict.verdict}</Badge>;
}

export function Runs() {
  const [filter, setFilter] = useState<HistoryFilter>({});
  const [compare, setCompare] = useState<string[]>([]);
  const selected = routeParams().get("run");
  const runs = useHistory(filter);
  const all = useHistory({ include_archived: true });
  const models = [...new Set((all.data ?? []).map((r) => r.model_id))];

  const toggle = (id: string) =>
    setCompare((list) => (list.includes(id) ? list.filter((x) => x !== id) : [...list, id].slice(-2)));

  return (
    <div className="space-y-4">
      <Card title="History">
        <div className="space-y-3 text-sm">
          <div className="flex flex-wrap items-end gap-3">
            <Field label="Status" htmlFor="runs-status">
              <select
                id="runs-status"
                className={`${inputClass} w-40`}
                value={filter.status ?? ""}
                onChange={(e) => setFilter((f) => ({ ...f, status: e.target.value || undefined }))}
              >
                <option value="">Any</option>
                {STATUSES.map((s) => (
                  <option key={s}>{s}</option>
                ))}
              </select>
            </Field>
            <Field label="Model" htmlFor="runs-model">
              <select
                id="runs-model"
                className={`${inputClass} w-48`}
                value={filter.model_id ?? ""}
                onChange={(e) => setFilter((f) => ({ ...f, model_id: e.target.value || undefined }))}
              >
                <option value="">Any</option>
                {models.map((m) => (
                  <option key={m}>{m}</option>
                ))}
              </select>
            </Field>
            <Field label="Run ID or config name" htmlFor="runs-text">
              <input
                id="runs-text"
                className={`${inputClass} w-56`}
                value={filter.text ?? ""}
                onChange={(e) => setFilter((f) => ({ ...f, text: e.target.value || undefined }))}
              />
            </Field>
            <label className="flex items-center gap-2 pb-2">
              <input
                type="checkbox"
                checked={!!filter.include_archived}
                onChange={(e) => setFilter((f) => ({ ...f, include_archived: e.target.checked || undefined }))}
              />
              Show archived runs
            </label>
          </div>
          {runs.error ? (
            <ErrorPanel error={runs.error} />
          ) : !runs.data ? (
            <Spinner />
          ) : runs.data.length === 0 ? (
            <p>No runs match.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left" aria-label="Runs">
                <thead>
                  <tr className="border-b border-slate-200 dark:border-slate-700">
                    <th className="py-1 pr-2">Compare</th>
                    <th className="pr-3">Run</th>
                    <th className="pr-3">Status</th>
                    <th className="pr-3">Model</th>
                    <th className="pr-3">Config</th>
                    <th className="pr-3">Best validation loss</th>
                    <th className="pr-3">Verdict</th>
                    <th>Started</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.data.map((r) => (
                    <tr key={r.id} className="border-b border-slate-100 dark:border-slate-800" aria-current={r.id === selected ? "true" : undefined}>
                      <td className="py-1 pr-2">
                        <input type="checkbox" aria-label={`Compare ${r.id}`} checked={compare.includes(r.id)} onChange={() => toggle(r.id)} />
                      </td>
                      <td className="pr-3">
                        <a className={LINK} href={routeHref("/runs", { run: r.id })}>
                          {r.id}
                        </a>
                        {r.archived && (
                          <span className="ml-1">
                            <Badge>archived</Badge>
                          </span>
                        )}
                      </td>
                      <td className="pr-3">
                        <Badge tone={statusTone(r.status)}>{r.status}</Badge>
                      </td>
                      <td className="pr-3">{r.model_id}</td>
                      <td className="pr-3">{r.config_name ? `${r.config_name} v${r.config_version}` : "—"}</td>
                      <td className="pr-3">{r.best_eval_loss != null ? fmt(r.best_eval_loss) : "—"}</td>
                      <td className="pr-3">{verdictBadge(r)}</td>
                      <td>{new Date(r.created_at).toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <p className="text-slate-600 dark:text-slate-400">Tick two runs to compare them. Open a run for its artifacts, exports and actions.</p>
        </div>
      </Card>
      {compare.length === 2 && <Compare a={compare[0] as string} b={compare[1] as string} />}
      {selected && <RunDetail key={selected} runId={selected} run={(all.data ?? []).find((r) => r.id === selected)} />}
    </div>
  );
}

// --------------------------------------------------------------------------- compare

function Compare({ a, b }: { a: string; b: string }) {
  const result = useQuery({
    queryKey: ["compare", a, b],
    queryFn: () => unwrap(api.GET("/api/compare-runs", { params: { query: { a, b } } })),
  });
  if (result.error) return <ErrorPanel error={result.error} />;
  if (!result.data) return <Spinner label="Comparing…" />;
  const { a: fa, b: fb } = result.data;
  const rows: [string, (f: Schemas["RunFigures"]) => string][] = [
    ["Status", (f) => f.run.status],
    ["Model", (f) => `${f.run.model_id} (${f.run.backend} on ${f.run.device}, ${f.run.precision})`],
    ["Dataset", (f) => f.run.dataset_id ?? "—"],
    ["Config", (f) => (f.run.config_name ? `${f.run.config_name} v${f.run.config_version}` : "—")],
    ["Best validation loss", (f) => (f.run.best_eval_loss != null ? fmt(f.run.best_eval_loss) : "—")],
    ["Best step", (f) => String(f.best_step ?? "—")],
    ["Updates done", (f) => `${f.steps_done ?? "—"} / ${f.total_steps ?? "—"}`],
    ["Stopped early", (f) => (f.stopped_early == null ? "—" : f.stopped_early ? "yes" : "no")],
    ["Peak memory", (f) => (f.peak_memory_gb != null ? `${f.peak_memory_gb} GB` : "—")],
    ["Training time", (f) => (f.train_seconds != null ? `${Math.round(f.train_seconds)} s` : "—")],
    ["Attempts", (f) => String(f.attempts)],
    [
      "Verdict",
      (f) =>
        f.run.verdict
          ? `${f.run.verdict.verdict} (${f.run.verdict.wins ?? "?"} won / ${f.run.verdict.losses ?? "?"} lost / ${f.run.verdict.ties ?? "?"} tied)`
          : "not evaluated",
    ],
  ];
  return (
    <Card title={`Compare ${a} and ${b}`}>
      <div className="space-y-4 text-sm">
        {result.data.note && <p className="text-amber-800 dark:text-amber-300">{result.data.note}</p>}
        <table className="w-full text-left" aria-label="Metrics and verdicts">
          <thead>
            <tr className="border-b border-slate-200 dark:border-slate-700">
              <th className="py-1 pr-3" />
              <th className="pr-3">{a}</th>
              <th>{b}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([label, value]) => (
              <tr key={label} className="border-b border-slate-100 dark:border-slate-800">
                <th className="py-1 pr-3 font-medium">{label}</th>
                <td className="pr-3">{value(fa)}</td>
                <td>{value(fb)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div>
          <h3 className="mb-1 font-semibold">Config differences</h3>
          {result.data.config_differences.length === 0 ? (
            <p>The effective configs are the same.</p>
          ) : (
            <table className="text-left" aria-label="Config differences">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-700">
                  <th className="py-1 pr-4">Setting</th>
                  <th className="pr-4">{a}</th>
                  <th>{b}</th>
                </tr>
              </thead>
              <tbody>
                {result.data.config_differences.map((d) => (
                  <tr key={d.key}>
                    <td className="py-1 pr-4 font-mono text-xs">{d.key}</td>
                    <td className="pr-4">{JSON.stringify(d.a)}</td>
                    <td>{JSON.stringify(d.b)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </Card>
  );
}

// --------------------------------------------------------------------------- one run

async function download(runId: string, path: string) {
  const response = await rawFetch(`/api/runs/${encodeURIComponent(runId)}/artifacts/file?path=${encodeURIComponent(path)}`);
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement("a");
  link.href = url;
  link.download = path.split("/").pop() ?? "artifact";
  link.click();
  URL.revokeObjectURL(url);
}

function size(bytes: number): string {
  if (bytes >= 2 ** 30) return `${(bytes / 2 ** 30).toFixed(1)} GB`;
  if (bytes >= 2 ** 20) return `${(bytes / 2 ** 20).toFixed(1)} MB`;
  if (bytes >= 2 ** 10) return `${(bytes / 2 ** 10).toFixed(1)} kB`;
  return `${bytes} B`;
}

function RunDetail({ runId, run }: { runId: string; run: Run | undefined }) {
  const client = useQueryClient();
  const report = useReportError();
  const artifacts = useQuery({
    queryKey: ["artifacts", runId],
    queryFn: () => unwrap(api.GET("/api/runs/{run_id}/artifacts", { params: { path: { run_id: runId } } })),
  });
  const exports = useQuery({
    queryKey: ["exports", runId],
    queryFn: () => unwrap(api.GET("/api/exports", { params: { query: { run_id: runId } } })),
    refetchInterval: 5_000,
  });
  const [mergeJob, setMergeJob] = useState<string | null>(null);
  const refresh = () => {
    for (const key of [["history"], ["artifacts", runId], ["exports", runId]]) void client.invalidateQueries({ queryKey: key });
  };
  const exportRun = useMutation({
    mutationFn: (merge: boolean) =>
      unwrap(api.POST("/api/runs/{run_id}/exports", { params: { path: { run_id: runId } }, body: { merge, allow_download: false } })),
    onSuccess: (started) => {
      if (started.job) setMergeJob(started.job.id);
      refresh();
    },
    onError: report,
  });
  const archive = useMutation({
    mutationFn: (archived: boolean) =>
      unwrap(api.POST("/api/runs/{run_id}/archive", { params: { path: { run_id: runId } }, body: { archived } })),
    onSuccess: refresh,
    onError: report,
  });
  const finished = run?.status === "succeeded";

  return (
    <Card title={`Run ${runId}`}>
      <div className="space-y-4 text-sm">
        {run && (
          <p className="flex flex-wrap items-center gap-2">
            <Badge tone={statusTone(run.status)}>{run.status}</Badge>
            {run.archived && <Badge>archived</Badge>}
            <span>
              {run.model_id} · {run.backend} on {run.device} ({run.precision}) · dataset {run.dataset_id ?? "—"}
            </span>
          </p>
        )}
        <div className="flex flex-wrap gap-3">
          <a className={LINK} href={monitorHref(runId)}>
            Run monitor
          </a>
          {finished && (
            <>
              <a className={LINK} href={routeHref("/evaluation", { run: runId })}>
                Evaluate
              </a>
              <a className={LINK} href={routeHref("/chat", { run: runId })}>
                Chat
              </a>
            </>
          )}
        </div>

        <section aria-label="Artifacts" className="space-y-1">
          <h3 className="font-semibold">Artifacts</h3>
          {artifacts.error ? (
            <ErrorPanel error={artifacts.error} />
          ) : !artifacts.data ? (
            <Spinner />
          ) : (
            <>
              <p className="text-slate-600 dark:text-slate-400">Folder: {artifacts.data.run_dir}</p>
              <ul className="space-y-0.5">
                {artifacts.data.artifacts.map((a) => (
                  <li key={a.path} className="flex flex-wrap items-center gap-2">
                    <span className="font-mono text-xs">
                      {a.path}
                      {a.kind === "folder" ? "/" : ""}
                    </span>
                    <span className="text-xs text-slate-600 dark:text-slate-400">
                      {size(a.bytes)}
                      {a.kind === "folder" ? `, ${a.files} files` : ""}
                    </span>
                    {a.downloadable && (
                      <Button variant="ghost" onClick={() => void download(runId, a.path).catch(report)} aria-label={`Download ${a.path}`}>
                        Download
                      </Button>
                    )}
                  </li>
                ))}
              </ul>
              {artifacts.data.external_exports.length > 0 && <p>Exports outside the run folder: {artifacts.data.external_exports.join(", ")}</p>}
            </>
          )}
        </section>

        <section aria-label="Export" className="space-y-2">
          <h3 className="font-semibold">Export</h3>
          <p className="text-slate-600 dark:text-slate-400">
            The adapter export copies the adapter with its metadata and base-model reference. A merged model is a standalone
            copy of base + adapter, checked against the adapter on 5 prompts; it runs as a job.
          </p>
          <div className="flex flex-wrap gap-2">
            <Button variant="secondary" onClick={() => exportRun.mutate(false)} disabled={!finished || exportRun.isPending}>
              Export the adapter
            </Button>
            <Button variant="secondary" onClick={() => exportRun.mutate(true)} disabled={!finished || exportRun.isPending || !!mergeJob}>
              Export a merged model
            </Button>
          </div>
          {!finished && <p>Only finished runs can be exported.</p>}
          {mergeJob && <JobProgress jobId={mergeJob} label="Merging" />}
          {exports.data && exports.data.length > 0 && (
            <ul className="space-y-0.5" aria-label="Exports">
              {exports.data.map((e) => (
                <li key={e.id} className="flex flex-wrap items-center gap-2">
                  <Badge tone={statusTone(e.status)}>{e.status}</Badge>
                  <span>{e.kind}</span>
                  <span className="font-mono text-xs">{e.path}</span>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section aria-label="Archive or delete" className="space-y-2">
          <h3 className="font-semibold">Archive or delete</h3>
          <div className="flex flex-wrap items-center gap-2">
            {run?.archived ? (
              <Button variant="secondary" onClick={() => archive.mutate(false)} disabled={archive.isPending}>
                Restore
              </Button>
            ) : (
              <Button variant="secondary" onClick={() => archive.mutate(true)} disabled={archive.isPending}>
                Archive
              </Button>
            )}
            <span className="text-slate-600 dark:text-slate-400">Archiving hides the run and keeps everything; you can restore it.</span>
          </div>
          <DeleteRun runId={runId} />
        </section>
      </div>
    </Card>
  );
}

function DeleteRun({ runId }: { runId: string }) {
  const client = useQueryClient();
  const report = useReportError();
  const [open, setOpen] = useState(false);
  const [typed, setTyped] = useState("");
  const remove = useMutation({
    mutationFn: () => unwrap(api.POST("/api/runs/{run_id}/delete", { params: { path: { run_id: runId } }, body: { confirm: typed } })),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["history"] });
      void client.invalidateQueries({ queryKey: ["runs"] });
      window.location.hash = routeHref("/runs");
    },
    onError: report,
  });
  if (!open) {
    return (
      <Button variant="danger" onClick={() => setOpen(true)}>
        Delete permanently…
      </Button>
    );
  }
  return (
    <div role="group" aria-label="Confirm permanent delete" className="space-y-2 rounded-md border border-rose-300 p-3 dark:border-rose-800">
      <p>
        This permanently deletes the run's adapter, checkpoints, evaluations (with your review judgements), exports inside
        the run folder and job logs. It can't be undone. Type <strong className="font-mono">{runId}</strong> to confirm.
      </p>
      <Field label="Run ID" htmlFor="delete-confirm">
        <input id="delete-confirm" className={`${inputClass} md:w-96`} value={typed} onChange={(e) => setTyped(e.target.value)} autoComplete="off" />
      </Field>
      <div className="flex gap-2">
        <Button variant="danger" onClick={() => remove.mutate()} disabled={typed !== runId || remove.isPending}>
          Delete this run
        </Button>
        <Button
          variant="secondary"
          onClick={() => {
            setOpen(false);
            setTyped("");
          }}
        >
          Keep it
        </Button>
      </div>
    </div>
  );
}
