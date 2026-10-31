// Dashboard: this machine and its backend, the active job, data sources, datasets, models, training
// configs, runs, jobs, disk use.
import { useQuery } from "@tanstack/react-query";

import { api, unwrap } from "../api/client";
import { Badge, Card, CountList, ErrorPanel, Spinner, Stat, statusTone } from "../components/ui";
import { monitorHref, useDatasets, useHistory, useJobs, useSources } from "../lib/queries";
import { fmt } from "../lib/monitor";
import { routeHref } from "../lib/route";
import { useModels } from "./Models";
import { useConfigs } from "./TrainingConfig";

const LINK = "text-sky-800 underline dark:text-sky-300";

export function Dashboard() {
  const doctor = useQuery({ queryKey: ["doctor"], queryFn: () => unwrap(api.GET("/api/doctor")), staleTime: 60_000 });
  const status = useQuery({
    queryKey: ["status"],
    queryFn: () => unwrap(api.GET("/api/status")),
    refetchInterval: 5_000,
  });
  const sources = useSources();
  const datasets = useDatasets();
  const jobs = useJobs(8);
  const models = useModels();
  const configs = useConfigs();
  const runs = useHistory({ limit: 5 });
  const active = jobs.data?.jobs.find((job) => job.status === "running" || job.status === "queued");

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Card title="This machine">
        {doctor.isPending ? (
          <Spinner label="Checking this machine…" />
        ) : doctor.error ? (
          <ErrorPanel error={doctor.error} />
        ) : (
          <div className="space-y-3 text-sm">
            <dl className="grid grid-cols-2 gap-3">
              <Stat label="System" value={doctor.data.machine.os_version} />
              <Stat label="Memory" value={`${doctor.data.machine.ram_gb} GB`} />
              <Stat
                label="Accelerator"
                value={
                  (doctor.data.machine.nvidia_gpus ?? []).length
                    ? (doctor.data.machine.nvidia_gpus ?? []).map((g) => g.name).join(", ")
                    : doctor.data.machine.apple_chip ?? "none (CPU)"
                }
              />
              <Stat
                label="Backend"
                value={
                  doctor.data.backend.selected
                    ? `Auto → ${doctor.data.backend.selected.backend} on ${doctor.data.backend.selected.device}`
                    : "none available"
                }
              />
            </dl>
            {doctor.data.backend.selected && (
              <p className="text-slate-600 dark:text-slate-400">Why: {doctor.data.backend.selected.reason}</p>
            )}
            {(doctor.data.backend.suggestions ?? []).map((s) => (
              <p key={s} className="text-amber-800 dark:text-amber-300">
                {s}
              </p>
            ))}
          </div>
        )}
      </Card>

      <Card title="Active job">
        {jobs.error ? (
          <ErrorPanel error={jobs.error} />
        ) : !jobs.data ? (
          <Spinner />
        ) : !active ? (
          <p className="text-sm">No job is running.</p>
        ) : (
          <div className="space-y-2 text-sm">
            <p className="flex flex-wrap items-center gap-2">
              <Badge tone={statusTone(active.status)}>{active.status}</Badge>
              <span className="font-medium">{active.type}</span>
              {active.progress != null && active.status === "running" && <span>{Math.round(active.progress * 100)}%</span>}
              <span className="text-xs text-slate-600 dark:text-slate-400">job {active.id}</span>
            </p>
            {active.holds_model_lock && <p>It is using the model; other model jobs wait until it finishes.</p>}
            {active.type === "train" && active.run_id ? (
              <a className={LINK} href={monitorHref(active.run_id)}>
                Open the run monitor
              </a>
            ) : (
              <p className="text-slate-600 dark:text-slate-400">It runs in the background; you can leave this page.</p>
            )}
          </div>
        )}
      </Card>

      <Card title="Training runs">
        {runs.error ? (
          <ErrorPanel error={runs.error} />
        ) : !runs.data ? (
          <Spinner />
        ) : runs.data.length === 0 ? (
          <p className="text-sm">
            No training run yet. Start one from{" "}
            <a className={LINK} href="#/preflight">
              Preflight
            </a>
            .
          </p>
        ) : (
          <div className="space-y-2">
          <ul className="space-y-1 text-sm">
            {runs.data.map((r) => (
              <li key={r.id} className="flex flex-wrap items-center gap-2">
                <Badge tone={statusTone(r.status)}>{r.status}</Badge>
                <a className={LINK} href={monitorHref(r.id)}>
                  {r.id}
                </a>
                {r.verdict && <Badge tone={statusTone(r.verdict.verdict === "accepted" ? "succeeded" : r.verdict.verdict === "rejected" ? "failed" : "warn")}>verdict: {r.verdict.verdict}</Badge>}
                <span className="text-xs text-slate-600 dark:text-slate-400">
                  {r.model_id} · {r.backend} on {r.device}
                  {r.best_eval_loss != null ? ` · best validation loss ${fmt(r.best_eval_loss)}` : ""}
                  {r.exports ? ` · ${r.exports} export${r.exports === 1 ? "" : "s"}` : ""}
                </span>
              </li>
            ))}
          </ul>
            <p className="text-sm">
              <a className={LINK} href="#/runs">
                All runs
              </a>
            </p>
          </div>
        )}
      </Card>

      <Card title="Data">
        {sources.error ? (
          <ErrorPanel error={sources.error} />
        ) : datasets.error ? (
          <ErrorPanel error={datasets.error} />
        ) : (
          <div className="space-y-3 text-sm">
            <dl className="grid grid-cols-2 gap-3">
              <Stat label="Data sources" value={sources.data?.length ?? "…"} />
              <Stat label="Datasets" value={datasets.data?.length ?? "…"} />
            </dl>
            {datasets.data?.length ? (
              <ul className="space-y-1">
                {datasets.data.slice(0, 5).map((d) => (
                  <li key={d.id} className="flex items-center gap-2">
                    <a className="text-sky-800 underline dark:text-sky-300" href="#/dataset">
                      {d.name}
                    </a>
                    <span className="text-xs text-slate-600 dark:text-slate-400">{d.id}</span>
                    <Badge tone={statusTone(d.status)}>{d.status}</Badge>
                  </li>
                ))}
              </ul>
            ) : (
              <p>
                No dataset yet. Start with <a className="text-sky-800 underline dark:text-sky-300" href="#/sources">Data sources</a>.
              </p>
            )}
          </div>
        )}
      </Card>

      <Card title="Models">
        {models.error ? (
          <ErrorPanel error={models.error} />
        ) : !models.data ? (
          <Spinner />
        ) : (
          <div className="space-y-3 text-sm">
            <dl className="grid grid-cols-3 gap-3">
              <Stat label="In the catalog" value={models.data.models.length} />
              <Stat label="Downloaded" value={models.data.models.filter((m) => m.cached).length} />
              <Stat label="Fit here (estimate)" value={models.data.models.filter((m) => m.compatibility?.label === "fits").length} />
            </dl>
            <p>
              <a className={LINK} href="#/models">
                See the models
              </a>{" "}
              with memory estimates for this machine, and download one.
            </p>
          </div>
        )}
      </Card>

      <Card title="Training configs">
        {configs.error ? (
          <ErrorPanel error={configs.error} />
        ) : !configs.data ? (
          <Spinner />
        ) : configs.data.length === 0 ? (
          <p className="text-sm">
            No training config yet.{" "}
            <a className={LINK} href="#/config">
              Make one
            </a>
            .
          </p>
        ) : (
          <ul className="space-y-1 text-sm">
            {configs.data.slice(0, 5).map((c) => (
              <li key={c.name} className="flex flex-wrap items-center gap-2">
                <span className="font-medium">
                  {c.name} v{c.version}
                </span>
                <span className="text-xs text-slate-600 dark:text-slate-400">
                  {c.model} · {c.preset}
                </span>
                <a className={LINK} href={routeHref("/preflight", { config: c.name })}>
                  Preflight
                </a>
              </li>
            ))}
          </ul>
        )}
      </Card>

      <Card title="Jobs">
        {jobs.error ? (
          <ErrorPanel error={jobs.error} />
        ) : !jobs.data ? (
          <Spinner />
        ) : jobs.data.jobs.length === 0 ? (
          <p className="text-sm">No jobs yet.</p>
        ) : (
          <div className="space-y-2 text-sm">
            <p>Model in use: {jobs.data.model_lock_holder ?? "none"}</p>
            <ul className="space-y-1">
              {jobs.data.jobs.map((job) => (
                <li key={job.id} className="flex flex-wrap items-center gap-2">
                  <Badge tone={statusTone(job.status)}>{job.status}</Badge>
                  <span className="font-medium">{job.type}</span>
                  {job.progress !== null && job.status === "running" && (
                    <span>{Math.round((job.progress ?? 0) * 100)}%</span>
                  )}
                  <span className="text-xs text-slate-600 dark:text-slate-400">{job.created_at}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </Card>

      <Card title="Disk use (MB)">
        {status.error ? (
          <ErrorPanel error={status.error} />
        ) : !status.data ? (
          <Spinner />
        ) : (
          <div className="space-y-2 text-sm">
            <CountList counts={Object.fromEntries(Object.entries(status.data.disk_use_mb).map(([k, v]) => [k, Math.round(v)]))} />
            <p className="text-slate-600 dark:text-slate-400">App home: {status.data.app_home}</p>
          </div>
        )}
      </Card>
    </div>
  );
}
