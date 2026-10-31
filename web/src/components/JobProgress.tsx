// Live progress of a job. The job keeps running if you leave the page; it's listed on the Dashboard.
import { ApiError } from "../api/client";
import { useJob, type JobState } from "../lib/useJob";
import { Badge, ErrorPanel, statusTone } from "./ui";

export function JobProgress({
  jobId,
  label,
  children,
}: {
  jobId: string;
  label: string;
  children?: (state: JobState) => React.ReactNode;
}) {
  const state = useJob(jobId);
  const percent = state.total ? Math.round((state.done / state.total) * 100) : null;
  return (
    <div className="space-y-2" aria-live="polite">
      <div className="flex items-center gap-2 text-sm">
        <span className="font-medium">{label}</span>
        <Badge tone={statusTone(state.status)}>{state.status}</Badge>
        <span className="text-xs text-slate-600 dark:text-slate-400">job {jobId}</span>
      </div>
      {state.status === "running" || state.status === "queued" ? (
        <div className="space-y-1">
          <progress
            className="h-2 w-full"
            max={100}
            value={percent ?? undefined}
            aria-label={`${label} progress`}
          />
          {state.message && <p className="text-xs text-slate-600 dark:text-slate-400">{state.message}</p>}
        </div>
      ) : null}
      {state.failure ? <ErrorPanel error={state.failure} /> : null}
      {state.outcome?.error ? <ErrorPanel error={jobError(state.outcome.error)} /> : null}
      {children?.(state)}
    </div>
  );
}

/** A failed job's error.json (message / hint / details) as an error the panel can show. */
export function jobError(error: Record<string, unknown>): ApiError {
  return new ApiError(
    0,
    String(error.message ?? "The job failed."),
    (error.hint as string | null) ?? null,
    (error.details as string | null) ?? null,
  );
}
