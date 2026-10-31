// Follow a job: live progress from its event stream, then its result or error.
import { useEffect, useState } from "react";

import { api, type Schemas, unwrap } from "../api/client";
import { streamJobEvents } from "../api/sse";

export interface JobState {
  status: string;
  done: number;
  total: number;
  message: string | null;
  outcome: Schemas["JobOutcome"] | null;
  failure: unknown;
}

const INITIAL: JobState = { status: "queued", done: 0, total: 0, message: null, outcome: null, failure: null };
const FINAL = new Set(["succeeded", "failed", "cancelled", "interrupted"]);

export function isFinal(status: string): boolean {
  return FINAL.has(status);
}

export function useJob(jobId: string | null): JobState {
  const [state, setState] = useState<JobState>(INITIAL);

  useEffect(() => {
    if (!jobId) return;
    const controller = new AbortController();
    setState(INITIAL);
    (async () => {
      try {
        await streamJobEvents(
          jobId,
          (event) => {
            const data = event.data as Record<string, unknown>;
            if (event.event === "progress") {
              setState((s) => ({
                ...s,
                done: Number(data.done ?? 0),
                total: Number(data.total ?? 0),
                message: (data.message as string | null) ?? s.message,
              }));
            } else if (event.event === "status") {
              setState((s) => ({ ...s, status: String(data.status) }));
            }
          },
          controller.signal,
        );
        const outcome = await unwrap(
          api.GET("/api/jobs/{job_id}/outcome", { params: { path: { job_id: jobId } } }),
        );
        setState((s) => ({ ...s, status: outcome.job.status, outcome }));
      } catch (error) {
        if ((error as Error).name !== "AbortError") setState((s) => ({ ...s, failure: error }));
      }
    })();
    return () => controller.abort();
  }, [jobId]);

  return state;
}
