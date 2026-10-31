// Shared server queries. State comes from the registry, so leaving a page and coming back shows
// the same jobs and results.
import { useQuery } from "@tanstack/react-query";

import { api, type Schemas, unwrap } from "../api/client";
import { routeHref } from "./route";

export function useSettings() {
  return useQuery({ queryKey: ["settings"], queryFn: () => unwrap(api.GET("/api/settings")) });
}

export function useJobs(limit = 20) {
  return useQuery({
    queryKey: ["jobs", limit],
    queryFn: () => unwrap(api.GET("/api/jobs", { params: { query: { limit } } })),
    refetchInterval: 5_000,
  });
}

/** The most recent job of a type (to resume watching it after leaving the page). */
export function useLatestJob(type: string): Schemas["JobInfo"] | null {
  const jobs = useJobs(50);
  return jobs.data?.jobs.find((job) => job.type === type) ?? null;
}

export function useSources() {
  return useQuery({ queryKey: ["sources"], queryFn: () => unwrap(api.GET("/api/sources")) });
}

export function useDatasets() {
  return useQuery({ queryKey: ["datasets"], queryFn: () => unwrap(api.GET("/api/datasets")) });
}

export function useRuns(limit = 50) {
  return useQuery({
    queryKey: ["runs", limit],
    queryFn: () => unwrap(api.GET("/api/runs", { params: { query: { limit } } })),
    refetchInterval: 5_000,
  });
}

export function monitorHref(runId: string): string {
  return routeHref("/monitor", { run: runId });
}

export interface HistoryFilter {
  status?: string;
  model_id?: string;
  text?: string;
  include_archived?: boolean;
  limit?: number;
}

/** Runs with their latest verdict (the Runs screen, pickers, the dashboard). */
export function useHistory(filter: HistoryFilter = {}) {
  return useQuery({
    queryKey: ["history", filter],
    queryFn: () => unwrap(api.GET("/api/history", { params: { query: filter } })),
    refetchInterval: 10_000,
  });
}
