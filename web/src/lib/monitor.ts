// Run monitor state: the server's snapshot (GET /api/runs/{id}/monitor) plus live job events.
// applyEvent follows the same rules as the server's fold (services/monitor.py), so a snapshot taken
// after the events and the snapshot taken before them plus applyEvent agree.
import type { Schemas } from "../api/client";
import type { SseEvent } from "../api/sse";

export type Monitor = Schemas["RunMonitor"];
export type LossPoint = Schemas["LossPoint"];

const LOG_LINES = 300;
const FINAL = new Set(["succeeded", "failed", "cancelled", "interrupted"]);

interface EventData {
  seq?: number;
  time?: string;
  [key: string]: unknown;
}

function num(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** A loss for display: 6 significant digits (the server's log uses the same). */
export function fmt(value: number): string {
  return Number(value.toPrecision(6)).toString();
}

/** The lowest validation loss so far (the earliest on a tie), as training picks it. */
export function bestOf(points: LossPoint[]): Schemas["BestCheckpoint"] | null {
  let best: LossPoint | null = null;
  for (const p of points) {
    if (p.eval_loss != null && (best === null || p.eval_loss < (best.eval_loss as number))) best = p;
  }
  return best ? { step: best.step, eval_loss: best.eval_loss as number, final: false } : null;
}

export function applyEvent(state: Monitor, event: SseEvent, jobId: string): Monitor {
  const data = event.data as EventData;
  const line = (level: string, message: string): Monitor["log"] =>
    [...state.log, { seq: data.seq ?? 0, time: data.time ?? "", job_id: jobId, level, message }].slice(-LOG_LINES);
  const next: Monitor = { ...state, last_seq: data.seq ?? state.last_seq };

  if (event.event === "progress") {
    next.step = num(data.done);
    next.max_steps = num(data.total) || state.max_steps;
  } else if (event.event === "train") {
    const step = Number(data.step);
    let points = state.points;
    const last = points[points.length - 1];
    // A resumed attempt restarts from its checkpoint: drop what came after it.
    if (last && last.job_id !== jobId) points = points.filter((p) => p.step < step);
    const point: LossPoint = {
      step,
      loss: num(data.loss),
      eval_loss: num(data.eval_loss),
      learning_rate: num(data.learning_rate),
      job_id: jobId,
    };
    next.points = [...points, point];
    next.best = bestOf(next.points);
    next.step = step;
    next.max_steps = num(data.max_steps) || state.max_steps;
    if (num(data.epoch) !== null) next.epoch = num(data.epoch);
    if (num(data.eta_s) !== null) {
      next.eta_s = num(data.eta_s);
      next.eta_at = data.time ?? null;
    }
    const memory = num(data.memory_gb);
    if (memory !== null) next.memory = { ...state.memory, peak_gb: Math.max(state.memory.peak_gb ?? 0, memory) };
    const total = next.max_steps ? `/${next.max_steps}` : "";
    const messages: string[] = [];
    if (point.eval_loss !== null) messages.push(`Step ${step}${total}: validation loss ${fmt(point.eval_loss as number)}`);
    if (point.loss !== null) messages.push(`Step ${step}${total}: train loss ${fmt(point.loss as number)}`);
    const added = messages.map((message) => ({ seq: data.seq ?? 0, time: data.time ?? "", job_id: jobId, level: "info", message }));
    next.log = [...state.log, ...added].slice(-LOG_LINES);
  } else if (event.event === "log") {
    const level = String(data.level ?? "info");
    const message = String(data.message ?? "");
    if (level === "warning") next.warnings = [...state.warnings, message];
    next.log = line(level, message);
  } else if (event.event === "status") {
    const status = String(data.status);
    const error = (data.error ?? null) as { message?: string } | null;
    let text = `Job ${jobId} is ${status}.`;
    if (error?.message) text += ` ${error.message}`;
    if (data.forced) text += " (force-stopped)";
    next.log = line(status === "failed" ? "error" : "info", text);
    if (state.job && state.job.id === jobId) next.job = { ...state.job, status };
    next.run = { ...state.run, status };
    next.can_cancel = !FINAL.has(status);
    if (FINAL.has(status)) next.eta_s = null;
  }
  return next;
}

/** "about 2 min 5 s left", counting down from the time of the last report. */
export function etaText(state: Pick<Monitor, "eta_s" | "eta_at">, now: number = Date.now()): string | null {
  if (state.eta_s == null) return null;
  const reported = state.eta_at ? Date.parse(state.eta_at) : NaN;
  const elapsed = Number.isNaN(reported) ? 0 : Math.max(0, (now - reported) / 1000);
  const left = Math.max(0, Math.round(state.eta_s - elapsed));
  if (left === 0) return "finishing";
  const min = Math.floor(left / 60);
  const s = left % 60;
  return `about ${min ? `${min} min ` : ""}${s} s left`;
}

/** What each status means for the person watching, and what they can do next. */
export const STATUS_HELP: Record<string, string> = {
  queued: "Waiting for the background worker to start.",
  running: "Training in the background. You can leave this page or close the browser; it keeps going.",
  succeeded: "Training finished. The best checkpoint is saved as the run's adapter.",
  failed: "Training stopped with an error (below). Checkpoints are kept; Resume continues from the last one.",
  cancelled: "You cancelled training. Checkpoints are kept; Resume continues from the last one.",
  interrupted:
    "The training process ended unexpectedly (the computer slept or restarted, or the process was killed). Resume continues from the last checkpoint.",
};
