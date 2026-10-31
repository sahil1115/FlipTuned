// Server-sent events over fetch (the token is a header, which EventSource can't send).
import { api, ApiError, rawFetch, unwrap } from "./client";

export interface SseEvent {
  event: string;
  data: unknown;
  id?: string;
}

/** Parse complete events out of `buffer`; returns the events and the unfinished rest. */
export function parseSse(buffer: string): { events: SseEvent[]; rest: string } {
  const normalized = buffer.replace(/\r\n/g, "\n");
  const blocks = normalized.split("\n\n");
  const rest = blocks.pop() ?? "";
  const events: SseEvent[] = [];
  for (const block of blocks) {
    let event = "message";
    let id: string | undefined;
    const data: string[] = [];
    for (const line of block.split("\n")) {
      if (!line || line.startsWith(":")) continue;
      const colon = line.indexOf(":");
      const field = colon === -1 ? line : line.slice(0, colon);
      const value = colon === -1 ? "" : line.slice(colon + 1).replace(/^ /, "");
      if (field === "event") event = value;
      else if (field === "id") id = value;
      else if (field === "data") data.push(value);
    }
    if (data.length) events.push({ event, data: JSON.parse(data.join("\n")), id });
  }
  return { events, rest };
}

export async function readSse(response: Response, onEvent: (event: SseEvent) => void): Promise<void> {
  const reader = response.body?.getReader();
  if (!reader) return;
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parsed = parseSse(buffer);
    buffer = parsed.rest;
    parsed.events.forEach(onEvent);
  }
  buffer += decoder.decode();
  parseSse(buffer + "\n\n").events.forEach(onEvent);
}

export async function streamJobEvents(
  jobId: string,
  onEvent: (event: SseEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  await followJob(jobId, { onEvent, signal });
}

const FINAL = new Set(["succeeded", "failed", "cancelled", "interrupted"]);

export type Connection = "connecting" | "live" | "reconnecting";

export interface FollowOptions {
  onEvent: (event: SseEvent) => void;
  signal: AbortSignal;
  /** Byte offset to start from (the SSE event ID), e.g. where a snapshot stopped reading. */
  from?: number;
  /** Events up to this `seq` were seen already (they are skipped if sent again). */
  afterSeq?: number;
  onConnection?: (state: Connection) => void;
  /** Wait before reconnect attempt `n` (1, 2, …), in ms. */
  retryDelay?: (attempt: number) => number;
}

/**
 * Follow a job's events until its final status, reconnecting when the stream drops (a server
 * restart, sleep, a network blip) and resuming from the last event ID. Events are passed on once
 * each, in order. A missing job or a wrong token (404 / 401) is not retried.
 */
export async function followJob(jobId: string, options: FollowOptions): Promise<void> {
  const { onEvent, signal, onConnection, retryDelay = (n) => Math.min(500 * 2 ** (n - 1), 5_000) } = options;
  let lastId = options.from;
  let seen = options.afterSeq ?? 0;
  let finished = false;
  let failures = 0;
  const pass = (event: SseEvent) => {
    if (event.id !== undefined) lastId = Number(event.id);
    const data = event.data as { seq?: number; status?: string };
    if (typeof data.seq === "number") {
      if (data.seq <= seen) return;
      seen = data.seq;
    }
    if (event.event === "status" && FINAL.has(String(data.status))) finished = true;
    onEvent(event);
  };
  onConnection?.("connecting");
  while (!finished) {
    try {
      const headers: Record<string, string> = lastId ? { "Last-Event-ID": String(lastId) } : {};
      const response = await rawFetch(`/api/jobs/${encodeURIComponent(jobId)}/events`, { signal, headers });
      onConnection?.("live");
      failures = 0;
      await readSse(response, pass);
      if (finished) break;
      // The server ends the stream at the final status; its event may land just after that.
      const job = await unwrap(api.GET("/api/jobs/{job_id}", { params: { path: { job_id: jobId } } }));
      if (FINAL.has(job.status)) {
        const rest = await rawFetch(`/api/jobs/${encodeURIComponent(jobId)}/events`, {
          signal,
          headers: lastId ? { "Last-Event-ID": String(lastId) } : {},
        });
        await readSse(rest, pass);
        if (!finished) pass({ event: "status", data: { status: job.status } });
        break;
      }
    } catch (error) {
      if ((error as Error).name === "AbortError") throw error;
      if (error instanceof ApiError && (error.status === 401 || error.status === 404)) throw error;
    }
    failures += 1;
    onConnection?.("reconnecting");
    await sleep(retryDelay(failures), signal);
  }
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) return reject(new DOMException("Aborted", "AbortError"));
    const timer = setTimeout(resolve, ms);
    signal.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(new DOMException("Aborted", "AbortError"));
      },
      { once: true },
    );
  });
}
