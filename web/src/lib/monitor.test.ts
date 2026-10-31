import { afterEach, describe, expect, it, vi } from "vitest";

import { TOKEN_KEY } from "../api/client";
import { followJob, type SseEvent } from "../api/sse";
import { chartRows } from "../components/LossChart";
import recorded from "../test/train-events.json";
import { emptyMonitor } from "../test/monitorFixture";
import { mockServer } from "../test/mockServer";
import { applyEvent, bestOf, etaText, type Monitor } from "./monitor";

// A real tiny-model CPU training job's events.jsonl (metrics only), recorded in Phase 12.
const EVENTS = recorded as Record<string, unknown>[];

function sseEvent(data: Record<string, unknown>): SseEvent {
  return { event: String(data.type), data };
}

describe("applyEvent on a recorded training stream", () => {
  const state = EVENTS.reduce<Monitor>((s, e) => applyEvent(s, sseEvent(e), "job_1"), emptyMonitor());
  const train = EVENTS.filter((e) => e.type === "train");
  const evals = train.filter((e) => e.eval_loss != null);

  it("collects every loss point and marks the lowest validation loss", () => {
    expect(state.points).toHaveLength(train.length);
    expect(evals.length).toBeGreaterThan(1);
    const lowest = Math.min(...evals.map((e) => e.eval_loss as number));
    expect(state.best?.eval_loss).toBe(lowest);
    expect(state.best?.step).toBe(evals.find((e) => e.eval_loss === lowest)?.step);
  });

  it("tracks step, total, memory and the final status", () => {
    const last = EVENTS.filter((e) => e.type === "progress").at(-1);
    expect(state.step).toBe(last?.done);
    expect(state.max_steps).toBe(last?.total);
    expect(state.memory.peak_gb).toBe(Math.max(...train.map((e) => e.memory_gb as number)));
    expect(state.run.status).toBe("succeeded");
    expect(state.can_cancel).toBe(false);
    expect(state.eta_s).toBeNull();
    expect(state.last_seq).toBe(EVENTS.at(-1)?.seq);
  });

  it("logs status changes and losses without any record text", () => {
    const messages = state.log.map((l) => l.message);
    expect(messages[0]).toBe("Job job_1 is queued.");
    expect(messages.at(-1)).toBe("Job job_1 is succeeded.");
    expect(messages.filter((m) => m.includes("validation loss"))).toHaveLength(evals.length);
    const warnings = EVENTS.filter((e) => e.type === "log").map((e) => e.message);
    for (const m of messages) {
      expect(/^Job job_1 is \w+\.$/.test(m) || /^Step \d+\/\d+: (train|validation) loss [\d.e-]+$/.test(m) || warnings.includes(m)).toBe(true);
    }
  });

  it("merges a step's train and validation loss into one chart row", () => {
    const rows = chartRows(state.points);
    expect(new Set(rows.map((r) => r.step)).size).toBe(rows.length);
    expect(rows.filter((r) => r.eval_loss != null)).toHaveLength(evals.length);
  });
});

describe("applyEvent rules", () => {
  it("replaces points after the checkpoint when a resumed attempt reports", () => {
    const train = (step: number, loss: number) => sseEvent({ type: "train", seq: step, step, max_steps: 8, loss, eval_loss: null });
    let state = [train(1, 3), train(2, 2.5), train(3, 2.4)].reduce((s, e) => applyEvent(s, e, "job_1"), emptyMonitor());
    state = applyEvent(state, train(3, 2.2), "job_2");
    expect(state.points.map((p) => [p.step, p.loss, p.job_id])).toEqual([
      [1, 3, "job_1"],
      [2, 2.5, "job_1"],
      [3, 2.2, "job_2"],
    ]);
  });

  it("keeps the earliest of equal validation losses as best", () => {
    const points = [3, 5].map((step) => ({ step, loss: null, eval_loss: 1.5, learning_rate: null, job_id: "j" }));
    expect(bestOf(points)).toEqual({ step: 3, eval_loss: 1.5, final: false });
  });

  it("collects warnings and failure messages", () => {
    let state = applyEvent(emptyMonitor(), sseEvent({ type: "log", seq: 1, level: "warning", message: "No keep-awake." }), "job_1");
    state = applyEvent(state, sseEvent({ type: "status", seq: 2, status: "failed", error: { message: "Out of memory." } }), "job_1");
    expect(state.warnings).toEqual(["No keep-awake."]);
    expect(state.log.at(-1)).toMatchObject({ level: "error", message: "Job job_1 is failed. Out of memory." });
    expect(state.job?.status).toBe("failed");
  });

  it("counts the time left down from the last report", () => {
    const at = "2026-10-30T10:00:00Z";
    expect(etaText({ eta_s: 130, eta_at: at }, Date.parse(at) + 5_000)).toBe("about 2 min 5 s left");
    expect(etaText({ eta_s: 3, eta_at: at }, Date.parse(at) + 9_000)).toBe("finishing");
    expect(etaText({ eta_s: null, eta_at: null })).toBeNull();
  });
});

describe("followJob", () => {
  afterEach(() => vi.unstubAllGlobals());

  const block = (event: Record<string, unknown>, id?: number) =>
    `event: ${String(event.type)}\n${id ? `id: ${id}\n` : ""}data: ${JSON.stringify(event)}\n\n`;

  it("reconnects from the last event ID, skips repeats, and stops at the final status", async () => {
    window.sessionStorage.setItem(TOKEN_KEY, "t0ken");
    const streams = [
      // first connection drops after two events
      block({ type: "progress", seq: 4, done: 1, total: 3 }) + block({ type: "progress", seq: 5, done: 2, total: 3 }, 120),
      // the server resends from 120, including one event already seen
      block({ type: "progress", seq: 5, done: 2, total: 3 }) +
        block({ type: "progress", seq: 6, done: 3, total: 3 }) +
        block({ type: "status", seq: 7, status: "succeeded" }, 200),
    ];
    const lastIds: (string | null)[] = [];
    mockServer({
      "GET /api/jobs/job_1/events": (_url: URL, _body: string, headers: Headers) => {
        lastIds.push(headers.get("Last-Event-ID"));
        return { sse: streams.shift() ?? "" };
      },
      "GET /api/jobs/job_1": { id: "job_1", status: "running" },
    });
    const seen: number[] = [];
    const connections: string[] = [];
    await followJob("job_1", {
      from: 100,
      afterSeq: 3,
      signal: new AbortController().signal,
      onEvent: (e) => seen.push((e.data as { seq: number }).seq),
      onConnection: (c) => connections.push(c),
      retryDelay: () => 1,
    });
    expect(lastIds).toEqual(["100", "120"]);
    expect(seen).toEqual([4, 5, 6, 7]);
    expect(connections).toEqual(["connecting", "live", "reconnecting", "live"]);
  });

  it("gives up on a missing job instead of retrying", async () => {
    window.sessionStorage.setItem(TOKEN_KEY, "t0ken");
    const calls = mockServer({ "GET /api/jobs/nope/events": () => ({ status: 404, body: { error: { message: "No job nope." } } }) });
    await expect(followJob("nope", { onEvent: () => {}, signal: new AbortController().signal, retryDelay: () => 1 })).rejects.toThrow(
      "No job nope.",
    );
    expect(calls).toHaveLength(1);
  });
});
