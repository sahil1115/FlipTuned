import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { TOKEN_KEY } from "../api/client";
import { applyEvent } from "../lib/monitor";
import { emptyMonitor } from "../test/monitorFixture";
import { mockServer } from "../test/mockServer";
import { Dashboard } from "./Dashboard";
import { RunMonitor } from "./RunMonitor";

function wrap(node: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{node}</QueryClientProvider>);
}

const block = (event: Record<string, unknown>, id?: number) =>
  `event: ${String(event.type)}\n${id ? `id: ${id}\n` : ""}data: ${JSON.stringify(event)}\n\n`;

const train = (seq: number, step: number, fields: Record<string, unknown>) => ({
  type: "train",
  seq,
  time: "2026-10-30T10:00:05+00:00",
  step,
  max_steps: 8,
  epoch: step / 8,
  loss: null,
  eval_loss: null,
  learning_rate: 1e-4,
  eta_s: 30,
  memory_gb: 0.6,
  ...fields,
});

const RUNS = [emptyMonitor().run];

beforeEach(() => {
  window.sessionStorage.setItem(TOKEN_KEY, "t0ken");
  window.location.hash = "#/monitor?run=run_1";
});
afterEach(() => vi.unstubAllGlobals());

describe("Run monitor", () => {
  it("shows live loss from the stream, then the cancelled state with Resume", async () => {
    const running = { ...emptyMonitor(), offset: 50, last_seq: 2, warnings: ["Dataset size: small"] };
    const events = [
      train(3, 1, { loss: 2.5 }),
      train(4, 2, { eval_loss: 2.2 }),
      { type: "progress", seq: 5, done: 2, total: 8 },
      { type: "status", seq: 6, status: "cancelled", error: null },
    ];
    // What the server answers afterwards: the same events folded in, and now resumable.
    const cancelled = {
      ...events.reduce((state, e) => applyEvent(state, { event: e.type, data: e }, "job_1"), running),
      can_resume: true,
    };
    let monitorCalls = 0;
    const calls = mockServer({
      "GET /api/runs": RUNS,
      "GET /api/runs/run_1/monitor": () => ({ body: monitorCalls++ === 0 ? running : cancelled }),
      "GET /api/jobs/job_1/events": () => ({
        sse: events.map((e, i) => block(e, i === events.length - 1 ? 400 : undefined)).join(""),
      }),
      "POST /api/runs/run_1/resume": { run_id: "run_1", job: { id: "job_2" } },
    });
    wrap(<RunMonitor />);

    const table = await screen.findByRole("table", { name: "Loss values" });
    await waitFor(() => expect(within(table).getAllByRole("row")).toHaveLength(3)); // header + 2 steps
    expect(within(table).getByText("2.2 (best)")).toBeInTheDocument();
    expect(screen.getByText("Dataset size: small")).toBeInTheDocument();
    const log = screen.getByRole("log", { name: "Event log" });
    expect(within(log).getByText(/Step 2\/8: validation loss 2.2/)).toBeInTheDocument();

    await screen.findByText(/You cancelled training/);
    expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled();
    const resume = await screen.findByRole("button", { name: "Resume" });
    await waitFor(() => expect(resume).toBeEnabled());
    expect(calls.find((c) => c.url === "/api/jobs/job_1/events")?.headers.get("Last-Event-ID")).toBe("50");

    await userEvent.click(resume);
    await waitFor(() => expect(calls.some((c) => c.method === "POST" && c.url === "/api/runs/run_1/resume")).toBe(true));
  });

  it("cancels the running job", async () => {
    const calls = mockServer({
      "GET /api/runs": RUNS,
      "GET /api/runs/run_1/monitor": { ...emptyMonitor(), step: 3, max_steps: 8, eta_s: 30, eta_at: new Date().toISOString() },
      "GET /api/jobs/job_1/events": () => ({ sse: ": keepalive\n\n" }),
      "GET /api/jobs/job_1": { id: "job_1", status: "running" },
      "POST /api/jobs/job_1/cancel": { job_id: "job_1", status: "cancelled", forced: false, already_finished: false },
    });
    wrap(<RunMonitor />);
    expect(await screen.findByText(/Training in the background/)).toBeInTheDocument();
    expect(screen.getByText("3 / 8")).toBeInTheDocument();
    expect(screen.getByText(/s left$/)).toBeInTheDocument();
    expect(screen.getByText("Peak memory (RAM)")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() => {
      const cancel = calls.find((c) => c.method === "POST" && c.url === "/api/jobs/job_1/cancel");
      expect(JSON.parse(cancel?.body ?? "{}")).toEqual({ timeout: 60 });
    });
  });

  it("explains the model lock and keeps Resume disabled", async () => {
    mockServer({
      "GET /api/runs": RUNS,
      "GET /api/runs/run_1/monitor": {
        ...emptyMonitor("job_1", "interrupted"),
        can_cancel: false,
        can_resume: false,
        resume_note: "Another job is using the model: job_9 (train, running). Resume when it has finished.",
        lock_holder: {
          job_id: "job_9",
          type: "train",
          status: "running",
          run_id: "run_9",
          message: "Another job is using the model: job_9 (train, running).",
        },
      },
    });
    wrap(<RunMonitor />);
    expect(await screen.findByText(/Another job is using the model: job_9/)).toBeInTheDocument();
    expect(screen.getByText(/ended unexpectedly/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Resume" })).toBeDisabled();
    expect(screen.getByRole("link", { name: "Watch it" })).toHaveAttribute("href", "#/monitor?run=run_9");
  });

  it("shows each status in plain words", async () => {
    for (const status of ["queued", "succeeded", "failed"]) {
      mockServer({
        "GET /api/runs": RUNS,
        "GET /api/runs/run_1/monitor": {
          ...emptyMonitor("job_1", status),
          can_cancel: status === "queued",
          error: status === "failed" ? { message: "Out of memory.", hint: "Use a smaller model." } : null,
          result: status === "succeeded" ? { best_step: 4, best_eval_loss: 1.9, steps_done: 8, total_steps: 8, stopped_early: false } : null,
        },
        "GET /api/jobs/job_1/events": () => ({ sse: ": keepalive\n\n" }),
        "GET /api/jobs/job_1": { id: "job_1", status },
      });
      const { unmount } = wrap(<RunMonitor />);
      const text = { queued: /Waiting for the background worker/, succeeded: /Training finished/, failed: /stopped with an error/ }[status];
      expect(await screen.findByText(text as RegExp)).toBeInTheDocument();
      if (status === "failed") expect(screen.getByText("What to do: Use a smaller model.")).toBeInTheDocument();
      if (status === "succeeded") expect(screen.getByRole("region", { name: "Result" })).toBeInTheDocument();
      unmount();
      vi.unstubAllGlobals();
    }
  });

  it("points to Preflight when there is no run", async () => {
    window.location.hash = "#/monitor";
    mockServer({ "GET /api/runs": [] });
    wrap(<RunMonitor />);
    expect(await screen.findByText(/No training runs yet/)).toBeInTheDocument();
  });
});

describe("Dashboard active job", () => {
  it("links a running training job to its monitor", async () => {
    const job = { ...emptyMonitor().job, status: "running", progress: 0.25 };
    mockServer({
      "GET /api/jobs": { jobs: [job], model_lock_holder: "job_1", reconciled: [] },
      "GET /api/runs": RUNS,
      "GET /api/doctor": { machine: { os_version: "x", ram_gb: 16 }, backend: { selected: null, suggestions: [] } },
      "GET /api/status": { disk_use_mb: {}, app_home: "/h" },
      "GET /api/sources": [],
      "GET /api/datasets": [],
      "GET /api/models": { models: [], estimate_note: "" },
      "GET /api/configs": [],
    });
    wrap(<Dashboard />);
    const card = await screen.findByRole("region", { name: "Active job" });
    expect(await within(card).findByRole("link", { name: "Open the run monitor" })).toHaveAttribute("href", "#/monitor?run=run_1");
    expect(within(card).getByText("25%")).toBeInTheDocument();
  });
});
