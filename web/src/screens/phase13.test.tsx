import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { TOKEN_KEY } from "../api/client";
import { mockServer } from "../test/mockServer";
import { Chat } from "./Chat";
import { Evaluation, keyToJudgement, Verdict } from "./Evaluation";
import { Runs } from "./Runs";
import { Settings } from "./Settings";

function wrap(node: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{node}</QueryClientProvider>);
}

const RUN = {
  id: "run_1",
  status: "succeeded",
  model_id: "tiny-qwen2",
  backend: "hf_peft",
  device: "cpu",
  precision: "fp32",
  dataset_id: "ds_1",
  config_name: "quick",
  config_version: 1,
  created_at: "2026-10-30T10:00:00+00:00",
  archived: false,
  best_eval_loss: 1.5,
  verdict: null,
  evaluations: 1,
  exports: 0,
};

const INFO = {
  id: "ev_1",
  run_id: "run_1",
  status: "ready",
  verdict: "inconclusive",
  eval_dir: "/h/runs/run_1/evaluation/ev_1",
  job_id: "job_e",
  created_at: "2026-10-30T10:00:00+00:00",
  updated_at: "2026-10-30T10:00:00+00:00",
  total: 2,
  judged: 0,
};

const SCORE = {
  evaluation_id: "ev_1",
  run_id: "run_1",
  verdict: "inconclusive" as const,
  reasons: ["Fewer than 50 test items (2)."],
  criteria: [
    {
      name: "test_items",
      stage: "inconclusive" as const,
      description: "At least 50 test items",
      value: 2,
      threshold: 50,
      met: false,
      applicable: true,
      display_value: "2",
      display_threshold: "50",
    },
    {
      name: "sign_test",
      stage: "rejected" as const,
      description: "Sign test p-value below alpha",
      value: 0.5,
      threshold: 0.05,
      met: false,
      applicable: true,
      display_value: "p = 0.5",
      display_threshold: "p = 0.05",
    },
  ],
  test_items: 2,
  judged: 2,
  review_coverage: 1,
  wins: 1,
  losses: 0,
  ties: 1,
  sign_test_p: 1,
  categories: [{ category: "python", items: 2, judged: 2, wins: 1, losses: 0, ties: 1 }],
  deterministic_pass_base: 50,
  deterministic_pass_tuned: 100,
  regression_pass_base: 90,
  regression_pass_tuned: 85,
  regressions: 1,
  prior_exposure_note: null,
  limitations: ["The test set is small."],
  eval_dir: "/h/runs/run_1/evaluation/ev_1",
  report_path: "/h/runs/run_1/evaluation/ev_1/report.md",
};

const pair = (position: number, judgement: string | null = null) => ({
  evaluation_id: "ev_1",
  item_id: `item_${position}`,
  position,
  category: "python",
  question: `Question ${position}?`,
  reference: "Reference answer.",
  answer_a: `Answer A${position}`,
  answer_b: `Answer B${position}`,
  judgement,
});

beforeEach(() => window.sessionStorage.setItem(TOKEN_KEY, "t0ken"));
afterEach(() => vi.unstubAllGlobals());

describe("Evaluation review", () => {
  it("maps the review keys", () => {
    expect(["1", "2", "3", "4", "a"].map(keyToJudgement)).toEqual(["A", "B", "tie", null, null]);
  });

  it("judges blind pairs with keys 1/2/3 and never sees which model wrote them", async () => {
    window.location.hash = "#/evaluation?run=run_1&eval=ev_1";
    const judged: { item_id: string; judgement: string }[] = [];
    const calls = mockServer({
      "GET /api/history": [RUN],
      "GET /api/runs/run_1/evaluations": [INFO],
      "GET /api/evaluations/ev_1": () => ({ body: { ...INFO, judged: judged.length } }),
      "GET /api/evaluations/ev_1/pairs": () => ({
        body: { evaluation_id: "ev_1", total: 2, judged: judged.length, pair: judged.length < 2 ? pair(judged.length + 1) : null },
      }),
      "POST /api/evaluations/ev_1/judgements": (_u: URL, body: string) => {
        judged.push(JSON.parse(body));
        return { body: { evaluation_id: "ev_1", item_id: "x", total: 2, judged: judged.length } };
      },
      "POST /api/evaluations/ev_1/score": SCORE,
    });
    wrap(<Evaluation />);
    expect(await screen.findByText("Question 1?")).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Answer A" })).toHaveTextContent("Answer A1");
    await userEvent.keyboard("1");
    expect(await screen.findByText("Question 2?")).toBeInTheDocument();
    await userEvent.keyboard("3");
    expect(await screen.findByText(/Every pair is reviewed/)).toBeInTheDocument();
    expect(judged).toEqual([
      { item_id: "item_1", judgement: "A" },
      { item_id: "item_2", judgement: "tie" },
    ]);
    // the review typing in a field doesn't count as a judgement
    expect(calls.filter((c) => c.url.endsWith("/judgements"))).toHaveLength(2);
    expect(await screen.findByRole("table", { name: "Verdict criteria" })).toBeInTheDocument();
  });

  it("ignores the keys while typing in a field", async () => {
    window.location.hash = "#/evaluation?run=run_1&eval=ev_1";
    const calls = mockServer({
      "GET /api/history": [RUN],
      "GET /api/runs/run_1/evaluations": [INFO],
      "GET /api/evaluations/ev_1": INFO,
      "GET /api/evaluations/ev_1/pairs": { evaluation_id: "ev_1", total: 2, judged: 0, pair: pair(1) },
      "POST /api/evaluations/ev_1/score": SCORE,
    });
    wrap(<Evaluation />);
    await screen.findByText("Question 1?");
    await userEvent.type(screen.getByLabelText("Only the first N test items (optional)"), "1");
    expect(calls.some((c) => c.url.endsWith("/judgements"))).toBe(false);
  });
});

describe("Verdict", () => {
  it("shows the verdict with every input, categories and limitations", () => {
    wrap(<Verdict result={SCORE} />);
    expect(screen.getByText("Verdict", { selector: "span" }).parentElement).toHaveTextContent("Verdictinconclusive");
    expect(screen.getByText("Fewer than 50 test items (2).")).toBeInTheDocument();
    const criteria = screen.getByRole("table", { name: "Verdict criteria" });
    expect(within(criteria).getAllByRole("row")).toHaveLength(3);
    expect(within(criteria).getByText("p = 0.05")).toBeInTheDocument();
    expect(screen.getByText("1 / 0 / 1")).toBeInTheDocument();
    expect(screen.getByText("50.0% → 100.0%")).toBeInTheDocument();
    expect(screen.getByText("90.0% → 85.0%")).toBeInTheDocument();
    expect(within(screen.getByRole("table", { name: "Results by category" })).getByText("python")).toBeInTheDocument();
    expect(within(screen.getByRole("list", { name: "Limitations" })).getByText("The test set is small.")).toBeInTheDocument();
    expect(screen.getByText("2", { selector: "dd" })).toBeInTheDocument(); // test-set size
  });
});

describe("Chat", () => {
  it("shows the no-retrieval banner, streams a reply and notes the idle release", async () => {
    window.location.hash = "#/chat?run=run_1";
    let sent: unknown = null;
    mockServer({
      "GET /api/history": [RUN],
      "GET /api/jobs": { jobs: [], model_lock_holder: null, reconciled: [] },
      "POST /api/chat/sessions": {
        run_id: "run_1",
        model: "tiny-qwen2",
        base: false,
        backend: "hf_peft",
        device: "cpu",
        precision: "fp32",
        idle_timeout_s: 600,
        job: { id: "job_c" },
      },
      "GET /api/jobs/job_c/events": () => ({
        sse: 'event: chat\ndata: {"seq": 3, "type": "chat", "state": "ready"}\n\n',
      }),
      "GET /api/jobs/job_c": { id: "job_c", status: "running" },
      "POST /api/chat/sessions/job_c/messages": (_u: URL, body: string) => {
        sent = JSON.parse(body);
        return { sse: 'event: delta\ndata: {"text": "Hello"}\n\nevent: delta\ndata: {"text": " there"}\n\nevent: done\ndata: {}\n\n' };
      },
    });
    wrap(<Chat />);
    expect(screen.getByRole("note")).toHaveTextContent("Model weights only — no document retrieval.");
    await userEvent.click(await screen.findByRole("button", { name: "Load the model and chat" }));
    expect(await screen.findByText(/released after 10 min without a message/)).toBeInTheDocument();
    const input = await screen.findByLabelText("Message");
    await waitFor(() => expect(input).toBeEnabled());
    await userEvent.type(input, "Hi?{Enter}");
    expect(await screen.findByText("Hello there")).toBeInTheDocument();
    expect(sent).toEqual({ turns: [{ role: "user", content: "Hi?" }] });
    expect(screen.getByRole("note")).toBeInTheDocument(); // the banner stays
  });
});

describe("Runs", () => {
  it("filters, compares two runs and deletes only after the run ID is typed", async () => {
    window.location.hash = "#/runs?run=run_1";
    const second = { ...RUN, id: "run_2", best_eval_loss: 1.4, verdict: { evaluation_id: "e", verdict: "rejected" } };
    const deleted: unknown[] = [];
    const calls = mockServer({
      "GET /api/history": (url: URL) => ({ body: url.searchParams.get("status") === "failed" ? [] : [RUN, second] }),
      "GET /api/compare-runs": {
        a: { run: RUN, best_step: 4, steps_done: 8, total_steps: 8, stopped_early: false, peak_memory_gb: 0.6, attempts: 1, train_seconds: 60 },
        b: { run: second, best_step: 6, steps_done: 8, total_steps: 8, stopped_early: false, peak_memory_gb: 0.6, attempts: 2, train_seconds: 90 },
        config_differences: [{ key: "training.learning_rate", a: 0.0001, b: 0.0002 }],
        same_dataset: true,
        note: null,
      },
      "GET /api/runs/run_1/artifacts": {
        run_id: "run_1",
        run_dir: "/h/runs/run_1",
        artifacts: [
          { path: "metadata.json", kind: "file", bytes: 2048, files: 1, downloadable: true },
          { path: "adapter", kind: "folder", bytes: 4096, files: 3, downloadable: false },
        ],
        external_exports: [],
      },
      "GET /api/exports": [],
      "POST /api/runs/run_1/delete": (_u: URL, body: string) => {
        deleted.push(JSON.parse(body));
        return { body: { run_id: "run_1", removed_folders: [], kept_exports: [] } };
      },
    });
    wrap(<Runs />);
    const table = await screen.findByRole("table", { name: "Runs" });
    expect(within(table).getByText("rejected")).toBeInTheDocument();
    await userEvent.selectOptions(screen.getByLabelText("Status"), "failed");
    expect(await screen.findByText("No runs match.")).toBeInTheDocument();
    await waitFor(() => expect(calls.some((c) => c.url === "/api/history")).toBe(true));
    await userEvent.selectOptions(screen.getByLabelText("Status"), "");

    await userEvent.click(await screen.findByLabelText("Compare run_1"));
    await userEvent.click(screen.getByLabelText("Compare run_2"));
    const diff = await screen.findByRole("table", { name: "Config differences" });
    expect(within(diff).getByText("training.learning_rate")).toBeInTheDocument();
    expect(screen.getByRole("table", { name: "Metrics and verdicts" })).toHaveTextContent("90 s");

    expect(await screen.findByRole("button", { name: "Download metadata.json" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Download adapter" })).toBeNull();
    await userEvent.click(screen.getByRole("button", { name: "Delete permanently…" }));
    const confirm = screen.getByRole("button", { name: "Delete this run" });
    expect(confirm).toBeDisabled();
    await userEvent.type(screen.getByLabelText("Run ID"), "run_");
    expect(confirm).toBeDisabled();
    await userEvent.type(screen.getByLabelText("Run ID"), "1");
    await userEvent.click(confirm);
    await waitFor(() => expect(deleted).toEqual([{ confirm: "run_1" }]));
  });
});

describe("Settings", () => {
  it("validates and saves options, and lists install suggestions", async () => {
    const saved: unknown[] = [];
    mockServer({
      "GET /api/settings": { app_home: "/h", allowed_roots: ["/data"], idle_timeout_min: 10, upload_max_mb: 50, dev_mode: false, port: 8765 },
      "PUT /api/settings": (_u: URL, body: string) => {
        saved.push(JSON.parse(body));
        return { body: { app_home: "/h", allowed_roots: ["/data"], idle_timeout_min: 5, upload_max_mb: 50, dev_mode: false, port: 9000 } };
      },
      "GET /api/doctor": {
        fliptuned_version: "0.1.0",
        machine: { os_version: "Windows 11", arch: "x86_64", python_version: "3.12", emulated: false, cpu_name: "CPU", cpu_count_logical: 8, ram_gb: 16, nvidia_gpus: [], apple_chip: null },
        packages: { torch: "2.9.0" },
        torch: { installed: true, build: "cpu", cuda_available: false },
        mlx: { installed: false },
        app_home: "/h",
        app_home_source: "flag",
        hf_cache: "/cache",
        disks: [],
        requested: {},
        backend: {
          selected: { backend: "hf_peft", device: "cpu", precision: "fp32", reason: "no GPU backend is available" },
          options: [{ backend: "hf_peft", device: "cpu", available: true, reason: "PyTorch on CPU" }],
          warnings: [],
          suggestions: ["NVIDIA GPU found but PyTorch is CPU-only → run `uv sync --extra cuda --extra web`"],
          install_command: "uv sync --extra cpu --extra web",
        },
        warnings: [],
      },
    });
    wrap(<Settings />);
    expect(await screen.findByText("/data")).toBeInTheDocument();
    expect(screen.getByText("/h")).toBeInTheDocument();
    const port = screen.getByLabelText("Port");
    await userEvent.clear(port);
    await userEvent.type(port, "80");
    expect(screen.getByText("Use a value from 1024 to 65535.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
    await userEvent.clear(port);
    await userEvent.type(port, "9000");
    const idle = screen.getByLabelText("Chat idle timeout (minutes)");
    await userEvent.clear(idle);
    await userEvent.type(idle, "5");
    await userEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(saved).toEqual([{ idle_timeout_min: 5, upload_max_mb: 50, port: 9000, dev_mode: false }]));
    expect(await screen.findByText("uv sync --extra cpu --extra web")).toBeInTheDocument();
    expect(screen.getByText(/PyTorch is CPU-only/)).toBeInTheDocument();
  });
});
