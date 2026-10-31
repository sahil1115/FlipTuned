import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { TOKEN_KEY } from "../api/client";
import { App } from "../App";
import { mockServer } from "../test/mockServer";
import trainSchema from "../test/train-schema.json";
import { Models } from "./Models";
import { Preflight } from "./Preflight";
import { TrainingConfig } from "./TrainingConfig";

function wrap(node: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{node}</QueryClientProvider>);
}

const CPU = { backend: "hf_peft", device: "cpu", precision: "fp32", reason: "no GPU backend is available" };
const TINY = {
  id: "tiny-qwen2",
  display_name: "Tiny Qwen2 (test)",
  params_b: 0.002,
  license: "apache-2.0",
  model_card_url: "https://huggingface.co/x/tiny",
  test_only: true,
  trainable: true,
  prefix_test_verified: "2026-10-29",
  backends_here: ["hf_peft"],
  selection: CPU,
  compatibility: { label: "fits", estimate_gb: 0.7, pool_gb: 11.8, pool: "RAM - 4 GB", is_estimate: true },
  cached: false,
  cached_by_backend: { hf_peft: false },
  download_gb: 0.01,
  notes: [],
};

const ROUTES = {
  "GET /api/settings": { app_home: "/h", allowed_roots: [], idle_timeout_min: 10, upload_max_mb: 50, dev_mode: true },
  "GET /api/datasets": [{ id: "ds_1", name: "fixtures", status: "built" }],
  "GET /api/jobs": { jobs: [], model_lock_holder: null, reconciled: [] },
  "GET /api/models": { models: [TINY], estimate_note: "Memory figures are catalog estimates." },
  "GET /api/configs/schema/train": trainSchema,
  "GET /api/configs": [],
  "GET /api/training-choices": {
    auto: CPU,
    auto_error: null,
    options: [
      { backend: "mlx", device: "apple_gpu", available: false, reason: "MLX needs a Mac with Apple Silicon", precisions: [] },
      { backend: "hf_peft", device: "cuda", available: false, reason: "no NVIDIA GPU found", precisions: [] },
      { backend: "hf_peft", device: "cpu", available: true, reason: "PyTorch on CPU", precisions: ["auto", "fp32"] },
    ],
    presets: { hf_peft: { quick: { max_epochs: 1 }, standard: {}, thorough: { max_epochs: 5 } }, mlx: {} },
  },
};

beforeEach(() => {
  window.sessionStorage.setItem(TOKEN_KEY, "t0ken");
});

afterEach(() => {
  vi.unstubAllGlobals();
  window.sessionStorage.clear();
});

describe("Training config", () => {
  it("starts in the beginner view and explains the auto-selected backend", async () => {
    mockServer(ROUTES);
    wrap(<TrainingConfig />);
    expect(await screen.findByText(/Auto → hf_peft on cpu \(fp32\)/)).toBeInTheDocument();
    expect(screen.getByText(/Why: no GPU backend is available/)).toBeInTheDocument();
    const details = screen.getByText("Advanced settings").closest("details")!;
    expect(details.open).toBe(false);
    // the epochs field shows the preset's value until the user overrides it
    const epochs = screen.getByLabelText("Max epochs");
    expect(epochs).toHaveAttribute("placeholder", "3");
    await userEvent.selectOptions(screen.getByLabelText("Preset"), "quick");
    expect(epochs).toHaveAttribute("placeholder", "1");
  });

  it("lists only the hardware this machine has, and says why the rest isn't there", async () => {
    mockServer(ROUTES);
    wrap(<TrainingConfig />);
    await screen.findByText(/Auto → hf_peft/);
    await userEvent.click(screen.getByText("Advanced settings"));
    const hardware = screen.getByLabelText("Backend and device");
    expect(within(hardware).getAllByRole("option").map((o) => o.textContent)).toEqual([
      "Auto (recommended)",
      "Transformers + PEFT on the CPU",
    ]);
    expect(screen.getByRole("list", { name: "Not available here" })).toHaveTextContent("no NVIDIA GPU found");
    await userEvent.selectOptions(hardware, "hf_peft:cpu");
    const precision = screen.getByLabelText("Precision");
    expect(within(precision).getAllByRole("option").map((o) => o.textContent)).toEqual(["Auto", "fp32"]);
    expect(screen.getByText(/Transformers \+ PEFT on the CPU \(set under Advanced settings\)/)).toBeInTheDocument();
  });

  it("validates schema fields live and sends nothing while one is wrong", async () => {
    const calls = mockServer(ROUTES);
    wrap(<TrainingConfig />);
    await screen.findByText(/Auto → hf_peft/);
    await userEvent.type(screen.getByLabelText("Name"), "cfg");
    await waitFor(() => expect(screen.getByRole("option", { name: /Tiny Qwen2/ })).toBeInTheDocument());
    await userEvent.selectOptions(screen.getByLabelText("Model"), "tiny-qwen2");
    await userEvent.click(screen.getByText("Advanced settings"));
    const lr = screen.getByLabelText("Learning rate");
    expect(lr).toHaveAttribute("placeholder", "0.0001");
    expect(screen.getByText(/How big each update step is/)).toBeInTheDocument(); // description from the schema
    await userEvent.type(lr, "2");
    expect(screen.getByText("Use a value above 0 and below 1.")).toBeInTheDocument(); // before saving
    await userEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(screen.getByText("Fix the highlighted fields first.")).toBeInTheDocument();
    expect(calls.filter((c) => c.method === "POST")).toEqual([]);
    await userEvent.click(screen.getByRole("button", { name: "Reset Learning rate" }));
    expect(screen.queryByText("Use a value above 0 and below 1.")).not.toBeInTheDocument();
  });

  it("saves only what the user changed and reports the version", async () => {
    let posted: unknown = null;
    mockServer({
      ...ROUTES,
      "POST /api/configs": (_url: URL, sent: string) => {
        posted = JSON.parse(sent);
        return { body: { id: 3, name: "cfg", version: 2, created: true } };
      },
    });
    wrap(<TrainingConfig />);
    await screen.findByText(/Auto → hf_peft/);
    await userEvent.type(screen.getByLabelText("Name"), "cfg");
    await waitFor(() => expect(screen.getByRole("option", { name: /Tiny Qwen2/ })).toBeInTheDocument());
    await userEvent.selectOptions(screen.getByLabelText("Model"), "tiny-qwen2");
    await userEvent.selectOptions(screen.getByLabelText("Preset"), "quick");
    await userEvent.type(screen.getByLabelText("Max epochs"), "2");
    await userEvent.click(screen.getByText("Advanced settings"));
    await userEvent.selectOptions(screen.getByLabelText("Gradient checkpointing"), "false");
    await userEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(await screen.findByText("Saved cfg v2.")).toBeInTheDocument();
    expect(posted).toEqual({
      config: {
        name: "cfg",
        model: { catalog_id: "tiny-qwen2" },
        preset: "quick",
        training: { max_epochs: 2, gradient_checkpointing: false },
      },
    });
  });

  it("duplicates a saved config under a new name", async () => {
    let body: unknown = null;
    mockServer({
      ...ROUTES,
      "GET /api/configs": [{ name: "base", version: 2, versions: 2, model: "tiny-qwen2", backend: "auto", preset: "quick", created_at: "t" }],
      "POST /api/configs/base/duplicate": (_url: URL, sent: string) => {
        body = JSON.parse(sent);
        return { body: { id: 9, name: "base-2", version: 1, created: true } };
      },
    });
    wrap(<TrainingConfig />);
    await userEvent.click(await screen.findByRole("button", { name: "Duplicate base" }));
    const name = screen.getByLabelText("Name of the copy of base");
    await userEvent.clear(name);
    await userEvent.type(name, "base-2");
    await userEvent.click(screen.getByRole("button", { name: "Make the copy" }));
    expect(await screen.findByText("Duplicated base as base-2 v1.")).toBeInTheDocument();
    expect(body).toEqual({ new_name: "base-2" });
    expect(screen.getByRole("link", { name: "Preflight base" })).toHaveAttribute("href", "#/preflight?config=base");
  });
});

const SAVED = [{ name: "cfg", version: 2, versions: 2, model: "tiny-qwen2", backend: "auto", preset: "quick", created_at: "t" }];

function preflightResult(ok: boolean) {
  return {
    ok,
    model: "tiny-qwen2",
    dataset_id: "ds_1",
    selection: CPU,
    plan: { total_updates: 6 },
    effective_training: { max_epochs: 1, learning_rate: 0.0001, eval_every_steps: 10, max_seq_length: 1024 },
    checks: [
      { name: "config", status: "pass", message: "Config 'cfg' is valid.", fix: null },
      ok
        ? { name: "cpu_time", status: "warn", message: "Training on the CPU: roughly 1 min.", fix: "Use a GPU." }
        : {
            name: "cpu_model_size",
            status: "fail",
            message: "'qwen3-1.7b' has 2.03B parameters, above the CPU training limit of 1B.",
            fix: "Use a model of about 1B parameters or less on CPU.",
          },
    ],
  };
}

describe("Preflight", () => {
  it("shows each check with its fix and keeps Start disabled on a failure", async () => {
    window.history.replaceState(null, "", "/#/preflight?config=cfg&version=1");
    let sent: unknown = null;
    const calls = mockServer({
      ...ROUTES,
      "GET /api/configs": SAVED,
      "POST /api/preflight": (_url: URL, body: string) => {
        sent = JSON.parse(body);
        return { body: preflightResult(false) };
      },
    });
    wrap(<Preflight />);
    expect(await screen.findByText("1 check failed. Fix it before training.")).toBeInTheDocument();
    expect(sent).toEqual({ saved_name: "cfg", saved_version: 1, allow_download: false });
    const checks = screen.getByRole("list", { name: "Preflight checks" });
    expect(within(checks).getByText("fail")).toBeInTheDocument();
    expect(within(checks).getByText("What to do: Use a model of about 1B parameters or less on CPU.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start training" })).toBeDisabled();
    expect(screen.getByText("Start is disabled while any check fails.")).toBeInTheDocument();
    expect(calls.some((c) => c.url === "/api/runs")).toBe(false);
  });

  it("enables Start when nothing fails and starts the run", async () => {
    window.history.replaceState(null, "", "/#/preflight");
    let started: unknown = null;
    mockServer({
      ...ROUTES,
      "GET /api/configs": SAVED,
      "POST /api/preflight": preflightResult(true),
      "POST /api/runs": (_url: URL, body: string) => {
        started = JSON.parse(body);
        return { status: 409, body: { error: { message: "The model is in use by job_x.", hint: "Wait for it." } } };
      },
    });
    window.history.replaceState(null, "", "/#/preflight");
    const view = wrap(<App />);
    expect(await view.findByText("Ready to train (1 warning).")).toBeInTheDocument();
    const start = screen.getByRole("button", { name: "Start training" });
    expect(start).toBeEnabled();
    await userEvent.click(start);
    expect(await screen.findByText("The model is in use by job_x.")).toBeInTheDocument(); // global error banner
    expect(started).toEqual({ saved_name: "cfg", saved_version: null, allow_download: false });
  });

  it("opens the run monitor once the run starts", async () => {
    window.history.replaceState(null, "", "/#/preflight");
    mockServer({
      ...ROUTES,
      "GET /api/configs": SAVED,
      "POST /api/preflight": preflightResult(true),
      "POST /api/runs": { run_id: "run_1", job: { id: "job_1" } },
    });
    wrap(<Preflight />);
    await userEvent.click(await screen.findByRole("button", { name: "Start training" }));
    await waitFor(() => expect(window.location.hash).toBe("#/monitor?run=run_1"));
  });

  it("names the job holding the model and keeps Start disabled", async () => {
    const holder = { id: "job_9", type: "evaluate", status: "running", run_id: "run_9" };
    mockServer({
      ...ROUTES,
      "GET /api/jobs": { jobs: [holder], model_lock_holder: "job_9", reconciled: [] },
      "GET /api/configs": SAVED,
      "POST /api/preflight": preflightResult(true),
    });
    wrap(<Preflight />);
    expect(await screen.findByText(/Another job is using the model: job_9 \(evaluate, running\)/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start training" })).toBeDisabled();
    expect(screen.getByText("Start is disabled while another job is using the model.")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Watch it" })).toHaveAttribute("href", "#/monitor?run=run_9");
  });
});

describe("Models", () => {
  it("shows the estimate badge, cached state and a download that runs as a job", async () => {
    let downloaded = false;
    mockServer({
      ...ROUTES,
      "POST /api/models/tiny-qwen2/download": () => {
        downloaded = true;
        return { body: { id: "job_dl", type: "model_download", status: "queued" } };
      },
    });
    wrap(<Models />);
    const card = await screen.findByRole("article", { name: "Tiny Qwen2 (test)" });
    expect(within(card).getByText("fits")).toBeInTheDocument();
    expect(within(card).getByText(/about 0.7 GB needed of 11.8 GB \(RAM - 4 GB\)/)).toBeInTheDocument();
    expect(within(card).getByText("not downloaded")).toBeInTheDocument();
    expect(within(card).getByText("test only")).toBeInTheDocument();
    expect(screen.getByText("Memory figures are catalog estimates.")).toBeInTheDocument();
    await userEvent.click(within(card).getByRole("button", { name: /Download/ }));
    await waitFor(() => expect(downloaded).toBe(true));
    expect(await within(card).findByText("job job_dl")).toBeInTheDocument();
  });
});
