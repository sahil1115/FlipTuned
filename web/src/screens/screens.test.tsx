import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError, TOKEN_KEY } from "../api/client";
import { App } from "../App";
import { CodeText } from "../components/CodeText";
import { ErrorPanel } from "../components/ui";
import { mockServer } from "../test/mockServer";
import { Dataset } from "./Dataset";

function wrap(node: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{node}</QueryClientProvider>);
}

const BASE_ROUTES = {
  "GET /api/settings": { app_home: "/h", allowed_roots: [], idle_timeout_min: 10, upload_max_mb: 50, dev_mode: true },
  "GET /api/sources": [{ id: "src_1", name: "faq", kind: "folder", root_path: "/d", status: "ingested", updated_at: "t" }],
  "GET /api/datasets": [],
  "GET /api/jobs": { jobs: [], model_lock_holder: null, reconciled: [] },
  "GET /api/models": {
    models: [{ id: "tiny-qwen2", display_name: "Tiny", params_b: 0.001, trainable: true, test_only: true }],
    estimate_note: "",
  },
};

beforeEach(() => {
  window.sessionStorage.setItem(TOKEN_KEY, "t0ken");
});

afterEach(() => {
  vi.unstubAllGlobals();
  window.sessionStorage.clear();
});

describe("ErrorPanel", () => {
  it("shows message, then what to do, then expandable details", () => {
    render(<ErrorPanel error={new ApiError(409, "The model is in use.", "Cancel job x.", "job_x (train)")} />);
    expect(screen.getByRole("alert")).toHaveTextContent("The model is in use.");
    expect(screen.getByText("What to do: Cancel job x.")).toBeInTheDocument();
    expect(screen.getByText("Details")).toBeInTheDocument();
  });
});

describe("CodeText", () => {
  it("renders code blocks verbatim", () => {
    const { container } = render(<CodeText text={"Hi\n```sql\nSELECT  1;\n```"} />);
    expect(container.querySelector("code")?.textContent).toBe("SELECT  1;");
    expect(screen.getByText("sql")).toBeInTheDocument();
  });
});

describe("Dataset build form", () => {
  it("shows validation errors and sends nothing until they're fixed", async () => {
    const calls = mockServer(BASE_ROUTES);
    wrap(<Dataset />);
    await userEvent.click(await screen.findByRole("button", { name: "Build dataset" }));
    expect(await screen.findByText("Give the dataset a name.")).toBeInTheDocument();
    expect(screen.getByText(/Choose the model/)).toBeInTheDocument();
    expect(calls.filter((c) => c.method === "POST")).toEqual([]);
    expect(calls.length).toBeGreaterThan(0);
    expect(calls.every((c) => c.headers.get("X-FlipTuned-Token") === "t0ken")).toBe(true);
  });

  it("shows the server's error with what to do", async () => {
    let posted: unknown = null;
    mockServer({
      ...BASE_ROUTES,
      "POST /api/datasets": (_url: URL, sent: string) => {
        posted = sent;
        return { status: 422, body: { error: { message: "The dataset config is not valid.", hint: "Fix the name.", details: "name: bad" } } };
      },
    });
    window.history.replaceState(null, "", "/#/dataset");
    const { container } = wrap(<App />); // failed actions show in the app's global error banner
    await userEvent.type(await screen.findByLabelText("Name"), "fixtures");
    await waitFor(() => expect(screen.getByRole("option", { name: /Tiny/ })).toBeInTheDocument());
    await userEvent.selectOptions(screen.getByLabelText("Model"), "tiny-qwen2");
    await userEvent.click(screen.getByRole("button", { name: "Build dataset" }));
    expect(await screen.findByText("The dataset config is not valid.")).toBeInTheDocument();
    expect(screen.getByText("What to do: Fix the name.")).toBeInTheDocument();
    expect(JSON.parse(String(posted))).toMatchObject({ config: { name: "fixtures", model_catalog_id: "tiny-qwen2" } });
    expect(container.querySelector("[role=alert]")).not.toBeNull();
  });
});

describe("App shell", () => {
  it("asks for the token link when there is none", () => {
    window.sessionStorage.clear();
    wrap(<App />);
    expect(screen.getByRole("alert")).toHaveTextContent("needs the access token");
  });

  it("shows only the screens that exist and a global error for failed actions", async () => {
    mockServer(BASE_ROUTES);
    window.history.replaceState(null, "", "/#/dataset");
    wrap(<App />);
    const nav = screen.getByRole("navigation", { name: "Main" });
    expect(Array.from(nav.querySelectorAll("a")).map((a) => a.textContent)).toEqual([
      "Dashboard",
      "Data sources",
      "Review",
      "Dataset",
      "Models",
      "Training config",
      "Preflight",
      "Run monitor",
      "Evaluation",
      "Chat",
      "Runs",
      "Settings",
    ]);
    expect(screen.getByRole("link", { name: "Dataset" })).toHaveAttribute("aria-current", "page");
    await userEvent.click(screen.getByRole("button", { name: /Dark theme/ }));
    expect(document.documentElement).toHaveClass("dark");
  });
});
