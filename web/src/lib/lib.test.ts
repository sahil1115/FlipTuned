import { describe, expect, it } from "vitest";

import { ApiError, captureToken, toApiError, TOKEN_KEY, unwrap } from "../api/client";
import { parseSse } from "../api/sse";
import { splitFences } from "../components/CodeText";
import { DEFAULT_FORM, datasetConfig, validateDatasetForm } from "./datasetForm";
import { buildIngestConfig, initialMapping, joinPath, validateMapping, validateSourceName } from "./ingest";

describe("token", () => {
  it("moves the #token fragment into sessionStorage and out of the address bar", () => {
    window.history.replaceState(null, "", "/#token=abc%2D123");
    expect(captureToken()).toBe("abc-123");
    expect(window.sessionStorage.getItem(TOKEN_KEY)).toBe("abc-123");
    expect(window.location.hash).toBe("#/");
  });
});

describe("errors", () => {
  it("keeps the server's message, hint and details", () => {
    const error = toApiError(404, { error: { message: "No run x.", hint: "See jobs.", details: "d" } });
    expect([error.status, error.message, error.hint, error.details]).toEqual([404, "No run x.", "See jobs.", "d"]);
  });

  it("explains a missing token", () => {
    const error = toApiError(401, null);
    expect(error.hint).toContain("fliptuned serve");
  });

  it("describes bodies without the error shape", () => {
    expect(toApiError(502, "<html>").message).toContain("502");
  });

  it("turns network failures into a clear error", async () => {
    await expect(unwrap(Promise.reject(new TypeError("Failed to fetch")))).rejects.toMatchObject({
      message: "Can't reach the FlipTuned server.",
    });
    const response = new Response(null, { status: 422 });
    await expect(
      unwrap(Promise.resolve({ error: { error: { message: "Bad." } }, response })),
    ).rejects.toBeInstanceOf(ApiError);
  });
});

describe("server-sent events", () => {
  it("parses complete events and keeps the unfinished rest", () => {
    const { events, rest } = parseSse(
      ': keepalive\n\nevent: progress\ndata: {"done":1}\n\nevent: status\ndata: {"status":"succeeded"}\nid: 120\n\nevent: prog',
    );
    expect(events).toEqual([
      { event: "progress", data: { done: 1 }, id: undefined },
      { event: "status", data: { status: "succeeded" }, id: "120" },
    ]);
    expect(rest).toBe("event: prog");
  });

  it("handles CRLF line endings", () => {
    expect(parseSse('event: delta\r\ndata: {"text":"a"}\r\n\r\n').events[0]?.data).toEqual({ text: "a" });
  });
});

describe("column mapping", () => {
  const headers = ["Question", "Answer", "Tags"];

  it("starts from the suggestion and accepts a complete mapping", () => {
    const mapping = initialMapping({ question: "Question", answer: "Answer", tags: null });
    expect(mapping).toEqual({ question: "Question", answer: "Answer" });
    expect(validateMapping(mapping, headers)).toEqual([]);
  });

  it("requires question and answer, real columns, and no column twice", () => {
    expect(validateMapping({}, headers)).toEqual([
      "Pick the column that holds the question.",
      "Pick the column that holds the answer.",
    ]);
    expect(validateMapping({ question: "Question", answer: "Question" }, headers)).toContain(
      'Column "Question" is used for both question and answer.',
    );
    expect(validateMapping({ question: "Q", answer: "Answer" }, headers)).toContain(
      'Column "Q" (question) isn\'t in this file.',
    );
  });

  it("validates the source name", () => {
    expect(validateSourceName("  ")).toBe("Give the data source a name.");
    expect(validateSourceName("x".repeat(101))).toContain("100");
    expect(validateSourceName("faq")).toBeNull();
  });

  it("joins paths in the root's native style", () => {
    expect(joinPath("C:\\data\\qa", "sub/faq.csv")).toBe("C:\\data\\qa\\sub\\faq.csv");
    expect(joinPath("/home/me/qa", "sub/faq.csv")).toBe("/home/me/qa/sub/faq.csv");
    expect(joinPath("/home/me/qa/", "faq.csv")).toBe("/home/me/qa/faq.csv");
  });

  it("builds one source per file with its mapping", () => {
    const scan = {
      root: "/data",
      tabular: [{ path: "a.csv", sheet: null, headers, suggested_mapping: {}, mapping_complete: true }],
      text_files: ["notes.md"],
    } as unknown as Parameters<typeof buildIngestConfig>[1];
    const config = buildIngestConfig(" faq ", scan, { "a.csv": { question: "Question", answer: "Answer", tags: "" } });
    expect(config).toEqual({
      name: "faq",
      sources: [
        { path: "/data/a.csv", columns: { question: "Question", answer: "Answer" } },
        { path: "/data/notes.md" },
      ],
    });
  });
});

describe("dataset form", () => {
  it("lists every problem", () => {
    const errors = validateDatasetForm({ ...DEFAULT_FORM, train: 1.2, seed: 1.5, maxSeqLength: 10 });
    expect(Object.keys(errors).sort()).toEqual(["maxSeqLength", "model", "name", "seed", "train"]);
  });

  it("checks that the split adds up to 1", () => {
    const errors = validateDatasetForm({ ...DEFAULT_FORM, name: "x", model: "m", train: 0.7 });
    expect(errors).toEqual({ split: "Train, valid and test must add up to 1." });
  });

  it("builds the config the server expects", () => {
    expect(datasetConfig({ ...DEFAULT_FORM, name: " fx ", model: "tiny-qwen2" })).toEqual({
      name: "fx",
      model_catalog_id: "tiny-qwen2",
      source_ids: [],
      split: { train: 0.8, valid: 0.1, test: 0.1 },
      seed: 42,
      max_seq_length: 1024,
    });
  });
});

describe("code rendering", () => {
  it("keeps code exactly and labels its language", () => {
    const text = "Use this:\n```python\nif x:\n    return  1\n```\nDone.";
    expect(splitFences(text)).toEqual([
      { kind: "text", content: "Use this:" },
      { kind: "code", content: "if x:\n    return  1", language: "python" },
      { kind: "text", content: "Done." },
    ]);
  });

  it("shows an unclosed fence as code", () => {
    expect(splitFences("```js\nlet a")).toEqual([{ kind: "code", content: "let a", language: "js" }]);
  });
});
