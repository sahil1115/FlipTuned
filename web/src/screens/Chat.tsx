// Chat: talk to a run's fine-tuned model (or its base model). Answers come from the model weights
// only: no document retrieval. The conversation lives in this page only; nothing is logged or saved.
// The chat worker holds the model and releases it after the idle timeout (Settings).
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

import { api, ApiError, rawFetch, type Schemas, unwrap } from "../api/client";
import { followJob, readSse } from "../api/sse";
import { useReportError } from "../App";
import { CodeText } from "../components/CodeText";
import { RunPicker } from "../components/RunPicker";
import { Button, Card, ErrorPanel, inputClass, Spinner } from "../components/ui";
import { useJobs } from "../lib/queries";
import { routeHref, routeParams } from "../lib/route";

interface Turn {
  role: "user" | "assistant";
  content: string;
}

type Phase = "loading" | "ready" | "ended";

export function Chat() {
  const run = routeParams().get("run");
  const [base, setBase] = useState(false);
  const [session, setSession] = useState<Schemas["ChatInfo"] | null>(null);
  const report = useReportError();
  const client = useQueryClient();
  const jobs = useJobs(50);
  const start = useMutation({
    mutationFn: () => unwrap(api.POST("/api/chat/sessions", { body: { run_id: run ?? "", base, max_new_tokens: 512, allow_download: false } })),
    onSuccess: (info) => {
      setSession(info);
      void client.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: report,
  });
  const holderId = jobs.data?.model_lock_holder ?? null;
  const holder = holderId ? jobs.data?.jobs.find((j) => j.id === holderId) : undefined;
  const release = useMutation({
    mutationFn: (jobId: string) => unwrap(api.DELETE("/api/chat/sessions/{job_id}", { params: { path: { job_id: jobId } } })),
    onSuccess: () => void client.invalidateQueries({ queryKey: ["jobs"] }),
    onError: report,
  });

  return (
    <div className="space-y-4">
      <p
        role="note"
        className="rounded-md border border-sky-300 bg-sky-50 p-3 text-sm text-sky-950 dark:border-sky-800 dark:bg-sky-950/60 dark:text-sky-100"
      >
        <strong>Model weights only — no document retrieval.</strong> Answers come from the model alone: it can't look up or
        check your documents, and it can be wrong. Nothing you type here is logged or saved.
      </p>
      {session ? (
        <Conversation key={session.job.id} session={session} onEnd={() => setSession(null)} />
      ) : (
        <Card title="Choose a model">
          <div className="space-y-3 text-sm">
            <RunPicker
              id="chat-run"
              value={run}
              onChange={(id) => {
                window.location.hash = routeHref("/chat", { run: id });
              }}
            />
            <fieldset className="flex flex-wrap gap-4">
              <legend className="mb-1 font-medium">Model</legend>
              <label className="flex items-center gap-2">
                <input type="radio" name="chat-model" checked={!base} onChange={() => setBase(false)} />
                Fine-tuned (base model + this run's adapter)
              </label>
              <label className="flex items-center gap-2">
                <input type="radio" name="chat-model" checked={base} onChange={() => setBase(true)} />
                Base model only (for comparison)
              </label>
            </fieldset>
            {holderId && (
              <p role="status" className="rounded-md border border-amber-300 bg-amber-50 p-2 text-amber-950 dark:border-amber-800 dark:bg-amber-950/50 dark:text-amber-100">
                Another job is using the model: {holderId}
                {holder ? ` (${holder.type}, ${holder.status})` : ""}. One model job runs at a time.{" "}
                {holder?.type === "chat_session" && (
                  <Button variant="secondary" onClick={() => release.mutate(holderId)} disabled={release.isPending}>
                    Release that chat model
                  </Button>
                )}
              </p>
            )}
            <Button onClick={() => start.mutate()} disabled={!run || start.isPending || !!holderId}>
              {start.isPending ? "Starting…" : "Load the model and chat"}
            </Button>
          </div>
        </Card>
      )}
    </div>
  );
}

function Conversation({ session, onEnd }: { session: Schemas["ChatInfo"]; onEnd: () => void }) {
  const jobId = session.job.id;
  const [phase, setPhase] = useState<Phase>("loading");
  const [endReason, setEndReason] = useState<string | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const client = useQueryClient();
  const minutes = Math.round((session.idle_timeout_s / 60) * 10) / 10;
  const box = useRef<HTMLOListElement>(null);

  // Follow the chat worker: "ready" once the model is loaded; the final status when it's released.
  useEffect(() => {
    const controller = new AbortController();
    followJob(jobId, {
      signal: controller.signal,
      onEvent: (event) => {
        const data = event.data as { state?: string; status?: string; error?: { message?: string } | null };
        if (event.event === "chat" && data.state === "ready") setPhase("ready");
        else if (event.event === "chat" && data.state) setEndReason(data.state);
        else if (event.event === "status" && ["succeeded", "failed", "cancelled", "interrupted"].includes(String(data.status))) {
          setPhase("ended");
          if (data.error?.message) setError(new ApiError(0, data.error.message));
          void client.invalidateQueries({ queryKey: ["jobs"] });
        }
      },
    }).catch((e: unknown) => {
      if ((e as Error).name !== "AbortError") setError(e);
    });
    return () => controller.abort();
  }, [jobId, client]);

  // Closing the tab releases the model (the conversation isn't kept anyway). Moving to another
  // screen keeps it until the idle timeout; this screen offers to release it when you come back.
  useEffect(() => {
    const onHide = () => {
      void rawFetch(`/api/chat/sessions/${encodeURIComponent(jobId)}`, { method: "DELETE", keepalive: true }).catch(() => {});
    };
    window.addEventListener("pagehide", onHide);
    return () => window.removeEventListener("pagehide", onHide);
  }, [jobId]);

  useEffect(() => {
    if (box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [turns]);

  async function send() {
    const text = draft.trim();
    if (!text || sending || phase !== "ready") return;
    const history: Turn[] = [...turns, { role: "user", content: text }];
    setTurns([...history, { role: "assistant", content: "" }]);
    setDraft("");
    setSending(true);
    setError(null);
    try {
      const response = await rawFetch(`/api/chat/sessions/${encodeURIComponent(jobId)}/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ turns: history }),
      });
      await readSse(response, (event) => {
        const data = event.data as { text?: string; message?: string };
        if (event.event === "delta" && data.text) {
          setTurns((list) => {
            const last = list[list.length - 1];
            return last ? [...list.slice(0, -1), { ...last, content: last.content + data.text }] : list;
          });
        } else if (event.event === "error") {
          setError(new ApiError(0, data.message ?? "The model couldn't answer.", "Try again, or load the model again."));
        }
      });
    } catch (e) {
      setError(e);
    } finally {
      setSending(false);
    }
  }

  async function close() {
    try {
      await unwrap(api.DELETE("/api/chat/sessions/{job_id}", { params: { path: { job_id: jobId } } }));
    } catch {
      // already released
    }
    onEnd();
  }

  return (
    <Card
      title={`${session.base ? "Base model" : "Fine-tuned model"}: ${session.model} (run ${session.run_id})`}
      actions={
        <Button variant="secondary" onClick={() => void close()}>
          {phase === "ended" ? "Choose another model" : "Close and release the model"}
        </Button>
      }
    >
      <div className="space-y-3 text-sm">
        <p className="text-slate-600 dark:text-slate-400">
          {session.backend} on {session.device} ({session.precision}). The model is released after {minutes} min without a
          message (Settings → idle timeout).
        </p>
        {phase === "loading" && <Spinner label="Loading the model…" />}
        {phase === "ended" && (
          <p role="status" className="rounded-md border border-amber-300 bg-amber-50 p-2 text-amber-950 dark:border-amber-800 dark:bg-amber-950/50 dark:text-amber-100">
            {endReason === "idle_timeout"
              ? `The model was released after ${minutes} min without a message.`
              : "The model was released."}{" "}
            Choose a model again to keep chatting (the conversation starts over).
          </p>
        )}
        {error ? <ErrorPanel error={error} /> : null}
        <ol ref={box} aria-label="Conversation" className="max-h-[32rem] space-y-3 overflow-y-auto">
          {turns.map((turn, index) => (
            <li
              key={index}
              className={
                turn.role === "user"
                  ? "rounded-md bg-sky-50 p-2 dark:bg-slate-800"
                  : "rounded-md border border-slate-200 p-2 dark:border-slate-700"
              }
            >
              <p className="mb-1 text-xs font-medium text-slate-600 dark:text-slate-400">{turn.role === "user" ? "You" : "Model"}</p>
              {turn.content ? <CodeText text={turn.content} /> : <Spinner label="Thinking…" />}
            </li>
          ))}
        </ol>
        <form
          className="flex flex-col gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            void send();
          }}
        >
          <label htmlFor="chat-input" className="font-medium">
            Message
          </label>
          <textarea
            id="chat-input"
            className={`${inputClass} min-h-24`}
            value={draft}
            disabled={phase !== "ready"}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send();
              }
            }}
          />
          <div className="flex items-center gap-2">
            <Button type="submit" disabled={phase !== "ready" || sending || !draft.trim()}>
              {sending ? "Answering…" : "Send"}
            </Button>
            <span className="text-xs text-slate-600 dark:text-slate-400">Enter sends, Shift+Enter adds a line.</span>
          </div>
        </form>
      </div>
    </Card>
  );
}
