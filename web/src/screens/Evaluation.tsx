// Evaluation: start an evaluation job, review blind A/B pairs (keys 1/2/3; come back any time, the
// judgements are in the registry), then the verdict with every input to it. The server never says
// which model wrote A or B; scoring un-blinds on the server.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { api, rawFetch, type Schemas, unwrap } from "../api/client";
import { useReportError } from "../App";
import { CodeText } from "../components/CodeText";
import { JobProgress } from "../components/JobProgress";
import { RunPicker } from "../components/RunPicker";
import { Badge, Button, Card, ErrorPanel, Field, inputClass, Spinner, Stat, statusTone } from "../components/ui";
import { routeHref, routeParams } from "../lib/route";

type Judgement = "A" | "B" | "tie";
type Info = Schemas["EvaluationInfo"];

/** Review keys: 1 = A is better, 2 = B is better, 3 = tie. */
export function keyToJudgement(key: string): Judgement | null {
  return key === "1" ? "A" : key === "2" ? "B" : key === "3" ? "tie" : null;
}

const VERDICT_TONE = { accepted: "good", rejected: "bad", inconclusive: "warn" } as const;

function go(run: string | null, evaluation?: string | null) {
  window.location.hash = routeHref("/evaluation", { run: run ?? undefined, eval: evaluation ?? undefined });
}

export function Evaluation() {
  const params = routeParams();
  const run = params.get("run");
  const evalId = params.get("eval");
  return (
    <div className="space-y-4">
      <RunPicker id="ev-run" value={run} onChange={(id) => go(id)} hint="Only finished runs can be evaluated." />
      {run && <Evaluations key={run} runId={run} selected={evalId} />}
      {run && evalId && <EvaluationDetail key={evalId} runId={run} evaluationId={evalId} />}
    </div>
  );
}

function Evaluations({ runId, selected }: { runId: string; selected: string | null }) {
  const client = useQueryClient();
  const report = useReportError();
  const list = useQuery({
    queryKey: ["evaluations", runId],
    queryFn: () => unwrap(api.GET("/api/runs/{run_id}/evaluations", { params: { path: { run_id: runId } } })),
    refetchInterval: 5_000,
  });
  const [limit, setLimit] = useState("");
  const [allowDownload, setAllowDownload] = useState(false);
  const limitError = limit && !(Number.isInteger(Number(limit)) && Number(limit) >= 1) ? "Use a whole number of 1 or more." : null;
  const start = useMutation({
    mutationFn: () =>
      unwrap(
        api.POST("/api/runs/{run_id}/evaluations", {
          params: { path: { run_id: runId } },
          body: { limit: limit ? Number(limit) : null, allow_download: allowDownload },
        }),
      ),
    onSuccess: (started) => {
      void client.invalidateQueries({ queryKey: ["evaluations", runId] });
      go(runId, started.evaluation_id);
    },
    onError: report,
  });

  return (
    <Card title="Evaluations of this run">
      <div className="space-y-4 text-sm">
        <p>
          An evaluation answers every held-out test question (and about 20 general coding prompts) with the base model and
          the fine-tuned model, using the run's backend, the same system prompt and greedy decoding. Then you compare the
          answers blind.
        </p>
        <div className="flex flex-wrap items-end gap-3">
          <Field
            label="Only the first N test items (optional)"
            htmlFor="ev-limit"
            hint="For a quick check. Below 50 items the verdict is 'inconclusive'."
            error={limitError}
          >
            <input id="ev-limit" className={`${inputClass} w-40`} inputMode="numeric" value={limit} onChange={(e) => setLimit(e.target.value.trim())} />
          </Field>
          <label className="flex items-center gap-2 pb-6">
            <input type="checkbox" checked={allowDownload} onChange={(e) => setAllowDownload(e.target.checked)} />
            Download model files if they're missing
          </label>
          <div className="pb-6">
            <Button onClick={() => start.mutate()} disabled={!!limitError || start.isPending}>
              {start.isPending ? "Starting…" : "Start an evaluation"}
            </Button>
          </div>
        </div>
        {list.error ? (
          <ErrorPanel error={list.error} />
        ) : !list.data ? (
          <Spinner />
        ) : list.data.length === 0 ? (
          <p>No evaluation of this run yet.</p>
        ) : (
          <ul className="space-y-1" aria-label="Evaluations">
            {list.data.map((e) => (
              <li key={e.id} className="flex flex-wrap items-center gap-2">
                <Badge tone={statusTone(e.status)}>{e.status}</Badge>
                {e.verdict && <Badge tone={VERDICT_TONE[e.verdict as keyof typeof VERDICT_TONE] ?? "neutral"}>{e.verdict}</Badge>}
                <a className="text-sky-800 underline dark:text-sky-300" href={routeHref("/evaluation", { run: runId, eval: e.id })} aria-current={e.id === selected ? "true" : undefined}>
                  {e.id}
                </a>
                <span className="text-xs text-slate-600 dark:text-slate-400">
                  {e.judged} of {e.total} pairs reviewed · {new Date(e.created_at).toLocaleString()}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </Card>
  );
}

function EvaluationDetail({ runId, evaluationId }: { runId: string; evaluationId: string }) {
  const info = useQuery({
    queryKey: ["evaluation", evaluationId],
    queryFn: () => unwrap(api.GET("/api/evaluations/{evaluation_id}", { params: { path: { evaluation_id: evaluationId } } })),
    refetchInterval: (q) => (q.state.data && ["queued", "running"].includes(q.state.data.status) ? 3_000 : false),
  });
  if (info.error) return <ErrorPanel error={info.error} />;
  if (!info.data) return <Spinner />;
  const e = info.data;
  if (e.status === "queued" || e.status === "running") {
    return (
      <Card title={`Evaluation ${e.id}`}>
        {e.job_id ? (
          <JobProgress jobId={e.job_id} label="Generating answers" />
        ) : (
          <Spinner label="Waiting for the job to start…" />
        )}
        <p className="mt-2 text-sm text-slate-600 dark:text-slate-400">
          Both models answer every question; this can take a while. You can leave this page and come back.
        </p>
      </Card>
    );
  }
  if (e.status !== "ready") {
    return (
      <Card title={`Evaluation ${e.id}`}>
        <p className="text-sm">
          This evaluation is <Badge tone={statusTone(e.status)}>{e.status}</Badge>: the answers weren't all generated. Start a
          new evaluation of run {runId}.
        </p>
      </Card>
    );
  }
  return (
    <>
      <Review info={e} />
      <ReviewSheet info={e} />
      <Results info={e} />
    </>
  );
}

// --------------------------------------------------------------------------- blind review

function Review({ info }: { info: Info }) {
  const client = useQueryClient();
  const report = useReportError();
  const [position, setPosition] = useState<number | null>(null); // null = next unreviewed
  const pair = useQuery({
    queryKey: ["pair", info.id, position],
    queryFn: () =>
      unwrap(
        api.GET("/api/evaluations/{evaluation_id}/pairs", {
          params: { path: { evaluation_id: info.id }, query: position ? { position } : {} },
        }),
      ),
  });
  const judge = useMutation({
    mutationFn: (judgement: Judgement) =>
      unwrap(
        api.POST("/api/evaluations/{evaluation_id}/judgements", {
          params: { path: { evaluation_id: info.id } },
          body: { item_id: pair.data?.pair?.item_id ?? "", judgement },
        }),
      ),
    onSuccess: () => {
      setPosition(null);
      void client.invalidateQueries({ queryKey: ["pair", info.id] });
      void client.invalidateQueries({ queryKey: ["evaluation", info.id] });
      void client.invalidateQueries({ queryKey: ["evaluations", info.run_id] });
      void client.invalidateQueries({ queryKey: ["score", info.id] });
    },
    onError: report,
  });
  const current = pair.data?.pair ?? null;

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      const judgement = keyToJudgement(event.key);
      if (judgement && current && !judge.isPending) {
        event.preventDefault();
        judge.mutate(judgement);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [current, judge]);

  const total = pair.data?.total ?? info.total;
  const judged = pair.data?.judged ?? info.judged;
  return (
    <Card title="Blind review">
      <div className="space-y-3 text-sm">
        <div className="flex flex-wrap items-center gap-3">
          <progress className="h-2 w-48" max={total || 1} value={judged} aria-label="Review progress" />
          <span role="status">
            {judged} of {total} pairs reviewed
          </span>
          <span className="text-slate-600 dark:text-slate-400">
            Keys: 1 = A is better, 2 = B is better, 3 = tie. You can stop and come back; your judgements are saved.
          </span>
        </div>
        {pair.error ? (
          <ErrorPanel error={pair.error} />
        ) : !pair.data ? (
          <Spinner />
        ) : !current ? (
          <p className="font-medium">Every pair is reviewed. The results below use all of them.</p>
        ) : (
          <div className="space-y-3">
            <p className="flex flex-wrap items-center gap-2">
              <span className="font-medium">
                Pair {current.position} of {total}
              </span>
              {current.category && <Badge>{current.category}</Badge>}
              {current.judgement && <Badge tone="good">your judgement: {current.judgement === "tie" ? "tie" : `${current.judgement} better`}</Badge>}
            </p>
            <section aria-label="Question" className="space-y-1">
              <h3 className="font-semibold">Question</h3>
              <CodeText text={current.question} />
            </section>
            <details>
              <summary className="cursor-pointer">Approved answer (reference)</summary>
              <CodeText text={current.reference} />
            </details>
            <div className="grid gap-3 lg:grid-cols-2">
              <section aria-label="Answer A" className="space-y-1 rounded-md border border-slate-200 p-2 dark:border-slate-700">
                <h3 className="font-semibold">Answer A</h3>
                <CodeText text={current.answer_a} />
              </section>
              <section aria-label="Answer B" className="space-y-1 rounded-md border border-slate-200 p-2 dark:border-slate-700">
                <h3 className="font-semibold">Answer B</h3>
                <CodeText text={current.answer_b} />
              </section>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Button onClick={() => judge.mutate("A")} disabled={judge.isPending} aria-keyshortcuts="1">
                A is better (1)
              </Button>
              <Button onClick={() => judge.mutate("B")} disabled={judge.isPending} aria-keyshortcuts="2">
                B is better (2)
              </Button>
              <Button variant="secondary" onClick={() => judge.mutate("tie")} disabled={judge.isPending} aria-keyshortcuts="3">
                Tie (3)
              </Button>
              <span className="mx-2 text-slate-400">|</span>
              <Button variant="ghost" onClick={() => setPosition(Math.max(1, current.position - 1))} disabled={current.position <= 1}>
                Previous pair
              </Button>
              <Button variant="ghost" onClick={() => setPosition(Math.min(total, current.position + 1))} disabled={current.position >= total}>
                Next pair
              </Button>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}

function ReviewSheet({ info }: { info: Info }) {
  const client = useQueryClient();
  const report = useReportError();
  const upload = useMutation({
    mutationFn: async (file: File) => {
      const response = await rawFetch(`/api/evaluations/${encodeURIComponent(info.id)}/review-sheet`, {
        method: "POST",
        body: await file.text(),
        headers: { "Content-Type": "text/csv" },
      });
      return (await response.json()) as Schemas["SheetImportResult"];
    },
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["pair", info.id] });
      void client.invalidateQueries({ queryKey: ["evaluation", info.id] });
      void client.invalidateQueries({ queryKey: ["score", info.id] });
    },
    onError: report,
  });
  return (
    <details className="rounded-lg border border-slate-200 bg-white p-4 text-sm dark:border-slate-700 dark:bg-slate-900">
      <summary className="cursor-pointer font-semibold">Or review in a spreadsheet</summary>
      <div className="mt-2 space-y-2">
        <p>
          The evaluation folder has <code>review_sheet.csv</code> ({info.eval_dir}). Fill in the judgement column with A, B or
          tie and upload it here; it's scored the same way as reviews made on this page.
        </p>
        <label className="flex items-center gap-2">
          Filled-in review sheet
          <input type="file" accept=".csv,text/csv" onChange={(e) => e.target.files?.[0] && upload.mutate(e.target.files[0])} />
        </label>
        {upload.data && (
          <p role="status">
            Recorded {Object.values(upload.data.recorded).reduce((a, b) => a + b, 0)} judgements; {upload.data.blank} rows left
            blank
            {upload.data.problems.length ? `; ${upload.data.problems.length} rows skipped` : ""}.
          </p>
        )}
      </div>
    </details>
  );
}

// --------------------------------------------------------------------------- results

function pct(value: number | null | undefined): string {
  return value == null ? "not run" : `${value.toFixed(1)}%`;
}

function Results({ info }: { info: Info }) {
  const score = useQuery({
    queryKey: ["score", info.id],
    queryFn: () => unwrap(api.POST("/api/evaluations/{evaluation_id}/score", { params: { path: { evaluation_id: info.id } } })),
  });
  if (score.error) return <ErrorPanel error={score.error} />;
  if (!score.data) return <Spinner label="Scoring…" />;
  return <Verdict result={score.data} />;
}

export function Verdict({ result }: { result: Schemas["ScoreResult"] }) {
  return (
    <Card title="Results">
      <div className="space-y-4 text-sm">
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-base font-semibold">Verdict</span>
          <span className="text-lg">
            <Badge tone={VERDICT_TONE[result.verdict]}>{result.verdict}</Badge>
          </span>
        </div>
        {result.reasons.length > 0 && (
          <ul className="list-disc space-y-1 pl-5" aria-label="Why">
            {result.reasons.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        )}
        <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="Test set size" value={result.test_items} />
          <Stat label="Pairs reviewed" value={`${result.judged} (${Math.round(result.review_coverage * 100)}%)`} />
          <Stat label="Fine-tuned / base / tie" value={`${result.wins} / ${result.losses} / ${result.ties}`} />
          <Stat label="Sign test" value={`p = ${Number(result.sign_test_p.toPrecision(4))}`} />
          <Stat label="Checks passed (base → tuned)" value={`${pct(result.deterministic_pass_base)} → ${pct(result.deterministic_pass_tuned)}`} />
          <Stat label="General coding set (base → tuned)" value={`${pct(result.regression_pass_base)} → ${pct(result.regression_pass_tuned)}`} />
          <Stat label="Regressions" value={result.regressions} />
        </dl>

        <div>
          <h3 className="mb-1 font-semibold">Every input to the verdict</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-left" aria-label="Verdict criteria">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-700">
                  <th className="py-1 pr-3">Criterion</th>
                  <th className="pr-3">Value</th>
                  <th className="pr-3">Needed</th>
                  <th className="pr-3">Met</th>
                  <th>If not met</th>
                </tr>
              </thead>
              <tbody>
                {result.criteria.map((c) => (
                  <tr key={c.name} className="border-b border-slate-100 dark:border-slate-800">
                    <td className="py-1 pr-3">{c.description}</td>
                    <td className="pr-3">{c.display_value}</td>
                    <td className="pr-3">{c.display_threshold}</td>
                    <td className="pr-3">
                      {!c.applicable ? <Badge>n/a</Badge> : c.met ? <Badge tone="good">yes</Badge> : <Badge tone="bad">no</Badge>}
                    </td>
                    <td>{c.stage}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div>
          <h3 className="mb-1 font-semibold">By category</h3>
          {result.categories.length === 0 ? (
            <p>No categories.</p>
          ) : (
            <table className="text-left" aria-label="Results by category">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-700">
                  <th className="py-1 pr-4">Category</th>
                  <th className="pr-4">Items</th>
                  <th className="pr-4">Reviewed</th>
                  <th className="pr-4">Fine-tuned won</th>
                  <th className="pr-4">Base won</th>
                  <th>Ties</th>
                </tr>
              </thead>
              <tbody>
                {result.categories.map((c) => (
                  <tr key={c.category}>
                    <td className="py-1 pr-4">{c.category}</td>
                    <td className="pr-4">{c.items}</td>
                    <td className="pr-4">{c.judged}</td>
                    <td className="pr-4">{c.wins}</td>
                    <td className="pr-4">{c.losses}</td>
                    <td>{c.ties}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {result.prior_exposure_note && (
          <p className="rounded-md border border-amber-300 bg-amber-50 p-2 text-amber-950 dark:border-amber-800 dark:bg-amber-950/50 dark:text-amber-100">
            {result.prior_exposure_note}
          </p>
        )}
        <div>
          <h3 className="mb-1 font-semibold">Limitations</h3>
          <ul className="list-disc space-y-1 pl-5" aria-label="Limitations">
            {result.limitations.map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
        </div>
        <p className="text-slate-600 dark:text-slate-400">
          The full report is in {result.report_path}. Falling training loss alone never counts as success; this verdict does.
        </p>
      </div>
    </Card>
  );
}
