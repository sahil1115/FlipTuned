// Review: filter records, read one with its code rendered, approve or reject singly or in bulk.
// Privacy findings come from the server already redacted (type and length only).
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api, type Schemas, unwrap } from "../api/client";
import { useReportError } from "../App";
import { CodeText } from "../components/CodeText";
import { Badge, Button, Card, CountList, ErrorPanel, Field, inputClass, Spinner, statusTone } from "../components/ui";
import { useSources } from "../lib/queries";

const STATUSES = ["pending_review", "approved", "rejected", "quarantined"] as const;
const PAGE = 50;
type Decision = "approved" | "rejected" | "pending_review";

function key(item: Schemas["ReviewItem"]): string {
  return `${item.source_id}/${item.record_id}`;
}

function firstLine(text: string | null | undefined): string {
  const line = (text ?? "").split("\n").find((l) => l.trim()) ?? "";
  return line.length > 120 ? `${line.slice(0, 117)}…` : line;
}

export function Review() {
  const [statuses, setStatuses] = useState<string[]>(["pending_review"]);
  const [source, setSource] = useState("");
  const [flag, setFlag] = useState("");
  const [category, setCategory] = useState("");
  const [onlyDuplicates, setOnlyDuplicates] = useState(false);
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [open, setOpen] = useState<string | null>(null);
  const sources = useSources();
  const client = useQueryClient();
  const report = useReportError();

  const filter = {
    sources: source ? [source] : [],
    statuses: statuses as Schemas["ReviewFilter"]["statuses"],
    flag: flag.trim() || null,
    category: category.trim() || null,
    only_duplicates: onlyDuplicates,
    include_source_blocks: false,
  };
  const records = useQuery({
    queryKey: ["records", filter, offset],
    queryFn: () => unwrap(api.POST("/api/records/search", { body: { filter, limit: PAGE, offset } })),
  });
  const decide = useMutation({
    mutationFn: (args: { items: Schemas["ReviewItem"][]; status: Decision }) =>
      unwrap(
        api.POST("/api/records/decisions", {
          body: {
            records: args.items.map((i) => ({ source_id: i.source_id, record_id: i.record_id })),
            status: args.status,
            origin: args.items.length > 1 ? "bulk" : "ui",
          },
        }),
      ),
    onSuccess: () => {
      setSelected(new Set());
      client.invalidateQueries({ queryKey: ["records"] });
    },
    onError: report,
  });

  const items = records.data?.items ?? [];
  const chosen = items.filter((i) => selected.has(key(i)));
  const detail = items.find((i) => key(i) === open) ?? null;
  const changeFilter = (fn: () => void) => {
    fn();
    setOffset(0);
    setSelected(new Set());
  };

  return (
    <div className="space-y-4">
      <Card title="Filters">
        <div className="grid gap-3 md:grid-cols-4">
          <fieldset className="space-y-1">
            <legend className="text-sm font-medium">Status</legend>
            {STATUSES.map((s) => (
              <label key={s} className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={statuses.includes(s)}
                  onChange={(e) =>
                    changeFilter(() =>
                      setStatuses((list) => (e.target.checked ? [...list, s] : list.filter((x) => x !== s))),
                    )
                  }
                />
                {s.replace("_", " ")}
              </label>
            ))}
          </fieldset>
          <Field label="Data source" htmlFor="filter-source">
            <select
              id="filter-source"
              className={inputClass}
              value={source}
              onChange={(e) => changeFilter(() => setSource(e.target.value))}
            >
              <option value="">All sources</option>
              {sources.data?.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Flag" htmlFor="filter-flag" hint="e.g. duplicate_question, broken_code_fence, secret">
            <input
              id="filter-flag"
              className={inputClass}
              value={flag}
              onChange={(e) => changeFilter(() => setFlag(e.target.value))}
            />
          </Field>
          <div className="space-y-3">
            <Field label="Category" htmlFor="filter-category">
              <input
                id="filter-category"
                className={inputClass}
                value={category}
                onChange={(e) => changeFilter(() => setCategory(e.target.value))}
              />
            </Field>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={onlyDuplicates}
                onChange={(e) => changeFilter(() => setOnlyDuplicates(e.target.checked))}
              />
              Only duplicate clusters
            </label>
          </div>
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card
          title={records.data ? `Records (${records.data.total})` : "Records"}
          actions={
            <div className="flex gap-1">
              <Button disabled={!chosen.length || decide.isPending} onClick={() => decide.mutate({ items: chosen, status: "approved" })}>
                Approve {chosen.length || ""}
              </Button>
              <Button
                variant="danger"
                disabled={!chosen.length || decide.isPending}
                onClick={() => decide.mutate({ items: chosen, status: "rejected" })}
              >
                Reject {chosen.length || ""}
              </Button>
            </div>
          }
        >
          {records.error ? (
            <ErrorPanel error={records.error} />
          ) : !records.data ? (
            <Spinner />
          ) : (
            <div className="space-y-2 text-sm">
              <CountList counts={records.data.by_status} />
              {items.length === 0 ? (
                <p>No records match these filters.</p>
              ) : (
                <>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={chosen.length === items.length}
                      onChange={(e) => setSelected(e.target.checked ? new Set(items.map(key)) : new Set())}
                    />
                    Select all on this page
                  </label>
                  <ul className="divide-y divide-slate-200 dark:divide-slate-700" aria-label="Records">
                    {items.map((item) => (
                      <li key={key(item)} className="flex items-start gap-2 py-1.5">
                        <input
                          type="checkbox"
                          aria-label={`Select ${item.record_id}`}
                          checked={selected.has(key(item))}
                          onChange={(e) =>
                            setSelected((set) => {
                              const next = new Set(set);
                              if (e.target.checked) next.add(key(item));
                              else next.delete(key(item));
                              return next;
                            })
                          }
                        />
                        <button
                          type="button"
                          className={`flex-1 text-left ${open === key(item) ? "font-semibold" : ""}`}
                          onClick={() => setOpen(key(item))}
                        >
                          <span className="mr-1">
                            <Badge tone={statusTone(item.status)}>{item.status}</Badge>
                          </span>
                          {item.category && <span className="mr-1 text-xs text-slate-600 dark:text-slate-400">{item.category}</span>}
                          {firstLine(item.question ?? item.text)}
                          {item.quality_flags.length + item.privacy_findings.length > 0 && (
                            <span className="ml-1">
                              <Badge tone="warn">{item.quality_flags.length + item.privacy_findings.length} flags</Badge>
                            </span>
                          )}
                        </button>
                      </li>
                    ))}
                  </ul>
                  <div className="flex items-center gap-2">
                    <Button variant="secondary" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>
                      Previous
                    </Button>
                    <span>
                      {offset + 1}–{offset + items.length} of {records.data.total}
                    </span>
                    <Button
                      variant="secondary"
                      disabled={offset + PAGE >= records.data.total}
                      onClick={() => setOffset(offset + PAGE)}
                    >
                      Next
                    </Button>
                  </div>
                </>
              )}
            </div>
          )}
        </Card>

        <Card title="Record">
          {detail ? (
            <RecordDetail item={detail} busy={decide.isPending} onDecide={(status) => decide.mutate({ items: [detail], status })} />
          ) : (
            <p className="text-sm">Select a record to read it.</p>
          )}
        </Card>
      </div>
    </div>
  );
}

function RecordDetail({
  item,
  busy,
  onDecide,
}: {
  item: Schemas["ReviewItem"];
  busy: boolean;
  onDecide: (status: Decision) => void;
}) {
  const decidable = item.record_type === "qa_pair" && item.status !== "quarantined";
  return (
    <article className="space-y-3 text-sm">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone={statusTone(item.status)}>{item.status}</Badge>
        {item.decision_stale && <Badge tone="warn">earlier decision is stale</Badge>}
        <span className="text-xs text-slate-600 dark:text-slate-400">
          {item.source_file} · {item.location}
        </span>
      </div>
      {item.record_type === "qa_pair" ? (
        <>
          <section>
            <h3 className="mb-1 font-semibold">Question</h3>
            <CodeText text={item.question ?? ""} />
          </section>
          <section>
            <h3 className="mb-1 font-semibold">Answer</h3>
            <CodeText text={item.answer ?? ""} />
          </section>
        </>
      ) : (
        <section>
          <h3 className="mb-1 font-semibold">Text block (never trains on its own)</h3>
          <CodeText text={item.text ?? ""} />
        </section>
      )}
      {item.tags.length > 0 && <p>Tags: {item.tags.join(", ")}</p>}
      {item.quality_flags.length > 0 && (
        <p>
          Quality flags:{" "}
          {item.quality_flags.map((f) => (
            <Badge key={f} tone="warn">
              {f}
            </Badge>
          ))}
        </p>
      )}
      {item.privacy_findings.length > 0 && (
        <div>
          <p className="font-medium">Privacy findings (values hidden)</p>
          <ul className="ml-4 list-disc">
            {item.privacy_findings.map((f) => (
              <li key={f}>{f}</li>
            ))}
          </ul>
        </div>
      )}
      {item.duplicate_cluster && <p>Duplicate cluster: {item.duplicate_cluster}</p>}
      {decidable ? (
        <div className="flex gap-2">
          <Button disabled={busy} onClick={() => onDecide("approved")}>
            Approve
          </Button>
          <Button variant="danger" disabled={busy} onClick={() => onDecide("rejected")}>
            Reject
          </Button>
          <Button variant="secondary" disabled={busy} onClick={() => onDecide("pending_review")}>
            Back to pending
          </Button>
        </div>
      ) : (
        <p className="text-slate-600 dark:text-slate-400">
          {item.status === "quarantined" ? "Quarantined records can't be approved." : "Text blocks can't be approved."}
        </p>
      )}
    </article>
  );
}
