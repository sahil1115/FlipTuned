// Small UI building blocks (Tailwind). Written for this project and kept in the repo.
import type { ButtonHTMLAttributes, ReactNode } from "react";

import { ApiError } from "../api/client";

type Variant = "primary" | "secondary" | "danger" | "ghost";

const VARIANTS: Record<Variant, string> = {
  primary: "bg-sky-700 text-white hover:bg-sky-800 dark:bg-sky-600 dark:hover:bg-sky-500",
  secondary:
    "border border-slate-300 bg-white text-slate-900 hover:bg-slate-100 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100 dark:hover:bg-slate-700",
  danger: "bg-rose-700 text-white hover:bg-rose-800 dark:bg-rose-600 dark:hover:bg-rose-500",
  ghost: "text-sky-800 hover:bg-sky-100 dark:text-sky-300 dark:hover:bg-slate-800",
};

export function Button({
  variant = "primary",
  className = "",
  type = "button",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  return (
    <button
      type={type}
      className={`inline-flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-50 ${VARIANTS[variant]} ${className}`}
      {...props}
    />
  );
}

export function Card({ title, actions, children }: { title: string; actions?: ReactNode; children: ReactNode }) {
  return (
    <section
      aria-label={title}
      className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900"
    >
      <div className="mb-3 flex items-center justify-between gap-2">
        <h2 className="text-base font-semibold">{title}</h2>
        {actions}
      </div>
      {children}
    </section>
  );
}

const TONES = {
  neutral: "bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-200",
  good: "bg-emerald-100 text-emerald-900 dark:bg-emerald-900/50 dark:text-emerald-200",
  warn: "bg-amber-100 text-amber-900 dark:bg-amber-900/50 dark:text-amber-200",
  bad: "bg-rose-100 text-rose-900 dark:bg-rose-900/50 dark:text-rose-200",
};

export function Badge({ tone = "neutral", children }: { tone?: keyof typeof TONES; children: ReactNode }) {
  return <span className={`inline-block rounded px-1.5 py-0.5 text-xs font-medium ${TONES[tone]}`}>{children}</span>;
}

export function statusTone(status: string): keyof typeof TONES {
  if (status === "approved" || status === "succeeded" || status === "pass") return "good";
  if (status === "rejected" || status === "failed" || status === "fail" || status === "quarantined") return "bad";
  if (status === "pending_review" || status === "cancelled" || status === "interrupted" || status === "warn")
    return "warn";
  return "neutral";
}

export const inputClass =
  "w-full rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800";

export function Field({
  label,
  hint,
  error,
  htmlFor,
  children,
}: {
  label: string;
  hint?: string;
  error?: string | null;
  htmlFor: string;
  children: ReactNode;
}) {
  return (
    <div className="space-y-1">
      <label htmlFor={htmlFor} className="block text-sm font-medium">
        {label}
      </label>
      {children}
      {hint && <p className="text-xs text-slate-600 dark:text-slate-400">{hint}</p>}
      {error && (
        <p role="alert" className="text-xs text-rose-700 dark:text-rose-300">
          {error}
        </p>
      )}
    </div>
  );
}

export function Stat({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div>
      <dt className="text-xs text-slate-600 dark:text-slate-400">{label}</dt>
      <dd className="text-lg font-semibold">{value}</dd>
    </div>
  );
}

export function CountList({ counts, empty = "none" }: { counts: Record<string, number>; empty?: string }) {
  const entries = Object.entries(counts);
  if (!entries.length) return <span className="text-sm text-slate-600 dark:text-slate-400">{empty}</span>;
  return (
    <ul className="flex flex-wrap gap-1">
      {entries.map(([key, count]) => (
        <li key={key}>
          <Badge tone={statusTone(key)}>
            {key}: {count}
          </Badge>
        </li>
      ))}
    </ul>
  );
}

/** Errors as the spec wants them: plain message, then what to do, then expandable details. */
export function ErrorPanel({ error, onDismiss }: { error: unknown; onDismiss?: () => void }) {
  const api = error instanceof ApiError ? error : null;
  const message = api?.message ?? (error instanceof Error ? error.message : "Something went wrong.");
  const hint = api ? api.hint : "Try again. If it keeps happening, check the server console.";
  return (
    <div
      role="alert"
      className="rounded-md border border-rose-300 bg-rose-50 p-3 text-sm text-rose-950 dark:border-rose-800 dark:bg-rose-950/60 dark:text-rose-100"
    >
      <div className="flex items-start justify-between gap-2">
        <p className="font-medium">{message}</p>
        {onDismiss && (
          <Button variant="ghost" onClick={onDismiss} aria-label="Dismiss error">
            ✕
          </Button>
        )}
      </div>
      {hint && <p className="mt-1">What to do: {hint}</p>}
      {api?.details && (
        <details className="mt-2">
          <summary className="cursor-pointer">Details</summary>
          <pre className="mt-1 whitespace-pre-wrap text-xs">{api.details}</pre>
        </details>
      )}
    </div>
  );
}

export function Spinner({ label = "Loading…" }: { label?: string }) {
  return (
    <p role="status" className="text-sm text-slate-600 dark:text-slate-400">
      {label}
    </p>
  );
}
