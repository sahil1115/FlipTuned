// Pick a finished training run (for evaluation and chat). The choice lives in the address.
import { useHistory } from "../lib/queries";
import { ErrorPanel, Field, inputClass, Spinner } from "./ui";

export function RunPicker({
  value,
  onChange,
  id,
  hint,
}: {
  value: string | null;
  onChange: (runId: string) => void;
  id: string;
  hint?: string;
}) {
  const runs = useHistory({ status: "succeeded" });
  if (runs.error) return <ErrorPanel error={runs.error} />;
  if (!runs.data) return <Spinner />;
  if (!runs.data.length) {
    return (
      <p className="text-sm">
        No finished training run yet. Train one from{" "}
        <a className="text-sky-800 underline dark:text-sky-300" href="#/preflight">
          Preflight
        </a>
        .
      </p>
    );
  }
  return (
    <Field label="Training run" htmlFor={id} hint={hint}>
      <select id={id} className={`${inputClass} md:w-auto`} value={value ?? ""} onChange={(e) => onChange(e.target.value)}>
        {!value && <option value="">Choose a run…</option>}
        {runs.data.map((r) => (
          <option key={r.id} value={r.id}>
            {r.id} · {r.model_id}
            {r.config_name ? ` · ${r.config_name} v${r.config_version}` : ""}
            {r.verdict ? ` · ${r.verdict.verdict}` : ""}
          </option>
        ))}
      </select>
    </Field>
  );
}
