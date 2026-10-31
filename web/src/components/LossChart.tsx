// Train and validation loss by optimizer update, with the best checkpoint marked. The numbers are
// also in a table (collapsed) for screen readers and copying.
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { fmt, type LossPoint } from "../lib/monitor";

interface Row {
  step: number;
  loss: number | null;
  eval_loss: number | null;
}

/** One row per step; a step's train and validation losses arrive as separate points. */
export function chartRows(points: LossPoint[]): Row[] {
  const rows = new Map<number, Row>();
  for (const p of points) {
    const row = rows.get(p.step) ?? { step: p.step, loss: null, eval_loss: null };
    if (p.loss != null) row.loss = p.loss;
    if (p.eval_loss != null) row.eval_loss = p.eval_loss;
    rows.set(p.step, row);
  }
  return [...rows.values()].sort((a, b) => a.step - b.step);
}

const TRAIN = "#0284c7"; // sky-600
const VALID = "#d97706"; // amber-600
const BEST = "#059669"; // emerald-600

export function LossChart({ points, best }: { points: LossPoint[]; best: { step: number; eval_loss: number; final: boolean } | null }) {
  const rows = chartRows(points);
  const last = rows.at(-1);
  if (!last) {
    return <p className="text-sm text-slate-600 dark:text-slate-400">No loss reported yet. The first point appears after the first update.</p>;
  }
  const summary =
    `Loss chart: ${rows.length} steps up to step ${last.step}.` +
    (best ? ` ${best.final ? "Best" : "Best so far"}: validation loss ${fmt(best.eval_loss)} at step ${best.step}.` : "");
  return (
    <figure className="space-y-2">
      <div role="img" aria-label={summary} className="h-64 w-full text-slate-700 dark:text-slate-300">
        <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 640, height: 256 }}>
          <LineChart data={rows} margin={{ top: 16, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid stroke="currentColor" strokeOpacity={0.15} />
            <XAxis dataKey="step" type="number" domain={["dataMin", "dataMax"]} allowDecimals={false} stroke="currentColor" fontSize={12} />
            <YAxis stroke="currentColor" fontSize={12} width={48} domain={["auto", "auto"]} />
            <Tooltip
              contentStyle={{ background: "rgb(15 23 42)", border: "none", borderRadius: 6, color: "white", fontSize: 12 }}
              labelFormatter={(step) => `Step ${String(step)}`}
              formatter={(value) => (typeof value === "number" ? fmt(value) : String(value))}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line name="Train loss" type="linear" dataKey="loss" stroke={TRAIN} dot={false} strokeWidth={2} connectNulls isAnimationActive={false} />
            <Line
              name="Validation loss"
              type="linear"
              dataKey="eval_loss"
              stroke={VALID}
              dot={{ r: 3 }}
              strokeWidth={2}
              connectNulls
              isAnimationActive={false}
            />
            {best && (
              <ReferenceLine
                x={best.step}
                stroke={BEST}
                strokeDasharray="4 3"
                label={{ value: best.final ? "best" : "best so far", position: "insideTopRight", fill: BEST, fontSize: 12 }}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <figcaption className="text-xs text-slate-600 dark:text-slate-400">
        {best
          ? `${best.final ? "Best checkpoint" : "Best so far"}: step ${best.step}, validation loss ${fmt(best.eval_loss)}. Falling loss alone doesn't show the model got better; the evaluation does.`
          : "The best checkpoint is marked once a validation loss is reported."}
      </figcaption>
      <details>
        <summary className="cursor-pointer text-xs">Loss values</summary>
        <table className="mt-1 text-xs" aria-label="Loss values">
          <thead>
            <tr>
              <th className="pr-4 text-left">Step</th>
              <th className="pr-4 text-left">Train loss</th>
              <th className="text-left">Validation loss</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.step}>
                <td className="pr-4">{r.step}</td>
                <td className="pr-4">{r.loss != null ? fmt(r.loss) : "—"}</td>
                <td>
                  {r.eval_loss != null ? fmt(r.eval_loss) : "—"}
                  {best?.step === r.step ? " (best)" : ""}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  );
}
