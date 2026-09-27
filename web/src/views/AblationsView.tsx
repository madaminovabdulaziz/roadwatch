"use client";
// Ablations and error analysis from public/data/ablations.json (scripts/write_ablations.py). Each
// ablation says where its numbers come from: measured now on the committed runs and track caches, or
// recorded in the SPEC decision that measured code which no longer exists.
import { Card, Section } from "@/components/ui";
import { className, classColor } from "@/lib/classes";
import { useJson } from "@/lib/data";

interface Ablation {
  title: string;
  metric: string;
  rows: { variant: string; value: number }[];
  takeaway: string;
  source: string;
}
interface Ablations {
  ablations: Ablation[];
  errors: {
    match_tiou: number;
    confusion: { labelled: string[]; predicted: string[]; counts: number[][] };
    boundaries: Record<
      string,
      { n: number; start_median_sec: number; end_median_sec: number }
    >;
  };
}

const label = (c: string) => (c === "none" ? "none" : className(c));
const fmt = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(3));
const signed = (v: number) => `${v > 0 ? "+" : ""}${v.toFixed(2)} s`;

export default function AblationsView() {
  const data = useJson<Ablations>("ablations.json");
  if (data.state !== "ok") return null;
  const { ablations, errors } = data.data;
  const c = errors.confusion;
  const peak = Math.max(1, ...c.counts.flat());
  return (
    <>
      <Section
        title="Ablations"
        changed="each of these decided a setting that ships: the numbers below are why the rule is the way it is."
      >
        <div className="ablation-grid">
          {ablations.map((a) => {
            const top = Math.max(...a.rows.map((r) => Math.abs(r.value)), 1e-9);
            return (
              <Card key={a.title}>
                <h3 className="ablation-title">{a.title}</h3>
                <p className="meta">{a.metric}</p>
                <div className="ablation-rows">
                  {a.rows.map((r) => (
                    <div key={r.variant} className="ablation-row">
                      <span className="ablation-variant">{r.variant}</span>
                      <span className="ablation-bar" aria-hidden="true">
                        <span
                          style={{
                            width: `${(100 * Math.abs(r.value)) / top}%`,
                          }}
                        />
                      </span>
                      <span className="ablation-value">{fmt(r.value)}</span>
                    </div>
                  ))}
                </div>
                <p className="ablation-takeaway">{a.takeaway}</p>
                <p className="meta">{a.source}</p>
              </Card>
            );
          })}
        </div>
      </Section>

      <Section
        title="Error analysis on our labels"
        changed="almost every error is a missed or an extra event, not one class read as another. Timing is tight where there are enough events (failure_to_yield within 0.05 s); the rare classes rest on one event each, so no timing correction was fitted."
      >
        <div className="table-wrap">
          <p className="table-note">
            Every predicted and labelled event matched regardless of class at
            tIoU ≥ {errors.match_tiou}. Rows: what was labelled; columns: what
            we predicted. &quot;none&quot; = missed (last column) or fired on
            unlabelled time (last row).
          </p>
          <table className="data-table confusion-table">
            <thead>
              <tr>
                <th>labelled \ predicted</th>
                {c.predicted.map((p) => (
                  <th key={p}>{label(p)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {c.labelled.map((g, i) => (
                <tr key={g}>
                  <td>
                    {g !== "none" && (
                      <span
                        className="class-swatch"
                        style={{ background: classColor(g) }}
                      />
                    )}
                    {label(g)}
                  </td>
                  {c.counts[i].map((n, j) => (
                    <td
                      key={c.predicted[j]}
                      className={g === c.predicted[j] ? "strong" : undefined}
                      style={{
                        background: n
                          ? `rgb(127 127 127 / ${(0.08 + (0.42 * n) / peak).toFixed(2)})`
                          : undefined,
                      }}
                    >
                      {n || ""}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="table-wrap">
          <p className="table-note">
            Median timing error of matched events of the same class (predicted
            minus labelled).
          </p>
          <table className="data-table">
            <thead>
              <tr>
                <th>Class</th>
                <th>Matched</th>
                <th>Start</th>
                <th>End</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(errors.boundaries).map(([cls, b]) => (
                <tr key={cls}>
                  <td>
                    <span
                      className="class-swatch"
                      style={{ background: classColor(cls) }}
                    />
                    {className(cls)}
                  </td>
                  <td>{b.n}</td>
                  <td>{signed(b.start_median_sec)}</td>
                  <td>{signed(b.end_median_sec)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </>
  );
}
