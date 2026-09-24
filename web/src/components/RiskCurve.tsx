"use client";
// Part B risk curve with the 0.5 alarm line and a cursor that follows the video's current time.
import Chart from "@/components/Chart";
import type { RiskPoint } from "@/lib/classes";

export default function RiskCurve({
  risk,
  duration,
  currentTime,
}: {
  risk: RiskPoint[];
  duration: number;
  currentTime: number;
}) {
  if (!risk.length) return <p className="text-sm text-zinc-500">No risk curve for this video.</p>;
  return (
    <Chart
      height={170}
      data={[
        {
          x: risk.map((p) => p[0]),
          y: risk.map((p) => p[1]),
          type: "scatter",
          mode: "lines",
          line: { color: "#f97316", width: 1.5 },
          fill: "tozeroy",
          fillcolor: "rgba(249,115,22,0.15)",
          name: "P(accident within 5 s)",
          hovertemplate: "%{x:.1f} s: %{y:.2f}<extra></extra>",
        },
      ]}
      layout={{
        margin: { l: 40, r: 8, t: 8, b: 32 },
        showlegend: false,
        xaxis: { range: [0, duration], title: { text: "time (s)" } },
        yaxis: { range: [0, 1], title: { text: "risk" } },
        shapes: [
          { type: "line", x0: 0, x1: duration, y0: 0.5, y1: 0.5, line: { color: "#ef4444", dash: "dot", width: 1 } },
          { type: "line", x0: currentTime, x1: currentTime, y0: 0, y1: 1, line: { color: "#fff", width: 1 } },
        ],
      }}
    />
  );
}
