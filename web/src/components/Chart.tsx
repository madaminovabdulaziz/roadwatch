"use client";
// Plotly on the client only (no SSR), with the small "basic" bundle and one dark theme for every chart.
import dynamic from "next/dynamic";
import type { Data, Layout } from "plotly.js";

const Plot = dynamic(
  async () => {
    const Plotly = (await import("plotly.js-basic-dist-min")).default;
    const createPlotlyComponent = (await import("react-plotly.js/factory")).default;
    return createPlotlyComponent(Plotly);
  },
  { ssr: false, loading: () => <div className="h-64 animate-pulse rounded-lg bg-white/5" /> },
);

const AXIS = { gridcolor: "rgba(255,255,255,0.08)", zerolinecolor: "rgba(255,255,255,0.15)", color: "#a1a1aa" };

export default function Chart({
  data,
  layout = {},
  height = 280,
}: {
  data: Data[];
  layout?: Partial<Layout>;
  height?: number;
}) {
  return (
    <Plot
      data={data}
      layout={{
        height,
        autosize: true,
        margin: { l: 48, r: 12, t: 12, b: 40 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#d4d4d8", size: 12 },
        legend: { orientation: "h", y: -0.2 },
        ...layout,
        xaxis: { ...AXIS, ...layout.xaxis },
        yaxis: { ...AXIS, ...layout.yaxis },
      }}
      config={{ displayModeBar: false, responsive: true }}
      useResizeHandler
      style={{ width: "100%" }}
    />
  );
}
