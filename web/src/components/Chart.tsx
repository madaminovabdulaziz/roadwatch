"use client";
// Plotly on the client only (no SSR), with the small "basic" bundle and one dark theme for every chart.
import dynamic from "next/dynamic";
import type { Data, Layout } from "plotly.js";

const Plot = dynamic(
  async () => {
    const Plotly = (await import("plotly.js-basic-dist-min")).default;
    const createPlotlyComponent = (await import("react-plotly.js/factory"))
      .default;
    return createPlotlyComponent(Plotly);
  },
  {
    ssr: false,
    loading: () => <div className="chart-loading" />,
  },
);

const AXIS = {
  gridcolor: "rgba(239,238,232,0.07)",
  zerolinecolor: "rgba(239,238,232,0.16)",
  linecolor: "rgba(239,238,232,0.16)",
  color: "#a3a79e",
};
// Series without a colour of their own (lighting per video, speed per lane): ink first, then greys and
// a cool blue; the orange accent is kept for accident risk.
const COLORWAY = [
  "#efeee8",
  "#8fb3c9",
  "#a3a79e",
  "#d9c7a3",
  "#6d726b",
  "#c9a0b8",
];

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
    <div className="chart-frame">
      <Plot
        data={data}
        layout={{
          height,
          autosize: true,
          margin: { l: 48, r: 12, t: 12, b: 40 },
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { color: "#a3a79e", size: 12, family: "var(--sans)" },
          colorway: COLORWAY,
          hoverlabel: {
            bgcolor: "#111312",
            bordercolor: "rgba(239,238,232,0.2)",
            font: { color: "#efeee8", family: "var(--sans)" },
          },
          // Above the plot: below it the legend collides with the x-axis title.
          legend: {
            orientation: "h",
            x: 0,
            y: 1.02,
            yanchor: "bottom",
            font: { color: "#a3a79e", size: 12 },
          },
          ...layout,
          xaxis: { ...AXIS, ...layout.xaxis },
          yaxis: { ...AXIS, ...layout.yaxis },
        }}
        config={{ displayModeBar: false, responsive: true }}
        useResizeHandler
        style={{ width: "100%" }}
      />
    </div>
  );
}
