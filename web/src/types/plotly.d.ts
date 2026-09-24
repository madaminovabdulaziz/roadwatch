// The basic Plotly bundle (scatter, bar, pie) and react-plotly's factory ship without types.
declare module "plotly.js-basic-dist-min" {
  const Plotly: object;
  export default Plotly;
}

declare module "react-plotly.js/factory" {
  import type { ComponentType } from "react";
  import type { PlotParams } from "react-plotly.js";
  export default function createPlotlyComponent(plotly: object): ComponentType<PlotParams>;
}
