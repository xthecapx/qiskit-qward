/** Tree-shaken ECharts: register only the charts and components the dashboard draws. */
import { LineChart, BarChart, GraphChart, ScatterChart } from "echarts/charts";
import {
  GridComponent,
  MarkAreaComponent,
  MarkLineComponent,
  TooltipComponent,
} from "echarts/components";
import * as echarts from "echarts/core";
import { SVGRenderer } from "echarts/renderers";

echarts.use([
  LineChart,
  BarChart,
  GraphChart,
  ScatterChart,
  GridComponent,
  MarkAreaComponent,
  MarkLineComponent,
  TooltipComponent,
  SVGRenderer,
]);

export { echarts };
export type { EChartsCoreOption, EChartsType } from "echarts/core";

export const ECHARTS_VERSION = echarts.version;

export const palette = {
  ink: "#34554e",
  teal: "#087363",
  accent: "#c44d16",
  muted: "#a2b6b2",
  blue: "#387b9a",
  reference: "#bd6617",
  threshold: "#9a5b12",
  band: "rgba(31,134,112,.10)",
  edge: "#c7d5ce",
  outline: "#142f2a",
};

export const theme = {
  textStyle: { fontFamily: "system-ui, sans-serif", color: palette.ink },
};
