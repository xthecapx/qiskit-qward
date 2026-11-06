<script lang="ts">
  import EChart from "../../lib/charts/EChart.svelte";
  import { palette } from "../../lib/charts/echarts";
  import { pct } from "../../lib/domain/format";
  import type { HistogramModel } from "../../lib/domain/histogram";

  let { model }: { model: HistogramModel } = $props();

  const height = $derived(Math.max(240, model.display.length * 26 + 65));
  const option = $derived({
    animationDurationUpdate: 350,
    grid: { left: 10, right: 20, top: 12, bottom: 36, containLabel: true },
    tooltip: {
      trigger: "axis",
      renderMode: "richText",
      valueFormatter: (v: number | number[]) => (Array.isArray(v) ? pct(v[0]) : pct(v)),
    },
    xAxis: {
      type: "value",
      min: 0,
      max: 1,
      axisLabel: { formatter: (v: number) => Math.round(v * 100) + "%" },
    },
    yAxis: {
      type: "category",
      inverse: true,
      data: model.display,
      axisLabel: { interval: 0, fontFamily: "monospace", fontSize: 11 },
    },
    series: [
      {
        name: "Observed",
        type: "bar",
        barMaxWidth: 13,
        data: model.display.map((label, i) => ({
          name: label,
          value: model.values[i],
          itemStyle: {
            color: model.blind ? palette.blue : model.allowed.has(label) ? palette.teal : palette.muted,
          },
        })),
      },
      {
        name: model.coarse ? "Task target" : "Ideal",
        type: "scatter",
        symbol: "diamond",
        symbolSize: 10,
        itemStyle: { color: palette.reference },
        data:
          model.hasReference && !model.blind
            ? model.markers.map((v, i) => [v, model.display[i]])
            : [],
      },
    ],
  });
</script>

<EChart
  id="hist"
  label="Observed outcome probabilities with reference markers"
  style="height: {height}px"
  {option}
  replace
/>
