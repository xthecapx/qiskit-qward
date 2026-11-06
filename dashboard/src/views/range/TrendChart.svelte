<script lang="ts">
  import EChart from "../../lib/charts/EChart.svelte";
  import { palette, type EChartsType } from "../../lib/charts/echarts";
  import type { Job } from "../../lib/data/types";
  import { axes, questions } from "../../lib/domain/catalog";
  import { unique } from "../../lib/domain/format";
  import { sweepAxis } from "../../lib/domain/sweep";
  import type { Explorer } from "../../lib/state/explorer.svelte";
  import IntervalLens, { type LensPoint } from "./IntervalLens.svelte";

  let { explorer }: { explorer: Explorer } = $props();

  let chart = $state<EChartsType>();
  let hovered = $state<LensPoint | null>(null);

  const showIntervals = $derived(explorer.question === "success");

  const option = $derived.by(() => {
    const { jobs, configs, axis, question: key, threshold, summary } = explorer;
    const dots = jobs
      .filter((j) => j[key] != null)
      .map((j) => ({ value: [j[axis], j[key] as number], job: j }));
    const boundary = summary.boundary;
    const shade =
      boundary != null && axis === sweepAxis(configs)
        ? [[{ xAxis: configs[boundary - 1][0][axis] }, { xAxis: configs[boundary][0][axis] }]]
        : [];
    const axisValues = unique(jobs.map((j) => j[axis]));
    return {
      animationDurationUpdate: 350,
      grid: { left: 58, right: 18, top: 36, bottom: 52 },
      tooltip: { show: false },
      xAxis: {
        type: "value",
        min: axisValues.length ? Math.max(0, Math.min(...axisValues) - 1) : 0,
        max: axisValues.length ? Math.max(...axisValues) + 1 : 1,
        minInterval: 1,
        name: axes[axis],
        nameLocation: "middle",
        nameGap: 29,
      },
      yAxis: {
        type: "value",
        name: questions[key].label,
        nameLocation: "end",
        nameTextStyle: { align: "left" },
        min: 0,
        max: 1,
        interval: showIntervals ? 0.1 : 0.25,
        axisLabel: {
          formatter: (v: number) => (key === "success" ? Math.round(v * 100) + "%" : String(v)),
        },
      },
      series: [
        {
          name: questions[key].label,
          type: "scatter",
          symbolSize: showIntervals ? 7 : 11,
          z: 3,
          itemStyle: { color: palette.teal },
          emphasis: { scale: false, itemStyle: { color: palette.accent, opacity: 1 } },
          data: dots,
          markArea: { silent: true, itemStyle: { color: palette.band }, data: shade },
          markLine: {
            silent: true,
            symbol: "none",
            label: { show: false },
            lineStyle: { color: palette.threshold, type: "dashed" },
            data: threshold == null ? [] : [{ yAxis: threshold }],
          },
        },
      ],
    };
  });

  // Any change of plotted data, or leaving the view, hides a stale lens.
  $effect(() => {
    void option;
    void explorer.view;
    hovered = null;
  });

  function ready(instance: EChartsType) {
    chart = instance;
    type Event = { data: { value: [number, number]; job: Job } };
    instance.on("mouseover", { seriesType: "scatter" }, (event) => {
      const { data } = event as unknown as Event;
      hovered = { job: data.job, value: data.value };
    });
    instance.on("mouseout", { seriesType: "scatter" }, () => {
      hovered = null;
    });
    instance.on("click", { seriesType: "scatter" }, (event) => {
      const { data } = event as unknown as Event;
      void explorer.openJob(data.job, document.getElementById("plot-title"));
    });
  }
</script>

<div class="plot-stage">
  <EChart
    id="trend"
    class="plot"
    label="Selected metric by circuit setting"
    {option}
    replace
    onready={ready}
  />
  {#if hovered && chart}
    <IntervalLens {explorer} {chart} point={hovered} {showIntervals} />
  {/if}
</div>
