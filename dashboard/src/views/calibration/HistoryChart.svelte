<script lang="ts">
  import EChart from "../../lib/charts/EChart.svelte";
  import { palette, type EChartsType } from "../../lib/charts/echarts";
  import { calibrationNumber, isDuration } from "../../lib/domain/calibration";
  import { calibrationStamp } from "../../lib/domain/format";
  import type { CalibrationState } from "../../lib/state/calibration.svelte";

  let { cal }: { cal: CalibrationState } = $props();

  const option = $derived.by(() => {
    const { dates, history, date, mode } = cal;
    const duration = isDuration(mode);
    return {
      grid: { left: 75, right: 25, top: 25, bottom: 60 },
      tooltip: {
        trigger: "axis",
        renderMode: "richText",
        formatter: (params: { dataIndex: number; value: number | null }[]) =>
          `${calibrationStamp(dates[params[0].dataIndex])} UTC\n${calibrationNumber(mode, params[0].value)}`,
      },
      xAxis: {
        type: "category",
        data: dates.map(calibrationStamp),
        axisLabel: { fontSize: 10, rotate: 15 },
      },
      yAxis: {
        type: "value",
        min: 0,
        name: duration ? "μs" : "%",
        axisLabel: { formatter: (v: number) => (duration ? String(v) : String(+(v * 100).toFixed(3))) },
      },
      series: [
        {
          type: "scatter",
          data: history,
          symbolSize: (_: unknown, p: { dataIndex: number }) => (dates[p.dataIndex] === date ? 15 : 9),
          itemStyle: { color: palette.teal },
        },
      ],
    };
  });

  function ready(chart: EChartsType) {
    chart.on("click", (event) => {
      const at = cal.dates[event.dataIndex];
      if (at) cal.pickDate(at);
    });
  }
</script>

<EChart
  id="cal-history"
  label="Historical calibration values for the selected component"
  {option}
  replace
  onready={ready}
/>
