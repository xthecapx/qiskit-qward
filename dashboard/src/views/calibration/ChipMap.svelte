<script lang="ts">
  import EChart from "../../lib/charts/EChart.svelte";
  import { palette, type EChartsType } from "../../lib/charts/echarts";
  import { calibrationNumber, scaleColor, schematic } from "../../lib/domain/calibration";
  import type { CalibrationState } from "../../lib/state/calibration.svelte";
  import { motion } from "../../lib/ui/motion.svelte";

  let { cal }: { cal: CalibrationState } = $props();
  let width = $state(600);

  const option = $derived.by(() => {
    const plotWidth = Math.max(width - 60, 100);
    const { nodes, links } = schematic(cal.current, plotWidth);
    const selected = cal.component;
    const pairMode = cal.mode === "pair";
    const many = nodes.length > 35;
    const data: object[] = nodes.map(({ id, x, y }) => {
      const item = pairMode ? undefined : cal.components.find((c) => c.id === id);
      const highlighted = selected?.qubits.includes(Number(id));
      return {
        id,
        name: `Q${id}`,
        x,
        y,
        symbolSize: many ? 14 : 22,
        value: item?.value,
        itemStyle: {
          color: scaleColor(item?.value, cal.domain, cal.mode),
          borderColor: highlighted ? palette.outline : "#fff",
          borderWidth: highlighted ? 4 : 1,
        },
      };
    });
    // Invisible anchors keep the schematic's scale stable as the selection changes.
    data.push(
      ...[
        [0, 0],
        [plotWidth, 300],
      ].map(([x, y], i) => ({ id: `anchor-${i}`, x, y, symbolSize: 0, label: { show: false }, tooltip: { show: false } })),
    );
    const edges = links.map(([id, qubits]) => {
      const item = cal.components.find((c) => c.id === id);
      return {
        source: String(qubits[0]),
        target: String(qubits[1]),
        componentId: id,
        lineStyle: {
          color: pairMode ? scaleColor(item?.value, cal.domain, cal.mode) : palette.edge,
          width: pairMode ? (id === selected?.id ? 7 : 4) : 2,
          opacity: 1,
          curveness: 0.06,
        },
      };
    });
    const mode = cal.mode;
    return {
      animationDurationUpdate: motion.reduced ? 0 : 300,
      tooltip: {
        trigger: "item",
        renderMode: "richText",
        formatter: (p: { dataType?: string; name: string; value?: number; data: { source?: string; target?: string } }) =>
          p.dataType === "edge"
            ? `Q${p.data.source} → Q${p.data.target}`
            : `${p.name}\n${calibrationNumber(mode, p.value)}`,
      },
      series: [
        {
          type: "graph",
          layout: "none",
          roam: true,
          data,
          links: edges,
          label: { show: true, position: "bottom", fontSize: 10 },
          emphasis: { focus: "adjacency" },
          left: 30,
          right: 30,
          top: 35,
          bottom: 35,
        },
      ],
    };
  });

  function ready(chart: EChartsType) {
    chart.on("click", (event) => {
      const data = event.data as { id?: string; componentId?: string };
      const id = cal.mode === "pair" ? data.componentId : data.id;
      if (id) cal.pickFromMap(id);
    });
  }
</script>

<div bind:clientWidth={width}>
  <EChart
    id="cal-map"
    label="Recorded qubit and gate calibration schematic"
    {option}
    replace
    onready={ready}
  />
</div>
