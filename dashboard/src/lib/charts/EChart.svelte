<script lang="ts">
  import { onMount } from "svelte";
  import { motion } from "../ui/motion.svelte";
  import { echarts, theme, type EChartsCoreOption, type EChartsType } from "./echarts";

  interface Props {
    /** Full option; replaced (`notMerge`) when `replace` is true, merged otherwise. */
    option: EChartsCoreOption;
    replace?: boolean;
    id?: string;
    class?: string;
    style?: string;
    label: string;
    /** Called once with the instance so views can bind events or convert coordinates. */
    onready?: (chart: EChartsType) => void;
  }

  let { option, replace = false, id, class: className, style, label, onready }: Props = $props();

  let node: HTMLDivElement;
  let chart = $state<EChartsType>();

  onMount(() => {
    const instance = echarts.init(node, null, { renderer: "svg" });
    chart = instance;
    onready?.(instance);
    const observer = new ResizeObserver(() => instance.resize());
    observer.observe(node);
    return () => {
      observer.disconnect();
      instance.dispose();
    };
  });

  $effect(() => {
    if (!chart) return;
    chart.setOption(
      { ...theme, animation: !motion.reduced, ...option },
      { notMerge: replace, lazyUpdate: false },
    );
    chart.resize();
  });
</script>

<div bind:this={node} {id} class={className} {style} role="img" aria-label={label}></div>
