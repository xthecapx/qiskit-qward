<script lang="ts" module>
  import type { Job } from "../../lib/data/types";
  export interface LensPoint {
    job: Job;
    value: [number, number];
  }
</script>

<script lang="ts">
  import type { EChartsType } from "../../lib/charts/echarts";
  import { algorithmName, axes, circuitDescription, questions, studyName } from "../../lib/domain/catalog";
  import { format, jobTime, pct } from "../../lib/domain/format";
  import type { Explorer } from "../../lib/state/explorer.svelte";

  interface Props {
    explorer: Explorer;
    chart: EChartsType;
    point: LensPoint;
    showIntervals: boolean;
  }
  let { explorer, chart, point, showIntervals }: Props = $props();

  let lens: HTMLElement;
  let position = $state({ left: 0, top: 0, origin: "0 0" });

  const job = $derived(point.job);
  const interval = $derived(showIntervals ? job.successInterval : undefined);
  const study = $derived(studyName(job.family));
  const meanPosition = $derived.by(() => {
    if (!interval) return 50;
    const [low, high] = interval;
    const t = high > low ? (high - point.value[1]) / (high - low) : 0.5;
    return 100 * Math.max(0, Math.min(1, t));
  });

  // Place beside the point, flipping left near the edge and clamping inside the chart.
  $effect(() => {
    void point;
    const [x, y] = chart.convertToPixel({ gridIndex: 0 }, point.value) as number[];
    const width = lens.offsetWidth;
    const height = lens.offsetHeight;
    const left = x + width + 16 <= chart.getWidth() ? x + 16 : x - width - 16;
    const top = Math.max(0, Math.min(y - height / 2, chart.getHeight() - height));
    const bounded = Math.max(0, Math.min(left, chart.getWidth() - width));
    position = { left: bounded, top, origin: `${x - bounded}px ${y - top}px` };
  });
</script>

<aside
  bind:this={lens}
  id="interval-lens"
  aria-label="Job measurement details"
  style:left="{position.left}px"
  style:top="{position.top}px"
  style:transform-origin={position.origin}
  style:--mean-position="{meanPosition}%"
>
  <p class="eyebrow">{interval ? "99% confidence interval" : questions[explorer.question].label}</p>
  {#if interval}
    <div class="lens-detail">
      <span class="lens-whisker" aria-hidden="true"><i></i></span>
      <dl>
        <div><dt>Upper bound</dt><dd>{pct(interval[1])}</dd></div>
        <div class="lens-estimate"><dt>Mean success</dt><dd>{pct(point.value[1])}</dd></div>
        <div><dt>Lower bound</dt><dd>{pct(interval[0])}</dd></div>
      </dl>
    </div>
  {:else}
    <strong id="lens-value">{format(explorer.question, point.value[1])}</strong>
  {/if}
  <dl class="lens-context">
    {#if study && study !== job.config}
      <div><dt>Study</dt><dd>{study}</dd></div>
    {/if}
    <div><dt>Circuit</dt><dd>{circuitDescription(job) || algorithmName(job.algorithm)}</dd></div>
    <div>
      <dt>{explorer.axis === "width" ? "Total logical qubits" : axes[explorer.axis]}</dt>
      <dd>{point.value[0]}</dd>
    </div>
    <div><dt>Repetition</dt><dd>{job.rep}</dd></div>
    <div><dt>Measured at</dt><dd>{jobTime(job)}</dd></div>
  </dl>
</aside>
