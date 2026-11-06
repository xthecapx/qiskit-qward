<script lang="ts">
  import type { QuestionKey } from "../../lib/data/types";
  import { algorithmName, axes, questionKeys, questions } from "../../lib/domain/catalog";
  import { comparator, format, plural } from "../../lib/domain/format";
  import type { Explorer } from "../../lib/state/explorer.svelte";
  import Icon from "../../lib/ui/Icon.svelte";
  import Metadata from "../../lib/ui/Metadata.svelte";
  import Select from "../../lib/ui/Select.svelte";
  import Findings from "./Findings.svelte";
  import PlotSettings from "./PlotSettings.svelte";
  import SweepTable from "./SweepTable.svelte";
  import TrendChart from "./TrendChart.svelte";

  let { explorer }: { explorer: Explorer } = $props();
  let settingsOpen = $state(false);
  let adjustButton: HTMLButtonElement;

  const key = $derived(explorer.question);
  const metric = $derived(questions[key].label);
</script>

<div id="plot-view">
  <div class="plot-toolbar" aria-label="Explore measurements">
    <Select
      id="algorithm"
      label="Algorithm"
      value={explorer.algorithm}
      options={explorer.algorithms}
      format={algorithmName}
      onchange={(v) => explorer.setAlgorithm(v)}
    />
    <Select
      id="question"
      label="Question"
      value={key}
      options={questionKeys}
      format={(k: QuestionKey) => questions[k].prompt}
      onchange={(v) => explorer.setQuestion(v)}
    />
    <button
      bind:this={adjustButton}
      type="button"
      id="adjust-plot"
      class="view-button"
      aria-haspopup="dialog"
      aria-controls="plot-settings"
      onclick={() => (settingsOpen = true)}><Icon name="sliders" size={17} />Adjust plot</button
    >
  </div>
  <div class="plot-caption">
    <h2 id="plot-title" tabindex="-1">{metric} by {axes[explorer.axis].toLowerCase()}</h2>
    <Metadata
      id="scope-caption"
      label="Plot context"
      entries={[
        ["chip", "Chip", explorer.backend],
        ["date", "Periods", explorer.periodLabel],
        ["group", "Experiments", explorer.family || "All experiments"],
      ]}
    />
    <p class="note">
      <span id="coverage">{plural(explorer.configs.length, "experiment")} · {explorer.jobs.length} jobs</span
      ><span
        >{explorer.threshold == null
          ? " · No target set"
          : ` · Target ${comparator(key)} ${format(key, explorer.threshold)}`}</span
      >
    </p>
  </div>
  <PlotSettings
    {explorer}
    open={settingsOpen}
    onclose={() => {
      settingsOpen = false;
      adjustButton.focus();
    }}
  />
  <div class="analysis">
    <div>
      <TrendChart {explorer} />
      <p class="note">
        {key === "success"
          ? "Each point is one job’s accepted-shot fraction. Hover to magnify its mean and 99% confidence bounds."
          : "Each point is one job."} Showing {explorer.collections} collection(s); dates are in UTC. Values
        are individual jobs, not averages. Select a point or a job value in the table to open its histogram.
      </p>
    </div>
    <Findings {explorer} />
  </div>
  <SweepTable {explorer} />
</div>
