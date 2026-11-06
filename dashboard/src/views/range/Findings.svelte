<script lang="ts">
  import { questions, tasks } from "../../lib/domain/catalog";
  import { comparator, format } from "../../lib/domain/format";
  import { acceptedSummary } from "../../lib/domain/histogram";
  import { passes } from "../../lib/domain/sweep";
  import type { Explorer } from "../../lib/state/explorer.svelte";
  import Metadata from "../../lib/ui/Metadata.svelte";
  import Tabs from "../../lib/ui/Tabs.svelte";
  import { jobMetadata } from "../job/metadata";

  let { explorer }: { explorer: Explorer } = $props();
  let tab = $state("overview");

  const key = $derived(explorer.question);
  const threshold = $derived(explorer.threshold);
  const job = $derived(explorer.selectedJob);
  const decision = $derived.by(() => {
    const value = job?.[key];
    if (value == null || threshold == null) return "";
    return ` · ${passes(key, value, threshold) ? "meets" : "misses"} the ${format(key, threshold)} target`;
  });
  const description = $derived(
    job && explorer.selectedReference
      ? [tasks[job.algorithm], job.description, acceptedSummary(explorer.selectedReference)]
          .filter(Boolean)
          .join(" ")
      : "",
  );
</script>

<article class="finding" aria-label="Experiment overview">
  <Tabs
    label="Experiment details"
    tabs={[
      { id: "overview", label: "Overview" },
      { id: "metric", label: "Metric" },
      { id: "job", label: "Selected job" },
    ]}
    selected={tab}
    onselect={(id) => (tab = id)}
  >
    {#snippet panel(id)}
      {#if id === "overview"}
        <h3 id="headline">{explorer.summary.title}</h3>
        <p id="explain">{explorer.summary.detail}</p>
      {:else if id === "metric"}
        <h3 id="metric">
          {questions[key].label}{threshold == null ? "" : ` ${comparator(key)} ${format(key, threshold)}`}
        </h3>
        <p>{questions[key].meaning}</p>
      {:else if job}
        <h3 id="selected-label">{job.config} · job {job.rep}</h3>
        <b>{questions[key].label} {format(key, job[key])}{decision}</b>
        <p>{description}</p>
        <Metadata id="selected-context" label="Selected run details" entries={jobMetadata(job)} />
      {/if}
    {/snippet}
  </Tabs>
</article>
