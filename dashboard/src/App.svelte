<script lang="ts">
  import { onMount } from "svelte";
  import { ECHARTS_VERSION } from "./lib/charts/echarts";
  import { loadIndex } from "./lib/data/load";
  import { unique } from "./lib/domain/format";
  import { Explorer } from "./lib/state/explorer.svelte";
  import { formatHash, parseHash } from "./lib/state/url";
  import CalibrationView from "./views/calibration/CalibrationView.svelte";
  import JobView from "./views/job/JobView.svelte";
  import RangeView from "./views/range/RangeView.svelte";

  import Tabs from "./lib/ui/Tabs.svelte";
  import CompareDecay from "./views/decay/CompareDecay.svelte";

  let section = $state("range");
  let decay = $state({ matrix: "similarity", pair: "QFT/PHASE-ESTIMATION", resource: "a.ComplexityMetrics.gate_based_metrics.two_qubit_count", outcome: "success" });
  let reports = $state<Record<string,string>>({});
  function readChoices() {
    const url = parseHash(location.hash);
    section = url.section === "decay" ? "decay" : "range";
    decay = {matrix: url.matrix === "direction" ? "direction" : "similarity", pair: url.pair || "QFT/PHASE-ESTIMATION", resource: url.resource || "a.ComplexityMetrics.gate_based_metrics.two_qubit_count", outcome: url.outcome || "success"};
  }

  let explorer = $state<Explorer>();
  let error = $state("");

  onMount(() => {
    readChoices();
    loadIndex()
      .then((data) => { reports = data.decayReports ?? {}; explorer = new Explorer(data, parseHash(location.hash)); })
      .catch((reason) => {
        console.error(reason);
        error =
          "The experiment data could not be loaded. Build the dashboard (npm run build) and serve its dist/ directory over HTTP.";
      });
    const onHash = () => {
      readChoices();
      if (explorer) {
        explorer.apply(parseHash(location.hash));
      }
    };
    addEventListener("hashchange", onHash);
    return () => removeEventListener("hashchange", onHash);
  });

  // Mirror state into the URL without adding history entries for every control change.
  $effect(() => {
    if (!explorer) return;
    const hash = formatHash({...explorer.toUrl(), ...(section === "decay" ? {section, ...decay} : {})});
    if (hash !== location.hash) history.replaceState(null, "", hash || location.pathname);
  });

  const view = $derived(explorer?.view ?? "range");
</script>

<nav>
  <div class="wrap nav-inner">
    <strong><span class="mark" aria-hidden="true">∿</span> QWARD</strong><span>Hardware experiments</span><a
      href="https://xthecapx.github.io/qiskit-qward/">Documentation ↗</a
    >
  </div>
</nav>
<main class="wrap">
  <header>
    <p class="eyebrow">Experiment explorer</p>
    <h1>QPU Benchmark</h1>
    {#if section === "range"}
    <p class="intro">
      Choose an algorithm and a question to see which circuit sizes met your target on this chip.
      Select a point to explore that job’s measurement outcomes.
    </p>
    <button
      type="button"
      id="open-calibration"
      class="view-button calibration-entry"
      disabled={!explorer}
      onclick={(event) => explorer?.calibration.open(null, event.currentTarget)}
      >Chip calibration history <span aria-hidden="true">↗</span></button
    >
    {:else}<p class="intro">Compare how circuit quality changes as two algorithms grow on the same chip.</p>{/if}
  </header>
  {#if error}
    <div id="error" role="alert">{error}</div>
  {/if}
  {#if explorer}
    <Tabs tabs={[{id:"range",label:"Operating range"},{id:"decay",label:"Compare decay"}]} label="Experiment sections" selected={section} onselect={(s) => section=s}>
    {#snippet panel(id)}
    {#if id === "decay" && explorer}
      <CompareDecay {explorer} {reports} choices={decay} onchange={(v) => decay=v} />
    {:else if explorer}
    <section class="surface" aria-label="Experiment explorer">
      <div class="explorer-path" id="explorer-path" aria-label="Explorer location">
        {#if view === "range"}
          <span aria-current="page">Operating range</span>
        {:else}
          <button
            type="button"
            id="range-return"
            onclick={() => explorer!.returnToRange()}
            >Operating range</button
          >
          {#if view === "job"}
            <span><span aria-hidden="true">{" / "}</span><span aria-current="page">Job histogram</span></span>
          {:else}
            {#if explorer.calibration.origin === "job"}
              <span
                ><span aria-hidden="true">{" / "}</span><button
                  type="button"
                  id="histogram-return"
                  onclick={() => explorer!.calibration.close()}>Job histogram</button
                ></span
              >
            {/if}
            <span aria-current="page">{" / Calibration"}</span>
          {/if}
        {/if}
      </div>
      <!-- Range and job views stay mounted while hidden so returning restores focus targets,
           open disclosures and scroll position exactly. -->
      <div hidden={view !== "range"}><RangeView {explorer} /></div>
      {#if explorer.selectedJob}
        <div hidden={view !== "job"}><JobView {explorer} /></div>
      {/if}
      {#if view === "calibration"}
        <CalibrationView {explorer} />
      {/if}
    </section>
    {/if}
    {/snippet}
    </Tabs>
  {:else if !error}
    <p role="status" class="note">Reading the exported observations…</p>
  {/if}
  <footer>
    <span id="source"
      >{explorer
        ? `Exported observations · ${explorer.jobsAll.length} jobs · ${unique(explorer.jobsAll.map((j) => j.algorithm)).length} algorithms · ECharts ${ECHARTS_VERSION}`
        : "Reading the exported snapshot…"}</span
    >
  </footer>
</main>
