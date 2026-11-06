<script lang="ts">
  import { algorithmName } from "../../lib/domain/catalog";
  import {
    calibrationNumber,
    componentLabel,
    isDuration,
    isGate,
    modeKeys,
    modes,
  } from "../../lib/domain/calibration";
  import { calibrationStamp, plural } from "../../lib/domain/format";
  import type { Explorer } from "../../lib/state/explorer.svelte";
  import Select from "../../lib/ui/Select.svelte";
  import ChipMap from "./ChipMap.svelte";
  import ComponentDetail from "./ComponentDetail.svelte";
  import HistoryChart from "./HistoryChart.svelte";

  let { explorer }: { explorer: Explorer } = $props();
  const cal = $derived(explorer.calibration);
  const job = $derived(cal.job);
  const duration = $derived(isDuration(cal.mode));

  const context = $derived.by(() => {
    if (cal.status === "ready" && !cal.available) {
      return "Detailed preparation calibration is unavailable for this selection. The histogram and its metrics remain available.";
    }
    if (!job) {
      return "Explore the hardware observed in our experiments. Dates refer to saved preparation calibration snapshots; coverage varies between snapshots.";
    }
    return `${algorithmName(job.algorithm)} · ${job.config} · run ${job.rep}. ${
      cal.expanded
        ? "Exploring chip history; return to the histogram to retain this job."
        : "Showing this circuit’s preparation snapshot and used hardware."
    } Preparation: ${calibrationStamp(cal.jobRecord?.at)} UTC. Execution calibration: ${calibrationStamp(job.executionCalibration?.at)} UTC.`;
  });
  const known = $derived(cal.history.filter((v) => v != null).length);
</script>

<div id="calibration-view">
  <div class="section-head">
    <div>
      <p class="eyebrow">Hardware context</p>
      <h2 id="calibration-title" tabindex="-1">
        {job && !cal.expanded ? "Calibration for this run" : "Chip calibration history"}
      </h2>
    </div>
    <button type="button" id="close-calibration" class="view-button" onclick={() => cal.close()}
      >← Back to {cal.origin === "job" ? "job histogram" : "operating range"}</button
    >
  </div>
  <p id="calibration-context" class="note">{context}</p>

  {#if cal.status === "loading" || cal.status === "idle"}
    <p role="status" class="note">Loading calibration snapshots…</p>
  {:else if cal.status === "error"}
    <p role="alert" class="error">{cal.error}</p>
  {:else}
    <div class="calibration-controls">
      <Select
        id="cal-chip"
        label="Chip"
        disabled={Boolean(job)}
        value={cal.chip}
        options={cal.chips}
        onchange={(v) => cal.setChip(v)}
      />
      <Select
        id="cal-date"
        label="Calibration snapshot (UTC)"
        disabled={cal.scoped}
        value={cal.date}
        options={cal.scoped ? [cal.date] : cal.dates}
        format={calibrationStamp}
        onchange={(v) => (cal.dateChoice = v)}
      />
      <Select
        id="cal-mode"
        label="Explore"
        value={cal.mode}
        options={modeKeys}
        format={(m) => modes[m].option}
        onchange={(v) => cal.setMode(v)}
      />
      {#if isGate(cal.mode) && cal.available}
        <Select
          id="cal-gate"
          label="Operation"
          value={cal.gate}
          options={cal.gates}
          onchange={(v) => (cal.gateChoice = v)}
        />
      {/if}
    </div>
    {#if !cal.available}
      <p id="cal-empty" role="status">
        No detailed calibration snapshots are available. Export calibrations.csv and rebuild to explore
        this chip.
      </p>
    {:else}
      <div id="cal-content">
        <div class="calibration-layout">
          <div class="calibration-map-panel">
            <div class="section-head">
              <h3>Where on the chip?</h3>
              <span id="cal-coverage" class="note"
                >{cal.components.filter((c) => c.value != null).length} / {cal.components.length} components
                with values</span
              >
            </div>
            <p class="note">
              Observed connectivity · schematic, not physical chip geometry. Select a qubit or
              connection. Drag to pan; scroll to zoom.
            </p>
            <ChipMap {cal} />
            <div class="cal-scale">
              <span>{calibrationNumber(cal.mode, duration ? cal.domain[1] : cal.domain[0])}</span><span
                class="cal-gradient"
                aria-hidden="true"
              ></span><span>{calibrationNumber(cal.mode, duration ? cal.domain[0] : cal.domain[1])}</span><span
                class="note">Gray: unavailable</span
              >
            </div>
            <p class="note">
              Fixed scale across recorded dates · {duration ? "longer is better" : "lower is better"}.
              Colors describe relative values, not a pass/fail threshold.
            </p>
          </div>
          <ComponentDetail {cal} />
        </div>
        <div class="calibration-history">
          <div class="section-head">
            <div>
              <p class="eyebrow">Through time</p>
              <h3 id="cal-history-title">
                {cal.component ? componentLabel(cal.component) : "Component"} · {modes[cal.mode].title}
              </h3>
            </div>
            {#if job}
              <button type="button" id="cal-expand-history" class="view-button" onclick={() => cal.toggleExpanded()}
                >{cal.expanded ? "Return to this run’s hardware" : "Explore this component’s history"}</button
              >
            {/if}
          </div>
          <p class="note">
            {plural(known, "recorded snapshot")} with a value. {job
              ? "History includes this component wherever it was observed on the same chip. "
              : ""}Snapshots are equally spaced (UTC); dots are observations and empty dates have no
            comparable value. Select a dot to inspect its snapshot.
          </p>
          <HistoryChart {cal} />
        </div>
        <details>
          <summary>Data sources and interpretation</summary>
          <p class="note">
            The map and component history use saved preparation snapshots. Only components used by the
            collected circuits are available; this is not a complete chip survey. Dynamic control-flow
            bodies were not expanded in the saved snapshots. Repeated references to the same value at the
            same calibration timestamp are counted once. Conflicting values are marked unavailable rather
            than averaged. Gate estimates, readout errors and coherence times describe different
            properties; they do not explain an individual histogram outcome on their own. A calibration
            timestamp identifies the backend snapshot, not necessarily the measurement time of each
            property.
          </p>
          <p class="note">
            {job
              ? `Execution source: ${job.executionCalibration?.source || "unavailable"}. Recorded circuit summary: ${calibrationNumber(cal.mode, job.executionCalibration?.values[cal.mode])}. Gate/readout means are instruction-weighted; coherence means are over used qubits. These summaries are not per-component execution measurements.`
              : "Open Calibration for this job from a histogram to compare its execution-time summary and preparation details."}
          </p>
          <p class="note">
            History shows recorded observations without interpolation or confidence intervals. No
            per-component calibration uncertainty was saved.
          </p>
        </details>
      </div>
    {/if}
  {/if}
</div>
