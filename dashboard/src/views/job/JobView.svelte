<script lang="ts">
  import { algorithmName } from "../../lib/domain/catalog";
  import { jobTime, pct, score } from "../../lib/domain/format";
  import { histogram } from "../../lib/domain/histogram";
  import { jobKey } from "../../lib/domain/scope";
  import { BIN_OPTIONS, type Explorer } from "../../lib/state/explorer.svelte";
  import Metadata from "../../lib/ui/Metadata.svelte";
  import Select from "../../lib/ui/Select.svelte";
  import ExperimentPicker from "./ExperimentPicker.svelte";
  import HistogramChart from "./HistogramChart.svelte";
  import { jobMetadata } from "./metadata";

  let { explorer }: { explorer: Explorer } = $props();
  let pickerOpen = $state(false);
  let pickerButton: HTMLButtonElement;

  const group = $derived(explorer.selectedConfig ?? []);
  const job = $derived(explorer.selectedJob!);
  const model = $derived(
    histogram(job, explorer.selectedReference!, explorer.question, explorer.bins),
  );
  const binLabels: Record<number, string> = {
    8: "Compact · up to 8 bitstrings",
    16: "Standard · up to 16 bitstrings",
    32: "Expanded · up to 32 bitstrings",
  };
</script>

<div id="histogram-view">
  <h2 id="histogram-title" tabindex="-1">{algorithmName(explorer.algorithm)} measurement outcomes</h2>
  <p class="histogram-intro">Compare the measured bitstrings with the answers this experiment expects.</p>
  <div class="settings histogram-selection">
    <div class="experiment-field">
      <span id="experiment-label">Experiment</span><button
        bind:this={pickerButton}
        id="choose-config"
        class="picker-trigger"
        type="button"
        aria-labelledby="experiment-label choose-config"
        aria-haspopup="dialog"
        aria-controls="experiment-picker"
        onclick={() => (pickerOpen = true)}>{job.description || job.config}</button
      >
    </div>
    <div class="run-field">
      <Select
        id="rep"
        label="Recorded run"
        aria-describedby="run-help"
        value={jobKey(job)}
        options={group.map(jobKey)}
        format={(key) => {
          const run = group.find((j) => jobKey(j) === key)!;
          return `Run ${run.rep} · ${jobTime(run)}`;
        }}
        onchange={(v) => (explorer.runChoice = v)}
      />
      <p id="run-help" class="field-help">One run is one IBM job executing this circuit repeatedly.</p>
    </div>
  </div>
  <ExperimentPicker
    {explorer}
    open={pickerOpen}
    onclose={() => {
      pickerOpen = false;
      pickerButton.focus();
    }}
  />
  <Metadata id="run-context" label="Run details" entries={jobMetadata(job)} />
  <div class="numbers" id="numbers">
    {#each [["Task success", pct(job.success)], ["DSR · peak dominance", score(job.dsr)], ["Full TVD", score(job.tvd)], ["Guessing baseline", pct(job.baseline)]] as [name, value] (name)}
      <div><span>{name}</span><b>{value}</b></div>
    {/each}
  </div>
  <div class="histogram-chart-heading">
    <h3 id="hist-chart-title">{model.title}</h3>
    <Select
      id="bins"
      label="Chart detail"
      aria-describedby="bins-help"
      value={explorer.bins}
      options={BIN_OPTIONS}
      format={(n) => binLabels[n]}
      onchange={(v) => (explorer.bins = v)}
    />
  </div>
  <p id="hist-note">{model.note}</p>
  <HistogramChart {model} />
  <p id="bins-help" class="note">{model.help}</p>
  <div class="hardware-context">
    <div>
      <span class="hardware-context-label">Behind this run</span>
      <p id="job-calibration-help" class="note">
        See how these histogram bits map to physical qubits and their calibration. Opens the chip
        explorer scoped to this run.
      </p>
    </div>
    <button
      type="button"
      id="job-calibration"
      class="text-action"
      aria-describedby="job-calibration-help"
      onclick={(event) => explorer.calibration.open(job, event.currentTarget)}
      >Inspect run calibration <span aria-hidden="true">→</span></button
    >
  </div>
  <details>
    <summary>View all outcome counts and reference values</summary>
    <div class="scroll">
      <table>
        <thead>
          <tr><th>Outcome</th><th>Shots</th><th>Observed</th><th>Ideal</th><th>Accepted</th></tr>
        </thead>
        <tbody id="outcomes">
          {#each model.rows as row (row.label)}
            <tr>
              <td>{row.label}</td>
              <td>{row.shots}</td>
              <td>{pct(row.observed)}</td>
              <td>{row.ideal == null ? "Unavailable" : pct(row.ideal)}</td>
              <td>{row.accepted == null ? "Unavailable" : row.accepted ? "Yes" : "No"}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </details>
</div>
