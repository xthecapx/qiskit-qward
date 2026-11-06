<script lang="ts">
  import type { AxisKey } from "../../lib/data/types";
  import { axes, questions } from "../../lib/domain/catalog";
  import { lowerIsBetter, plural } from "../../lib/domain/format";
  import { ALL_GROUPS, ALL_PERIODS } from "../../lib/domain/scope";
  import type { Explorer } from "../../lib/state/explorer.svelte";
  import Dialog from "../../lib/ui/Dialog.svelte";
  import Icon from "../../lib/ui/Icon.svelte";
  import Select from "../../lib/ui/Select.svelte";

  let { explorer, open, onclose }: { explorer: Explorer; open: boolean; onclose: () => void } =
    $props();

  const metric = $derived(questions[explorer.question].label);
  const periodIds = $derived([ALL_PERIODS, ...explorer.periods.map((p) => p.id)]);
  const periodText = (id: string) =>
    id === ALL_PERIODS ? "All periods" : (explorer.periods.find((p) => p.id === id)?.label ?? id);
</script>

<Dialog id="plot-settings" labelledby="settings-title" {open} {onclose}>
  <form method="dialog">
    <h2 id="settings-title">Adjust plot</h2>
    <div class="plot-setting-sections">
      <fieldset>
        <legend><Icon name="filter" />Jobs to include</legend>
        <Select
          id="backend"
          label="Chip"
          value={explorer.backend}
          options={explorer.backends}
          onchange={(v) => explorer.setBackend(v)}
        />
        <Select
          id="window"
          label="Periods (UTC)"
          value={explorer.period}
          options={periodIds}
          format={periodText}
          onchange={(v) => explorer.setPeriod(v)}
        />
        <Select
          id="family"
          label="Experiment group"
          value={explorer.family}
          options={[ALL_GROUPS, ...explorer.families]}
          format={(g) => g || "All experiments"}
          onchange={(v) => explorer.setFamily(v)}
        />
      </fieldset>
      <fieldset>
        <legend><Icon name="compare" />How to compare jobs</legend>
        <Select
          id="axis"
          label="Horizontal axis"
          value={explorer.axis}
          options={Object.keys(axes) as AxisKey[]}
          format={(k) => axes[k]}
          onchange={(v) => (explorer.axis = v)}
        />
        <label
          ><span>{lowerIsBetter(explorer.question) ? "Maximum" : "Minimum"} {metric.toLowerCase()}</span>
          <input
            id="target"
            type="number"
            min="0"
            max="1"
            step="any"
            placeholder="No target"
            aria-describedby="target-help"
            aria-invalid={explorer.targetInvalid}
            bind:value={
              () => explorer.targetText,
              (v) => (explorer.targetText = v == null ? "" : String(v))
            }
          />
        </label>
        <p class="note" id="target-help">
          {explorer.question === "success"
            ? "Enter a fraction, e.g. 0.5 for 50%."
            : "Enter a value from 0 to 1."} Leave blank to explore without a cutoff.
        </p>
      </fieldset>
    </div>
    <div class="settings-footer">
      <span id="settings-coverage" class="note" role="status"
        >{plural(explorer.configs.length, "experiment")} · {explorer.jobs.length} jobs · {explorer.collections}
        collection(s)</span
      >
      <button class="view-button" type="submit">Done</button>
    </div>
  </form>
</Dialog>
