<script lang="ts">
  import {
    bitMappings,
    calibrationNumber,
    componentLabel,
    isGate,
    modes,
    type CalibrationMode,
  } from "../../lib/domain/calibration";
  import { unique } from "../../lib/domain/format";
  import type { CalibrationState } from "../../lib/state/calibration.svelte";
  import Select from "../../lib/ui/Select.svelte";

  let { cal }: { cal: CalibrationState } = $props();

  const drawings: Record<CalibrationMode, string> = {
    readout:
      '<circle cx="40" cy="42" r="19"/><path d="M60 42h50"/><rect x="110" y="20" width="65" height="44" rx="8"/><path d="M122 50a20 20 0 0 1 40 0m-20-1 14-18"/><text x="34" y="47">q</text>',
    single: '<path d="M15 42h50m65 0h65"/><rect x="65" y="18" width="65" height="48" rx="9"/><text x="77" y="48">Gate</text>',
    pair: '<path d="M15 23h180M15 65h180M105 23v42"/><circle cx="105" cy="23" r="7"/><circle cx="105" cy="65" r="7"/>',
    t1: '<path d="M25 20h60m40 45h60M105 23v35m-6-6 6 6 6-6"/><text x="38" y="43">|1⟩</text><text x="140" y="49">|0⟩</text>',
    t2: '<circle cx="48" cy="42" r="23"/><path d="M48 42l17-16M83 42h35"/><circle cx="155" cy="42" r="23"/><path d="M155 42l17-16m-17 16 20 4m-20-4-9 18m9-18-15-8"/>',
  };

  const item = $derived(cal.component);
  const directional = $derived.by(() => {
    if (!item || cal.mode !== "readout") return [];
    return (
      [
        ["prob_meas1_prep0", "Prepared 0 → reported 1"],
        ["prob_meas0_prep1", "Prepared 1 → reported 0"],
      ] as const
    ).map(([key, title]) => {
      const distinct = unique(
        cal.current.map((r) => r.qubits[item.id]?.[key]).filter((v): v is number => v != null),
      );
      return { title, value: distinct.length > 1 ? "Conflicting records" : calibrationNumber("readout", distinct[0]) };
    });
  });
  const mapping = $derived.by(() => {
    if (!cal.job) return "Open from a job histogram to see how physical qubits map to its measured bits.";
    const mappings = item ? bitMappings(cal.jobRecord, item.qubits) : [];
    return mappings.length
      ? `${mappings.join("; ")}. Bit 0 is the rightmost bit of the flattened histogram label. Repeated measurements may overwrite bits; mappings include recorded measurement steps.`
      : "This component has no recorded direct measurement mapping for the selected job.";
  });
</script>

<aside class="calibration-detail">
  <label
    >Find a qubit or connection<input
      type="search"
      id="cal-search"
      placeholder="Search, e.g. Q24"
      autocomplete="off"
      bind:value={cal.query}
    /></label
  >
  <Select
    id="cal-component"
    label="Component"
    disabled={!cal.filtered.length}
    value={item?.id ?? ""}
    options={cal.filtered.map((c) => c.id)}
    format={(id) => componentLabel(cal.filtered.find((c) => c.id === id)!)}
    onchange={(id) => cal.selectComponent(id)}
  />
  {#if !cal.filtered.length}<p class="note">No matching components.</p>{/if}
  <h3 id="cal-component-title">
    {item ? `${componentLabel(item)}${isGate(cal.mode) ? ` · ${cal.gate.toUpperCase()}` : ""}` : "No component selected"}
  </h3>
  <div id="cal-drawing" aria-hidden="true">
    <svg viewBox="0 0 210 85" fill="none" stroke="currentColor" stroke-width="2">{@html drawings[cal.mode]}</svg>
  </div>
  <div id="cal-value" class="calibration-value">
    {item?.conflict ? "Conflicting records" : calibrationNumber(cal.mode, item?.value)}
  </div>
  <p>{modes[cal.mode].explanation}</p>
  <dl class="calibration-facts">
    {#each directional as fact (fact.title)}
      <div><dt>{fact.title}</dt><dd>{fact.value}</dd></div>
    {/each}
  </dl>
  <p id="cal-bit-mapping" class="note">{mapping}</p>
</aside>
