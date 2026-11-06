<script lang="ts">
  import { algorithmName, qubitSummary } from "../../lib/domain/catalog";
  import { unique } from "../../lib/domain/format";
  import type { Explorer } from "../../lib/state/explorer.svelte";
  import Dialog from "../../lib/ui/Dialog.svelte";
  import Icon from "../../lib/ui/Icon.svelte";

  let { explorer, open, onclose }: { explorer: Explorer; open: boolean; onclose: () => void } =
    $props();

  let query = $state("");
  let search: HTMLInputElement;
  let results: HTMLDivElement;
  const buttons: HTMLButtonElement[] = $state([]);

  const selected = $derived(explorer.selectedConfig?.[0].config);
  const matches = $derived.by(() => {
    const words = query.trim().toLowerCase().split(/\s+/);
    return explorer.configs
      .filter(([job]) => {
        const text = [job.config, job.family, job.description].join(" ").toLowerCase();
        return words.every((word) => text.includes(word));
      })
      .sort((a, b) => Number(b[0].config === selected) - Number(a[0].config === selected));
  });
  const first = $derived(explorer.configs[0]?.[0]);

  // Each opening starts from a clear search with focus in the field.
  $effect(() => {
    if (!open) return;
    query = "";
    queueMicrotask(() => search.focus());
  });
  $effect(() => {
    void matches;
    if (results) results.scrollTop = 0;
  });

  function choose(config: string) {
    explorer.selectConfig(config);
    onclose();
  }

  function searchKeys(event: KeyboardEvent) {
    const items = buttons.filter(Boolean);
    if (!items.length) return;
    if (event.key === "ArrowDown") (event.preventDefault(), items[0].focus());
    if (event.key === "ArrowUp") (event.preventDefault(), items.at(-1)!.focus());
    if (event.key === "Enter") (event.preventDefault(), items[0].click());
  }

  function itemKeys(event: KeyboardEvent, index: number) {
    const items = buttons.filter(Boolean);
    if (event.key === "ArrowDown") (event.preventDefault(), items[(index + 1) % items.length].focus());
    if (event.key === "ArrowUp")
      (event.preventDefault(), index ? items[index - 1].focus() : search.focus());
  }
</script>

<Dialog id="experiment-picker" labelledby="picker-title" {open} {onclose}>
  <div class="section-head picker-heading">
    <div>
      <p class="eyebrow">{first ? `${algorithmName(first.algorithm)} · ${first.backend}` : ""}</p>
      <h2 id="picker-title">Choose an experiment</h2>
    </div>
    <button type="button" class="picker-close" aria-label="Close experiment picker" onclick={onclose}
      ><Icon name="close" /></button
    >
  </div>
  <p class="note picker-intro">
    Choose a circuit to inspect its measured outcomes. Results follow your operating-range filters.
  </p>
  <label for="experiment-search">Find an experiment</label>
  <div class="picker-search">
    <Icon name="search" />
    <input
      bind:this={search}
      bind:value={query}
      id="experiment-search"
      type="search"
      placeholder="Search description, group or ID…"
      autocomplete="off"
      aria-controls="experiment-results"
      aria-describedby="match-count"
      onkeydown={searchKeys}
    />
  </div>
  <p id="match-count" class="note" role="status">
    {matches.length} of {explorer.configs.length} experiments
  </p>
  <div bind:this={results} id="experiment-results" role="group" aria-label="Matching experiments">
    {#each matches as group, index (group[0].config)}
      {@const job = group[0]}
      {@const current = job.config === selected}
      <button
        bind:this={buttons[index]}
        type="button"
        aria-current={current}
        onclick={() => choose(job.config)}
        onkeydown={(event) => itemKeys(event, index)}
      >
        <span class="experiment-choice-heading">
          <strong>{job.description || job.config}</strong>
          {#if current}<span class="experiment-selected">✓ Selected</span>{/if}
        </span>
        <span class="experiment-choice-context"
          >{unique([job.family, job.config].filter(Boolean)).join(" · ")}</span
        >
        <span class="experiment-choice-metadata">
          <span><Icon name="qubits" />{qubitSummary(job)}</span>
          <span><Icon name="depth" />ISA depth {job.depth}</span>
          <span
            ><Icon name="runs" />{group.length} recorded run{group.length === 1 ? "" : "s"}</span
          >
        </span>
      </button>
    {/each}
  </div>
  {#if !matches.length}
    <div id="picker-empty">
      <strong>No experiments found</strong>
      <p>Try a shorter description, group name or experiment ID.</p>
      <button
        type="button"
        class="view-button"
        onclick={() => {
          query = "";
          search.focus();
        }}>Clear search</button
      >
    </div>
  {/if}
  <div class="picker-footer">
    <span>Select a circuit to view its recorded runs.</span><span class="picker-key-hint"
      >↑ ↓ Navigate · Enter Open · Esc Close</span
    >
  </div>
</Dialog>
