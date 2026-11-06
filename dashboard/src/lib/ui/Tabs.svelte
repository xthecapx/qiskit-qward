<script lang="ts" module>
  export interface TabItem {
    id: string;
    label: string;
  }
</script>

<script lang="ts">
  import type { Snippet } from "svelte";

  interface Props {
    tabs: TabItem[];
    label: string;
    selected: string;
    onselect: (id: string) => void;
    panel: Snippet<[string]>;
  }

  let { tabs, label, selected, onselect, panel }: Props = $props();
  const buttons: HTMLButtonElement[] = [];

  /** WAI-ARIA tabs pattern: arrows move and activate; Home/End jump. */
  function keydown(event: KeyboardEvent, index: number) {
    const next = {
      ArrowLeft: (index + tabs.length - 1) % tabs.length,
      ArrowRight: (index + 1) % tabs.length,
      Home: 0,
      End: tabs.length - 1,
    }[event.key];
    if (next === undefined) return;
    event.preventDefault();
    onselect(tabs[next].id);
    buttons[next].focus();
  }
</script>

<div class="finding-tabs" role="tablist" aria-label={label}>
  {#each tabs as tab, index (tab.id)}
    <button
      bind:this={buttons[index]}
      id="tab-{tab.id}"
      type="button"
      role="tab"
      aria-controls="panel-{tab.id}"
      aria-selected={tab.id === selected}
      tabindex={tab.id === selected ? 0 : -1}
      onclick={() => onselect(tab.id)}
      onkeydown={(event) => keydown(event, index)}>{tab.label}</button
    >
  {/each}
</div>
{#each tabs as tab (tab.id)}
  <div
    id="panel-{tab.id}"
    role="tabpanel"
    aria-labelledby="tab-{tab.id}"
    tabindex="0"
    hidden={tab.id !== selected}
  >
    {#if tab.id === selected}{@render panel(tab.id)}{/if}
  </div>
{/each}
