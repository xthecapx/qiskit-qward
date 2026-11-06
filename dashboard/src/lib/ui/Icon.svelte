<script lang="ts" module>
  /** Static, trusted SVG bodies (24×24). Styling comes from the surrounding CSS. */
  export const icons = {
    chip: '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4m6-4v4M9 18v4m6-4v4M2 9h4m-4 6h4m12-6h4m-4 6h4"/>',
    shots: '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><path d="M12 2v3m0 14v3M2 12h3m14 0h3"/>',
    bits: '<rect x="3" y="5" width="7" height="14" rx="2"/><path d="M16 8l2-2v12m-3 0h6"/>',
    depth: '<path d="M3 6h18M3 12h18M3 18h18"/><rect x="6" y="3" width="4" height="6" rx="1"/><rect x="14" y="9" width="4" height="6" rx="1"/><rect x="6" y="15" width="4" height="6" rx="1"/>',
    runs: '<path d="M20 7a9 9 0 0 0-15-2L2 8m0-6v6h6m-4 9a9 9 0 0 0 15 2l3-3m0 6v-6h-6"/>',
    qubits: '<circle cx="5" cy="12" r="3"/><circle cx="18" cy="5" r="3"/><circle cx="18" cy="19" r="3"/><path d="m8 10 7-4m-7 8 7 4"/>',
    experiment: '<path d="M4 3h10l6 6v12H4zM14 3v6h6M8 13h8m-8 4h5"/>',
    date: '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M7 2v6m10-6v6M3 11h18m-13 4h2m4 0h2"/>',
    group: '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
    filter: '<path d="M3 5h18l-7 8v5l-4 2v-7z"/>',
    compare: '<path d="M4 3v17h17M8 15l4-5 4 2 4-6"/>',
    sliders: '<path d="M3 7h7m4 0h7M3 17h11m4 0h3"/><circle cx="12" cy="7" r="2"/><circle cx="16" cy="17" r="2"/>',
    close: '<path d="m6 6 12 12M18 6 6 18"/>',
    search: '<circle cx="10" cy="10" r="6"/><path d="m15 15 5 5"/>',
  } as const;
  export type IconName = keyof typeof icons;
</script>

<script lang="ts">
  let { name, size }: { name: IconName; size?: number } = $props();
</script>

{#if size}
  <svg aria-hidden="true" focusable="false" viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" stroke-width="2">{@html icons[name]}</svg>
{:else}
  <svg aria-hidden="true" focusable="false" viewBox="0 0 24 24">{@html icons[name]}</svg>
{/if}
