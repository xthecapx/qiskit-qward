<script lang="ts" module>
  import type { IconName } from "./Icon.svelte";
  export type MetadataEntry = [icon: IconName, label: string, value: string | number];
</script>

<script lang="ts">
  import Icon from "./Icon.svelte";

  let { id, label, entries }: { id?: string; label: string; entries: MetadataEntry[] } = $props();
</script>

<dl {id} class="context-metadata" aria-label={label}>
  {#each entries as [icon, term, value] (term)}
    <div>
      <dt><Icon name={icon} />{term}</dt>
      <dd>{value}</dd>
    </div>
  {/each}
</dl>
