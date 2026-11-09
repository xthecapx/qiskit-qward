<script lang="ts" generics="T extends string | number">
  import type { HTMLSelectAttributes } from "svelte/elements";

  interface Props extends Omit<HTMLSelectAttributes, "value" | "onchange"> {
    label: string;
    value: T;
    options: readonly T[];
    format?: (value: T) => string;
    onchange: (value: T) => void;
  }

  let { label, value, options, format = String, onchange, ...rest }: Props = $props();

  function change(event: Event) {
    const index = (event.currentTarget as HTMLSelectElement).selectedIndex;
    if (index >= 0) onchange(options[index]);
  }
</script>

<label
  >{label}<select aria-label={label} {...rest} onchange={change}>
    {#each options as option, index (option)}
      <option value={index} selected={option === value}>{format(option)}</option>
    {/each}
  </select></label
>
