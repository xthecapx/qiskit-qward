<script lang="ts">
  import type { Snippet } from "svelte";

  interface Props {
    id: string;
    open: boolean;
    labelledby: string;
    onclose: () => void;
    children: Snippet;
  }

  let { id, open, labelledby, onclose, children }: Props = $props();
  let dialog: HTMLDialogElement;

  // The native modal provides focus trapping, inertness and Escape handling.
  $effect(() => {
    if (open && !dialog.open) dialog.showModal();
    else if (!open && dialog.open) dialog.close();
  });
</script>

<dialog bind:this={dialog} {id} aria-labelledby={labelledby} {onclose}>
  {@render children()}
</dialog>
