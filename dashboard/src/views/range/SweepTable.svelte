<script lang="ts">
  import { format, jobTime, unique } from "../../lib/domain/format";
  import { jobKey } from "../../lib/domain/scope";
  import type { Explorer } from "../../lib/state/explorer.svelte";

  let { explorer }: { explorer: Explorer } = $props();
  const key = $derived(explorer.question);
</script>

<details>
  <summary>Compare the experiments shown</summary>
  <div class="scroll">
    <table>
      <thead>
        <tr>
          <th>Experiment</th>
          <th>Group</th>
          <th>Qubits</th>
          <th>ISA depth</th>
          <th>Job values</th>
          <th>Target result</th>
        </tr>
      </thead>
      <tbody id="settings-table">
        {#each explorer.configs as group, i (group[0].config)}
          <tr>
            <td>{group[0].config}</td>
            <td>{group[0].family}</td>
            <td>{group[0].width}</td>
            <td>{unique(group.map((j) => j.depth)).join(", ")}</td>
            <td>
              {#each group as job (jobKey(job))}
                {@const value = format(key, job[key])}
                <button
                  type="button"
                  class="job-value"
                  title="Run {job.rep} · {jobTime(job)}"
                  aria-label="{job.config}, repetition {job.rep}, {jobTime(job)}, {value}. Open histogram"
                  onclick={(event) => explorer.openJob(job, event.currentTarget)}>{value}</button
                >
              {/each}
            </td>
            <td>{explorer.summary.states[i]}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
</details>
