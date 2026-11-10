<script lang="ts">
  import EChart from "../../lib/charts/EChart.svelte";
  import Select from "../../lib/ui/Select.svelte";
  import { algorithmName } from "../../lib/domain/catalog";
  import type { Explorer } from "../../lib/state/explorer.svelte";
  import { bounds, chart, extraLoss, fixed, key, medianLoss, signed, resources, roles, short, targets, type Snapshot } from "./model";
  type Choices={matrix:string;pair:string;resource:string;outcome:string};
  let {explorer,reports,choices,onchange}:{explorer:Explorer;reports:Record<string,string>;choices:Choices;onchange:(v:Choices)=>void}=$props();
  let data=$state<Snapshot>();
  let error=$state("");
  $effect(()=>{
    const backend=explorer.backend, filename=reports[backend], controller=new AbortController();
    data=undefined; error="";
    if(filename) {
      fetch(new URL("data/"+filename,document.baseURI),{signal:controller.signal}).then(async r=>{
        if(!r.ok) throw Error("fetch"); const s:Snapshot=await r.json();
        if(s.schemaVersion!==1 || s.backend!==backend || !s.panels) throw Error("schema");
        if(!controller.signal.aborted)data=s;
      }).catch(()=>{if(!controller.signal.aborted)error="The decay measurements could not be loaded. Reload to try again.";});
    }
    return ()=>controller.abort();
  });
  const metrics=$derived(Object.keys(data?.panels??{}));
  const metric=$derived(metrics.includes(choices.resource)?choices.resource:metrics[0]);
  const outcomes=$derived(Object.keys(data?.panels[metric]??{}));
  const outcome=$derived(outcomes.includes(choices.outcome)?choices.outcome:outcomes[0]);
  const cells=$derived(data?.panels[metric]?.[outcome]??[]);
  const directional = $derived(choices.matrix === "direction");
  const signedGaps = $derived(new Map(cells.map(c => [key(c), medianLoss(c)])));
  const rankedPairs = $derived(cells.map(c => ({
    id: key(c),
    name: pairLabel(key(c)),
    gap: !directional && c.grouped === null ? null : c.gap,
  })).sort((a, b) => {
    if (a.gap === b.gap) return a.name.localeCompare(b.name, "en");
    if (a.gap === null) return 1;
    if (b.gap === null) return -1;
    return a.gap - b.gap;
  }));
  const cell=$derived(cells.find(c=>key(c)===choices.pair || c.b+"/"+c.a===choices.pair)??cells.find(c=>key(c)===rankedPairs[0]?.id));
  const periods=$derived(Object.entries(cell?.periods??{}));
  const reversed = $derived(!!cell && choices.pair === cell.b+"/"+cell.a);
  const row = $derived(cell && reversed ? cell.b : cell?.a ?? "");
  const column = $derived(cell && reversed ? cell.a : cell?.b ?? "");
  function directionValue(c: typeof cells[number], rowId: string) {
    const value = signedGaps.get(key(c));
    return value == null ? null : rowId === c.a ? value : -value;
  }
  const axes=$derived(bounds(periods.map(p=>p[1])));
  const ids=$derived(data?.algorithms??[]);
  const unit=$derived(outcome==="success"?"pp":"points");
  const pairOptions = $derived(new Map(rankedPairs.map(({id, name, gap}) => [
    id, directional ? name : name + " · " + (gap === null ? "Insufficient overlap" : fixed(gap) + " " + unit),
  ])));
  const ylabel=$derived(outcome==="success"?"Success rate (%)":(targets[outcome]+" × 100"));
  function change(v:Partial<Choices>){onchange({...choices,...v});}
  function pairLabel(k:string){return k.split("/").map(algorithmName).join(" / ");}
  function lookup(a:string,b:string){return cells.find(c=>(c.a===a&&c.b===b)||(c.a===b&&c.b===a));}
  const dates = $derived(new Map(explorer.periods.map(period => {
    const { campaign, window } = period.jobs[0];
    return [campaign + " / window " + window, period.dateLabel] as const;
  })));
  function dateLabel(period: string) { return dates.get(period) ?? "Date unavailable"; }
</script>

<section class="surface compare" aria-label="Decay comparison">
  <div class="settings controls">
    <Select label="Chip" value={explorer.backend} options={explorer.backends} onchange={(b)=>explorer.setBackend(b)}/>
    {#if cell}
      <Select label="Algorithm pair" value={key(cell)} options={[...pairOptions.keys()]} format={id=>pairOptions.get(id)??pairLabel(id)} onchange={pair=>change({pair})}/>
      <Select label="Resource" value={metric} options={metrics} format={v=>resources[v]??v} onchange={resource=>change({resource})}/>
      <Select label="Outcome" value={outcome} options={outcomes} format={v=>targets[v]??v} onchange={outcome=>change({outcome})}/>
    {/if}
  </div>
  {#if error}<p role="alert">{error}</p>
  {:else if !reports[explorer.backend]}<p role="status">No decay analysis matches the current export for {explorer.backend}.</p>
  {:else if !data}<p role="status">Loading decay measurements…</p>
  {:else if cell}
    <div class="heading"><div><p class="eyebrow">{roles[cell.role]??cell.role}</p>
      <h2>{algorithmName(cell.a)} and {algorithmName(cell.b)}</h2>
      <p class="note">How does {targets[outcome]} change for these two algorithms across shared values of {resources[metric].toLowerCase()} in each window on {explorer.backend}?</p></div>
      <div class="stat"><strong>{directional ? signed(directionValue(cell,row)) : fixed(cell.gap)} {unit}</strong><span>{directional ? `Median extra loss of ${short(column)} relative to ${short(row)}` : "Median gap in fitted change"}</span></div>
    </div>
    {#if directional}
      <p class="verdict">{periods.length} eligible {periods.length === 1 ? "window" : "windows"}. {#if periods.length}{short(column)} had a lower fitted change than {short(row)} in {periods.filter(([,v])=>extraLoss(v,reversed)>0).length} of {periods.length}.{/if}</p>
    {:else}<p class="verdict">{cell.grouped===true?"Grouped together":cell.grouped===false?"In separate clusters":"Insufficient overlap for clustering"} · {data.tolerance}-{unit} tolerance · {periods.length} eligible {periods.length===1?"period":"periods"}</p>{/if}
    {#if periods.length}
      <div class="legend"><span>● {algorithmName(cell.a)} · solid</span><span>◆ {algorithmName(cell.b)} · dashed</span></div>
      <p class="note">{ylabel}. Points show configuration means at shared levels. Lines show the linear fit for each algorithm.</p>
      <div class="panels">{#each periods as [p,values] (p)}
        <article><h3>{dateLabel(p)}</h3>
        <EChart option={chart(values,cell,axes,resources[metric]??metric,ylabel)} replace label={pairLabel(key(cell))+" on "+dateLabel(p)} style="height:195px;width:100%"/>
        </article>{/each}</div>
      <h3>How much did the fitted trends differ in each window?</h3>
      <div class="scroll"><table><thead><tr><th>Collection date (UTC)</th><th>Shared levels</th><th>{short(cell.a)} slope</th><th>{short(cell.b)} slope</th><th>{directional ? `Extra loss of ${short(column)} vs ${short(row)}` : "Gap"} ({unit})</th><th>Profile ρ</th></tr></thead><tbody>
      {#each periods as [p,v]}<tr><td>{dateLabel(p)}</td><td>{v.points.length}</td><td>{fixed(100*v.slopeA)}</td><td>{fixed(100*v.slopeB)}</td><td>{directional ? signed(extraLoss(v,reversed)) : fixed(v.gap)}</td><td>{fixed(v.rho)}</td></tr>{/each}</tbody></table></div>
      <p class="note">Slopes use {unit} per resource unit. The gap compares fitted changes across the shared range. Spearman ρ compares the rankings of the observed means.</p>
    {:else}
      <p>No period has the required {data.minLevels} shared levels. The largest overlap is {Math.max(0,...Object.values(cell.overlap))}.</p>
    {/if}
    <div class="matrix-heading">
      <h3>All algorithm pairs</h3>
      <div class="settings matrix-settings">
        <Select label="Matrix view" value={choices.matrix} options={["similarity","direction"]} format={v=>v==="direction"?"Direction":"Similarity"} onchange={matrix=>change({matrix})}/>
      </div>
    </div>
    <div class="matrix-intro">
      <h4 id="matrix-guide">
        {directional ? "Which algorithm shows the larger fitted decrease in" : "Which algorithm pairs show similar changes in"}
        {targets[outcome].toLowerCase()} across the shared range of {resources[metric].toLowerCase()}?
      </h4>
      <div class="matrix-callout">
        <p id="matrix-reading">
          {#if directional}
            Read from row to column. A positive value means the column algorithm fell more or gained less.
            A negative value means the row algorithm did.
          {:else}
            Values closer to zero mean the two fitted lines changed by similar amounts over the shared range.
            Darker cells indicate smaller gaps.
          {/if}
          Each cell shows the median {directional ? "signed difference" : "absolute gap"} across eligible windows,
          in {outcome === "success" ? "percentage points" : "points on the outcome × 100 scale"}.
        </p>
        <p class="matrix-example" aria-live="polite">
          <strong>{directional ? short(row) + " → " + short(column) : short(cell.a) + " / " + short(cell.b)}</strong>
          {#if periods.length >= (directional ? 1 : data.minPeriods)}
            · Window {directional ? "differences" : "gaps"} ({unit}):
            {periods.map(([,v]) => directional ? signed(extraLoss(v,reversed)) : fixed(v.gap)).join(", ")}.
            Median <strong>{directional ? signed(directionValue(cell,row)) : fixed(cell.gap)} {unit}</strong>.
          {:else}
            · {periods.length} of {directional ? 1 : data.minPeriods} required windows have at least {data.minLevels} shared resource levels.
          {/if}
        </p>
      </div>
    </div>
    <div class="scroll"><table class="matrix" aria-labelledby="matrix-guide" aria-describedby="matrix-reading"><thead><tr><th>Pair</th>{#each ids as a}<th>{short(a)}</th>{/each}</tr></thead><tbody>
    {#each ids as a}<tr><th>{short(a)}</th>{#each ids as b}{@const c=lookup(a,b)}<td>
    {#if c}{@const selected=directional ? a===row && b===column : key(c)===key(cell)}
      {@const delta=directionValue(c,a)}
      <button class:close={!directional && c.grouped===true} class:chosen={selected} aria-pressed={selected} aria-label={directional ? `${short(a)} to ${short(b)}: ${signed(delta)} ${unit}` : pairLabel(key(c))} onclick={()=>change({pair:directional ? a+"/"+b : key(c)})}
      style:background-color={directional ? (delta === null || delta === 0 ? "#f1f2ee" : delta > 0 ? "#f6ddca" : "#d9ece6") : c.grouped===null||c.gap===null?"#f1f2ee":`hsl(162 36% ${94-50*Math.exp(-c.gap/data.tolerance)}%)`}>
      {directional ? signed(delta,1) : c.grouped===null?"—":c.gap?.toFixed(1)}</button>{:else}·{/if}</td>{/each}</tr>{/each}
    </tbody></table></div>
    <p class="note">
      {#if directional}
        A dash means no window has {data.minLevels} shared levels.
      {:else}
        The clustering tolerance is {data.tolerance} {unit}. A dash means fewer than {data.minPeriods} eligible windows.
      {/if}
      A dot marks the same algorithm. Select a cell to inspect its fitted lines.
    </p>
  {/if}
</section>
<style>
  .matrix-heading { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px; }
  .matrix-settings { margin: 0; width: min(100%, 12rem); }
  .matrix-intro { display: grid; gap: 8px; }
  #matrix-guide { margin: 0; font-size: 1rem; font-weight: 700; line-height: 1.45; }
  .matrix-callout { background: #f1f6f3; border-inline-start: 3px solid var(--teal); padding: 10px 14px; line-height: 1.5; }
  .matrix-callout p { margin: 0; }
  .matrix-callout .matrix-example { margin-top: 6px; }
.compare{padding:clamp(12px,2.3vw,28px);display:grid;gap:14px}.controls{border-bottom:1px solid var(--line);padding-bottom:16px}.controls :global(label){flex:1 1 145px;min-width:0}.controls :global(select){width:100%;min-width:0}
.heading{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}.heading h2{font:500 clamp(1.6rem,3vw,2.5rem) Georgia,serif}.stat{border-left:3px solid var(--teal);padding-left:14px;display:grid}.stat strong{font-size:1.55rem}.stat span{font-size:.78rem;color:var(--muted)}.verdict{font-weight:650}.legend{display:flex;gap:20px;flex-wrap:wrap;font-size:.85rem;font-weight:650}.legend span:first-child{color:var(--teal)}.legend span:last-child{color:#a03e10}.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,255px),1fr));gap:10px}.panels article{min-width:0;border:1px solid var(--line);padding:12px 6px 0;border-radius:9px}.panels h3{padding:0 8px}.scroll{max-width:100%;overflow:auto}.compare{min-width:0}.matrix{table-layout:fixed;min-width:500px}.matrix th,.matrix td{text-align:center;padding:2px}.matrix button{width:100%;min-height:34px;border:0;border-radius:3px;color:var(--ink);cursor:pointer;font:inherit}.matrix button.chosen{outline:2px solid var(--ink);outline-offset:-2px}.matrix button:hover{box-shadow:inset 0 0 0 1px var(--teal)}
</style>
