<script lang="ts">
  import EChart from "../../lib/charts/EChart.svelte";
  import Select from "../../lib/ui/Select.svelte";
  import { algorithmName } from "../../lib/domain/catalog";
  import type { Explorer } from "../../lib/state/explorer.svelte";
  import { bounds, chart, fixed, key, resources, roles, short, targets, type Snapshot } from "./model";
  type Choices={pair:string;resource:string;outcome:string};
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
  const rankedPairs = $derived(cells.map(c => ({
    id: key(c),
    name: pairLabel(key(c)),
    gap: c.grouped === null ? null : c.gap,
  })).sort((a, b) => {
    if (a.gap === b.gap) return a.name.localeCompare(b.name, "en");
    if (a.gap === null) return 1;
    if (b.gap === null) return -1;
    return a.gap - b.gap;
  }));
  const cell=$derived(cells.find(c=>key(c)===choices.pair)??cells.find(c=>key(c)===rankedPairs[0]?.id));
  const periods=$derived(Object.entries(cell?.periods??{}));
  const axes=$derived(bounds(periods.map(p=>p[1])));
  const ids=$derived(data?.algorithms??[]);
  const unit=$derived(outcome==="success"?"pp":"points");
  const pairOptions = $derived(new Map(rankedPairs.map(({id, name, gap}) => [
    id, name + " · " + (gap === null ? "Insufficient overlap" : fixed(gap) + " " + unit),
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
      <p class="note">{targets[outcome]} by {resources[metric]?.toLowerCase()??metric} · {explorer.backend}</p></div>
      <div class="stat"><strong>{fixed(cell.gap)} {unit}</strong><span>Median gap in fitted change</span></div>
    </div>
    <p class="verdict">{cell.grouped===true?"Grouped together":cell.grouped===false?"In separate clusters":"Insufficient overlap for clustering"} · {data.tolerance}-{unit} tolerance · {periods.length} eligible {periods.length===1?"period":"periods"}</p>
    {#if periods.length}
      <div class="legend"><span>● {algorithmName(cell.a)} · solid</span><span>◆ {algorithmName(cell.b)} · dashed</span></div>
      <p class="note">{ylabel}. Points show configuration means at shared levels. Lines show the linear fit for each algorithm.</p>
      <div class="panels">{#each periods as [p,values] (p)}
        <article><h3>{dateLabel(p)}</h3>
        <EChart option={chart(values,cell,axes,resources[metric]??metric,ylabel)} replace label={pairLabel(key(cell))+" on "+dateLabel(p)} style="height:195px;width:100%"/>
        </article>{/each}</div>
      <h3>Period measurements</h3>
      <div class="scroll"><table><thead><tr><th>Collection date (UTC)</th><th>Shared levels</th><th>{short(cell.a)} slope</th><th>{short(cell.b)} slope</th><th>Gap ({unit})</th><th>Profile ρ</th></tr></thead><tbody>
      {#each periods as [p,v]}<tr><td>{dateLabel(p)}</td><td>{v.points.length}</td><td>{fixed(100*v.slopeA)}</td><td>{fixed(100*v.slopeB)}</td><td>{fixed(v.gap)}</td><td>{fixed(v.rho)}</td></tr>{/each}</tbody></table></div>
      <p class="note">Slopes use {unit} per resource unit. The gap compares fitted changes across the shared range. Spearman ρ compares the rankings of the observed means.</p>
    {:else}
      <p>No period has the required {data.minLevels} shared levels. The largest overlap is {Math.max(0,...Object.values(cell.overlap))}.</p>
    {/if}
    <h3>All algorithm pairs</h3>
    <p id="matrix-guide">How differently does {targets[outcome].toLowerCase()} change as {resources[metric]?.toLowerCase()??metric} increases?
      In each period, compare how much each fitted line rises or falls from the smallest to the largest shared resource value.
      Each cell shows the median absolute difference between those changes, in {outcome==="success"?"percentage points":"points on the target × 100 scale"}.</p>
    {#if periods.length >= data.minPeriods && cell.gap !== null}
      <p class="matrix-example" aria-live="polite"><strong>{short(cell.a)} / {short(cell.b)}</strong> ·
        Period gaps ({unit}): {periods.map(([,v])=>fixed(v.gap)).join(", ")}.
        Median <strong>{fixed(cell.gap)} {unit}</strong>, displayed as <strong>{cell.gap.toFixed(1)}</strong> in the matrix.</p>
    {/if}
    <div class="scroll"><table class="matrix"><thead><tr><th>Pair</th>{#each ids as a}<th>{short(a)}</th>{/each}</tr></thead><tbody>
    {#each ids as a}<tr><th>{short(a)}</th>{#each ids as b}{@const c=lookup(a,b)}<td>
    {#if c}<button class:close={c.grouped===true} class:chosen={key(c)===key(cell)} aria-pressed={key(c)===key(cell)} aria-label={pairLabel(key(c))} onclick={()=>change({pair:key(c)})}
      style:background-color={c.grouped===null||c.gap===null?"#f1f2ee":`hsl(162 36% ${94-50*Math.exp(-c.gap/data.tolerance)}%)`}>
      {c.grouped===null?"—":c.gap?.toFixed(1)}</button>{:else}·{/if}</td>{/each}</tr>{/each}
    </tbody></table></div><p class="note">Closer to zero means more similar fitted changes. Two algorithms can have a small gap even when both lose a lot of success. Darker cells mark smaller gaps. The clustering tolerance is {data.tolerance} {unit}.
      A dash means fewer than {data.minPeriods} eligible periods. A dot marks the same algorithm. Select a cell to inspect its lines.</p>
  {/if}
</section>
<style>
  .matrix-example{border-left:3px solid var(--teal);background:#f1f6f3;padding:10px 14px;line-height:1.5}
.compare{padding:clamp(12px,2.3vw,28px);display:grid;gap:14px}.controls{border-bottom:1px solid var(--line);padding-bottom:16px}.controls :global(label){flex:1 1 145px;min-width:0}.controls :global(select){width:100%;min-width:0}
.heading{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}.heading h2{font:500 clamp(1.6rem,3vw,2.5rem) Georgia,serif}.stat{border-left:3px solid var(--teal);padding-left:14px;display:grid}.stat strong{font-size:1.55rem}.stat span{font-size:.78rem;color:var(--muted)}.verdict{font-weight:650}.legend{display:flex;gap:20px;flex-wrap:wrap;font-size:.85rem;font-weight:650}.legend span:first-child{color:var(--teal)}.legend span:last-child{color:#a03e10}.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,255px),1fr));gap:10px}.panels article{min-width:0;border:1px solid var(--line);padding:12px 6px 0;border-radius:9px}.panels h3{padding:0 8px}.scroll{max-width:100%;overflow:auto}.compare{min-width:0}.matrix{table-layout:fixed;min-width:500px}.matrix th,.matrix td{text-align:center;padding:2px}.matrix button{width:100%;min-height:34px;border:0;border-radius:3px;color:var(--ink);cursor:pointer;font:inherit}.matrix button.chosen{outline:2px solid var(--ink);outline-offset:-2px}.matrix button:hover{box-shadow:inset 0 0 0 1px var(--teal)}
</style>
