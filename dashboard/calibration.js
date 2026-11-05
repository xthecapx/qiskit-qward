"use strict";

// One static explorer, opened either from the overview or from a particular job.
(() => {
  const records = snapshot.calibrations || {};
  const modes = {
    readout: ["Readout error", "Average probability of reporting the wrong state when a known 0 or 1 is prepared. The two directions below can differ."],
    single: ["Single-qubit gate error", "The provider’s calibration error estimate for this operation on one physical qubit. Compare the same operation across dates."],
    pair: ["Two-qubit gate error", "The provider’s calibration error estimate for this operation on the ordered pair shown. The line represents an operation between qubits, not a separately measured coupler error."],
    t1: ["T1 · relaxation time", "Characteristic time for an excited qubit to lose energy. Longer times indicate slower relaxation; this is a duration, not an error probability."],
    t2: ["T2 · coherence time", "Characteristic coherence time, affected by energy relaxation and dephasing. Longer times indicate longer coherence; this is a duration, not an error probability."],
  };
  let job = null, origin, savedScroll, opener, map, history, component, historyExpanded = false;
  let current = [], components = [], dates = [], domain = [0, 1];
  const mode = () => el("cal-mode").value;
  const timed = () => ["t1", "t2"].includes(mode());
  const stamp = value => {
    if (!value) return "Timestamp unavailable";
    const date = new Date(value.replace(" ", "T"));
    return Number.isNaN(date.valueOf()) ? value : date.toISOString().slice(0, 16).replace("T", " ");
  };
  const number = value => value == null ? "Unavailable" : timed()
    ? `${value.toFixed(1)} μs` : `${(value * 100).toFixed(3)}%`;
  const allForChip = () => Object.values(records).filter(r => r.backend === el("cal-chip").value && r.at);
  const values = record => {
    if (["readout", "single", "pair"].includes(mode())) {
      return record.operations.filter(op => mode() === "readout" ? op.name === "measure"
        : op.name === el("cal-gate").value && op.qubits.length === (mode() === "pair" ? 2 : 1))
        .map(op => ({ id: op.qubits.join("→"), qubits: op.qubits, value: op.error }));
    }
    return Object.entries(record.qubits).map(([q, data]) => ({
      id: q, qubits: [Number(q)], value: data[`${mode()}_s`] == null ? null : data[`${mode()}_s`] * 1e6,
    }));
  };
  function merge(rows) {
    const merged = new Map();
    for (const record of rows) for (const item of values(record)) {
      if (!merged.has(item.id)) merged.set(item.id, { ...item, seen: new Set() });
      if (item.value != null) merged.get(item.id).seen.add(item.value);
    }
    return [...merged.values()].map(item => ({ ...item, conflict: item.seen.size > 1,
      value: item.seen.size === 1 ? [...item.seen][0] : null })).sort((a, b) => compareText(a.id, b.id));
  }
  const label = item => item.qubits.map(q => `Q${q}`).join(" → ");
  function color(value) {
    if (value == null) return "#bbc7c2";
    let t = (value - domain[0]) / (domain[1] - domain[0] || 1);
    if (timed()) t = 1 - t;
    const start = [15, 126, 113], end = [204, 93, 48];
    return `rgb(${start.map((c, i) => Math.round(c + (end[i] - c) * t)).join(",")})`;
  }
  function fillDates(preferred) {
    dates = unique(allForChip().map(r => r.at)).sort((a, b) => new Date(a.replace(" ", "T")) - new Date(b.replace(" ", "T")));
    options("cal-date", dates, preferred || dates.at(-1), stamp);
    el("cal-date").disabled = Boolean(job && !historyExpanded);
    update();
  }
  function update() {
    const rows = allForChip();
    const available = rows.length > 0 && (!job || historyExpanded || Boolean(records[job.ref]?.at));
    el("cal-empty").hidden = available;
    el("cal-content").hidden = !available;
    el("cal-gate-field").hidden = true;
    if (!available) {
      el("calibration-context").textContent = "Detailed preparation calibration is unavailable for this selection. The histogram and its metrics remain available.";
      return;
    }
    const gateMode = ["single", "pair"].includes(mode());
    el("cal-gate-field").hidden = !gateMode;
    const gates = unique(rows.flatMap(r => r.operations.filter(op => op.name !== "measure" && op.qubits.length === (mode() === "pair" ? 2 : 1)).map(op => op.name))).sort();
    options("cal-gate", gates, gates.includes(el("cal-gate").value) ? el("cal-gate").value : mode() === "single" ? "sx" : gates[0]);
    current = job && !historyExpanded ? (records[job.ref] ? [records[job.ref]] : [])
      : rows.filter(r => r.at === el("cal-date").value);
    components = merge(current);
    const known = rows.flatMap(values).map(item => item.value).filter(v => v != null);
    domain = known.length ? [Math.min(...known), Math.max(...known)] : [0, 1];
    el("cal-scale-low").textContent = number(timed() ? domain[1] : domain[0]);
    el("cal-scale-high").textContent = number(timed() ? domain[0] : domain[1]);
    el("cal-scope-note").textContent = `Fixed scale across recorded dates · ${timed() ? "longer is better" : "lower is better"}. Colors describe relative values, not a pass/fail threshold.`;
    el("cal-coverage").textContent = `${components.filter(c => c.value != null).length} / ${components.length} components with values`;
    el("calibration-title").textContent = job && !historyExpanded ? "Calibration for this run" : "Chip calibration history";
    el("calibration-context").textContent = job
      ? `${names[job.algorithm] || job.algorithm} · ${job.config} · run ${job.rep}. ${historyExpanded ? "Exploring chip history; return to the histogram to retain this job." : "Showing this circuit’s preparation snapshot and used hardware."} Preparation: ${stamp(records[job.ref]?.at)} UTC. Execution calibration: ${stamp(job.executionCalibration?.at)} UTC.`
      : "Explore the hardware observed in our experiments. Dates refer to saved preparation calibration snapshots; coverage varies between snapshots.";
    const executionValue = job?.executionCalibration?.values[mode()];
    el("cal-execution").textContent = job ? `Execution source: ${job.executionCalibration?.source || "unavailable"}. Recorded circuit summary: ${number(executionValue)}. Gate/readout means are instruction-weighted; coherence means are over used qubits. These summaries are not per-component execution measurements.` : "Open Calibration for this job from a histogram to compare its execution-time summary and preparation details.";
    el("cal-expand-history").hidden = !job;
    el("cal-expand-history").textContent = historyExpanded ? "Return to this run’s hardware" : "Explore this component’s history";
    chooseComponents();
  }
  function chooseComponents() {
    const query = el("cal-search").value.toLowerCase().replaceAll(" ", "");
    const filtered = components.filter(c => label(c).toLowerCase().replaceAll(" ", "").includes(query));
    if (!filtered.some(c => c.id === component)) {
      component = [...filtered].sort((a, b) => {
        if (a.value == null) return 1;
        if (b.value == null) return -1;
        return timed() ? a.value - b.value : b.value - a.value;
      })[0]?.id;
    }
    options("cal-component", filtered.map(c => c.id), component, id => label(filtered.find(c => c.id === id)));
    el("cal-component").disabled = !filtered.length;
    el("cal-component-empty").hidden = filtered.length > 0;
    renderMap();
    renderDetail();
  }
  function renderMap() {
    const qs = unique(current.flatMap(r => Object.keys(r.qubits))).sort(compareText);
    const links = new Map();
    for (const r of current) for (const op of r.operations.filter(o => o.qubits.length === 2)) {
      if (op.qubits.every(q => qs.includes(String(q)))) links.set(op.qubits.join("→"), op.qubits);
    }
    // Deterministic breadth-first schematic. No invented physical chip coordinates.
    const neighbors = new Map(qs.map(q => [q, new Set()]));
    for (const [a, b] of links.values()) { neighbors.get(String(a)).add(String(b)); neighbors.get(String(b)).add(String(a)); }
    const levels = new Map(); let offset = 0;
    for (const root of qs) {
      if (levels.has(root)) continue;
      const queue = [root]; levels.set(root, offset);
      for (let i = 0; i < queue.length; i++) for (const next of neighbors.get(queue[i])) {
        if (!levels.has(next)) { levels.set(next, levels.get(queue[i]) + 1); queue.push(next); }
      }
      offset = Math.max(...levels.values()) + 2;
    }
    const layers = new Map();
    for (const [q, level] of levels) { if (!layers.has(level)) layers.set(level, []); layers.get(level).push(q); }
    const selected = components.find(c => c.id === component);
    const plotWidth = Math.max(el("cal-map").clientWidth - 60, 100);
    const maxLevel = Math.max(...levels.values(), 1);
    const maxRows = Math.max(...[...layers.values()].map(row => row.length), 2);
    const data = qs.map(q => {
      const row = layers.get(levels.get(q));
      const item = mode() === "pair" ? null : components.find(c => c.id === q);
      return { id: q, name: `Q${q}`, x: 15 + levels.get(q) / maxLevel * (plotWidth - 30), y: 150 + (row.indexOf(q) - (row.length - 1) / 2) * (220 / maxRows) + (levels.get(q) % 2 ? 16 : -16),
        symbolSize: qs.length > 35 ? 14 : 22, value: item?.value,
        itemStyle: { color: color(item?.value), borderColor: selected?.qubits.includes(Number(q)) ? "#142f2a" : "#fff", borderWidth: selected?.qubits.includes(Number(q)) ? 4 : 1 },
      };
    });
    data.push(...[[0, 0], [plotWidth, 300]].map(([x, y], i) => ({ id: `anchor-${i}`, x, y, symbolSize: 0, label: { show: false }, tooltip: { show: false } })));
    const edges = [...links].map(([id, qubits]) => {
      const item = components.find(c => c.id === id);
      return { source: String(qubits[0]), target: String(qubits[1]), componentId: id,
        lineStyle: { color: mode() === "pair" ? color(item?.value) : "#c7d5ce", width: mode() === "pair" ? (id === component ? 7 : 4) : 2, opacity: 1, curveness: 0.06 } };
    });
    if (!map) {
      map = echarts.init(el("cal-map"), null, { renderer: "svg" });
      map.on("click", event => {
        const id = mode() === "pair" ? event.data.componentId : event.data.id;
        if (!components.some(c => c.id === id)) return;
        component = id; el("cal-search").value = ""; chooseComponents();
      });
    }
    map.setOption({ ...chartTheme, animationDurationUpdate: reducedMotion.matches ? 0 : 300,
      tooltip: { trigger: "item", renderMode: "richText", formatter: p => p.dataType === "edge" ? `Q${p.data.source} → Q${p.data.target}` : `${p.name}\n${number(p.value)}` },
      series: [{ type: "graph", layout: "none", roam: true, data, links: edges, label: { show: true, position: "bottom", fontSize: 10 }, emphasis: { focus: "adjacency" }, left: 30, right: 30, top: 35, bottom: 35 }],
    }, true);
    map.resize();
  }
  function renderDetail() {
    const item = components.find(c => c.id === component);
    el("cal-component-title").textContent = item ? `${label(item)}${["single", "pair"].includes(mode()) ? ` · ${el("cal-gate").value.toUpperCase()}` : ""}` : "No component selected";
    el("cal-value").textContent = item?.conflict ? "Conflicting records" : number(item?.value);
    el("cal-explanation").textContent = modes[mode()][1];
    const drawings = {
      readout: '<circle cx="40" cy="42" r="19"/><path d="M60 42h50"/><rect x="110" y="20" width="65" height="44" rx="8"/><path d="M122 50a20 20 0 0 1 40 0m-20-1 14-18"/><text x="34" y="47">q</text>',
      single: '<path d="M15 42h50m65 0h65"/><rect x="65" y="18" width="65" height="48" rx="9"/><text x="77" y="48">Gate</text>',
      pair: '<path d="M15 23h180M15 65h180M105 23v42"/><circle cx="105" cy="23" r="7"/><circle cx="105" cy="65" r="7"/>',
      t1: '<path d="M25 20h60m40 45h60M105 23v35m-6-6 6 6 6-6"/><text x="38" y="43">|1⟩</text><text x="140" y="49">|0⟩</text>',
      t2: '<circle cx="48" cy="42" r="23"/><path d="M48 42l17-16M83 42h35"/><circle cx="155" cy="42" r="23"/><path d="M155 42l17-16m-17 16 20 4m-20-4-9 18m9-18-15-8"/>',
    };
    el("cal-drawing").innerHTML = `<svg viewBox="0 0 210 85" fill="none" stroke="currentColor" stroke-width="2">${drawings[mode()]}</svg>`;
    el("cal-directional").replaceChildren();
    if (item && mode() === "readout") {
      for (const [key, title] of [["prob_meas1_prep0", "Prepared 0 → reported 1"], ["prob_meas0_prep1", "Prepared 1 → reported 0"]]) {
        const distinct = unique(current.map(r => r.qubits[item.id]?.[key]).filter(v => v != null));
        const div = document.createElement("div"), dt = document.createElement("dt"), dd = document.createElement("dd");
        dt.textContent = title; dd.textContent = distinct.length > 1 ? "Conflicting records" : number(distinct[0]);
        div.append(dt, dd); el("cal-directional").append(div);
      }
    }
    const measurements = job ? records[job.ref]?.measurements || [] : [];
    const mappings = item ? measurements.flatMap(m => m.qubits.flatMap((q, i) => item.qubits.includes(q) ? [`Q${q} → bit ${m.bits[i]}`] : [])) : [];
    el("cal-bit-mapping").textContent = job ? (mappings.length ? `${unique(mappings).join("; ")}. Bit 0 is the rightmost bit of the flattened histogram label. Repeated measurements may overwrite bits; mappings include recorded measurement steps.` : "This component has no recorded direct measurement mapping for the selected job.") : "Open from a job histogram to see how physical qubits map to its measured bits.";
    renderHistory(item);
  }
  function renderHistory(item) {
    if (!history) history = echarts.init(el("cal-history"), null, { renderer: "svg" });
    const series = dates.map(at => merge(allForChip().filter(r => r.at === at)).find(c => c.id === component)?.value ?? null);
    const known = series.filter(v => v != null);
    el("cal-history-title").textContent = `${item ? label(item) : "Component"} · ${modes[mode()][0]}`;
    el("cal-history-note").textContent = `${known.length} recorded snapshot${known.length === 1 ? "" : "s"} with a value. ${job ? "History includes this component wherever it was observed on the same chip. " : ""}Snapshots are equally spaced (UTC); dots are observations and empty dates have no comparable value. Select a dot to inspect its snapshot.`;
    history.setOption({ ...chartTheme, animation: !reducedMotion.matches,
      grid: { left: 75, right: 25, top: 25, bottom: 60 },
      tooltip: { trigger: "axis", renderMode: "richText", formatter: params => `${stamp(dates[params[0].dataIndex])} UTC\n${number(params[0].value)}` },
      xAxis: { type: "category", data: dates.map(stamp), axisLabel: { fontSize: 10, rotate: 15 } },
      yAxis: { type: "value", min: 0, name: timed() ? "μs" : "%", axisLabel: { formatter: v => timed() ? v : +(v * 100).toFixed(3) } },
      series: [{ type: "scatter", data: series, symbolSize: (_, p) => dates[p.dataIndex] === el("cal-date").value ? 15 : 9, itemStyle: { color: "#087363" } }],
    }, true);
    history.off("click");
    history.on("click", p => {
      historyExpanded = true; el("cal-date").disabled = false;
      el("cal-date").value = dates[p.dataIndex]; update();
    });
    history.resize();
  }
  function open(fromJob, trigger) {
    job = fromJob; opener = trigger; savedScroll = window.scrollY;
    origin = el("histogram-view").hidden ? "plot-view" : "histogram-view";
    el("plot-view").hidden = true; el("histogram-view").hidden = true;
    el("interval-lens").hidden = true; el("calibration-view").hidden = false;
    el("detail-location").hidden = true;
    el("calibration-parent").hidden = origin !== "histogram-view";
    el("calibration-location").hidden = false;
    el("close-calibration").textContent = origin === "histogram-view" ? "← Back to job histogram" : "← Back to operating range";
    historyExpanded = false; component = null; el("cal-search").value = "";
    options("cal-chip", unique(Object.values(records).map(r => r.backend)), job?.backend || el("backend").value);
    el("cal-chip").disabled = Boolean(job);
    el("cal-mode").value = "readout";
    fillDates(job ? records[job.ref]?.at : null);
    el("calibration-title").focus({ preventScroll: true });
    el("calibration-view").scrollIntoView({ block: "start", behavior: "instant" });
  }
  function close() {
    el("calibration-view").hidden = true; el("calibration-location").hidden = true;
    el("calibration-parent").hidden = true;
    el("detail-location").hidden = origin !== "histogram-view";
    el(origin).hidden = false;
    if (origin === "histogram-view") { renderRun(); hist?.resize(); } else trend.resize();
    opener.focus({ preventScroll: true }); window.scrollTo({ top: savedScroll, behavior: "instant" });
  }
  el("open-calibration").onclick = () => { if (!el("calibration-view").hidden) close(); open(null, el("open-calibration")); };
  el("job-calibration").onclick = () => open(jobs.find(j => j.config === selectedConfig && String(j.rep) === el("rep").value), el("job-calibration"));
  el("close-calibration").onclick = close;
  el("histogram-return").onclick = close;
  el("cal-chip").onchange = () => fillDates();
  for (const id of ["cal-date", "cal-gate"]) el(id).onchange = update;
  el("cal-mode").onchange = () => { el("cal-gate").value = ""; component = null; update(); };
  el("cal-search").oninput = chooseComponents;
  el("cal-component").onchange = () => { component = el("cal-component").value; renderMap(); renderDetail(); };
  el("cal-expand-history").onclick = () => {
    historyExpanded = !historyExpanded;
    el("cal-date").disabled = !historyExpanded;
    if (!historyExpanded) { el("cal-date").value = records[job.ref].at; el("cal-search").value = ""; }
    update();
  };
  // An existing ancestor breadcrumb may also leave calibration.
  el("range-return").addEventListener("click", () => {
    el("calibration-view").hidden = true; el("calibration-location").hidden = true;
    el("calibration-parent").hidden = true;
  });
  new ResizeObserver(() => { if (!el("calibration-view").hidden) { map?.resize(); history?.resize(); } }).observe(el("calibration-view"));
})();
