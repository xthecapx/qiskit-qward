"use strict";

const el = (id) => document.getElementById(id);
const unique = (values) => [...new Set(values)];
const pct = (value) =>
  value == null ? "Unavailable" : (100 * value).toFixed(1) + "%";
const score = (value) => (value == null ? "Unavailable" : value.toFixed(3));
const axes = {
  width: "Logical qubits",
  bits: "Measured bits",
  depth: "Executed ISA depth",
  gates: "Two-qubit gates",
};
const names = {
  "BERNSTEIN-VAZIRANI": "Bernstein–Vazirani",
  "PHASE-ESTIMATION": "Phase estimation",
  "RANDOM-VOLUMETRIC": "Random volumetric",
  VTP: "vTP",
  GROVER: "Grover",
  SHOR: "Shor",
};
const questions = {
  success: [
    "How often did I get an accepted answer?",
    "Success rate",
    "Fraction of all shots that landed in the task’s accepted answer set.",
  ],
  dsr: [
    "Did expected peaks dominate competitors?",
    "Differential Success Rate",
    "Contrast of the average accepted peak against the strongest individual competitor. Compare within the same answer set, measured width and shot budget.",
  ],
  contrast: [
    "Did any observed candidate stand out?",
    "Top-two contrast",
    "Uses the leading counts alone. The rank test gives separate evidence about their ordering.",
  ],
  coarse: [
    "Did accepted answers have the desired balance?",
    "Coarse TVD",
    "Distance to equal target weights across accepted answers, with one remaining bin for all other outcomes. Lower is closer.",
  ],
  tvd: [
    "How far was the full output from the ideal?",
    "Full TVD",
    "Probability mass that would need to move to match the frozen reference. Lower is closer.",
  ],
  hellinger: [
    "How well did the whole distribution agree?",
    "Hellinger fidelity",
    "Squared sum of the geometric means of corresponding observed and reference probabilities. Higher values indicate closer agreement of the classical distributions.",
  ],
  chance: [
    "How much did success improve over guessing?",
    "Chance-corrected success",
    "Success above the uniform-guess baseline, rescaled to the remaining attainable gain and clipped at zero.",
  ],
};
const windowKey = (j) => JSON.stringify([j.campaign, j.window]);
const format = (key, value) => (key === "success" ? pct(value) : score(value));
const lower = (key) => ["coarse", "tvd"].includes(key);
const compareText = (a, b) => a.localeCompare(b, "en", { numeric: true });
let windowJobs, jobs, configs, trend, hist;
const chartTheme = {
  textStyle: { fontFamily: "system-ui, sans-serif", color: "#34554e" },
};

function options(id, values, preferred, label = (v) => v) {
  const select = el(id),
    previous = select.value;
  select.replaceChildren(...values.map((v) => new Option(label(v), v)));
  if (values.includes(preferred)) select.value = preferred;
  else if (values.includes(previous)) select.value = previous;
}
function scope() {
  const backend = snapshot.jobs.filter(
    (j) => j.backend === el("backend").value,
  );
  const windows = unique(backend.map(windowKey)).sort((a, b) =>
    latest(backend.filter((j) => windowKey(j) === b)).localeCompare(
      latest(backend.filter((j) => windowKey(j) === a)),
    ),
  );
  options("window", windows, null, (v) => {
    const [campaign, w] = JSON.parse(v);
    return `W${w} · ${campaign}`;
  });
  windowJobs = backend.filter((j) => windowKey(j) === el("window").value);
  options(
    "algorithm",
    unique(windowJobs.map((j) => j.algorithm)).sort(),
    el("algorithm").value || "QFT",
    (v) => names[v] || v,
  );
  families();
}
function latest(rows) {
  return rows
    .map((j) => j.date)
    .sort()
    .at(-1);
}
function families() {
  const rows = windowJobs.filter((j) => j.algorithm === el("algorithm").value);
  const families = unique(rows.map((j) => j.family)).sort(compareText);
  options(
    "family",
    families,
    families.includes(el("family").value)
      ? el("family").value
      : "SR · round trip",
  );
  sweep();
}
function sweep() {
  jobs = windowJobs.filter(
    (j) =>
      j.algorithm === el("algorithm").value && j.family === el("family").value,
  );
  configs = unique(jobs.map((j) => j.config))
    .map((id) =>
      jobs.filter((j) => j.config === id).sort((a, b) => a.rep - b.rep),
    )
    .sort(
      (a, b) =>
        (a[0].designDepth ?? a[0].width) - (b[0].designDepth ?? b[0].width) ||
        compareText(a[0].config, b[0].config),
    );
  el("coverage").textContent =
    `${configs.length} setting${configs.length === 1 ? "" : "s"} · ${jobs.length} jobs · ${latest(jobs)} UTC`;
  renderSweep(true);
}
function target() {
  const input = el("target");
  return input.validity.valid && input.value !== ""
    ? input.valueAsNumber
    : null;
}
function result(group, key, threshold) {
  if (group.length < 2 || group.some((j) => j[key] == null))
    return "Insufficient evidence";
  if (threshold == null) return "No target";
  const ok = group.map((j) =>
    lower(key) ? j[key] <= threshold : j[key] >= threshold,
  );
  return ok.every(Boolean) ? "Meets" : ok.some(Boolean) ? "Mixed" : "Below";
}
function cells(body, values) {
  const row = body.insertRow();
  for (const value of values) row.insertCell().textContent = value;
}
function renderSweep(selectBoundary = false) {
  const key = el("question").value,
    threshold = target(),
    axis = el("axis").value;
  const states = configs.map((g) => result(g, key, threshold));
  const missed = states.indexOf("Below"),
    first = configs[0][0],
    last = configs.at(-1)[0];
  const bracket =
    missed > 0 &&
    states.slice(0, missed).every((s) => s === "Meets") &&
    states.slice(missed).every((s) => s === "Below");
  let title, detail;
  if (threshold == null) {
    title = "Explore the measured values";
    detail = "Set a target between 0 and 1 to classify the tested settings.";
  } else if (configs.length === 1) {
    title = "One setting tested";
    detail = `${first.config} · ${states[0]}. More settings are needed to locate a boundary.`;
  } else if (bracket) {
    const pass = configs[missed - 1],
      miss = configs[missed];
    title = `${pass[0].config} met the target. ${miss[0].config} missed it.`;
    detail = `${pass[0].config} jobs: ${pass.map((j) => format(key, j[key])).join(" and ")}. ${miss[0].config} jobs: ${miss.map((j) => format(key, j[key])).join(" and ")}.`;
    if (key === "success" && miss.every((j) => j.dsr != null)) {
      detail += ` DSR at ${miss[0].config}: ${miss.map((j) => score(j.dsr)).join(" and ")}.`;
    }
  } else if (states.every((v) => v === "Meets")) {
    title = `Met the target through ${last.config}`;
    detail = "Every tested setting in this family met the requirement.";
  } else if (states.every((v) => v === "Below")) {
    title = "No tested setting met the target";
    detail = `The measured sweep covers ${first.config} through ${last.config}.`;
  } else {
    title = "No single measured cutoff";
    detail =
      "Inspect each setting for reversals, split repetitions or missing evidence.";
  }
  el("headline").textContent = title;
  el("explain").textContent = detail;
  el("metric").textContent =
    questions[key][1] +
    (threshold == null
      ? ""
      : ` ${lower(key) ? "≤" : "≥"} ${format(key, threshold)}`);
  el("meaning").textContent = questions[key][2];
  const body = el("settings-table");
  body.replaceChildren();
  configs.forEach((g, i) =>
    cells(body, [
      g[0].config,
      g[0].width,
      unique(g.map((j) => j.depth)).join(", "),
      g.map((j) => format(key, j[key])).join(" / "),
      states[i],
    ]),
  );
  const dots = jobs
    .filter((j) => j[key] != null)
    .map((j) => ({
      value: [j[axis], j[key]],
      name: j.config,
      config: j.config,
      rep: j.rep,
    }));
  const sweepAxis = first.designDepth !== null ? "depth" : "width";
  const shade =
    bracket && axis === sweepAxis
      ? [
          [
            { xAxis: configs[missed - 1][0][axis] },
            { xAxis: configs[missed][0][axis] },
          ],
        ]
      : [];
  const axisValues = unique(jobs.map((j) => j[axis]));
  trend.setOption({
    ...chartTheme,
    animation: el("motion").checked,
    animationDurationUpdate: 350,
    grid: { left: 58, right: 18, top: 36, bottom: 52 },
    tooltip: {
      trigger: "item",
      renderMode: "richText",
      formatter: (p) =>
        `${p.data.config} · job ${p.data.rep}\n${questions[key][1]}  ${format(key, p.value[1])}\n${axes[axis]}  ${p.value[0]}`,
    },
    xAxis: {
      type: "value",
      min: Math.max(0, Math.min(...axisValues) - 1),
      max: Math.max(...axisValues) + 1,
      minInterval: 1,
      name: axes[axis],
      nameLocation: "middle",
      nameGap: 29,
    },
    yAxis: {
      type: "value",
      name: questions[key][1],
      nameLocation: "end",
      nameTextStyle: { align: "left" },
      min: 0,
      max: 1,
      interval: 0.25,
      axisLabel: {
        formatter: (v) =>
          key === "success" ? Math.round(v * 100) + "%" : String(v),
      },
    },
    series: [
      {
        name: questions[key][1],
        type: "scatter",
        symbolSize: 11,
        itemStyle: { color: "#087363" },
        data: dots,
        markArea: {
          silent: true,
          itemStyle: { color: "rgba(31,134,112,.10)" },
          data: shade,
        },
        markLine: {
          silent: true,
          symbol: "none",
          label: { show: false },
          lineStyle: { color: "#9a5b12", type: "dashed" },
          data: threshold == null ? [] : [{ yAxis: threshold }],
        },
      },
    ],
  });
  const preferred = selectBoundary
    ? configs[bracket ? missed : 0][0].config
    : el("config").value;
  options(
    "config",
    configs.map((g) => g[0].config),
    preferred,
  );
  repetitions();
}
function repetitions(preferred) {
  const group = configs.find((g) => g[0].config === el("config").value);
  options(
    "rep",
    group.map((j) => String(j.rep)),
    preferred,
    (v) => "Job " + v,
  );
  renderRun();
}
function renderRun() {
  const job = jobs.find(
    (j) => j.config === el("config").value && String(j.rep) === el("rep").value,
  );
  const { accepted, ideal } = snapshot.refs[job.ref];
  const allowed = new Set(accepted),
    counts = job.counts;
  const labels = unique([
    ...Object.keys(counts),
    ...allowed,
    ...Object.keys(ideal || {}),
  ]).sort((a, b) => (counts[b] || 0) - (counts[a] || 0) || compareText(a, b));
  const key = el("question").value,
    coarse = key === "coarse",
    blind = key === "contrast";
  const probability = (label) => (counts[label] || 0) / job.shots;
  const reference = (label) =>
    coarse ? (allowed.has(label) ? 1 / allowed.size : 0) : ideal?.[label] || 0;
  const hasRef = coarse
    ? allowed.size > 0
    : ideal != null && Object.keys(ideal).length > 0;
  const ranked = coarse ? labels.filter((k) => allowed.has(k)) : labels;
  const limit = Number(el("bins").value);
  const selected = blind
    ? ranked.slice(0, limit)
    : unique([
        ...ranked
          .filter((k) => allowed.has(k) || reference(k) > 0)
          .slice(0, Math.ceil(limit / 2)),
        ...ranked.slice(0, limit),
        ...ranked.filter((k) => allowed.has(k)),
      ]).slice(0, limit);
  const rest = labels.filter((k) => !selected.includes(k)),
    display = [...selected];
  if (rest.length) display.push("Other outcomes");
  const values = display.map((k) =>
    k === "Other outcomes"
      ? rest.reduce((n, v) => n + probability(v), 0)
      : probability(k),
  );
  const markers = display.map((k) =>
    k === "Other outcomes"
      ? rest.reduce((n, v) => n + reference(v), 0)
      : reference(k),
  );
  el("run-context").textContent =
    `${job.description} · ${job.shots.toLocaleString()} shots · ${job.bits} measured bits · ISA depth ${job.depth} · ${job.date} UTC`;
  const numbers = el("numbers");
  numbers.replaceChildren();
  for (const [name, value] of [
    ["Task success", pct(job.success)],
    ["DSR · peak dominance", score(job.dsr)],
    ["Full TVD", score(job.tvd)],
    ["Guessing baseline", pct(job.baseline)],
  ]) {
    const cell = document.createElement("div"),
      label = document.createElement("span"),
      val = document.createElement("b");
    label.textContent = name;
    val.textContent = value;
    cell.append(label, val);
    numbers.append(cell);
  }
  const competitor = labels.find((k) => !allowed.has(k));
  const notes = {
    dsr: `Mean accepted peak ${allowed.size ? pct(labels.filter((k) => allowed.has(k)).reduce((s, k) => s + probability(k), 0) / allowed.size) : "unavailable"}. Strongest individual competitor ${competitor ?? "none"} (${pct(competitor ? probability(competitor) : 0)}).`,
    contrast: `Leading counts: ${labels
      .slice(0, 2)
      .map((k) => counts[k] || 0)
      .join(
        " / ",
      )}. Rank ${job.rank || "unavailable"} · p = ${job.rankP || "unavailable"} · α = ${job.alpha || "unavailable"}.`,
    coarse:
      "Orange markers show equal target weights for accepted answers. The full table below retains the original ideal probabilities.",
  };
  const legend = blind
    ? "Outcomes ordered by observed counts."
    : `Teal bars are accepted labels. ${hasRef ? "Orange diamonds show " + (coarse ? "the task target." : (job.exact ? "the exact" : "the sampled") + " ideal reference.") : "Reference unavailable."}`;
  el("hist-note").textContent = [
    notes[key],
    legend,
    rest.length ? "Other outcomes combines omitted labels." : "",
  ]
    .filter(Boolean)
    .join(" ");
  const height = Math.max(240, display.length * 26 + 65);
  el("hist").style.height = height + "px";
  hist.resize();
  hist.setOption({
    ...chartTheme,
    animation: el("motion").checked,
    animationDurationUpdate: 350,
    grid: { left: 10, right: 20, top: 12, bottom: 36, containLabel: true },
    tooltip: {
      trigger: "axis",
      renderMode: "richText",
      valueFormatter: (v) => (Array.isArray(v) ? pct(v[0]) : pct(v)),
    },
    xAxis: {
      type: "value",
      min: 0,
      max: 1,
      axisLabel: { formatter: (v) => Math.round(v * 100) + "%" },
    },
    yAxis: {
      type: "category",
      inverse: true,
      data: display,
      axisLabel: { interval: 0, fontFamily: "monospace", fontSize: 11 },
    },
    series: [
      {
        name: "Observed",
        type: "bar",
        barMaxWidth: 13,
        data: display.map((k, i) => ({
          name: k,
          value: values[i],
          itemStyle: {
            color: blind ? "#387b9a" : allowed.has(k) ? "#087363" : "#a2b6b2",
          },
        })),
      },
      {
        name: coarse ? "Task target" : "Ideal",
        type: "scatter",
        symbol: "diamond",
        symbolSize: 10,
        itemStyle: { color: "#bd6617" },
        data: hasRef && !blind ? markers.map((v, i) => [v, display[i]]) : [],
      },
    ],
  });
  const body = el("outcomes");
  body.replaceChildren();
  labels.forEach((k) =>
    cells(body, [
      k,
      counts[k] || 0,
      pct(probability(k)),
      ideal ? pct(ideal[k] || 0) : "Unavailable",
      accepted ? (allowed.has(k) ? "Yes" : "No") : "Unavailable",
    ]),
  );
}
function init() {
  el("motion").checked = !matchMedia("(prefers-reduced-motion: reduce)")
    .matches;
  trend = echarts.init(el("trend"), null, { renderer: "svg" });
  hist = echarts.init(el("hist"), null, { renderer: "svg" });
  options("backend", unique(snapshot.jobs.map((j) => j.backend)), "ibm_fez");
  options(
    "question",
    Object.keys(questions),
    "success",
    (k) => questions[k][0],
  );
  el("target").value = "0.5";
  el("source").textContent =
    `Exported observations · ${snapshot.jobs.length} jobs · ${unique(snapshot.jobs.map((j) => j.algorithm)).length} algorithms · ECharts 5.6.0`;
  for (const id of ["backend", "window"]) el(id).onchange = scope;
  el("algorithm").onchange = families;
  el("family").onchange = sweep;
  el("question").onchange = () => {
    el("target").value = el("question").value === "success" ? "0.5" : "";
    renderSweep();
  };
  el("axis").onchange = () => renderSweep();
  el("target").oninput = () => renderSweep();
  el("config").onchange = () => repetitions();
  el("rep").onchange = renderRun;
  el("bins").onchange = renderRun;
  el("motion").onchange = () => renderSweep();
  trend.on("click", (p) => {
    if (p.data?.config) {
      el("config").value = p.data.config;
      repetitions(String(p.data.rep));
      el("config").scrollIntoView({ block: "center" });
    }
  });
  new ResizeObserver(() => {
    trend.resize();
    hist.resize();
  }).observe(document.querySelector("main"));
  scope();
}
try {
  init();
} catch (error) {
  el("error").hidden = false;
  el("error").textContent =
    "The experiment snapshot could not be opened. Build the dashboard and open its dist/index.html.";
  console.error(error);
}
