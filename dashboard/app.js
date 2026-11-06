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
const jobKey = (j) => JSON.stringify([j.ref, j.rep]);
const jobTime = (j) => j.finishedAt ? new Date(j.finishedAt).toLocaleString("en-GB", {
  day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit",
  second: "2-digit", timeZone: "UTC", hourCycle: "h23",
}) + " UTC" : "Time unavailable";
const format = (key, value) => (key === "success" ? pct(value) : score(value));
const lower = (key) => ["coarse", "tvd"].includes(key);
const compareText = (a, b) => a.localeCompare(b, "en", { numeric: true });
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
let windowJobs, jobs, configs, selectedConfig, trend, hist;
const chartTheme = {
  textStyle: { fontFamily: "system-ui, sans-serif", color: "#34554e" },
};

const metadataIcons = {
  chip: '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4m6-4v4M9 18v4m6-4v4M2 9h4m-4 6h4m12-6h4m-4 6h4"/>',
  shots: '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><path d="M12 2v3m0 14v3M2 12h3m14 0h3"/>',
  bits: '<rect x="3" y="5" width="7" height="14" rx="2"/><path d="M16 8l2-2v12m-3 0h6"/>',
  depth: '<path d="M3 6h18M3 12h18M3 18h18"/><rect x="6" y="3" width="4" height="6" rx="1"/><rect x="14" y="9" width="4" height="6" rx="1"/><rect x="6" y="15" width="4" height="6" rx="1"/>',
  runs: '<path d="M20 7a9 9 0 0 0-15-2L2 8m0-6v6h6m-4 9a9 9 0 0 0 15 2l3-3m0 6v-6h-6"/>',
  qubits: '<circle cx="5" cy="12" r="3"/><circle cx="18" cy="5" r="3"/><circle cx="18" cy="19" r="3"/><path d="m8 10 7-4m-7 8 7 4"/>',
  experiment: '<path d="M4 3h10l6 6v12H4zM14 3v6h6M8 13h8m-8 4h5"/>',
  date: '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M7 2v6m10-6v6M3 11h18m-13 4h2m4 0h2"/>',
  group: '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
};

function renderMetadata(id, entries) {
  el(id).replaceChildren(...entries.map(([icon, label, value]) => {
    const item = document.createElement("div");
    const term = document.createElement("dt"), detail = document.createElement("dd");
    term.innerHTML = `<svg aria-hidden="true" focusable="false" viewBox="0 0 24 24">${metadataIcons[icon]}</svg>`;
    term.append(document.createTextNode(label));
    detail.textContent = value;
    item.append(term, detail);
    return item;
  }));
}

function jobMetadata(job) {
  return [
    ["chip", "Chip", job.backend],
    ["shots", "Shots", job.shots.toLocaleString()],
    ["bits", "Measured bits", job.bits],
    ["depth", "ISA depth", job.depth],
    ["experiment", "Experiment", job.config],
    ["date", "Measured at", jobTime(job)],
  ];
}

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
  const windows = new Map();
  for (const job of backend) {
    const id = windowKey(job);
    if (!windows.has(id)) windows.set(id, []);
    windows.get(id).push(job);
  }
  const times = new Map([...windows].map(([id, rows]) => [
    id, rows.map(j => j.finishedAt).filter(Boolean).sort(),
  ]));
  const ordered = [...windows.keys()].sort((a, b) =>
    (times.get(b).at(-1) || "").localeCompare(times.get(a).at(-1) || ""),
  );
  options("window", ["", ...ordered], null, id => {
    if (id === "") return "All periods";
    const dates = times.get(id);
    if (!dates.length) return "Date unavailable";
    const start = dates[0], end = dates.at(-1);
    const date = value => new Date(value).toLocaleDateString("en-US", {
      month: "short", day: "numeric", year: "numeric", timeZone: "UTC",
    });
    const time = value => new Date(value).toLocaleTimeString("en-GB", {
      hour: "2-digit", minute: "2-digit", timeZone: "UTC", hourCycle: "h23",
    });
    const stamp = value => `${date(value)} · ${time(value)}`;
    const finish = date(start) === date(end) ? time(end) : stamp(end);
    return `${stamp(start)}–${finish}`;
  });
  windowJobs = el("window").value === "" ? backend : windows.get(el("window").value);
  options(
    "algorithm",
    unique(windowJobs.map((j) => j.algorithm)).sort(),
    el("algorithm").value || "QFT",
    (v) => names[v] || v,
  );
  experimentGroups();
}
function experimentGroups() {
  const rows = windowJobs.filter((j) => j.algorithm === el("algorithm").value);
  const families = unique(rows.map((j) => j.family)).sort(compareText);
  options(
    "family",
    ["", ...families],
    "",
    (group) => group || "All experiments",
  );
  sweep();
}
function sweep() {
  jobs = windowJobs.filter(
    (j) =>
      j.algorithm === el("algorithm").value &&
      (!el("family").value || j.family === el("family").value),
  );
  configs = unique(jobs.map((j) => j.config))
    .map((id) =>
      jobs.filter((j) => j.config === id).sort((a, b) => (a.finishedAt || "").localeCompare(b.finishedAt || "") || a.rep - b.rep),
    )
    .sort(
      (a, b) =>
        (a[0].designDepth ?? a[0].width) - (b[0].designDepth ?? b[0].width) ||
        compareText(a[0].config, b[0].config),
    );
  el("coverage").textContent =
    `${configs.length} experiment${configs.length === 1 ? "" : "s"} · ${jobs.length} jobs`;
  el("settings-coverage").textContent = `${el("coverage").textContent} · ${unique(jobs.map(windowKey)).length} collection(s)`;
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
  return row;
}
function showJobLens(point) {
  const lens = el("interval-lens");
  const [axisValue, value] = point.value;
  const key = el("question").value;
  const study = point.family
    .replace(/^SP · period (\d+)$/, "Period detection at increasing widths (period $1)")
    .replace(/^PV · period (\d+)$/, "Period variation at fixed width (period $1)")
    .replace(/^SR · round trip$/, "Round-trip scaling");
  el("lens-group-row").hidden = !study || study === point.config;
  el("lens-group").textContent = study;
  el("lens-circuit").textContent = point.description || names[el("algorithm").value] || el("algorithm").value;
  const axis = el("axis").value;
  el("lens-axis").textContent = axis === "width" ? "Total logical qubits" : axes[axis];
  el("lens-axis-value").textContent = axisValue;
  el("lens-rep").textContent = point.rep;
  el("lens-time").textContent = jobTime(point);
  el("lens-interval").hidden = !point.interval;
  el("lens-value").hidden = Boolean(point.interval);
  if (point.interval) {
    const [low, high] = point.interval;
    el("lens-heading").textContent = "99% confidence interval";
    el("lens-high").textContent = pct(high);
    el("lens-mean").textContent = pct(value);
    el("lens-low").textContent = pct(low);
    const meanPosition = high > low ? (high - value) / (high - low) : 0.5;
    lens.style.setProperty("--mean-position", `${100 * Math.max(0, Math.min(1, meanPosition))}%`);
  } else {
    el("lens-heading").textContent = questions[key][1];
    el("lens-value").textContent = format(key, value);
  }
  lens.hidden = false;
  const [x, y] = trend.convertToPixel({ gridIndex: 0 }, point.value);
  const width = lens.offsetWidth, height = lens.offsetHeight;
  const left = x + width + 16 <= trend.getWidth() ? x + 16 : x - width - 16;
  const top = Math.max(0, Math.min(y - height / 2, trend.getHeight() - height));
  const boundedLeft = Math.max(0, Math.min(left, trend.getWidth() - width));
  lens.style.left = `${boundedLeft}px`;
  lens.style.top = `${top}px`;
  lens.style.transformOrigin = `${x - boundedLeft}px ${y - top}px`;
}
function renderSweep(selectBoundary = false) {
  el("interval-lens").hidden = true;
  const key = el("question").value,
    threshold = target(),
    axis = el("axis").value;
  const states = configs.map((g) => result(g, key, threshold));
  const missed = states.indexOf("Below"),
    first = configs[0][0],
    last = configs.at(-1)[0];
  const combined = unique(jobs.map((j) => j.family)).length > 1;
  const bracket =
    !combined &&
    missed > 0 &&
    states.slice(0, missed).every((s) => s === "Meets") &&
    states.slice(missed).every((s) => s === "Below");
  let title, detail;
  if (threshold == null) {
    title = "Explore the measured values";
    detail = "Set a target between 0 and 1 to classify the tested settings.";
  } else if (combined) {
    const passed = states.filter((state) => state === "Meets").length;
    title = `${passed} of ${configs.length} experiments met the target`;
    detail = "Each experiment is assessed across all jobs in the selected periods. Choose a group of related experiments to inspect its measured boundary.";
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
  const metric = questions[key][1];
  el("plot-title").textContent = `${metric} by ${axes[axis].toLowerCase()}`;
  renderMetadata("scope-caption", [
    ["chip", "Chip", el("backend").value],
    ["date", "Periods", el("window").selectedOptions[0].text],
    ["group", "Experiments", el("family").selectedOptions[0].text],
  ]);
  el("target-caption").textContent = threshold == null
    ? " · No target set"
    : ` · Target ${lower(key) ? "≤" : "≥"} ${format(key, threshold)}`;
  el("target-label").textContent = `${lower(key) ? "Maximum" : "Minimum"} ${metric.toLowerCase()}`;
  el("target-help").textContent = (key === "success" ? "Enter a fraction, e.g. 0.5 for 50%." : "Enter a value from 0 to 1.") + " Leave blank to explore without a cutoff.";
  const showIntervals = key === "success";
  el("point-guide").textContent = showIntervals
    ? "Each point is one job’s accepted-shot fraction. Hover to magnify its mean and 99% confidence bounds."
    : "Each point is one job.";
  el("point-guide").textContent += ` Showing ${unique(jobs.map(windowKey)).length} collection(s); dates are in UTC. Values are individual jobs, not averages.`;
  const body = el("settings-table");
  body.replaceChildren();
  configs.forEach((group, i) => {
    const row = cells(body, [
      group[0].config, group[0].family, group[0].width,
      unique(group.map(job => job.depth)).join(", "),
    ]);
    const values = row.insertCell();
    for (const job of group) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "job-value";
      button.textContent = format(key, job[key]);
      button.setAttribute("aria-label", `${job.config}, repetition ${job.rep}, ${jobTime(job)}, ${button.textContent}. Open histogram`);
      button.title = `Run ${job.rep} · ${jobTime(job)}`;
      button.onclick = () => openJob(job, button);
      values.append(button);
    }
    row.insertCell().textContent = states[i];
  });
  const dots = jobs
    .filter((j) => j[key] != null)
    .map((j) => ({
      value: [j[axis], j[key]],
      name: `${j.config} · job ${j.rep}`,
      family: j.family,
      description: j.countingQubits != null && j.width === j.countingQubits + 1
        ? `${j.countingQubits} counting qubits + 1 ancilla`
        : j.description,
      config: j.config,
      rep: j.rep,
      ref: j.ref,
      finishedAt: j.finishedAt,
      campaign: j.campaign,
      window: j.window,
      interval: showIntervals ? j.successInterval : null,
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
    animation: !reducedMotion.matches,
    animationDurationUpdate: 350,
    grid: { left: 58, right: 18, top: 36, bottom: 52 },
    tooltip: { show: false },
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
      interval: showIntervals ? 0.1 : 0.25,
      axisLabel: {
        formatter: (v) =>
          key === "success" ? Math.round(v * 100) + "%" : String(v),
      },
    },
    series: [
      {
        name: questions[key][1],
        type: "scatter",
        symbolSize: showIntervals ? 7 : 11,
        z: 3,
        itemStyle: { color: "#087363" },
        emphasis: { scale: false, itemStyle: { color: "#c44d16", opacity: 1 } },
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
  if (selectBoundary || !configs.some(([job]) => job.config === selectedConfig)) {
    selectedConfig = configs[bracket ? missed : 0][0].config;
  }
  repetitions();
}
function renderExperimentChoices() {
  const words = el("experiment-search").value.trim().toLowerCase().split(/\s+/);
  const matches = configs.map(([job]) => job).filter(job => {
    const text = [job.config, job.family, job.description].join(" ").toLowerCase();
    return words.every(word => text.includes(word));
  }).sort((a, b) => Number(b.config === selectedConfig) - Number(a.config === selectedConfig));
  el("picker-scope").textContent = `${names[jobs[0].algorithm] || jobs[0].algorithm} · ${jobs[0].backend}`;
  el("match-count").textContent = `${matches.length} of ${configs.length} experiments`;
  el("picker-empty").hidden = matches.length > 0;
  el("experiment-results").replaceChildren(...matches.map((job, index) => {
    const button = document.createElement("button");
    button.type = "button";
    const selected = job.config === selectedConfig;
    button.setAttribute("aria-current", String(selected));
    const heading = document.createElement("span");
    heading.className = "experiment-choice-heading";
    const name = document.createElement("strong");
    name.textContent = job.description || job.config;
    heading.append(name);
    if (selected) {
      const badge = document.createElement("span");
      badge.className = "experiment-selected";
      badge.textContent = "✓ Selected";
      heading.append(badge);
    }
    const detail = document.createElement("span");
    detail.className = "experiment-choice-context";
    detail.textContent = unique([job.family, job.config].filter(Boolean)).join(" · ");
    const metadata = document.createElement("span");
    metadata.className = "experiment-choice-metadata";
    const runCount = configs.find(group => group[0].config === job.config).length;
    for (const [icon, text] of [
      ["qubits", job.countingQubits != null ? `${job.width} qubits (${job.countingQubits} counting + ${job.width - job.countingQubits} auxiliary)` : `${job.width} logical qubits`],
      ["depth", `ISA depth ${job.depth}`],
      ["runs", `${runCount} recorded run${runCount === 1 ? "" : "s"}`],
    ]) {
      const item = document.createElement("span");
      item.innerHTML = `<svg aria-hidden="true" focusable="false" viewBox="0 0 24 24">${metadataIcons[icon]}</svg>`;
      item.append(document.createTextNode(text));
      metadata.append(item);
    }
    button.append(heading, detail, metadata);
    button.onkeydown = event => {
      const buttons = el("experiment-results").children;
      if (event.key === "ArrowDown") { event.preventDefault(); buttons[(index + 1) % buttons.length].focus(); }
      if (event.key === "ArrowUp") { event.preventDefault(); index ? buttons[index - 1].focus() : el("experiment-search").focus(); }
    };
    button.onclick = () => {
      selectedConfig = job.config;
      repetitions();
      el("experiment-picker").close();
    };
    return button;
  }));
  el("experiment-results").scrollTop = 0;
}
function repetitions(preferred) {
  const group = configs.find((g) => g[0].config === selectedConfig);
  el("choose-config").textContent = group[0].description || selectedConfig;
  options(
    "rep",
    group.map(jobKey),
    preferred,
    (v) => {
      const job = group.find(j => jobKey(j) === v);
      return `Run ${job.rep} · ${jobTime(job)}`;
    },
  );
  renderRun();
}
function selectedJob() {
  return jobs.find(j => j.config === selectedConfig && jobKey(j) === el("rep").value);
}
function renderRun() {
  const job = selectedJob();
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
  const task = {
    GROVER: "Grover searches for a marked bitstring.",
    GHZ: "The GHZ circuit prepares a superposition of the all-zero and all-one strings.",
    "BERNSTEIN-VAZIRANI": "Bernstein–Vazirani recovers the hidden bitstring.",
    QFT: "This experiment checks the measured output of a Fourier-transform circuit.",
    "PHASE-ESTIMATION": "Phase estimation encodes the gate’s phase in the counting register.",
    VTP: "The protocol tests whether the payload is recovered after teleportation.",
    "RANDOM-VOLUMETRIC": "This experiment tests output recovery as a random circuit grows in size or depth.",
    SHOR: "This circuit samples phase peaks associated with the order-finding step of Shor’s algorithm.",
  };
  const acceptedText = allowed.size > 0 && allowed.size <= 4
    ? `Accepted outcomes: ${[...allowed].join(", ")}.`
    : allowed.size ? `${allowed.size} outcomes satisfy the task.` : "Accepted outcomes unavailable.";
  el("experiment-description").textContent = [task[job.algorithm], job.description, acceptedText].filter(Boolean).join(" ");
  const threshold = target();
  const value = job[key];
  let decision = "";
  if (value != null && threshold != null) {
    const passes = lower(key) ? value <= threshold : value >= threshold;
    decision = ` · ${passes ? "meets" : "misses"} the ${format(key, threshold)} target`;
  }
  el("selected-label").textContent = `${job.config} · job ${job.rep}`;
  el("selected-result").textContent = `${questions[key][1]} ${format(key, value)}${decision}`;
  renderMetadata("selected-context", jobMetadata(job));
  if (el("histogram-view").hidden) return;
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
  renderMetadata("run-context", jobMetadata(job));
  el("bins-help").textContent = `${selected.length} individual bitstring${selected.length === 1 ? "" : "s"} shown${rest.length ? `; ${rest.length.toLocaleString()} more grouped as “Other outcomes”` : "; no outcomes grouped"}. ${blind ? "Largest observed counts are shown first." : "Expected outcomes and largest observed counts are prioritized."} This changes only the chart; all shots contribute to the metrics.`;
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
  el("hist-chart-title").textContent = blind || !hasRef ? "Observed outcomes" : "Observed and expected outcomes";
  const height = Math.max(240, display.length * 26 + 65);
  el("hist").style.height = height + "px";
  if (!hist) hist = echarts.init(el("hist"), null, { renderer: "svg" });
  hist.resize();
  hist.setOption({
    ...chartTheme,
    animation: !reducedMotion.matches,
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
function selectFindingTab(selected) {
  for (const tab of document.querySelectorAll(".finding-tabs button")) {
    const active = tab === selected;
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    el(tab.getAttribute("aria-controls")).hidden = !active;
  }
}

let returnFocus, rangeScroll;
function setJobView(open) {
  el("plot-view").hidden = open;
  el("range-location").hidden = open;
  el("histogram-view").hidden = !open;
  el("range-return").hidden = !open;
  el("detail-location").hidden = !open;
  el("interval-lens").hidden = true;
}
function openJob(job, opener) {
  returnFocus = opener;
  rangeScroll = window.scrollY;
  setJobView(true);
  el("histogram-title").textContent = el("algorithm").selectedOptions[0].text + " measurement outcomes";
  selectedConfig = job.config;
  repetitions(jobKey(job));
  el("histogram-title").focus({ preventScroll: true });
  el("range-return").scrollIntoView({ block: "nearest" });
}
function init() {
  el("adjust-plot").onclick = () => el("plot-settings").showModal();
  const tabs = [...document.querySelectorAll(".finding-tabs button")];
  tabs.forEach((tab, index) => {
    tab.onclick = () => selectFindingTab(tab);
    tab.onkeydown = (event) => {
      const nextIndex = {
        ArrowLeft: (index + tabs.length - 1) % tabs.length,
        ArrowRight: (index + 1) % tabs.length,
        Home: 0,
        End: tabs.length - 1,
      }[event.key];
      if (nextIndex === undefined) return;
      event.preventDefault();
      selectFindingTab(tabs[nextIndex]);
      tabs[nextIndex].focus();
    };
  });
  trend = echarts.init(el("trend"), null, { renderer: "svg" });
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
  el("algorithm").onchange = experimentGroups;
  el("family").onchange = sweep;
  el("question").onchange = () => {
    el("target").value = el("question").value === "success" ? "0.5" : "";
    renderSweep();
  };
  el("axis").onchange = () => renderSweep();
  el("target").oninput = () => renderSweep();
  el("choose-config").onclick = () => {
    el("experiment-search").value = "";
    renderExperimentChoices();
    el("experiment-picker").showModal();
    el("experiment-search").focus();
  };
  el("close-picker").onclick = () => el("experiment-picker").close();
  el("reset-experiment-search").onclick = () => {
    el("experiment-search").value = "";
    renderExperimentChoices();
    el("experiment-search").focus();
  };
  el("experiment-search").oninput = renderExperimentChoices;
  el("experiment-search").onkeydown = event => {
    const first = el("experiment-results").firstElementChild;
    if (!first) return;
    if (event.key === "ArrowDown") { event.preventDefault(); first.focus(); }
    if (event.key === "ArrowUp") { event.preventDefault(); el("experiment-results").lastElementChild.focus(); }
    if (event.key === "Enter") { event.preventDefault(); first.click(); }
  };
  el("rep").onchange = renderRun;
  el("bins").onchange = renderRun;
  trend.on("mouseover", { seriesType: "scatter" }, ({ data }) => {
    showJobLens(data);
  });
  trend.on("mouseout", { seriesType: "scatter" }, () => {
    el("interval-lens").hidden = true;
  });
  el("range-return").onclick = () => {
    setJobView(false);
    trend.resize();
    returnFocus.focus({ preventScroll: true });
    window.scrollTo({ top: rangeScroll, behavior: "instant" });
  };
  trend.on("click", { seriesType: "scatter" }, ({ data }) => openJob(data, el("plot-title")));
  new ResizeObserver(() => {
    if (el("histogram-view").hidden) trend.resize();
    else hist?.resize();
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
