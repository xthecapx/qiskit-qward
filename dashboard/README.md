# QPU Benchmark · QWARD dashboard

A static explorer for experiment results and historical QPU calibration, built with **Svelte 5, TypeScript and Vite**, with tree-shaken Apache ECharts 5.6. There is no backend: a Python step validates the exported tables into JSON, and Vite produces a static site for local use or GitHub Pages at `/qiskit-qward/dashboard/`.

## Quick start

From the **repository root**, with Python 3.11+ and Node.js 22.12+:

```sh
npm ci --prefix dashboard          # once, or after package-lock.json changes
python3 dashboard/build.py         # validate tables → dashboard/public/data/*.json
npm run dev --prefix dashboard     # http://localhost:5173 with live reload
```

For the production build, as deployed:

```sh
npm run build --prefix dashboard   # → dashboard/dist/
npm run preview --prefix dashboard # http://127.0.0.1:8774/
```

The page fetches its data, so it must be served over HTTP; opening `dist/index.html` from disk does not work. Rerun `build.py` after the exported tables change; the dev server reloads source edits automatically.

## Architecture

```
build.py (stdlib)            Validated tables → public/data/index.json (jobs + references)
                                              → public/data/calibrations.json (loaded on demand)
src/lib/data/                Types mirroring build.py output, schema-checked fetch
src/lib/domain/              Pure logic: scope/periods, sweep verdicts and boundary,
                             histogram binning, calibration merge/schematic, formatting
src/lib/stats/               Pure statistics (Wilson interval); add new estimators here
src/lib/state/               Explorer + CalibrationState (Svelte runes), URL hash sync
src/lib/ui/                  Select, Dialog, Tabs, Metadata, Icon
src/lib/charts/              EChart wrapper (init, resize, dispose, reduced motion) + palette
src/views/{range,job,calibration}/   One folder per screen
tests/unit/                  Vitest for domain/stats/state (no DOM)
tests/e2e/                   Playwright for the browser flows below
```

Rules that keep the code consistent as features are added:

- **Calculations stay in `lib/domain` and `lib/stats`.** They are plain TypeScript with unit tests. Components only display results.
- **State lives in `Explorer`.** Choices the user makes are stored as-is, and the effective values are `$derived`. If a choice is unavailable in the current scope, the view falls back to a default and keeps the choice for later. Views never read the DOM to get state.
- **Charts go through `EChart.svelte`.** A new chart type is registered once in `lib/charts/echarts.ts`, which keeps the bundle tree-shaken.
- **Heavy or publication statistics are computed in Python at build time** and shipped as data. The browser computes only what reacts to interaction (thresholds, filters, interval levels), so the numbers on the page match the thesis.
- **The URL hash holds the view state.** Every view can be shared and reloaded as a link, e.g. `#a=GROVER&q=dsr&v=job&c=S4-1&r=<ref>:2`. Add new state fields in `lib/state/url.ts`.
- **Change data shapes in three places:** `build.py`, `src/lib/data/types.ts`, and `SCHEMA_VERSION` (bump it in both files). The page refuses data with a mismatched schema version.

## Data inputs

`build.py --tables DIR --data-out DIR` reads, by default, `qward/examples/papers/dataset_out/`. The tables must all come from the **same export**:

| File | Contents |
| --- | --- |
| `dataset-001.csv`, `dataset-002.csv`, … | One row per completed job: circuit information, counts, metrics, timing, and execution calibration summaries |
| `dataset_manifest.json` | Ordered CSV parts, byte sizes, and per-part/total row counts |
| `references.csv` | Frozen accepted labels and ideal probabilities, joined by `run_id` |
| `calibrations.csv` | Used-qubit preparation properties, unique gate operations, and ordered measurement-to-classical-bit mappings, joined by `run_id` |

Exports split the dataset into files of at most 20 MB each. Every part repeats the column header, and no job row is split across parts. The builder reads every part named in the manifest. It refuses missing parts, inconsistent headers, and sizes or counts that don't match the manifest. A legacy `dataset.csv` is accepted when no manifest exists. It also checks that jobs are unique, that counts sum to the shot total, that accepted-set sizes match, that ideal probabilities sum to 1, and that calibration joins match their jobs and chip. Older exports without `calibrations.csv` still build; the calibration view then shows a message that calibration data is unavailable.

Only selected scientific fields are published. Raw journal paths, account details, batch IDs, and IBM job IDs are left out.

## Updating the measurements

Exporting and testing require the QWARD Python environment. If it is not already installed, create a development environment from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]" pytest
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Installation may need network access; building the dashboard from existing tables does not.

Once completed jobs have been recovered to the local experiment records, export and rebuild:

```sh
python -m qward.examples.papers.dataset_export \
  --backend ibm_fez \
  --out qward/examples/papers/dataset_out
python3 dashboard/build.py
```

If you use the campaign wrapper and have `uv` installed, the equivalent export command is:

```sh
./qward/examples/papers/run_campaign.sh export
```

Export reads local journals and their `artifacts/.../prepared.json` files; it does not submit jobs or query IBM. Review `dataset_out/export_report.json` for errors and warnings. Keep all three tables together when committing updated data, then rerun `python3 dashboard/build.py`. The wrapper exports completed records across periods; it is not restricted to one campaign window.

For collecting and recovering new QPU results, follow the [IBM Fez experiment plan](../docs/ibm_fez_dataset_plan.md) and the [campaign driver](../qward/examples/papers/run_campaign.sh). Those operations have their own IBM credentials and usage requirements. They are separate from running the dashboard.

## Verification during development

None of these submit IBM jobs. One-time setup, from the repository root:

```sh
npm ci --prefix dashboard
npx --prefix dashboard playwright install chromium   # browser for the e2e tests
```

Full check before pushing. These are the same steps CI runs:

```sh
cd dashboard
python3 build.py      # validate tables → public/data/*.json
npm run check         # svelte-check / TypeScript, warnings fail
npm test              # Vitest unit tests for domain, stats and URL state
npm run build         # production build → dist/
npm run e2e           # Playwright browser flows against dist/
cd ..
uv run -m pytest tests/test_calibration_tables.py tests/test_dataset_export.py -q
uvx black --check dashboard/build.py tests/
```

To reproduce the GitHub Pages layout, serve the build under the same subpath:

```sh
mkdir -p /tmp/pages/qiskit-qward
npm run build --prefix dashboard -- --outDir /tmp/pages/qiskit-qward/dashboard
python3 -m http.server 8790 --bind 127.0.0.1 --directory /tmp/pages
# open http://127.0.0.1:8790/qiskit-qward/dashboard/
```

`tests/e2e/flows.spec.ts` automates these flows. Any UI change should keep them passing:

1. Change the algorithm, question, and measurement period; open a job's histogram and return.
2. Search the experiment picker and select a result by keyboard. Escape keeps the previous selection and returns focus.
3. Change Chart detail and confirm omitted bitstrings are grouped while the metrics and the full table stay unchanged.
4. Open Inspect run calibration, change the property, expand to chip history, and return to the same run and chart settings.
5. Open Chip calibration history on its own, search components, and check there is no horizontal scroll at phone width.

## GitHub Pages deployment

The deployment strategy is the same as before. [`.github/workflows/docs.yml`](../.github/workflows/docs.yml) builds the Sphinx docs, then:

```sh
npm ci --prefix dashboard
python dashboard/build.py
npm run check --prefix dashboard && npm test --prefix dashboard
npm run build --prefix dashboard -- --outDir ../docs/_build/html/dashboard
```

It runs on relevant pushes to `main` or `stable/**` and on manual dispatch. A failed check or build stops the workflow before deployment. It deploys with `peaceiris/actions-gh-pages` only from `main`. A manual dispatch from another branch runs every build step but skips the deploy, which makes it a safe dry run. Vite's `base: "./"` keeps every asset path relative, so the same build works at **`/qiskit-qward/dashboard/`** and locally.

On pull requests, [`.github/workflows/dashboard.yml`](../.github/workflows/dashboard.yml) runs the same checks plus the Playwright flows, but only when dashboard sources or exported tables change. See the [workflow dependency guide](../.github/workflows/README.md#when-workflows-run).

Commit dashboard sources, `package-lock.json` and the exported tables. Do not commit `dist/` or `public/data/`; both are generated. Dependabot updates npm dependencies weekly as one grouped PR.

The separate [IBM campaign workflow](../.github/workflows/ibm-campaign.md) collects new measurements and validates them with `python dashboard/build.py`. It needs Python only, not Node. Its results branch is merged manually to publish.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| "The experiment data could not be loaded" | Run `python3 dashboard/build.py` before `npm run dev`/`build`, and serve over HTTP rather than opening the file directly. |
| Schema version error in the page | `build.py` and `src/lib/data/types.ts` disagree; bump `SCHEMA_VERSION` in both and rebuild. |
| Old results | Rerun `build.py`; with `preview`, also rerun `npm run build`. |
| Address already in use | Stop the previous server with Ctrl+C, or pass `--port`. |
| `npm install` fails with `Cannot read properties of null (reading 'edgesOut')` | Known npm 10 resolver bug for fresh installs; use `npm ci`, or `npx npm@11 install` when changing dependencies. |
| Missing CSV parts, `dataset_manifest.json`, or `references.csv` | Use the default committed export, or pass the correct directory with `--tables`. |
| Calibration unavailable | Check that the matching `calibrations.csv` was exported and included before rebuilding. Detailed data is limited to the preparation snapshots actually captured. |
| Invalid reference, calibration, or shot-total error | Re-export the tables together and inspect `export_report.json`; do not manually force mismatched joins. |
| Export cannot find a preparation artifact | Restore the local journal's associated artifact directory. Viewing an existing tabulated export does not require these raw records. |
| Preview works but published site is stale | Check the docs workflow's build and deploy steps, the branch/ref, and the repository's Pages source. |

## Reading the results

The default 0.5 success requirement is an editable demonstration. The graph initially shows every recorded job for the selected algorithm and chip across all periods. Select an individual collection in Adjust plot → Periods (UTC) to restrict it. Each success-rate point is one job’s accepted shots divided by its total shots. Hovering over a point highlights the job and opens a magnified view of its two-sided 99% Wilson interval based on the exact shot counts. Compare separate points for variation across repetitions. The target classification uses the observed rates. Hover a point for its interval, experiment, group, repetition and measurement date. Histogram run choices use a unique record/repetition identity, so runs numbered alike in different collections remain separately selectable. The combined summary counts experiments meeting the target. Choose a group to inspect its boundary. Boundaries describe observed jobs within the chosen chip, periods and circuit family; when all periods are selected, agreement is assessed across every included job. A setting needs at least two jobs and agreement across all its observed repetitions. Missing metrics, reversals and disagreements are shown in the sweep table. The two sides of a bracket identify tested configurations. The axis selector changes their plotted coordinate.

Choose a question to select its supporting quantity. DSR compares the average accepted peak to the largest competing outcome. The counts-only question uses top-two contrast and the recorded rank test. Coarse TVD uses the campaign’s equal accepted-answer weights. Selecting an experiment or repetition updates its description, accepted outcomes and measured result against the active target. In the histogram, open Experiment to search the current plot’s experiments by configuration ID, group, or description. Select a result to change jobs; Escape keeps the current selection. The histogram exposes every recorded or expected outcome in its accompanying table, including zero-count expected labels. The chart combines omitted outcomes into a single probability-preserving bin.

## Calibration explorer

Use **Chip calibration history** at the top for independent exploration, or **Inspect run calibration** below a histogram for that run's hardware. The latter preserves the experiment, run, chart detail and scroll position on return. A recorded run is one IBM job; its individual shots are repeated circuit executions. Chart detail controls the number of individual bitstrings drawn, not the shots included in metrics.

Readout, individual native gate types, and T1/T2 have separate views with explanatory schematics. Search for a physical qubit or an ordered pair, select it in the graph or selector, and inspect its recorded history. The connectivity drawing is a schematic of observed operations, not the physical geometry of the entire chip. The color domain stays fixed across dates for the selected property and operation; colors are relative, not pass/fail classifications. Snapshot dates are equally spaced in the history plot. Missing observations remain gaps and no interpolation or calibration uncertainty is invented.

Detailed values come from **preparation snapshots**, not execution-time per-element measurements. The job context shows both timestamps and exposes the saved execution summary under Data sources. Gate/readout execution means are weighted by instruction occurrence. Repeated uses of a component in the preparation map/history are deduplicated. Conflicting values at the same timestamp remain unavailable. Dynamic control-flow bodies were not expanded by the original capture, and repeated measurements can overwrite classical bits. Bit 0 is the rightmost bit after removing register-separating spaces from histogram labels.

## Suggestions to evaluate next

- Try the histogram → calibration → histogram path first: is the relationship between observed outcomes and hardware context clear without implying causation?
- Decide whether comparing two saved dates side by side is more useful than a component-by-date heatmap once more monthly samples exist.
- Preserve detailed execution-time snapshots in future collection. That would let the component view show the actual execution calibration rather than preparation details alongside execution summaries.
