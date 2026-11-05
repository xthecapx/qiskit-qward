# QPU Benchmark · QWARD dashboard

A standalone, static explorer for experiment results and historical QPU calibration. It uses HTML, CSS, vanilla JavaScript, and locally bundled Apache ECharts 5.6.0. It runs without a backend, Node package installation, or IBM credentials. The Python build reads exported tables and produces a self-contained site for local use or GitHub Pages.

## Quick start

Run these commands from the **repository root**, using Python 3.11 or later (matching the project's supported minimum):

```sh
python3 dashboard/build.py
python3 -m http.server 8774 --bind 127.0.0.1 --directory dashboard/dist
```

Open **http://127.0.0.1:8774/** in a browser. Keep the server terminal open; press **Ctrl+C** there to stop it. If port 8774 is occupied, use another port in both the command and URL.

The default build uses the committed tables in `qward/examples/papers/dataset_out/`. Python's standard library is sufficient to build and serve the dashboard; you do not need to install QWARD to view these results. A modern browser is required. You can also open `dashboard/dist/index.html` directly in a browser that permits local files; all scripts, charts, and data are bundled locally.

After editing source files or updating the tables, run the build again and refresh the browser. The server serves files as they change, but **there is no automatic rebuild or browser reload**.

## Project layout

| Path | Purpose |
| --- | --- |
| `dashboard/index.html` | Page structure, dialogs, and navigation |
| `dashboard/styles.css` | Responsive layout, metadata icons styling, and visual states |
| `dashboard/app.js` | Operating-range plot, histogram, and experiment picker |
| `dashboard/calibration.js` | Chip and run-specific calibration explorer |
| `dashboard/build.py` | Standard-library table validation and static site generation |
| `dashboard/vendor/` | Bundled ECharts, license, and notices |
| `dashboard/dist/` | Generated local site; ignored by Git |
| `qward/examples/papers/dataset_out/` | Exported input tables and export reports |
| `qward/examples/papers/dataset_export.py` | Offline exporter for completed experiment records |
| `qward/examples/papers/calibration_tables.py` | Preparation calibration and bit-mapping export |

Edit the source files in `dashboard/`, not copies in `dashboard/dist/`. The generated `snapshot.js` contains the public data used by the page and is regenerated on every build.

## Build options

```sh
python3 dashboard/build.py --help

# Build from a different set of exported tables.
python3 dashboard/build.py --tables /path/to/export --out /path/to/site

# Build into the documentation site's deployment directory, as CI does.
python3 dashboard/build.py --out docs/_build/html/dashboard
```

The input directory must contain these files from the **same export**:

| File | Contents |
| --- | --- |
| `dataset.csv` | One row per completed job: circuit information, counts, metrics, timing, and execution calibration summaries |
| `references.csv` | Frozen accepted labels and ideal probabilities, joined by `run_id` |
| `calibrations.csv` | Used-qubit preparation properties, unique gate operations, and ordered measurement-to-classical-bit mappings, joined by `run_id` |

Older exports without `calibrations.csv` still display the success dashboard and show a calibration availability message. If a calibration table is supplied, its joins must match the jobs and chip. The builder also checks job uniqueness, shot totals, accepted-set sizes, and ideal probability totals. Correct or re-export inconsistent inputs rather than combining files from different exports.

The site publishes selected scientific fields. Raw journal paths, account details, batch IDs, and IBM job IDs are omitted. Measurement periods show the first and last job completion times in UTC, using month names; campaign window numbers are retained internally for grouping but hidden in the period labels. The page initially selects the latest measured period and QFT when available.

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
python dashboard/build.py
```

If you use the campaign wrapper and have `uv` installed, the equivalent export command is:

```sh
./qward/examples/papers/run_campaign.sh export
```

Export reads local journals and their `artifacts/.../prepared.json` files; it does not submit jobs or query IBM. Review `dataset_out/export_report.json` for errors and warnings. Keep all three tables together when committing updated data, then rebuild and refresh the preview. The wrapper exports completed records across periods; it is not restricted to one campaign window.

For collecting and recovering new QPU results, follow the [IBM Fez experiment plan](../docs/ibm_fez_dataset_plan.md) and the [campaign driver](../qward/examples/papers/run_campaign.sh). Those operations have their own IBM credentials and usage requirements. They are separate from running the dashboard.

## Verification during development

The following checks do not submit IBM jobs. Run the Python checks in the development environment above:

```sh
python dashboard/build.py
python -m pytest tests/test_calibration_tables.py tests/test_dataset_export.py -q
python -m black --check .
```

If Node.js is available, check JavaScript syntax without installing packages:

```sh
node --check dashboard/app.js
node --check dashboard/calibration.js
```

For the full repository lint job, use `uvx tox -elint` with Python 3.11 available, as configured in [tox.ini](../tox.ini). The targeted tests cover export behavior, calibration identities and joins, missing calibration sidecars, and the distinction between preparation and execution data. They do not replace browser checks.

After rebuilding, check these flows in the browser:

1. Change the algorithm, question, and measurement period; select a plot point to open its histogram.
2. Search the experiment picker, select a result with the mouse or keyboard, and verify the recorded run and histogram update. Escape should keep the previous selection.
3. Change Chart detail and confirm omitted bitstrings are grouped while the metrics remain unchanged.
4. Open Inspect run calibration, select components and properties, expand to chip history, and return to the same histogram run and chart settings.
5. Open Chip calibration history independently; check dates, search, unavailable values, and layout at narrow screen widths.

## GitHub Pages deployment

[`.github/workflows/docs.yml`](../.github/workflows/docs.yml) builds the Sphinx documentation, then runs:

```sh
python dashboard/build.py --out docs/_build/html/dashboard
```

The workflow builds on pushes and pull requests targeting `main` or `stable/**`, and supports manual dispatch. Its deploy step runs only when the workflow's ref is `main`. It publishes the combined documentation output through `peaceiris/actions-gh-pages` using `GITHUB_TOKEN`; the repository's Pages configuration must serve the published branch.

The dashboard's deployment path is **`/qiskit-qward/dashboard/`**, alongside the documentation. Commit dashboard sources and the exported tables; do not commit the generated `dashboard/dist/` directory. The docs workflow builds from those committed tables.

The separate [IBM campaign workflow](../.github/workflows/ibm-campaign.md) can collect the full research profile using a collection date, preserve recovery checkpoints, and open a results PR against `main`. Its first scheduled test is November 5, 2026 at 10:00 a.m. Bogotá (15:00 UTC), after the setup is merged and repository secrets are configured. Merge the results PR to publish the new measurements. A reduced monthly campaign is planned separately; the full 344-job window is not automatically repeated monthly.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Old layout or old results | Rebuild, confirm you are serving `dashboard/dist`, and refresh the browser; use a hard refresh if assets are cached. |
| Address already in use | Stop the previous preview with Ctrl+C, or select another local port. |
| Missing `dataset.csv` or `references.csv` | Use the default committed export, or pass the correct directory with `--tables`. |
| Calibration unavailable | Check that the matching `calibrations.csv` was exported and included before rebuilding. Detailed data is limited to the preparation snapshots actually captured. |
| Invalid reference, calibration, or shot-total error | Re-export the tables together and inspect `export_report.json`; do not manually force mismatched joins. |
| Export cannot find a preparation artifact | Restore the local journal's associated artifact directory. Viewing an existing tabulated export does not require these raw records. |
| Preview works but published site is stale | Check the docs workflow's build and deploy steps, the branch/ref, and the repository's Pages source. |

## Reading the results

The default 0.5 success requirement is an editable demonstration. The graph initially shows every experiment for the selected algorithm and measurement period. Each success-rate point is one job’s accepted shots divided by its total shots. Hovering over a point highlights the job and opens a magnified view of its two-sided 99% Wilson interval based on the exact shot counts. Compare separate points for variation across repetitions. The target classification uses the observed rates. Hover a point for its interval, experiment, group and repetition. The combined summary counts experiments meeting the target. Choose a group to inspect its boundary. Boundaries describe observed jobs within a chip, campaign, window and circuit family. A setting needs at least two jobs and agreement across all its observed repetitions. Missing metrics, reversals and disagreements are shown in the sweep table. The two sides of a bracket identify tested configurations. The axis selector changes their plotted coordinate.

Choose a question to select its supporting quantity. DSR compares the average accepted peak to the largest competing outcome. The counts-only question uses top-two contrast and the recorded rank test. Coarse TVD uses the campaign’s equal accepted-answer weights. Selecting an experiment or repetition updates its description, accepted outcomes and measured result against the active target. In the histogram, open Experiment to search the current plot’s experiments by configuration ID, group, or description. Select a result to change jobs; Escape keeps the current selection. The histogram exposes every recorded or expected outcome in its accompanying table, including zero-count expected labels. The chart combines omitted outcomes into a single probability-preserving bin.

## Calibration explorer

Use **Chip calibration history** at the top for independent exploration, or **Inspect run calibration** below a histogram for that run's hardware. The latter preserves the experiment, run, chart detail and scroll position on return. A recorded run is one IBM job; its individual shots are repeated circuit executions. Chart detail controls the number of individual bitstrings drawn, not the shots included in metrics.

Readout, individual native gate types, and T1/T2 have separate views with explanatory schematics. Search for a physical qubit or an ordered pair, select it in the graph or selector, and inspect its recorded history. The connectivity drawing is a schematic of observed operations, not the physical geometry of the entire chip. The color domain stays fixed across dates for the selected property and operation; colors are relative, not pass/fail classifications. Snapshot dates are equally spaced in the history plot. Missing observations remain gaps and no interpolation or calibration uncertainty is invented.

Detailed values come from **preparation snapshots**, not execution-time per-element measurements. The job context shows both timestamps and exposes the saved execution summary under Data sources. Gate/readout execution means are weighted by instruction occurrence. Repeated uses of a component in the preparation map/history are deduplicated. Conflicting values at the same timestamp remain unavailable. Dynamic control-flow bodies were not expanded by the original capture, and repeated measurements can overwrite classical bits. Bit 0 is the rightmost bit after removing register-separating spaces from histogram labels.

## Suggestions to evaluate next

- Try the histogram → calibration → histogram path first: is the relationship between observed outcomes and hardware context clear without implying causation?
- Decide whether comparing two saved dates side by side is more useful than a component-by-date heatmap once more monthly samples exist.
- Preserve detailed execution-time snapshots in future collection. That would let the component view show the actual execution calibration rather than preparation details alongside execution summaries.
