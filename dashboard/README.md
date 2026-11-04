# QWARD experiment dashboard

Build from the repository root with Python 3.9 or later.

```sh
python3 dashboard/build.py
open dashboard/dist/index.html
```

The standalone site also works on a static HTTP server. The existing Pages workflow builds it into the site's `dashboard/` directory alongside the documentation. The page uses locally bundled Apache ECharts 5.6.0. Its license and notices are in `vendor/`.

## Updating the measurements

Run the existing campaign export after collecting jobs. Keep `dataset.csv` and `references.csv` from the same export together. The latter contains one frozen accepted set and ideal distribution per preparation, joined using `meta.run_id`. Rebuild to include the new results. Use `--tables DIRECTORY` to select another export or `--out DIRECTORY` to set the static output directory.

The build uses only the tabulated inputs and Python’s standard library. It validates the joins and shot totals and publishes selected scientific fields. Raw journal paths, account details, batch IDs and IBM job IDs are omitted. Window labels show the first and last job completion times in UTC. The page opens the latest measured window and all QFT experiments when available.

## Interpreting the first release

The default 0.5 success requirement is an editable demonstration. The graph initially shows every experiment for the selected algorithm and window. Hover a point for its experiment, group and repetition. The combined summary counts experiments meeting the target. Choose a group to inspect its boundary. Boundaries describe observed jobs within a chip, campaign, window and circuit family. A setting needs at least two jobs and agreement across all its observed repetitions. Missing metrics, reversals and disagreements are shown in the sweep table. The two sides of a bracket identify tested configurations. The axis selector changes their plotted coordinate.

Choose a question to select its supporting quantity. DSR compares the average accepted peak to the largest competing outcome. The counts-only question uses top-two contrast and the recorded rank test. Coarse TVD uses the campaign’s equal accepted-answer weights. Selecting an experiment or repetition updates its description, accepted outcomes and measured result against the active target. In the histogram, open Experiment to search the current plot’s experiments by configuration ID, group, or description. Select a result to change jobs; Escape keeps the current selection. The histogram exposes every recorded or expected outcome in its accompanying table, including zero-count expected labels. The chart combines omitted outcomes into a single probability-preserving bin.
