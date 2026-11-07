# GitHub Actions workflows

This directory contains a number of workflows for use with [GitHub Actions](https://docs.github.com/actions).  They specify what standards should be expected for development of this software, including pull requests.  These workflows are designed to work out of the box for any research software prototype, especially those based on [Qiskit](https://qiskit.org/).

## When workflows run

No workflows run on a schedule. Automatic runs require both a matching branch
and a matching changed path. Pull requests keep relevant style, ecosystem JSON,
and citation checks. Full tests, coverage, and documentation/dashboard builds run
only after relevant pushes to `main` or `stable/**` (including merges), or manually
from the Actions tab. Manual dispatch is not restricted by changed paths.

| Workflow | Relevant changes | Pull request | Push to `main` / `stable/**` | Manual |
| --- | --- | --- | --- | --- |
| Styles check | Python files anywhere; lint/package configuration and core dependencies | Yes | Yes | No |
| Ecosystem JSON check | `ecosystem.json`, `tox.ini`, its workflow | Yes | Yes | No |
| Citation preview | `CITATION.bib`, its workflow | On `main` | On `main` | No |
| Latest version tests | Library/example Python, tests, campaign test inputs, package/test configuration | No | Yes | Yes |
| Code coverage | Same suite inputs as tests, plus `.coveragerc` | No | Yes | Yes |
| Documentation and dashboard | Docs, dashboard sources, exported tables, library API Python, docs/package configuration | No | Yes | Yes |
| Dashboard checks | Dashboard sources (except README), exported tables, its workflow | Yes | No | Yes |
| IBM campaign collection | Explicit dispatch only | No | No | Yes, select `main` |

Each filtered workflow also watches its own YAML file. Editing one workflow does
not trigger the others unless another changed file matches their filters.

The less obvious dependencies are intentional:

- The test suite covers research examples as well as the library. Campaign window
  templates (`*/plans/*-w*.json`), the Shor growth profile, `run_campaign.sh`, and
  `scripts/ibm_campaign_ci.py` are direct test inputs.
- `tests/test_dataset_export.py` and `tests/test_calibration_tables.py` exercise
  `dashboard/build.py`, so changing that builder runs tests and coverage. Changing
  only the dashboard's Svelte/TypeScript sources or npm dependencies does not; those
  are covered by the dashboard checks on pull requests and by the docs build.
- Sphinx imports the library for API pages. Library Python changes rebuild docs;
  research example Python changes do not. The dashboard reads the committed
  `qward/examples/papers/dataset_out/` tables, not raw QPU records, caches, or
  date-specific campaign plans. Updating those raw files alone does not rebuild
  the site; merging updated exports does.
- `setup.py`, `pyproject.toml`, `MANIFEST.in`, `qward/VERSION.txt`,
  `requirements.qward.txt`, and `tox.ini` affect multiple environments, so they
  trigger the applicable workflows. `requirements.docs.txt` affects only docs.
  These workflows do not consume `uv.lock`; updating it alone triggers none.

For example, a dashboard CSS-only PR runs the dashboard checks and, once merged, the docs/dashboard publisher; a
documentation Markdown-only merge runs the docs builder; an ecosystem JSON-only
change runs only its JSON check. README edits, research reports, and raw experiment
results do not start the Python test matrix.

Release tagging still runs when `qward/VERSION.txt` changes on `main`. Package
publishing still runs on `v*` tag pushes or manual dispatch.

## IBM campaign collection (`ibm-campaign.yml`)

Selects the full Grover/BV growth window (391 jobs), the Grover/BV-only window (68 jobs), or a historical profile (344, 39 or 381 jobs) on Fez, Kingston, or Marrakesh with a backend/date identity. Separate prepare, submit, recover, and export steps use bounded batches, durable checkpoints, and read-only recovery retries. Results stay on a branch; no PR is created. This workflow is manual-only. See the [setup, credentials, and recovery guide](ibm-campaign.md). The default `validate` mode is offline; select `run` explicitly to start a collection, `resume` to submit only untouched batches, or `recover` to retrieve existing results. Set the collection date in the dispatch form for each new campaign.

## Dashboard checks (`dashboard.yml`)

Pull-request gate for the Svelte dashboard: `npm ci`, `python build.py` (table validation), `svelte-check`, Vitest unit tests, the Vite production build, and the Playwright browser flows in `dashboard/tests/e2e/`. It does not deploy; publishing happens in `docs.yml` after merge. Playwright artifacts are uploaded on failure.

## Styles check (`lint.yml`)

This workflow checks that the code is formatted properly and follows the style guide by installing tox and running the [lint environment](/tests/#lint-environment) (`tox -elint`). It watches Python files throughout the repository because Black runs on `.`; Pylint and mypy have narrower library/test scopes. Non-Python dashboard assets, Markdown, notebooks, and result tables do not trigger it.

## Latest version tests (`test_latest_versions.yml`)

This workflow installs the latest version of tox and runs [the current repository's tests](/tests/#test-py-environments) under each supported Python version on Linux and under a single Python version on macOS and Windows. This is the primary testing workflow. It runs on relevant pushes to `main` and `stable/**`, or manually. It does not run on pull requests or a timer.

<!-- ## Development version tests (`test_development_versions.yml`)

This workflow installs tox and modifies `requirements.txt` to use the _development_ versions of all Qiskit packages.  For all other packages, the latest version is installed.  This workflow runs on two versions of Python: the minimum supported version and the maximum supported version.  Its purpose is to identify as soon as possible (i.e., before a Qiskit release) when changes in Qiskit will break the current repository.  This workflow runs for all code changes, as well as on a timer once per day.

## Minimum version tests (`test_minimum_versions.yml`)

This workflow first installs the minimum supported tox version (the `minversion` specified in [`tox.ini`](/tox.ini)) and then installs the _minimum_ compatible version of each package listed in `requirements.txt` and `requirements-dev.txt`.  The purpose of this workflow is to make sure the minimum version specifiers in these files are accurate, i.e., that the tests actually pass with these versions.  This workflow uses a single Python version, typically the oldest supported version, as the minimum supported versions of each package may not be compatible with the most recent Python release.

Under the hood, this workflow uses a regular expression to change each `>=` and `~=` specifier in the requirements files to instead be `==`, as pip [does not support](https://github.com/pypa/pip/issues/8085) resolving the minimum versions of packages directly.  Unfortunately, this means that the workflow will only install the minimum version of a package if it is _explicitly_ listed in one of the requirements files with a minimum version.  For instance, a requirements file that simply lists `qiskit>=0.34` will actually install `qiskit==0.34` (i.e., the minimum version of the _meta_-package) along with the latest versions of `qiskit-terra` and `qiskit-aer`, unless their minimum versions are specified explicitly as well. -->

## Code coverage (`coverage.yml`)

This workflow tests the [coverage environment](/tests/#coverage-environment) on a single version of Python by installing tox and running `tox -ecoverage`. It runs on relevant pushes to `main` and `stable/**`, or manually, rather than on pull requests. Its suite path filters stay aligned with the latest-version tests.

## Documentation and dashboard (`docs.yml`)

Builds the Sphinx documentation and experiment dashboard on relevant pushes to
`main` and `stable/**`, or manually. Dashboard assets and exported tables rebuild
the combined site because docs and dashboard share one GitHub Pages deployment.
`dashboard/README.md` and generated `dashboard/dist/` output are excluded. Only
`main` deploys the output to GitHub Pages. Pull requests do not run this build.

## Citation preview (`citation.yml`)

This workflow is only triggered when the `CITATION.bib` file is changed.  It ensures that the file contains only ASCII characters ([escaped codes](https://en.wikibooks.org/wiki/LaTeX/Special_Characters#Escaped_codes) are preferred, as then the `bib` file will work even when `inputenc` is not used).  It also compiles a sample LaTeX document which includes the citation in its bibliography and uploads the resulting PDF as an artifact so it can be previewed (e.g., before merging a pull request).
