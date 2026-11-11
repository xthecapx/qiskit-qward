# QWARD GitHub Actions

QWARD retains its library checks, releases, and Sphinx documentation publication.
Thesis experiments, data exports, and dashboard publication now belong to
[estimating-quantum-circuit](https://github.com/xthecapx/estimating-quantum-circuit).

- `test_latest_versions.yml`: library/example tests on relevant main/stable pushes
  or manual dispatch.
- `coverage.yml`: coverage for the remaining test suite.
- `lint.yml`: Python and configuration style checks.
- `docs.yml`: Sphinx documentation, retained slides, and documentation publishing.
- `ecosystem.yml` and `citation.yml`: ecosystem metadata and citation checks.
- `create-tag.yml` and `publish.yml`: version tagging and package publication.

The thesis-specific IBM collection and dashboard workflows have been removed.
QWARD documentation links to the thesis dashboard instead of building a second
copy. Raw thesis records and exported tables are not inputs to these workflows.
The report/manuscripts and their assets remain for a later migration.
