# Data Encoding for Quantum Machine Learning: Paper Compilation

## Quick Start

```bash
cd paper/
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

## Requirements

- LaTeX distribution (TeX Live 2023+ or MiKTeX)
- Required packages: amsmath, amssymb, amsthm, braket, graphicx, booktabs,
  algorithm, algpseudocode, hyperref, biblatex (with biber backend)
- Biber (for bibliography processing)

## File Structure

```
paper/
  main.tex            # Complete paper source
  references.bib      # Bibliography (22 entries)
  figures/
    fig1_radar_profiles.png          # Dataset statistical profiles
    fig2_heatmap_encoding_dataset.png # Accuracy heatmap
    fig3_quantum_vs_classical.png     # Quantum vs classical comparison
    fig4_boxplot_accuracy.png         # Accuracy distributions
    fig5_circuit_resources.png        # Circuit resource comparison
    fig6_summary_dashboard.png        # Combined summary
  README.md           # This file
```

## Compilation Notes

1. Run `pdflatex` twice after `biber` to resolve all cross-references
   and citations.
2. The paper uses `biblatex` with `biber` backend (not `bibtex`).
3. All figures are PNG at 300 DPI. For journal submission, convert to
   PDF or EPS if required.
4. The paper targets approximately 8-10 pages in two-column format.

## Paper Overview

- **Title**: Data Encoding for Quantum Machine Learning: A Systematic
  Study of Encoding-Data Compatibility
- **Sections**: Abstract, Introduction, Background, Encoding Methods,
  Classical Preprocessing, Experimental Design, Results, Discussion,
  Limitations, Conclusion
- **Key results**: 5 encodings x 8 datasets x 4 preprocessings = 67
  valid configurations, 335 fold evaluations
- **Figures**: 5 publication figures + 1 summary dashboard

## Source Material

All content derived from phase reports in `../reports/`:
- Phase 1: Literature review, dataset selection, evaluation framework
- Phase 2: Encoding theory, expressibility analysis, preprocessing theory
- Phase 3: Test methodology (TDD)
- Phase 4: Implementation details
- Phase 5: Experimental results, statistical analysis, visualizations
- Phase 6: Synthesis, hypothesis verdicts, recommendations
