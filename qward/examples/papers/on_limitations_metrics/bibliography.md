# Bibliography

Working bibliography for *On the Limitation of QC Metrics*. Keys follow
existing project citations where possible. Add Elicit hits to
[notes/literature.md](notes/literature.md) first, then promote them here.

## Primary measurement-theory source

```
@book{alma991005387727907681_2015,
  address={Boca Raton, Florida},
  edition={3rd ed.},
  series={Chapman & hall/CRC innovations in software engineering
          and software development},
  title={Software metrics: a rigorous and practical approach},
  ISBN={0-429-10622-X},
  publisher={CRC Press},
  author={Fenton, Norman E. and Bieman, James},
  year={2015},
  note={PDF on disk: docs/papers/metrics.pdf}
}
```

Related Fenton papers to cite alongside the book:

- Fenton, N. E. Software measurement: a necessary scientific basis.
  *IEEE Transactions on Software Engineering*, 20(3):199–206, 1994.
- Fenton, N. When a software measure is not a measure.
  *Software Engineering Journal*, 7(5):357–362, 1992.
- Basili, V. R., Caldiera, G., and Rombach, H. D. The Goal Question
  Metric Approach. In *Encyclopedia of Software Engineering*, 1994.

## Metric axioms and quantum distances

Must-cite from Elicit Question A:

```
@article{gilchristDistanceMeasuresComparea_2005,
  title={Distance measures to compare real and ideal quantum processes},
  volume={71},
  DOI={10.1103/PhysRevA.71.062310},
  number={6},
  journal={Physical Review A},
  author={Gilchrist, Alexei and Langford, Nathan K. and Nielsen, Michael A.},
  year={2005},
  pages={062310}
}
@article{luoInformationalDistanceQuantumstate_2004,
  title={Informational distance on quantum-state space},
  volume={69},
  DOI={10.1103/PhysRevA.69.032106},
  number={3},
  journal={Physical Review A},
  author={Luo, Shunlong and Zhang, Qiang},
  year={2004},
  pages={032106}
}
@article{maFidelityInducedDistance_2009,
  title={Fidelity induced distance measures for quantum states},
  volume={373},
  DOI={10.1016/j.physleta.2009.07.042},
  number={38},
  journal={Physics Letters A},
  author={Ma, Zhihao and Zhang, Fu-Lin and Chen, Jing-Ling},
  year={2009},
  pages={3407--3409}
}
```

Supporting Question A hits (cite if the subsection needs them):

```
@article{endresNewMetricProbability_2003,
  title={A new metric for probability distributions},
  volume={49},
  DOI={10.1109/TIT.2003.813506},
  number={7},
  journal={IEEE Transactions on Information Theory},
  author={Endres, D. M. and Schindelin, J. E.},
  year={2003},
  pages={1858--1860}
}
@article{mendoncaAlternativeFidelityMeasure_2008,
  title={Alternative fidelity measure between quantum states},
  volume={78},
  DOI={10.1103/PhysRevA.78.052330},
  number={5},
  journal={Physical Review A},
  author={Mendon{\c{c}}a, Paulo E. M. F. and Napolitano, Reginaldo d. J.
          and Marchiolli, Marcelo A. and Foster, Christopher J.
          and Liang, Yeong-Cherng},
  year={2008},
  pages={052330}
}
@article{chenSuperFidelityRelated_2011,
  title={Super fidelity and related metrics},
  volume={9},
  DOI={10.2478/s11534-010-0123-8},
  number={4},
  journal={Open Physics},
  author={Chen, Zhi-Hua and Ma, Zhihao and Zhang, Fu-Lin and Chen, Jing-Ling},
  year={2011},
  pages={1036--1042}
}
@article{spehnerQuantumCorrelationsDistinguishability_2014,
  title={Quantum correlations and distinguishability of quantum states},
  volume={55},
  DOI={10.1063/1.4885832},
  number={7},
  journal={Journal of Mathematical Physics},
  author={Spehner, Dominique},
  year={2014},
  pages={075211}
}
@article{zhangLowerBoundFidelity_2013,
  title={A lower bound on the fidelity between two states in terms of their Bures distance},
  journal={arXiv:1305.2023},
  author={Zhang, Lin and Wu, Junde},
  year={2013}
}
```

Logged but not for the main text unless a reviewer asks:
`reebHilbertsProjectiveMetric_2011`,
`tothQuantumWassersteinDistance_2025`.

Also keep:

- Nielsen, M. A. and Chuang, I. L. *Quantum Computation and Quantum
  Information*. Cambridge University Press, 2010.
- Hellinger, E. Neue Begründung der Theorie quadratischer Formen von
  unendlichvielen Veränderlichen. *Journal für die reine und
  angewandte Mathematik*, 136:210–271, 1909.
- QuantumBenchmarkZoo, "Fidelities and Errors" figure-of-merit table.
- Qiskit `hellinger_fidelity` / `hellinger_distance` (HF \(=\mathrm{BC}^2\)).
- Uhlmann, A. The "transition probability" in the state space of a
  *-algebra. *Reports on Mathematical Physics*, 9(2):273–279, 1976.

## QC success / fidelity reviews and testing

```
@inproceedings{mohapatraBenchmarkingFidelityMetricsa_2025,
  title={Benchmarking Fidelity Metrics of Quantum Computers},
  volume={01},
  url={https://ieeexplore.ieee.org/document/11250150},
  DOI={10.1109/QCE65121.2025.00045},
  booktitle={2025 IEEE International Conference on Quantum Computing
             and Engineering (QCE)},
  author={Mohapatra, Shubdeep and Patil, Hrushikesh Pramod
          and Liu, Ji and Zhou, Huiyang},
  year={2025},
  pages={327--337}
}
```
- Fortunato, D. et al. A Survey on Testing and Analysis of Quantum
  Software. arXiv:2410.00650, 2024.
- *Is Measurement Enough? Rethinking Output Validation in Quantum
  Program Testing.* ASE 2025.
- Andrews, E., Jayasena, A., and Mishra, P. A Survey of Functional
  Testing and Validation of Quantum Circuits. *IEEE Design & Test*,
  2026.
- Hacaloglu, T., Soubra, H., Bourque, P., and Abran, A. Quantum
  Software Size: What Do We Measure, How and Why? A Systematic
  Literature Review. *IEEE Access*, 14:86893–86914, 2026.
  doi:10.1109/ACCESS.2026.3700957.
  https://ieeexplore.ieee.org/document/11554035
  Local PDF:
  `qward/examples/papers/on_limitations_metrics/pdf/Quantum_Software_Size_What_Do_We_Measure_How_and_Why_A_Systematic_Literature_Review.pdf`.
  Cite as the closest QSE "what/how/why" review; it measures *size*,
  not job-level execution success.
- QUTest distribution asserts (TVD, Hellinger, KL, chi-squared) as
  examples of testers treating distances as oracles.

## Project-internal sources (reuse, do not re-cite as novelty)

- QCE26 submission 219 and reviews:
  `qward/examples/papers/main.tex`,
  `qward/examples/papers/main-review.md`.
- DSR profile implementation:
  `qward/metrics/differential_success_rate.py`,
  `qward/schemas/dsr_profile_schema.py`.
- Existing jobs:
  `qward/examples/papers/DSR_result.csv`,
  `qward/examples/papers/narrative_assessment.md`,
  `qward/examples/papers/broad_ideal_experiment_results.md`.
