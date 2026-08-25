# Research checklist

Complete these items before drafting the article. Do not invent proofs:
cite a standard theorem or write a short self-contained argument in
[notes/axiom-audit.md](notes/axiom-audit.md) or
[notes/dsr-typing.md](notes/dsr-typing.md).

## Measurement theory

- [x] Extract Fenton and Bieman 2015, Ch. 1–4 and 6, with page pointers.
      See [notes/fenton-bieman.md](notes/fenton-bieman.md).
- [x] Map every audited QC quantity onto representation condition and
      scale type. First pass is in [notes/axiom-audit.md](notes/axiom-audit.md);
      tighten after Elicit Packet C.
- [x] Decide and defend the scale type of \(p_E\) (absolute versus
      ratio). Fenton treats counts as absolute and derived proportions
      as needing an explicit argument.

## Mathematical statements

- [x] Cite the \(L^1\) proof that TVD is a metric.
- [x] Cite the \(L^2\) / Hellinger-integral proof that Hellinger
      distance is a metric.
- [x] Write the monotone threshold-invariance lemma
      (single job, fixed reference).
- [x] Write why arithmetic means of \(F_H\) and \(H\) need not
      preserve group rankings.
- [x] Write the composition bound for TVD and the corresponding
      inequality for \(s=1-\mathrm{TVD}\): \(s(P,R)\ge s(P,Q)+s(Q,R)-1\).
- [x] If auditing Hellinger infidelity \(1-F_H\), give a
      three-distribution counter-example that it is not a metric.
      This quantity was excluded from the audit; no unsupported M4 claim is made.
- [x] Prove the \(K=1\) collapse:
      \(\mathrm{coarse\_tvd}=1-p_E\) and
      \(\mathrm{coarse\_HF}=p_E\).
- [x] Prove that coarse TVD / Hellinger distance inherit M1–M4 as
      push-forwards onto the \(K+1\) simplex.
- [x] Verify the Bures / sine-distance citations: use Ma et al.,
      *Phys. Lett. A* 373:3407 (2009) plus Gilchrist et al. (2005).

## Literature gates

- [x] First novelty-gate pass:
      [notes/literature.md](notes/literature.md).
- [x] Run Elicit Questions A, B, and C and log hits in
      [notes/literature.md](notes/literature.md).
      Novelty gate: tentatively **open** (no occupier).
- [x] Read Mohapatra et al. 2025, ASE 2025 "Is Measurement Enough?",
      and Paltenghi and Pradel, arXiv:2410.00650, with the follow-up columns.
- [ ] If Zotero MCP is available later, tag the same queries against
      the local library.

## Empirical gates

- [x] Design the read-only aggregation analysis:
      [notes/consequence-gate.md](notes/consequence-gate.md).
- [x] Run Test A, Test B, and Test C on
      `qward/examples/papers/DSR_result.csv` (author will run this;
      no new QPU jobs).
- [x] If Test B finds no ranking inversion, reframe the evaluation as a
      methodological / negative-results section rather than claiming
      empirically changed decisions.

## Case study

- [x] Formal typing of every DSR profile component:
      [notes/dsr-typing.md](notes/dsr-typing.md).
- [ ] Recheck the toy histogram in [notes/examples.md](notes/examples.md)
      against `DSRProfiler` if desired.
