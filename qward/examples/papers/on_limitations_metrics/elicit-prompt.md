# Elicit.com prompts

Elicit wants a **precise research question**, not a keyword list.
Paste **one question per search**. After each run, fill the follow-up
columns and copy hits into [notes/literature.md](notes/literature.md).

The paper's novelty claim is *not* that Hellinger distance is a metric
and Hellinger fidelity is not. That is standard. The claim to test is
whether anyone already applies **representational measurement theory**
or **GQM** to **job-level quantum execution-success quantities**.

Start with Question C (novelty gate). Use A and B for background
citations.

---

## Question C — novelty gate (run this first)

Paste exactly:

```
Have software measurement theory or the Goal-Question-Metric (GQM)
paradigm been applied to decide which quantity — success rate,
Hellinger fidelity, total variation distance, ESP, or PST — should be
used to mark a quantum-circuit execution as successful?
```

Purpose: a hit that already audits those quantities with Fenton's
representation condition, scale types, or a GQM tree for "mark this
QPU job successful" would shrink the paper to a decision guide plus
DSR typing.

If this question is still too broad for Elicit, use the narrower
variant:

```
Does any study use Fenton's representational measurement theory
(representation condition, scale type, or meaningfulness) to validate
Hellinger fidelity, total variation distance, or raw success rate as
measures of quantum job success?
```

---

## Question A — metric axioms (background citations)

Elicit rejected the previous wording as too vague. Use the version
below: one focus, explicit concept split, and a stated literature
scope.

**Research focus.** Mathematical classification only: which of four
named quantities is a *distance metric* (M1–M4) and which is a
*similarity*.

**Concepts to keep distinct.**

- Hellinger *distance* \(H\) and total variation *distance* (TVD)
  are candidates for a metric.
- Hellinger *fidelity* \(F_H=\mathrm{BC}^2\) and Uhlmann / density-
  matrix *fidelity* are similarities (they equal 1 when the two
  objects coincide). They are not metrics and should not be tested
  against the triangle inequality.
- Do not treat \(1-F_H\) as automatically a metric.

**Literature-review scope.** Theoretical papers in probability
theory and quantum information, roughly 2000–present, that state or
prove the metric-space axioms or an explicit fidelity-versus-distance
distinction. Exclude: software-size metrics, COSMIC/LOC, empirical
QPU benchmarking without axioms, and papers that only report a
numerical fidelity.

Paste exactly:

```
In probability theory and quantum information (theoretical results,
approximately 2000–present), which publications prove or state that
Hellinger distance and total variation distance satisfy the four
metric-space axioms (non-negativity, identity of indiscernibles,
symmetry, and triangle inequality), and which publications
distinguish those distances from Hellinger fidelity and Uhlmann
fidelity as similarities that equal 1 when the two objects coincide
rather than as metrics?

Exclude software-size measurement, project-estimation metrics, and
empirical hardware benchmark papers that report a fidelity number
without discussing metric axioms.
```

If Elicit still asks for a narrower question, paste this single-
distinction variant:

```
Do theoretical papers in quantum information treat Hellinger
fidelity and Uhlmann fidelity as metrics on states or distributions,
or do they treat them as similarities and reserve the metric-space
axioms for Hellinger distance, total variation distance, or Bures
distance?
```

Purpose: collect the standard mathematical sources so the paper cites
them instead of reproving textbook facts. Expected hits include
Gilchrist, Langford and Nielsen (2005) and related fidelity-distance
work. Hits that only benchmark IBM/Rigetti fidelities are out of
scope for Question A; send those to Question B.

---

## Question B — reviews and success criteria (competitors)

Paste exactly:

```
What reviews or empirical studies compare Hellinger fidelity, total
variation distance, probability of successful trials (PST), and
estimated success probability (ESP) as criteria for validating the
output of a quantum circuit execution?
```

Purpose: find surveys that already treat HF, TVD, PST, and ESP as
execution oracles, and see whether they ask if those quantities are
metrics or valid measures. Expected hits include Mohapatra et al.
(2025) and quantum software-testing surveys.

---

## Follow-up columns

For every paper Elicit returns, record:

| Column | Question |
|---|---|
| C1 M1–M4 | Does the paper state the four metric-space axioms? |
| C2 Proof | Does it prove or cite that Hellinger distance or TVD is a metric? |
| C3 Fidelity vs distance | Does it distinguish fidelity / similarity from distance? |
| C4 Fenton / representation | Does it invoke representational measurement theory, scale types, or meaningfulness? |
| C5 GQM | Does it use Goal-Question-Metric for choosing a success quantity? |
| C6 Job-level criterion | Does it propose how to mark a single quantum job successful? |
| C7 Reference | Does it require known \(E\), a full ideal \(P^\star\), tomography, or an inverse circuit? **Elicit will not fill this.** Write it as our taxonomy in Section 5 (see below). |
| C8 Hardware | Is there empirical work on IBM, Rigetti, or another QPU? |

## How to interpret the results

- Question A hits with C1–C3 = yes and C4–C5 = no are expected. They
  confirm that the mathematical audit is background, not contribution.
- Question B hits with C6 = yes and C4 = no are the closest
  competitors (Mohapatra 2025; ASE 2025; Fortunato et al.). Position
  against them as correlation or testing surveys, not
  measurement-theory audits.
- Question C is decisive. If several papers have C4 or C5 = yes *and*
  apply that machinery to HF / TVD / success rate, the novelty gate
  fails and the outline must shrink.

## First-pass status

See [notes/literature.md](notes/literature.md). The first web search
did not find a paper that jointly applies Fenton / GQM to job-level
quantum execution-success quantities. Confirm or refute that with
Question C before drafting the article.
