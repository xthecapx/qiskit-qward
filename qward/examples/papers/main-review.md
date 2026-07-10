SUBMISSION: 219
TITLE: Differential Success Rate (DSR): A Distribution-Aware Score for Validating Quantum Algorithm Outputs

----------------------- REVIEW 1 ---------------------

SUBMISSION: 219
TITLE: Differential Success Rate (DSR): A Distribution-Aware Score for Validating Quantum Algorithm Outputs

----------- Overall evaluation -----------
SCORE: -2 (reject)
----- TEXT:
Summary:
The paper introduces a new metric, Differential Success Rate (DSR), to quantify the algorithmic degradation of quantum circuits. The proposed metric is intended to assess whether a given implementation of a quantum algorithm produces correct results on current noisy quantum hardware. The authors evaluate DSR against Hellinger Fidelity (HF) and Total Variation Distance Fidelity (TVDF) using IBM and Rigetti quantum backends across multiple configurations of three selected quantum algorithms.

Strengths

1. The paper is generally well written and easy to follow.

2. Evaluating the metric across two different quantum hardware providers (IBM and Rigetti) strengthens the empirical analysis and demonstrates broader applicability.

3. The comparison with existing benchmarking metrics provides useful insights into how DSR relates to the current state of the art.


Weaknesses

1. The motivation for introducing a new metric is not sufficiently convincing. Existing metrics already provide similar information, and DSR appears to suffer from many of the same limitations as HF and TVDF. The authors claim that DSR does not require ideal simulation of the true probability distribution because it relies only on the expected quantum state and the observed histogram. However, the paper also states that DSR is applicable only to quantum programs with known target output states, which weakens the claimed distinction from existing approaches. I do not agree with the authors on the usability claim that the DSR is not intended to compete or replace the existing metrics but rather the objective of enabling non-experts in QC to determine whether a given quantum algorithm operates as expected. This is exactly what other metrics do, so why a new metric with the same insights. One has to compete with other metrics to showcase the need.

   In practice, HF and TVDF can be applied to almost any quantum circuit, even when the target output state or ideal simulation is unavailable. For example, one can apply the inverse circuit and measure deviation from the zero state. While this approach increases sensitivity to noise due to greater circuit depth, it still preserves the applicability of HF and TVDF. In contrast, the presented analysis suggests that DSR depends heavily not only on circuit depth but also on the ratio between target output states and noisy outputs. Consequently, applying DSR using a zero target state may lead to unreliable results. This raises concerns regarding the suitability of DSR for larger and noisier circuits and hardware platforms.

2. The evaluation is limited in several important aspects. Although the authors compare the metrics across varying circuit depths and qubit counts, the results indicate that DSR exhibits more complex and less predictable behavior than the existing metrics. Circuit depth alone does not appear to be an adequate characterization criterion for DSR, since the metric is also strongly influenced by the number of target states and the circuit structure itself.

   Furthermore, the paper shows that HF and TVDF remain more robust under highly noisy hardware conditions. As noted by the authors, the discrepancy in DSR on the Ankaa-3 backend arises because the noise causes the strongest competing peak to match the average probability of the expected state, thereby driving DSR toward zero. In contrast, HF and TVDF, which evaluate overlap between probability distributions, still capture partial agreement between the expected and observed outputs.


Recommendations

1. Provide a stronger and more technically grounded motivation for introducing a new threshold-based metric in the current era of noisy quantum computing, especially given the growing focus on variational quantum algorithms.

2. Strengthen and expand the empirical evaluation by including a broader range of circuits. The current results indicate that DSR is highly sensitive to circuit configuration, complexity, and algorithm type. Evaluating DSR on more diverse circuits (particularly those whose expected output states can be verified using ideal simulation)would provide stronger empirical evidence and deeper theoretical insight.

3. Include a more detailed discussion explaining under which conditions higher or lower DSR values, or reduced variance in DSR, are preferable to HF and TVDF. This is particularly important if DSR is intended to complement existing metrics while providing similar qualitative insights.


----------------------- REVIEW 2 ---------------------

SUBMISSION: 219
TITLE: Differential Success Rate (DSR): A Distribution-Aware Score for Validating Quantum Algorithm Outputs

----------- Overall evaluation -----------
SCORE: 1 (weak accept)
----- TEXT:
Summary
The authors present Differential Success Rate: a new fidelity measure based on comparing state distribution peaks of expected results. The aim of this work is to expand existing fidelity measures with a more interpretable metric. DSR distinguishes itself by presenting a score ties to algorithm success probability, rather than the capability to match a full state distribution.

Strengths
- Validation of quantum circuits is a critical component in many aspects of quantum computing, range from benchmarking, operational validation and Quantum Architecture Search. As such, providing extensions to the existing tool-set is a welcome contribution.
- DSR in itself is indeed interpretable: if the score reaches 0, there is no chance of success, whereas for high values the DSR indicates validity of the execution. As this method targets users with little quantum knowledge, this is a accessible metric to validate quantum circuit executions.
- The metric matches well with expectations in the comparison to Hellinger Fidelity.
- While the authors adres challenges in the section "Validity considerations", these particular challenger apply for all discussed fidelity measures.

Weaknesses
- My main concern is that it is not clear what the true benefit of this metric is over other existing metrics. The main justification of the metric is that the score decreases faster for worse results. This should result in a threshold behavior, but this is not evident from the provided data. The DSR values in some cases present values of 0.51, 0.28 and 0.11, whereas with threshold behavior I would expect only highs and lows with nothing in between. I would have liked that an actual threshold value for this metric would be given by the authors, or that a clear benefit of DSR over HF/TVDF is given.

- Secondly, the applicability is very limited. As the authors discuss, this is limited to a set of known state solutions. Therefore it can not be applied to e.g. variational algorithms or any other algorithm that requires a full state distribution.

Justification and recommendation
I believe there is some merit in this paper, as it opens the discussion on fidelity metrics and there could be practical use for this metric. However, I believe the authors need to make a stronger case on where DSR outperforms HF or TVDF, as the justification in the paper is rather weak. For these reasons, I recommend a weak accept (+1).


----------------------- REVIEW 3 ---------------------

SUBMISSION: 219
TITLE: Differential Success Rate (DSR): A Distribution-Aware Score for Validating Quantum Algorithm Outputs

----------- Overall evaluation -----------
SCORE: 0 (borderline paper)
----- TEXT:
The paper has a clear contribution, but also some limitations. The main contribution is the introduction of a new metric (DSR) for validating quantum algorithm outputs, and the authors provide an extensive experimental evaluation on real IBM and Rigetti hardware.

The paper proposes Differential Success Rate (DSR), a lightweight metric that evaluates whether expected outcomes stand out from competing peaks in a measured quantum output distribution. DSR does not require simulation of the ideal output distribution and is computed directly from the observed histogram and expected target states. The authors evaluate DSR on 1,281 executions of Grover, QFT, and a teleportation variant across IBM and Rigetti quantum hardware and compare it against HF and TVDF.

Strengths
•Clear and well-motivated problem: output validation remains challenging in NISQ computing.
•Simple metric with low computational overhead.
•The paper is generally well written and easy to follow.

Weaknesses
•The novelty is somewhat incremental.
•Applicability is limited to algorithms with known expected outcomes.
•The evaluation focuses primarily on Grover, QFT, and a teleportation variant. Additional studies on other algorithm families would strengthen generality claims.

Overall Assessment
The contribution is useful. However, the conceptual novelty is moderate rather than high, and the applicability of the metric is narrower than the paper sometimes suggests.
