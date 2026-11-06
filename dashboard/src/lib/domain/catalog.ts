import type { AxisKey, Job, QuestionKey } from "../data/types";

export const axes: Record<AxisKey, string> = {
  width: "Logical qubits",
  depth: "Executed ISA depth",
  bits: "Measured bits",
  gates: "Two-qubit gates",
};

const names: Record<string, string> = {
  "BERNSTEIN-VAZIRANI": "Bernstein–Vazirani",
  "PHASE-ESTIMATION": "Phase estimation",
  "RANDOM-VOLUMETRIC": "Random volumetric",
  VTP: "vTP",
  GROVER: "Grover",
  SHOR: "Shor",
};
export const algorithmName = (id: string) => names[id] ?? id;

export interface Question {
  prompt: string;
  label: string;
  meaning: string;
}

export const questions: Record<QuestionKey, Question> = {
  success: {
    prompt: "How often did I get an accepted answer?",
    label: "Success rate",
    meaning: "Fraction of all shots that landed in the task’s accepted answer set.",
  },
  dsr: {
    prompt: "Did expected peaks dominate competitors?",
    label: "Differential Success Rate",
    meaning:
      "Contrast of the average accepted peak against the strongest individual competitor. Compare within the same answer set, measured width and shot budget.",
  },
  contrast: {
    prompt: "Did any observed candidate stand out?",
    label: "Top-two contrast",
    meaning:
      "Uses the leading counts alone. The rank test gives separate evidence about their ordering.",
  },
  coarse: {
    prompt: "Did accepted answers have the desired balance?",
    label: "Coarse TVD",
    meaning:
      "Distance to equal target weights across accepted answers, with one remaining bin for all other outcomes. Lower is closer.",
  },
  tvd: {
    prompt: "How far was the full output from the ideal?",
    label: "Full TVD",
    meaning:
      "Probability mass that would need to move to match the frozen reference. Lower is closer.",
  },
  hellinger: {
    prompt: "How well did the whole distribution agree?",
    label: "Hellinger fidelity",
    meaning:
      "Squared sum of the geometric means of corresponding observed and reference probabilities. Higher values indicate closer agreement of the classical distributions.",
  },
  chance: {
    prompt: "How much did success improve over guessing?",
    label: "Chance-corrected success",
    meaning:
      "Success above the uniform-guess baseline, rescaled to the remaining attainable gain and clipped at zero.",
  },
};
export const questionKeys = Object.keys(questions) as QuestionKey[];

export const tasks: Record<string, string> = {
  GROVER: "Grover searches for a marked bitstring.",
  GHZ: "The GHZ circuit prepares a superposition of the all-zero and all-one strings.",
  "BERNSTEIN-VAZIRANI": "Bernstein–Vazirani recovers the hidden bitstring.",
  QFT: "This experiment checks the measured output of a Fourier-transform circuit.",
  "PHASE-ESTIMATION": "Phase estimation encodes the gate’s phase in the counting register.",
  VTP: "The protocol tests whether the payload is recovered after teleportation.",
  "RANDOM-VOLUMETRIC":
    "This experiment tests output recovery as a random circuit grows in size or depth.",
  SHOR: "This circuit samples phase peaks associated with the order-finding step of Shor’s algorithm.",
};

/** Long-form names for the QFT study families. */
export function studyName(family: string): string {
  return family
    .replace(/^SP · period (\d+)$/, "Period detection at increasing widths (period $1)")
    .replace(/^PV · period (\d+)$/, "Period variation at fixed width (period $1)")
    .replace(/^SR · round trip$/, "Round-trip scaling");
}

/** Phase-detection QFT circuits use one ancilla beyond the counting register. */
export function circuitDescription(job: Job): string {
  return job.countingQubits != null && job.width === job.countingQubits + 1
    ? `${job.countingQubits} counting qubits + 1 ancilla`
    : job.description;
}

export function qubitSummary(job: Job): string {
  return job.countingQubits != null
    ? `${job.width} qubits (${job.countingQubits} counting + ${job.width - job.countingQubits} auxiliary)`
    : `${job.width} logical qubits`;
}
