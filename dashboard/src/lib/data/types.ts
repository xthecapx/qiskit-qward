/** Shapes written by build.py. Keep in step with `load_tables()` and SCHEMA_VERSION. */

export const SCHEMA_VERSION = 1;

export type MetricKey =
  | "success"
  | "dsr"
  | "tvd"
  | "hellinger"
  | "coarse"
  | "chance"
  | "baseline";

export type QuestionKey = Exclude<MetricKey, "baseline"> | "contrast";
export type AxisKey = "width" | "bits" | "depth" | "gates";
export type CalibrationValueKey = "readout" | "single" | "pair" | "t1" | "t2";

export interface ExecutionCalibration {
  at: string | null;
  source: string | null;
  values: Record<CalibrationValueKey, number | null>;
}

export interface Job extends Record<MetricKey, number | null> {
  backend: string;
  campaign: string;
  window: string;
  algorithm: string;
  family: string;
  config: string;
  description: string;
  width: number;
  bits: number;
  depth: number;
  gates: number;
  designDepth: number | null;
  shots: number;
  rep: number;
  ref: string;
  finishedAt: string;
  exact: boolean;
  counts: Record<string, number>;
  contrast: number;
  rank: string;
  alpha: string;
  rankP: string;
  executionCalibration: ExecutionCalibration;
  countingQubits?: number;
  successInterval?: [number, number];
}

export interface Reference {
  accepted: string[] | null;
  ideal: Record<string, number> | null;
}

export interface QubitProperties {
  prob_meas0_prep1?: number | null;
  prob_meas1_prep0?: number | null;
  t1_s?: number | null;
  t2_s?: number | null;
}

export interface Operation {
  name: string;
  qubits: number[];
  error: number | null;
}

export interface Measurement {
  qubits: number[];
  bits: number[];
}

export interface CalibrationRecord {
  backend: string;
  at: string;
  collectedAt: string;
  source: string;
  qubits: Record<string, QubitProperties>;
  operations: Operation[];
  measurements: Measurement[];
}

export interface IndexData {
  decayReports?: Record<string, string>;
  schemaVersion: number;
  jobs: Job[];
  refs: Record<string, Reference>;
}

export interface CalibrationData {
  schemaVersion: number;
  calibrations: Record<string, CalibrationRecord>;
}
