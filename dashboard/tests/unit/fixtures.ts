import type { Job } from "../../src/lib/data/types";

/** Minimal job with sensible defaults; override what a test cares about. */
export function job(overrides: Partial<Job> = {}): Job {
  return {
    backend: "ibm_fez",
    campaign: "full",
    window: "1",
    algorithm: "SHOR",
    family: "a=7",
    config: "M3",
    description: "",
    width: 7,
    bits: 4,
    depth: 10,
    gates: 4,
    designDepth: null,
    shots: 100,
    rep: 1,
    ref: "ref",
    finishedAt: "2026-11-04T13:00:00Z",
    exact: true,
    counts: { "0000": 100 },
    contrast: 1,
    rank: "verified",
    alpha: "0.01",
    rankP: "0",
    executionCalibration: {
      at: null,
      source: null,
      values: { readout: null, single: null, pair: null, t1: null, t2: null },
    },
    success: 1,
    dsr: 1,
    tvd: 0,
    hellinger: 1,
    coarse: 0,
    chance: 1,
    baseline: 0.0625,
    ...overrides,
  };
}
