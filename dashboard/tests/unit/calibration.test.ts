import { describe, expect, it } from "vitest";
import type { CalibrationRecord } from "../../src/lib/data/types";
import {
  UNAVAILABLE,
  bitMappings,
  gatesFor,
  mergeComponents,
  scaleColor,
  schematic,
  snapshotDates,
  worstFirst,
} from "../../src/lib/domain/calibration";

function record(overrides: Partial<CalibrationRecord> = {}): CalibrationRecord {
  return {
    backend: "ibm_fez",
    at: "2026-11-04 07:36:35-05:00",
    collectedAt: "",
    source: "prepare_snapshot",
    qubits: {
      "1": { prob_meas0_prep1: 0.01, prob_meas1_prep0: 0.02, t1_s: 1e-4, t2_s: 5e-5 },
      "2": { t1_s: 2e-4, t2_s: null },
    },
    operations: [
      { name: "measure", qubits: [1], error: 0.015 },
      { name: "sx", qubits: [1], error: 0.0002 },
      { name: "cz", qubits: [1, 2], error: 0.003 },
    ],
    measurements: [{ qubits: [1, 2], bits: [0, 1] }],
    ...overrides,
  };
}

describe("mergeComponents", () => {
  it("deduplicates identical values and marks conflicts unavailable", () => {
    const a = record();
    const b = record({ operations: [{ name: "measure", qubits: [1], error: 0.02 }] });
    expect(mergeComponents([a, a], "readout", "")).toEqual([
      { id: "1", qubits: [1], value: 0.015, conflict: false },
    ]);
    expect(mergeComponents([a, b], "readout", "")[0]).toMatchObject({ value: null, conflict: true });
  });

  it("converts coherence times to microseconds", () => {
    const merged = mergeComponents([record()], "t1", "");
    expect(merged.map((c) => c.value)).toEqual([100, 200]);
    expect(mergeComponents([record()], "t2", "")[1].value).toBeNull();
  });

  it("filters gates by name and arity", () => {
    expect(mergeComponents([record()], "pair", "cz")[0]).toMatchObject({ id: "1→2", value: 0.003 });
    expect(gatesFor([record()], "single")).toEqual(["sx"]);
    expect(gatesFor([record()], "pair")).toEqual(["cz"]);
  });
});

describe("colour and defaults", () => {
  it("inverts the scale for durations and greys out missing values", () => {
    expect(scaleColor(0, [0, 1], "readout")).toBe("rgb(15,126,113)");
    expect(scaleColor(0, [0, 1], "t1")).toBe("rgb(204,93,48)");
    expect(scaleColor(null, [0, 1], "t1")).toBe(UNAVAILABLE);
  });

  it("selects the worst component first", () => {
    const merged = mergeComponents([record()], "t1", "");
    expect(worstFirst(merged, "t1")?.id).toBe("1");
  });
});

describe("schematic and mapping", () => {
  it("lays out connected qubits on successive levels", () => {
    const { nodes, links } = schematic([record()], 300);
    expect(links).toEqual([["1→2", [1, 2]]]);
    expect(nodes[0].x).toBeLessThan(nodes[1].x);
  });

  it("maps physical qubits to classical bits", () => {
    expect(bitMappings(record(), [2])).toEqual(["Q2 → bit 1"]);
    expect(bitMappings(undefined, [2])).toEqual([]);
  });

  it("orders snapshot dates chronologically across offsets", () => {
    const dates = snapshotDates([
      record({ at: "2026-11-05 01:00:00+00:00" }),
      record({ at: "2026-11-04 22:00:00-05:00" }),
    ]);
    expect(dates).toEqual(["2026-11-05 01:00:00+00:00", "2026-11-04 22:00:00-05:00"]);
  });
});
