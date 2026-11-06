// Ported from the former periods.test.cjs, now against pure functions with no DOM shim.
import { describe, expect, it } from "vitest";
import {
  ALL_PERIODS,
  groupConfigs,
  inPeriod,
  jobKey,
  periodLabel,
  periods,
  pick,
  selectJobs,
} from "../../src/lib/domain/scope";
import { job } from "./fixtures";

const fixture = [
  job({ config: "M3", campaign: "full", window: "1", ref: "old", finishedAt: "2026-11-04T13:00:00Z" }),
  job({ config: "M3", campaign: "growth", window: "2", ref: "new", finishedAt: "2026-11-06T00:00:00Z" }),
  job({ config: "M4", width: 8, campaign: "growth", window: "2", ref: "wider", finishedAt: "2026-11-06T00:01:00Z" }),
  job({ algorithm: "QFT", config: "SP3", family: "SP", width: 3, ref: "qft", finishedAt: "" }),
  job({ backend: "ibm_kingston", ref: "other-chip" }),
];
const refs = (jobs: { ref: string }[]) => jobs.map((j) => j.ref);

describe("measurement periods", () => {
  it("default to all periods, with individual collections selectable", () => {
    const all = inPeriod(fixture, "ibm_fez", ALL_PERIODS);
    expect(refs(selectJobs(all, "SHOR", ""))).toEqual(["old", "new", "wider"]);
    const growth = periods(fixture, "ibm_fez").find((p) => p.id.includes("growth"))!;
    expect(refs(selectJobs(inPeriod(fixture, "ibm_fez", growth.id), "SHOR", ""))).toEqual(["new", "wider"]);
    expect(refs(selectJobs(all, "QFT", ""))).toEqual(["qft"]);
  });

  it("are ordered newest first and labelled in UTC", () => {
    const list = periods(fixture, "ibm_fez");
    expect(list[0].id).toContain("growth");
    expect(list[0].label).toBe("Nov 6, 2026 · 00:00–00:01");
    expect(periodLabel([])).toBe("Date unavailable");
    expect(periodLabel(["2026-11-04T13:00:00Z", "2026-11-05T01:00:00Z"])).toBe(
      "Nov 4, 2026 · 13:00–Nov 5, 2026 · 01:00",
    );
  });

  it("exclude other chips across all periods", () => {
    expect(refs(inPeriod(fixture, "ibm_kingston", ALL_PERIODS))).toEqual(["other-chip"]);
  });
});

describe("job identity", () => {
  it("keeps same-numbered runs from different collections distinct", () => {
    const [m3] = groupConfigs(selectJobs(inPeriod(fixture, "ibm_fez", ALL_PERIODS), "SHOR", ""));
    expect(m3.map(jobKey)).toEqual(["old:1", "new:1"]);
    expect(new Set(m3.map(jobKey)).size).toBe(2);
  });

  it("orders configurations by design depth then width", () => {
    const configs = groupConfigs([
      job({ config: "B", width: 5 }),
      job({ config: "A", width: 9, designDepth: 2 }),
      job({ config: "C", width: 3 }),
    ]);
    expect(configs.map((g) => g[0].config)).toEqual(["A", "C", "B"]);
  });
});

describe("pick", () => {
  it("prefers the first available choice and falls back to the first value", () => {
    expect(pick(["a", "b"], "z", "b")).toBe("b");
    expect(pick(["a", "b"], null, "z")).toBe("a");
    expect(pick([], "a")).toBeUndefined();
  });
});
