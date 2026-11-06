import { describe, expect, it } from "vitest";
import { groupConfigs } from "../../src/lib/domain/scope";
import { summarize, verdict } from "../../src/lib/domain/sweep";
import { job } from "./fixtures";

const runs = (config: string, width: number, ...success: number[]) =>
  success.map((s, i) => job({ config, width, success: s, rep: i + 1, ref: `${config}-${i}` }));

describe("verdict", () => {
  it("needs two jobs with the metric", () => {
    expect(verdict(runs("A", 2, 0.9), "success", 0.5)).toBe("Insufficient evidence");
    expect(verdict([...runs("A", 2, 0.9), job({ success: null })], "success", 0.5)).toBe(
      "Insufficient evidence",
    );
  });

  it("classifies agreement across repetitions", () => {
    expect(verdict(runs("A", 2, 0.9, 0.8), "success", 0.5)).toBe("Meets");
    expect(verdict(runs("A", 2, 0.9, 0.3), "success", 0.5)).toBe("Mixed");
    expect(verdict(runs("A", 2, 0.1, 0.3), "success", 0.5)).toBe("Below");
    expect(verdict(runs("A", 2, 0.9, 0.8), "success", null)).toBe("No target");
  });

  it("treats TVD as lower-is-better", () => {
    const group = [job({ tvd: 0.1 }), job({ tvd: 0.2, ref: "b" })];
    expect(verdict(group, "tvd", 0.3)).toBe("Meets");
    expect(verdict(group, "tvd", 0.05)).toBe("Below");
  });
});

describe("summarize", () => {
  it("locates a clean boundary within one family", () => {
    const jobs = [...runs("S2", 2, 0.9, 0.8), ...runs("S3", 3, 0.7, 0.6), ...runs("S4", 4, 0.2, 0.1)];
    const summary = summarize(groupConfigs(jobs), jobs, "success", 0.5);
    expect(summary.boundary).toBe(2);
    expect(summary.title).toBe("S3 met the target. S4 missed it.");
    expect(summary.detail).toContain("S4 jobs: 20.0% and 10.0%.");
    expect(summary.detail).toContain("DSR at S4");
  });

  it("reports no single cutoff on reversals", () => {
    const jobs = [...runs("S2", 2, 0.2, 0.1), ...runs("S3", 3, 0.9, 0.8)];
    const summary = summarize(groupConfigs(jobs), jobs, "success", 0.5);
    expect(summary.boundary).toBeNull();
    expect(summary.title).toBe("No single measured cutoff");
  });

  it("counts experiments when several families are combined", () => {
    const jobs = [...runs("A", 2, 0.9, 0.8), ...runs("B", 3, 0.1, 0.2).map((j) => ({ ...j, family: "other" }))];
    const summary = summarize(groupConfigs(jobs), jobs, "success", 0.5);
    expect(summary.combined).toBe(true);
    expect(summary.title).toBe("1 of 2 experiments met the target");
  });

  it("handles an empty selection", () => {
    expect(summarize([], [], "success", 0.5).title).toBe("No jobs in this selection");
  });
});
