import { describe, expect, it } from "vitest";
import { OTHER, acceptedSummary, histogram } from "../../src/lib/domain/histogram";
import { job } from "./fixtures";

const counts = { "00": 50, "01": 30, "10": 15, "11": 5 };
const reference = { accepted: ["00"], ideal: { "00": 0.9, "01": 0.1 } };

describe("histogram", () => {
  it("keeps every outcome in the table, including expected labels with zero counts", () => {
    const model = histogram(job({ counts: { "01": 100 } }), reference, "success", 8);
    expect(model.rows.map((r) => r.label)).toEqual(["01", "00"]);
    expect(model.rows[1]).toMatchObject({ shots: 0, ideal: 0.9, accepted: true });
  });

  it("groups omitted outcomes into one probability-preserving bin", () => {
    const model = histogram(job({ counts }), reference, "success", 2);
    expect(model.display).toEqual(["00", "01", OTHER]);
    expect(model.values.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(model.values[2]).toBeCloseTo(0.2, 12);
    expect(model.groupedCount).toBe(2);
  });

  it("uses equal target weights for coarse TVD", () => {
    const model = histogram(job({ counts }), { accepted: ["00", "11"], ideal: null }, "coarse", 8);
    expect(model.display).toEqual(["00", "11", OTHER]);
    expect(model.markers).toEqual([0.5, 0.5, 0]);
    expect(model.hasReference).toBe(true);
  });

  it("ignores the reference for the counts-only question", () => {
    const model = histogram(job({ counts }), reference, "contrast", 8);
    expect(model.blind).toBe(true);
    expect(model.title).toBe("Observed outcomes");
    expect(model.note).toContain("Leading counts: 50 / 30.");
  });

  it("describes DSR against the strongest competitor", () => {
    const model = histogram(job({ counts }), reference, "dsr", 8);
    expect(model.note).toContain("Mean accepted peak 50.0%. Strongest individual competitor 01 (30.0%).");
  });
});

describe("acceptedSummary", () => {
  it("lists small answer sets and counts large ones", () => {
    expect(acceptedSummary({ accepted: ["1", "2"], ideal: null })).toBe("Accepted outcomes: 1, 2.");
    expect(acceptedSummary({ accepted: ["1", "2", "3", "4", "5"], ideal: null })).toBe(
      "5 outcomes satisfy the task.",
    );
    expect(acceptedSummary({ accepted: null, ideal: null })).toBe("Accepted outcomes unavailable.");
  });
});
