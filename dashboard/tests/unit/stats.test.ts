import { describe, expect, it } from "vitest";
import { normalQuantile, wilson } from "../../src/lib/stats/wilson";

describe("normalQuantile", () => {
  it("matches reference quantiles", () => {
    expect(normalQuantile(0.5)).toBeCloseTo(0, 9);
    expect(normalQuantile(0.975)).toBeCloseTo(1.959963985, 8);
    expect(normalQuantile(0.995)).toBeCloseTo(2.575829304, 8);
    expect(normalQuantile(0.005)).toBeCloseTo(-2.575829304, 8);
  });
});

describe("wilson", () => {
  // Values from build.py's success_interval for the same inputs (statistics.NormalDist).
  it("matches the Python builder at 99%", () => {
    const [low, high] = wilson(775, 1024);
    expect(low).toBeCloseTo(0.7207224752112611, 9);
    expect(high).toBeCloseTo(0.7896425451608609, 9);
  });

  it("pins the bounds at 0 and 1 for extreme counts", () => {
    expect(wilson(0, 100)[0]).toBe(0);
    expect(wilson(100, 100)[1]).toBe(1);
  });

  it("narrows with lower confidence", () => {
    const [l99, h99] = wilson(50, 100, 0.99);
    const [l95, h95] = wilson(50, 100, 0.95);
    expect(h95 - l95).toBeLessThan(h99 - l99);
  });
});
