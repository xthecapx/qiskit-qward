import { describe, expect, it } from "vitest";
import { formatHash, parseHash } from "../../src/lib/state/url";

describe("URL state", () => {
  it("round-trips view state", () => {
    const state = {
      algorithm: "GROVER",
      question: "dsr",
      target: "0.6",
      view: "calibration" as const,
      config: "S4-1",
      run: "abc:2",
      calMode: "pair",
      calScoped: true,
      calOrigin: "job" as const,
    };
    expect(parseHash(formatHash(state))).toEqual(state);
  });

  it("ignores unknown views and keys", () => {
    expect(parseHash("#v=elsewhere&zz=1&a=QFT")).toEqual({ algorithm: "QFT" });
    expect(formatHash({})).toBe("");
  });
});
