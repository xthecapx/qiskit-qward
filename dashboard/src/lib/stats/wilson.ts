/**
 * Inverse standard normal CDF (Acklam's rational approximation, |error| < 1.2e-9).
 * Used for interactive interval levels; build.py computes the published 99% intervals.
 */
export function normalQuantile(p: number): number {
  if (!(p > 0 && p < 1)) throw new RangeError("p must be in (0, 1)");
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const low = 0.02425;
  if (p < low) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - low) return -normalQuantile(1 - p);
  const q = p - 0.5;
  const r = q * q;
  return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/** Two-sided Wilson score interval for a binomial proportion. Mirrors build.py's success_interval. */
export function wilson(successes: number, trials: number, level = 0.99): [number, number] {
  if (trials <= 0) throw new RangeError("trials must be positive");
  const z = normalQuantile(1 - (1 - level) / 2);
  const p = successes / trials;
  const denominator = trials + z ** 2;
  const center = (successes + z ** 2 / 2) / denominator;
  const radius = (z * Math.sqrt(trials * p * (1 - p) + z ** 2 / 4)) / denominator;
  return [successes === 0 ? 0 : center - radius, successes === trials ? 1 : center + radius];
}
