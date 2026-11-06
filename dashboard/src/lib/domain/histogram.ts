import type { Job, QuestionKey, Reference } from "../data/types";
import { compareText, pct, unique } from "./format";

export const OTHER = "Other outcomes";

export interface OutcomeRow {
  label: string;
  shots: number;
  observed: number;
  ideal: number | null;
  accepted: boolean | null;
}

export interface HistogramModel {
  /** Every recorded or expected outcome, ordered by observed counts. */
  rows: OutcomeRow[];
  display: string[];
  values: number[];
  markers: number[];
  hasReference: boolean;
  shownCount: number;
  groupedCount: number;
  blind: boolean;
  coarse: boolean;
  allowed: Set<string>;
  note: string;
  title: string;
  help: string;
}

/**
 * Choose which bitstrings the chart draws. Expected outcomes are prioritised (half the budget),
 * then the most frequent observations; omitted labels are combined into one
 * probability-preserving bin. Metrics always use all shots.
 */
export function histogram(
  job: Job,
  reference: Reference,
  key: QuestionKey,
  limit: number,
): HistogramModel {
  const { accepted, ideal } = reference;
  const allowed = new Set(accepted ?? []);
  const counts = job.counts;
  const labels = unique([...Object.keys(counts), ...allowed, ...Object.keys(ideal ?? {})]).sort(
    (a, b) => (counts[b] ?? 0) - (counts[a] ?? 0) || compareText(a, b),
  );
  const coarse = key === "coarse";
  const blind = key === "contrast";
  const probability = (label: string) => (counts[label] ?? 0) / job.shots;
  const target = (label: string) =>
    coarse ? (allowed.has(label) ? 1 / allowed.size : 0) : (ideal?.[label] ?? 0);
  const hasReference = coarse ? allowed.size > 0 : ideal != null && Object.keys(ideal).length > 0;
  const ranked = coarse ? labels.filter((k) => allowed.has(k)) : labels;
  const selected = blind
    ? ranked.slice(0, limit)
    : unique([
        ...ranked.filter((k) => allowed.has(k) || target(k) > 0).slice(0, Math.ceil(limit / 2)),
        ...ranked.slice(0, limit),
        ...ranked.filter((k) => allowed.has(k)),
      ]).slice(0, limit);
  const chosen = new Set(selected);
  const rest = labels.filter((k) => !chosen.has(k));
  const display = rest.length ? [...selected, OTHER] : [...selected];
  const sum = (fn: (label: string) => number) => rest.reduce((n, v) => n + fn(v), 0);
  const values = display.map((k) => (k === OTHER ? sum(probability) : probability(k)));
  const markers = display.map((k) => (k === OTHER ? sum(target) : target(k)));

  const competitor = labels.find((k) => !allowed.has(k));
  const acceptedMean = allowed.size
    ? pct(labels.filter((k) => allowed.has(k)).reduce((s, k) => s + probability(k), 0) / allowed.size)
    : "unavailable";
  const notes: Partial<Record<QuestionKey, string>> = {
    dsr: `Mean accepted peak ${acceptedMean}. Strongest individual competitor ${competitor ?? "none"} (${pct(competitor ? probability(competitor) : 0)}).`,
    contrast: `Leading counts: ${labels
      .slice(0, 2)
      .map((k) => counts[k] ?? 0)
      .join(" / ")}. Rank ${job.rank || "unavailable"} · p = ${job.rankP || "unavailable"} · α = ${job.alpha || "unavailable"}.`,
    coarse:
      "Orange markers show equal target weights for accepted answers. The full table below retains the original ideal probabilities.",
  };
  const legend = blind
    ? "Outcomes ordered by observed counts."
    : `Teal bars are accepted labels. ${
        hasReference
          ? "Orange diamonds show " +
            (coarse ? "the task target." : (job.exact ? "the exact" : "the sampled") + " ideal reference.")
          : "Reference unavailable."
      }`;
  const note = [notes[key], legend, rest.length ? "Other outcomes combines omitted labels." : ""]
    .filter(Boolean)
    .join(" ");
  const help = `${selected.length} individual bitstring${selected.length === 1 ? "" : "s"} shown${
    rest.length ? `; ${rest.length.toLocaleString()} more grouped as “Other outcomes”` : "; no outcomes grouped"
  }. ${
    blind ? "Largest observed counts are shown first." : "Expected outcomes and largest observed counts are prioritized."
  } This changes only the chart; all shots contribute to the metrics.`;

  return {
    rows: labels.map((label) => ({
      label,
      shots: counts[label] ?? 0,
      observed: probability(label),
      ideal: ideal ? (ideal[label] ?? 0) : null,
      accepted: accepted ? allowed.has(label) : null,
    })),
    display,
    values,
    markers,
    hasReference,
    shownCount: selected.length,
    groupedCount: rest.length,
    blind,
    coarse,
    allowed,
    note,
    title: blind || !hasReference ? "Observed outcomes" : "Observed and expected outcomes",
    help,
  };
}

export function acceptedSummary(reference: Reference): string {
  const allowed = reference.accepted ?? [];
  if (allowed.length > 0 && allowed.length <= 4) return `Accepted outcomes: ${allowed.join(", ")}.`;
  return allowed.length ? `${allowed.length} outcomes satisfy the task.` : "Accepted outcomes unavailable.";
}
