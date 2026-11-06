import type { Job, QuestionKey } from "../data/types";
import { format, lowerIsBetter, score, unique } from "./format";
import type { Config } from "./scope";

export type Verdict = "Meets" | "Mixed" | "Below" | "No target" | "Insufficient evidence";

export function passes(key: QuestionKey, value: number, threshold: number): boolean {
  return lowerIsBetter(key) ? value <= threshold : value >= threshold;
}

/** A setting needs at least two jobs, all with the metric, to be classified. */
export function verdict(group: Job[], key: QuestionKey, threshold: number | null): Verdict {
  if (group.length < 2 || group.some((j) => j[key] == null)) return "Insufficient evidence";
  if (threshold == null) return "No target";
  const ok = group.map((j) => passes(key, j[key] as number, threshold));
  return ok.every(Boolean) ? "Meets" : ok.some(Boolean) ? "Mixed" : "Below";
}

export interface SweepSummary {
  states: Verdict[];
  /** Index of the first failing setting when the sweep has a clean Meets→Below cutoff. */
  boundary: number | null;
  combined: boolean;
  title: string;
  detail: string;
}

export function summarize(
  configs: Config[],
  jobs: Job[],
  key: QuestionKey,
  threshold: number | null,
): SweepSummary {
  const states = configs.map((group) => verdict(group, key, threshold));
  const missed = states.indexOf("Below");
  const combined = unique(jobs.map((j) => j.family)).length > 1;
  const bracket =
    !combined &&
    missed > 0 &&
    states.slice(0, missed).every((s) => s === "Meets") &&
    states.slice(missed).every((s) => s === "Below");
  const first = configs[0]?.[0];
  const last = configs.at(-1)?.[0];
  let title: string;
  let detail: string;
  if (!first || !last) {
    title = "No jobs in this selection";
    detail = "Choose another chip, period or experiment group.";
  } else if (threshold == null) {
    title = "Explore the measured values";
    detail = "Set a target between 0 and 1 to classify the tested settings.";
  } else if (combined) {
    const passed = states.filter((state) => state === "Meets").length;
    title = `${passed} of ${configs.length} experiments met the target`;
    detail =
      "Each experiment is assessed across all jobs in the selected periods. Choose a group of related experiments to inspect its measured boundary.";
  } else if (configs.length === 1) {
    title = "One setting tested";
    detail = `${first.config} · ${states[0]}. More settings are needed to locate a boundary.`;
  } else if (bracket) {
    const pass = configs[missed - 1];
    const miss = configs[missed];
    const values = (group: Job[]) => group.map((j) => format(key, j[key])).join(" and ");
    title = `${pass[0].config} met the target. ${miss[0].config} missed it.`;
    detail = `${pass[0].config} jobs: ${values(pass)}. ${miss[0].config} jobs: ${values(miss)}.`;
    if (key === "success" && miss.every((j) => j.dsr != null)) {
      detail += ` DSR at ${miss[0].config}: ${miss.map((j) => score(j.dsr)).join(" and ")}.`;
    }
  } else if (states.every((v) => v === "Meets")) {
    title = `Met the target through ${last.config}`;
    detail = "Every tested setting in this family met the requirement.";
  } else if (states.every((v) => v === "Below")) {
    title = "No tested setting met the target";
    detail = `The measured sweep covers ${first.config} through ${last.config}.`;
  } else {
    title = "No single measured cutoff";
    detail = "Inspect each setting for reversals, split repetitions or missing evidence.";
  }
  return { states, boundary: bracket ? missed : null, combined, title, detail };
}

/** The axis on which configurations were swept; the shaded bracket only applies there. */
export const sweepAxis = (configs: Config[]) =>
  configs[0]?.[0].designDepth != null ? "depth" : "width";
