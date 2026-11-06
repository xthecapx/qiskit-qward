import type { Job, QuestionKey } from "../data/types";

export const unique = <T>(values: Iterable<T>): T[] => [...new Set(values)];

export const compareText = (a: string, b: string) => a.localeCompare(b, "en", { numeric: true });

export const pct = (value: number | null | undefined) =>
  value == null ? "Unavailable" : (100 * value).toFixed(1) + "%";

export const score = (value: number | null | undefined) =>
  value == null ? "Unavailable" : value.toFixed(3);

export const format = (key: QuestionKey | "baseline", value: number | null | undefined) =>
  key === "success" ? pct(value) : score(value);

/** Metrics where a smaller value is better. */
export const lowerIsBetter = (key: QuestionKey) => key === "coarse" || key === "tvd";

export const comparator = (key: QuestionKey) => (lowerIsBetter(key) ? "≤" : "≥");

export const plural = (count: number, word: string) => `${count} ${word}${count === 1 ? "" : "s"}`;

export function jobTime(job: Pick<Job, "finishedAt">): string {
  if (!job.finishedAt) return "Time unavailable";
  return (
    new Date(job.finishedAt).toLocaleString("en-GB", {
      day: "numeric",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      timeZone: "UTC",
      hourCycle: "h23",
    }) + " UTC"
  );
}

/** Calibration timestamps are ISO-like with a space separator. Show minutes in UTC. */
export function calibrationStamp(value: string | null | undefined): string {
  if (!value) return "Timestamp unavailable";
  const date = new Date(value.replace(" ", "T"));
  return Number.isNaN(date.valueOf()) ? value : date.toISOString().slice(0, 16).replace("T", " ");
}

export const calibrationTime = (value: string) => new Date(value.replace(" ", "T")).valueOf();
