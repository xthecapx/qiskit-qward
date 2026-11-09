import type { Job } from "../data/types";
import { compareText, unique } from "./format";

/** One measurement period is one (campaign, window) collection. */
export const windowKey = (job: Pick<Job, "campaign" | "window">) =>
  JSON.stringify([job.campaign, job.window]);

/** Run numbers repeat across collections, so the reference identifies the job. */
export const jobKey = (job: Pick<Job, "ref" | "rep">) => `${job.ref}:${job.rep}`;

export const ALL_PERIODS = "";
export const ALL_GROUPS = "";

export interface Period {
  id: string;
  label: string;
  dateLabel: string;
  jobs: Job[];
}

const shortDate = (value: string) =>
  new Date(value).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  });
const shortTime = (value: string) =>
  new Date(value).toLocaleTimeString("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC",
    hourCycle: "h23",
  });

export function periodLabel(times: string[]): string {
  if (!times.length) return "Date unavailable";
  const start = times[0];
  const end = times.at(-1)!;
  const stamp = (value: string) => `${shortDate(value)} · ${shortTime(value)}`;
  const finish = shortDate(start) === shortDate(end) ? shortTime(end) : stamp(end);
  return `${stamp(start)}–${finish}`;
}

/** Collections on one chip, newest first by their last completion. */
export function periods(jobs: Job[], backend: string): Period[] {
  const grouped = new Map<string, Job[]>();
  for (const job of jobs) {
    if (job.backend !== backend) continue;
    const id = windowKey(job);
    if (!grouped.has(id)) grouped.set(id, []);
    grouped.get(id)!.push(job);
  }
  return [...grouped]
    .map(([id, rows]) => {
      const times = rows.map((j) => j.finishedAt).filter(Boolean).sort();
      return { id, rows, times };
    })
    .sort((a, b) => (b.times.at(-1) ?? "").localeCompare(a.times.at(-1) ?? ""))
    .map(({ id, rows, times }) => {
      const first = times.length ? shortDate(times[0]) : "Date unavailable";
      const last = times.length ? shortDate(times[times.length - 1]) : first;
      return { id, label: periodLabel(times), dateLabel: first === last ? first : `${first} – ${last}`, jobs: rows };
    });
}

export function inPeriod(jobs: Job[], backend: string, period: string): Job[] {
  return jobs.filter(
    (j) => j.backend === backend && (period === ALL_PERIODS || windowKey(j) === period),
  );
}

export const algorithmsIn = (jobs: Job[]) => unique(jobs.map((j) => j.algorithm)).sort();

export const familiesIn = (jobs: Job[], algorithm: string) =>
  unique(jobs.filter((j) => j.algorithm === algorithm).map((j) => j.family)).sort(compareText);

export function selectJobs(jobs: Job[], algorithm: string, family: string): Job[] {
  return jobs.filter(
    (j) => j.algorithm === algorithm && (family === ALL_GROUPS || j.family === family),
  );
}

/** A configuration's jobs in time order. Configurations sort by designed depth, then width. */
export type Config = Job[];

export function groupConfigs(jobs: Job[]): Config[] {
  return unique(jobs.map((j) => j.config))
    .map((id) =>
      jobs
        .filter((j) => j.config === id)
        .sort((a, b) => (a.finishedAt || "").localeCompare(b.finishedAt || "") || a.rep - b.rep),
    )
    .sort(
      (a, b) =>
        (a[0].designDepth ?? a[0].width) - (b[0].designDepth ?? b[0].width) ||
        compareText(a[0].config, b[0].config),
    );
}

/** Keep the requested value when available, else the previous one, else the first. */
export function pick<T>(values: T[], ...preferred: (T | null | undefined)[]): T | undefined {
  for (const value of preferred) if (value != null && values.includes(value)) return value;
  return values[0];
}
