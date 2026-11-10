/**
 * View state <-> URL hash. Every view is a shareable link, e.g.
 * #a=GROVER&q=dsr&t=0.6&v=job&c=S4-1&r=<ref>:2
 * Unknown or stale values are ignored by the explorer, which falls back to defaults.
 */
export type ViewName = "range" | "job" | "calibration";

export interface UrlState {
  section?: string;
  pair?: string;
  matrix?: string;
  resource?: string;
  outcome?: string;
  backend?: string;
  period?: string;
  algorithm?: string;
  family?: string;
  question?: string;
  target?: string;
  axis?: string;
  view?: ViewName;
  config?: string;
  run?: string;
  bins?: string;
  calChip?: string;
  calDate?: string;
  calMode?: string;
  calGate?: string;
  calComponent?: string;
  calScoped?: boolean;
  calExpanded?: boolean;
  calOrigin?: ViewName;
}

const keys: Record<Exclude<keyof UrlState, "calScoped" | "calExpanded">, string> = {
  section: "section", pair: "pair", matrix: "matrix", resource: "resource", outcome: "outcome",
  backend: "b",
  period: "p",
  algorithm: "a",
  family: "f",
  question: "q",
  target: "t",
  axis: "x",
  view: "v",
  config: "c",
  run: "r",
  bins: "n",
  calChip: "cc",
  calDate: "cd",
  calMode: "cm",
  calGate: "cg",
  calComponent: "cx",
  calOrigin: "co",
};
const views = new Set<string>(["range", "job", "calibration"]);

export function parseHash(hash: string): UrlState {
  const params = new URLSearchParams(hash.replace(/^#/, ""));
  const state: UrlState = {};
  for (const [name, key] of Object.entries(keys) as [keyof typeof keys, string][]) {
    const value = params.get(key);
    if (value == null) continue;
    if ((name === "view" || name === "calOrigin") && !views.has(value)) continue;
    (state as Record<string, string>)[name] = value;
  }
  if (params.get("cs") === "1") state.calScoped = true;
  if (params.get("ce") === "1") state.calExpanded = true;
  return state;
}

export function formatHash(state: UrlState): string {
  const params = new URLSearchParams();
  for (const [name, key] of Object.entries(keys) as [keyof typeof keys, string][]) {
    const value = state[name];
    if (value != null && value !== "") params.set(key, value);
  }
  if (state.calScoped) params.set("cs", "1");
  if (state.calExpanded) params.set("ce", "1");
  const text = params.toString();
  return text ? `#${text}` : "";
}
