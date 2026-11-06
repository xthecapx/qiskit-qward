import type { CalibrationRecord, CalibrationValueKey } from "../data/types";
import { calibrationTime, compareText, unique } from "./format";

export type CalibrationMode = CalibrationValueKey;

export const modes: Record<CalibrationMode, { option: string; title: string; explanation: string }> = {
  readout: {
    option: "Readout errors",
    title: "Readout error",
    explanation:
      "Average probability of reporting the wrong state when a known 0 or 1 is prepared. The two directions below can differ.",
  },
  single: {
    option: "Single-qubit gate errors",
    title: "Single-qubit gate error",
    explanation:
      "The provider’s calibration error estimate for this operation on one physical qubit. Compare the same operation across dates.",
  },
  pair: {
    option: "Two-qubit gate errors",
    title: "Two-qubit gate error",
    explanation:
      "The provider’s calibration error estimate for this operation on the ordered pair shown. The line represents an operation between qubits, not a separately measured coupler error.",
  },
  t1: {
    option: "T1 · relaxation time",
    title: "T1 · relaxation time",
    explanation:
      "Characteristic time for an excited qubit to lose energy. Longer times indicate slower relaxation; this is a duration, not an error probability.",
  },
  t2: {
    option: "T2 · coherence time",
    title: "T2 · coherence time",
    explanation:
      "Characteristic coherence time, affected by energy relaxation and dephasing. Longer times indicate longer coherence; this is a duration, not an error probability.",
  },
};
export const modeKeys = Object.keys(modes) as CalibrationMode[];

/** Durations are better when longer; error rates are better when lower. */
export const isDuration = (mode: CalibrationMode) => mode === "t1" || mode === "t2";
export const isGate = (mode: CalibrationMode) => mode === "single" || mode === "pair";

export function calibrationNumber(mode: CalibrationMode, value: number | null | undefined): string {
  if (value == null) return "Unavailable";
  return isDuration(mode) ? `${value.toFixed(1)} μs` : `${(value * 100).toFixed(3)}%`;
}

export interface ComponentValue {
  id: string;
  qubits: number[];
  value: number | null;
}

export interface Component extends ComponentValue {
  conflict: boolean;
}

export const componentLabel = (item: Pick<ComponentValue, "qubits">) =>
  item.qubits.map((q) => `Q${q}`).join(" → ");

export function componentValues(
  record: CalibrationRecord,
  mode: CalibrationMode,
  gate: string,
): ComponentValue[] {
  if (mode === "readout" || isGate(mode)) {
    const width = mode === "pair" ? 2 : 1;
    return record.operations
      .filter((op) =>
        mode === "readout" ? op.name === "measure" : op.name === gate && op.qubits.length === width,
      )
      .map((op) => ({ id: op.qubits.join("→"), qubits: op.qubits, value: op.error }));
  }
  const field = mode === "t1" ? "t1_s" : "t2_s";
  return Object.entries(record.qubits).map(([q, data]) => {
    const seconds = data[field];
    return { id: q, qubits: [Number(q)], value: seconds == null ? null : seconds * 1e6 };
  });
}

/**
 * Merge repeated observations of a component. Identical values are counted once;
 * conflicting values at the same timestamp become unavailable rather than averaged.
 */
export function mergeComponents(
  records: CalibrationRecord[],
  mode: CalibrationMode,
  gate: string,
): Component[] {
  const merged = new Map<string, ComponentValue & { seen: Set<number> }>();
  for (const record of records) {
    for (const item of componentValues(record, mode, gate)) {
      if (!merged.has(item.id)) merged.set(item.id, { ...item, seen: new Set() });
      if (item.value != null) merged.get(item.id)!.seen.add(item.value);
    }
  }
  return [...merged.values()]
    .map(({ seen, ...item }) => ({
      ...item,
      conflict: seen.size > 1,
      value: seen.size === 1 ? [...seen][0] : null,
    }))
    .sort((a, b) => compareText(a.id, b.id));
}

export const gatesFor = (records: CalibrationRecord[], mode: CalibrationMode) =>
  unique(
    records.flatMap((r) =>
      r.operations
        .filter((op) => op.name !== "measure" && op.qubits.length === (mode === "pair" ? 2 : 1))
        .map((op) => op.name),
    ),
  ).sort();

export const snapshotDates = (records: CalibrationRecord[]) =>
  unique(records.map((r) => r.at)).sort((a, b) => calibrationTime(a) - calibrationTime(b));

/** Fixed colour domain over every recorded date for the selected property. */
export function valueDomain(
  records: CalibrationRecord[],
  mode: CalibrationMode,
  gate: string,
): [number, number] {
  const known = records
    .flatMap((r) => componentValues(r, mode, gate))
    .map((item) => item.value)
    .filter((v): v is number => v != null);
  return known.length ? [Math.min(...known), Math.max(...known)] : [0, 1];
}

const LOW = [15, 126, 113];
const HIGH = [204, 93, 48];
export const UNAVAILABLE = "#bbc7c2";

/** Teal is better, orange is worse. The scale is relative, not a pass/fail threshold. */
export function scaleColor(
  value: number | null | undefined,
  domain: [number, number],
  mode: CalibrationMode,
): string {
  if (value == null) return UNAVAILABLE;
  let t = (value - domain[0]) / (domain[1] - domain[0] || 1);
  if (isDuration(mode)) t = 1 - t;
  return `rgb(${LOW.map((c, i) => Math.round(c + (HIGH[i] - c) * t)).join(",")})`;
}

/** The default component is the worst-performing one with a value. */
export function worstFirst(components: Component[], mode: CalibrationMode): Component | undefined {
  return [...components].sort((a, b) => {
    if (a.value == null) return 1;
    if (b.value == null) return -1;
    return isDuration(mode) ? a.value - b.value : b.value - a.value;
  })[0];
}

export const matchesQuery = (item: Component, query: string) =>
  componentLabel(item)
    .toLowerCase()
    .replaceAll(" ", "")
    .includes(query.toLowerCase().replaceAll(" ", ""));

export interface SchematicNode {
  id: string;
  x: number;
  y: number;
}

/**
 * Deterministic breadth-first schematic of observed connectivity. No invented physical
 * chip coordinates: disconnected components are laid out left to right with a gap.
 */
export function schematic(
  records: CalibrationRecord[],
  plotWidth: number,
): { nodes: SchematicNode[]; links: [string, number[]][] } {
  const qubits = unique(records.flatMap((r) => Object.keys(r.qubits))).sort(compareText);
  const present = new Set(qubits);
  const links = new Map<string, number[]>();
  for (const record of records) {
    for (const op of record.operations) {
      if (op.qubits.length === 2 && op.qubits.every((q) => present.has(String(q)))) {
        links.set(op.qubits.join("→"), op.qubits);
      }
    }
  }
  const neighbors = new Map(qubits.map((q) => [q, new Set<string>()]));
  for (const [a, b] of links.values()) {
    neighbors.get(String(a))!.add(String(b));
    neighbors.get(String(b))!.add(String(a));
  }
  const levels = new Map<string, number>();
  let offset = 0;
  for (const root of qubits) {
    if (levels.has(root)) continue;
    const queue = [root];
    levels.set(root, offset);
    for (let i = 0; i < queue.length; i++) {
      for (const next of neighbors.get(queue[i])!) {
        if (!levels.has(next)) {
          levels.set(next, levels.get(queue[i])! + 1);
          queue.push(next);
        }
      }
    }
    offset = Math.max(...levels.values()) + 2;
  }
  const layers = new Map<number, string[]>();
  for (const [q, level] of levels) {
    if (!layers.has(level)) layers.set(level, []);
    layers.get(level)!.push(q);
  }
  const maxLevel = Math.max(...levels.values(), 1);
  const maxRows = Math.max(...[...layers.values()].map((row) => row.length), 2);
  const nodes = qubits.map((q) => {
    const level = levels.get(q)!;
    const row = layers.get(level)!;
    return {
      id: q,
      x: 15 + (level / maxLevel) * (plotWidth - 30),
      y: 150 + (row.indexOf(q) - (row.length - 1) / 2) * (220 / maxRows) + (level % 2 ? 16 : -16),
    };
  });
  return { nodes, links: [...links] };
}

/** Physical-qubit → classical-bit mappings for a component in a job's recorded measurements. */
export function bitMappings(record: CalibrationRecord | undefined, qubits: number[]): string[] {
  if (!record) return [];
  return unique(
    record.measurements.flatMap((m) =>
      m.qubits.flatMap((q, i) => (qubits.includes(q) ? [`Q${q} → bit ${m.bits[i]}`] : [])),
    ),
  );
}
