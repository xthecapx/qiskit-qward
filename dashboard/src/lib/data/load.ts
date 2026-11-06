import { SCHEMA_VERSION, type CalibrationData, type IndexData } from "./types";

async function fetchJson<T extends { schemaVersion: number }>(name: string): Promise<T> {
  const response = await fetch(new URL(`data/${name}`, document.baseURI));
  if (!response.ok) throw new Error(`Could not load ${name} (HTTP ${response.status}).`);
  const payload = (await response.json()) as T;
  if (payload.schemaVersion !== SCHEMA_VERSION) {
    throw new Error(
      `${name} uses schema ${payload.schemaVersion}; this page expects ${SCHEMA_VERSION}. Rebuild the dashboard.`,
    );
  }
  return payload;
}

export const loadIndex = () => fetchJson<IndexData>("index.json");

let calibrations: Promise<CalibrationData> | undefined;
/** Calibrations are large and only needed by the explorer, so fetch once on demand. */
export function loadCalibrations(): Promise<CalibrationData> {
  calibrations ??= fetchJson<CalibrationData>("calibrations.json").catch((error) => {
    calibrations = undefined;
    throw error;
  });
  return calibrations;
}
