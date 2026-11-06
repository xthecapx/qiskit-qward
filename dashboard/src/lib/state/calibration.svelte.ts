import { tick } from "svelte";
import { loadCalibrations } from "../data/load";
import type { CalibrationRecord, Job } from "../data/types";
import {
  gatesFor,
  isGate,
  matchesQuery,
  mergeComponents,
  modeKeys,
  snapshotDates,
  valueDomain,
  worstFirst,
  type CalibrationMode,
} from "../domain/calibration";
import { unique } from "../domain/format";
import { pick } from "../domain/scope";
import type { Explorer } from "./explorer.svelte";
import type { UrlState, ViewName } from "./url";

type LoadState = "idle" | "loading" | "ready" | "error";

/**
 * Calibration explorer state. Opened either for the whole chip or scoped to one job,
 * where it shows that job's preparation snapshot until the user expands to chip history.
 */
export class CalibrationState {
  status = $state<LoadState>("idle");
  error = $state("");
  records = $state.raw<Record<string, CalibrationRecord>>({});

  /** The job whose hardware is inspected, or null for independent exploration. */
  job = $state.raw<Job | null>(null);
  origin = $state<ViewName>("range");
  expanded = $state(false);
  chipChoice = $state("");
  dateChoice = $state("");
  mode = $state<CalibrationMode>("readout");
  gateChoice = $state("");
  componentChoice = $state<string | null>(null);
  query = $state("");

  private opener: HTMLElement | null = null;
  private savedScroll = 0;

  private readonly explorer: Explorer = null!;

  constructor(explorer: Explorer) {
    this.explorer = explorer;
  }

  // ---- data --------------------------------------------------------------------------------
  async ensureLoaded() {
    if (this.status === "ready" || this.status === "loading") return;
    this.status = "loading";
    try {
      this.records = (await loadCalibrations()).calibrations;
      this.status = "ready";
    } catch (error) {
      this.error = error instanceof Error ? error.message : String(error);
      this.status = "error";
    }
  }

  chips = $derived(unique(Object.values(this.records).map((r) => r.backend)).sort());
  chip = $derived(
    this.job?.backend ?? pick(this.chips, this.chipChoice, this.explorer.backend) ?? "",
  );
  scoped = $derived(this.job != null && !this.expanded);
  jobRecord = $derived(this.job ? this.records[this.job.ref] : undefined);
  chipRecords = $derived(
    Object.values(this.records).filter((r) => r.backend === this.chip && r.at),
  );
  dates = $derived(snapshotDates(this.chipRecords));
  date = $derived(
    this.scoped
      ? (this.jobRecord?.at ?? "")
      : (pick(this.dates, this.dateChoice, this.jobRecord?.at, this.dates.at(-1)) ?? ""),
  );
  available = $derived(
    this.chipRecords.length > 0 && (!this.job || this.expanded || Boolean(this.jobRecord?.at)),
  );
  gates = $derived(isGate(this.mode) ? gatesFor(this.chipRecords, this.mode) : []);
  gate = $derived(
    pick(this.gates, this.gateChoice, this.mode === "single" ? "sx" : undefined) ?? "",
  );
  current = $derived(
    this.scoped
      ? this.jobRecord
        ? [this.jobRecord]
        : []
      : this.chipRecords.filter((r) => r.at === this.date),
  );
  components = $derived(mergeComponents(this.current, this.mode, this.gate));
  domain = $derived(valueDomain(this.chipRecords, this.mode, this.gate));
  filtered = $derived(this.components.filter((c) => matchesQuery(c, this.query)));
  component = $derived(
    this.filtered.find((c) => c.id === this.componentChoice) ?? worstFirst(this.filtered, this.mode),
  );
  /** One value per snapshot date for the selected component; null where not observed. */
  history = $derived.by(() => {
    const id = this.component?.id;
    return this.dates.map((at) =>
      id == null
        ? null
        : (mergeComponents(
            this.chipRecords.filter((r) => r.at === at),
            this.mode,
            this.gate,
          ).find((c) => c.id === id)?.value ?? null),
    );
  });

  // ---- actions -----------------------------------------------------------------------------
  setMode(mode: CalibrationMode) {
    this.mode = mode;
    this.gateChoice = "";
    this.componentChoice = null;
  }
  setChip(chip: string) {
    this.chipChoice = chip;
    this.dateChoice = "";
    this.componentChoice = null;
  }
  selectComponent(id: string) {
    this.componentChoice = id;
  }
  /** From the map: selecting clears any search so the choice is visible. */
  pickFromMap(id: string) {
    if (!this.components.some((c) => c.id === id)) return;
    this.query = "";
    this.componentChoice = id;
  }
  /** From the history chart: jump to that snapshot, expanding beyond the job if needed. */
  pickDate(at: string) {
    this.expanded = true;
    this.dateChoice = at;
  }
  toggleExpanded() {
    this.expanded = !this.expanded;
    if (!this.expanded) this.query = "";
    this.dateChoice = this.expanded ? this.date : "";
  }

  async open(job: Job | null, opener: HTMLElement | null) {
    if (this.explorer.view === "calibration") await this.close();
    this.opener = opener;
    this.savedScroll = window.scrollY;
    this.origin = this.explorer.view;
    this.job = job;
    this.expanded = false;
    this.componentChoice = null;
    this.query = "";
    this.mode = "readout";
    this.dateChoice = "";
    this.explorer.view = "calibration";
    void this.ensureLoaded();
    await tick();
    document.getElementById("calibration-title")?.focus({ preventScroll: true });
    document.getElementById("calibration-view")?.scrollIntoView({ block: "start", behavior: "instant" });
  }

  async close() {
    this.explorer.view = this.origin;
    this.job = null;
    await tick();
    // Views are unmounted while hidden, so the opener is usually re-created: find it by id.
    const fallback = this.origin === "job" ? "histogram-title" : "plot-title";
    const opener = this.opener?.isConnected
      ? this.opener
      : (document.getElementById(this.opener?.id ?? "") ?? document.getElementById(fallback));
    opener?.focus({ preventScroll: true });
    window.scrollTo({ top: this.savedScroll, behavior: "instant" });
  }

  // ---- URL ---------------------------------------------------------------------------------
  apply(state: UrlState) {
    if (state.view !== "calibration") return;
    this.origin = state.calOrigin === "job" ? "job" : "range";
    this.job = state.calScoped ? (this.explorer.selectedJob ?? null) : null;
    if (this.origin === "job" && !this.explorer.selectedJob) this.origin = "range";
    this.expanded = Boolean(state.calExpanded);
    this.chipChoice = state.calChip ?? "";
    this.dateChoice = state.calDate ?? "";
    if (state.calMode && (modeKeys as string[]).includes(state.calMode)) {
      this.mode = state.calMode as CalibrationMode;
    }
    this.gateChoice = state.calGate ?? "";
    this.componentChoice = state.calComponent ?? null;
    this.explorer.view = "calibration";
    void this.ensureLoaded();
  }

  toUrl(): Partial<UrlState> {
    if (this.explorer.view !== "calibration") return {};
    return {
      calOrigin: this.origin === "job" ? "job" : undefined,
      calScoped: this.job != null,
      calExpanded: this.expanded,
      calChip: this.job ? undefined : this.chip,
      calDate: this.scoped ? undefined : this.date,
      calMode: this.mode === "readout" ? undefined : this.mode,
      calGate: isGate(this.mode) ? this.gate : undefined,
      calComponent: this.componentChoice ?? undefined,
    };
  }
}
