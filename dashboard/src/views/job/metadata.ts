import type { Job } from "../../lib/data/types";
import { jobTime } from "../../lib/domain/format";
import type { MetadataEntry } from "../../lib/ui/Metadata.svelte";

export const jobMetadata = (job: Job): MetadataEntry[] => [
  ["chip", "Chip", job.backend],
  ["shots", "Shots", job.shots.toLocaleString()],
  ["bits", "Measured bits", job.bits],
  ["depth", "ISA depth", job.depth],
  ["experiment", "Experiment", job.config],
  ["date", "Measured at", jobTime(job)],
];
