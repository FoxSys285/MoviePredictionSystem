export type LabGroupId = "rating";
export type Criterion = { key: string; name: string };
export type LabGroup = {
  id: LabGroupId;
  title: string;
  question: string;
  algorithms: { id: string; name: string; description: string; output: string }[];
  metrics: Criterion[];
};
export type DatasetStats = {
  ratings: number;
  users: number;
  movies: number;
  catalog_movies: number;
  genres: number;
  rating_min: number;
  rating_max: number;
  rating_mean: number;
  sparsity: number;
};
export type PresetSetting = { label: string; value: string };
export type LabPreset = {
  id: string;
  version: number;
  label: string;
  common: PresetSetting[];
  algorithms: Record<string, { settings: PresetSetting[]; reduction: { method: "pca"; components: number } | null }>;
  tuning: boolean;
};
export type LabCatalog = {
  version: number;
  source: string;
  groups: LabGroup[];
  presets?: Partial<Record<LabGroupId, LabPreset>>;
  data_ready: boolean;
  statistics?: DatasetStats;
};
export type LabPoint = { x: string | number; y: number | null };
export type LabChart = {
  index: number;
  metric_key?: string;
  title: string;
  kind: "bar";
  x_label: string;
  y_label: string;
  series: { label: string; points: LabPoint[] }[];
  split: string;
  baseline?: { label: string; value: number };
};
export type LabRow = { algorithm: string; name: string; split: string; variant_id?: string; [key: string]: unknown };
export type LabResult = {
  schema_version: number;
  group: LabGroupId;
  rows: LabRow[];
  charts: LabChart[];
  notes: string[];
  selection?: string;
  selected_variant_id?: string;
  config?: Record<string, unknown>;
  best_configs?: Record<string, unknown>;
  baseline?: { name: string; fit_scope: string; train_ratings: number; prediction: number; rows: { split: string; ratings: number; rmse: number; mae: number }[] };
  cv_protocol?: string;
  cv_folds?: number;
  cv?: Record<string, { metrics: Record<string, { mean: number; std: number }>; folds: unknown[] }>;
  runtime?: Record<string, unknown>;
  provenance: {
    sampled_ratings: number;
    sampled_users: number;
    sample?: DatasetStats;
    prepared?: DatasetStats;
    protocol?: string;
    splits: Record<string, { ratings: number; users?: number; movies?: number }>;
  };
};
export type LabJob = {
  id: string;
  group: LabGroupId;
  status: "queued" | "running" | "completed" | "failed" | "cancelled" | "interrupted";
  progress: number;
  message: string;
  created_at: string;
  finished_at?: string;
  cancel_requested?: boolean;
  resume_available?: boolean;
  completed_work_units?: number;
  epoch_checkpoint_available?: boolean;
  request?: { algorithms?: string[]; reduction_choice?: "off" | "pca" | null; reduction_mode?: "off" | "enabled" | "compare"; reduction_methods?: "pca"[] };
  result?: LabResult;
};
export const busy = (job: LabJob) => ["queued", "running"].includes(job.status);
export const metricNumber = (value: unknown, language = "vi") => {
  if (typeof value !== "number" || !Number.isFinite(value)) return "N/A";
  const tiny = value !== 0 && Math.abs(value) < 1e-6;
  return value.toLocaleString(language === "en" ? "en-US" : "vi-VN", tiny
    ? { notation: "scientific", maximumSignificantDigits: 6 }
    : { maximumFractionDigits: 8 });
};
