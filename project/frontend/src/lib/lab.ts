export type LabGroupId = "rating" | "classification" | "clustering" | "association" | "pca" | "reinforcement";
export type MetricReference = { value:number; operator:"gte"|"lte"|"gt"; kind:"quality"|"budget"|"screening"|"baseline"; unit:string; minimum:number; maximum:number|null; rationale:string; interpretation:string; caution:string };
export type Criterion = { key: string; name: string; definition: string; interpretation: string; question: string; reference?:MetricReference };
export type ChartGuide = { name: string; axes: string; question: string };
export type LabGroup = { id: LabGroupId; title: string; question: string; algorithms: { id: string; name: string; description: string; output: string }[]; metrics: Criterion[]; charts: ChartGuide[] };
export type DatasetStats = { ratings:number; users:number; movies:number; catalog_movies:number; genres:number; rating_min:number; rating_max:number; rating_mean:number; sparsity:number };
export type LabCatalog = { version: number; source: string; groups: LabGroup[]; data_ready: boolean; statistics?:DatasetStats };
export type LabPoint = { x: string | number; y: number | null };
export type LabChart = { index: number; metric_key?: string; title: string; kind: "bar" | "line" | "scatter"; x_label: string; y_label: string; series: { label: string; points: LabPoint[] }[]; split: string };
export type LabRow = { algorithm: string; name: string; split: string; [key: string]: unknown };
export type LabRule = { antecedents:string[]; consequents:string[]; support:number; confidence:number; lift:number };
export type LabResult = {
  schema_version: number; group: LabGroupId; rows: LabRow[]; charts: LabChart[]; notes: string[]; selection?:string;
  best_configs?:Record<string,unknown>;
  cv_protocol?:string; cv_folds?:number;
  cv?:Record<string,{metrics:Record<string,{mean:number;std:number;n:number}>;folds:unknown[]}>;
  class_balance?:Record<string,{positive:number;negative:number;positive_fraction:number}>;
  rules?:LabRule[]; search?:Record<string,unknown>[];
  clusters?:{algorithm:string;profiles:{cluster:number;users:number;genre_means:Record<string,number>}[]}[];
  runtime?:{isolated_process?:boolean;memory_budget_mb?:number;worker_peak_private_mb?:number;system_min_available_mb?:number;processor:string;cpu_count:number;ram_bytes:number;device:string;python:string;platform:string};
  provenance: { exclusions?:{excluded_users:number;excluded_ratings:number}; protocol?: string; sampled_ratings: number; sampled_users: number; sample?:DatasetStats; prepared?:DatasetStats; splits: Record<string, { ratings: number }> };
};
export type LabJob = { id: string; group: LabGroupId; status: "queued" | "running" | "completed" | "failed" | "cancelled" | "interrupted"; progress: number; message: string; created_at: string; finished_at?: string; cancel_requested?: boolean; result?: LabResult };
export const busy = (job: LabJob) => ["queued", "running"].includes(job.status);
export const metricNumber = (value: unknown, language = "vi") => {
  if (typeof value !== "number" || !Number.isFinite(value)) return "N/A";
  const tiny=value!==0 && Math.abs(value)<1e-6;
  return value.toLocaleString(language === "en" ? "en-US" : "vi-VN", tiny
    ? { notation:"scientific",maximumSignificantDigits:6 }
    : { maximumFractionDigits:8 });
};
