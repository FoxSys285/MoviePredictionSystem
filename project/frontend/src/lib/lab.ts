export type LabGroupId = "rating" | "classification" | "clustering" | "association" | "pca";
export type Criterion = { key: string; name: string; definition: string; interpretation: string; question: string };
export type ChartGuide = { name: string; axes: string; question: string };
export type LabGroup = { id: LabGroupId; title: string; question: string; algorithms: { id: string; name: string; description: string; output: string }[]; metrics: Criterion[]; charts: ChartGuide[] };
export type LabCatalog = { version: number; source: string; groups: LabGroup[]; data_ready: boolean };
export type LabPoint = { x: string | number; y: number | null };
export type LabChart = { index: number; metric_key: string; title: string; kind: "bar"; x_label: string; y_label: string; series: { label: string; points: LabPoint[] }[]; split: "comparison" };
export type LabRow = { algorithm: string; name: string; split: "validation" | "test"; [key: string]: unknown };
export type LabResult = { schema_version: number; group: LabGroupId; rows: LabRow[]; charts: LabChart[]; notes: string[]; provenance: { sampled_ratings: number; sampled_users: number; splits: Record<string, { ratings: number }> } };
export type LabJob = { id: string; group: LabGroupId; status: "queued" | "running" | "completed" | "failed" | "cancelled" | "interrupted"; progress: number; message: string; created_at: string; cancel_requested?: boolean; result?: LabResult };
export const busy = (job: LabJob) => ["queued", "running"].includes(job.status);
export const metricNumber = (value: unknown, language = "vi") => {
  if (typeof value !== "number" || !Number.isFinite(value)) return "N/A";
  const tiny=value!==0 && Math.abs(value)<1e-6;
  return value.toLocaleString(language === "en" ? "en-US" : "vi-VN", tiny
    ? { notation:"scientific",maximumSignificantDigits:6 }
    : { maximumFractionDigits:8 });
};
