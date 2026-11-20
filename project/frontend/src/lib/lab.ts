export type LabGroupId = "rating" | "classification" | "clustering" | "association" | "pca" | "reinforcement";
export type MetricReference = { value:number; operator:"gte"|"lte"|"gt"; kind:"quality"|"budget"|"screening"|"baseline"; unit:string; minimum:number; maximum:number|null; rationale:string; interpretation:string; caution:string };
export type Criterion = { key: string; name: string; definition: string; interpretation: string; question: string; reference?:MetricReference };
export type ChartGuide = { name: string; axes: string; question: string };
export type LabGroup = { id: LabGroupId; title: string; question: string; algorithms: { id: string; name: string; description: string; output: string }[]; metrics: Criterion[]; charts: ChartGuide[] };
export type DatasetStats = { ratings:number; users:number; movies:number; catalog_movies:number; genres:number; rating_min:number; rating_max:number; rating_mean:number; sparsity:number };
export type LabCatalog = { version: number; source: string; groups: LabGroup[]; data_ready: boolean; statistics?:DatasetStats; dataset?:{splits?:Record<string,{ratings:number;users:number;movies:number}>;settings?:{data?:{user_sample_fraction?:number}}} };
export type LabPoint = { x: string | number; y: number | null };
export type LabChart = { index: number; metric_key?: string; title: string; kind: "bar" | "line" | "scatter"; x_label: string; y_label: string; series: { label: string; points: LabPoint[] }[]; split: string };
export type LabRow = { algorithm: string; name: string; split: string; [key: string]: unknown };
export type LabRule = { antecedents:string[]; consequents:string[]; support:number; confidence:number; lift:number; antecedent_ids?:number[]; consequent_ids?:number[]; group?:number; variant_id?:string; support_in_group?:number; support_global?:number; confidence_global?:number; lift_global?:number };
export type ReductionVariant = { variant_id:string; method:string|null; components:number|null; [key:string]:unknown };
export type GroupedAprioriVariant = { variant_id:string; k:number; rule_summary:{rules:number;support_in_group_mean:number|null;support_global_mean:number|null;fallback_groups:number;empty_transactions:number}; groups:{group:number;users:number;status:string;rule_count:number;itemsets?:number}[]; validation_ndcg:number|null; validation_coverage:number; model_bytes:number; fit_metadata:Record<string,unknown> };
export type LabResult = {
  schema_version: number; group: LabGroupId; rows: LabRow[]; charts: LabChart[]; notes: string[]; selection?:string;
  reduction?:{mode:string;variants:ReductionVariant[];selected_variant_id?:string|null;selection_metric?:string;feature_family?:string;transaction_reduction?:boolean;reason?:string};
  rl_protocol?:{name:string;training_split:string;evaluation:string;checkpoint_selection:string;checkpoint_step:number;test_used_for_selection:boolean;action_limit_requested:number;action_limit_actual:number;horizon:number;train_support_query_users:number;train_excluded_support_query_users:number;train_sweeps_requested:number;train_steps_budget:number;train_steps_actual:number;validation_sampled_users:number;test_sampled_users:number;training_user_limit:number;validation_cohort?:Record<string,number>;test_cohort?:Record<string,number>};
  paired_test_uncertainty?:{reference_policy:string;policies:Record<string,{users:number;mean_delta:number|null;ci95:number[]|null;reason?:string|null}>};
  best_configs?:Record<string,unknown>;
  association_data?:{mode:"full_prepared_train";train_ratings:number;train_users:number;train_movies:number;positive_ratings:number;positive_movies:number;candidate_movies:number;retained_positive_ratings:number;empty_transactions:number;transaction_bytes:number;grouped?:Record<string,unknown>};
  group_rules?:LabRule[]; grouped_variants?:GroupedAprioriVariant[];
  clustering_protocol?:string; cluster_min_coverage?:number; profile_kind?:string;
  training?:{split:string;ratings:number;eligible_ratings:number;eligible_users:number;scope:string};
  recommendation?:{k?:number;cohort_users?:number;candidate_movies:number;protocol?:string;rows:{algorithm:string;variant_id:string;split:string;recall_at_10:number|null;ndcg_at_10:number|null;evaluated_users:number;sampled_users:number;clustered_users?:number;clustered_recall_at_10?:number|null;clustered_ndcg_at_10?:number|null;group_assigned_users?:number;group_coverage?:number;recommendation_coverage?:number;fallback_users?:number;candidate_pool_size?:number;eligible_positives?:number;candidate_eligible_positives?:number;candidate_positive_coverage?:number|null}[]};
  cv_protocol?:string; cv_folds?:number;
  cv?:Record<string,{metrics:Record<string,{mean:number;std:number;n:number}>;folds:unknown[]}>;
  class_balance?:Record<string,{positive:number;negative:number;positive_fraction:number}>;
  rules?:LabRule[]; search?:Record<string,unknown>[];
  clusters?:{algorithm:string;variant_id?:string;qualified?:boolean;stability?:{adjusted_rand_mean:number|null;joint_coverage_mean:number;comparisons:number};profiles:{cluster:number;users:number;genre_means:Record<string,number>}[]}[];
  runtime?:{isolated_process?:boolean;memory_budget_mb?:number;worker_peak_private_mb?:number;system_min_available_mb?:number;processor:string;cpu_count:number;ram_bytes:number;device:string;python:string;platform:string};
  provenance: { split_files?:Record<string,string>|null;source_prepared_files?:Record<string,string>;metadata_file?:string; association_source?:{mode:"full_prepared_train";train_file:string;movies_file:string;heldout_files:string[];prepared_user_sample_fraction:number|null}; exclusions?:{excluded_users:number;excluded_ratings:number}; protocol?: string; sampled_ratings: number; sampled_users: number; sample?:DatasetStats; prepared?:DatasetStats; splits: Record<string, { ratings: number }> };
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
