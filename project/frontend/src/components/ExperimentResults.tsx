import { useState } from "react";
import { Activity, Database, Download, Film, Tags, Users } from "lucide-react";
import { LabChartFigure } from "./LabChart";
import { AssociationTrainingSummary } from "./AssociationDataSource";
import { apiResourceUrl } from "../lib/api";
import { metricNumber, type DatasetStats, type LabGroup, type LabJob, type LabResult, type LabRow, type LabRule } from "../lib/lab";
import { usePreferences } from "../lib/preferences";

function ExperimentPipeline({group,result}:{group:LabGroup;result:LabJob["result"]}) {
  const {t}=usePreferences();
  if(!result)return null;
  const stages:Record<string,string[]>={
    rating:["Train theo thời gian từng user","Đặc trưng rating; loại nhãn của hàng Train","Scaler / PCA / TruncatedSVD fit trong Train hoặc fold","Mô hình và CV thời gian","Chọn cấu hình trên Validation","Refit Train + Validation, đánh giá Test"],
    classification:["Train theo thời gian từng user","Đặc trưng và chuẩn hóa fit trên Train","Reducer chỉ nén block tương tác user–genre","Logistic / Random Forest","Chọn threshold và model bằng Validation","Refit Train + Validation, đánh giá Test"],
    clustering:["Hồ sơ sở thích và tần suất từ Train","Chia user fit / giữ lại","Scaler / reducer fit trên user fit","K-Means hoặc DBSCAN; tìm lại cấu hình","Fit cuối trên toàn bộ Train","Đánh giá ranking tương lai Validation / Test"],
    reinforcement:["Train support / query theo thời gian","Reducer chỉ nén profile support","Giữ nguyên action mask và tín hiệu động","DQN học; Random / Popularity dùng cùng catalog","Chọn checkpoint bằng Validation","Phục hồi checkpoint và đánh giá Test đóng băng"],
    association:["Toàn bộ prepared Train và transaction dương","Apriori toàn cục giữ làm baseline","Reducer nén profile user Train rồi phân nhóm","Apriori riêng theo nhóm; transaction không bị nén","Chọn theo NDCG Validation","Test ranking trong catalog phim Train"],
    pca:["Profile user–genre Train","Scale và fit PCA / TruncatedSVD","Transform Validation / Test","Đo lỗi tái tạo và phương sai thành phần chiếu","Chọn theo Validation","Không diễn giải là độ chính xác rating"],
  };
  const variant=result.reduction?.selected_variant_id;
  return <section className="lab-section lab-pipeline" aria-label={t("Pipeline của thí nghiệm")}>
    <div className="lab-section-heading"><h2>{t("Pipeline của thí nghiệm")}</h2>{variant&&<span className="lab-badge">{t("Validation chọn {variant}",{variant})}</span>}</div>
    <ol>{stages[group.id].map((stage,index)=><li key={stage}><span>{String(index+1).padStart(2,"0")}</span><p>{t(stage)}</p></li>)}</ol>
    {group.id==="pca"&&<p className="lab-muted">{t("TruncatedSVD chiếu không centering; explained variance ratio là Var(projection) chia tổng phương sai đầu vào đã scale, không phải tỷ lệ năng lượng.")}</p>}
    {result.reduction&&<p className="lab-muted">{t("Chế độ {mode}; các representation dùng chung dữ liệu và seed.",{mode:result.reduction.mode})} {result.reduction.feature_family?t("Feature family: {family}.",{family:result.reduction.feature_family}):""} {result.reduction.transaction_reduction===false?t("Transaction Apriori giữ nguyên dạng boolean."):""}</p>}
  </section>;
}

function GroupedAprioriDetails({result}:{result:NonNullable<LabJob["result"]>}) {
  const {t}=usePreferences();
  if(!result.grouped_variants?.length)return null;
  return <>
    <section className="lab-section"><h2>{t("Apriori theo nhóm · support và coverage")}</h2><p className="lab-muted">{t("Support trong nhóm và support toàn cục dùng mẫu số khác nhau, nên được trình bày riêng. Reducer fit trên profile user Train; transaction rỗng vẫn nằm trong mẫu số của nhóm.")}</p>
      <div className="lab-table-scroll"><table><thead><tr><th>{t("Variant")}</th><th>{t("k nhóm")}</th><th>{t("Số luật")}</th><th>{t("Support trong nhóm")}</th><th>{t("Support toàn cục")}</th><th>{t("Nhóm fallback")}</th><th>{t("Validation NDCG@10")}</th><th>{t("Coverage nhóm")}</th></tr></thead><tbody>{result.grouped_variants.map(v=><tr key={v.variant_id}><th>{v.variant_id}</th><td>{v.k}</td><td>{v.rule_summary.rules.toLocaleString("vi-VN")}</td><td>{metricNumber(v.rule_summary.support_in_group_mean)}</td><td>{metricNumber(v.rule_summary.support_global_mean)}</td><td>{v.rule_summary.fallback_groups.toLocaleString("vi-VN")}</td><td>{metricNumber(v.validation_ndcg)}</td><td>{(v.validation_coverage*100).toFixed(1)}%</td></tr>)}</tbody></table></div>
    </section>
    {result.group_rules&&result.group_rules.length>0&&<section className="lab-section"><h2>{t("Luật Apriori trong nhóm")}</h2><div className="lab-table-scroll"><table><thead><tr><th>{t("Variant / nhóm")}</th><th>{t("Phim đã thích")}</th><th>{t("Phim gợi ý")}</th><th>{t("Support trong nhóm")}</th><th>{t("Support toàn cục")}</th><th>{t("Confidence nhóm")}</th><th>{t("Lift nhóm")}</th></tr></thead><tbody>{result.group_rules.slice(0,100).map((rule,index)=><tr key={rule.variant_id+"/"+rule.group+"/"+index}><th>{rule.variant_id} · nhóm {rule.group}</th><td>{rule.antecedents.join(" + ")}</td><td>{rule.consequents.join(" + ")}</td><td>{metricNumber(rule.support_in_group)}</td><td>{metricNumber(rule.support_global)}</td><td>{metricNumber(rule.confidence)}</td><td>{metricNumber(rule.lift)}</td></tr>)}</tbody></table></div></section>}
  </>;
}

function AprioriRanking({recommendation}:{recommendation:NonNullable<LabResult["recommendation"]>}) {
  const {t}=usePreferences();
  const baseline= (split:string)=>recommendation.rows.find(row=>row.algorithm==="apriori_global"&&row.split===split);
  return <section className="lab-section"><h2>{t("Apriori · xếp hạng trên phim tương lai")}</h2>
    <p className="lab-muted">{t("Catalog gồm {movies} phim thích phổ biến trên Train. Phim ngoài catalog không thể được gợi ý và vẫn làm giảm Recall; cohort và catalog được dùng chung cho các biến thể.",{movies:recommendation.candidate_movies.toLocaleString("vi-VN")})}</p>
    <div className="lab-table-scroll"><table><thead><tr><th>{t("Variant")}</th><th>{t("Tập")}</th><th>{t("User có positive")}</th><th>NDCG@10</th><th>{t("Δ NDCG so Apriori toàn cục")}</th><th>{t("Coverage nhóm")}</th><th>{t("Fallback user")}</th><th>{t("Positive trong catalog")}</th></tr></thead><tbody>{recommendation.rows.map(row=>{const global=baseline(row.split);const delta=row.ndcg_at_10!==null&&global?.ndcg_at_10!=null?row.ndcg_at_10-global.ndcg_at_10:null;return <tr key={row.variant_id+"/"+row.split}><th>{row.variant_id}</th><td>{row.split}</td><td>{row.evaluated_users.toLocaleString("vi-VN")}</td><td>{metricNumber(row.ndcg_at_10)}</td><td>{row.algorithm==="apriori_global"?"—":metricNumber(delta)}</td><td>{typeof row.group_coverage==="number"?(row.group_coverage*100).toFixed(1)+"%":"—"}</td><td>{typeof row.fallback_users==="number"?row.fallback_users.toLocaleString("vi-VN"):"—"}</td><td>{metricNumber(row.candidate_positive_coverage)}</td></tr>;})}</tbody></table></div>
  </section>;
}

export function DatasetSummary({stats,label}:{stats?:DatasetStats;label:string}) {
  const {t}=usePreferences();
  if(!stats)return null;
  const cells:[[string,unknown],...[string,unknown][]]=[
    ["Lượt đánh giá",stats.ratings],["Người dùng",stats.users],["Phim có đánh giá",stats.movies],["Thể loại",stats.genres],
    ["Phim trong danh mục",stats.catalog_movies],
  ];
  const icons=[Activity,Users,Film,Tags,Database];
  return <section className="lab-section lab-dataset-summary"><div className="lab-section-heading"><div className="lab-section-title"><span className="lab-section-icon"><Database size={21}/></span><div><h2>{t(label)}</h2><p>{t("Thống kê được tính từ dữ liệu MovieLens đã chuẩn bị.")}</p></div></div><span className="lab-badge">MovieLens 25M</span></div>
    <div className="lab-stat-grid">{cells.map(([name,value],index)=>{const Icon=icons[index];return <div className={`lab-stat-card tone-${index%4}`} key={name}><span className="lab-stat-icon"><Icon size={19}/></span><div><span>{t(name)}</span><strong>{typeof value==="number"?value.toLocaleString("vi-VN"):"Không có dữ liệu"}</strong></div></div>;})}</div></section>;
}

function Rules({rules}:{rules:LabRule[]}) {
  const [sort,setSort]=useState<"lift"|"support"|"confidence">("lift");
  const [limit,setLimit]=useState(50);
  const ordered=[...rules].sort((a,b)=>b[sort]-a[sort]);
  return <section className="lab-section"><div className="lab-section-heading"><h2>{rules.length.toLocaleString("vi-VN")} luật kết hợp</h2><label>Sắp xếp <select aria-label="Sắp xếp luật" value={sort} onChange={e=>setSort(e.target.value as typeof sort)}><option value="lift">Lift</option><option value="confidence">Confidence</option><option value="support">Support</option></select></label></div>
    <div className="lab-table-scroll"><table><thead><tr><th>Phim đã thích</th><th>Phim gợi ý</th><th>Support</th><th>Confidence</th><th>Lift</th></tr></thead><tbody>{ordered.slice(0,limit).map((r,i)=><tr key={i}><td>{r.antecedents.join(" + ")}</td><td>{r.consequents.join(" + ")}</td><td>{metricNumber(r.support)}</td><td>{metricNumber(r.confidence)}</td><td>{metricNumber(r.lift)}</td></tr>)}</tbody></table></div>
    {limit<rules.length&&<button className="lab-outline" onClick={()=>setLimit(v=>v+50)}>Xem thêm 50 luật</button>}
  </section>;
}

type ResultRecord = Record<string, unknown>;

function numericValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function shownValue(value: unknown, language = "vi"): string {
  return numericValue(value) === null ? "Không có dữ liệu" : metricNumber(value, language);
}

function variantId(row: LabRow): string {
  return String(row.variant_id || row.algorithm || "Không có dữ liệu");
}

function baseAlgorithm(row: LabRow): string {
  return String(row.base_algorithm || row.algorithm || "");
}

function representation(row: LabRow): string {
  const method = row.reduction_method;
  const components = numericValue(row.components);
  if (typeof method === "string") {
    return `${method === "pca" ? "PCA" : "TruncatedSVD"}${components ? ` · ${components} chiều` : ""}`;
  }
  return row.representation && row.representation !== "raw" && row.representation !== "not_applicable"
    ? String(row.representation)
    : "Biểu diễn gốc";
}

function algorithmName(id: string, group: LabGroup, fallback?: string): string {
  return group.algorithms.find((item) => item.id === id)?.name || fallback || id || "Không có dữ liệu";
}

function MiniBars({
  title,
  unit,
  items,
  lowerIsBetter = false,
}: {
  title: string;
  unit: string;
  items: { label: string; value: number | null }[];
  lowerIsBetter?: boolean;
}) {
  const valid = items.filter((item) => item.value !== null);
  if (valid.length < 2) return null;
  const values = valid.map((item) => item.value as number);
  const low = Math.min(...values);
  const high = Math.max(...values);
  const span = high - low;
  return (
    <article className="lab-result-chart">
      <h4>{title}</h4>
      <p>{unit}{lowerIsBetter ? " · thấp hơn tốt hơn" : " · cao hơn tốt hơn"}</p>
      {valid.map((item) => {
        const value = item.value as number;
        const width = span < 1e-12 ? 100 : lowerIsBetter
          ? 100 * (high - value) / span
          : 100 * (value - low) / span;
        return (
          <div className="lab-bar-row" key={item.label}>
            <span title={item.label}>{item.label}</span>
            <span className="lab-bar-track"><i style={{ width: `${Math.max(2, width)}%` }}/></span>
            <strong>{metricNumber(value)}</strong>
          </div>
        );
      })}
    </article>
  );
}

function ReductionSummary({ result, group }: { result: LabResult; group: LabGroup }) {
  const selectedId = String(result.selected_variant_id || result.reduction?.selected_variant_id || "");
  const relevantRows = result.rows.filter((row) =>
    ["test", "train_history", "train"].includes(row.split),
  );
  const reduced = new Map<string, LabRow>();
  const raw = new Set<string>();
  for (const row of relevantRows) {
    const meta = row.reduction as ResultRecord | undefined;
    const method = row.reduction_method || meta?.method;
    const owner = baseAlgorithm(row);
    if (method === "pca" || method === "truncated_svd") {
      if (!reduced.has(variantId(row))) reduced.set(variantId(row), row);
    } else if (row.representation === "raw" || row.representation === "raw_profile") {
      raw.add(owner);
    }
  }
  const variants = [...reduced.values()];
  if (!variants.length) {
    return (
      <div className="lab-reduction-results">
        <article className="lab-reduction-result-card">
          <h4>Nhóm này đang dùng biểu diễn gốc</h4>
          <p>{relevantRows.length
            ? [...new Set(relevantRows.map((row) => algorithmName(baseAlgorithm(row), group, String(row.name))))].join(" · ")
            : "Không có dữ liệu giảm chiều trong báo cáo này."}</p>
        </article>
      </div>
    );
  }

  return (
    <>
      <div className="lab-reduction-results">
        {variants.map((row) => {
          const meta = (row.reduction || {}) as ResultRecord;
          const method = String(row.reduction_method || meta.method || "");
          const input = numericValue(meta.input_dimensions);
          const output = numericValue(meta.output_dimensions) ?? numericValue(row.components);
          const percent = input !== null && output !== null && input > 0
            ? 100 * (input - output) / input
            : null;
          const variance = numericValue(meta.explained_variance_ratio_train);
          const fitSeconds = numericValue(meta.fit_seconds) ?? numericValue(row.reduction_fit_seconds);
          const transformSeconds = numericValue(meta.transform_seconds);
          const owner = baseAlgorithm(row);
          const selected = selectedId === variantId(row);
          const competing = raw.has(owner) || variants.some(
            (item) => baseAlgorithm(item) === owner && variantId(item) !== variantId(row),
          );
          const status = selected || !competing ? "Đang dùng" : "Chỉ đối chứng";
          const featureLabel = String(row.feature_family || meta.feature_family || "hồ sơ sở thích");
          return (
            <article className="lab-reduction-result-card" key={variantId(row)}>
              <header>
                <h4>{row.name ? String(row.name) : algorithmName(owner, group)}</h4>
                <span className={`lab-reduction-status${selected ? " active" : ""}`}>{status}</span>
              </header>
              <p>{method === "pca" ? "PCA" : "TruncatedSVD"} · {featureLabel}</p>
              <p>{input !== null && output !== null
                ? `${input.toLocaleString("vi-VN")} → ${output.toLocaleString("vi-VN")} chiều · giảm ${percent?.toFixed(1)}%`
                : `Không có dữ liệu chiều đầu vào → đầu ra · preset ${output ?? "Không có dữ liệu"} chiều`}</p>
              <p>Phương sai giữ lại: {variance === null ? "Không có dữ liệu" : `${(variance * 100).toFixed(2)}%`}</p>
              <p>Fit + transform reducer: {fitSeconds === null && transformSeconds === null
                ? "Không có dữ liệu"
                : `${metricNumber((fitSeconds ?? 0) + (transformSeconds ?? 0))} giây`}</p>
              <p>{competing
                ? "Đối chiếu cùng thuật toán; kết quả được chọn bằng Validation."
                : "Chưa có đối chứng cùng thuật toán để kết luận tăng tốc hoặc tăng chất lượng."}</p>
            </article>
          );
        })}
      </div>
      {raw.size > 0 && <p className="lab-result-selection">
        Giữ biểu diễn gốc: {[...raw].map((id) => algorithmName(id, group)).join(" · ")}.
      </p>}
    </>
  );
}

function PerformanceTable({
  result,
  group,
  selectedId,
  language,
}: {
  result: LabResult;
  group: LabGroup;
  selectedId: string;
  language: string;
}) {
  const rows = result.rows ?? [];
  const ids = [...new Set(rows.map(variantId))];
  if (!ids.length) return <p className="lab-muted">Không có dữ liệu kết quả.</p>;
  const find = (id: string, split: string) => rows.find((row) => variantId(row) === id && row.split === split);
  const recommendation = result.recommendation?.rows ?? [];
  const rec = (id: string, split: string) => recommendation.find((row) => row.variant_id === id && row.split === split);
  const selected = (id: string) => id === selectedId;
  const resultMetricKeys = group.id === "rating" ? ["rmse", "mae"]
    : group.id === "classification" ? ["f1", "precision", "recall"] : [];

  if (resultMetricKeys.length) {
    const label = (key: string) => ({ rmse: "RMSE", mae: "MAE", f1: "F1", precision: "Precision", recall: "Recall" }[key] || key);
    return (
      <div className="lab-table-scroll">
        <table className="lab-result-table"><thead>
          <tr><th rowSpan={2}>Thuật toán / biến thể</th><th rowSpan={2}>Biểu diễn</th>
            {resultMetricKeys.map((key) => <th key={key} colSpan={2}>{label(key)}</th>)}
            <th rowSpan={2}>Fit / xử lý (s)</th><th rowSpan={2}>Model (KiB)</th></tr>
          <tr>{resultMetricKeys.flatMap((key) => ["Validation", "Test"].map((split) => <th key={`${key}/${split}`}>{split}</th>))}</tr>
        </thead><tbody>
          {ids.map((id) => {
            const validation = find(id, "validation");
            const test = find(id, "test");
            const sample = validation || test || rows.find((row) => variantId(row) === id)!;
            return <tr key={id} className={selected(id) ? "is-selected" : ""}>
              <th>{algorithmName(baseAlgorithm(sample), group, String(sample.name))}
                {selected(id) && <span className="selected-label">Validation chọn</span>}</th>
              <td>{representation(sample)}</td>
              {resultMetricKeys.flatMap((key) => [validation, test].map((row, index) =>
                <td key={`${key}/${index}`}>{shownValue(row?.[key], language)}</td>))}
              <td>{shownValue(test?.fit_seconds, language)}</td>
              <td>{test ? shownValue(numericValue(test.model_bytes) === null ? null : Number(test.model_bytes) / 1024, language) : "Không có dữ liệu"}</td>
            </tr>;
          })}
        </tbody></table>
      </div>
    );
  }

  if (group.id === "clustering") {
    const models = rows.filter((row) => row.split === "train_history");
    return <div className="lab-table-scroll"><table className="lab-result-table"><thead><tr>
      <th>Thuật toán / biểu diễn</th><th>Silhouette</th><th>Số cụm</th><th>Coverage</th><th>Noise</th>
      <th>Validation NDCG@10</th><th>Test NDCG@10</th><th>Fit model (s)</th><th>Model (KiB)</th>
    </tr></thead><tbody>{models.map((row) => {
      const id = variantId(row); const val = rec(id, "validation"); const test = rec(id, "test");
      return <tr key={id} className={selected(id) ? "is-selected" : ""}>
        <th>{algorithmName(baseAlgorithm(row), group, String(row.name))}{selected(id) && <span className="selected-label">Validation chọn</span>}</th>
        <td>{shownValue(row.silhouette, language)}</td><td>{shownValue(row.clusters, language)}</td>
        <td>{shownValue(row.coverage, language)}</td><td>{numericValue(row.noise_fraction) === null ? "Không có dữ liệu" : `${(Number(row.noise_fraction) * 100).toFixed(1)}%`}</td>
        <td>{shownValue(val?.ndcg_at_10, language)}</td><td>{shownValue(test?.ndcg_at_10, language)}</td>
        <td>{shownValue(row.fit_seconds, language)}</td><td>{numericValue(row.model_bytes) === null ? "Không có dữ liệu" : metricNumber(Number(row.model_bytes) / 1024, language)}</td>
      </tr>;
    })}</tbody></table></div>;
  }

  if (group.id === "association") {
    return <div className="lab-table-scroll"><table className="lab-result-table"><thead><tr>
      <th>Thuật toán / biểu diễn</th><th>Số luật</th><th>Validation NDCG@10</th><th>Test NDCG@10</th><th>Coverage</th><th>Huấn luyện (s)</th><th>Model (KiB)</th>
    </tr></thead><tbody>{rows.map((row) => {
      const id = variantId(row); const val = rec(id, "validation"); const test = rec(id, "test");
      return <tr key={id} className={selected(id) ? "is-selected" : ""}>
        <th>{String(row.name || id)}{selected(id) && <span className="selected-label">Validation chọn</span>}</th>
        <td>{shownValue(row.rule_count, language)}</td><td>{shownValue(val?.ndcg_at_10, language)}</td>
        <td>{shownValue(test?.ndcg_at_10, language)}</td>
        <td>{shownValue(val?.recommendation_coverage ?? val?.group_coverage, language)}</td>
        <td>{shownValue(row.model_training_seconds, language)}</td>
        <td>{numericValue(row.model_bytes) === null ? "Không có dữ liệu" : metricNumber(Number(row.model_bytes) / 1024, language)}</td>
      </tr>;
    })}</tbody></table></div>;
  }

  if (group.id === "reinforcement") {
    const metric = "average_reward";
    return <div className="lab-table-scroll"><table className="lab-result-table"><thead><tr>
      <th>Chính sách</th><th>Biểu diễn</th><th>Reward Validation</th><th>Reward Test</th><th>Coverage Test</th><th>Training (s)</th><th>Model (KiB)</th>
    </tr></thead><tbody>{ids.map((id) => {
      const validation = find(id, "validation"); const test = find(id, "test"); const sample = validation || test!;
      return <tr key={id} className={selected(id) ? "is-selected" : ""}>
        <th>{String(sample.name || id)}{selected(id) && <span className="selected-label">Validation chọn</span>}</th>
        <td>{representation(sample)}</td><td>{shownValue(validation?.[metric], language)}</td><td>{shownValue(test?.[metric], language)}</td>
        <td>{shownValue(test?.user_coverage ?? test?.observed_feedback_fraction, language)}</td><td>{shownValue(test?.fit_seconds, language)}</td>
        <td>{numericValue(test?.model_bytes) === null ? "Không có dữ liệu" : metricNumber(Number(test?.model_bytes) / 1024, language)}</td>
      </tr>;
    })}</tbody></table></div>;
  }

  return <div className="lab-table-scroll"><table className="lab-result-table"><thead><tr>
    <th>Thuật toán</th><th>Tập</th>{group.metrics.slice(0, 3).map((metric) => <th key={metric.key}>{metric.name}</th>)}<th>Fit (s)</th>
  </tr></thead><tbody>{rows.map((row, index) => <tr key={`${variantId(row)}/${row.split}/${index}`}>
    <th>{String(row.name || variantId(row))}</th><td>{row.split}</td>
    {group.metrics.slice(0, 3).map((metric) => <td key={metric.key}>{shownValue(row[metric.key], language)}</td>)}
    <td>{shownValue(row.fit_seconds, language)}</td>
  </tr>)}</tbody></table></div>;
}

function CrossValidation({ result, group, language }: { result: LabResult; group: LabGroup; language: string }) {
  if (!["rating", "classification"].includes(group.id)) return null;
  const entries = Object.entries(result.cv ?? {});
  const metricKeys = group.id === "rating" ? ["rmse", "mae"] : ["f1", "precision", "recall"];
  const metricNames: Record<string, string> = { rmse: "RMSE", mae: "MAE", f1: "F1", precision: "Precision", recall: "Recall" };
  const chartValues: { label: string; value: number | null }[] = [];
  const temporal = result.cv_protocol === "per_user_expanding_time_v1";
  const foldCount = result.cv_folds ?? Math.max(0, ...entries.map(([, data]) => data.folds?.length ?? 0));
  const foldIndices = Array.from({ length: foldCount }, (_, index) => index);
  const cvTitle = foldCount ? `${foldCount}-fold` : "Cross Validation";
  return (
    <section className="lab-result-row" aria-labelledby="lab-cv-title">
      <h3 id="lab-cv-title">{temporal ? `${cvTitle} theo thời gian trên Train` : foldCount ? `${cvTitle} Cross Validation` : cvTitle}</h3>
      <p>{temporal ? `${cvTitle} theo thời gian trên Train.` : "Không có metadata CV; giao thức lịch sử cần được kiểm tra trong chi tiết."}</p>
      {entries.length ? <div className="lab-table-scroll"><table className="lab-result-table"><thead><tr>
        <th>Thuật toán / biến thể</th>{metricKeys.flatMap((key) => [...foldIndices.map((index) => `Fold ${index + 1}`), "Trung bình ± độ lệch chuẩn"].map((label) => <th key={`${key}/${label}`}>{metricNames[key]} · {label}</th>))}
      </tr></thead><tbody>{entries.map(([id, data]) => {
        const folds = (data.folds ?? []) as ResultRecord[];
        return <tr key={id}><th>{id}</th>{metricKeys.flatMap((key) => {
          const values = foldIndices.map((index) => numericValue(folds[index]?.[key]));
          const summary = data.metrics[key];
          if (key === (group.id === "rating" ? "rmse" : "f1")) {
            values.forEach((value, index) => chartValues.push({ label: `${id} · Fold ${index + 1}`, value }));
          }
          return [
            ...values.map((value, index) => <td key={`${key}/${index}`}>{shownValue(value, language)}</td>),
            <td key={`${key}/mean`}>{summary
              ? `${metricNumber(summary.mean, language)} ± ${metricNumber(summary.std, language)}`
              : "Không có dữ liệu"}</td>,
          ];
        })}</tr>;
      })}</tbody></table></div> : <p className="lab-muted">Không có dữ liệu CV.</p>}
      {chartValues.filter((item) => item.value !== null).length > 1 && (
        <div className="lab-cv-chart"><MiniBars
          title={group.id === "rating" ? "RMSE theo fold" : "F1 theo fold"}
          unit={group.id === "rating" ? "RMSE · thấp hơn tốt hơn" : "F1 · cao hơn tốt hơn"}
          items={chartValues.filter((item) => item.value !== null)}
          lowerIsBetter={group.id === "rating"}
        /></div>
      )}
    </section>
  );
}

export function ExperimentResults({ job, group }: { job: LabJob; group: LabGroup }) {
  const { t, language } = usePreferences();
  const result = job.result!;
  const selectedId = String(result.selected_variant_id || result.reduction?.selected_variant_id || "");
  const recommendations = result.recommendation?.rows ?? [];
  let qualityItems: { label: string; value: number | null }[] = [];
  let qualityTitle = "Chất lượng trên Validation";
  let qualityUnit = "Validation";
  let qualityLower = false;
  if (group.id === "rating" || group.id === "classification") {
    const key = group.id === "rating" ? "rmse" : "f1";
    qualityItems = result.rows.filter((row) => row.split === "validation").map((row) => ({ label: variantId(row), value: numericValue(row[key]) }));
    qualityTitle = group.id === "rating" ? "RMSE Validation" : "F1 Validation";
    qualityUnit = group.id === "rating" ? "RMSE" : "F1";
    qualityLower = group.id === "rating";
  } else if (group.id === "clustering" || group.id === "association") {
    qualityItems = recommendations.filter((row) => row.split === "validation").map((row) => ({ label: row.variant_id, value: numericValue(row.ndcg_at_10) }));
    qualityTitle = "NDCG@10 Validation";
    qualityUnit = "NDCG@10";
  } else if (group.id === "reinforcement") {
    qualityItems = result.rows.filter((row) => row.split === "validation").map((row) => ({ label: variantId(row), value: numericValue(row.average_reward) }));
    qualityTitle = "Reward Validation";
    qualityUnit = "Reward trung bình";
  }
  const timeSplit = group.id === "clustering" ? "train_history"
    : group.id === "association" ? "train"
      : group.id === "pca" ? "validation" : "test";
  const timeRows = result.rows.filter((row) => row.split === timeSplit);
  const fitItems = timeRows.map((row) => ({ label: variantId(row), value: numericValue(row.fit_seconds) }));
  const sizeItems = timeRows.map((row) => ({ label: variantId(row), value: numericValue(row.model_bytes) === null ? null : Number(row.model_bytes) / 1024 }));
  const splitSummary = Object.entries(result.provenance?.splits ?? {})
    .map(([split, data]) => `${split}: ${shownValue(data.ratings, language)} ratings`).join(" · ");
  const detailedTimings = result.pipeline_timings;
  const hasTimings = detailedTimings && Object.keys(detailedTimings).length > 0;

  return (
    <section id="lab-results" className="lab-section lab-results" data-lab-group={group.id} aria-label={t("Kết quả thực nghiệm")}>
      <div className="lab-section-heading">
        <div><span className="lab-kicker">EXPERIMENT REPORT</span><h2>{t("Kết quả thực nghiệm")}</h2></div>
        <div className="lab-downloads">{["csv", "xlsx", "json"].map((format) => <a className="lab-outline" key={format} href={apiResourceUrl(`/experiment-lab/runs/${job.id}/export/${format}`)}><Download size={14}/>{format.toUpperCase()}</a>)}</div>
      </div>
      <p className="lab-muted">
        {shownValue(result.provenance?.sampled_users, language)} users · {shownValue(result.provenance?.sampled_ratings, language)} ratings · Seed {shownValue(result.config?.seed, language)}
        {splitSummary ? ` · ${splitSummary}` : ""}
      </p>

      <section className="lab-result-row" aria-labelledby="lab-reduction-results-title">
        <h3 id="lab-reduction-results-title">Kết quả giảm chiều và kết luận</h3>
        <ReductionSummary result={result} group={group}/>
        {result.selection && <p className="lab-result-selection">{result.selection}</p>}
      </section>

      <section className="lab-result-row" aria-labelledby="lab-training-results-title">
        <h3 id="lab-training-results-title">Kết quả huấn luyện</h3>
        <p>Validation dùng chọn mô hình; Test chỉ báo cáo cuối. Thời gian fit không phải tổng thời gian toàn pipeline.</p>
        <PerformanceTable result={result} group={group} selectedId={selectedId} language={language}/>
      </section>

      <section className="lab-result-row" aria-labelledby="lab-comparison-charts-title">
        <h3 id="lab-comparison-charts-title">Biểu đồ so sánh</h3>
        <div className="lab-result-charts">
          <MiniBars title={qualityTitle} unit={qualityUnit} items={qualityItems} lowerIsBetter={qualityLower}/>
          <MiniBars title="Thời gian fit" unit={group.id === "association" ? "Khai phá luật · giây" : "Giây"} items={fitItems} lowerIsBetter/>
          {sizeItems.some((item) => item.value !== null) && <MiniBars title="Kích thước mô hình" unit="KiB" items={sizeItems} lowerIsBetter/>}
          {qualityItems.filter((item) => item.value !== null).length < 2 && fitItems.filter((item) => item.value !== null).length < 2 && <p className="lab-muted">Cần ít nhất hai kết quả có số đo để vẽ biểu đồ so sánh.</p>}
        </div>
      </section>

      <CrossValidation result={result} group={group} language={language}/>

      <details className="lab-detail-panel lab-result-details">
        <summary>Chi tiết protocol và chẩn đoán</summary>
        <ExperimentPipeline group={group} result={result}/>
        {group.id === "clustering" && result.training && <div className="lab-insight lab-cluster-training"><strong>Dữ liệu huấn luyện phân cụm</strong><p>Train: {result.training.ratings.toLocaleString("vi-VN")} ratings · {result.training.eligible_users.toLocaleString("vi-VN")} user đủ lịch sử.</p></div>}
        {group.id === "association" && <AssociationTrainingSummary result={result}/>}
        {group.id === "association" && result.recommendation && <AprioriRanking recommendation={result.recommendation}/>}
        {result.rules && <Rules rules={result.rules}/>}
        <GroupedAprioriDetails result={result}/>
        {result.clusters && <section className="lab-section"><h3>Hồ sơ genre theo cụm</h3><div className="lab-table-scroll"><table><thead><tr><th>Thuật toán</th><th>Cụm</th><th>Users Train</th><th>Genres nổi bật</th></tr></thead><tbody>{result.clusters.flatMap((cluster) => cluster.profiles.map((profile) => <tr key={`${cluster.variant_id || cluster.algorithm}/${profile.cluster}`}><th>{cluster.variant_id || cluster.algorithm}</th><td>{profile.cluster < 0 ? "Noise" : profile.cluster}</td><td>{profile.users}</td><td>{Object.entries(profile.genre_means).map(([genre, score]) => `${genre}: ${score.toFixed(2)}`).join(" · ")}</td></tr>))}</tbody></table></div></section>}
        {result.search && <details className="lab-detail-panel"><summary>Kết quả tìm kiếm cấu hình</summary><div className="lab-table-scroll"><table><thead><tr><th>Thuật toán</th><th>Cấu hình</th><th>Train</th><th>Validation giữ lại</th><th>Bao phủ</th></tr></thead><tbody>{result.search.map((row, index) => <tr key={index}><th>{String(row.algorithm || "Không có dữ liệu")}</th><td>{JSON.stringify(row.config || "Không có dữ liệu")}</td><td>{shownValue(row.silhouette, language)}</td><td>{shownValue(row.validation_silhouette, language)}</td><td>{shownValue(row.validation_coverage, language)}</td></tr>)}</tbody></table></div></details>}
        {result.class_balance && <section className="lab-section"><h3>Phân bố nhãn</h3>{Object.entries(result.class_balance).map(([split, balance]) => <p key={split}>{split}: Không thích {balance.negative.toLocaleString("vi-VN")} · Thích {balance.positive.toLocaleString("vi-VN")} · {shownValue(balance.positive_fraction * 100, language)}%</p>)}</section>}
        {hasTimings && <section className="lab-section"><h3>Thời gian pipeline (giây)</h3><div className="lab-preset-common">{Object.entries(detailedTimings!).map(([key, value]) => <div key={key}><dt>{({feature_creation_seconds:"Tạo đặc trưng",reducer_fit_seconds:"Fit reducer",transform_seconds:"Transform",model_training_seconds:"Huấn luyện mô hình",evaluation_seconds:"Đánh giá",total_pipeline_seconds:"Tổng pipeline"} as Record<string,string>)[key] || key}</dt><dd>{shownValue(value, language)}</dd></div>)}</div></section>}
        <details className="lab-detail-panel"><summary>Các biểu đồ chẩn đoán</summary><div className="lab-charts">{result.charts.map((chart) => <LabChartFigure key={chart.index} chart={chart}/>)}</div></details>
        <details className="lab-detail-panel"><summary>Cấu hình và metadata</summary><pre>{JSON.stringify({config: result.config, best_configs: result.best_configs, provenance: result.provenance, runtime: result.runtime, protocol: result.rl_protocol, notes: result.notes}, null, 2)}</pre></details>
      </details>
    </section>
  );
}
