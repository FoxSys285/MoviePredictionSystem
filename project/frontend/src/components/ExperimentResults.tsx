import { useEffect, useId, useState } from "react";
import { BarChart3 } from "lucide-react";
import { useSearchParams } from "react-router-dom";
import { api, apiResourceUrl } from "../lib/api";
import { usePreferences } from "../lib/preferences";
import { ExperimentCharts, type ExperimentChartRow } from "./ExperimentCharts";

type Metrics = { rmse: number; mae: number };
type Scope = {
  mode: "sampled_users" | "all_users";
  included_users: number;
  included_ratings: number;
  source_users: number;
  source_ratings: number;
  split_report: { ratings: Record<string, number> };
};
type Job = {
  id: string;
  status: "queued" | "running" | "completed" | "failed";
  progress: number;
  stage: string;
  message: string;
  current_variant?: string;
  created_at: string;
  scope?: Scope;
};
type VariantResult = {
  validation: Record<string, Metrics>;
  test: Record<string, Metrics>;
  best_parameters: Record<string, Record<string, unknown>>;
  best_iterations: Record<string, number>;
};
type Run = { job: Job; result: { scope: Scope; variants: Record<string, VariantResult> } | null };

const names: Record<string, string> = {
  linear: "Linear Regression", xgboost: "XGBoost", svd: "SVD", ncf: "NCF",
  global_mean: "Global Mean",
};
const phases: Record<string, string> = {
  data: "Chuẩn bị dữ liệu", search: "GridSearchCV trên Train",
  validation: "Đánh giá Validation", finished: "Đã đánh giá Validation",
  refit: "Refit Train + Validation", test: "Đánh giá Test", completed: "Đã hoàn tất",
};

export function ExperimentResults() {
  const { t, language } = usePreferences();
  const [params] = useSearchParams();
  const [jobs, setJobs] = useState<Job[] | null>(null);
  const [selected, setSelected] = useState(params.get("run") || "");
  const [run, setRun] = useState<Run | null>(null);
  const [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0);
  const [showCharts, setShowCharts] = useState(false);
  const chartsId = useId();
  const locale = language === "vi" ? "vi-VN" : "en-US";
  const number = (value: number) => value.toLocaleString(locale);

  useEffect(() => {
    const controller = new AbortController();
    setError("");
    api<Job[]>("/experiments/runs", { signal: controller.signal }).then((data) => {
      if (controller.signal.aborted) return;
      setJobs(data);
      setSelected((previous) => data.some((job) => job.id === previous) ? previous : data[0]?.id || "");
    }).catch((cause: Error) => { if (!controller.signal.aborted) setError(cause.message); });
    return () => controller.abort();
  }, [refresh]);

  useEffect(() => {
    if (!selected) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    setRun(null);
    async function poll() {
      try {
        const data = await api<Run>(`/experiments/runs/${selected}`, { signal: controller.signal });
        if (controller.signal.aborted) return;
        setRun(data);
        setError("");
        if (["running", "queued"].includes(data.job.status)) timer = setTimeout(poll, 3000);
      } catch (cause) {
        if (!controller.signal.aborted) setError((cause as Error).message);
      }
    }
    void poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [selected, refresh]);

  const scope = run?.result?.scope || run?.job.scope;
  const variants = Object.entries(run?.result?.variants || {});
  const rows: ExperimentChartRow[] = [];
  if (variants.length) {
    const reference = variants[0][1];
    rows.push({ key: "global_mean", label: "Global Mean", validation: reference.validation.global_mean,
      test: reference.test.global_mean });
    for (const [variant, result] of variants) {
      for (const algorithm of ["linear", "xgboost", "svd", "ncf"]) {
        if (!result.validation[algorithm] || !result.test[algorithm]) continue;
        const name = variant === "improved" && algorithm === "linear"
          ? "Linear Regression (Ridge)" : names[algorithm];
        rows.push({ key: `${variant}-${algorithm}`,
          label: `${t(variant === "baseline" ? "Mô hình cơ sở" : "Mô hình cải tiến")} · ${name}`,
          validation: result.validation[algorithm], test: result.test[algorithm] });
      }
    }
  }
  const stage = run?.job.stage.split(":") || [];
  const metric = (value: number) => value.toLocaleString(locale, {
    minimumFractionDigits: 4, maximumFractionDigits: 4,
  });

  return <div className="experiment-results">
    <div className="experiment-results-toolbar">
      <label>{t("Lượt thực nghiệm")}
        <select value={selected} onChange={(event) => setSelected(event.target.value)} disabled={!jobs?.length}>
          {!jobs?.length && <option value="">{t("Chưa có lượt thực nghiệm")}</option>}
          {jobs?.map((job) => <option key={job.id} value={job.id}>{job.id}</option>)}
        </select>
      </label>
      <button className="button button-outline" onClick={() => setRefresh((value) => value + 1)}>
        {t("Tải lại kết quả")}
      </button>
    </div>
    {error && <p role="alert" className="experiment-results-error">{t(error)}</p>}
    {!error && jobs === null && <p role="status">{t("Đang tải kết quả thực nghiệm...")}</p>}
    {!error && jobs?.length === 0 && <p>{t("Chưa có lượt thực nghiệm được lưu.")}</p>}
    {run && <>
      <p className="experiment-results-status" role="status" aria-live="polite">
        {t(run.job.status === "completed" ? "Đã hoàn tất" : run.job.status === "failed" ? "Thực nghiệm gặp lỗi" : "Đang chạy thực nghiệm")}
        {run.job.status === "running" && ` · ${t(phases[stage[0]] || stage[0])}${stage[1] ? ` · ${names[stage[1]]}` : ""} · ${run.job.progress}%`}
      </p>
      {["running", "queued"].includes(run.job.status) && <progress value={run.job.progress} max={100} aria-label={t("Tiến độ thực nghiệm")} />}
      {run.job.status === "failed" && <p role="alert" className="experiment-results-error">{run.job.message}</p>}
      {scope && <section className="experiment-result-scope" aria-label={t("Phạm vi dữ liệu thực nghiệm")}>
        <p><strong>{t(scope.mode === "sampled_users" ? "Kết quả trên mẫu người dùng" : "Toàn bộ dữ liệu")}</strong></p>
        <p>{number(scope.included_users)} {t("Người dùng")} · {number(scope.included_ratings)} {t("Lượt đánh giá")}</p>
        <p>{t("Nguồn dữ liệu")}: {number(scope.source_users)} {t("Người dùng")} · {number(scope.source_ratings)} {t("Lượt đánh giá")}</p>
        <dl className="experiment-split-summary">
          {Object.entries(scope.split_report.ratings).map(([split, count]) => <div key={split}>
            <dt>{split === "train" ? "Train" : split === "validation" ? "Validation" : "Test"}</dt>
            <dd>{number(count)}</dd>
          </div>)}
        </dl>
      </section>}
      {rows.length > 0 && <>
        <div className="experiment-result-table-toolbar">
          <button type="button" className="button button-outline" aria-expanded={showCharts} aria-controls={chartsId}
            onClick={() => setShowCharts((previous) => !previous)}>
            <BarChart3 size={17} aria-hidden="true" />{t("Biểu đồ")}
          </button>
        </div>
        <div className="experiment-result-table-wrap" tabIndex={0} aria-label={t("Bảng kết quả thực nghiệm")}>
          <table className="experiment-result-table">
            <caption>{t("RMSE và MAE càng thấp càng tốt. Test được đánh giá sau khi chốt mô hình.")}</caption>
            <thead><tr><th>{t("Mô hình")}</th><th>Validation RMSE</th><th>Validation MAE</th><th>Test RMSE</th><th>Test MAE</th></tr></thead>
            <tbody>{rows.map((row) => <tr key={row.key}><th scope="row">{row.label}</th>
              <td>{metric(row.validation.rmse)}</td><td>{metric(row.validation.mae)}</td>
              <td>{metric(row.test.rmse)}</td><td>{metric(row.test.mae)}</td>
            </tr>)}</tbody>
          </table>
        </div>
        <div id={chartsId} hidden={!showCharts}>
          {showCharts && <ExperimentCharts rows={rows} />}
        </div>
        <a className="button button-outline" href={apiResourceUrl(`/experiments/runs/${selected}/result`)}>{t("Tải kết quả JSON")}</a>
        <details className="experiment-result-parameters"><summary>{t("Siêu tham số đã chọn")}</summary>
          {variants.map(([variant, result]) => <section key={variant}>
            <h3>{t(variant === "baseline" ? "Mô hình cơ sở" : "Mô hình cải tiến")}</h3>
            <pre>{JSON.stringify({ parameters: result.best_parameters, best_iterations: result.best_iterations }, null, 2)}</pre>
          </section>)}
        </details>
      </>}
    </>}
  </div>;
}
