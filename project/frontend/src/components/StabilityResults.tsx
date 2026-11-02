import { useState } from "react";
import { Download } from "lucide-react";
import { SectionTitle } from "./ui";
import { LineChart, experimentColor } from "./ExperimentCharts";
import { apiResourceUrl } from "../lib/api";
import { usePreferences } from "../lib/preferences";

type Metric = "rmse" | "mae" | "ndcg_at_10";
type Stats = { mean: number | null; std: number | null; n: number };
type Summary = { algorithm: string; metrics: Record<Metric | "fit_seconds", Stats> };
type Scores = { rmse: number; mae: number; ndcg_at_10: number | null; ranking_users: number; known_pair_fraction: number };
type SeedRun = { algorithm: string; model_seed: number; fit_seconds: number; validation: Scores; test: Scores };
type FoldRun = { algorithm: string; fold: number; fit_seconds: number; train: { ratings: number; max_timestamp: number }; heldout: { ratings: number; min_timestamp: number; max_timestamp: number }; evaluation: Scores };
type Interval = { algorithm_a: string; algorithm_b: string; metric: Metric; users: number; delta: number | null; ci_low: number | null; ci_high: number | null; conclusion: "a_better" | "b_better" | "uncertain" | "insufficient_users" };
export type StabilityReport = {
  run_id: string; status: "running" | "complete";
  model_seeds: number[]; temporal_folds: number; bootstrap_repetitions: number;
  reference_rating: string; reference_ranking: string;
  seed_runs: SeedRun[]; temporal_runs: FoldRun[];
  seed_summary: Record<"validation" | "test", Summary[]>; temporal_summary: Summary[];
  bootstrap: Partial<Record<"validation" | "test", Interval[]>>;
};
const valid = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);

export function StabilityResults({ report, split, names }: {
  report?: StabilityReport | null; split: "validation" | "test"; names: Record<string, string>;
}) {
  const { t, language } = usePreferences();
  const [metric, setMetric] = useState<Metric>("rmse");
  if (!report) return null;
  const number = (value: unknown, digits = 5) => valid(value) ? value.toLocaleString(language === "en" ? "en-US" : "vi-VN", { minimumFractionDigits: digits, maximumFractionDigits: digits }) : "—";
  const name = (value: string) => names[value] || value;
  const label = metric === "ndcg_at_10" ? "NDCG@10" : metric.toUpperCase();
  const summary = report.seed_summary[split];
  const algorithms = [...new Set(report.seed_runs.map((row) => row.algorithm))].sort();
  const candidates = summary.filter((row) => valid(row.metrics[metric].mean));
  const best = candidates.length ? (metric === "ndcg_at_10" ? Math.max : Math.min)(...candidates.map((row) => row.metrics[metric].mean!)) : null;
  const leaders = candidates.filter((row) => row.metrics[metric].mean === best).map((row) => name(row.algorithm)).join(", ");
  const temporalCandidates = report.temporal_summary.filter((row) => valid(row.metrics[metric].mean));
  const temporalBest = temporalCandidates.length ? (metric === "ndcg_at_10" ? Math.max : Math.min)(...temporalCandidates.map((row) => row.metrics[metric].mean!)) : null;
  const temporalLeaders = temporalCandidates.filter((row) => row.metrics[metric].mean === temporalBest).map((row) => name(row.algorithm)).join(", ");
  const intervals = (report.bootstrap[split] || []).filter((row) => row.metric === metric);
  const finiteIntervals = intervals.filter((row) => valid(row.ci_low) && valid(row.ci_high) && valid(row.delta));
  const min = Math.min(0, ...finiteIntervals.map((row) => row.ci_low!));
  const max = Math.max(0, ...finiteIntervals.map((row) => row.ci_high!));
  const pad = Math.max(.0001, max - min) * .15;
  const x = (value: number) => 110 + (value - (min - pad)) / (max - min + 2 * pad) * 410;
  const conclusion = (value: Interval) => value.conclusion === "a_better" ? t("Ủng hộ A") : value.conclusion === "b_better" ? t("Ủng hộ B") : value.conclusion === "uncertain" ? t("Chưa đủ bằng chứng") : t("Chưa đủ người dùng");
  const table = (rows: Summary[], category: string) => <div className="experiment-table-wrap"><table className="experiment-table stability-summary-table">
    <caption>{category} · {label} · mean ± sample std</caption>
    <thead><tr><th>{t("Mô hình")}</th><th>{label} {metric === "ndcg_at_10" ? "↑" : "↓"}</th><th>{t("Số lần đo")}</th><th>{t("Train trung bình (s)")}</th></tr></thead>
    <tbody>{rows.map((row) => <tr key={row.algorithm}><th scope="row">{name(row.algorithm)}</th><td>{number(row.metrics[metric].mean)} ± {number(row.metrics[metric].std)}</td><td>{row.metrics[metric].n}</td><td>{number(row.metrics.fit_seconds.mean, 2)}</td></tr>)}</tbody>
  </table></div>;
  return <section className="experiment-stability-results" aria-label={t("Kết quả đánh giá độ ổn định")}>
    <div className="experiment-results-heading"><SectionTitle>{t("Kết quả đánh giá độ ổn định")}</SectionTitle><span className={`experiment-plan-badge ${report.status === "complete" ? "is-complete" : ""}`}>{t(report.status === "complete" ? "Đã hoàn thành" : "Đang chạy · tải lại để cập nhật")}</span></div>
    <p className="subtle-note">{t("Đã đo {seeds} lượt seed và {folds} lượt theo thời gian. Seed và CI theo tập {split}; temporal CV chỉ dùng phần train ban đầu.", { seeds: report.seed_runs.length, folds: report.temporal_runs.length, split })}</p>
    <div className="experiment-split stability-metric-selector" role="group" aria-label={t("Chỉ số đánh giá độ ổn định")}>
      {(["rmse", "mae", "ndcg_at_10"] as const).map((value) => <button key={value} className={metric === value ? "selected" : ""} aria-pressed={metric === value} onClick={() => setMetric(value)}>{value === "ndcg_at_10" ? "NDCG@10" : value.toUpperCase()}</button>)}
    </div>
    <div className="experiment-visual-grid">
      <article className="experiment-chart experiment-figure">
        <span className="experiment-figure-label">A · {split} · {report.model_seeds.length} SEED</span><h3>{t("Độ biến động qua seed")}</h3>
        <LineChart title={t("Độ biến động qua seed")} ticks={report.model_seeds} xLabel="Model seed" yLabel={label} series={algorithms.map((algorithm) => ({
          label: name(algorithm), color: experimentColor(algorithm), values: report.seed_runs.filter((row) => row.algorithm === algorithm && valid(row[split][metric])).map((row) => ({ x: row.model_seed, y: row[split][metric]! })),
        }))} />
        {table(summary, t("Lặp seed"))}
        <p className="experiment-insight"><b>{t("Nhận xét:")}</b> {leaders ? t("Trung bình {metric} tốt nhất: {models} ({value}).", { metric: label, models: leaders, value: number(best) }) : t("Chưa đủ số liệu.")} {t("Đường gần trùng nhau có thể che khuất nhau; đọc mean ± std và bảng từng run. Std bằng 0 nghĩa là không thấy biến động trong các seed đã thử.")}</p>
        <p className="experiment-axis-note">{t("Cả user và phim đã có trong train: {warm}% rating của tập {split}. Sai số tổng và std nhỏ có thể chịu ảnh hưởng lớn từ các trường hợp mới dùng prior.", { warm: number((report.seed_runs[0]?.[split].known_pair_fraction || 0) * 100, 2), split })}</p>
        <details className="experiment-chart-data"><summary>{t("Xem từng lần huấn luyện")}</summary><div className="experiment-data-scroll"><table><thead><tr><th>{t("Mô hình")}</th><th>Seed</th><th>{label}</th></tr></thead><tbody>{report.seed_runs.map((row) => <tr key={`${row.algorithm}-${row.model_seed}`}><td>{name(row.algorithm)}</td><td>{row.model_seed}</td><td>{number(row[split][metric])}</td></tr>)}</tbody></table></div></details>
      </article>
      <article className="experiment-chart experiment-figure">
        <span className="experiment-figure-label">B · TEMPORAL CV · {report.temporal_folds} FOLD</span><h3>{t("Độ biến động theo thời gian")}</h3>
        <LineChart title={t("Độ biến động theo thời gian")} ticks={[1, 2, 3, 4, 5]} xLabel="Fold" yLabel={label} series={algorithms.map((algorithm) => ({ label: name(algorithm), color: experimentColor(algorithm), values: report.temporal_runs.filter((row) => row.algorithm === algorithm && valid(row.evaluation[metric])).map((row) => ({ x: row.fold, y: row.evaluation[metric]! })) }))} />
        {table(report.temporal_summary, t("Giai đoạn thời gian"))}
        <p className="experiment-insight"><b>{t("Nhận xét:")}</b> {temporalLeaders && t("Trung bình {metric} tốt nhất: {models} ({value}).", { metric: label, models: temporalLeaders, value: number(temporalBest) })} {t("Biến động qua fold gồm thay đổi dữ liệu, người dùng mới và lượng train tăng dần; không quy toàn bộ cho ngẫu nhiên của thuật toán. Std qua fold được giữ riêng với std qua seed.")}</p>
        <details className="experiment-chart-data"><summary>{t("Xem dữ liệu từng fold")}</summary><div className="experiment-data-scroll"><table><thead><tr><th>Fold</th><th>{t("Mô hình")}</th><th>Train / heldout</th><th>{label}</th><th>Warm %</th><th>Ranking users</th></tr></thead><tbody>{report.temporal_runs.map((row) => <tr key={`${row.algorithm}-${row.fold}`}><td>{row.fold}</td><td>{name(row.algorithm)}</td><td>{row.train.ratings} / {row.heldout.ratings}</td><td>{number(row.evaluation[metric])}</td><td>{number(row.evaluation.known_pair_fraction * 100, 2)}</td><td>{row.evaluation.ranking_users}</td></tr>)}</tbody></table></div></details>
      </article>
    </div>
    <article className="experiment-chart experiment-figure stability-bootstrap">
      <span className="experiment-figure-label">C · {split} · {report.bootstrap_repetitions.toLocaleString()} BOOTSTRAP</span><h3>{t("Khoảng tin cậy của chênh lệch")}</h3>
      <p>{t("Δ = A − B; B là mô hình tham chiếu chọn từ validation. Điểm là chênh lệch quan sát, đoạn ngang là CI percentile 95%, đường đứt là 0.")}</p>
      {finiteIntervals.length > 0 && <svg className="experiment-ci-plot" viewBox={`0 0 560 ${100 + intervals.length * 48}`} role="img" aria-label={t("Khoảng tin cậy của chênh lệch")}>
        <title>{label} · Δ(A − B) · CI 95%</title>
        <line className="plot-checkpoint" x1={x(0)} x2={x(0)} y1={15} y2={intervals.length * 48 + 15} />
        {finiteIntervals.map((row, index) => <g key={row.algorithm_a}>
          <text x={8} y={36 + index * 48}>{name(row.algorithm_a)}</text>
          <line stroke={experimentColor(row.algorithm_a)} strokeWidth="3" x1={x(row.ci_low!)} x2={x(row.ci_high!)} y1={32 + index * 48} y2={32 + index * 48} />
          <circle fill={experimentColor(row.algorithm_a)} cx={x(row.delta!)} cy={32 + index * 48} r="5"><title>{number(row.delta)} [{number(row.ci_low)}; {number(row.ci_high)}]</title></circle>
        </g>)}
        {[min - pad, 0, max + pad].map((tick, index) => <text key={index} x={x(tick)} y={intervals.length * 48 + 43} textAnchor="middle">{number(tick, 4)}</text>)}
        <text x={315} y={intervals.length * 48 + 74} textAnchor="middle">Δ {label} · A − B</text>
      </svg>}
      <div className="experiment-table-wrap"><table className="experiment-table"><caption>{label} · {split} · CI 95%</caption><thead><tr><th>A</th><th>B</th><th>Δ(A − B)</th><th>CI 95%</th><th>{t("Người dùng")}</th><th>{t("Diễn giải")}</th></tr></thead><tbody>{intervals.map((row) => <tr key={row.algorithm_a}><th scope="row">{name(row.algorithm_a)}</th><td>{name(row.algorithm_b)}</td><td>{number(row.delta)}</td><td>[{number(row.ci_low)}; {number(row.ci_high)}]</td><td>{row.users}</td><td>{conclusion(row)}</td></tr>)}</tbody></table></div>
      {!intervals.length && <p className="experiment-empty">{t("Bootstrap chưa hoàn tất cho tập này.")}</p>}
      {intervals.length > 0 && <p className="experiment-insight"><b>{t("Kết quả từng cặp:")}</b> {intervals.map((row) => `${name(row.algorithm_a)} / ${name(row.algorithm_b)}: ${conclusion(row)}`).join("; ")}.</p>}
      <p className="experiment-insight"><b>{t("Nhận xét:")}</b> {t("RMSE/MAE: CI hoàn toàn âm ủng hộ A; NDCG: CI hoàn toàn dương ủng hộ A. Khoảng chứa 0 chưa chứng minh được chênh lệch và không chứng minh tương đương.")} {t("Các CI là từng cặp, mang tính khám phá, chưa điều chỉnh nhiều phép so sánh. Chúng chỉ đo bất định theo người dùng của mô hình gốc đã lưu, không đo biến động qua seed.")}</p>
    </article>
    <div className="stability-downloads">{([['stability', 'Báo cáo độ ổn định (.json)'], ['seeds', 'Từng seed (.csv)'], ['temporal', 'Từng fold (.csv)'], ['bootstrap', 'Khoảng tin cậy (.csv)']] as const).map(([id, text]) => <a key={id} className="button button-outline" href={apiResourceUrl(`/experiments/exports/${id}`)}><Download size={15} />{t(text)}</a>)}</div>
  </section>;
}
