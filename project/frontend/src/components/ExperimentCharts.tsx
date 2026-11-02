import { useState } from "react";
import { SectionTitle } from "./ui";
import { usePreferences } from "../lib/preferences";

export type ExperimentResult = {
  algorithm: string;
  rmse?: number | null;
  mae?: number | null;
  ndcg_at_10?: number | null;
  ranking_users?: number;
  best_epoch?: number;
  learning_curve?: { epoch: number; train_mse?: number; validation_rmse?: number }[];
};
type HistoryPoint = { support_ratings: number; users: number; query_ratings: number; rmse: number | null; mae: number | null };
export type ExperimentAnalysis = {
  run_id: string;
  cold_history: Partial<Record<"validation" | "test", {
    evaluated_users: number;
    eligible_users: number;
    results: { algorithm: string; points: HistoryPoint[] }[];
  }>>;
};
const colors = ["#de81b0", "#54b9bc", "#c8a547", "#9d97ee", "#76b774", "#da9570", "#88a9d8"];
const algorithms = ["ridge", "xgboost", "svd", "ncf", "baseline", "item_knn", "nmf"];
export const experimentColor = (algorithm: string) => colors[Math.max(0, algorithms.indexOf(algorithm)) % colors.length];
const color = experimentColor;
const valid = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value) && value >= 0;
type Series = { label: string; color: string; values: { x: number; y: number }[] };

export function LineChart({ series, ticks, xLabel, yLabel, title, marker }: {
  series: Series[]; ticks: number[]; xLabel: string; yLabel: string; title: string; marker?: number;
}) {
  const { t, language } = usePreferences();
  const number = (value: number) => value.toLocaleString(language === "en" ? "en-US" : "vi-VN", { maximumFractionDigits: 3 });
  const values = series.flatMap((item) => item.values);
  if (!values.length) return <p className="experiment-empty">{t("Chưa có dữ liệu để vẽ biểu đồ.")}</p>;
  const width = 560, height = 290, left = 64, right = 22, top = 32, bottom = 52;
  const maxY = Math.max(.01, ...values.map((point) => point.y)) * 1.12;
  const minX = Math.min(...ticks), maxX = Math.max(...ticks);
  const x = (value: number) => left + (value - minX) / Math.max(1, maxX - minX) * (width - left - right);
  const y = (value: number) => height - bottom - value / maxY * (height - top - bottom);
  return <>
    <svg className="experiment-line-plot" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
      <title>{title}</title>
      <text x={left} y={16} className="plot-axis-title">{yLabel}</text>
      {[0, 1, 2, 3, 4].map((step) => {
        const value = maxY * step / 4;
        return <g key={step}><line className="plot-grid" x1={left} x2={width - right} y1={y(value)} y2={y(value)} /><text x={left - 9} y={y(value) + 4} textAnchor="end">{number(value)}</text></g>;
      })}
      {ticks.map((tick) => <text key={tick} x={x(tick)} y={height - bottom + 21} textAnchor="middle">{tick}</text>)}
      <text x={(width + left - right) / 2} y={height - 8} textAnchor="middle" className="plot-axis-title">{xLabel}</text>
      {marker !== undefined && <line className="plot-checkpoint" x1={x(marker)} x2={x(marker)} y1={top} y2={height - bottom}><title>{t("Checkpoint tốt nhất: epoch {epoch}", { epoch: marker })}</title></line>}
      {series.map((item) => <g key={item.label}>
        <polyline fill="none" stroke={item.color} strokeWidth="2.5" points={item.values.map((point) => `${x(point.x)},${y(point.y)}`).join(" ")} />
        {item.values.map((point) => <circle key={point.x} cx={x(point.x)} cy={y(point.y)} r="4" fill={item.color}><title>{item.label}: {point.x} → {number(point.y)}</title></circle>)}
      </g>)}
    </svg>
    <div className="experiment-legend">{series.map((item) => <span key={item.label}><i style={{ background: item.color }} />{item.label}</span>)}</div>
    <details className="experiment-chart-data"><summary>{t("Xem số liệu biểu đồ")}</summary><div className="experiment-data-scroll"><table>
      <caption>{title}</caption><thead><tr><th>{t("Chuỗi dữ liệu")}</th><th>{xLabel}</th><th>{yLabel}</th></tr></thead>
      <tbody>{series.flatMap((item) => item.values.map((point) => <tr key={`${item.label}-${point.x}`}><th scope="row">{item.label}</th><td>{point.x}</td><td>{number(point.y)}</td></tr>))}</tbody>
    </table></div></details>
  </>;
}

export function ExperimentCharts({ results, training, analysis, split, names }: {
  results: ExperimentResult[]; training: ExperimentResult[]; analysis?: ExperimentAnalysis | null;
  split: "validation" | "test"; names: Record<string, string>;
}) {
  const { t, language } = usePreferences();
  const [historyMetric, setHistoryMetric] = useState<"rmse" | "mae">("rmse");
  const number = (value: number, digits = 4) => value.toLocaleString(language === "en" ? "en-US" : "vi-VN", { maximumFractionDigits: digits, minimumFractionDigits: digits });
  const name = (algorithm: string) => names[algorithm] || algorithm;
  const errors = results.filter((row) => valid(row.rmse) || valid(row.mae));
  const maxError = Math.max(.01, ...errors.flatMap((row) => [row.rmse, row.mae].filter(valid)));
  const winners = (metric: "rmse" | "mae" | "ndcg_at_10", highest = false) => {
    const rows = results.filter((row) => valid(row[metric]));
    const best = (highest ? Math.max : Math.min)(...rows.map((row) => row[metric]!));
    return { label: rows.filter((row) => row[metric] === best).map((row) => name(row.algorithm)).join(", "), value: best };
  };
  const bestRMSE = winners("rmse"), bestMAE = winners("mae"), bestNDCG = winners("ndcg_at_10", true);
  const ranking = results.filter((row) => valid(row.ndcg_at_10));
  const maxNDCG = Math.max(.01, ...ranking.map((row) => row.ndcg_at_10!));
  const cohort = analysis?.cold_history[split];
  const historySeries: Series[] = (cohort?.results || []).map((row) => ({
    label: name(row.algorithm), color: color(row.algorithm),
    values: row.points.filter((point) => valid(point[historyMetric])).map((point) => ({ x: point.support_ratings, y: point[historyMetric]! })),
  }));
  const ncf = training.find((row) => row.algorithm === "ncf");
  const curve = ncf?.learning_curve || [];
  const trainSeries = curve.filter((row) => valid(row.train_mse)).map((row) => ({ x: row.epoch, y: Math.sqrt(row.train_mse!) }));
  const validationSeries = curve.filter((row) => valid(row.validation_rmse)).map((row) => ({ x: row.epoch, y: row.validation_rmse! }));
  const bestPoint = validationSeries.find((point) => point.x === ncf?.best_epoch);
  const lastPoint = validationSeries.at(-1);
  const start = cohort?.results.map((row) => row.points.find((point) => point.support_ratings === 0)?.[historyMetric]).filter(valid) || [];
  const end = cohort?.results.map((row) => row.points.find((point) => point.support_ratings === 20)?.[historyMetric]).filter(valid) || [];
  const improved = cohort?.results.filter((row) => {
    const first = row.points.find((point) => point.support_ratings === 0)?.[historyMetric];
    const last = row.points.find((point) => point.support_ratings === 20)?.[historyMetric];
    return valid(first) && valid(last) && last < first;
  }).map((row) => name(row.algorithm)) || [];
  return <section className="experiment-visuals" aria-label={t("Biểu đồ và nhận xét thực nghiệm")}>
    <SectionTitle>{t("Biểu đồ và nhận xét thực nghiệm")}</SectionTitle>
    <p className="subtle-note">{t("Ba biểu đồ đầu theo tập đang chọn. Đường học NCF luôn dùng train/validation của run đã lưu.")}</p>
    <div className="experiment-visual-grid">
      <article className="experiment-chart experiment-figure">
        <span className="experiment-figure-label">01 · {split}</span><h3>{t("Sai số dự đoán RMSE / MAE")}</h3>
        <p>{t("Cùng thang điểm rating; thanh bắt đầu từ 0, ngắn hơn là tốt hơn.")}</p>
        {errors.length ? <>
          <div className="experiment-legend"><span><i style={{ background: "#de81b0" }} />RMSE</span><span><i style={{ background: "#54b9bc" }} />MAE</span></div>
          <div className="experiment-grouped-bars" role="img" aria-label={t("Sai số dự đoán RMSE / MAE")}>
            {errors.map((row) => <div className="experiment-bar-group" key={row.algorithm}><strong>{name(row.algorithm)}</strong>{(["rmse", "mae"] as const).map((metric) => <div className="experiment-metric-bar" key={metric}><span>{metric.toUpperCase()}</span><div><i style={{ width: valid(row[metric]) ? `${row[metric]! / maxError * 100}%` : 0, background: metric === "rmse" ? "#de81b0" : "#54b9bc" }} /></div><b>{valid(row[metric]) ? number(row[metric]!) : "—"}</b></div>)}</div>)}
          </div><p className="experiment-axis-note">0 → {number(maxError, 2)} · {t("điểm sai số")}</p>
          <p className="experiment-insight"><b>{t("Nhận xét:")}</b> {t("RMSE thấp nhất: {model} ({value}).", { model: bestRMSE.label || "—", value: valid(bestRMSE.value) ? number(bestRMSE.value) : "—" })} {t("MAE thấp nhất: {model} ({value}).", { model: bestMAE.label || "—", value: valid(bestMAE.value) ? number(bestMAE.value) : "—" })} {t("Hai chỉ số có thể chọn mô hình khác nhau vì RMSE phạt mạnh lỗi lớn. Một run chưa xác nhận ưu thế ổn định.")}</p>
        </> : <p className="experiment-empty">{t("Chưa đo RMSE/MAE trên tập này.")}</p>}
      </article>
      <article className="experiment-chart experiment-figure">
        <span className="experiment-figure-label">02 · {split}</span><h3>{t("Chất lượng gợi ý NDCG@10")}</h3>
        <p>{t("Cao hơn là tốt hơn; phim có rating thật ≥ 4 được xem là phù hợp.")}</p>
        {ranking.length ? <>
          <div className="experiment-ranking-bars" role="img" aria-label={t("Chất lượng gợi ý NDCG@10")}>
            {ranking.map((row) => <div className="experiment-metric-bar" key={row.algorithm}><span>{name(row.algorithm)}</span><div><i style={{ width: `${row.ndcg_at_10! / maxNDCG * 100}%`, background: color(row.algorithm) }} /></div><b>{number(row.ndcg_at_10!)}</b></div>)}
          </div><p className="experiment-axis-note">0 → {number(maxNDCG)} · {t("trục phóng theo dữ liệu, miền NDCG là [0; 1]")}</p>
          <p className="experiment-insight"><b>{t("Nhận xét:")}</b> {bestNDCG.value > 0 ? t("NDCG@10 cao nhất: {model} ({value}).", { model: bestNDCG.label, value: number(bestNDCG.value) }) : t("Chưa mô hình nào đưa được phim phù hợp vào Top 10 trong nhóm đánh giá này.")} {t("Giá trị 0 là đã đo nhưng không có lượt trúng Top 10; dấu — là chưa đo. Kết quả dựa trên {users} người dùng, chưa đại diện cho mọi người dùng.", { users: Math.max(0, ...ranking.map((row) => row.ranking_users || 0)) })}</p>
        </> : <p className="experiment-empty">{t("Chưa đo NDCG@10 trên tập này.")}</p>}
      </article>
      <article className="experiment-chart experiment-figure">
        <span className="experiment-figure-label">03 · {split} · {t("NGƯỜI DÙNG MỚI")}</span>
        <div className="experiment-figure-heading"><h3>{t("Chất lượng theo lượng lịch sử")}</h3><div className="experiment-split" role="group" aria-label={t("Chỉ số theo lượng lịch sử")}>
          {(["rmse", "mae"] as const).map((metric) => <button key={metric} className={historyMetric === metric ? "selected" : ""} aria-pressed={historyMetric === metric} onClick={() => setHistoryMetric(metric)}>{metric.toUpperCase()}</button>)}
        </div></div>
        <p>{t("Cùng người dùng ngoài train, cùng tập query; chỉ thay đổi 0 / 5 / 10 / 20 rating hỗ trợ có trước query.")}</p>
        {historySeries.some((item) => item.values.length) ? <>
          <LineChart series={historySeries} ticks={[0, 5, 10, 20]} xLabel={t("Số rating hỗ trợ")} yLabel={`${historyMetric.toUpperCase()} ↓`} title={t("Chất lượng theo lượng lịch sử")} />
          <p className="experiment-insight"><b>{t("Nhận xét:")}</b> {improved.length ? t("Từ 0 lên 20 rating, sai số giảm ở: {models}.", { models: improved.join(", ") }) : t("Thêm 20 rating chưa làm giảm sai số so với 0 rating ở các mô hình được đo.")} {t("Không mặc định thêm lịch sử luôn tốt hơn; cần đọc từng đường và kiểm tra độ ổn định.")}</p>
          <p className="experiment-axis-note">{t("Đã đo {users}/{eligible} người dùng đủ điều kiện; mỗi người có ít nhất 5 rating query sau 20 rating hỗ trợ. Sai số gộp theo rating, chỉ xét phim đã có trong train.", { users: cohort?.evaluated_users, eligible: cohort?.eligible_users })} {t("Mức 0 dùng prior của mô hình, chưa cá nhân hóa. Trọng số phim và cấu hình được giữ cố định.")}</p>
          {!!start.length && !!end.length && <p className="experiment-axis-note">{t("Miền sai số tại 0 rating: {start}; tại 20 rating: {end}.", { start: `${number(Math.min(...start))}–${number(Math.max(...start))}`, end: `${number(Math.min(...end))}–${number(Math.max(...end))}` })}</p>}
        </> : <p className="experiment-empty">{t("Chưa có kết quả 0/5/10/20 rating cho run và tập này. Biểu đồ sẽ xuất hiện sau khi chạy phép đo lịch sử; không suy diễn từ điểm tổng.")}</p>}
      </article>
      <article className="experiment-chart experiment-figure">
        <span className="experiment-figure-label">04 · TRAIN / VALIDATION</span><h3>{t("Đường học của NCF")}</h3>
        <p>{t("Căn bậc hai của train MSE và validation RMSE theo epoch. Đường đứt đánh dấu checkpoint được chọn.")}</p>
        {curve.length ? <>
          <LineChart series={[{ label: "√MSE train", color: "#54b9bc", values: trainSeries }, { label: "RMSE validation", color: "#de81b0", values: validationSeries }]} ticks={curve.map((row) => row.epoch)} xLabel="Epoch" yLabel={t("Sai số rating ↓")} title={t("Đường học của NCF")} marker={ncf?.best_epoch} />
          <p className="experiment-insight"><b>{t("Nhận xét:")}</b> {bestPoint ? t("Checkpoint tốt nhất ở epoch {epoch}, validation RMSE = {value}.", { epoch: bestPoint.x, value: number(bestPoint.y) }) : t("Chưa lưu đủ validation RMSE để xác định checkpoint.")} {bestPoint && lastPoint && lastPoint.y > bestPoint.y ? t("Validation ở epoch cuối cao hơn checkpoint tốt nhất; giữ checkpoint tốt nhất thay vì lấy epoch cuối.") : t("Cần theo dõi đồng thời train và validation để nhận biết dấu hiệu quá khớp.")}</p>
          <p className="experiment-axis-note">{t("Train MSE được tích lũy trong mini-batch khi có dropout và chưa clip; validation đo sau epoch, tắt dropout và clip [0,5; 5]. Chênh lệch hai đường không tự chứng minh quá khớp. Test không dùng để chọn epoch.")}</p>
        </> : <p className="experiment-empty">{t("Run này chưa lưu đường học NCF.")}</p>}
      </article>
    </div>
  </section>;
}

export function StabilityDesign({ completed = false }: { completed?: boolean }) {
  const { t } = usePreferences();
  return <section className="experiment-stability" aria-label={t("Thiết kế thực nghiệm lặp lại")}>
    <div className="experiment-results-heading"><SectionTitle>{t("Thiết kế thực nghiệm lặp lại")}</SectionTitle><span className={`experiment-plan-badge ${completed ? "is-complete" : ""}`}>{t(completed ? "Đã thực hiện trên mẫu hiện tại" : "Kế hoạch · chưa chạy")}</span></div>
    <p className="subtle-note">{t("Lặp huấn luyện để đo độ ổn định; bootstrap lấy mẫu lại kết quả dự đoán để đo độ chắc chắn. Ba nguồn biến động được báo cáo riêng.")}</p>
    <div className="experiment-stability-grid">
      {[
        ["A", "05 seed · cùng một split", "Độ ngẫu nhiên khi huấn luyện", "Giữ nguyên dữ liệu, đặc trưng, tham số và tập người dùng ranking. Chỉ thay model seed: 13, 42, 73, 101, 137; không lấy mẫu lại dataset.", "Báo từng lần chạy và mean ± std qua seed cho RMSE, MAE, NDCG@10; Ridge có thể không đổi. Không chọn seed thắng để đại diện."],
        ["B", "05 fold · tiến theo thời gian", "Khả năng ổn định qua giai đoạn", "Chia riêng 70% train thành 6 khối thời gian B1…B6. Fold 1 học B1, đánh giá B2; …; fold 5 học B1…B5, đánh giá B6. Seed cố định 42, fit lại đặc trưng ở mỗi fold.", "Báo điểm từng fold và mean ± std qua giai đoạn. Không trộn std qua fold với std qua seed. Nếu dùng fold để tuning, đây là kết quả phát triển, không thay thế test độc lập."],
        ["C", "2.000 bootstrap · ghép cặp", "Độ chắc chắn của chênh lệch", "Lấy mẫu người dùng có hoàn lại; mỗi lượt dùng đúng cùng mẫu cho hai mô hình. Giữ toàn bộ rating của user, tính lại RMSE/MAE; NDCG dùng điểm từng user trên cùng tập ứng viên.", "Báo Δ(A − B) và khoảng tin cậy 95%. Với RMSE/MAE, khoảng hoàn toàn âm ủng hộ A; với NDCG, khoảng hoàn toàn dương ủng hộ A. Khoảng chứa 0: chưa đủ bằng chứng về chênh lệch."],
      ].map(([id, title, question, method, output]) => <article key={id}>
        <span className="experiment-stability-id">{id}</span><h3>{t(title)}</h3><h4>{t(question)}</h4><p>{t(method)}</p><p className="experiment-stability-output">{t(output)}</p>
      </article>)}
    </div>
    <div className="experiment-decision"><h3>{t("Quy tắc kết luận đã định trước")}</h3>
      <ol>
        <li>{t("Chọn cấu hình trên validation: RMSE cho rating, NDCG@10 cho gợi ý. Chọn mô hình tham chiếu trước khi so chênh lệch; không dùng test để đổi cấu hình hoặc chọn seed.")}</li>
        <li>{t("Báo chất lượng trung bình, độ biến động và CI của chênh lệch; mean ± std không phải khoảng tin cậy 95%. CI bootstrap của một run không đo được biến động do huấn luyện.")}</li>
        <li>{t("Xét thêm mức cải thiện có ý nghĩa sử dụng và chi phí p95/RAM. Nếu chênh lệch nhỏ hoặc chưa chắc chắn, ưu tiên mô hình đơn giản và nhanh hơn đáp ứng yêu cầu.")}</li>
      </ol>
      <p>{t(completed ? "Đã lưu từng lần chạy, dấu vết dữ liệu/mã nguồn, seed, fold, tham số và dự đoán theo người dùng. Kết quả seed, thời gian và bootstrap được trình bày riêng ở trên." : "Đầu ra cần lưu: run_id, dấu vết split, data seed, model seed, fold, tham số, điểm từng người dùng và thời gian. Các kết quả hiện tại vẫn là một split; chưa hiển thị mean ± std hay CI khi chưa đo.")}</p>
    </div>
  </section>;
}
