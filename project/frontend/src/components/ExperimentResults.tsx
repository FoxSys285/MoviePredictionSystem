import { Download } from "lucide-react";
import { LabChartFigure } from "./LabChart";
import { apiResourceUrl } from "../lib/api";
import { metricNumber, type DatasetStats, type LabGroup, type LabJob, type LabResult } from "../lib/lab";

export function DatasetSummary({ stats, label }: { stats?: DatasetStats; label: string }) {
  const cards = [
    ["Ratings", stats?.ratings],
    ["Users", stats?.users],
    ["Movies", stats?.movies],
    ["Genres", stats?.genres],
  ] as const;
  return (
    <section className="lab-dataset-summary" aria-label={label}>
      <span className="lab-kicker">{label}</span>
      <div>{cards.map(([name, value]) => (
        <article key={name}><span>{name}</span><strong>{value?.toLocaleString("vi-VN") ?? "—"}</strong></article>
      ))}</div>
    </section>
  );
}

function valueText(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? metricNumber(value) : "—";
}

type Fold = { fold?: number; rmse?: number; mae?: number };

function CrossValidation({ result }: { result: LabResult }) {
  const entries = Object.entries(result.cv ?? {});
  if (!entries.length) return null;
  const foldCount = result.cv_folds ?? 5;
  return (
    <section className="lab-result-row" aria-labelledby="lab-cv-title">
      <h3 id="lab-cv-title">Cross validation trên Train</h3>
      <p>GridSearchCV dùng 5 fold trên Train để chọn tham số. Validation và Test được giữ riêng.</p>
      <div className="lab-table-scroll">
        <table className="lab-result-table">
          <thead><tr><th>Thuật toán</th><th>Chỉ số</th>
            {Array.from({ length: foldCount }, (_, index) => <th key={index}>Fold {index + 1}</th>)}
            <th>Trung bình ± độ lệch chuẩn</th>
          </tr></thead>
          <tbody>{entries.flatMap(([id, data]) => (["rmse", "mae"] as const).map((metric) => {
            const folds = (data.folds ?? []) as Fold[];
            const summary = data.metrics[metric];
            return <tr key={`${id}-${metric}`}>
              <th>{id}</th><td>{metric.toUpperCase()}</td>
              {Array.from({ length: foldCount }, (_, index) => <td key={index}>{valueText(folds[index]?.[metric])}</td>)}
              <td>{summary ? `${valueText(summary.mean)} ± ${valueText(summary.std)}` : "—"}</td>
            </tr>;
          }))}</tbody>
        </table>
      </div>
    </section>
  );
}

export function SimpleExperimentResults({ job, group }: { job: LabJob | null; group: LabGroup }) {
  const result = job?.status === "completed" ? job.result : undefined;
  const rows = result?.rows ?? [];
  const ids = [...new Set(rows.map((row) => String(row.variant_id ?? row.algorithm)))];
  const findRow = (id: string, split: string) => rows.find(
    (row) => String(row.variant_id ?? row.algorithm) === id && row.split === split,
  );
  const charts = (result?.charts ?? []).filter((chart) => chart.kind === "bar");

  return (
    <section id="lab-results" className="lab-section lab-results" data-lab-group={group.id} aria-label="Kết quả hồi quy">
      <div className="lab-section-heading">
        <div><span className="lab-kicker">REGRESSION REPORT</span><h2>Kết quả hồi quy</h2></div>
        {job && <div className="lab-downloads">{["csv", "xlsx", "json"].map((format) => (
          <a className="lab-outline" key={format} href={apiResourceUrl(`/experiment-lab/runs/${job.id}/export/${format}`)}>
            <Download size={14}/>{format.toUpperCase()}
          </a>
        ))}</div>}
      </div>

      {!result ? <p className="lab-muted">Kết quả và biểu đồ hồi quy sẽ xuất hiện sau khi lần chạy hoàn tất.</p> : <>
        <p className="lab-muted">
          {result.provenance.sampled_users.toLocaleString("vi-VN")} users · {result.provenance.sampled_ratings.toLocaleString("vi-VN")} ratings · Seed {String(result.config?.seed ?? "—")}
        </p>
        {result.baseline && <p className="lab-muted">
          Global Mean dự đoán mọi rating bằng trung bình Train ({valueText(result.baseline.prediction)}). RMSE và MAE càng thấp càng tốt.
        </p>}

        <div className="lab-table-scroll">
          <table className="lab-result-table">
            <thead><tr><th rowSpan={2}>Thuật toán</th>
              {["Validation", "Test"].map((split) => <th key={split} colSpan={3}>{split}</th>)}
            </tr><tr>{["validation", "test"].flatMap((split) => ["RMSE", "MAE", "Fit (s)"].map((metric) => (
              <th key={`${split}-${metric}`}>{metric}</th>
            )))}</tr></thead>
            <tbody>{ids.map((id) => {
              const validation = findRow(id, "validation");
              const test = findRow(id, "test");
              const selected = id === result.selected_variant_id;
              return <tr key={id} className={selected ? "is-selected" : undefined}>
                <th>{String(validation?.name ?? test?.name ?? id)}{selected && <span className="selected-label">Được chọn theo Validation</span>}</th>
                {[validation, test].flatMap((row) => [row?.rmse, row?.mae, row?.fit_seconds].map((value, index) => (
                  <td key={`${row?.split ?? "missing"}-${index}`}>{valueText(value)}</td>
                )))}</tr>;
            })}
            {!ids.length && <tr><td colSpan={7}>Lần chạy chưa có dòng kết quả.</td></tr>}
            </tbody>
          </table>
        </div>

        {charts.length > 0 && <section className="lab-result-row" aria-label="Biểu đồ hồi quy">
          <h3>Biểu đồ so sánh</h3>
          <div className="lab-result-charts">{charts.map((chart) => <LabChartFigure key={chart.index} chart={chart}/>)}</div>
        </section>}
        <CrossValidation result={result}/>
        {result.selection && <p className="lab-result-selection">{result.selection}</p>}
        <details className="lab-detail-panel">
          <summary>Cấu hình, log tham số và nguồn dữ liệu</summary>
          <pre>{JSON.stringify({ config: result.config, best_configs: result.best_configs, provenance: result.provenance, runtime: result.runtime, notes: result.notes }, null, 2)}</pre>
        </details>
      </>}
    </section>
  );
}
