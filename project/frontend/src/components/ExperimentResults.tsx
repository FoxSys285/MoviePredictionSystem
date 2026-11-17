import { useState } from "react";
import { Download } from "lucide-react";
import { LabChartFigure } from "./LabChart";
import { MetricAssessment } from "./MetricReferences";
import type { ReferenceValues } from "../lib/metric-references";
import { apiResourceUrl } from "../lib/api";
import { metricNumber, type DatasetStats, type LabGroup, type LabJob, type LabRule } from "../lib/lab";

export function DatasetSummary({stats,label}:{stats?:DatasetStats;label:string}) {
  if(!stats)return null;
  const cells:[[string,unknown],...[string,unknown][]]=[
    ["Ratings",stats.ratings],["Users",stats.users],["Movies có ratings",stats.movies],["Genres",stats.genres],
    ["Rating min / max",`${stats.rating_min} / ${stats.rating_max}`],["Rating mean",metricNumber(stats.rating_mean)],
    ["Sparsity",`${(stats.sparsity*100).toFixed(3)}%`],["Movies trong catalog",stats.catalog_movies],
  ];
  return <section className="lab-section"><div className="lab-section-heading"><h2>{label}</h2><span className="lab-badge">MovieLens 25M</span></div>
    <div className="lab-stat-grid">{cells.map(([name,value])=><div key={name}><span>{name}</span><strong>{typeof value==="number"?value.toLocaleString("vi-VN"):String(value)}</strong></div>)}</div></section>;
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

export function ExperimentResults({job,group,references={}}:{job:LabJob;group:LabGroup;references?:ReferenceValues}) {
  const result=job.result!;
  const cv=result.cv;
  return <section id="lab-results" className="lab-section lab-results" aria-label="Kết quả thực nghiệm">
    <div className="lab-section-heading"><div><span className="lab-kicker">EXPERIMENT REPORT</span><h2>Kết quả thực nghiệm</h2></div><div className="lab-downloads">{["csv","xlsx","json"].map(format=><a className="lab-outline" key={format} href={apiResourceUrl(`/experiment-lab/runs/${job.id}/export/${format}`)}><Download size={14}/>{format.toUpperCase()}</a>)}</div></div>
    <p className="lab-muted">{result.provenance.sampled_users.toLocaleString("vi-VN")} users · {result.provenance.sampled_ratings.toLocaleString("vi-VN")} ratings thật · Seed 42</p>
    <div className="lab-split-cards">{Object.entries(result.provenance.splits).map(([s,f])=><div key={s}><span>{s}</span><strong>{f.ratings.toLocaleString("vi-VN")}</strong><small>ratings</small></div>)}</div>
    <DatasetSummary stats={result.provenance.sample} label="Dataset của lần chạy"/>
    {result.selection&&<div className="lab-insight"><strong>Nhận xét & lựa chọn mô hình</strong><p>{result.selection}</p></div>}
    <div className="lab-table-scroll lab-comparison"><table><caption>So sánh chất lượng và chi phí tính toán</caption><thead><tr><th>Thuật toán / cấu hình</th><th>Tập đánh giá</th>{group.metrics.map(m=><th key={m.key}>{m.name}</th>)}<th>Model size (KiB)</th>{group.id==="clustering"&&<><th>Số cụm</th><th>Noise</th></>}</tr></thead>
      <tbody>{result.rows.map((r,i)=><tr key={i}><th scope="row">{r.name}</th><td>{r.split}</td>{group.metrics.map(m=><td key={m.key} data-split={r.split} data-metric={m.key}>{metricNumber(r[m.key])}<MetricAssessment metric={m} row={r} result={result} values={references} metrics={group.metrics}/></td>)}<td>{typeof r.model_bytes==="number"?metricNumber(r.model_bytes/1024):"N/A"}</td>{group.id==="clustering"&&<><td>{metricNumber(r.clusters)}</td><td>{typeof r.noise_fraction==="number"?`${(r.noise_fraction*100).toFixed(2)}%`:"N/A"}</td></>}</tr>)}</tbody></table></div>
    {cv&&<section className="lab-section"><h2>3-Fold Cross Validation · Mean ± Std</h2><p className="lab-muted">Chỉ tính trên Train 70%. F1 dùng lớp Thích, ngưỡng CV = 0,5.</p><div className="lab-table-scroll"><table><thead><tr><th>Thuật toán</th>{group.metrics.map(m=><th key={m.key}>{m.name}</th>)}</tr></thead><tbody>{Object.entries(cv).map(([a,data])=><tr key={a}><th>{group.algorithms.find(x=>x.id===a)?.name||a}</th>{group.metrics.map(m=><td key={m.key}>{data.metrics[m.key]?`${metricNumber(data.metrics[m.key].mean)} ± ${metricNumber(data.metrics[m.key].std)}`:"N/A"}</td>)}</tr>)}</tbody></table></div></section>}
    {result.class_balance&&<section className="lab-section"><h2>Phân bố nhãn & Confusion Matrix</h2><div className="lab-balance-grid">{Object.entries(result.class_balance).map(([s,b])=><div key={s}><strong>{s}</strong><p>Không thích: {b.negative.toLocaleString("vi-VN")} · Thích: {b.positive.toLocaleString("vi-VN")}</p><progress max={1} value={b.positive_fraction}/><small>Lớp Thích: {(b.positive_fraction*100).toFixed(1)}%</small></div>)}</div>
      <div className="lab-table-scroll"><table><thead><tr><th>Thuật toán</th><th>Tập</th><th>Precision</th><th>Recall</th><th>TN</th><th>FP</th><th>FN</th><th>TP</th></tr></thead><tbody>{result.rows.map((r,i)=>{const cm=r.confusion_matrix as number[][]|undefined;return <tr key={i}><th>{r.name}</th><td>{r.split}</td><td>{metricNumber(r.precision)}</td><td>{metricNumber(r.recall)}</td>{[cm?.[0]?.[0],cm?.[0]?.[1],cm?.[1]?.[0],cm?.[1]?.[1]].map((v,j)=><td key={j}>{metricNumber(v)}</td>)}</tr>;})}</tbody></table></div></section>}
    {result.rules&&<Rules rules={result.rules}/>}
    {result.clusters&&<section className="lab-section"><h2>Đặc trưng Genre từng cụm</h2><div className="lab-table-scroll"><table><thead><tr><th>Thuật toán</th><th>Cụm</th><th>Users Train</th><th>Genres có rating trung bình cao nhất</th></tr></thead><tbody>{result.clusters.flatMap(c=>c.profiles.map(p=><tr key={`${c.algorithm}/${p.cluster}`}><th>{c.algorithm}</th><td>{p.cluster<0?"Noise":p.cluster}</td><td>{p.users}</td><td>{Object.entries(p.genre_means).map(([g,v])=>`${g}: ${v.toFixed(2)}`).join(" · ")}</td></tr>))}</tbody></table></div></section>}
    {result.search&&<details className="lab-detail-panel"><summary>Kết quả tìm kiếm K / eps / min_samples</summary><div className="lab-table-scroll"><table><thead><tr><th>Thuật toán</th><th>Cấu hình</th><th>Silhouette Train</th><th>DBI Train</th><th>Noise</th></tr></thead><tbody>{result.search.map((s,i)=><tr key={i}><td>{String(s.algorithm)}</td><td>{JSON.stringify(s.config)}</td><td>{metricNumber(s.silhouette)}</td><td>{metricNumber(s.dbi)}</td><td>{metricNumber(s.noise_fraction)}</td></tr>)}</tbody></table></div></details>}
    <div className="lab-section-heading lab-figures-heading"><h2>Biểu đồ đánh giá</h2><span className="lab-badge">{result.charts.length} biểu đồ</span></div><div className="lab-charts">{result.charts.map(c=><LabChartFigure key={c.index} chart={c}/>)}</div>
    <details className="lab-detail-panel"><summary>Cấu hình được chọn</summary><pre>{JSON.stringify(result.best_configs,null,2)}</pre></details>
    <section className="lab-section"><h2>Phương pháp & giới hạn đánh giá</h2>{result.notes.map(n=><p className="lab-muted lab-result-note" key={n}>{n}</p>)}{group.id==="reinforcement"&&<p className="lab-muted">Tỷ lệ hành động có phản hồi quan sát trên Test: {result.rows.filter(r=>r.split==="test").map(r=>`${r.name}: ${typeof r.observed_feedback_fraction==="number"?(r.observed_feedback_fraction*100).toFixed(2):"N/A"}%`).join(" · ")}</p>}
    {result.runtime&&<p className="lab-muted">{result.runtime.device} · {result.runtime.processor||result.runtime.platform} · {result.runtime.cpu_count} logical CPUs · RAM {(result.runtime.ram_bytes/2**30).toFixed(1)} GiB · Python {result.runtime.python}. Thời gian: giây; RSS là bộ nhớ toàn tiến trình.</p>}</section>
  </section>;
}
