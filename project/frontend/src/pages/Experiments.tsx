import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Activity, AlertCircle, BookOpen, CheckCircle2, ChevronRight, Download, FlaskConical, Layers, Loader2, Minimize2, Network, Play, RefreshCw, Route, Square, Target } from "lucide-react";
import { PageHero } from "../components/ui";
import { LabChartFigure } from "../components/LabChart";
import { api, apiResourceUrl } from "../lib/api";
import { busy, GROUP_METRICS, LAB_METRICS, type Criterion, type LabCatalog, type LabGroup, type LabJob, type LabResult } from "../lib/lab";
import { usePreferences } from "../lib/preferences";

const icons = [Activity, Target, Layers, Network, Minimize2, Route];
const statuses: Record<string, string> = { queued: "Đang chờ", running: "Đang chạy", completed: "Đã hoàn tất", failed: "Có lỗi", cancelled: "Đã dừng", interrupted: "Bị gián đoạn" };
const defaults = { max_users: 150, seed: 42, repeats: 1, epochs: 8, positive_threshold: 4, n_clusters: 4, eps: 1.5, min_samples: 5, max_items: 80, min_support: .08, episodes: 1200, cluster_object: "movies" };

function CriteriaTable({ items, caption }: { items: Criterion[]; caption: string }) {
  const { t } = usePreferences();
  return <div className="lab-table-scroll lab-criteria"><table><caption>{caption}</caption><thead><tr><th>{t("Tiêu chí")}</th><th>{t("Đánh giá gì / cách xác định")}</th><th>{t("Cách diễn giải")}</th><th>{t("Câu hỏi được trả lời")}</th></tr></thead><tbody>{items.map(item => <tr key={item.name}><th scope="row">{item.name}</th><td>{item.definition}</td><td>{item.interpretation}</td><td>{item.question}</td></tr>)}</tbody></table></div>;
}

function ResultPanel({ result, job, group }: { result: LabResult; job: LabJob; group: LabGroup }) {
  const { t, language } = usePreferences();
  const [split, setSplit] = useState("validation");
  const splits = [...new Set(result.rows.map(r => r.split))];
  const selected = splits.includes(split) ? split : splits[0];
  const rows = result.rows.filter(r => r.split === selected);
  const charts = result.charts.filter(c => c.split === selected || c.split === "train");
  const display = (v: unknown, digits = 4) => typeof v === "number" && Number.isFinite(v) ? v.toLocaleString(language === "en" ? "en-US" : "vi-VN", { maximumFractionDigits: Number.isInteger(v) ? 0 : digits }) : typeof v === "boolean" ? t(v ? "Có" : "Không") : "N/A";
  return <section id="lab-results" className="lab-section lab-results" aria-label={t("Kết quả thực nghiệm")}>
    <div className="lab-section-heading"><div><span className="lab-kicker">03 / {t("KẾT QUẢ")}</span><h2>{t("So sánh từ lần chạy thực tế")}</h2></div><div className="lab-downloads"><a href={apiResourceUrl(`/experiment-lab/runs/${job.id}/export/csv`)} className="lab-outline"><Download size={15} />CSV</a><a href={apiResourceUrl(`/experiment-lab/runs/${job.id}/export/json`)} className="lab-outline"><Download size={15} />JSON</a></div></div>
    <p className="lab-muted">{result.selection}</p>
    <div className="lab-split-toolbar"><div className="lab-segments" role="group" aria-label={t("Tập dữ liệu đánh giá")}>{splits.map(s => <button key={s} type="button" aria-pressed={s === selected} className={s === selected ? "selected" : ""} onClick={() => setSplit(s)}>{s === "simulation" ? t("Mô phỏng") : s === "train" ? "Train" : s === "test" ? "Test" : "Validation"}</button>)}</div><span>{t("N/A: chưa đo hoặc không xác định. ± là std qua seed.")}</span></div>
    {result.class_balance?.[selected] && <p className="lab-balance">{t("Phân bố nhãn")}: {t("Thích")} {display(result.class_balance[selected].positive)} · {t("Không thích")} {display(result.class_balance[selected].negative)}</p>}
    <div className="lab-table-scroll lab-comparison"><table><caption>{t("Bảng so sánh kết quả")} · {selected}</caption><thead><tr><th>{t("Thuật toán / cấu hình")}</th>{GROUP_METRICS[group.id].map(key => <th key={key}>{key === "coverage_at_20" ? "Catalog coverage@20" : LAB_METRICS[key] || key}</th>)}</tr></thead><tbody>{rows.map((row,i) => <tr key={`${row.algorithm}-${i}`}><th scope="row">{row.name}</th>{GROUP_METRICS[group.id].map(key => <td key={key} title={row[key] == null ? t("Chỉ số chưa được đo hoặc không xác định với dữ liệu này.") : undefined}>{display(row[key])}{row.std?.[key] !== undefined && <small>± {display(row.std[key])}</small>}</td>)}</tr>)}</tbody></table></div>
    <details className="lab-details"><summary>{t("Phạm vi dữ liệu, cấu hình và lưu ý khi kết luận")}</summary><div className="lab-provenance-grid">{Object.entries(result.provenance.splits).map(([name, part]) => <div key={name}><b>{name}</b><strong>{display(part.ratings)}</strong><span>rating · {display(part.users)} user · {display(part.movies)} {t("phim")}</span><small>{new Date(part.min_timestamp*1000).toISOString().slice(0,10)} → {new Date(part.max_timestamp*1000).toISOString().slice(0,10)}</small></div>)}</div><p>{t("Mẫu thực nghiệm")}: {display(result.provenance.sampled_users)} user · {display(result.provenance.sampled_ratings)} / {display(result.provenance.prepared_ratings)} rating {t("đã chuẩn bị")} · data seed {result.provenance.data_seed}.</p><p className="lab-hash">Run: {job.id}<br />SHA-256 mẫu: {result.provenance.sample_sha256}</p><pre>{JSON.stringify(job.request, null, 2)}</pre><ul>{result.notes.map((note,i) => <li key={i}>{note}</li>)}</ul></details>
    {!!result.common_subset?.length && <details className="lab-details"><summary>{t("Đối chiếu trên tập điểm không nhiễu chung")}</summary><div className="lab-table-scroll"><table><thead><tr><th>{t("Thuật toán")}</th><th>{t("Số điểm chung")}</th><th>Silhouette</th><th>Davies–Bouldin</th></tr></thead><tbody>{result.common_subset.map(row => <tr key={row.algorithm}><th>{row.algorithm}</th><td>{row.points}</td><td>{display(row.silhouette)}</td><td>{display(row.davies_bouldin)}</td></tr>)}</tbody></table></div></details>}
    {!!result.bootstrap?.[selected]?.length && <details className="lab-details"><summary>{t("Khoảng tin cậy chênh lệch · bootstrap ghép cặp theo user")}</summary><p>{t("Khoảng 95% của chênh lệch (mô hình − tham chiếu), 200 lần bootstrap trên seed mô hình đầu. Với sai số, khoảng hoàn toàn âm ủng hộ mô hình; chứa 0 là chưa đủ bằng chứng. Đây không phải biến động huấn luyện qua seed.")}</p><div className="lab-table-scroll"><table><thead><tr><th>{t("Mô hình")}</th><th>{t("Tham chiếu")}</th><th>{t("Tiêu chí")}</th><th>CI low</th><th>CI high</th><th>User</th></tr></thead><tbody>{result.bootstrap[selected].map(row => <tr key={`${row.algorithm}-${row.metric}`}><th>{row.algorithm}</th><td>{row.reference}</td><td>{row.metric}</td><td>{display(row.ci_low)}</td><td>{display(row.ci_high)}</td><td>{row.users}</td></tr>)}</tbody></table></div></details>}
    {result.rules && <details className="lab-details"><summary>{t("Xem các luật kết hợp đã khai phá")} · {result.rules.length}</summary><div className="lab-table-scroll"><table className="lab-rules"><thead><tr><th>{t("Luật")}</th><th>Support</th><th>Confidence</th><th>Lift</th></tr></thead><tbody>{result.rules.map((rule,i) => <tr key={i}><th>{rule.antecedents.join(" + ")} → {rule.consequents.join(" + ")}</th><td>{display(rule.support)}</td><td>{display(rule.confidence)}</td><td>{display(rule.lift)}</td></tr>)}</tbody></table></div>{!result.rules.length && <p>{t("Không có luật vượt ngưỡng hiện tại.")}</p>}</details>}
    <div className="lab-section-heading lab-figures-heading"><div><h2>{t("Biểu đồ phân tích")}</h2><p className="lab-muted">{t("Các điểm và cột được tạo từ số liệu đã lưu. Bấm chú giải để ẩn/hiện chuỗi dữ liệu.")}</p></div><span className="lab-badge">{charts.length} {t("biểu đồ")}</span></div>
    <div className="lab-charts">{charts.map((c,i) => <LabChartFigure key={`${job.id}-${selected}-${i}`} chart={c} guide={group.charts[c.index]} />)}</div>
  </section>;
}

export function Experiments() {
  const { t } = usePreferences();
  const [params, setParams] = useSearchParams();
  const [catalog, setCatalog] = useState<LabCatalog | null>(null);
  const [catalogError, setCatalogError] = useState("");
  const [runs, setRuns] = useState<LabJob[]>([]);
  const [chosen, setChosen] = useState<Record<string,string>>({});
  const [job, setJob] = useState<LabJob | null>(null);
  const [config, setConfig] = useState(defaults);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [reload, setReload] = useState(0);
  const groupId = params.get("group") || "rating";
  const group = catalog?.groups.find(g => g.id === groupId) || catalog?.groups[0];
  const groupRuns = runs.filter(r => r.group === group?.id);
  const selectedId = group ? chosen[group.id] || groupRuns[0]?.id : undefined;
  const active = runs.find(busy);
  useEffect(() => {
    const controller = new AbortController();
    setCatalogError("");
    api<LabCatalog>("/experiment-lab/catalog", { signal: controller.signal }).then(setCatalog).catch(e => { if (!controller.signal.aborted) setCatalogError((e as Error).message); });
    return () => controller.abort();
  }, [reload]);
  useEffect(() => {
    const controller = new AbortController();
    let timer: number;
    const load = async () => {
      try { const data = await api<LabJob[]>("/experiment-lab/runs", { signal: controller.signal }); if (!controller.signal.aborted) setRuns(data); }
      catch (e) { if (!controller.signal.aborted) setError((e as Error).message); }
      if (!controller.signal.aborted) timer = window.setTimeout(load, 1800);
    };
    void load();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [reload]);
  useEffect(() => {
    setJob(null);
    if (!selectedId) return;
    const controller = new AbortController();
    let timer: number;
    const load = async () => {
      try {
        const data = await api<LabJob>(`/experiment-lab/runs/${selectedId}`, { signal: controller.signal });
        if (controller.signal.aborted) return;
        setJob(data);
        if (busy(data)) timer = window.setTimeout(load, 1000);
      } catch (e) { if (!controller.signal.aborted) setError((e as Error).message); }
    };
    void load();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [selectedId, reload]);
  useEffect(() => { setError(""); }, [group?.id]);
  async function start() {
    if (!group) return;
    setSubmitting(true); setError("");
    try {
      const response = await api<LabJob>("/experiment-lab/runs", { method: "POST", body: JSON.stringify({ ...config, group: group.id, repeats: ["association","pca"].includes(group.id) ? 1 : config.repeats }) });
      setRuns(v => [response,...v]); setChosen(v => ({ ...v, [group.id]: response.id })); setJob(response);
    } catch (e) { setError((e as Error).message); }
    finally { setSubmitting(false); }
  }
  async function stop(id: string) {
    try { await api(`/experiment-lab/runs/${id}/cancel`, { method: "POST" }); setReload(v => v+1); }
    catch (e) { setError((e as Error).message); }
  }
  const numberField = (key: Exclude<keyof typeof defaults,"cluster_object">, label: string, min: number, max: number, step=1) => <label key={key}>{t(label)}<input type="number" min={min} max={max} step={step} required value={config[key]} onChange={e => setConfig(v => ({ ...v, [key]: e.target.value === "" ? "" : Number(e.target.value) }))} /></label>;
  return <div className="lab-page">
    <PageHero title={t("THỰC NGHIỆM")} subtitle={t("Từ câu hỏi nghiên cứu đến bằng chứng thực nghiệm")} variant="experiments" />
    <main id="main" className="container lab-layout">
      <aside className="lab-sidebar" aria-label={t("Danh mục nhóm thuật toán")}>
        <div className="lab-sidebar-heading"><FlaskConical size={20} /><div><strong>{t("Phòng thực nghiệm")}</strong><span>MovieLens 25M</span></div></div><p className="lab-sidebar-label">{t("NHÓM THUẬT TOÁN")}</p>
        <nav>{catalog?.groups.map((g,i) => { const Icon = icons[i]; const done = runs.some(r => r.group === g.id && r.status === "completed"); return <button type="button" key={g.id} className={group?.id === g.id ? "selected" : ""} aria-current={group?.id === g.id ? "page" : undefined} onClick={() => { setParams({ group: g.id }); requestAnimationFrame(() => document.querySelector('.lab-content')?.scrollIntoView({ block: 'start', behavior: 'instant' })); }}><Icon size={18} /><span>{t(g.title)}<small>{g.algorithms.length} {t("thuật toán")}</small></span>{done ? <CheckCircle2 size={14} className="lab-nav-done" /> : <ChevronRight size={14} />}</button>; })}</nav>
        <div className="lab-sidebar-note"><b>06 {t("nhóm")} · 12 {t("thuật toán")}</b><p>{t("So sánh trong cùng nhiệm vụ, trên cùng dữ liệu và điều kiện đánh giá.")}</p></div><a className="lab-document" href={apiResourceUrl("/experiment-lab/document")}><BookOpen size={16} />{t("Tài liệu mô tả")}<Download size={14} /></a>
      </aside>
      <div className="lab-content">
        {catalogError ? <section className="lab-alert" role="alert"><AlertCircle /><div><h2>{t("Chưa tải được phần thực nghiệm")}</h2><p>{catalogError}</p><button className="lab-outline" onClick={() => setReload(v => v+1)}><RefreshCw size={15} />{t("Thử lại")}</button></div></section> : !group ? <div className="lab-loading" role="status"><Loader2 className="lab-spin" />{t("Đang tải danh mục thuật toán…")}</div> : <>
          <header className="lab-group-header"><span className="lab-kicker">EXPERIMENT / {String((catalog?.groups.findIndex(g => g.id === group.id) || 0)+1).padStart(2,"0")}</span><div><h1>{t(group.title)}</h1><span className="lab-badge">{group.algorithms.length} {t("thuật toán")}</span></div><p>{group.question}</p></header>
          <nav className="lab-jump-links" aria-label={t("Điều hướng trong nhóm")}><a href="#lab-metrics">{t("Tiêu chí so sánh")}</a><a href="#lab-run"><Play size={12} />{t("Chạy thực nghiệm")}</a>{job?.status === "completed" && <a href="#lab-results">{t("Kết quả thực nghiệm")}</a>}</nav>
          <div className="lab-algorithms">{group.algorithms.map((algorithm,i) => <article key={algorithm.id}><span className="lab-algorithm-number">{String(i+1).padStart(2,"0")}</span><h2>{algorithm.name}</h2><p>{algorithm.description}</p><div><span>{t("ĐẦU RA")}</span>{algorithm.output}</div></article>)}</div>
          <section id="lab-metrics" className="lab-section"><div className="lab-section-heading"><div><span className="lab-kicker">01 / {t("THIẾT KẾ ĐÁNH GIÁ")}</span><h2>{t("Tiêu chí so sánh")}</h2></div><span className="lab-source">{t("Theo tài liệu")} · {catalog?.source}</span></div><p className="lab-muted">{t("Mỗi tiêu chí gắn với một câu hỏi nghiên cứu. ↑ cao hơn tốt hơn; ↓ thấp hơn tốt hơn. Các tiêu chí mô tả cần được đọc trong bối cảnh.")}</p><CriteriaTable items={group.metrics} caption={`${t("Tiêu chí đánh giá")} · ${group.title}`} />
            <details className="lab-details"><summary>{t("Danh mục biểu đồ và câu hỏi phân tích")} · {group.charts.length}</summary><div className="lab-chart-plan">{group.charts.map((c,i) => <article key={c.name}><span>{String(i+1).padStart(2,"0")}</span><div><h3>{c.name}</h3><p>{c.axes}</p><p><b>{t("Câu hỏi:")}</b> {c.question}</p></div></article>)}</div></details>
            <details className="lab-details"><summary>{t("Tiêu chí chi phí tính toán dùng chung")}</summary><CriteriaTable items={catalog?.resources || []} caption={t("Thời gian, bộ nhớ và chi phí phục vụ")} /></details>
            {group.id === "rating" && <details className="lab-details"><summary>{t("Đánh giá bổ sung cho gợi ý top-N")}</summary><CriteriaTable items={catalog?.ranking_metrics || []} caption={t("Tiêu chí xếp hạng top-N")} /></details>}
          </section>
          <section id="lab-run" className="lab-section lab-run-panel" aria-label={t("Chạy thực nghiệm")}><div className="lab-section-heading"><div><span className="lab-kicker">02 / {t("THỰC HIỆN")}</span><h2>{t("Chạy thực nghiệm nhóm")}</h2></div><span className={`lab-data-status ${catalog?.data_ready ? "ready" : ""}`}><i />{t(catalog?.data_ready ? "Dữ liệu đã sẵn sàng" : "Chưa chuẩn bị dữ liệu")}</span></div><p className="lab-muted">{t("Chạy toàn bộ thuật toán trong nhóm và các baseline phù hợp. Lấy mẫu trọn lịch sử user từ dữ liệu đã chuẩn bị, chia lại theo thời gian 70/15/15; cấu hình và kết quả được lưu riêng cho từng lần chạy.")}</p>
            {group.id === "reinforcement" && <p className="lab-simulation-notice"><AlertCircle size={18} />{t("Môi trường mô phỏng từ train. Phần thưởng là ước lượng; kết quả không đại diện phản hồi thực của người dùng.")}</p>}
            <form onSubmit={e => { e.preventDefault(); void start(); }}><fieldset disabled={submitting || !!active} className="lab-config"><legend className="sr-only">{t("Cấu hình thực nghiệm")}</legend>{numberField("max_users", "Số user tối đa", 10, 1000)}{numberField("seed", "Seed dữ liệu / mô hình", 0, 1000000)}
              {!["association","pca"].includes(group.id) && numberField("repeats", "Số seed huấn luyện", 1, 3)}{group.id === "rating" && numberField("epochs", "Epochs SVD / NCF", 2, 30)}{["classification","association"].includes(group.id) && numberField("positive_threshold", "Ngưỡng rating thích", 1, 5, .5)}
              {group.id === "clustering" && <><label>{t("Đối tượng phân cụm")}<select value={config.cluster_object} onChange={e => setConfig(v => ({ ...v, cluster_object: e.target.value }))}><option value="movies">{t("Phim")}</option><option value="users">User</option></select></label>{numberField("n_clusters", "Số cụm K-Means", 2, 12)}{numberField("eps", "DBSCAN · eps", .1, 10, .1)}{numberField("min_samples", "DBSCAN · min_samples", 2, 30)}</>}
              {["clustering","association"].includes(group.id) && numberField("max_items", "Số phim tối đa", 10, 120)}{group.id === "association" && numberField("min_support", "min_support", .02, .5, .01)}{group.id === "reinforcement" && numberField("episodes", "Episode huấn luyện", 100, 5000, 100)}
            </fieldset><div className="lab-run-actions"><button className="lab-primary" type="submit" disabled={submitting || !!active || !catalog?.data_ready}>{submitting ? <Loader2 className="lab-spin" size={17} /> : <Play size={17} />}{t("Chạy thực nghiệm nhóm")}</button><span>{group.id === "association" ? t("Độ dài tập mục tối đa 2 · confidence ≥ 0,3") : group.id === "pca" ? t("So sánh các số chiều và baseline không PCA") : t("Chỉ tạo thanh sai số khi có nhiều seed thực sự được đo.")}</span></div></form>
            {error && <div className="lab-alert" role="alert"><AlertCircle size={18} /><p>{error}</p><button className="lab-outline" onClick={() => { setError(""); setReload(v => v+1); }}>{t("Thử lại")}</button></div>}
            {active && active.id !== job?.id && <div className="lab-active-other" role="status"><Loader2 className="lab-spin" size={16} /><p>{t("Đang chạy nhóm")}: {catalog?.groups.find(g => g.id === active.group)?.title}. {t("Hãy đợi hoàn tất hoặc dừng lần chạy này.")}</p><button className="lab-outline" onClick={() => void stop(active.id)}><Square size={13} />{t("Dừng")}</button></div>}
            {!!groupRuns.length && <label className="lab-history">{t("Lịch sử thực nghiệm")}<select value={selectedId || ""} onChange={e => setChosen(v => ({ ...v, [group.id]: e.target.value }))}>{groupRuns.map(r => <option key={r.id} value={r.id}>{new Date(r.created_at).toLocaleString("vi-VN")} · {t(statuses[r.status])} · {r.id.slice(-8)}</option>)}</select></label>}
            {job && <div className={`lab-job lab-job-${job.status}`} role="status" aria-live="polite"><div>{busy(job) ? <Loader2 size={20} className="lab-spin" /> : job.status === "completed" ? <CheckCircle2 size={20} /> : <AlertCircle size={20} />}<div><strong>{t(statuses[job.status])} {busy(job) ? `· ${job.progress}%` : ""}</strong><p>{job.message}</p><small>{job.id}</small></div>{busy(job) && <button className="lab-outline" disabled={job.cancel_requested} onClick={() => void stop(job.id)}><Square size={13} />{t(job.cancel_requested ? "Đang dừng…" : "Dừng")}</button>}</div>{busy(job) && <progress aria-label={t("Tiến độ thực nghiệm")} max={100} value={job.progress} />}</div>}
          </section>
          {job?.result && job.status === "completed" ? <ResultPanel key={job.id} result={job.result} job={job} group={group} /> : !job && <section className="lab-empty" role="status"><FlaskConical size={30} /><h2>{t("Chưa có kết quả thực nghiệm")}</h2><p>{t("Chọn cấu hình và chạy nhóm để tạo bảng so sánh cùng biểu đồ từ dữ liệu thực tế.")}</p></section>}
          <details className="lab-details lab-protocol"><summary>{t("Quy trình đánh giá và nguồn tham khảo")}</summary>{catalog?.protocol.map((text,i) => <p key={i}>{text}</p>)}<ol>{catalog?.references.map((url,i) => <li key={url}><a href={url} target="_blank" rel="noreferrer">[{i+1}] {new URL(url).hostname} ↗</a></li>)}</ol></details>
        </>}
      </div>
    </main>
  </div>;
}
