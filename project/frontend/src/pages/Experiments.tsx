import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Activity, AlertCircle, CheckCircle2, ChevronRight, Download, FlaskConical, Layers, Loader2, Minimize2, Network, Play, RefreshCw, Square, Target } from "lucide-react";
import { PageHero } from "../components/ui";
import { LabChartFigure } from "../components/LabChart";
import { ExperimentResults, DatasetSummary } from "../components/ExperimentResults";
import { ExperimentConfiguration, experimentDefaults } from "../components/ExperimentConfig";
import { MetricReferenceTable, useMetricReferenceSettings } from "../components/MetricReferences";
import { api, apiResourceUrl } from "../lib/api";
import { busy, metricNumber, type LabCatalog, type LabGroup, type LabJob } from "../lib/lab";
import { usePreferences } from "../lib/preferences";

const icons=[Activity,Target,Layers,Network,Minimize2,FlaskConical];
const statuses: Record<string,string>={queued:"Đang chờ",running:"Đang chạy",completed:"Đã hoàn tất",failed:"Có lỗi",cancelled:"Đã dừng",interrupted:"Bị gián đoạn"};

function Results({job,group}:{job:LabJob;group:LabGroup}) {
  const {t,language}=usePreferences();
  const result=job.result!;
  const historicMetrics=result.charts.filter(c=>c.metric_key).map(c=>({key:c.metric_key!,name:c.title,definition:"",interpretation:"",question:""}));
  if(historicMetrics.length)group={...group,metrics:historicMetrics};
  const algorithms=[...new Map(result.rows.map(r=>[r.algorithm,r.name])).entries()];
  const lookup=new Map(result.rows.map(r=>[`${r.algorithm}/${r.split}`,r]));
  const charts=group.metrics.map(metric=>result.charts.find(c=>c.metric_key===metric.key)).filter(c=>c!==undefined);
  return <section id="lab-results" className="lab-section lab-results" aria-label={t("Kết quả thực nghiệm")}>
    <div className="lab-section-heading"><h2>{t("Kết quả Validation và Test")}</h2><div className="lab-downloads">{["csv","json"].map(format=><a key={format} className="lab-outline" href={apiResourceUrl(`/experiment-lab/runs/${job.id}/export/${format}`)}><Download size={14}/>{format.toUpperCase()}</a>)}</div></div>
    <p className="lab-muted">{result.provenance.sampled_users.toLocaleString("vi-VN")} user · {result.provenance.sampled_ratings.toLocaleString("vi-VN")} rating. N/A: {t("không đủ dữ liệu hoặc chỉ số không xác định.")}</p>
    {["rating","classification"].includes(group.id)&&<p className="lab-muted">{result.provenance.protocol==="per_user_temporal_v1"?"Giao thức: chia thời gian trong từng user (70/15/15).":"Kết quả cũ: chia thời gian toàn cục; chỉ số phân loại dùng lớp thích. Chạy lại để áp dụng giao thức mới."}</p>}
    <div className="lab-table-scroll lab-comparison"><table><caption>{t("Bảng so sánh kết quả")} · Validation &amp; Test</caption><thead>
      <tr><th rowSpan={2}>{t("Thuật toán / cấu hình")}</th>{["Validation","Test"].map(split=><th key={split} colSpan={group.metrics.length} className={`lab-split-label lab-split-${split.toLowerCase()}`}>{split}</th>)}</tr>
      <tr>{["validation","test"].flatMap(split=>group.metrics.map(metric=><th key={`${split}/${metric.key}`}>{metric.name}</th>))}</tr>
    </thead><tbody>{algorithms.map(([aid,name])=><tr key={aid}><th scope="row">{name}</th>{["validation","test"].flatMap(split=>group.metrics.map(metric=><td key={`${split}/${metric.key}`} data-split={split} data-metric={metric.key}>{metricNumber(lookup.get(`${aid}/${split}`)?.[metric.key],language)}</td>))}</tr>)}</tbody></table></div>
    {result.notes.map(note=><p className="lab-muted lab-result-note" key={note}>{note}</p>)}
    <div className="lab-section-heading lab-figures-heading"><h2>{t("Biểu đồ so sánh")}</h2><span className="lab-badge">{group.metrics.length} {t("chỉ số")} · {group.metrics.length} {t("biểu đồ")}</span></div>
    <div className="lab-charts">{charts.map(c=><LabChartFigure key={c.metric_key} chart={c}/>)}</div>
  </section>;
}

export function Experiments() {
  const {t}=usePreferences();
  const [params,setParams]=useSearchParams();
  const [catalog,setCatalog]=useState<LabCatalog|null>(null),[catalogError,setCatalogError]=useState("");
  const [runs,setRuns]=useState<LabJob[]>([]),[chosen,setChosen]=useState<Record<string,string>>({});
  const [job,setJob]=useState<LabJob|null>(null),[config,setConfig]=useState(experimentDefaults);
  const [algorithmChoices,setAlgorithmChoices]=useState<Record<string,string[]>>({});
  const referenceSettings=useMetricReferenceSettings();
  const [submitting,setSubmitting]=useState(false),[error,setError]=useState(""),[reload,setReload]=useState(0);
  const groupId=params.get("group")||"rating";
  const group=catalog?.groups.find(g=>g.id===groupId)||catalog?.groups[0];
  const groupRuns=runs.filter(r=>r.group===group?.id);
  const selectedId=group ? chosen[group.id]||groupRuns[0]?.id : undefined;
  const visibleJob=job?.group===group?.id ? job : null;
  const active=runs.find(busy);
  const algorithms=group ? algorithmChoices[group.id]??group.algorithms.map(a=>a.id) : [];
  useEffect(()=>{
    const c=new AbortController();setCatalogError("");
    api<LabCatalog>("/experiment-lab/catalog",{signal:c.signal}).then(setCatalog).catch(e=>{if(!c.signal.aborted)setCatalogError((e as Error).message);});
    return ()=>c.abort();
  },[reload]);
  useEffect(()=>{
    const c=new AbortController();let timer:number;
    const load=async()=>{
      try {const data=await api<LabJob[]>("/experiment-lab/runs",{signal:c.signal});if(!c.signal.aborted)setRuns(data);}
      catch(e){if(!c.signal.aborted)setError((e as Error).message);}
      if(!c.signal.aborted)timer=window.setTimeout(load,1800);
    };
    void load();return ()=>{c.abort();window.clearTimeout(timer);};
  },[reload]);
  useEffect(()=>{
    setJob(null);if(!selectedId)return;
    const c=new AbortController();let timer:number;
    const load=async()=>{
      try {const data=await api<LabJob>(`/experiment-lab/runs/${selectedId}`,{signal:c.signal});if(c.signal.aborted)return;setJob(data);if(busy(data))timer=window.setTimeout(load,1000);}
      catch(e){if(!c.signal.aborted)setError((e as Error).message);}
    };
    void load();return ()=>{c.abort();window.clearTimeout(timer);};
  },[selectedId,reload]);
  useEffect(()=>{setError("");},[group?.id]);
  async function start(){
    if(!group)return;setSubmitting(true);setError("");
    try {const response=await api<LabJob>("/experiment-lab/runs",{method:"POST",body:JSON.stringify({...config,group:group.id,algorithms})});setRuns(v=>[response,...v]);setChosen(v=>({...v,[group.id]:response.id}));setJob(response);}
    catch(e){setError((e as Error).message);}finally{setSubmitting(false);}
  }
  async function stop(id:string){try {await api(`/experiment-lab/runs/${id}/cancel`,{method:"POST"});setReload(v=>v+1);}catch(e){setError((e as Error).message);}}
  return <div className="lab-page"><PageHero title={t("THỰC NGHIỆM")} subtitle="MovieLens Research Studio · Huấn luyện thật. Đánh giá minh bạch." variant="experiments"/>
    <main id="main" className="container lab-layout"><aside className="lab-sidebar" aria-label={t("Danh mục nhóm thuật toán")}>
      <div className="lab-sidebar-heading"><FlaskConical size={20}/><div><strong>{t("Phòng thực nghiệm")}</strong><span>MovieLens 25M</span></div></div><p className="lab-sidebar-label">{t("NHÓM THUẬT TOÁN")}</p>
      <nav>{catalog?.groups.map((g,i)=>{const Icon=icons[i];return <button type="button" key={g.id} className={group?.id===g.id?"selected":""} aria-current={group?.id===g.id?"page":undefined} onClick={()=>{setParams({group:g.id});requestAnimationFrame(()=>document.querySelector(".lab-content")?.scrollIntoView({block:"start",behavior:"instant"}));}}><Icon size={18}/><span>{t(g.title)}<small>{g.algorithms.length} {t("thuật toán")} · {g.metrics.length} {t("chỉ số")}</small></span><ChevronRight size={14}/></button>;})}</nav>
    </aside><div className="lab-content">
      {catalogError?<section className="lab-alert" role="alert"><AlertCircle/><div><h2>{t("Chưa tải được phần thực nghiệm")}</h2><p>{catalogError}</p><button className="lab-outline" onClick={()=>setReload(v=>v+1)}><RefreshCw size={15}/>{t("Thử lại")}</button></div></section>:!group?<div className="lab-loading" role="status"><Loader2 className="lab-spin"/>{t("Đang tải danh mục thuật toán…")}</div>:<>
        <header className="lab-group-header"><span className="lab-kicker">RESEARCH WORKSPACE / {String(catalog!.groups.indexOf(group)+1).padStart(2,"0")}</span><div><h1>{t(group.title)}</h1><span className="lab-badge">{group.algorithms.length} thuật toán</span></div><p className="lab-objective">{group.question}</p></header>
        <div className="lab-protocol-strip"><span><b>42</b> Random Seed</span><span><b>70 / 15 / 15</b> Train / Validation / Test</span><span><b>{["rating","classification"].includes(group.id)?"5-Fold CV":"Train only"}</b> {["rating","classification"].includes(group.id)?"Đặc trưng riêng từng fold":"Fit đặc trưng"}</span></div>
        <DatasetSummary stats={catalog?.statistics} label="Dataset đã chuẩn bị"/>
        <nav className="lab-jump-links" aria-label={t("Điều hướng trong nhóm")}><a href="#lab-run"><Play size={12}/>{t("Chạy thực nghiệm")}</a>{visibleJob?.status==="completed"&&<a href="#lab-results">{t("Kết quả thực nghiệm")}</a>}</nav>
        <div className="lab-algorithms">{group.algorithms.map(a=><article key={a.id}><h2>{a.name}</h2><p>{a.description}</p></article>)}</div>
        <MetricReferenceTable group={group} values={referenceSettings.settings[group.id]||{}} onChange={(key,value)=>referenceSettings.setValue(group.id,key,value)} onReset={()=>referenceSettings.reset(group.id)}/>
        <section id="lab-run" className="lab-section lab-run-panel" aria-label={t("Chạy thực nghiệm")}><div className="lab-section-heading"><h2>{t("Chạy thực nghiệm nhóm")}</h2><span className="lab-data-status"><i/>{t(catalog?.data_ready?"Dữ liệu đã sẵn sàng":"Chưa chuẩn bị dữ liệu")}</span></div>
          <form onInvalidCapture={e=>{const details=(e.target as HTMLElement).closest("details");if(details)details.open=true;}} onSubmit={e=>{e.preventDefault();void start();}}><ExperimentConfiguration key={group.id} group={group} value={config} onChange={setConfig} disabled={submitting||!!active} algorithms={algorithms} setAlgorithms={v=>setAlgorithmChoices(previous=>({...previous,[group.id]:v}))}/><div className="lab-run-actions"><button className="lab-primary" type="submit" disabled={submitting||!!active||!catalog?.data_ready||!algorithms.length}><Play size={17}/>{algorithms.length===1?"Chạy thuật toán đã chọn":t("Chạy thực nghiệm nhóm")}</button><span className="lab-muted">{algorithms.length} thuật toán được chọn</span></div></form>
          {error&&<div className="lab-alert" role="alert"><AlertCircle size={18}/><p>{error}</p><button className="lab-outline" onClick={()=>{setError("");setReload(v=>v+1);}}>{t("Thử lại")}</button></div>}
          {active&&active.id!==visibleJob?.id&&<div className="lab-active-other" role="status"><Loader2 size={16} className="lab-spin"/><p>{t("Đang chạy nhóm")}: {catalog?.groups.find(g=>g.id===active.group)?.title}</p><button className="lab-outline" onClick={()=>void stop(active.id)}>{t("Dừng")}</button></div>}
          {!!groupRuns.length&&<label className="lab-history">{t("Lịch sử thực nghiệm")}<select value={selectedId||""} onChange={e=>setChosen(v=>({...v,[group.id]:e.target.value}))}>{groupRuns.map(r=><option key={r.id} value={r.id}>{new Date(r.created_at).toLocaleString("vi-VN")} · {t(statuses[r.status])}</option>)}</select></label>}
          {visibleJob&&<div className={`lab-job lab-job-${visibleJob.status}`} role="status" aria-live="polite"><div>{busy(visibleJob)?<Loader2 size={20} className="lab-spin"/>:visibleJob.status==="completed"?<CheckCircle2 size={20}/>:<AlertCircle size={20}/>}<div><strong>{t(statuses[visibleJob.status])}{busy(visibleJob)?` · ${visibleJob.progress}%`:""}</strong><p>{visibleJob.message}</p></div>{busy(visibleJob)&&<button className="lab-outline" disabled={visibleJob.cancel_requested} onClick={()=>void stop(visibleJob.id)}><Square size={13}/>{t("Dừng")}</button>}</div>{busy(visibleJob)&&<progress aria-label={t("Tiến độ thực nghiệm")} max={100} value={visibleJob.progress}/>}</div>}
        </section>
        {visibleJob?.status==="completed"&&visibleJob.result?.schema_version===3?<ExperimentResults key={visibleJob.id} job={visibleJob} group={group} references={referenceSettings.settings[group.id]||{}}/>:visibleJob?.status==="completed"&&visibleJob.result?.schema_version===2?<><p className="lab-alert">Kết quả lịch sử dùng giao thức cũ. Chạy lại để áp dụng pipeline mới và 5-Fold CV.</p><Results key={visibleJob.id} job={visibleJob} group={group}/></>:!visibleJob&&<section className="lab-empty" role="status"><FlaskConical size={28}/><h2>{t("Chưa có kết quả thực nghiệm")}</h2><p>Chọn thuật toán và chạy để tạo báo cáo từ dữ liệu MovieLens thật.</p></section>}
      </>}
    </div></main>
  </div>;
}
