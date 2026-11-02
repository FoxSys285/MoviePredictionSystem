import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Activity, AlertCircle, CheckCircle2, ChevronRight, Download, FlaskConical, Layers, Loader2, Minimize2, Network, Play, RefreshCw, Square, Target } from "lucide-react";
import { PageHero } from "../components/ui";
import { LabChartFigure } from "../components/LabChart";
import { api, apiResourceUrl } from "../lib/api";
import { busy, metricNumber, type LabCatalog, type LabGroup, type LabJob } from "../lib/lab";
import { usePreferences } from "../lib/preferences";

const icons=[Activity,Target,Layers,Network,Minimize2];
const statuses: Record<string,string>={queued:"Đang chờ",running:"Đang chạy",completed:"Đã hoàn tất",failed:"Có lỗi",cancelled:"Đã dừng",interrupted:"Bị gián đoạn"};
const defaults={max_users:150,seed:42,epochs:8,positive_threshold:4,n_clusters:4,eps:1.5,min_samples:5,max_items:80,min_support:.08,cluster_object:"movies"};

function Results({job,group}:{job:LabJob;group:LabGroup}) {
  const {t,language}=usePreferences();
  const result=job.result!;
  const algorithms=[...new Map(result.rows.map(r=>[r.algorithm,r.name])).entries()];
  const lookup=new Map(result.rows.map(r=>[`${r.algorithm}/${r.split}`,r]));
  const charts=group.metrics.map(metric=>result.charts.find(c=>c.metric_key===metric.key)).filter(c=>c!==undefined);
  return <section id="lab-results" className="lab-section lab-results" aria-label={t("Kết quả thực nghiệm")}>
    <div className="lab-section-heading"><h2>{t("Kết quả Validation và Test")}</h2><div className="lab-downloads">{["csv","json"].map(format=><a key={format} className="lab-outline" href={apiResourceUrl(`/experiment-lab/runs/${job.id}/export/${format}`)}><Download size={14}/>{format.toUpperCase()}</a>)}</div></div>
    <p className="lab-muted">{result.provenance.sampled_users.toLocaleString("vi-VN")} user · {result.provenance.sampled_ratings.toLocaleString("vi-VN")} rating. N/A: {t("không đủ dữ liệu hoặc chỉ số không xác định.")}</p>
    <div className="lab-table-scroll lab-comparison"><table><caption>{t("Bảng so sánh kết quả")} · Validation &amp; Test</caption><thead>
      <tr><th rowSpan={2}>{t("Thuật toán / cấu hình")}</th>{["Validation","Test"].map(split=><th key={split} colSpan={group.metrics.length} className={`lab-split-label lab-split-${split.toLowerCase()}`}>{split}</th>)}</tr>
      <tr>{["validation","test"].flatMap(split=>group.metrics.map(metric=><th key={`${split}/${metric.key}`}>{metric.name}</th>))}</tr>
    </thead><tbody>{algorithms.map(([aid,name])=><tr key={aid}><th scope="row">{name}</th>{["validation","test"].flatMap(split=>group.metrics.map(metric=><td key={`${split}/${metric.key}`} data-split={split} data-metric={metric.key}>{metricNumber(lookup.get(`${aid}/${split}`)?.[metric.key],language)}</td>))}</tr>)}</tbody></table></div>
    {result.notes.slice(0,1).map(note=><p className="lab-muted lab-result-note" key={note}>{note}</p>)}
    <div className="lab-section-heading lab-figures-heading"><h2>{t("Biểu đồ so sánh")}</h2><span className="lab-badge">{group.metrics.length} {t("chỉ số")} · {group.metrics.length} {t("biểu đồ")}</span></div>
    <div className="lab-charts">{charts.map(c=><LabChartFigure key={c.metric_key} chart={c}/>)}</div>
  </section>;
}

export function Experiments() {
  const {t}=usePreferences();
  const [params,setParams]=useSearchParams();
  const [catalog,setCatalog]=useState<LabCatalog|null>(null),[catalogError,setCatalogError]=useState("");
  const [runs,setRuns]=useState<LabJob[]>([]),[chosen,setChosen]=useState<Record<string,string>>({});
  const [job,setJob]=useState<LabJob|null>(null),[config,setConfig]=useState(defaults);
  const [submitting,setSubmitting]=useState(false),[error,setError]=useState(""),[reload,setReload]=useState(0);
  const groupId=params.get("group")||"rating";
  const group=catalog?.groups.find(g=>g.id===groupId)||catalog?.groups[0];
  const groupRuns=runs.filter(r=>r.group===group?.id);
  const selectedId=group ? chosen[group.id]||groupRuns[0]?.id : undefined;
  const visibleJob=job?.group===group?.id ? job : null;
  const active=runs.find(busy);
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
    try {const response=await api<LabJob>("/experiment-lab/runs",{method:"POST",body:JSON.stringify({...config,group:group.id,repeats:1})});setRuns(v=>[response,...v]);setChosen(v=>({...v,[group.id]:response.id}));setJob(response);}
    catch(e){setError((e as Error).message);}finally{setSubmitting(false);}
  }
  async function stop(id:string){try {await api(`/experiment-lab/runs/${id}/cancel`,{method:"POST"});setReload(v=>v+1);}catch(e){setError((e as Error).message);}}
  const field=(key:Exclude<keyof typeof defaults,"cluster_object">,label:string,min:number,max:number,step=1)=><label key={key}>{t(label)}<input type="number" min={min} max={max} step={step} required value={config[key]} onChange={e=>setConfig(v=>({...v,[key]:e.target.value===""?"":Number(e.target.value)}))}/></label>;
  return <div className="lab-page"><PageHero title={t("THỰC NGHIỆM")} subtitle={t("So sánh thuật toán theo các chỉ số chính")} variant="experiments"/>
    <main id="main" className="container lab-layout"><aside className="lab-sidebar" aria-label={t("Danh mục nhóm thuật toán")}>
      <div className="lab-sidebar-heading"><FlaskConical size={20}/><div><strong>{t("Phòng thực nghiệm")}</strong><span>MovieLens 25M</span></div></div><p className="lab-sidebar-label">{t("NHÓM THUẬT TOÁN")}</p>
      <nav>{catalog?.groups.map((g,i)=>{const Icon=icons[i];return <button type="button" key={g.id} className={group?.id===g.id?"selected":""} aria-current={group?.id===g.id?"page":undefined} onClick={()=>{setParams({group:g.id});requestAnimationFrame(()=>document.querySelector(".lab-content")?.scrollIntoView({block:"start",behavior:"instant"}));}}><Icon size={18}/><span>{t(g.title)}<small>{g.algorithms.length} {t("thuật toán")} · {g.metrics.length} {t("chỉ số")}</small></span><ChevronRight size={14}/></button>;})}</nav>
    </aside><div className="lab-content">
      {catalogError?<section className="lab-alert" role="alert"><AlertCircle/><div><h2>{t("Chưa tải được phần thực nghiệm")}</h2><p>{catalogError}</p><button className="lab-outline" onClick={()=>setReload(v=>v+1)}><RefreshCw size={15}/>{t("Thử lại")}</button></div></section>:!group?<div className="lab-loading" role="status"><Loader2 className="lab-spin"/>{t("Đang tải danh mục thuật toán…")}</div>:<>
        <header className="lab-group-header"><span className="lab-kicker">EXPERIMENT</span><div><h1>{t(group.title)}</h1><span className="lab-badge">{group.metrics.length} {t("chỉ số")} · {group.metrics.length} {t("biểu đồ")}</span></div></header>
        <nav className="lab-jump-links" aria-label={t("Điều hướng trong nhóm")}><a href="#lab-run"><Play size={12}/>{t("Chạy thực nghiệm")}</a>{visibleJob?.status==="completed"&&<a href="#lab-results">{t("Kết quả thực nghiệm")}</a>}</nav>
        <div className="lab-algorithms">{group.algorithms.map(a=><article key={a.id}><h2>{a.name}</h2><p>{a.description}</p></article>)}</div>
        <section id="lab-metrics" className="lab-section"><h2>{t("Tiêu chí so sánh")}</h2><div className="lab-table-scroll lab-criteria"><table><caption>{t("Tiêu chí đánh giá")} · {group.title}</caption><thead><tr><th>{t("Tiêu chí")}</th><th>{t("Đánh giá gì / cách xác định")}</th><th>{t("Cách diễn giải")}</th><th>{t("Câu hỏi được trả lời")}</th></tr></thead><tbody>{group.metrics.map(m=><tr key={m.key}><th scope="row">{m.name}</th><td>{m.definition}</td><td>{m.interpretation}</td><td>{m.question}</td></tr>)}</tbody></table></div></section>
        <section id="lab-run" className="lab-section lab-run-panel" aria-label={t("Chạy thực nghiệm")}><div className="lab-section-heading"><h2>{t("Chạy thực nghiệm nhóm")}</h2><span className="lab-data-status"><i/>{t(catalog?.data_ready?"Dữ liệu đã sẵn sàng":"Chưa chuẩn bị dữ liệu")}</span></div>
          <form onSubmit={e=>{e.preventDefault();void start();}}><fieldset className="lab-config" disabled={submitting||!!active}><legend className="sr-only">{t("Cấu hình thực nghiệm")}</legend>{field("max_users","Số user tối đa",10,1000)}{field("seed","Seed dữ liệu / mô hình",0,1000000)}
            {group.id==="rating"&&field("epochs","Epochs SVD / NCF",2,30)}{["classification","association"].includes(group.id)&&field("positive_threshold","Ngưỡng rating thích",1,5,.5)}
            {group.id==="clustering"&&<><label>{t("Đối tượng phân cụm")}<select value={config.cluster_object} onChange={e=>setConfig(v=>({...v,cluster_object:e.target.value}))}><option value="movies">{t("Phim")}</option><option value="users">User</option></select></label>{field("n_clusters","Số cụm K-Means",2,12)}{field("eps","DBSCAN · eps",.1,10,.1)}{field("min_samples","DBSCAN · min_samples",2,30)}</>}
            {["clustering","association"].includes(group.id)&&field("max_items","Số phim tối đa",10,120)}{group.id==="association"&&field("min_support","min_support",.02,.5,.01)}
          </fieldset><div className="lab-run-actions"><button className="lab-primary" type="submit" disabled={submitting||!!active||!catalog?.data_ready}><Play size={17}/>{t("Chạy thực nghiệm nhóm")}</button></div></form>
          {error&&<div className="lab-alert" role="alert"><AlertCircle size={18}/><p>{error}</p><button className="lab-outline" onClick={()=>{setError("");setReload(v=>v+1);}}>{t("Thử lại")}</button></div>}
          {active&&active.id!==visibleJob?.id&&<div className="lab-active-other" role="status"><Loader2 size={16} className="lab-spin"/><p>{t("Đang chạy nhóm")}: {catalog?.groups.find(g=>g.id===active.group)?.title}</p><button className="lab-outline" onClick={()=>void stop(active.id)}>{t("Dừng")}</button></div>}
          {!!groupRuns.length&&<label className="lab-history">{t("Lịch sử thực nghiệm")}<select value={selectedId||""} onChange={e=>setChosen(v=>({...v,[group.id]:e.target.value}))}>{groupRuns.map(r=><option key={r.id} value={r.id}>{new Date(r.created_at).toLocaleString("vi-VN")} · {t(statuses[r.status])}</option>)}</select></label>}
          {visibleJob&&<div className={`lab-job lab-job-${visibleJob.status}`} role="status" aria-live="polite"><div>{busy(visibleJob)?<Loader2 size={20} className="lab-spin"/>:visibleJob.status==="completed"?<CheckCircle2 size={20}/>:<AlertCircle size={20}/>}<div><strong>{t(statuses[visibleJob.status])}{busy(visibleJob)?` · ${visibleJob.progress}%`:""}</strong><p>{visibleJob.message}</p></div>{busy(visibleJob)&&<button className="lab-outline" disabled={visibleJob.cancel_requested} onClick={()=>void stop(visibleJob.id)}><Square size={13}/>{t("Dừng")}</button>}</div>{busy(visibleJob)&&<progress aria-label={t("Tiến độ thực nghiệm")} max={100} value={visibleJob.progress}/>}</div>}
        </section>
        {visibleJob?.status==="completed"&&visibleJob.result?.schema_version===2?<Results key={visibleJob.id} job={visibleJob} group={group}/>:!visibleJob&&<section className="lab-empty" role="status"><FlaskConical size={28}/><h2>{t("Chưa có kết quả thực nghiệm")}</h2><p>{t("Chạy nhóm để xem Validation và Test cùng lúc.")}</p></section>}
      </>}
    </div></main>
  </div>;
}
