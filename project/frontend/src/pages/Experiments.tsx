import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Activity, AlertCircle, ChevronRight, FlaskConical, Loader2, Play, RefreshCw } from "lucide-react";
import { PageHero } from "../components/ui";
import { SimpleExperimentResults, DatasetSummary } from "../components/ExperimentResults";
import { AlgorithmSelection } from "../components/ExperimentConfig";
import { ExperimentLog } from "../components/ExperimentLog";
import { ExperimentJobStatus } from "../components/ExperimentJobStatus";
import { ExperimentMetricGuide } from "../components/ExperimentMetricGuide";
import { ExperimentConfigurationSummary } from "../components/ExperimentConfigurationSummary";
import { api } from "../lib/api";
import { ExperimentSuccessSound } from "../lib/experiment-sound";
import { busy, type LabCatalog, type LabJob } from "../lib/lab";
import { runAlgorithmTitle } from "../lib/experiment-runs";
import { usePreferences } from "../lib/preferences";

type ReductionChoice = "off" | "pca";

const statuses: Record<string,string>={queued:"Đang chờ",running:"Đang chạy",completed:"Đã hoàn tất",failed:"Có lỗi",cancelled:"Đã dừng",interrupted:"Bị gián đoạn"};

export function Experiments() {
  const {t}=usePreferences();
  const [successSound]=useState(()=>new ExperimentSuccessSound());
  const [params,setParams]=useSearchParams();
  const [catalog,setCatalog]=useState<LabCatalog|null>(null),[catalogError,setCatalogError]=useState("");
  const [runs,setRuns]=useState<LabJob[]>([]),[chosen,setChosen]=useState<Record<string,string>>({});
  const [job,setJob]=useState<LabJob|null>(null);
  const [algorithmChoices,setAlgorithmChoices]=useState<Record<string,string[]>>({});
  const [reductionChoices,setReductionChoices]=useState<Record<string,ReductionChoice>>({});
  const [submitting,setSubmitting]=useState(false),[error,setError]=useState(""),[reload,setReload]=useState(0);
  const [submissionStartedAt,setSubmissionStartedAt]=useState<string|null>(null);
  const group=catalog?.groups.find(g=>g.id==="rating");
  const groupRuns=runs.filter(r=>r.group===group?.id);
  const selectedId=group ? chosen[group.id]||groupRuns[0]?.id : undefined;
  const visibleJob=job?.group===group?.id ? job : null;
  const active=runs.find(busy);
  const requestedAlgorithmId=params.get("algorithm");
  const requestedAlgorithm=group?.algorithms.find(algorithm=>algorithm.id===requestedAlgorithmId);
  const algorithms=group ? algorithmChoices[group.id]??[requestedAlgorithm?.id||group.algorithms[0]?.id].filter((id):id is string=>!!id) : [];
  const activeAlgorithmId=requestedAlgorithm?.id||(algorithms.length===1?algorithms[0]:undefined);
  const selectedAlgorithm=group?.algorithms.find(algorithm=>algorithm.id===activeAlgorithmId);
  const algorithmIndex=Math.max(0,group?.algorithms.findIndex(algorithm=>algorithm.id===activeAlgorithmId)??0);
  const basePreset=group ? catalog?.presets?.[group.id] : undefined;
  const reductionChoice:ReductionChoice=(group && reductionChoices[group.id]) || "off";
  const preset=basePreset ? {...basePreset,algorithms:Object.fromEntries(
    Object.entries(basePreset.algorithms).map(([id,config])=>{
      const eligible=group?.id==="rating" && id==="linear";
      const reduction=eligible && reductionChoice!=="off" ? {method:reductionChoice,components:8} : null;
      const settings=config.settings.map(setting=>setting.label==="Biểu diễn" && eligible
        ? {...setting,value:reduction ? `PCA · 8 chiều` : "Gốc · không giảm chiều"}
        : setting);
      return [id,{...config,settings,reduction}];
    })
  )} : undefined;
  const statusJob:LabJob|null=submitting&&group&&submissionStartedAt?{
    id:"submitting",group:group.id,status:"queued",progress:0,
    message:"Đang gửi yêu cầu thực nghiệm…",created_at:submissionStartedAt,
  }:visibleJob;
  useEffect(()=>{
    const arm=()=>successSound.arm();
    window.addEventListener("pointerdown",arm,{capture:true,once:true});
    window.addEventListener("keydown",arm,{capture:true,once:true});
    return ()=>{
      window.removeEventListener("pointerdown",arm,true);
      window.removeEventListener("keydown",arm,true);
      successSound.close();
    };
  },[successSound]);
  useEffect(()=>{
    const c=new AbortController();setCatalogError("");
    api<LabCatalog>("/experiment-lab/catalog",{signal:c.signal}).then(data=>setCatalog({...data,groups:data.groups.filter(group=>group.id==="rating")})).catch(e=>{if(!c.signal.aborted)setCatalogError((e as Error).message);});
    return ()=>c.abort();
  },[reload]);
  useEffect(()=>{
    const c=new AbortController();let timer:number;
    const load=async()=>{
      try {const data=await api<LabJob[]>("/experiment-lab/runs",{signal:c.signal});if(!c.signal.aborted){successSound.observe(data);setRuns(data);}}
      catch(e){if(!c.signal.aborted)setError((e as Error).message);}
      if(!c.signal.aborted)timer=window.setTimeout(load,1800);
    };
    void load();return ()=>{c.abort();window.clearTimeout(timer);};
  },[reload,successSound]);
  useEffect(()=>{
    setJob(null);if(!selectedId)return;
    const c=new AbortController();let timer:number;
    const load=async()=>{
      try {const data=await api<LabJob>(`/experiment-lab/runs/${selectedId}`,{signal:c.signal});if(c.signal.aborted)return;successSound.observe([data]);setJob(data);if(busy(data))timer=window.setTimeout(load,1000);}
      catch(e){if(!c.signal.aborted)setError((e as Error).message);}
    };
    void load();return ()=>{c.abort();window.clearTimeout(timer);};
  },[selectedId,reload,successSound]);
  useEffect(()=>{setError("");},[group?.id]);
  async function start(){
    if(!group)return;setSubmissionStartedAt(new Date().toISOString());setSubmitting(true);setError("");
    successSound.arm();
    try {
      if(!preset)throw new Error("Chưa tải được preset chuẩn cho nhóm này.");
      const effectiveReduction=algorithms.includes("linear")?reductionChoice:"off";
      const response=await api<LabJob>("/experiment-lab/runs",{method:"POST",body:JSON.stringify({group:group.id,algorithms,preset_id:preset.id,reduction_choice:effectiveReduction})});
      successSound.track(response.id);setRuns(v=>[response,...v]);setChosen(v=>({...v,[group.id]:response.id}));setJob(response);
    }
    catch(e){setError((e as Error).message);}finally{setSubmitting(false);setSubmissionStartedAt(null);}
  }
  async function resumeRun(id:string){
    setSubmitting(true);setError("");
    try{const response=await api<LabJob>(`/experiment-lab/runs/${id}/resume`,{method:"POST"});successSound.track(response.id);setRuns(v=>[response,...v.filter(r=>r.id!==response.id)]);setChosen(v=>({...v,[response.group]:response.id}));setJob(response);}
    catch(e){setError((e as Error).message);setReload(v=>v+1);}finally{setSubmitting(false);}
  }
  async function stop(id:string){try {await api(`/experiment-lab/runs/${id}/cancel`,{method:"POST"});setReload(v=>v+1);}catch(e){setError((e as Error).message);}}
  return <div className="lab-page"><PageHero title={t("THỰC NGHIỆM")} subtitle="MovieLens Research Studio" variant="experiments"/>
    <main id="main" className="container lab-layout"><aside className="lab-sidebar" aria-label={t("Danh sách thuật toán")}>
      <div className="lab-sidebar-heading"><FlaskConical size={20}/><div><strong>{t("Phòng thực nghiệm")}</strong><span>MovieLens 25M</span></div></div><p className="lab-sidebar-label">{t("THUẬT TOÁN")}</p>
      <nav aria-label={t("Danh sách thuật toán hồi quy")}>{group?.algorithms.map(algorithm=><button type="button" key={algorithm.id} data-algorithm-id={algorithm.id} className={activeAlgorithmId===algorithm.id?"selected":""} aria-current={activeAlgorithmId===algorithm.id?"page":undefined} onClick={()=>{if(!group)return;setAlgorithmChoices(previous=>({...previous,[group.id]:[algorithm.id]}));setParams({group:group.id,algorithm:algorithm.id});requestAnimationFrame(()=>document.querySelector(".lab-content")?.scrollIntoView({block:"start",behavior:"instant"}));}}><Activity size={18}/><span>{t(algorithm.name)}</span><ChevronRight size={14}/></button>)}</nav>
    </aside><div className="lab-content">
      {catalogError?<section className="lab-alert" role="alert"><AlertCircle/><div><h2>{t("Chưa tải được phần thực nghiệm")}</h2><p>{catalogError}</p><button className="lab-outline" onClick={()=>setReload(v=>v+1)}><RefreshCw size={15}/>{t("Thử lại")}</button></div></section>:!group?<div className="lab-loading" role="status"><Loader2 className="lab-spin"/>{t("Đang tải danh mục thuật toán…")}</div>:<>
        <header className="lab-group-header"><span className="lab-kicker">REGRESSION / {String(algorithmIndex+1).padStart(2,"0")}</span><div><h1>{t(selectedAlgorithm?.name||group.title)}</h1><span className="lab-badge">{t(group.title)}</span></div><p className="lab-objective">{selectedAlgorithm?.description||group.question}</p></header>
        <DatasetSummary stats={catalog?.statistics} label="Dataset đã chuẩn bị"/>
        <nav className="lab-jump-links" aria-label={t("Điều hướng trong thuật toán")}><a href="#lab-run"><Play size={12}/>{t("Chạy thực nghiệm")}</a>{visibleJob?.status==="completed"&&<a href="#lab-results">{t("Kết quả thực nghiệm")}</a>}</nav>
        <AlgorithmSelection group={group} selected={algorithms} disabled={submitting||!!active}
          reductionEnabled={reductionChoice!=="off"}
          onReductionChange={enabled=>setReductionChoices(previous=>({...previous,[group.id]:enabled?"pca":"off"}))}
          onChange={value=>{setAlgorithmChoices(previous=>({...previous,[group.id]:value}));const nextActive=activeAlgorithmId&&value.includes(activeAlgorithmId)?activeAlgorithmId:value[0];setParams(nextActive?{group:group.id,algorithm:nextActive}:{group:group.id});}}/>
        <ExperimentMetricGuide/>
        <ExperimentConfigurationSummary group={group} preset={preset} algorithms={algorithms}/>
        <section id="lab-run" className="lab-section lab-run-panel" aria-label={t("Chạy thực nghiệm")}><div className="lab-section-heading"><h2>{t("Chạy thực nghiệm")}</h2><span className="lab-data-status"><i/>{t(catalog?.data_ready?"Dữ liệu đã sẵn sàng":"Chưa chuẩn bị dữ liệu")}</span></div>
          <form onSubmit={e=>{e.preventDefault();void start();}}>
            <div className="lab-run-actions"><button className="lab-primary" type="submit" disabled={submitting||!!active||!catalog?.data_ready||!preset||!algorithms.length}><Play size={17}/>{algorithms.length===1?t("Chạy thuật toán đã chọn"):t("Chạy {count} thuật toán",{count:String(algorithms.length)})}</button><span className="lab-muted">{algorithms.length} thuật toán được chọn</span></div>
          </form>
          {error&&<div className="lab-alert" role="alert"><AlertCircle size={18}/><p>{error}</p><button className="lab-outline" onClick={()=>{setError("");setReload(v=>v+1);}}>{t("Thử lại")}</button></div>}
          {active&&active.id!==visibleJob?.id&&<div className="lab-active-other" role="status"><Loader2 size={16} className="lab-spin"/><p>{t("Đang có thực nghiệm khác chạy.")}</p><button className="lab-outline" onClick={()=>void stop(active.id)}>{t("Dừng")}</button></div>}
          {!!groupRuns.length&&<label className="lab-history">{t("Lịch sử thực nghiệm")}<select value={selectedId||""} onChange={e=>setChosen(v=>({...v,[group.id]:e.target.value}))}>{groupRuns.map(r=><option key={r.id} value={r.id}>{new Date(r.created_at).toLocaleString("vi-VN")} · {runAlgorithmTitle(r,group,t)} · {t(statuses[r.status])}</option>)}</select></label>}
          {statusJob&&<ExperimentJobStatus key={`status-${statusJob.id}`} job={statusJob} statusLabel={t(statuses[statusJob.status])} onCancel={!submitting?()=>void stop(statusJob.id):undefined}/>}
          {!active&&visibleJob?.resume_available&&<div className="lab-run-actions"><button className="lab-primary" type="button" disabled={submitting} onClick={()=>void resumeRun(visibleJob.id)}><Play size={16}/>{t(visibleJob.epoch_checkpoint_available?"Tiếp tục từ checkpoint":"Chạy tiếp phần còn lại")}</button><span className="lab-muted">{t("Đã lưu {count} đơn vị công việc{epoch}; báo cáo hoàn tất chỉ xuất hiện sau khi chạy xong.",{count:String(visibleJob.completed_work_units||0),epoch:visibleJob.epoch_checkpoint_available?" và checkpoint epoch":""})}</span></div>}
          {!submitting&&visibleJob&&<ExperimentLog key={visibleJob.id} job={visibleJob}/>}
        </section>
        <SimpleExperimentResults job={visibleJob} group={group}/>
      </>}
    </div></main>
  </div>;
}
