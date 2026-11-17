import { useEffect, useState } from "react";
import { AlertCircle, CheckCircle2, Clock3, Loader2, Square } from "lucide-react";
import { busy, type LabJob } from "../lib/lab";
import { usePreferences } from "../lib/preferences";

function elapsedLabel(milliseconds:number) {
  if(!Number.isFinite(milliseconds))return "--:--";
  const total=Math.max(0,Math.floor(milliseconds/1000));
  const seconds=String(total%60).padStart(2,"0");
  const minutes=String(Math.floor(total/60)%60).padStart(2,"0");
  return total>=3600?`${String(Math.floor(total/3600)).padStart(2,"0")}:${minutes}:${seconds}`:`${minutes}:${seconds}`;
}

export function ExperimentJobStatus({job,statusLabel,onCancel}:{job:LabJob;statusLabel:string;onCancel?:()=>void}) {
  const {t}=usePreferences();
  const active=busy(job);
  const [now,setNow]=useState(Date.now);
  useEffect(()=>{
    if(!active)return;
    const tick=()=>setNow(Date.now());
    tick();
    const timer=window.setInterval(tick,1000);
    document.addEventListener("visibilitychange",tick);
    return ()=>{window.clearInterval(timer);document.removeEventListener("visibilitychange",tick);};
  },[active]);
  const end=active?now:Date.parse(job.finished_at||"");
  const duration=end-Date.parse(job.created_at);
  const seconds=Number.isFinite(duration)?Math.max(0,Math.floor(duration/1000)):undefined;

  return <section className={`lab-job lab-job-${job.status}`} role="status" aria-live="polite" aria-label={t("Trạng thái thực nghiệm")}>
    <div className="lab-job-header">
      <span className="lab-job-icon">{active?<Loader2 size={21} className="lab-spin"/>:job.status==="completed"?<CheckCircle2 size={21}/>:<AlertCircle size={21}/>}</span>
      <div className="lab-job-copy"><strong>{statusLabel}{active?` · ${job.progress}%`:""}</strong><p>{job.message}</p></div>
      <div className="lab-job-timer" aria-live="off">
        <span><Clock3 size={13}/>{t(active?"Thời gian đã chạy":"Tổng thời gian")}</span>
        <time role="timer" aria-label={t("Thời gian thực nghiệm")} dateTime={seconds===undefined?undefined:`PT${seconds}S`}>{elapsedLabel(duration)}</time>
      </div>
      {active&&onCancel&&<button type="button" className="lab-outline" disabled={job.cancel_requested} onClick={onCancel}><Square size={13}/>{t("Dừng")}</button>}
    </div>
    {active&&<progress aria-label={t("Tiến độ thực nghiệm")} max={100} value={job.progress}/>}
  </section>;
}
