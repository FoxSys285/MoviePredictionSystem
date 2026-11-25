import { useEffect, useMemo, useState } from "react";
import { AlertCircle, GitCompare, Loader2 } from "lucide-react";
import { api } from "../lib/api";
import { metricNumber, type LabGroup, type LabJob, type LabRow } from "../lib/lab";
import { runAlgorithmIds, runAlgorithmTitle } from "../lib/experiment-runs";
import { usePreferences } from "../lib/preferences";

type MetricSpec={key:string;name:string;split:string;splitName:string;recommendation?:boolean};
const EMPTY_RUN_IDS:string[]=[];

function metricsFor(group:LabGroup):MetricSpec[] {
  const splits:[string,string][]=group.id==="rating"||group.id==="classification"||group.id==="reinforcement"
    ? [["validation","Validation"],["test","Test"]]
    : group.id==="clustering" ? [["train_history","Train"]]
      : [["train","Train"]];
  const specs:MetricSpec[]=group.metrics.flatMap(metric=>splits.map(([split,splitName])=>({key:metric.key,name:metric.name,split,splitName})));
  if(group.id==="clustering"||group.id==="association"){
    const recommendationMetrics:[string,string][]=[["ndcg_at_10","NDCG@10"],["recall_at_10","Recall@10"]];
    const recommendationSplits:[string,string][]=[["validation","Validation"],["test","Test"]];
    for(const [key,name] of recommendationMetrics){
      for(const [split,splitName] of recommendationSplits)specs.push({key,name,split,splitName,recommendation:true});
    }
  }
  return specs;
}

function identity(row:LabRow):string {
  return String(row.variant_id||row.algorithm||"");
}

function algorithmRows(job:LabJob,algorithmId:string):LabRow[] {
  const rows=(job.result?.rows??[]).filter(row=>String(row.base_algorithm||row.algorithm)===algorithmId);
  if(!rows.length)return [];
  const selected=String(job.result?.selected_variant_id||job.result?.reduction?.selected_variant_id||"");
  if(selected){
    const selectedRows=rows.filter(row=>identity(row)===selected);
    if(selectedRows.length)return selectedRows;
  }
  const raw=rows.filter(row=>!row.reduction_method||row.representation==="raw"||row.representation==="global_transactions");
  return raw.length?raw:rows;
}

function valueFor(job:LabJob,algorithmId:string,spec:MetricSpec):unknown {
  const result=job.result;
  if(!result)return undefined;
  const table:Record<string,unknown>[]=spec.recommendation
    ? (result.recommendation?.rows??[]) as unknown as Record<string,unknown>[]
    : algorithmRows(job,algorithmId);
  const rows=(table??[]).filter(row=>{
    const base=String(row.base_algorithm||row.algorithm||"");
    const matchesAlgorithm=base===algorithmId||row.variant_id===algorithmId||(
      !!spec.recommendation&&String(row.variant_id||"")===String(result.selected_variant_id||result.reduction?.selected_variant_id||"")
    );
    return matchesAlgorithm&&row.split===spec.split;
  });
  const selected=String(result.selected_variant_id||result.reduction?.selected_variant_id||"");
  const selectedRow=selected?rows.find(row=>row.variant_id===selected):undefined;
  const value=(selectedRow||rows[0])?.[spec.key];
  if(value!==undefined&&value!==null)return value;
  if(job.group==="association"&&spec.key==="support"){
    const summary=(selectedRow||rows[0])?.support_in_group_mean;
    if(typeof summary==="number")return summary;
  }
  if(job.group==="association"&&["support","confidence","lift"].includes(spec.key)){
    const rules=selected?result.group_rules?.filter(rule=>rule.variant_id===selected):result.rules;
    const values=(rules??[]).map(rule=>spec.key==="support"?(rule.support_in_group??rule.support):spec.key==="confidence"?rule.confidence:rule.lift).filter((item):item is number=>typeof item==="number"&&Number.isFinite(item));
    if(values.length)return values.reduce((sum,item)=>sum+item,0)/values.length;
  }
  return value;
}

export function AlgorithmRunComparison({groups,runs,group,onGroupChange}:{groups:LabGroup[];runs:LabJob[];group:LabGroup;onGroupChange:(groupId:string)=>void}) {
  const {t,language}=usePreferences();
  const [selectedByGroup,setSelectedByGroup]=useState<Record<string,string[]>>({});
  const [jobs,setJobs]=useState<LabJob[]>([]);
  const [loading,setLoading]=useState(false);
  const [error,setError]=useState("");
  const selectedIds=selectedByGroup[group.id]||EMPTY_RUN_IDS;
  const history=runs.filter(run=>run.group===group.id&&run.status==="completed"&&runAlgorithmIds(run).length===1);
  const specs=useMemo(()=>metricsFor(group),[group]);

  useEffect(()=>{
    if(!selectedIds.length){setJobs([]);setLoading(false);setError("");return;}
    const controller=new AbortController();
    setJobs([]);setLoading(true);setError("");
    Promise.all(selectedIds.map(id=>api<LabJob>(`/experiment-lab/runs/${id}`,{signal:controller.signal})))
      .then(data=>{
        if(controller.signal.aborted)return;
        setJobs(data.filter(job=>job.group===group.id&&job.status==="completed"&&runAlgorithmIds(job).length===1));
      })
      .catch(reason=>{if(!controller.signal.aborted)setError((reason as Error).message);})
      .finally(()=>{if(!controller.signal.aborted)setLoading(false);});
    return ()=>controller.abort();
  },[group.id,selectedIds]);

  function toggleRun(id:string,checked:boolean){
    setSelectedByGroup(previous=>{
      const current=previous[group.id]||[];
      return {...previous,[group.id]:checked?[...current.filter(value=>value!==id),id]:current.filter(value=>value!==id)};
    });
  }

  return <>
    <header className="lab-group-header lab-comparison-header"><span className="lab-kicker">RUN HISTORY / COMPARISON</span><div><h1>{t("So sánh thuật toán")}</h1><span className="lab-badge">{t(group.title)}</span></div><p className="lab-objective">{t("Chọn các lần chạy một thuật toán trong cùng nhóm để đối chiếu kết quả đã lưu.")}</p></header>
    <section className="lab-section lab-run-panel lab-run-comparison" aria-label={t("So sánh thuật toán")}>
      <div className="lab-section-heading"><div className="lab-section-title"><span className="lab-section-icon"><GitCompare size={20}/></span><div><h2>{t("Chọn nhóm và lịch sử")}</h2><p>{t("Chỉ hiển thị lần chạy đã hoàn tất và chỉ gồm một thuật toán.")}</p></div></div><span className="lab-badge">{t("{count} lần chạy được chọn",{count:String(selectedIds.length)})}</span></div>
      <label className="lab-comparison-group">{t("Nhóm thuật toán")}
        <select value={group.id} onChange={event=>onGroupChange(event.target.value)}>
          {groups.map(item=><option key={item.id} value={item.id}>{t(item.title)}</option>)}
        </select>
      </label>
      <div className="lab-comparison-history" aria-label={t("Lịch sử thực nghiệm để so sánh")}>
        {history.map(run=>{
          const checked=selectedIds.includes(run.id);
          return <label className={`lab-comparison-run${checked?" selected":""}`} key={run.id}>
            <input type="checkbox" checked={checked} onChange={event=>toggleRun(run.id,event.target.checked)}/>
            <span><strong>{runAlgorithmTitle(run,groups,t)}</strong><small>{new Date(run.created_at).toLocaleString(language==="en"?"en-US":"vi-VN")} · {run.id}</small></span>
          </label>;
        })}
      </div>
      {!history.length&&<p className="lab-muted lab-comparison-empty">{t("Chưa có lần chạy một thuật toán đã hoàn tất trong nhóm này.")}</p>}
      {selectedIds.length===1&&<p className="lab-muted lab-comparison-hint">{t("Chọn thêm ít nhất một lần chạy cùng nhóm để so sánh.")}</p>}
      {error&&<div className="lab-alert" role="alert"><AlertCircle size={18}/><p>{error}</p></div>}
      {loading&&<div className="lab-loading lab-comparison-loading" role="status"><Loader2 className="lab-spin"/><span>{t("Đang tải kết quả so sánh…")}</span></div>}
      {jobs.length>=2&&!loading&&!error&&<div className="lab-table-scroll lab-comparison-table-wrap"><table className="lab-comparison-table"><caption>{t("Kết quả các lần chạy đã chọn")}</caption><thead><tr><th scope="col">{t("Chỉ số / tập")}</th>{jobs.map(job=><th scope="col" key={job.id}><strong>{runAlgorithmTitle(job,groups,t)}</strong><small>{new Date(job.created_at).toLocaleString(language==="en"?"en-US":"vi-VN")}</small></th>)}</tr></thead><tbody>{specs.map(spec=><tr key={`${spec.key}/${spec.split}`}><th scope="row"><span>{spec.name}</span><small>{t(spec.splitName)}</small></th>{jobs.map(job=>{const algorithmId=runAlgorithmIds(job)[0];return <td key={job.id}>{metricNumber(valueFor(job,algorithmId,spec),language)}</td>;})}</tr>)}</tbody></table></div>}
      {selectedIds.length>=2&&jobs.length<2&&!loading&&!error&&<p className="lab-muted lab-comparison-hint">{t("Các lần chạy đã chọn chưa có đủ kết quả để so sánh.")}</p>}
    </section>
  </>;
}
