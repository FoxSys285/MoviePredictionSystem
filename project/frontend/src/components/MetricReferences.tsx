import { useEffect, useState } from "react";
import { ChevronDown, RotateCcw } from "lucide-react";
import type { Criterion, LabGroup, LabResult, LabRow } from "../lib/lab";
import { assessMetric, referenceLabel, referenceValue, type ReferenceSettings, type ReferenceValues } from "../lib/metric-references";
import { usePreferences } from "../lib/preferences";

const storageKey="movie-lab-metric-references-v1";
function readSettings():ReferenceSettings {
  try {const value=JSON.parse(localStorage.getItem(storageKey)||"{}");return value&&typeof value==="object"&&!Array.isArray(value)?value:{};}catch{return {};}
}

export function useMetricReferenceSettings() {
  const [settings,setSettings]=useState<ReferenceSettings>(readSettings);
  function update(next:ReferenceSettings) {setSettings(next);try{localStorage.setItem(storageKey,JSON.stringify(next));}catch{/* Settings remain usable in this session. */}}
  return {settings,setValue:(group:string,key:string,value:number)=>update({...settings,[group]:{...settings[group],[key]:value}}),
    reset:(group:string)=>{const next={...settings};delete next[group];update(next);}};
}

function ReferenceInput({metric,value,onChange}:{metric:Criterion;value:number;onChange:(value:number)=>void}) {
  const [draft,setDraft]=useState(String(value));
  useEffect(()=>setDraft(String(value)),[value]);
  const ref=metric.reference!;
  return <label className="lab-reference-input"><span className="sr-only">Mốc {metric.name}</span>
    <input type="number" aria-label={`Mốc ${metric.name}`} min={ref.minimum} max={ref.maximum??undefined} step={metric.key==="components"?1:"any"} value={draft}
      onChange={e=>{setDraft(e.target.value);if(e.target.value!==""&&e.target.validity.valid&&Number.isFinite(e.target.valueAsNumber))onChange(e.target.valueAsNumber);}}
      onBlur={()=>setDraft(String(value))}/><span>{ref.unit}</span></label>;
}

export function MetricReferenceTable({group,values,onChange,onReset}:{group:LabGroup;values:ReferenceValues;onChange:(key:string,value:number)=>void;onReset:()=>void}) {
  const [expanded,setExpanded]=useState(false);
  const {t}=usePreferences();
  return <section id="lab-metrics" className="lab-section"><div className="lab-section-heading"><h2>Tiêu chí so sánh & mốc đánh giá</h2><div className="lab-reference-actions"><button type="button" className="lab-outline lab-reference-toggle" aria-expanded={expanded} aria-controls="lab-criteria-content" onClick={()=>setExpanded(value=>!value)}><ChevronDown size={14} className={expanded?"is-open":""}/>{t(expanded?"Thu gọn":"Mở rộng")}</button><button type="button" className="lab-outline" onClick={onReset}><RotateCcw size={14}/>Khôi phục mốc mặc định</button></div></div>
    <div id="lab-criteria-content" className="lab-reference-content" hidden={!expanded}>
      <p className="lab-muted lab-reference-intro">Mốc khởi đầu do dự án lựa chọn, có thể điều chỉnh và được nhớ trên trình duyệt này. Nhãn “Tốt theo mốc” chỉ áp dụng cho từng chỉ số; không chứng nhận toàn bộ mô hình. Thay mốc không thay đổi kết quả huấn luyện hay mô hình đã được chọn.</p>
      <div className="lab-table-scroll lab-criteria"><table><caption>Tiêu chí đánh giá · {group.title}</caption><thead><tr><th>Tiêu chí</th><th>Đánh giá gì / cách xác định</th><th>Mốc tham khảo</th></tr></thead><tbody>{group.metrics.map(m=><tr key={m.key}><th scope="row">{m.name}</th><td>{m.definition}</td><td>{m.reference?<><ReferenceInput metric={m} value={referenceValue(m,values)!} onChange={v=>onChange(m.key,v)}/><strong className="lab-reference-target">{referenceLabel(m,values)}</strong><small>Mặc định: {m.reference.value} {m.reference.unit}</small></>:"Chưa có mốc"}</td></tr>)}</tbody></table></div>
    </div>
  </section>;
}

export function MetricAssessment({metric,row,result,values,metrics}:{metric:Criterion;row:LabRow;result:LabResult;values?:ReferenceValues;metrics?:Criterion[]}) {
  const status=assessMetric(metric,row,result,values,metrics);
  return <span className="lab-metric-assessment" title={status.explanation}><span className={`lab-verdict lab-verdict-${status.tone}`} data-verdict={status.tone}>{status.label}</span>{status.reference&&<small>{status.reference}</small>}</span>;
}
