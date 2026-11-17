import { metricNumber, type Criterion, type LabResult, type LabRow } from "./lab";

export type ReferenceValues=Record<string,number>;
export type ReferenceSettings=Record<string,ReferenceValues>;
export type Assessment={tone:"good"|"bad"|"neutral"|"caution";label:string;explanation:string;reference?:string};

export function referenceValue(metric:Criterion,values:ReferenceValues={}) {
  const ref=metric.reference;
  if(!ref)return undefined;
  const value=values[metric.key];
  return typeof value==="number"&&Number.isFinite(value)&&value>=ref.minimum&&(ref.maximum===null||value<=ref.maximum)?value:ref.value;
}

export function referenceLabel(metric:Criterion,values:ReferenceValues={}) {
  const ref=metric.reference,value=referenceValue(metric,values);
  if(!ref||value===undefined)return "Chưa có mốc";
  const operator=ref.operator==="lte"?"≤":ref.operator==="gte"?"≥":">";
  return `${ref.kind==="baseline"?"Δ ":""}${operator} ${metricNumber(value)} ${ref.unit.replace(/^Δ /,"")}`;
}

export function assessMetric(metric:Criterion,row:LabRow,result:LabResult,values:ReferenceValues={},metrics:Criterion[]=[]):Assessment {
  const raw=row[metric.key],ref=metric.reference,target=referenceValue(metric,values);
  if(typeof raw!=="number"||!Number.isFinite(raw))return {tone:"neutral",label:"Không đủ dữ liệu",explanation:"Chỉ số không xác định; không thay bằng 0 và không kết luận tốt/xấu."};
  if(!ref||target===undefined)return {tone:"neutral",label:"Chưa có mốc",explanation:"Cần mốc cùng đơn vị đo để diễn giải."};
  if(ref.kind==="baseline") {
    if(row.algorithm!=="q_learning")return {tone:"neutral",label:"Baseline đối chiếu",explanation:"Policy tham chiếu để đánh giá mức cải thiện của Q-Learning."};
    const baselines=["random","popular"].map(a=>result.rows.find(r=>r.algorithm===a&&r.split===row.split));
    if(!baselines.every(r=>typeof r?.[metric.key]==="number"&&Number.isFinite(r[metric.key])))return {tone:"neutral",label:"Thiếu baseline",explanation:"Chạy cùng cả Random và Popularity để xác định mốc; không dùng 0 thay cho baseline thiếu."};
    if(!baselines.every(r=>typeof row.evaluation_episodes==="number"&&r!.evaluation_episodes===row.evaluation_episodes))return {tone:"caution",label:"Khác điều kiện đo",explanation:"Cần cùng số episode khi so policy."};
    const base=Math.max(...baselines.map(r=>Number(r![metric.key]))),delta=raw-base;
    const equal=Math.abs(delta-target)<=1e-12;
    return {tone:equal?"neutral":delta>target?"good":"bad",label:equal?"Ngang mốc cải thiện":delta>target?"Vượt mốc baseline":"Chưa vượt baseline",
      explanation:`Baseline tốt nhất = ${metricNumber(base)}; Δ thực tế = ${metricNumber(delta)}. ${ref.caution}`,
      reference:`Δ ${metricNumber(delta)} · yêu cầu > ${metricNumber(target)}`};
  }
  const passed=ref.operator==="lte"?raw<=target:ref.operator==="gte"?raw>=target:raw>target;
  const reference=referenceLabel(metric,values);
  if(["silhouette","dbi"].includes(metric.key)&&typeof row.noise_fraction==="number") {
    const coverage=typeof row.coverage==="number"?row.coverage:1-row.noise_fraction;
    const minimum=result.cluster_min_coverage??.8;
    if(coverage<minimum||row.qualified===false)return {tone:"caution",label:coverage<minimum?"Bao phủ chưa đạt":"Cấu hình chưa đạt",explanation:`Chỉ số chỉ tính trên user không noise. Bao phủ ${(coverage*100).toFixed(1)}%; yêu cầu ≥${(minimum*100).toFixed(1)}% và ít nhất hai cụm hợp lệ khi chọn cấu hình.`,reference};
  }
  if(metric.key==="lift"&&ref.operator==="gt"&&Math.abs(raw-target)<=1e-12)return {tone:"neutral",label:"Bằng mốc Lift",explanation:target===1?"Lift =1: không tăng đồng xuất hiện so với độc lập.":"Lift bằng mốc đã chọn; chưa vượt mốc.",reference};
  if(metric.key==="accuracy"&&passed&&result.class_balance) {
    const train=result.class_balance.train,balance=result.class_balance[row.split];
    if(train&&balance&&balance.positive+balance.negative>0) {
      const majority=train.positive>=train.negative?balance.positive:balance.negative;
      const baseline=majority/(balance.positive+balance.negative);
      if(raw<=baseline)return {tone:"caution",label:"Đạt mốc; chưa vượt nền",explanation:`Accuracy đạt mốc ${metricNumber(target)} nhưng không vượt baseline luôn đoán lớp đa số Train (${metricNumber(baseline)} trên ${row.split}).`,reference:`${reference} · nền ${metricNumber(baseline)}`};
    }
  }
  if(metric.key==="components"&&passed) {
    const evrMetric=metrics.find(m=>m.key==="explained_variance_ratio"),errorMetric=metrics.find(m=>m.key==="reconstruction_error");
    const evrTarget=evrMetric&&referenceValue(evrMetric,values),errorTarget=errorMetric&&referenceValue(errorMetric,values);
    if(evrTarget===undefined||errorTarget===undefined||typeof row.explained_variance_ratio!=="number"||typeof row.reconstruction_error!=="number"||row.explained_variance_ratio<evrTarget||row.reconstruction_error>errorTarget)return {tone:"caution",label:"Gọn; cần giữ thông tin",explanation:"Trong ngân sách chiều nhưng phương sai/lỗi tái tạo chưa đồng thời đạt mốc. Không kết luận giảm chiều tốt chỉ vì ít components.",reference};
  }
  return {tone:passed?"good":"bad",label:ref.kind==="budget"?(passed?"Trong ngân sách":"Vượt ngân sách"):ref.kind==="screening"?(passed?"Đạt mốc sàng lọc":"Chưa đạt sàng lọc"):(passed?"Tốt theo mốc":"Chưa đạt mốc"),explanation:`${raw} ${passed?"đạt":"chưa đạt"} mốc ${reference}. ${ref.caution}`,reference};
}
