import type { LabGroup, LabGroupId, LabJob } from "./lab";

const groupLabels:Partial<Record<LabGroupId,string>>={
  rating:"Hồi quy",
  classification:"Phân loại",
  clustering:"Phân cụm",
  association:"Luật kết hợp",
  pca:"Giảm chiều",
  reinforcement:"Học tăng cường",
};

export function runAlgorithmIds(run:LabJob):string[] {
  const requested=run.request?.algorithms;
  if(Array.isArray(requested)&&requested.length)return [...new Set(requested)];
  return [...new Set((run.result?.rows??[]).map(row=>String(row.base_algorithm||row.algorithm)).filter(Boolean))];
}

function shortAlgorithmName(name:string):string {
  return name.replace(/\s+(?:regression|regressor|classifier)$/i,"");
}

function reductionTitle(run:LabJob,translate:(value:string)=>string):string {
  if(run.group==="reinforcement"&&!runAlgorithmIds(run).includes("dqn"))return translate("Không giảm chiều");
  const request=run.request;
  const choice=request?.reduction_choice;
  if(choice==="off")return translate("Không giảm chiều");
  if(choice==="pca")return translate("PCA");
  if(choice==="truncated_svd")return translate("TruncatedSVD");

  const config=run.result?.config||{};
  const mode=request?.reduction_mode||config.reduction_mode||run.result?.reduction?.mode;
  if(mode==="off")return translate("Không giảm chiều");
  const configuredMethods=request?.reduction_methods||
    (Array.isArray(config.reduction_methods)?config.reduction_methods.filter((value):value is "pca"|"truncated_svd"=>value==="pca"||value==="truncated_svd"):undefined);
  const methods=configuredMethods?.length?configuredMethods:
    run.result?.reduction?.variants?.map(variant=>variant.method).filter((value):value is string=>value==="pca"||value==="truncated_svd")||[];
  const titles=[...new Set(methods)].map(method=>translate(method==="pca"?"PCA":"TruncatedSVD"));
  return titles.length?titles.join(", "):translate("Không giảm chiều");
}

export function runAlgorithmTitle(run:LabJob,groups:LabGroup[],translate:(value:string)=>string):string {
  const group=groups.find(item=>item.id===run.group);
  const groupName=translate(groupLabels[run.group]||group?.title||"Thực nghiệm");
  const ids=runAlgorithmIds(run);
  const algorithms=ids.length>=3?`${ids.length} ${translate("thuật toán")}`:
    ids.map(id=>shortAlgorithmName(group?.algorithms.find(item=>item.id===id)?.name||id.replaceAll("_"," "))).join(", ");
  const algorithmLabel=algorithms||translate("thuật toán");
  return `${groupName} - ${algorithmLabel} - ${reductionTitle(run,translate)}`;
}
