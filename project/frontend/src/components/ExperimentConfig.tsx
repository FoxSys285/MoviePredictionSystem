import type { LabGroup } from "../lib/lab";

export const experimentDefaults={max_users:150,seed:42,epochs:8,patience:3,tune:true,
  k_values:[3,4,6],eps_values:[1,1.5,2],min_samples_values:[3,5],min_user_ratings:5,
  max_items:80,min_support:.08,min_confidence:.3,max_rule_length:2,components:[2,4,8],
  episodes:500,learning_rate:.2,discount_factor:.9,exploration_rate:.2,
  parameters:{factors:16,hidden_size:32,learning_rate:.005,regularization:.02,n_estimators:80,max_depth:6,c:1,min_samples_leaf:3,dropout:.1,batch_size:2048}};
export type ExperimentConfigValue=typeof experimentDefaults;

function ListInput({label,value,min,max,integer=true,onChange}:{label:string;value:number[];min:number;max:number;integer?:boolean;onChange:(v:number[])=>void}) {
  // Keep an editing draft so trailing commas do not disappear while typing.
  return <label>{label}<input defaultValue={value.join(", ")} required onChange={e=>{const values=e.target.value.split(",").map(s=>s.trim());const numbers=values.map(Number);const valid=values.every(Boolean)&&numbers.every(n=>Number.isFinite(n)&&n>=min&&n<=max&&(!integer||Number.isInteger(n)));e.target.setCustomValidity(valid?"":`Nhập danh sách phân cách bằng dấu phẩy, từ ${min} đến ${max}.`);if(valid)onChange(numbers);}}/></label>;
}

export function ExperimentConfiguration({group,value,onChange,disabled,algorithms,setAlgorithms}:{group:LabGroup;value:ExperimentConfigValue;onChange:(v:ExperimentConfigValue)=>void;disabled:boolean;algorithms:string[];setAlgorithms:(v:string[])=>void}) {
  const set=(key:keyof ExperimentConfigValue,v:unknown)=>onChange({...value,[key]:v});
  const field=(key:keyof ExperimentConfigValue,label:string,min:number,max:number,step=1)=><label key={key}>{label}<input type="number" required min={min} max={max} step={step} value={value[key] as number} onChange={e=>set(key,e.target.value===""?"":Number(e.target.value))}/></label>;
  const param=(key:keyof typeof value.parameters,label:string,min:number,max:number,step=1)=><label key={key}>{label}<input type="number" required min={min} max={max} step={step} value={value.parameters[key]} onChange={e=>set("parameters",{...value.parameters,[key]:e.target.value===""?"":Number(e.target.value)})}/></label>;
  const list=(key:"k_values"|"eps_values"|"min_samples_values"|"components",label:string,min:number,max:number,integer=true)=><ListInput key={`${group.id}/${key}`} label={label} value={value[key]} min={min} max={max} integer={integer} onChange={v=>set(key,v)}/>;
  return <fieldset className="lab-config-wrapper" disabled={disabled}><legend className="sr-only">Cấu hình thực nghiệm</legend>
    <div className="lab-algorithm-select">{group.algorithms.map(a=><label className={algorithms.includes(a.id)?"selected":""} key={a.id}><input type="checkbox" checked={algorithms.includes(a.id)} onChange={e=>setAlgorithms(e.target.checked?[...algorithms,a.id]:algorithms.filter(id=>id!==a.id))}/><span>{a.name}</span></label>)}</div>
    <details className="lab-parameter-details"><summary><span><strong>Cấu hình thông số</strong><small>Ngân sách dữ liệu, hyperparameters và các thiết lập của nhóm</small></span><span className="lab-collapse-label"><span className="lab-when-closed">Mở rộng</span><span className="lab-when-open">Thu gọn</span><span className="lab-collapse-arrow" aria-hidden="true">⌄</span></span></summary>
    <div className="lab-config">{field("max_users","Số user tối đa",10,5000)}
      <label>Random Seed<input value="42" readOnly/></label><label>Train / Validation / Test<input value="70% / 15% / 15%" readOnly/></label>
      {["rating","classification"].includes(group.id)&&<><label>Cross Validation<input value="5-Fold trên Train 70%" readOnly/></label><label className="lab-check-label"><input type="checkbox" checked={value.tune} onChange={e=>set("tune",e.target.checked)}/>Thử thêm cấu hình đơn giản hơn</label></>}
      {group.id==="rating"&&<>{field("epochs","Epochs SVD / NCF",2,100)}{field("patience","NCF · Early Stopping patience",1,20)}{param("factors","SVD / NCF · Factors",2,128)}{param("hidden_size","NCF · Hidden size",4,256)}{param("dropout","NCF · Dropout",0,.9,.05)}{param("batch_size","NCF · Batch size",32,16384)}{param("learning_rate","Learning rate mô hình",.0001,.1,.0001)}{param("regularization","SVD / NCF · Regularization",0,10,.01)}</>}
      {["rating","classification"].includes(group.id)&&<>{param("n_estimators","XGBoost / Forest · Trees",10,500)}{param("max_depth","XGBoost / Forest · Max depth",1,20)}</>}
      {group.id==="classification"&&<><label>Ngưỡng rating Thích<input value="rating ≥ 3,5" readOnly/></label>{param("c","Logistic · C",.001,100,.001)}{param("min_samples_leaf","Forest · Min samples leaf",1,50)}</>}
      {group.id==="clustering"&&<>{list("k_values","K-Means · Danh sách K",2,30)}{list("eps_values","DBSCAN · Danh sách eps",.1,20,false)}{list("min_samples_values","DBSCAN · Danh sách min_samples",2,100)}</>}
      {["clustering","pca"].includes(group.id)&&field("min_user_ratings","Số ratings Train tối thiểu / user",2,100)}
      {group.id==="association"&&<><label>Ngưỡng phim yêu thích<input value="rating ≥ 4,0" readOnly/></label>{field("max_items","Số phim tối đa",10,500)}{field("min_support","min_support",.001,1,.001)}{field("min_confidence","min_confidence",.01,1,.01)}{field("max_rule_length","Số phim tối đa trong itemset",2,3)}</>}
      {group.id==="pca"&&list("components","Danh sách Components",1,128)}
      {group.id==="reinforcement"&&<>{field("episodes","Episodes",10,20000)}{field("learning_rate","Q-Learning · Learning Rate",.01,1,.01)}{field("discount_factor","Q-Learning · Discount Factor",0,1,.01)}{field("exploration_rate","Q-Learning · Exploration Rate",0,1,.01)}</>}
    </div></details><p className="lab-muted">Chọn một thuật toán để chạy riêng, hoặc nhiều thuật toán để so sánh. Kết quả luôn ghi rõ số ratings thực tế trong mẫu.</p>
  </fieldset>;
}
