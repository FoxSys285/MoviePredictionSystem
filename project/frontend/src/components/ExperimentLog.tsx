import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { ArrowDown, Download, Terminal } from "lucide-react";
import { api, ApiError, apiResourceUrl } from "../lib/api";
import { busy, type LabJob } from "../lib/lab";

type LogEntry = { id:number; time:string; level:string; message:string; progress?:number };
type LogPage = { entries:LogEntry[]; next_cursor:number; has_more:boolean; reset:boolean };
const MAX_VISIBLE_LINES = 5000;

export function ExperimentLog({job}:{job:LabJob}) {
  const [entries,setEntries] = useState<LogEntry[]>([]);
  const [following,setFollowing] = useState(true);
  const [error,setError] = useState("");
  const viewport = useRef<HTMLDivElement>(null);
  const lastScrollTop = useRef(0);
  const cursor = useRef(0);
  const active = busy(job);

  useEffect(()=>{
    const controller = new AbortController();
    let timer:number|undefined;
    const load = async()=>{
      try {
        const page = await api<LogPage>(
          `/experiment-lab/runs/${job.id}/logs?cursor=${cursor.current}&limit=500`,
          {signal:controller.signal},
        );
        if(controller.signal.aborted)return;
        const advanced = page.next_cursor>cursor.current;
        cursor.current = page.next_cursor;
        setError("");
        if(page.entries.length||page.reset){
          setEntries(current=>[...(page.reset?[]:current),...page.entries].slice(-MAX_VISIBLE_LINES));
        }
        if(active||page.has_more){
          timer=window.setTimeout(load,page.has_more&&advanced?100:1000);
        }
      } catch(cause) {
        if(controller.signal.aborted)return;
        if(cause instanceof ApiError&&cause.status===404){
          setError("Máy chủ chưa hỗ trợ nhật ký. Khởi động lại ứng dụng sau khi lượt hiện tại kết thúc.");
        } else {
          setError("Mất kết nối nhật ký, đang thử kết nối lại…");
          timer=window.setTimeout(load,2000);
        }
      }
    };
    void load();
    return ()=>{controller.abort();window.clearTimeout(timer);};
  },[job.id,job.status,active]);

  useLayoutEffect(()=>{
    if(following&&viewport.current){
      viewport.current.scrollTop=viewport.current.scrollHeight;
    }
  },[entries,following]);

  useEffect(()=>{
    const element=viewport.current;
    if(!element)return;
    const observer=new ResizeObserver(()=>{
      if(following)element.scrollTop=element.scrollHeight;
    });
    observer.observe(element);
    return ()=>observer.disconnect();
  },[following]);

  return <section className="lab-log" aria-label="Nhật ký thực nghiệm">
    <div className="lab-log-heading">
      <div><Terminal size={16}/><h3>Nhật ký thực nghiệm</h3>
        <span className={`lab-log-state${active?" is-live":""}`}><i/>{active?"Đang nhận log":"Đã kết thúc"}</span>
      </div>
      <div className="lab-log-actions">
        {!following&&<button type="button" onClick={()=>setFollowing(true)}><ArrowDown size={14}/>Về log mới nhất</button>}
        {entries.length>0&&<a href={apiResourceUrl(`/experiment-lab/runs/${job.id}/logs/download`)}><Download size={14}/>Tải log</a>}
      </div>
    </div>
    {error&&<p className="lab-log-notice" role="status">{error}</p>}
    <div ref={viewport} className="lab-log-viewport" role="log" aria-label="Nội dung log thực nghiệm"
      aria-live="polite" aria-atomic="false" aria-relevant="additions" tabIndex={0}
      onScroll={event=>{
        const element=event.currentTarget;
        const nearBottom=element.scrollHeight-element.scrollTop-element.clientHeight<40;
        if(nearBottom)setFollowing(true);
        else if(element.scrollTop<lastScrollTop.current)setFollowing(false);
        lastScrollTop.current=element.scrollTop;
      }}>
      {!entries.length&&<p className="lab-log-placeholder">{active
        ?`Đang chờ log từ worker… ${job.message}`
        :"Lần chạy này chưa có nhật ký. Các lần chạy mới sẽ lưu log tại đây."}</p>}
      {entries.map(entry=><div className={`lab-log-line lab-log-${entry.level}`} key={entry.id}>
        <time dateTime={entry.time} title={new Date(entry.time).toLocaleString("vi-VN")}>{new Date(entry.time).toLocaleTimeString("vi-VN",{hour12:false})}</time>
        <span className="lab-log-level">{entry.level.toUpperCase()}</span>
        <span className="lab-log-message">{entry.message}</span>
      </div>)}
    </div>
    <div className="lab-log-footer"><span>{entries.length.toLocaleString("vi-VN")} dòng{entries.length===MAX_VISIBLE_LINES?" gần nhất; tải log để xem toàn bộ":""}</span>
      <span>{following?"Tự theo dõi log mới":"Đang xem log cũ"}</span>
    </div>
  </section>;
}
