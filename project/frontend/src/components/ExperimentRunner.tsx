import { useEffect, useRef, useState } from "react";
import { Clock3, Download, Play } from "lucide-react";
import { Link, useSearchParams } from "react-router-dom";
import { api, apiResourceUrl, ApiError } from "../lib/api";
import { usePreferences } from "../lib/preferences";

type Variant = "baseline" | "improved";
type Job = {
  id: string; variant: string; status: "queued" | "running" | "completed" | "failed";
  progress: number; stage: string; message: string; created_at: string; completed_at?: string;
  algorithms?: string[];
};
type LogEvent = { time: string; stage: string; message: string; cursor: number };
type Logs = { entries: LogEvent[]; next_cursor: number; has_more: boolean; reset: boolean };

const names: Record<string, string> = { linear: "Linear Regression", xgboost: "XGBoost", svd: "SVD", ncf: "NCF" };
const phases: Record<string, string> = {
  queued: "Đang khởi động thực nghiệm", data: "Chuẩn bị dữ liệu", search: "GridSearchCV trên Train",
  validation: "Đánh giá Validation", finished: "Đã đánh giá Validation", refit: "Refit Train + Validation",
  test: "Đánh giá Test", completed: "Đã hoàn tất", failed: "Thực nghiệm gặp lỗi",
};
const active = (job: Job | null) => job?.status === "queued" || job?.status === "running";
const storageKey = (variant: Variant) => `movie-experiment-run-${variant}`;
function savedRun(variant: Variant) {
  try { return localStorage.getItem(storageKey(variant)) || ""; } catch { return ""; }
}
function remember(variant: Variant, id: string) {
  try {
    if (id) localStorage.setItem(storageKey(variant), id);
    else localStorage.removeItem(storageKey(variant));
  } catch { /* Monitoring still works if browser storage is unavailable. */ }
}
function duration(milliseconds: number) {
  const seconds = Math.max(0, Math.floor(milliseconds / 1000));
  return [Math.floor(seconds / 3600), Math.floor(seconds / 60) % 60, seconds % 60]
    .map((value) => String(value).padStart(2, "0")).join(":");
}

export function ExperimentRunner({ variant, algorithms, onBusyChange, disabledReason }: {
  variant: Variant; algorithms: string[]; onBusyChange: (variant: Variant, busy: boolean) => void;
  disabledReason?: string;
}) {
  const { t, language } = usePreferences();
  const [params, setParams] = useSearchParams();
  const [runId, setRunId] = useState(() => params.get("run") || savedRun(variant));
  const [job, setJob] = useState<Job | null>(null);
  const [logs, setLogs] = useState<LogEvent[]>([]);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [now, setNow] = useState(Date.now());
  const logElement = useRef<HTMLDivElement>(null);
  const followLog = useRef(true);
  const mounted = useRef(true);
  const restoring = Boolean(runId && !job);
  const busy = submitting || restoring || active(job);

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);
  useEffect(() => { onBusyChange(variant, busy); }, [variant, busy, onBusyChange]);
  useEffect(() => {
    if (!busy) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [busy]);

  useEffect(() => {
    if (!runId) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    let cursor = 0;
    setLogs([]);
    followLog.current = true;
    async function poll() {
      try {
        const data = await api<{ job: Job }>(`/experiments/runs/${runId}`, { signal: controller.signal });
        if (controller.signal.aborted) return;
        if (data.job.variant !== variant && data.job.variant !== "both") {
          setRunId("");
          remember(variant, "");
          return;
        }
        setJob(data.job);
        setNow(Date.now());
        remember(variant, runId);
        const events = await api<Logs>(`/experiments/runs/${runId}/logs?cursor=${cursor}`, { signal: controller.signal });
        if (controller.signal.aborted) return;
        cursor = events.next_cursor;
        setLogs((previous) => [...(events.reset ? [] : previous), ...events.entries].slice(-400));
        setError("");
        if (active(data.job) || events.has_more) timer = setTimeout(poll, events.has_more ? 100 : 1500);
      } catch (cause) {
        if (controller.signal.aborted) return;
        if (cause instanceof ApiError && cause.status === 404) {
          remember(variant, "");
          setRunId("");
          setJob(null);
        } else {
          setError((cause as Error).message);
          timer = setTimeout(poll, 3000);
        }
      }
    }
    void poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [runId, variant]);

  useEffect(() => {
    if (followLog.current && logElement.current) logElement.current.scrollTop = logElement.current.scrollHeight;
  }, [logs]);

  async function start() {
    if (!algorithms.length || busy || disabledReason) return;
    setSubmitting(true);
    setError("");
    try {
      const data = await api<Job>("/experiments/runs", {
        method: "POST", body: JSON.stringify({ variant, algorithms, max_users: 1000, seed: 42 }),
      });
      // A navigation can unmount this pane while the server starts its worker.
      remember(variant, data.id);
      if (!mounted.current) return;
      setJob(data);
      setRunId(data.id);
      setParams({ section: variant, run: data.id }, { replace: true });
      setNow(Date.now());
    } catch (cause) {
      if (mounted.current) setError((cause as Error).message);
    } finally {
      if (mounted.current) setSubmitting(false);
    }
  }

  const [phase, algorithm] = job?.stage.split(":") || [];
  const progress = Math.max(0, Math.min(100, job?.progress || 0));
  const elapsed = job ? (job.completed_at ? Date.parse(job.completed_at) : now) - Date.parse(job.created_at) : 0;
  const clock = (value: string) => new Date(value).toLocaleTimeString(language === "vi" ? "vi-VN" : "en-US");

  return <section className="experiment-runner" aria-label={t("Chạy thực nghiệm")}>
    <div className="experiment-run-actions">
      <button type="button" className="button button-primary" disabled={busy || !algorithms.length || Boolean(disabledReason)} onClick={() => void start()}>
        <Play size={17} aria-hidden="true" />{t(busy ? "Đang chạy thực nghiệm" : "Chạy thực nghiệm")}
      </button>
      {!busy && (disabledReason ? <span>{t(disabledReason)}</span>
        : !algorithms.length && <span>{t("Chọn ít nhất một thuật toán để thực nghiệm.")}</span>)}
    </div>
    <p className="experiment-run-scope">{t("Mẫu tối đa 1.000 user, giữ đầy đủ lịch sử đánh giá. Chia 70/15/15 theo từng user; seed 42. Lượt chạy này refit trên Train 70%.")}</p>
    {error && <p role="alert" className="experiment-results-error">{t(error)}</p>}
    {(job || submitting || restoring) && <div className="experiment-run-monitor">
      <div className="experiment-run-status">
        <div>
          <strong role="status">{t(job ? phases[phase] || phase : "Đang khởi động thực nghiệm")}{algorithm ? ` · ${names[algorithm] || algorithm}` : ""}</strong>
          {job && <small>{job.id} · {(job.algorithms || []).map((id) => names[id] || id).join(", ")}</small>}
        </div>
        <span className="experiment-run-clock"><Clock3 size={17} aria-hidden="true" />
          <span>{t("Thời gian")}</span><output aria-label={t("Thời gian thực nghiệm")}>{duration(elapsed)}</output>
        </span>
      </div>
      <div className="experiment-run-progress"><progress max={100} value={progress} aria-label={t("Tiến độ thực nghiệm")} /><span>{progress.toFixed(0)}%</span></div>
      {job?.status === "failed" && <p role="alert" className="experiment-results-error">{job.message}</p>}
      <div className="experiment-log-heading"><h4>{t("Log thực nghiệm")}</h4>
        {job && logs.length > 0 && <a href={apiResourceUrl(`/experiments/runs/${job.id}/logs/download`)}><Download size={15} aria-hidden="true" />{t("Tải log")}</a>}
      </div>
      <div className="experiment-log" role="log" aria-label={t("Log thực nghiệm")} aria-live="off" tabIndex={0} ref={logElement}
        onScroll={(event) => { const node = event.currentTarget; followLog.current = node.scrollHeight - node.scrollTop - node.clientHeight < 40; }}>
        {logs.length ? logs.map((entry) => <div key={entry.cursor} className={entry.stage === "failed" ? "is-error" : undefined}>
          <time dateTime={entry.time}>{clock(entry.time)}</time><span>{entry.message}</span>
        </div>) : <p>{t("Đang chờ log từ tiến trình thực nghiệm...")}</p>}
      </div>
      {job?.status === "completed" && <Link className="button button-outline" to={`?section=evaluation&run=${job.id}`}>{t("Xem kết quả thực nghiệm")}</Link>}
    </div>}
  </section>;
}
