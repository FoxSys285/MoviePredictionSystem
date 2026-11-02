import { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { Check, LoaderCircle, LogIn, Star, StarHalf } from "lucide-react";
import { api, ApiError, type PersonalRating } from "../lib/api";
import { scoreFormat } from "../data/movies";
import { useViewer } from "../lib/viewer";
import { usePreferences } from "../lib/preferences";

export function MovieRating({ movieId }: { movieId: number }) {
  const { t } = usePreferences();
  const { user, authLoading, onRatingsChanged, expireSession } = useViewer();
  const location = useLocation();
  const [value, setValue] = useState("");
  const [saved, setSaved] = useState<number | null>(null);
  const [pending, setPending] = useState(false);
  const [busy, setBusy] = useState(false);
  const [prompt, setPrompt] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [hovered, setHovered] = useState(0);
  const [attempt, setAttempt] = useState(0);
  const [loadedKey, setLoadedKey] = useState("");
  const profileKey = `${user?.id || "guest"}:${movieId}`;
  const loadingRating = authLoading || pending || Boolean(user && loadedKey !== profileKey);
  useEffect(() => {
    setSaved(null); setValue(""); setPrompt(false); setError(""); setMessage("");
    if (!user) { setPending(false); return; }
    const controller = new AbortController();
    setPending(true);
    api<PersonalRating>(`/me/ratings/${movieId}`, { signal: controller.signal })
      .then((data) => { if (!controller.signal.aborted) { setSaved(data.rating); setValue(data.rating === null ? "" : String(data.rating)); setLoadedKey(`${user.id}:${movieId}`); } })
      .catch((cause: Error) => { if (!controller.signal.aborted) {
        if (cause instanceof ApiError && cause.status === 401) expireSession();
        setError(cause.message);
      } })
      .finally(() => { if (!controller.signal.aborted) setPending(false); });
    return () => controller.abort();
  }, [user?.id, movieId, attempt]);
  function choose(score: string) {
    setValue(score); setMessage("");
    if (!user) setPrompt(true);
  }
  async function save() {
    if (!user) { setPrompt(true); return; }
    if (!value) { setError(t("Hãy chọn điểm đánh giá từ 0,5 đến 5.")); return; }
    setBusy(true); setError(""); setMessage("");
    try {
      const data = await api<PersonalRating>(`/me/ratings/${movieId}`, { method: "PUT", body: JSON.stringify({ rating: Number(value) }) });
      setSaved(data.rating); setMessage(t("Đã lưu đánh giá của bạn vào Watch History.")); onRatingsChanged();
    } catch (cause) {
      if (cause instanceof ApiError && cause.status === 401) { expireSession(); setPrompt(true); }
      else setError((cause as Error).message);
    } finally { setBusy(false); }
  }
  const shown = hovered || Number(value);
  const next = encodeURIComponent(location.pathname + location.search);
  return <section className="personal-rating" aria-label={t("Đánh giá phim của bạn")}>
    <div className="rating-panel-heading"><span className="feature-icon"><Star size={23} /></span><div>
      <h2>{t("Bạn đánh giá phim này thế nào?")}</h2>
      <p>{saved !== null ? t("Bạn đã chấm {rating}/5. Bạn có thể cập nhật điểm bất cứ lúc nào.", { rating: scoreFormat(saved) }) : t("Chấm điểm phim đã xem để xây dựng lịch sử và nhận gợi ý phù hợp hơn.")}</p>
    </div></div>
    <div className="rating-controls">
      <div className="rating-stars" onPointerLeave={() => setHovered(0)} role="group" aria-label={t("Chọn số sao")}>
        {[1, 2, 3, 4, 5].map((score) => <button type="button" key={score} aria-label={t("Chấm {score} sao", { score })}
          aria-pressed={Number(value) === score} disabled={loadingRating || busy}
          onPointerEnter={() => setHovered(score)} onFocus={() => setHovered(score)} onBlur={() => setHovered(0)} onClick={() => choose(String(score))}>
          {shown > score - 1 && shown < score ? <StarHalf size={26} fill="currentColor" /> : <Star size={26} fill={shown >= score ? "currentColor" : "none"} />}
        </button>)}
      </div>
      <label className="rating-select-label">{t("Điểm của bạn")}
        <select aria-label={t("Điểm đánh giá của bạn")} value={value} disabled={loadingRating || busy} onChange={(event) => choose(event.target.value)}>
          <option value="">{t("Chọn điểm")}</option>
          {Array.from({ length: 10 }, (_, index) => (index + 1) / 2).map((score) => <option key={score} value={score}>{scoreFormat(score)} / 5</option>)}
        </select>
      </label>
      <button className="button" disabled={loadingRating || busy} onClick={save}>
        {busy || pending ? <LoaderCircle className="spin" size={16} /> : <Check size={16} />}
        {busy ? t("Đang lưu...") : saved !== null ? t("Cập nhật đánh giá") : t("Lưu đánh giá")}
      </button>
    </div>
    {prompt && !user && <div className="rating-login-prompt" role="alert"><LogIn size={19} /><div>
      <p>{t("Vui lòng đăng nhập để đánh giá phim và lưu lịch sử của bạn.")}</p>
      <Link to={`/login?next=${next}`}>{t("Đăng nhập để đánh giá")}</Link><span> · </span><Link to={`/register?next=${next}`}>{t("Tạo tài khoản")}</Link>
    </div></div>}
    {message && <p className="rating-success" role="status"><Check size={15} />{message}</p>}
    {error && <p className="form-error" role="alert">{error} <button className="text-button" onClick={() => setAttempt(attempt + 1)}>{t("Tải lại đánh giá")}</button></p>}
  </section>;
}
