import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, ChevronLeft, ChevronRight, Clock3, LoaderCircle, Star, Trash2 } from "lucide-react";
import { api, ApiError, type HistoryResponse } from "../lib/api";
import { useViewer } from "../lib/viewer";
import { movieFromRecord, numberFormat, scoreFormat } from "../data/movies";
import { PageHero } from "../components/ui";
import { MovieCard } from "../components/MovieCard";
import { usePreferences } from "../lib/preferences";

export function WatchHistory() {
  const { t, language } = usePreferences();
  const { user, authLoading, ratingsVersion, onRatingsChanged, expireSession } = useViewer();
  const [data, setData] = useState<HistoryResponse | null>(null);
  const [page, setPage] = useState(1);
  const [pending, setPending] = useState(true);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  const [removeId, setRemoveId] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => { setPage(1); setData(null); }, [user?.id]);
  useEffect(() => {
    if (authLoading || !user) return;
    const controller = new AbortController();
    setPending(true); setError("");
    api<HistoryResponse>(`/me/ratings?offset=${(page - 1) * 20}&limit=20`, { signal: controller.signal })
      .then((value) => { if (!controller.signal.aborted) {
        if (page > 1 && !value.items.length) setPage(Math.max(1, Math.ceil(value.total / 20)));
        else setData(value);
      } })
      .catch((cause: Error) => { if (!controller.signal.aborted) {
        if (cause instanceof ApiError && cause.status === 401) expireSession();
        setError(cause.message);
      } })
      .finally(() => { if (!controller.signal.aborted) setPending(false); });
    return () => controller.abort();
  }, [user?.id, authLoading, ratingsVersion, page, attempt]);
  async function remove(movieId: number) {
    setBusy(true); setError("");
    try {
      await api(`/me/ratings/${movieId}`, { method: "DELETE" });
      setRemoveId(null); onRatingsChanged();
    } catch (cause) {
      if (cause instanceof ApiError && cause.status === 401) expireSession();
      else setError((cause as Error).message);
    } finally { setBusy(false); }
  }
  return <>
    <PageHero title={t("LỊCH SỬ ĐÁNH GIÁ")} subtitle={t("Những phim bạn đã đánh giá")} variant="history" />
    <main id="main" className="container history-main">
      {authLoading ? <div className="empty-state" role="status"><LoaderCircle className="spin" size={30} /><p>{t("Đang kiểm tra tài khoản...")}</p></div>
        : !user ? <div className="empty-state history-gate"><Clock3 size={40} /><h1>{t("Lịch sử điện ảnh của riêng bạn")}</h1><p>{t("Đăng nhập để xem những phim đã đánh giá và tiếp tục xây dựng sở thích của bạn.")}</p><Link className="button" to="/login?next=%2Fwatch-history">{t("Đăng nhập để xem lịch sử")} <ArrowRight size={16} /></Link></div>
        : <>
          <div className="history-heading"><div><span className="eyebrow">{t("HÀNH TRÌNH ĐIỆN ẢNH")}</span><h1>{t("Phim bạn đã đánh giá")}</h1><p>{user.profile_type === "dataset" ? t("Hồ sơ demo MovieLens User {id} · Đã nhập {count} đánh giá gốc. Thay đổi trên website được lưu riêng.", { id: user.dataset_user_id, count: numberFormat(user.imported_rating_count) }) : t("Điểm của bạn được lưu riêng và giúp cá nhân hóa các gợi ý.")}</p></div><span className="history-count"><Clock3 size={16} />{numberFormat(data?.total || 0)} {t("phim")}</span></div>
          {error && <div className="rail-error" role="alert"><p>{error}</p><button className="button button-outline" onClick={() => setAttempt(attempt + 1)}>{t("Thử lại lịch sử")}</button></div>}
          {pending ? <div className="empty-state" role="status"><LoaderCircle className="spin" size={30} /><p>{t("Đang tải lịch sử đánh giá...")}</p></div>
            : !error && !data?.items.length ? <div className="empty-state"><Star size={38} /><h2>{t("Bắt đầu với bộ phim đầu tiên")}</h2><p>{t("Mở chi tiết một phim và chấm điểm. Phim đó sẽ xuất hiện ở đây.")}</p><Link className="button" to="/gallery">{t("Khám phá phim")} <ArrowRight size={16} /></Link></div>
            : data && <div className="movie-grid history-grid">{data.items.map((item) => <article className="history-entry" key={item.movieId}>
              <MovieCard movie={movieFromRecord(item)} />
              <div className="history-entry-meta"><span><Star size={13} fill="currentColor" /> {t("Bạn chấm")} <b>{scoreFormat(item.user_rating)}/5</b></span>
                <time dateTime={new Date(item.updated_at).toISOString()}>{new Date(item.updated_at).toLocaleDateString(language === "vi" ? "vi-VN" : "en-US")}</time>
                <div className="history-entry-actions"><Link to={`/movies/${item.movieId}`}>{t("Sửa đánh giá")}</Link><button aria-label={`${t("Xóa đánh giá")} ${movieFromRecord(item).title}`} onClick={() => setRemoveId(item.movieId)}><Trash2 size={13} /></button></div>
                {removeId === item.movieId && <div className="history-remove-confirm"><span>{t("Xóa đánh giá này?")}</span><button disabled={busy} onClick={() => remove(item.movieId)}>{t("Xác nhận xóa")}</button><button disabled={busy} onClick={() => setRemoveId(null)}>{t("Hủy")}</button></div>}
              </div>
            </article>)}</div>}
          {data && data.total > 20 && <nav className="pagination" aria-label={t("Phân trang lịch sử")}><button disabled={page <= 1 || pending} onClick={() => setPage(page - 1)} aria-label={t("Lịch sử trang trước")}><ChevronLeft size={16} /></button><span>{page} / {Math.ceil(data.total / 20)}</span><button disabled={page * 20 >= data.total || pending} onClick={() => setPage(page + 1)} aria-label={t("Lịch sử trang tiếp")}><ChevronRight size={16} /></button></nav>}
        </>}
    </main>
  </>;
}
