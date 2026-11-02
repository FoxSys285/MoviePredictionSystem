import { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { ArrowRight, Sparkles } from "lucide-react";
import { api, ALGORITHM_NAMES, PRIMARY_ALGORITHMS, type Recommendations as RecommendationData } from "../lib/api";
import { useViewer } from "../lib/viewer";
import { movieFromRecord } from "../data/movies";
import { MovieRail, RailSkeleton } from "./MovieRail";
import { SectionTitle, Select } from "./ui";
import { usePreferences } from "../lib/preferences";
import { scrollToElement } from "../lib/motion";

export function Recommendations() {
  const { t } = usePreferences();
  const { user, authLoading, ratingsVersion } = useViewer();
  const location = useLocation();
  const [result, setData] = useState<RecommendationData | null>(null);
  const [dataOwner, setDataOwner] = useState("");
  const owner = String(user?.id || "guest");
  const data = dataOwner === owner ? result : null;
  const [algorithm, setAlgorithm] = useState("auto");
  const [pending, setPending] = useState(true);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  useEffect(() => { setAlgorithm("auto"); setData(null); }, [user?.id]);
  useEffect(() => {
    if (authLoading) return;
    const controller = new AbortController();
    setPending(true); setError("");
    api<RecommendationData>(`/recommendations?limit=20&algorithm=${algorithm}`, { signal: controller.signal })
      .then((value) => { if (!controller.signal.aborted) { setData(value); setDataOwner(owner); } })
      .catch((cause: Error) => { if (!controller.signal.aborted) setError(cause.message); })
      .finally(() => { if (!controller.signal.aborted) setPending(false); });
    return () => controller.abort();
  }, [user?.id, authLoading, ratingsVersion, algorithm, attempt]);
  const subtitle = !user ? t("Khám phá phim nổi bật. Đăng nhập và chấm điểm để nhận gợi ý dành riêng cho bạn.")
    : data?.mode === "model" ? data.strategy === "trained_dataset_user"
      ? t("{algorithm} sử dụng hồ sơ User {id} đã học từ dataset.", { algorithm: ALGORITHM_NAMES[data.algorithm!], id: data.dataset_user_id })
      : t("{algorithm} cập nhật hồ sơ từ {count} đánh giá của bạn, giữ cố định tham số phim.", { algorithm: ALGORITHM_NAMES[data.algorithm!], count: data.history_count })
    : data?.mode === "content" ? t("Gợi ý theo thể loại từ {count} đánh giá{profile}. Chưa có điểm dự đoán từ mô hình.", { count: data.history_count, profile: data.profile_type === "dataset" ? ` trong hồ sơ MovieLens User ${data.dataset_user_id}` : " của bạn" })
    : t("Gợi ý ban đầu từ cộng đồng. Hãy đánh giá vài phim để bắt đầu cá nhân hóa.");
  const next = encodeURIComponent(location.pathname + location.search);
  const available = data?.available_algorithms || [];
  const references = available.filter((name) => !PRIMARY_ALGORITHMS.some((primary) => primary === name));
  const actions = <div className="recommendation-controls">
    <div className="recommendation-algorithm">
      <span>{t("Thuật toán gợi ý")}</span>
      <Select label={t("Mô hình gợi ý phim")} value={algorithm} onChange={setAlgorithm}>
        <option value="auto">{t("Tự động chọn mô hình")}</option>
        {PRIMARY_ALGORITHMS.map((name) => <option key={name} value={name} disabled={!available.includes(name)}>
          {ALGORITHM_NAMES[name]}{!available.includes(name) ? ` · ${t("Chưa sẵn sàng")}` : ""}
        </option>)}
        {references.length > 0 && <optgroup label={t("Mô hình tham chiếu")}>
          {references.map((name) => <option key={name} value={name}>{ALGORITHM_NAMES[name]}</option>)}
        </optgroup>}
      </Select>
    </div>
    <Link className="rail-link" to={user ? "/watch-history" : `/login?next=${next}`}>
      {user ? t("Lịch sử của bạn") : t("Đăng nhập")}<ArrowRight size={15} />
    </Link>
  </div>;
  return <div className="recommendation-section" data-mode={pending || authLoading ? "loading" : data?.mode || "loading"}>
    <div className="recommendation-label"><Sparkles size={13} /> {t("DÀNH CHO BẠN")}
      {data?.mode === "model" && <span>{t("Điểm dự đoán /5")}</span>}
      {user && data?.mode === "content" && <span>{t("Theo lịch sử đánh giá")}</span>}
      {user?.profile_type === "dataset" && <span>{t("Hồ sơ demo · User {id}", { id: user.dataset_user_id })}</span>}
    </div>
    {actions}
    {data?.model_status === "no_known_history_items" && !pending && <p className="rail-subtitle" role="note">
      {t("Các phim bạn đã chấm chưa có trong dữ liệu mô hình đã học. Danh sách tạm thời dựa trên thể loại hoặc cộng đồng; chưa có điểm dự đoán cá nhân.")}
    </p>}
    {data && data.history_count === 0 && !pending && <div className="recommendation-onboarding" role="note">
      <Sparkles size={18} aria-hidden="true" /><div>
        <strong>{t("Bắt đầu khám phá gu phim của bạn")}</strong>
        <p>{t("Chưa có đánh giá nên danh sách hiện dựa trên điểm cộng đồng. Đổi thuật toán lúc này vẫn dùng gợi ý ban đầu. Hãy chấm điểm 5 phim đã xem để cung cấp thêm thông tin về sở thích; gợi ý sẽ cập nhật sau mỗi lần chấm.")}</p>
        <Link to={user ? "/gallery#catalog-start" : `/login?next=${next}`} onClick={() => {
          if (user && location.pathname === "/gallery") scrollToElement(document.getElementById("catalog-start"));
        }}>{t(user ? "Chọn phim để đánh giá" : "Đăng nhập để bắt đầu")}<ArrowRight size={14} /></Link>
      </div>
    </div>}
    {pending || authLoading || (!data && !error) ? <>
      <div className="rail-heading"><div><SectionTitle>{t("Các phim có thể bạn yêu thích")}</SectionTitle><p className="rail-subtitle">{t("Đang chuẩn bị danh sách cho bạn...")}</p></div></div>
      <RailSkeleton />
    </> : error ? <div className="rail-error" role="alert"><SectionTitle>{t("Các phim có thể bạn yêu thích")}</SectionTitle><p>{error}</p><button className="button button-outline" onClick={() => setAttempt(attempt + 1)}>{t("Thử lại gợi ý")}</button></div>
    : data?.items.length ? <MovieRail title={t("Các phim có thể bạn yêu thích")} subtitle={subtitle}
      movies={data.items.map(movieFromRecord)}
      predictions={Object.fromEntries(data.items.map((movie) => [movie.movieId, movie.predicted_rating]))} />
    : <div className="rail-empty"><SectionTitle>{t("Các phim có thể bạn yêu thích")}</SectionTitle><p>{t("Chưa có phim gợi ý phù hợp. Hãy khám phá danh mục hoặc cập nhật đánh giá của bạn.")}</p></div>}
  </div>;
}
