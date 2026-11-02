import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft, ExternalLink, Film, LoaderCircle, Star } from "lucide-react";
import { api, type MovieDetails } from "../lib/api";
import { genreLabels, movieFromRecord, numberFormat, scoreFormat } from "../data/movies";
import { MovieCard } from "./MovieCard";
import { MovieArtwork } from "./MovieArtwork";
import { SectionTitle } from "./ui";
import { MovieRating } from "./MovieRating";
import { usePreferences } from "../lib/preferences";

export function MovieDetail() {
  const { t } = usePreferences();
  const { movieId } = useParams();
  const [detail, setDetail] = useState<MovieDetails | null>(null);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setDetail(null);
    setError("");
    api<MovieDetails>(`/movies/${encodeURIComponent(movieId || "")}`, { signal: controller.signal })
      .then((data) => {
        if (controller.signal.aborted) return;
        setDetail(data);
        document.title = `MOVIE — ${movieFromRecord(data).title}`;
      })
      .catch((cause: Error) => { if (!controller.signal.aborted) setError(cause.message); });
    return () => controller.abort();
  }, [movieId, attempt]);

  const movie = detail && movieFromRecord(detail);
  const maxCount = Math.max(1, ...(detail?.rating_distribution.map((bucket) => bucket.count) || []));
  return (
    <main id="main" className="container movie-page">
      <nav className="movie-breadcrumb" aria-label={t("Đường dẫn")}>
        <Link to="/gallery"><ArrowLeft size={16} /> {t("Bộ sưu tập phim")}</Link>
        <span>/</span><span>{t("Chi tiết phim")}</span>
      </nav>
      {error ? (
        <div className="empty-state" role="alert">
          <Film size={36} /><h1>{t("Chưa thể hiển thị phim")}</h1><p>{error}</p>
          <button className="button button-outline" onClick={() => setAttempt(attempt + 1)}>{t("Thử lại")}</button>
          <Link className="button" to="/gallery">{t("Trở lại bộ sưu tập")}</Link>
        </div>
      ) : !movie || !detail ? (
        <div className="empty-state" role="status"><LoaderCircle className="spin" size={30} /><p>{t("Đang tải thông tin phim...")}</p></div>
      ) : (
        <>
          <section className="movie-overview">
            <div className="movie-page-art"><MovieArtwork movie={movie} /></div>
            <div className="movie-overview-copy">
              <span className="eyebrow">{t("MOVIELENS · CHI TIẾT PHIM")}</span>
              <h1>{movie.title}</h1>
              <p className="movie-original-title">{detail.title}</p>
              <div className="movie-genre-list">
                {movie.genres.map((genre) => <span key={genre}>{t(genreLabels[genre] || genre)}</span>)}
              </div>
              <div className="movie-rating-summary">
                <Star size={28} fill="currentColor" />
                <strong>{movie.rating === null ? "—" : scoreFormat(movie.rating)}<small>/5</small></strong>
                <div><b>{t("Điểm trung bình MovieLens")}</b><span>{numberFormat(movie.ratingCount)} {t("lượt đánh giá")}</span></div>
              </div>
              <dl className="movie-facts">
                <div><dt>{t("Năm phát hành")}</dt><dd>{movie.year || t("Chưa có thông tin")}</dd></div>
                <div><dt>{t("Mã phim MovieLens")}</dt><dd>#{movie.id}</dd></div>
                <div><dt>{t("Thang đánh giá")}</dt><dd>{t("0,5–5 · Bước 0,5")}</dd></div>
                <div><dt>{t("Nguồn điểm")}</dt><dd>{t("Toàn bộ ratings.csv")}</dd></div>
              </dl>
              <p className="movie-data-note">{t("Thông tin tên phim, năm và thể loại được lấy từ dataset. MovieLens không cung cấp nội dung cốt truyện, diễn viên hay thời lượng; bạn có thể xem thêm qua liên kết bên dưới.")}</p>
              <div className="movie-external-links">
                {detail.imdb_url && <a className="button button-outline" href={detail.imdb_url} target="_blank" rel="noreferrer">{t("Xem trên IMDb")} <ExternalLink size={15} /></a>}
                {detail.tmdb_url && <a className="button button-outline" href={detail.tmdb_url} target="_blank" rel="noreferrer">{t("Xem trên TMDB")} <ExternalLink size={15} /></a>}
              </div>
              {movie.poster && <p className="subtle-note">{t("Ảnh là tranh minh họa, không thuộc dataset.")}</p>}
            </div>
          </section>
          <MovieRating movieId={movie.id} />
          <div className="movie-insights">
            <section className="movie-insight-panel">
              <SectionTitle>{t("Phân bố điểm đánh giá")}</SectionTitle>
              <p className="subtle-note">{t("Số lượt đánh giá ở từng mức điểm trong MovieLens.")}</p>
              <div className="rating-histogram" role="img" aria-label={`Phân bố ${numberFormat(movie.ratingCount)} lượt đánh giá`}>
                {detail.rating_distribution.map((bucket) => (
                  <div className="rating-bar-row" key={bucket.rating}>
                    <span>{scoreFormat(bucket.rating)} ★</span>
                    <div className="rating-bar-track"><div style={{ width: `${bucket.count / maxCount * 100}%` }} /></div>
                    <span>{numberFormat(bucket.count)}</span>
                  </div>
                ))}
              </div>
              {!movie.ratingCount && <p className="subtle-note">{t("Phim chưa có lượt đánh giá trong dataset.")}</p>}
            </section>
            <section className="movie-insight-panel">
              <SectionTitle>{t("Từ khóa từ cộng đồng")}</SectionTitle>
              <p className="subtle-note">{t("Các nhãn được người dùng gắn trong tags.csv, kèm số lần xuất hiện.")}</p>
              <div className="movie-tag-list">
                {detail.tags.map(({ tag, count }) => <span key={tag}>{tag}<small>{numberFormat(count)}</small></span>)}
              </div>
              {!detail.tags.length && <p className="subtle-note">{t("Chưa có từ khóa cho phim này trong dataset.")}</p>}
            </section>
          </div>
          {detail.related.length > 0 && (
            <section className="movie-related">
              <SectionTitle>{t("Khám phá phim cùng thể loại")}</SectionTitle>
              <p className="subtle-note">{t("Ưu tiên phim có nhiều thể loại chung và nhiều lượt đánh giá.")}</p>
              <div className="movie-grid">{detail.related.map((item) => <MovieCard key={item.movieId} movie={movieFromRecord(item)} />)}</div>
            </section>
          )}
        </>
      )}
    </main>
  );
}
