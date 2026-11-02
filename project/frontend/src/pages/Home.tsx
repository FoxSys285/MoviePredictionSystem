import { Link } from "react-router-dom";
import { ArrowRight, BrainCircuit, Play } from "lucide-react";
import { SectionTitle } from "../components/ui";
import { MovieCard } from "../components/MovieCard";
import { movieFromRecord, numberFormat } from "../data/movies";
import type { Dataset } from "../lib/api";
import { Recommendations } from "../components/Recommendations";
import { GenreShelves } from "../components/GenreShelves";
import { usePreferences } from "../lib/preferences";

export function Home({
  dataset,
  onSettings,
}: {
  dataset: Dataset | null;
  onSettings: () => void;
}) {
  const { t } = usePreferences();
  return (
    <main id="main">
      <section className="home-hero">
        <div className="container home-hero-content">
          <span className="eyebrow">
            <span /> {t("ĐIỆN ẢNH GẶP GỠ CÔNG NGHỆ")}
          </span>
          <h1>
            {t("MỖI BỘ PHIM.")}
            <br />
            {t("MỘT THẾ GIỚI.")}
            <br />
            <span>{t("DÀNH CHO BẠN.")}</span>
          </h1>
          <p>
            {t("Khám phá những câu chuyện chạm đến cảm xúc.")}
            <br />
            {t("Để học máy giúp bạn tìm bộ phim tiếp theo.")}
          </p>
          <div className="home-buttons">
            <Link to="/gallery" className="button">
              <Play size={17} fill="currentColor" /> {t("Khám phá phim")}
            </Link>
            <button className="button button-outline" onClick={onSettings}>
              <BrainCircuit size={18} /> {t("Công nghệ dự đoán")}
            </button>
          </div>
          <div className="home-stats">
            <div>
              <strong>{dataset ? numberFormat(dataset.rating_count) : "—"}</strong>
              <span>{t("lượt đánh giá")}</span>
            </div>
            <div>
              <strong>{dataset ? numberFormat(dataset.movie_count) : "—"}</strong>
              <span>{t("bộ phim")}</span>
            </div>
            <div>
              <strong>{dataset ? numberFormat(dataset.user_count) : "—"}</strong>
              <span>{t("người dùng MovieLens")}</span>
            </div>
          </div>
          <p className="home-source">{dataset ? t("Dữ liệu thực tế từ MovieLens · Điểm 0,5–5 · Phim từ {start} đến {end}.", { start: dataset.year_min, end: dataset.year_max }) : t("Đang tải thống kê dataset MovieLens...")}</p>
        </div>
        <a className="home-scroll" href="#featured-movies" aria-label={t("Cuộn đến phim nổi bật")}>
          {t("MỘT TRẢI NGHIỆM ĐIỆN ẢNH MỚI")} <span>↓</span>
        </a>
      </section>
      <div className="container home-recommendations"><Recommendations /></div>
      <section id="featured-movies" className="container home-collection">
        <div className="section-heading">
          <SectionTitle>{t("Phim được đánh giá nhiều nhất")}</SectionTitle>
          <Link to="/gallery">
            {t("Xem bộ sưu tập")} <ArrowRight size={16} />
          </Link>
        </div>
        <div className="movie-grid">
          {dataset?.popular_movies.map((item) => (
            <MovieCard key={item.movieId} movie={movieFromRecord(item)} />
          ))}
        </div>
        <p className="subtle-note">
          {t("Xếp hạng theo số lượt đánh giá trong dataset · Điểm trung bình /5.")}
        </p>
      </section>
      <section className="container home-discovery"><GenreShelves /></section>
    </main>
  );
}
