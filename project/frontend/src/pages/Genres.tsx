import { useEffect, useState, type CSSProperties } from "react";
import { Link } from "react-router-dom";
import { ArrowUpRight, Clapperboard, LoaderCircle, Search } from "lucide-react";
import { api, type Dataset } from "../lib/api";
import { genreLabels, numberFormat } from "../data/movies";
import { PageHero } from "../components/ui";
import { usePreferences } from "../lib/preferences";

const normalize = (value: string) => value.toLocaleLowerCase("vi-VN").normalize("NFD").replace(/\p{M}/gu, "").replace(/đ/g, "d");

export function Genres({ dataset }: { dataset: Dataset | null }) {
  const { t } = usePreferences();
  const [loaded, setLoaded] = useState<Dataset | null>(null);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (dataset) return;
    const controller = new AbortController();
    setError("");
    api<Dataset>("/dataset", { signal: controller.signal })
      .then((value) => { if (!controller.signal.aborted) setLoaded(value); })
      .catch((cause: Error) => { if (!controller.signal.aborted) setError(cause.message); });
    return () => controller.abort();
  }, [dataset, attempt]);
  const data = dataset || loaded;
  const genres = [...(data?.genres || []), ...(data?.quality.movies_without_genres ? [{ name: "(no genres listed)", movie_count: data.quality.movies_without_genres }] : [])];
  const filtered = genres.filter((genre) => normalize(`${genreLabels[genre.name] || genre.name} ${genre.name}`).includes(normalize(query.trim())));
  return <>
    <PageHero title={t("THỂ LOẠI")} subtitle={t("Mỗi thể loại, một thế giới điện ảnh")} variant="genres" />
    <main id="main" className="container genres-main">
      <div className="genres-heading"><div><span className="eyebrow">{t("TÌM CẢM HỨNG TIẾP THEO")}</span><h1>{t("Tất cả thể loại")}</h1><p>{t("Chọn một thể loại để mở Gallery với bộ lọc tương ứng.")}</p></div>
        <label className="search-field"><Search size={17} /><input aria-label={t("Tìm kiếm thể loại")} placeholder={t("Tìm thể loại...")} value={query} onChange={(event) => setQuery(event.target.value)} /></label>
      </div>
      {error && !data ? <div className="rail-error" role="alert"><p>{error}</p><button className="button" onClick={() => setAttempt(attempt + 1)}>{t("Thử lại thể loại")}</button></div>
        : !data ? <div className="empty-state" role="status"><LoaderCircle className="spin" size={30} /><p>{t("Đang tải các thể loại...")}</p></div>
        : !filtered.length ? <div className="empty-state"><Search size={35} /><h2>{t("Chưa tìm thấy thể loại phù hợp")}</h2><button className="button button-outline" onClick={() => setQuery("")}>{t("Xóa tìm thể loại")}</button></div>
        : <div className="genre-directory">{filtered.map((genre) => <Link key={genre.name} className="genre-tile"
          style={{ "--genre-hue": (genres.findIndex((item) => item.name === genre.name) * 47 + 12) % 360 } as CSSProperties}
          to={`/gallery?genre=${encodeURIComponent(genre.name)}`} aria-label={`${t("Khám phá thể loại")} ${t(genreLabels[genre.name] || genre.name)}`}>
          <div className="genre-tile-art" aria-hidden="true"><span /><span /><span /><Clapperboard size={31} /></div>
          <span className="genre-tile-count">{numberFormat(genre.movie_count)} {t("phim")}</span>
          <div className="genre-tile-copy"><h2>{t(genreLabels[genre.name] || genre.name)}</h2><span>{genre.name === "(no genres listed)" ? t("Chưa được phân loại") : t(genre.name)}</span></div>
          <ArrowUpRight className="genre-tile-arrow" size={20} />
        </Link>)}</div>}
    </main>
  </>;
}
