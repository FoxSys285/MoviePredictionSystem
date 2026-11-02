import { useEffect, useRef, useState } from "react";
import { useLocation, useSearchParams } from "react-router-dom";
import { ChevronLeft, ChevronRight, LoaderCircle, Search, X } from "lucide-react";
import { PageHero, Select } from "../components/ui";
import { MovieCard } from "../components/MovieCard";
import { genreLabels, movieFromRecord, numberFormat, type CatalogMovie, type Movie } from "../data/movies";
import { api, type Dataset } from "../lib/api";
import { scrollToElement } from "../lib/motion";
import { Recommendations } from "../components/Recommendations";
import { SectionTitle } from "../components/ui";
import { usePreferences } from "../lib/preferences";

type CatalogResponse = { total: number; items: CatalogMovie[] };

export function Gallery({ dataset }: { dataset: Dataset | null }) {
  const { t, language } = usePreferences();
  const [params, setParams] = useSearchParams();
  const location = useLocation();
  useEffect(() => {
    if (location.hash === "#catalog-start") scrollToElement(document.getElementById("catalog-start"));
  }, [location.hash]);
  const query = params.get("q") || "";
  const genre = params.get("genre") || "all";
  const year = params.get("year") || "all";
  const rating = params.get("rating") || "all";
  const sort = params.get("sort") || "popular";
  const page = Math.max(1, parseInt(params.get("page") || "1", 10) || 1);
  const [pending, setPending] = useState(true);
  const [catalog, setCatalog] = useState<Movie[]>([]);
  const [total, setTotal] = useState(0);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  const searchRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (params.get("focus") === "search") searchRef.current?.focus();
  }, [params]);

  function setFilter(name: string, value: string) {
    // History changes immediately; a previous React render can still hold old filters.
    const next = new URLSearchParams(window.location.search);
    next.delete("focus");
    if (value === "all" || !value) next.delete(name);
    else next.set(name, value);
    if (name !== "page") next.delete("page");
    setParams(next, { replace: true });
    if (name === "page") scrollToElement(document.getElementById("catalog-start"));
  }

  useEffect(() => {
    const controller = new AbortController();
    setPending(true);
    setError("");
    const timer = window.setTimeout(async () => {
      const filters = new URLSearchParams({
        q: query, offset: String((page - 1) * 12), limit: "12", genre, year,
        min_rating: rating === "all" ? "0" : rating, sort,
      });
      try {
        const data = await api<CatalogResponse>(`/movies?${filters}`, { signal: controller.signal });
        if (controller.signal.aborted) return;
        setCatalog(data.items.map(movieFromRecord));
        setTotal(data.total);
      } catch (cause) {
        if (!controller.signal.aborted) setError((cause as Error).message);
      } finally {
        if (!controller.signal.aborted) setPending(false);
      }
    }, 200);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [query, genre, year, rating, sort, page, attempt]);

  const genres = dataset?.genres.map((item) => item.name) || Object.keys(genreLabels).filter((name) => name !== "(no genres listed)");
  const options = ["all", ...genres, ...(dataset?.quality.movies_without_genres ? ["(no genres listed)"] : [])];
  const chips = ["all", ...["Action", "Horror", "Romance", "Comedy", "Sci-Fi", "Adventure", "Animation"].filter((name) => genres.includes(name))];
  const pages = Math.max(1, Math.ceil(total / 12));
  const reset = () => setParams({}, { replace: true });
  const activeFilters = [
    ...(query ? [{ key: "q", label: `“${query}”` }] : []),
    ...(genre !== "all" ? [{ key: "genre", label: t(genreLabels[genre] || genre) }] : []),
    ...(year !== "all" ? [{ key: "year", label: year === "classic" ? t("Trước 2000") : `${year}–${Number(year) + 9}` }] : []),
    ...(rating !== "all" ? [{ key: "rating", label: t("Từ {rating}/5", { rating: rating.replace(".", language === "vi" ? "," : ".") }) }] : []),
  ];

  return (
    <>
      <PageHero title={t("BỘ SƯU TẬP")} subtitle={t("Khám phá bộ sưu tập MovieLens")} variant="gallery" />
      <main id="main" className="container gallery-main">
        <Recommendations />
        <div className="catalog-section-heading"><SectionTitle>{t("Khám phá toàn bộ phim")}</SectionTitle></div>
        <div id="catalog-start" className="filter-toolbar">
          <label className="search-field">
            <Search size={18} />
            <input ref={searchRef} maxLength={200} aria-label={t("Tìm kiếm tên phim")} aria-keyshortcuts="/" placeholder={t("Tìm kiếm phim...")}
              value={query} onChange={(event) => setFilter("q", event.target.value)} />
            {!query && <kbd className="search-shortcut" aria-hidden="true">/</kbd>}
            {query && <button className="clear-search" onClick={() => setFilter("q", "")} aria-label={t("Xóa tìm kiếm")}><X size={15} /></button>}
          </label>
          <Select label={t("Thể loại")} value={genre} onChange={(value) => setFilter("genre", value)}>
            {options.map((value) => <option key={value} value={value}>{value === "all" ? t("Thể loại") : t(genreLabels[value] || value)}</option>)}
          </Select>
          <Select label={t("Năm phát hành")} value={year} onChange={(value) => setFilter("year", value)}>
            <option value="all">{t("Năm phát hành")}</option>
            {(dataset?.year_max || 0) >= 2020 && <option value="2020">2020 – 2029</option>}
            <option value="2010">2010 – 2019</option><option value="2000">2000 – 2009</option><option value="classic">{t("Trước 2000")}</option>
          </Select>
          <Select label={t("Đánh giá")} value={rating} onChange={(value) => setFilter("rating", value)}>
            <option value="all">{t("Đánh giá /5")}</option><option value="4.5">{t("Từ 4,5 ★")}</option><option value="4">{t("Từ 4,0 ★")}</option><option value="3.5">{t("Từ 3,5 ★")}</option>
          </Select>
          <Select label={t("Sắp xếp phim")} value={sort} onChange={(value) => setFilter("sort", value)} className="sort-select">
            <option value="popular">{t("Sắp xếp: Phổ biến nhất")}</option><option value="rating">{t("Đánh giá cao nhất")}</option><option value="newest">{t("Mới nhất trong dataset")}</option><option value="title">{t("Tên phim A – Z")}</option>
          </Select>
        </div>
        {activeFilters.length > 0 && <div className="active-filters" aria-label={t("Bộ lọc đang áp dụng")}>
          <span>{t("Đang lọc")}</span>
          {activeFilters.map((filter) => <button key={filter.key} onClick={() => setFilter(filter.key, "all")}
            aria-label={`${t("Bỏ lọc")} ${filter.label}`}>{filter.label}<X size={12} /></button>)}
          <button className="reset-filters" onClick={reset}>{t("Xóa tất cả")}</button>
        </div>}
        <div className="genre-chips" role="group" aria-label={t("Lọc theo thể loại")}>
          {chips.map((value) => <button key={value} className={genre === value ? "selected" : ""} aria-pressed={genre === value}
            onClick={() => setFilter("genre", value)}>{value === "all" ? t("Tất cả") : t(genreLabels[value] || value)}</button>)}
        </div>
        <div className="catalog-caption" aria-live="polite">
          <span>{pending ? <><LoaderCircle size={12} className="spin" /> {t("Đang tải bộ sưu tập...")}</> : error ? t("Chưa tải được bộ sưu tập") : `${numberFormat(total)} ${t("bộ phim")} · MovieLens`}</span>
          <span>{t("Điểm trung bình /5 · Toàn bộ lượt đánh giá")}</span>
        </div>
        {pending ? (
          <div className="movie-grid skeleton-grid" role="status" aria-label={t("Đang tải phim từ dataset")}>
            {Array.from({ length: 12 }, (_, index) => <div className="movie-skeleton" aria-hidden="true" key={index}>
              <div className="skeleton-poster" /><div className="skeleton-copy"><span /><span /></div>
            </div>)}
          </div>
        ) : error ? (
          <div className="empty-state" role="alert"><h2>{t("Chưa thể tải danh mục phim")}</h2><p>{error}</p><button className="button" onClick={() => setAttempt(attempt + 1)}>{t("Thử lại")}</button></div>
        ) : catalog.length ? (
          <div className="movie-grid catalog-grid">{catalog.map((movie) => <MovieCard key={movie.id} movie={movie} />)}</div>
        ) : (
          <div className="empty-state"><Search size={35} /><h2>{t("Chưa tìm thấy bộ phim phù hợp")}</h2><p>{t("Thử một tên phim hoặc thể loại khác để khám phá thêm.")}</p><button className="button button-outline" onClick={reset}>{t("Xóa bộ lọc")}</button></div>
        )}
        {!error && !pending && <nav className="pagination" aria-label={t("Phân trang phim")}>
          <button disabled={page <= 1} aria-label={t("Trang trước")} onClick={() => setFilter("page", String(page - 1))}><ChevronLeft size={17} /></button>
          {Array.from({ length: Math.min(5, pages) }, (_, index) => {
            const number = Math.max(1, Math.min(page - 2, pages - 4)) + index;
            return <button key={number} className={page === number ? "active" : ""} aria-current={page === number ? "page" : undefined}
              onClick={() => setFilter("page", String(number))}>{number}</button>;
          })}
          <button disabled={page >= pages} aria-label={t("Trang tiếp theo")} onClick={() => setFilter("page", String(page + 1))}><ChevronRight size={17} /></button>
        </nav>}
      </main>
    </>
  );
}
