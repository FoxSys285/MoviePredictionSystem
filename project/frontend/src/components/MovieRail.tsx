import { useEffect, useRef, useState, type ReactNode } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { MovieCard } from "./MovieCard";
import { SectionTitle } from "./ui";
import { reducedMotion } from "../lib/motion";
import { scoreFormat, type Movie } from "../data/movies";
import { usePreferences } from "../lib/preferences";

export function MovieRail({ title, subtitle, movies, actions, predictions }: {
  title: string;
  subtitle?: string;
  movies: Movie[];
  actions?: ReactNode;
  predictions?: Record<number, number | null>;
}) {
  const { t } = usePreferences();
  const ref = useRef<HTMLDivElement>(null);
  const [edges, setEdges] = useState({ start: true, end: true });
  const identity = movies.map((movie) => movie.id).join(",");
  useEffect(() => {
    const rail = ref.current;
    if (!rail) return;
    let frame = 0;
    function update() {
      frame = 0;
      const next = { start: rail!.scrollLeft < 3,
        end: rail!.scrollLeft + rail!.clientWidth >= rail!.scrollWidth - 3 };
      setEdges((old) => old.start === next.start && old.end === next.end ? old : next);
    }
    const schedule = () => { if (!frame) frame = requestAnimationFrame(update); };
    rail.scrollTo({ left: 0, behavior: "instant" });
    update();
    const resize = new ResizeObserver(schedule);
    resize.observe(rail);
    rail.addEventListener("scroll", schedule, { passive: true });
    return () => { resize.disconnect(); rail.removeEventListener("scroll", schedule); cancelAnimationFrame(frame); };
  }, [identity]);
  function move(direction: number) {
    ref.current?.scrollBy({ left: direction * ref.current.clientWidth * .82,
      behavior: reducedMotion() ? "instant" : "smooth" });
  }
  return <section className="movie-rail-section" aria-label={title}>
    <div className="rail-heading">
      <div><SectionTitle>{title}</SectionTitle>{subtitle && <p className="rail-subtitle">{subtitle}</p>}</div>
      <div className="rail-actions">{actions}
        <div className="rail-buttons">
          <button className="rail-arrow" disabled={edges.start} onClick={() => move(-1)} aria-label={`${t("Cuộn trái")} ${title}`}><ChevronLeft size={18} /></button>
          <button className="rail-arrow" disabled={edges.end} onClick={() => move(1)} aria-label={`${t("Cuộn phải")} ${title}`}><ChevronRight size={18} /></button>
        </div>
      </div>
    </div>
    <div ref={ref} className="movie-rail" tabIndex={0} aria-label={`${t("Danh sách phim")} ${title}`}>
      {movies.map((movie) => <div className="rail-item" key={movie.id}>
        <MovieCard movie={movie} />
        {predictions?.[movie.id] != null && <span className="prediction-caption">{t("Dự đoán cho bạn")} <b>{scoreFormat(predictions[movie.id]!)} /5</b></span>}
      </div>)}
    </div>
  </section>;
}

export function RailSkeleton() {
  const { t } = usePreferences();
  return <div className="rail-skeleton" role="status" aria-label={t("Đang tải danh sách phim")}>
    {Array.from({ length: 6 }, (_, index) => <div className="movie-skeleton" aria-hidden="true" key={index}>
      <div className="skeleton-poster" /><div className="skeleton-copy"><span /><span /></div>
    </div>)}
  </div>;
}
