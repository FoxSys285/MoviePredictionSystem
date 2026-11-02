import { Link } from "react-router-dom";
import { Star } from "lucide-react";
import type { CSSProperties } from "react";
import { numberFormat, scoreFormat, type Movie } from "../data/movies";
import { posterPattern } from "../data/posterPatterns";
import { useCardMotion } from "../lib/motion";
import { MovieArtwork } from "./MovieArtwork";
import { usePreferences } from "../lib/preferences";

export function MovieCard({ movie }: { movie: Movie }) {
  const { t } = usePreferences();
  const motion = useCardMotion<HTMLAnchorElement>();
  return (
    <Link {...motion} className="movie-card" to={`/movies/${movie.id}`}
      style={{ "--poster-accent": posterPattern(movie).primary } as CSSProperties}
      aria-label={`${t("Xem chi tiết phim")} ${movie.title}`}>
      <MovieArtwork movie={movie} />
      <div className="movie-card-copy">
        <h3>{movie.title}</h3>
        <div className="movie-meta">
          <span><Star size={14} fill="currentColor" />
            {movie.rating !== null ? `${scoreFormat(movie.rating)}/5` : t("Chưa có điểm")}
          </span>
          <span>{movie.year || "—"}</span>
        </div>
        <p className="movie-votes">{numberFormat(movie.ratingCount)} {t("lượt đánh giá")}</p>
      </div>
    </Link>
  );
}
