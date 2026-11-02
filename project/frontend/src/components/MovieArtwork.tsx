import { useState, type CSSProperties } from "react";
import { ArrowUpRight } from "lucide-react";
import type { Movie } from "../data/movies";
import { posterPattern } from "../data/posterPatterns";
import { PosterPattern } from "./PosterPattern";
import { usePreferences } from "../lib/preferences";

export function MovieArtwork({ movie }: { movie: Movie }) {
  const { t } = usePreferences();
  const [failedSource, setFailedSource] = useState("");
  const [loadedSource, setLoadedSource] = useState("");
  const source = movie.poster ? `/assets/posters/${movie.poster}.webp` : "";
  const hasImage = Boolean(source) && failedSource !== source;
  const palette = posterPattern(movie);
  return (
    <div
      className={`poster-wrap ${hasImage ? "has-poster" : "has-pattern"} ${loadedSource === source && source ? "is-loaded" : ""}`}
      data-artwork={hasImage ? "image" : "pattern"}
      style={{ "--poster-accent": palette.primary } as CSSProperties}
      role="img"
      aria-label={`${t(hasImage ? "Tranh minh họa" : "Bìa đồ họa")} ${movie.title}`}
    >
      <PosterPattern movie={movie} />
      {hasImage && <img
        className="poster-image"
        src={source}
        alt=""
        aria-hidden="true"
        loading="lazy"
        onLoad={() => setLoadedSource(source)}
        onError={() => setFailedSource(source)}
      />}
      <div className="poster-vignette" />
      {hasImage && <span className="poster-tagline">{t("TRANH MINH HỌA")}</span>}
      <span className={`poster-title ${hasImage ? `poster-title-${movie.poster}` : "poster-title-pattern"}`}>{movie.title}</span>
      <span className="poster-hover"><span>{t("Xem chi tiết")} <ArrowUpRight size={17} /></span></span>
    </div>
  );
}
