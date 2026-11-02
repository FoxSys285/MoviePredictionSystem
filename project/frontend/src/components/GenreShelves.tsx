import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, Grid2X2 } from "lucide-react";
import { api, type GenreShelf } from "../lib/api";
import { genreLabels, movieFromRecord, numberFormat } from "../data/movies";
import { MovieRail, RailSkeleton } from "./MovieRail";
import { SectionTitle } from "./ui";
import { usePreferences } from "../lib/preferences";

export function GenreShelves() {
  const { t } = usePreferences();
  const [shelves, setShelves] = useState<GenreShelf[]>([]);
  const [pending, setPending] = useState(true);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setPending(true); setError("");
    api<{ items: GenreShelf[] }>("/discovery/genres?limit=5&movies_per_genre=10", { signal: controller.signal })
      .then((data) => { if (!controller.signal.aborted) setShelves(data.items); })
      .catch((cause: Error) => { if (!controller.signal.aborted) setError(cause.message); })
      .finally(() => { if (!controller.signal.aborted) setPending(false); });
    return () => controller.abort();
  }, [attempt]);
  return <div className="home-genre-shelves" aria-label={t("Phim theo thể loại")}>
    {pending ? <><SectionTitle>{t("Khám phá theo thể loại")}</SectionTitle><RailSkeleton /></>
      : error ? <div className="rail-error" role="alert"><p>{error}</p><button className="button button-outline" onClick={() => setAttempt(attempt + 1)}>{t("Thử lại phim theo thể loại")}</button></div>
      : shelves.map((shelf, index) => <MovieRail key={shelf.name} title={t(genreLabels[shelf.name] || shelf.name)}
        subtitle={`${numberFormat(shelf.movie_count)} ${t("phim")} · ${t("Những phim được đánh giá nhiều trong thể loại")}`}
        movies={shelf.movies.map(movieFromRecord)} actions={<>
          {index === 0 && <Link className="all-genres-link" to="/genres"><Grid2X2 size={14} />{t("Tất cả thể loại")}</Link>}
          <Link className="rail-link" to={`/gallery?genre=${encodeURIComponent(shelf.name)}`}>{t("Xem tất cả")}<ArrowRight size={15} /></Link>
        </>} />)}
  </div>;
}
