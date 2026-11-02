export type CatalogMovie = {
  movieId: number;
  title: string;
  genres: string;
  year: number | null;
  mean_rating: number | null;
  rating_count: number;
  imdbId: string | null;
  tmdbId: number | null;
};

export type Movie = {
  id: number;
  title: string;
  year: number | null;
  rating: number | null;
  ratingCount: number;
  genres: string[];
  poster?: string;
};

// Illustrations are mapped only to verified matching MovieLens IDs.
const posters: Record<number, string> = {
  79132: "inception", 109487: "interstellar", 58559: "dark-knight",
  2959: "fight-club", 134130: "martian", 858: "godfather",
  136864: "batman-superman", 4993: "lord-rings", 122912: "avengers",
  204698: "joker",
};

export const genreLabels: Record<string, string> = {
  Action: "Hành động", Adventure: "Phiêu lưu", Animation: "Hoạt hình",
  Children: "Thiếu nhi", Comedy: "Hài hước", Crime: "Tội phạm",
  Documentary: "Tài liệu", Drama: "Chính kịch", Fantasy: "Kỳ ảo",
  "Film-Noir": "Phim noir", Horror: "Kinh dị", IMAX: "IMAX",
  Musical: "Nhạc kịch", Mystery: "Bí ẩn", Romance: "Tình cảm",
  "Sci-Fi": "Khoa học viễn tưởng", Thriller: "Giật gân", War: "Chiến tranh",
  Western: "Viễn Tây", "(no genres listed)": "Chưa có thể loại",
};

export function movieFromRecord(item: CatalogMovie): Movie {
  return {
    id: item.movieId,
    title: item.title.replace(/\s*\(\d{4}\)\s*$/, ""),
    year: item.year,
    rating: item.mean_rating,
    ratingCount: item.rating_count,
    genres: item.genres.split("|"),
    poster: posters[item.movieId],
  };
}

const numberLocale = () => typeof document !== "undefined" && document.documentElement.lang === "en" ? "en-US" : "vi-VN";
export const numberFormat = (value: number) => value.toLocaleString(numberLocale());
export const scoreFormat = (value: number) => value.toLocaleString(numberLocale(), {
  minimumFractionDigits: 1, maximumFractionDigits: 2,
});
