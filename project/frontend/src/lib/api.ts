import type { CatalogMovie } from "../data/movies";

const API_ROOT = import.meta.env.VITE_API_URL || "/api/v1";

export const apiResourceUrl = (path: string) => `${API_ROOT}${path}`;

export class ApiError extends Error {
  constructor(message: string, public status: number) { super(message); }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_ROOT}${path}`, {
      credentials: "include",
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new Error("Chưa kết nối được máy chủ. Vui lòng thử lại sau.");
  }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = typeof data?.detail === "string" ? data.detail : undefined;
    if (response.status === 503 && path === "/predictions") {
      throw new Error(
        "Mô hình chưa được huấn luyện. Bạn có thể khám phá phim và quay lại dự đoán sau.",
      );
    }
    throw new ApiError(detail || "Yêu cầu chưa thực hiện được. Vui lòng thử lại.", response.status);
  }
  return data as T;
}

export type User = { id: number; name: string; email: string;
  profile_type: "personal" | "dataset"; dataset_user_id: number | null;
  imported_rating_count: number; profile_modified: boolean };
export type DemoAccount = { name: string; email: string; password: string;
  dataset_user_id: number; imported_rating_count: number };
export type DemoAccounts = { accounts: DemoAccount[]; notice: string; source: string };
export const PRIMARY_ALGORITHMS = ["ridge", "xgboost", "svd", "ncf"] as const;
export type Algorithm = typeof PRIMARY_ALGORITHMS[number] | "baseline" | "item_knn" | "nmf";
export type Prediction = {
  predicted_rating: number;
  algorithm: Algorithm;
  cold_start: boolean;
  run_id: string;
};
export type Metric = {
  algorithm: Algorithm;
  rmse: number;
  mae: number;
  fit_seconds: number;
};
export type Experiment = {
  validation: { best_algorithm: Algorithm; results: Metric[] };
  test: { results: Metric[] } | null;
};

export type RatingBucket = { rating: number; count: number };
export type Dataset = {
  name: string;
  movie_count: number;
  rated_movie_count: number;
  unrated_movie_count: number;
  rating_count: number;
  user_count: number;
  mean_rating: number;
  tag_count: number;
  link_count: number;
  genre_count: number;
  genres: { name: string; movie_count: number }[];
  year_min: number | null;
  year_max: number | null;
  rating_start: string;
  rating_end: string;
  matrix_density: number;
  rating_distribution: RatingBucket[];
  quality: {
    movies_without_year: number;
    movies_without_genres: number;
    ratings_without_movie: number;
    empty_tags: number;
  };
  popular_movies: CatalogMovie[];
};

export type MovieDetails = CatalogMovie & {
  tags: { tag: string; count: number }[];
  rating_distribution: RatingBucket[];
  imdb_url: string | null;
  tmdb_url: string | null;
  related: CatalogMovie[];
};

export type PersonalRating = { movie_id: number; rating: number | null; updated_at: number | null };
export type HistoryMovie = CatalogMovie & { user_rating: number; updated_at: number };
export type HistoryResponse = { items: HistoryMovie[]; total: number; offset: number; limit: number };
export type RecommendedMovie = CatalogMovie & { predicted_rating: number | null };
export type Recommendations = {
  items: RecommendedMovie[];
  mode: "popular" | "content" | "model";
  authenticated: boolean;
  algorithm: Algorithm | null;
  run_id: string | null;
  history_count: number;
  model_status: "not_trained" | "ready" | "no_known_history_items";
  available_algorithms: Algorithm[];
  requested_algorithm: Algorithm | "auto";
  cold_start: boolean;
  strategy: "community_popularity" | "genre_profile" | "trained_dataset_user" | "profile_fold_in";
  profile_type: "personal" | "dataset";
  dataset_user_id: number | null;
};
export type GenreShelf = { name: string; movie_count: number; movies: CatalogMovie[] };

export const ALGORITHM_NAMES: Record<Algorithm, string> = {
  ridge: "Linear / Ridge",
  xgboost: "XGBoost",
  ncf: "NCF",
  baseline: "Baseline Bias",
  item_knn: "Item-KNN",
  svd: "SVD",
  nmf: "NMF",
};
