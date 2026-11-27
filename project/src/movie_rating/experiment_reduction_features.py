"""Train-only movie metadata PCA representation for rating regression."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class MovieFeatureRepresentation:
    """One movie per row: multi-hot genres, release year, and missing-year flag."""

    def fit(self, frame, movies, components=None):
        if movies.movieId.duplicated().any():
            raise ValueError("Movie metadata must have unique movieId values.")
        self.metadata_ = movies.set_index("movieId")
        self.fit_movie_ids_ = np.sort(frame.movieId.unique())
        fit_metadata = self.metadata_.reindex(self.fit_movie_ids_)
        self.genres_ = sorted({genre for value in fit_metadata.genres.fillna("")
                               for genre in value.split("|")
                               if genre and genre != "(no genres listed)"})
        years = self._years(fit_metadata)
        self.year_fill_ = float(years.median()) if years.notna().any() else 0.0
        table = self._table(self.fit_movie_ids_)
        self.feature_names_ = [*self.genres_, "release_year", "missing_release_year"]
        self.scaler_ = StandardScaler().fit(table)
        self.reducer_ = None
        if components is not None:
            maximum = min(len(table) - 1, table.shape[1])
            if not 1 <= components <= maximum:
                raise ValueError(f"Movie PCA components must be between 1 and {maximum}.")
            self.reducer_ = PCA(n_components=components, svd_solver="full")
            self.reducer_.fit(self.scaler_.transform(table))
        return self

    @staticmethod
    def _years(metadata):
        if "title" not in metadata:
            return pd.Series(np.nan, index=metadata.index)
        return pd.to_numeric(metadata.title.fillna("").str.extract(r"\((\d{4})\)\s*$")[0],
                             errors="coerce")

    def _table(self, movie_ids):
        metadata = self.metadata_.reindex(movie_ids)
        genres = metadata.genres.fillna("").str.split("|")
        years = self._years(metadata)
        genre_columns = np.asarray([[float(g in values) for g in self.genres_]
                                    for values in genres], dtype=np.float32).reshape(
                                        len(metadata), len(self.genres_))
        return np.column_stack((genre_columns, years.fillna(self.year_fill_).to_numpy(),
                                years.isna().to_numpy())).astype(np.float32)

    def transform(self, frame):
        # Transform each unique movie once, then map back to rating rows.
        ids, inverse = np.unique(frame.movieId.to_numpy(), return_inverse=True)
        table = self.scaler_.transform(self._table(ids))
        if self.reducer_ is not None:
            table = self.reducer_.transform(table)
        return table[inverse]
