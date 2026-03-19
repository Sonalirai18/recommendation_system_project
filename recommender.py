from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Recommendation:
    movie_id: int
    title: str
    predicted_score: float
    genres: str


class MovieRecommender:
    """A simple user-based collaborative filtering recommender system.

    The model loads ratings, creates a user-item matrix, computes user-user
    cosine similarity, and predicts scores for unseen movies using weighted
    ratings from similar users.
    """

    def __init__(self, ratings_path: str | Path, movies_path: str | Path) -> None:
        self.ratings_path = Path(ratings_path)
        self.movies_path = Path(movies_path)
        self.ratings = pd.read_csv(self.ratings_path)
        self.movies = pd.read_csv(self.movies_path)
        self.user_item = self._build_user_item_matrix()
        self.user_similarity = self._build_similarity_matrix()

    def _build_user_item_matrix(self) -> pd.DataFrame:
        return self.ratings.pivot_table(
            index="userId",
            columns="movieId",
            values="rating",
            aggfunc="mean",
        )

    def _build_similarity_matrix(self) -> pd.DataFrame:
        filled = self.user_item.fillna(0)
        similarity = cosine_similarity(filled)
        sim_df = pd.DataFrame(
            similarity,
            index=self.user_item.index,
            columns=self.user_item.index,
        )
        np.fill_diagonal(sim_df.values, 0.0)
        return sim_df

    def get_top_n_recommendations(self, user_id: int, n: int = 5) -> List[Recommendation]:
        if user_id not in self.user_item.index:
            return self._popular_recommendations(n=n)

        seen_movies = set(self.ratings.loc[self.ratings["userId"] == user_id, "movieId"].tolist())
        neighbors = self.user_similarity.loc[user_id].sort_values(ascending=False)

        recommendations: List[Recommendation] = []
        for movie_id in self.user_item.columns:
            if movie_id in seen_movies:
                continue

            movie_ratings = self.user_item[movie_id]
            mask = movie_ratings.notna()
            if not mask.any():
                continue

            relevant_neighbors = neighbors[mask]
            if relevant_neighbors.empty or relevant_neighbors.abs().sum() == 0:
                continue

            weighted_sum = float((movie_ratings[mask] * relevant_neighbors).sum())
            similarity_sum = float(relevant_neighbors.abs().sum())
            predicted = weighted_sum / similarity_sum if similarity_sum else 0.0

            movie_row = self.movies.loc[self.movies["movieId"] == movie_id].iloc[0]
            recommendations.append(
                Recommendation(
                    movie_id=int(movie_id),
                    title=str(movie_row["title"]),
                    predicted_score=round(predicted, 2),
                    genres=str(movie_row["genres"]),
                )
            )

        recommendations.sort(key=lambda rec: rec.predicted_score, reverse=True)
        if recommendations:
            return recommendations[:n]
        return self._popular_recommendations(n=n)

    def _popular_recommendations(self, n: int = 5) -> List[Recommendation]:
        avg = (
            self.ratings.groupby("movieId", as_index=False)["rating"]
            .mean()
            .sort_values("rating", ascending=False)
        )
        merged = avg.merge(self.movies, on="movieId", how="left").head(n)
        return [
            Recommendation(
                movie_id=int(row.movieId),
                title=str(row.title),
                predicted_score=round(float(row.rating), 2),
                genres=str(row.genres),
            )
            for row in merged.itertuples(index=False)
        ]
