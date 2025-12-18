"""
Utility functions for data loading and recommendations.
Extracted from Streamlit app for use in Flask.
"""

from pathlib import Path
import json
import pandas as pd
import joblib
from functools import lru_cache

DATA_DIR = Path("streamlit") / "data"
ANALYTICS_DIR = DATA_DIR / "analytics"
MODELS_DIR = Path("streamlit") / "models"

# Global cache for data
_data_cache = None
_analytics_cache = None
_models_cache = None


@lru_cache(maxsize=1)
def load_core_data():
    """Load core exported datasets from parquet files (optimized for speed)."""
    try:
        movies = pd.read_parquet(DATA_DIR / "movies_clean.parquet")
        ratings = pd.read_parquet(DATA_DIR / "ratings_clean.parquet")
        movie_stats = pd.read_parquet(DATA_DIR / "movie_stats.parquet")
        weighted_pop = pd.read_parquet(DATA_DIR / "weighted_popularity.parquet")
        movies_cb = pd.read_parquet(DATA_DIR / "movies_cb.parquet")
        
        # Optimize data types
        if 'movieId' in movies.columns:
            movies['movieId'] = movies['movieId'].astype('int32')
        if 'userId' in ratings.columns:
            ratings['userId'] = ratings['userId'].astype('int32')
        if 'movieId' in ratings.columns:
            ratings['movieId'] = ratings['movieId'].astype('int32')
        if 'timestamp' in ratings.columns:
            if ratings['timestamp'].dtype == 'int64':
                pass
            elif pd.api.types.is_datetime64_any_dtype(ratings['timestamp']):
                ratings['timestamp'] = (ratings['timestamp'] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')
                ratings['timestamp'] = ratings['timestamp'].astype('int64')
        if 'movieId' in movie_stats.columns:
            movie_stats['movieId'] = movie_stats['movieId'].astype('int32')
        if 'movieId' in weighted_pop.columns:
            weighted_pop['movieId'] = weighted_pop['movieId'].astype('int32')
        if 'movieId' in movies_cb.columns:
            movies_cb['movieId'] = movies_cb['movieId'].astype('int32')
        
        # Create indexes
        if 'movieId' in movies.columns:
            movies.set_index('movieId', inplace=True, drop=False)
        if 'movieId' in movie_stats.columns:
            movie_stats.set_index('movieId', inplace=True, drop=False)
        if 'movieId' in weighted_pop.columns:
            weighted_pop.set_index('movieId', inplace=True, drop=False)
        if 'movieId' in movies_cb.columns:
            movies_cb.set_index('movieId', inplace=True, drop=False)
        
    except Exception as e:
        return {
            "movies": pd.DataFrame(),
            "ratings": pd.DataFrame(),
            "movie_stats": pd.DataFrame(),
            "weighted_pop": pd.DataFrame(),
            "movies_cb": pd.DataFrame(),
            "summary": None,
        }

    summary_stats = None
    try:
        if (DATA_DIR / "summary_stats.json").exists():
            with open(DATA_DIR / "summary_stats.json", "r") as f:
                summary_stats = json.load(f)
    except Exception:
        summary_stats = None

    return {
        "movies": movies,
        "ratings": ratings,
        "movie_stats": movie_stats,
        "weighted_pop": weighted_pop,
        "movies_cb": movies_cb,
        "summary": summary_stats,
    }


@lru_cache(maxsize=1)
def load_analytics():
    """Load pre-computed analytics."""
    analytics = {}

    def _read_json(name):
        path = ANALYTICS_DIR / name
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return None

    def _read_parquet(name):
        path = ANALYTICS_DIR / name
        if path.exists():
            df = pd.read_parquet(path)
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]) or pd.api.types.is_timedelta64_dtype(df[col]):
                    continue
                elif df[col].dtype == 'int64':
                    df[col] = df[col].astype('int32')
                elif df[col].dtype == 'float64':
                    df[col] = df[col].astype('float32')
            return df
        return None

    analytics["rater_distribution"] = _read_json("rater_distribution.json")
    analytics["genre_performance"] = _read_json("genre_performance.json")
    analytics["hidden_gems"] = _read_json("hidden_gems.json")
    analytics["yearly_stats"] = _read_json("yearly_stats.json")
    analytics["tag_analytics"] = _read_json("tag_analytics.json")

    analytics["user_stats_df"] = _read_parquet("user_stats.parquet")
    analytics["genre_perf_df"] = _read_parquet("genre_performance.parquet")
    analytics["hidden_gems_df"] = _read_parquet("hidden_gems.parquet")
    analytics["yearly_stats_df"] = _read_parquet("yearly_stats.parquet")
    analytics["tag_analytics_df"] = _read_parquet("tag_analytics.parquet")

    return analytics


@lru_cache(maxsize=1)
def load_models():
    """Load ML models without feature matrix."""
    models = {
        "tfidf": None,
        "content_nn": None,
        "feature_matrix": None,
        "movies_cb": None,
        "rf": None,
        "scaler": None,
        "feature_info": None,
    }

    tfidf_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    nn_path = MODELS_DIR / "content_nn_model.pkl"
    movies_cb_path = DATA_DIR / "movies_cb.parquet"
    
    if tfidf_path.exists() and nn_path.exists() and movies_cb_path.exists():
        tfidf = joblib.load(tfidf_path)
        nn = joblib.load(nn_path)
        movies_cb = pd.read_parquet(movies_cb_path)
        movies_cb["features"] = movies_cb["features"].fillna("")
        movies_cb['movieId'] = movies_cb['movieId'].astype('int32')
        if 'movieId' in movies_cb.columns:
            movies_cb.set_index('movieId', inplace=True, drop=False)

        models["tfidf"] = tfidf
        models["content_nn"] = nn
        models["movies_cb"] = movies_cb

    rf_path = MODELS_DIR / "random_forest_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    feature_info_path = MODELS_DIR / "feature_info.json"

    if rf_path.exists() and scaler_path.exists() and feature_info_path.exists():
        models["rf"] = joblib.load(rf_path)
        models["scaler"] = joblib.load(scaler_path)
        with open(feature_info_path, "r") as f:
            models["feature_info"] = json.load(f)

    return models


@lru_cache(maxsize=1)
def get_feature_matrix(tfidf, movies_cb):
    """Lazy-load TF-IDF feature matrix."""
    if tfidf is None or movies_cb is None:
        return None
    return tfidf.transform(movies_cb["features"])


def get_user_rated_movies(data, user_id, max_rows=200):
    """Get user's rated movies."""
    ratings = data["ratings"]
    movies = data["movies"]
    user_ratings = ratings[ratings["userId"] == user_id].copy()
    if user_ratings.empty:
        return pd.DataFrame()
    movies_cols = movies[["movieId", "title", "genres", "release_year"]].reset_index(drop=True) if movies.index.name == 'movieId' else movies[["movieId", "title", "genres", "release_year"]]
    user_ratings = user_ratings.merge(movies_cols, on="movieId", how="left")
    return user_ratings.nlargest(max_rows, "rating")


def popularity_recommendations(data, user_id, top_n=20, min_ratings=10):
    """Get popularity recommendations."""
    weighted_pop = data["weighted_pop"].copy()
    ratings = data["ratings"]
    user_ratings = ratings[ratings["userId"] == user_id]
    seen = set(user_ratings["movieId"].unique())
    recs = weighted_pop[~weighted_pop["movieId"].isin(seen)].copy()
    if "num_ratings" in recs.columns:
        recs = recs[recs["num_ratings"] >= min_ratings]
    sort_col = "weighted_score" if "weighted_score" in recs.columns else "avg_rating"
    return recs.nlargest(top_n, sort_col)


def content_based_for_user(data, _models, user_id, top_n=20):
    """Content-based recommendations for user."""
    if _models is None or _models["content_nn"] is None or _models["movies_cb"] is None:
        return pd.DataFrame()

    ratings = data["ratings"]
    movies_cb = _models["movies_cb"]
    nn = _models["content_nn"]
    tfidf = _models["tfidf"]
    
    feature_matrix = get_feature_matrix(tfidf, movies_cb)
    if feature_matrix is None:
        return pd.DataFrame()

    user_ratings = ratings[ratings["userId"] == user_id]
    high_rated = user_ratings[user_ratings["rating"] >= 4.0]
    movies_cb_ids = set(movies_cb.index if hasattr(movies_cb.index, '__iter__') else movies_cb["movieId"].unique())
    high_rated = high_rated[high_rated["movieId"].isin(movies_cb_ids)]

    if high_rated.empty:
        return pd.DataFrame()

    all_scores = {}
    movies_cb_reset = movies_cb.reset_index() if movies_cb.index.name == 'movieId' else movies_cb
    
    for _, row in high_rated.iterrows():
        movie_id = row["movieId"]
        rating = row["rating"]
        if movies_cb.index.name == 'movieId':
            if movie_id not in movies_cb.index:
                continue
            idx = movies_cb.index.get_loc(movie_id)
        else:
            idx_list = movies_cb_reset.index[movies_cb_reset["movieId"] == movie_id].tolist()
            if not idx_list:
                continue
            idx = idx_list[0]

        distances, indices = nn.kneighbors(
            feature_matrix[idx],
            n_neighbors=min(20, len(movies_cb)),
        )
        for i, dist in zip(indices[0][1:], distances[0][1:]):
            mid = int(movies_cb_reset.iloc[i]["movieId"])
            score = (1.0 - float(dist)) * float(rating)
            if mid not in all_scores:
                all_scores[mid] = 0.0
            all_scores[mid] += score

    if not all_scores:
        return pd.DataFrame()

    scores_df = pd.DataFrame({"movieId": list(all_scores.keys()), "score": list(all_scores.values())})
    movies_cols = data["movies"][["movieId", "title", "genres", "release_year"]].reset_index(drop=True) if data["movies"].index.name == 'movieId' else data["movies"][["movieId", "title", "genres", "release_year"]]
    scores_df = scores_df.merge(movies_cols, on="movieId", how="left").sort_values("score", ascending=False)
    seen = set(user_ratings["movieId"].unique())
    scores_df = scores_df[~scores_df["movieId"].isin(seen)]
    return scores_df.head(top_n)


def content_based_similar_movie(_models, movie_id, top_n=10):
    """Find similar movies."""
    if _models is None or _models["content_nn"] is None or _models["movies_cb"] is None:
        return pd.DataFrame()

    movies_cb = _models["movies_cb"]
    nn = _models["content_nn"]
    tfidf = _models["tfidf"]
    
    feature_matrix = get_feature_matrix(tfidf, movies_cb)
    if feature_matrix is None:
        return pd.DataFrame()

    if movies_cb.index.name == 'movieId' and movie_id in movies_cb.index:
        idx = movies_cb.index.get_loc(movie_id)
    elif 'movieId' in movies_cb.columns:
        idx_list = movies_cb.index[movies_cb["movieId"] == movie_id].tolist()
        if not idx_list:
            return pd.DataFrame()
        idx = idx_list[0]
    else:
        return pd.DataFrame()

    distances, indices = nn.kneighbors(
        feature_matrix[idx],
        n_neighbors=min(top_n + 1, len(movies_cb)),
    )

    rows = []
    for i, dist in zip(indices[0][1:], distances[0][1:]):
        m = movies_cb.iloc[i]
        rows.append({
            "movieId": int(m["movieId"] if "movieId" in m else movies_cb.index[i]),
            "title": m["title"],
            "genres": m["genres"],
            "similarity": float(1.0 - float(dist)),
        })

    return pd.DataFrame(rows)

