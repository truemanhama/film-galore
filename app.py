"""
Streamlit Application
Movie Analytics & Recommendation System

This version is wired to the exported artifacts created by `export_artifacts.py`:
- Cleaned data in `streamlit/data/`
- Analytics in `streamlit/data/analytics/`
- Models in `streamlit/models/`
"""

from pathlib import Path
import json

import pandas as pd
import joblib
import streamlit as st


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Team 6 Movie Recommender",
    page_icon="🎬",
    layout="wide"
)


DATA_DIR = Path("streamlit") / "data"
ANALYTICS_DIR = DATA_DIR / "analytics"
MODELS_DIR = Path("streamlit") / "models"


# -----------------------------------------------------------------------------
# Cached loaders
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_core_data():
    """Load core exported datasets from parquet files (optimized for speed)."""
    try:
        # Use parquet files for faster loading and smaller file sizes
        movies = pd.read_parquet(DATA_DIR / "movies_clean.parquet")
        ratings = pd.read_parquet(DATA_DIR / "ratings_clean.parquet")
        movie_stats = pd.read_parquet(DATA_DIR / "movie_stats.parquet")
        weighted_pop = pd.read_parquet(DATA_DIR / "weighted_popularity.parquet")
        movies_cb = pd.read_parquet(DATA_DIR / "movies_cb.parquet")
        
        # Optimize data types for faster operations and smaller memory footprint
        movies['movieId'] = movies['movieId'].astype('int32')
        ratings['userId'] = ratings['userId'].astype('int32')
        ratings['movieId'] = ratings['movieId'].astype('int32')
        # Keep timestamp as int64 (can be large Unix timestamps)
        if ratings['timestamp'].dtype == 'int64':
            pass  # Already int64, keep it
        elif pd.api.types.is_datetime64_any_dtype(ratings['timestamp']):
            ratings['timestamp'] = (ratings['timestamp'] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')
            ratings['timestamp'] = ratings['timestamp'].astype('int64')
        movie_stats['movieId'] = movie_stats['movieId'].astype('int32')
        weighted_pop['movieId'] = weighted_pop['movieId'].astype('int32')
        movies_cb['movieId'] = movies_cb['movieId'].astype('int32')
        
        # Create indexes for faster lookups (keep movieId as column for merges)
        if 'movieId' in movies.columns:
            movies.set_index('movieId', inplace=True, drop=False)
        if 'movieId' in movie_stats.columns:
            movie_stats.set_index('movieId', inplace=True, drop=False)
        if 'movieId' in weighted_pop.columns:
            weighted_pop.set_index('movieId', inplace=True, drop=False)
        if 'movieId' in movies_cb.columns:
            movies_cb.set_index('movieId', inplace=True, drop=False)
        
    except FileNotFoundError as e:
        st.error(
            "One or more exported data files are missing. "
            "Please run the export cell in the notebook first "
            "(see `NOTEBOOK_EXPORT_INSTRUCTIONS.md`)."
        )
        st.stop()

    # Optional analytics
    summary_stats = None
    if (DATA_DIR / "summary_stats.json").exists():
        with open(DATA_DIR / "summary_stats.json", "r") as f:
            summary_stats = json.load(f)

    return {
        "movies": movies,
        "ratings": ratings,
        "movie_stats": movie_stats,
        "weighted_pop": weighted_pop,
        "movies_cb": movies_cb,
        "summary": summary_stats,
    }


@st.cache_data(show_spinner=False)
def load_analytics():
    """Load pre-computed analytics for the dashboard from parquet files."""
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
            # Optimize data types (skip datetime and timedelta columns)
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]) or pd.api.types.is_timedelta64_dtype(df[col]):
                    continue  # Skip datetime/timedelta columns
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


@st.cache_resource(show_spinner=False)
def load_models():
    """Load exported ML models (content-based + RF)."""
    models = {
        "tfidf": None,
        "content_nn": None,
        "feature_matrix": None,
        "movies_cb": None,
        "rf": None,
        "scaler": None,
        "feature_info": None,
    }

    # Content-based
    tfidf_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    nn_path = MODELS_DIR / "content_nn_model.pkl"
    movies_cb_path = DATA_DIR / "movies_cb.csv"

    movies_cb_path = DATA_DIR / "movies_cb.parquet"
    if tfidf_path.exists() and nn_path.exists() and movies_cb_path.exists():
        tfidf = joblib.load(tfidf_path)
        nn = joblib.load(nn_path)
        movies_cb = pd.read_parquet(movies_cb_path)
        movies_cb["features"] = movies_cb["features"].fillna("")
        movies_cb['movieId'] = movies_cb['movieId'].astype('int32')
        # Keep movieId as column for lookups
        if 'movieId' in movies_cb.columns:
            movies_cb.set_index('movieId', inplace=True, drop=False)
        feature_matrix = tfidf.transform(movies_cb["features"])

        models["tfidf"] = tfidf
        models["content_nn"] = nn
        models["feature_matrix"] = feature_matrix
        models["movies_cb"] = movies_cb

    # Random Forest rating model
    rf_path = MODELS_DIR / "random_forest_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    feature_info_path = MODELS_DIR / "feature_info.json"

    if rf_path.exists() and scaler_path.exists() and feature_info_path.exists():
        rf = joblib.load(rf_path)
        scaler = joblib.load(scaler_path)
        with open(feature_info_path, "r") as f:
            feature_info = json.load(f)
        models["rf"] = rf
        models["scaler"] = scaler
        models["feature_info"] = feature_info

    return models


# -----------------------------------------------------------------------------
# Recommendation helpers
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_user_rated_movies(data, user_id, max_rows=200):
    """Get user's rated movies with optimized lookups."""
    ratings = data["ratings"]
    movies = data["movies"]
    # Use vectorized filtering and merge
    user_ratings = ratings[ratings["userId"] == user_id].copy()
    if user_ratings.empty:
        return pd.DataFrame()
    # Use movieId column directly (kept as column even with index)
    movies_cols = movies[["movieId", "title", "genres", "release_year"]].reset_index(drop=True) if movies.index.name == 'movieId' else movies[["movieId", "title", "genres", "release_year"]]
    user_ratings = user_ratings.merge(movies_cols, on="movieId", how="left")
    return user_ratings.nlargest(max_rows, "rating")


@st.cache_data(show_spinner=False)
def popularity_recommendations(data, user_id, top_n=20, min_ratings=10):
    """Get popularity recommendations with optimized filtering."""
    weighted_pop = data["weighted_pop"].copy()
    ratings = data["ratings"]

    # Use vectorized operations for better performance
    user_ratings = ratings[ratings["userId"] == user_id]
    seen = set(user_ratings["movieId"].unique())
    
    # Use movieId column directly
    recs = weighted_pop[~weighted_pop["movieId"].isin(seen)].copy()

    if "num_ratings" in recs.columns:
        recs = recs[recs["num_ratings"] >= min_ratings]

    sort_col = "weighted_score" if "weighted_score" in recs.columns else "avg_rating"
    return recs.nlargest(top_n, sort_col)


@st.cache_data(show_spinner=False)
def content_based_for_user(data, _models, user_id, top_n=20):
    """Aggregate content-based recommendations from user's highly-rated movies (optimized)."""
    if _models["content_nn"] is None or _models["feature_matrix"] is None:
        return pd.DataFrame()

    ratings = data["ratings"]
    movies_cb = _models["movies_cb"]
    nn = _models["content_nn"]
    feature_matrix = _models["feature_matrix"]

    user_ratings = ratings[ratings["userId"] == user_id]
    # Consider movies rated >= 4.0 and that exist in the CB subset
    high_rated = user_ratings[user_ratings["rating"] >= 4.0]
    # Use index for faster lookup
    movies_cb_ids = set(movies_cb.index if hasattr(movies_cb.index, '__iter__') else movies_cb["movieId"].unique())
    high_rated = high_rated[high_rated["movieId"].isin(movies_cb_ids)]

    if high_rated.empty:
        return pd.DataFrame()

    all_scores = {}
    # Reset index if needed for positional access
    movies_cb_reset = movies_cb.reset_index() if movies_cb.index.name == 'movieId' else movies_cb
    
    for _, row in high_rated.iterrows():
        movie_id = row["movieId"]
        rating = row["rating"]
        # Use index-based lookup if available
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
            n_neighbors=min(50, len(movies_cb)),
        )
        for i, dist in zip(indices[0][1:], distances[0][1:]):  # skip self
            mid = int(movies_cb_reset.iloc[i]["movieId"])
            score = (1.0 - float(dist)) * float(rating)
            if mid not in all_scores:
                all_scores[mid] = 0.0
            all_scores[mid] += score

    if not all_scores:
        return pd.DataFrame()

    scores_df = pd.DataFrame({"movieId": list(all_scores.keys()), "score": list(all_scores.values())})
    # Use movieId column directly for merge
    movies_cols = data["movies"][["movieId", "title", "genres", "release_year"]].reset_index(drop=True) if data["movies"].index.name == 'movieId' else data["movies"][["movieId", "title", "genres", "release_year"]]
    scores_df = scores_df.merge(movies_cols, on="movieId", how="left").sort_values("score", ascending=False)

    # Remove already seen movies
    seen = set(user_ratings["movieId"].unique())
    scores_df = scores_df[~scores_df["movieId"].isin(seen)]

    return scores_df.head(top_n)


@st.cache_data(show_spinner=False)
def content_based_similar_movie(_models, movie_id, top_n=10):
    """Find similar movies using content-based model (optimized)."""
    if _models["content_nn"] is None or _models["feature_matrix"] is None:
        return pd.DataFrame()

    movies_cb = _models["movies_cb"]
    nn = _models["content_nn"]
    feature_matrix = _models["feature_matrix"]

    # Use index-based lookup if available, otherwise use column
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
    for i, dist in zip(indices[0][1:], distances[0][1:]):  # skip self
        m = movies_cb.iloc[i]
        rows.append(
            {
                "movieId": int(m["movieId"] if "movieId" in m else movies_cb.index[i]),
                "title": m["title"],
                "genres": m["genres"],
                "similarity": float(1.0 - float(dist)),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------
st.title("🎬 Movie Recommendation System")
st.markdown("### Powered by Exported ML Models & Analytics")

st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Home", "Get Recommendations", "Find Similar Movies", "Business Insights Dashboard"],
)


data = load_core_data()
analytics = load_analytics()
models = load_models()


# -----------------------------------------------------------------------------
# Page: Home
# -----------------------------------------------------------------------------
if page == "Home":
    st.header("Welcome to the Movie Recommendation System")
    st.markdown(
        """
        This application is backed by the exported MovieLens pipeline and provides:

        - **Personalized Recommendations** based on multiple models
        - **Similar Movies** using a content-based recommender
        - **Business Insights** from pre-computed analytics
        """
    )

    # Dataset statistics
    movies = data["movies"]
    ratings = data["ratings"]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Movies", f"{len(movies):,}")
    with col2:
        st.metric("Total Ratings", f"{len(ratings):,}")
    with col3:
        st.metric("Unique Users", f"{ratings['userId'].nunique():,}")
    with col4:
        st.metric("Avg Rating", f"{ratings['rating'].mean():.2f}")

    st.subheader("Trending Now (Weighted Popularity)")
    top_trending = data["weighted_pop"].head(20)
    if top_trending.empty:
        st.info("Weighted popularity file is empty or missing.")
    else:
        for i, row in top_trending.iterrows():
            with st.container():
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.write(f"**{row['title']}**")
                    st.caption(f"Genres: {row['genres']}")
                with c2:
                    rating = row.get("avg_rating", None)
                    if pd.notna(rating):
                        st.metric("Score", f"{rating:.2f}")
                st.divider()


# -----------------------------------------------------------------------------
# Page: Get Recommendations (per-user)
# -----------------------------------------------------------------------------
elif page == "Get Recommendations":
    st.header("Get Personalized Recommendations")

    ratings = data["ratings"]
    movies = data["movies"]

    # User selection (limited for performance, but sorted)
    # Cache unique users list to avoid recomputation
    if 'unique_users' not in st.session_state:
        st.session_state.unique_users = sorted(ratings["userId"].unique())
    selected_user = st.selectbox("Select a User ID", st.session_state.unique_users)

    # Show movies this user has rated
    st.subheader("Movies You've Rated")
    user_rated = get_user_rated_movies(data, selected_user)
    if user_rated.empty:
        st.info("No ratings found for this user in the exported sample.")
    else:
        st.dataframe(
            user_rated[["title", "genres", "release_year", "rating", "timestamp"]],
            use_container_width=True,
        )

    st.subheader("Choose Recommendation Models")
    model_choices = st.multiselect(
        "Select one or more models",
        [
            "Top Popular (Weighted)",
            "Content-Based (Similar to Your Likes)",
        ],
        default=["Top Popular (Weighted)", "Content-Based (Similar to Your Likes)"],
    )

    top_n = st.slider("Number of recommendations per model", 5, 30, 10)

    if st.button("Generate Recommendations"):
        with st.spinner("Computing recommendations..."):
            if "Top Popular (Weighted)" in model_choices:
                st.markdown("### 📈 Top Popular (Weighted)")
                pop_recs = popularity_recommendations(data, selected_user, top_n)
                if pop_recs.empty:
                    st.info("No popularity-based recommendations available.")
                else:
                    st.dataframe(
                        pop_recs[["title", "genres", "avg_rating", "num_ratings"]],
                        use_container_width=True,
                    )

            if "Content-Based (Similar to Your Likes)" in model_choices:
                st.markdown("### 🎯 Content-Based (Similar To Your Highly-Rated Movies)")
                cb_recs = content_based_for_user(data, models, selected_user, top_n)
                if cb_recs.empty:
                    st.info(
                        "Content-based model artifacts are missing, or this user has no "
                        "highly-rated movies in the content-based subset."
                    )
                else:
                    st.dataframe(
                        cb_recs[["title", "genres", "score"]],
                        use_container_width=True,
                    )


# -----------------------------------------------------------------------------
# Page: Find Similar Movies
# -----------------------------------------------------------------------------
elif page == "Find Similar Movies":
    st.header("Find Similar Movies")

    movies = data["movies"]
    
    # Cache movie titles list for faster access
    if 'movie_titles' not in st.session_state:
        st.session_state.movie_titles = movies["title"].tolist()
    selected_title = st.selectbox("Search for a movie", st.session_state.movie_titles)

    if st.button("Find Similar Movies"):
        with st.spinner("Finding similar movies..."):
            row = movies[movies["title"] == selected_title].iloc[0]
            movie_id = int(row["movieId"])

            # Prefer content-based recommender if available; fallback to simple genre Jaccard
            similar_df = content_based_similar_movie(models, movie_id, top_n=10)

            if similar_df.empty:
                # Fallback: simple genre-based similarity using exported movies
                selected_genres = set(str(row["genres"]).split("|"))
                scores = []
                for _, m in movies.iterrows():
                    if int(m["movieId"]) == movie_id:
                        continue
                    g = set(str(m["genres"]).split("|"))
                    union = selected_genres.union(g)
                    if not union:
                        continue
                    sim = len(selected_genres.intersection(g)) / len(union)
                    if sim > 0:
                        scores.append(
                            {
                                "movieId": int(m["movieId"]),
                                "title": m["title"],
                                "genres": m["genres"],
                                "similarity": sim,
                            }
                        )
                similar_df = (
                    pd.DataFrame(scores).sort_values("similarity", ascending=False).head(10)
                )

            if similar_df.empty:
                st.info("No similar movies could be found.")
            else:
                for i, r in similar_df.iterrows():
                    with st.container():
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.write(f"**{r['title']}**")
                            st.caption(f"Genres: {r['genres']}")
                        with c2:
                            st.metric("Similarity", f"{float(r['similarity'])*100:.1f}%")
                        st.divider()


# -----------------------------------------------------------------------------
# Page: Business Insights Dashboard
# -----------------------------------------------------------------------------
elif page == "Business Insights Dashboard":
    st.header("Business Insights Dashboard")

    movies = data["movies"]
    ratings = data["ratings"]

    # Key metrics
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_rating = ratings["rating"].mean()
        st.metric("Average Rating", f"{avg_rating:.2f}")

    with col2:
        total_users = ratings["userId"].nunique()
        st.metric("Total Users", f"{total_users:,}")

    with col3:
        total_movies = ratings["movieId"].nunique()
        st.metric("Rated Movies", f"{total_movies:,}")

    with col4:
        ratings_per_user = len(ratings) / max(total_users, 1)
        st.metric("Ratings per User", f"{ratings_per_user:.1f}")

    # Genre performance (from analytics, if available)
    st.subheader("Top Genres by Average Rating")
    genre_perf_df = analytics.get("genre_perf_df")
    if genre_perf_df is not None and not genre_perf_df.empty:
        top_genres = genre_perf_df.sort_values("avg_rating", ascending=False).head(10)
        st.bar_chart(top_genres.set_index("genre")["avg_rating"])
    else:
        st.info("Genre performance analytics not found; falling back to quick computation.")
        # Quick fallback using exported data
        genre_ratings = []
        for _, row in movies.iterrows():
            for g in str(row["genres"]).split("|"):
                if g and g != "(no genres listed)":
                    genre_ratings.append({"genre": g, "movieId": row["movieId"]})
        if genre_ratings:
            genre_df = pd.DataFrame(genre_ratings)
            gp = (
                genre_df.merge(ratings, on="movieId")
                .groupby("genre")["rating"]
                .agg(["mean", "count"])
                .reset_index()
            )
            gp = gp[gp["count"] >= 100].sort_values("mean", ascending=False).head(10)
            gp.columns = ["genre", "avg_rating", "num_ratings"]
            st.bar_chart(gp.set_index("genre")["avg_rating"])

    # Hidden gems
    st.subheader("Hidden Gems (High Rating, Low Visibility)")
    hidden_gems_df = analytics.get("hidden_gems_df")
    if hidden_gems_df is not None and not hidden_gems_df.empty:
        st.dataframe(
            hidden_gems_df[["title", "genres", "avg_rating", "num_ratings"]].head(20),
            use_container_width=True,
        )
    else:
        st.info("Hidden gems analytics not found.")

    # Tag analytics
    st.subheader("Top Tags by Usage")
    tag_df = analytics.get("tag_analytics_df")
    if tag_df is not None and not tag_df.empty:
        st.bar_chart(tag_df.set_index("tag")["num_uses"])
    else:
        st.info("Tag analytics not found.")


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
This app is backed by:
- Exported MovieLens data (cleaned & optimized)
- Content-based recommendation model (TF-IDF + Nearest Neighbors)
- Pre-computed analytics (genres, tags, hidden gems, temporal trends)
"""
)
