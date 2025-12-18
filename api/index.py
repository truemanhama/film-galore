"""
Flask application for Movie Recommendation System
Optimized for Vercel serverless deployment
"""

from flask import Flask, render_template, jsonify, request
from functools import wraps
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from utils import (
    load_core_data, load_analytics, load_models,
    get_user_rated_movies, popularity_recommendations,
    content_based_for_user, content_based_similar_movie
)

# Determine paths based on environment
if os.environ.get('VERCEL'):
    # Vercel deployment
    template_dir = BASE_DIR / "templates"
    static_dir = BASE_DIR / "static"
else:
    # Local development
    template_dir = BASE_DIR / "templates"
    static_dir = BASE_DIR / "static"

app = Flask(__name__, 
            template_folder=str(template_dir),
            static_folder=str(static_dir))

# Cache data on first load
_data = None
_analytics = None
_models = None


def get_data():
    """Lazy load data."""
    global _data
    if _data is None:
        _data = load_core_data()
    return _data


def get_analytics():
    """Lazy load analytics."""
    global _analytics
    if _analytics is None:
        _analytics = load_analytics()
    return _analytics


def get_models():
    """Lazy load models."""
    global _models
    if _models is None:
        _models = load_models()
    return _models


def df_to_dict(df):
    """Convert DataFrame to list of dicts."""
    if df.empty:
        return []
    return df.to_dict('records')


@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@app.route('/recommendations')
def recommendations():
    """Recommendations page."""
    return render_template('recommendations.html')


@app.route('/similar')
def similar():
    """Similar movies page."""
    return render_template('similar.html')


@app.route('/insights')
def insights():
    """Business insights page."""
    return render_template('insights.html')


# API Routes
@app.route('/api/stats')
def api_stats():
    """Get dataset statistics."""
    try:
        data = get_data()
        movies = data["movies"]
        ratings = data["ratings"]
        
        if movies.empty or ratings.empty:
            return jsonify({"error": "Data not available"}), 500
        
        return jsonify({
            "total_movies": len(movies),
            "total_ratings": len(ratings),
            "unique_users": int(ratings['userId'].nunique()) if 'userId' in ratings.columns else 0,
            "avg_rating": float(ratings['rating'].mean()) if 'rating' in ratings.columns else 0.0
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/trending')
def api_trending():
    """Get trending movies."""
    try:
        data = get_data()
        top_trending = data["weighted_pop"].head(20)
        return jsonify(df_to_dict(top_trending))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/users')
def api_users():
    """Get list of user IDs."""
    try:
        data = get_data()
        ratings = data["ratings"]
        if ratings.empty or 'userId' not in ratings.columns:
            return jsonify([])
        users = sorted(ratings["userId"].unique().tolist())
        return jsonify(users)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/user/<int:user_id>/ratings')
def api_user_ratings(user_id):
    """Get user's rated movies."""
    try:
        data = get_data()
        user_rated = get_user_rated_movies(data, user_id)
        return jsonify(df_to_dict(user_rated))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/user/<int:user_id>/recommendations')
def api_user_recommendations(user_id):
    """Get recommendations for user."""
    try:
        data = get_data()
        model_type = request.args.get('type', 'popular')
        
        if model_type == 'popular':
            recs = popularity_recommendations(data, user_id)
        elif model_type == 'content':
            models = get_models()
            recs = content_based_for_user(data, models, user_id)
        else:
            return jsonify({"error": "Invalid model type"}), 400
        
        return jsonify(df_to_dict(recs))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/movies')
def api_movies():
    """Get list of movies."""
    try:
        data = get_data()
        movies = data["movies"]
        if movies.empty:
            return jsonify([])
        # Return limited list for dropdown
        movies_list = movies[["movieId", "title"]].head(1000).to_dict('records')
        return jsonify(movies_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/movie/<int:movie_id>/similar')
def api_similar_movies(movie_id):
    """Get similar movies."""
    try:
        models = get_models()
        similar_df = content_based_similar_movie(models, movie_id)
        return jsonify(df_to_dict(similar_df))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/movie/search')
def api_movie_search():
    """Search for movies."""
    try:
        query = request.args.get('q', '').lower()
        data = get_data()
        movies = data["movies"]
        
        if movies.empty or not query:
            return jsonify([])
        
        # Simple search by title
        matches = movies[movies["title"].str.lower().str.contains(query, na=False)]
        results = matches[["movieId", "title", "genres"]].head(20).to_dict('records')
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/analytics/genres')
def api_analytics_genres():
    """Get genre performance analytics."""
    try:
        analytics = get_analytics()
        genre_perf_df = analytics.get("genre_perf_df")
        if genre_perf_df is None or genre_perf_df.empty:
            return jsonify([])
        top_genres = genre_perf_df.sort_values("avg_rating", ascending=False).head(10)
        return jsonify(df_to_dict(top_genres))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/analytics/hidden-gems')
def api_analytics_hidden_gems():
    """Get hidden gems."""
    try:
        analytics = get_analytics()
        hidden_gems_df = analytics.get("hidden_gems_df")
        if hidden_gems_df is None or hidden_gems_df.empty:
            return jsonify([])
        return jsonify(df_to_dict(hidden_gems_df.head(20)))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/analytics/tags')
def api_analytics_tags():
    """Get tag analytics."""
    try:
        analytics = get_analytics()
        tag_df = analytics.get("tag_analytics_df")
        if tag_df is None or tag_df.empty:
            return jsonify([])
        return jsonify(df_to_dict(tag_df))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Export app for Vercel
# Vercel will automatically detect Flask app
if __name__ == '__main__':
    app.run(debug=True)

