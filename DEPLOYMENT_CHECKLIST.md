# Deployment Checklist

## Pre-Deployment Checklist

### ✅ Code Preparation
- [x] All dependencies listed in `requirements.txt`
- [x] Streamlit configuration file created (`.streamlit/config.toml`)
- [x] `.gitignore` file created to exclude unnecessary files
- [x] Code cleaned (removed unused imports)
- [x] Documentation updated (README.md)

### 📁 Required Files for Deployment

Ensure these files are present in your repository:

**Data Files** (must be committed):
- `streamlit/data/movies_clean.parquet`
- `streamlit/data/ratings_clean.parquet`
- `streamlit/data/movie_stats.parquet`
- `streamlit/data/weighted_popularity.parquet`
- `streamlit/data/movies_cb.parquet`
- `streamlit/data/summary_stats.json` (optional)
- `streamlit/data/top_movies_homepage.json` (optional)

**Analytics Files** (in `streamlit/data/analytics/`):
- `rater_distribution.json`
- `genre_performance.json` and/or `genre_performance.parquet`
- `hidden_gems.json` and/or `hidden_gems.parquet`
- `yearly_stats.json` and/or `yearly_stats.parquet`
- `tag_analytics.json` and/or `tag_analytics.parquet`
- `user_stats.parquet`

**Model Files** (in `streamlit/models/`):
- `tfidf_vectorizer.pkl`
- `content_nn_model.pkl`
- `random_forest_model.pkl` (optional)
- `scaler.pkl` (optional)
- `feature_info.json` (optional)

### 🚀 Deployment Steps

#### For Streamlit Cloud (Easiest)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select repository: `film-galore`
   - Main file path: `app.py`
   - Click "Deploy"

3. **Verify Deployment**
   - Check that all data files are accessible
   - Test all app pages
   - Monitor for any errors in logs

#### For Heroku

1. **Install Heroku CLI** (if not already installed)

2. **Login and Create App**
   ```bash
   heroku login
   heroku create your-app-name
   ```

3. **Deploy**
   ```bash
   git push heroku main
   ```

4. **Open App**
   ```bash
   heroku open
   ```

#### For Docker

1. **Build Image**
   ```bash
   docker build -t film-galore .
   ```

2. **Run Container**
   ```bash
   docker run -p 8501:8501 film-galore
   ```

3. **Access App**
   - Open browser to `http://localhost:8501`

### ⚠️ Important Notes

1. **File Size Limits**
   - Streamlit Cloud free tier: 1GB repository limit
   - Heroku: 500MB slug size limit
   - If files are too large, consider:
     - Using Git LFS for large files
     - Compressing data files further
     - Using external storage (S3, etc.)

2. **Memory Considerations**
   - The app loads all data into memory on startup
   - Monitor memory usage during deployment
   - Consider using smaller sample datasets if needed

3. **Performance**
   - First load may be slow (caching models/data)
   - Subsequent loads will be faster due to Streamlit caching
   - Consider using `@st.cache_resource` for models (already implemented)

4. **Security**
   - No sensitive data should be in the repository
   - Use `.streamlit/secrets.toml` for any API keys (not committed)
   - Review `.gitignore` to ensure sensitive files are excluded

### 🐛 Troubleshooting

**Issue: FileNotFoundError**
- Solution: Ensure all required data/model files are committed to the repository

**Issue: Memory Error**
- Solution: Reduce dataset size or upgrade deployment tier

**Issue: Slow Loading**
- Solution: This is normal on first load. Subsequent loads use caching.

**Issue: Port Already in Use (Local)**
- Solution: Change port: `streamlit run app.py --server.port=8502`

### 📊 Post-Deployment

- [ ] Test all app pages
- [ ] Verify recommendations are working
- [ ] Check analytics dashboard
- [ ] Monitor app performance
- [ ] Set up error monitoring (optional)

