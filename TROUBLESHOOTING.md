# Troubleshooting Guide

## Health Check Error: "connection reset by peer"

### Error Message
```
The service has encountered an error while checking the health of the Streamlit app: 
Get "http://localhost:8501/healthz": read tcp 127.0.0.1:32884->127.0.0.1:8501: 
read: connection reset by peer
```

### Root Causes & Solutions

#### 1. App Crashing During Startup
**Symptoms:** App loads momentarily then crashes, health check fails

**Solutions Applied:**
- ✅ Added comprehensive error handling in data loading functions
- ✅ Added defensive checks for empty dataframes throughout the app
- ✅ Changed `st.stop()` to return empty dataframes instead of stopping
- ✅ Added try-except blocks around all data loading operations
- ✅ Added validation checks on all pages before accessing data

**What to Check:**
- Ensure all required data files are present in `streamlit/data/` directory
- Verify file permissions are correct
- Check deployment logs for specific error messages

#### 2. Memory Issues
**Symptoms:** App starts but crashes when loading large datasets

**Solutions:**
- The app uses `@st.cache_data` and `@st.cache_resource` to optimize memory
- Data types are optimized (int32 instead of int64 where possible)
- Consider reducing dataset size if memory is limited

**What to Check:**
- Monitor memory usage during deployment
- Check if your deployment platform has memory limits
- Consider using smaller sample datasets for initial deployment

#### 3. Health Check Endpoint Mismatch
**Symptoms:** Health check fails but app might actually be running

**Solutions Applied:**
- ✅ Updated Dockerfile health check to use correct endpoint: `/_stcore/health`
- ✅ Added proper health check timing (40s start period, 30s interval)
- ✅ Installed curl in Dockerfile for health checks

**Note:** Some deployment platforms use their own health check endpoints. If you see `/healthz` in errors, it's likely the platform's own check, not ours.

#### 4. File Path Issues
**Symptoms:** FileNotFoundError during startup

**Solutions Applied:**
- ✅ All paths use `Path()` objects for cross-platform compatibility
- ✅ Added error handling that returns empty dataframes instead of crashing
- ✅ Added clear error messages indicating which files are missing

**What to Check:**
- Verify file structure matches expected layout
- Ensure files are committed to repository (not in .gitignore)
- Check case sensitivity of file names

### Quick Fixes

1. **Check Deployment Logs**
   ```bash
   # For Streamlit Cloud: Check logs in the dashboard
   # For Docker: docker logs <container-id>
   # For Heroku: heroku logs --tail
   ```

2. **Verify File Structure**
   ```
   streamlit/
   ├── data/
   │   ├── movies_clean.parquet
   │   ├── ratings_clean.parquet
   │   ├── movie_stats.parquet
   │   ├── weighted_popularity.parquet
   │   ├── movies_cb.parquet
   │   └── analytics/
   └── models/
       ├── tfidf_vectorizer.pkl
       ├── content_nn_model.pkl
       └── ...
   ```

3. **Test Locally First**
   ```bash
   streamlit run app.py
   ```
   If it works locally but fails in deployment, it's likely a file or memory issue.

4. **Check File Sizes**
   - Large files (>100MB) might cause issues
   - Consider compressing or using Git LFS

### Common Deployment Platform Issues

#### Streamlit Cloud
- **File Size Limit:** 1GB total repository size
- **Memory:** Limited on free tier
- **Solution:** Ensure all files are committed, check repository size

#### Heroku
- **Slug Size Limit:** 500MB
- **Memory:** 512MB on free tier (may need upgrade)
- **Solution:** Use `Procfile` and `setup.sh` provided

#### Docker
- **Memory:** Depends on host system
- **Solution:** Adjust health check timing if startup is slow

### Still Having Issues?

1. **Enable Debug Mode**
   Add to `.streamlit/config.toml`:
   ```toml
   [logger]
   level = "debug"
   ```

2. **Check Specific Error**
   Look for the actual Python error in logs, not just the health check error

3. **Reduce Data Size**
   Temporarily use smaller datasets to test if it's a memory issue

4. **Contact Support**
   Share:
   - Deployment platform
   - Full error logs
   - File structure
   - Memory/storage limits

