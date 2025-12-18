# Vercel Deployment Guide

## Issue: Serverless Function Size Limit

Vercel has a **250 MB unzipped size limit** for serverless functions. The movie data and model files exceed this limit.

## Solutions

### Option 1: Use External Storage (Recommended)

Store large files (parquet, pkl) in external storage and load them at runtime.

#### Setup Steps:

1. **Upload files to external storage** (choose one):
   - **AWS S3** (recommended)
   - **Google Cloud Storage**
   - **Cloudflare R2**
   - **Any CDN with public URLs**

2. **Set environment variables in Vercel**:
   ```
   USE_EXTERNAL_STORAGE=true
   EXTERNAL_STORAGE_URL=https://your-bucket.s3.amazonaws.com
   ```

3. **Update `utils.py`** to download files from external storage when needed.

### Option 2: Reduce Data Size

1. **Use smaller sample datasets** for deployment
2. **Compress files** before committing
3. **Remove unnecessary files**

### Option 3: Use Vercel Blob Storage (New)

Vercel offers Blob Storage for large files:

```python
from vercel_blob import put, head

# Upload files to Vercel Blob
# Then access via URLs
```

### Option 4: Hybrid Approach

- Keep small JSON files in the repo
- Load large parquet/pkl files from external storage at runtime
- Cache in memory after first load

## Quick Fix: Exclude Large Files

I've created `.vercelignore` to exclude large files. However, you'll need to:

1. **Host data files elsewhere** (S3, etc.)
2. **Update `utils.py`** to fetch from external URLs
3. **Set environment variables** for storage URLs

## Recommended: AWS S3 Setup

1. **Create S3 bucket**
   ```bash
   aws s3 mb s3://film-galore-data
   ```

2. **Upload files**
   ```bash
   aws s3 sync streamlit/data s3://film-galore-data/data
   aws s3 sync streamlit/models s3://film-galore-data/models
   ```

3. **Make bucket public** (or use signed URLs)

4. **Set Vercel environment variables**:
   ```
   USE_EXTERNAL_STORAGE=true
   EXTERNAL_STORAGE_URL=https://film-galore-data.s3.amazonaws.com
   ```

5. **Update `utils.py`** to download files from S3 when needed

## Alternative: Use Sample Data

For a demo/prototype, you can:
- Use smaller sample datasets
- Pre-process data to reduce size
- Use JSON instead of Parquet for smaller files

## Current Status

The `.vercelignore` file will prevent large files from being deployed, but the app won't work without the data. You need to implement one of the solutions above.

