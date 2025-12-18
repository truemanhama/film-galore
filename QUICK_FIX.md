# Quick Fix for Vercel 250MB Limit

## Immediate Solution

The deployment fails because data files exceed 250MB. Here are your options:

### Option 1: Use External Storage (Best for Production)

1. **Upload files to AWS S3** (or similar):
   ```bash
   # Install AWS CLI if needed
   aws s3 mb s3://film-galore-data
   aws s3 sync streamlit/data s3://film-galore-data/data --acl public-read
   aws s3 sync streamlit/models s3://film-galore-data/models --acl public-read
   ```

2. **Set Vercel Environment Variables**:
   - Go to Vercel Dashboard → Your Project → Settings → Environment Variables
   - Add:
     ```
     USE_EXTERNAL_STORAGE=true
     EXTERNAL_STORAGE_URL=https://film-galore-data.s3.amazonaws.com
     ```

3. **Redeploy** - The app will download files from S3 at runtime

### Option 2: Use Sample/Mock Data (Quick Demo)

Create a minimal version that works without large files:

1. **Create sample data files** (small JSON files)
2. **Update `utils.py`** to handle missing files gracefully
3. **Show demo mode** message to users

### Option 3: Use Vercel Blob Storage

Vercel's new Blob Storage for large files:

```python
from vercel_blob import put, head, list, del_blob

# Upload files to Vercel Blob
# Access via blob URLs
```

### Option 4: Split into Multiple Functions

- Keep app code in one function
- Create separate API functions for data loading
- Use Vercel's Edge Functions for lighter operations

## Current Status

✅ `.vercelignore` created - excludes large files
✅ `utils.py` updated - supports external storage
⚠️ **You must set up external storage or the app won't have data**

## Next Steps

1. **Choose an option above**
2. **Set environment variables in Vercel**
3. **Redeploy**

The app code is ready - you just need to configure data storage!

