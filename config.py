"""
Configuration for data storage.
Supports both local files and external storage (S3, etc.)
"""

import os
from pathlib import Path

# Determine if running on Vercel
IS_VERCEL = os.environ.get('VERCEL') == '1'

# Data storage configuration
USE_EXTERNAL_STORAGE = os.environ.get('USE_EXTERNAL_STORAGE', 'false').lower() == 'true'
EXTERNAL_STORAGE_URL = os.environ.get('EXTERNAL_STORAGE_URL', '')

# Local paths (for development)
DATA_DIR = Path("streamlit") / "data"
ANALYTICS_DIR = DATA_DIR / "analytics"
MODELS_DIR = Path("streamlit") / "models"

# For Vercel, you can use environment variables to point to external storage
# Example: S3 bucket, Google Cloud Storage, or another CDN
VERCEL_DATA_BASE_URL = os.environ.get('VERCEL_DATA_BASE_URL', '')

def get_data_path(filename):
    """Get path to data file, supporting external storage."""
    if USE_EXTERNAL_STORAGE and EXTERNAL_STORAGE_URL:
        return f"{EXTERNAL_STORAGE_URL}/{filename}"
    return DATA_DIR / filename

def get_model_path(filename):
    """Get path to model file, supporting external storage."""
    if USE_EXTERNAL_STORAGE and EXTERNAL_STORAGE_URL:
        return f"{EXTERNAL_STORAGE_URL}/models/{filename}"
    return MODELS_DIR / filename

