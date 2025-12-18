# Film Galore - Movie Recommendation System

A comprehensive Movie Analytics & Recommendation System built with Streamlit, featuring personalized recommendations, similar movie discovery, and business insights.

## Features

- **Personalized Recommendations**: Get movie recommendations based on multiple models
  - Top Popular (Weighted) recommendations
  - Content-Based recommendations (similar to your highly-rated movies)
- **Similar Movies**: Find movies similar to any selected movie using content-based filtering
- **Business Insights Dashboard**: Pre-computed analytics including:
  - Genre performance metrics
  - Hidden gems (high rating, low visibility)
  - Tag analytics
  - Yearly statistics
  - User statistics

## Project Structure

```
film-galore/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── streamlit/
│   ├── data/                      # Data files (parquet format)
│   │   ├── movies_clean.parquet
│   │   ├── ratings_clean.parquet
│   │   ├── movie_stats.parquet
│   │   ├── weighted_popularity.parquet
│   │   ├── movies_cb.parquet
│   │   └── analytics/             # Pre-computed analytics
│   └── models/                    # ML models
│       ├── tfidf_vectorizer.pkl
│       ├── content_nn_model.pkl
│       ├── random_forest_model.pkl
│       ├── scaler.pkl
│       └── feature_info.json
└── README.md
```

## Local Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd film-galore
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

   The app will open in your browser at `http://localhost:8501`

## Deployment

### Streamlit Cloud (Recommended)

1. **Push your code to GitHub**
   - Make sure all data files and models are committed
   - Ensure `.gitignore` excludes sensitive files but includes necessary data

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository and branch
   - Set the main file path to `app.py`
   - Click "Deploy"

3. **Important Notes for Streamlit Cloud**
   - Ensure all data files in `streamlit/data/` and `streamlit/models/` are committed
   - File size limit: Free tier has a 1GB repository limit
   - The app will automatically install dependencies from `requirements.txt`

### Other Deployment Options

#### Heroku

1. Create a `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. Create `setup.sh`:
   ```bash
   mkdir -p ~/.streamlit/
   echo "[server]
   headless = true
   port = $PORT
   enableCORS = false
   " > ~/.streamlit/config.toml
   ```

3. Deploy:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

#### Docker

1. Create `Dockerfile`:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8501
   HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
   ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. Build and run:
   ```bash
   docker build -t film-galore .
   docker run -p 8501:8501 film-galore
   ```

## Requirements

- Python 3.8+
- All dependencies listed in `requirements.txt`

## Data Requirements

The application expects the following files to be present:

**Data Files** (in `streamlit/data/`):
- `movies_clean.parquet`
- `ratings_clean.parquet`
- `movie_stats.parquet`
- `weighted_popularity.parquet`
- `movies_cb.parquet`

**Analytics Files** (in `streamlit/data/analytics/`):
- `rater_distribution.json`
- `genre_performance.json` / `genre_performance.parquet`
- `hidden_gems.json` / `hidden_gems.parquet`
- `yearly_stats.json` / `yearly_stats.parquet`
- `tag_analytics.json` / `tag_analytics.parquet`
- `user_stats.parquet`

**Model Files** (in `streamlit/models/`):
- `tfidf_vectorizer.pkl`
- `content_nn_model.pkl`
- `random_forest_model.pkl` (optional)
- `scaler.pkl` (optional)
- `feature_info.json` (optional)

## Troubleshooting

- **FileNotFoundError**: Ensure all required data and model files are in the correct directories
- **Memory issues**: The app uses caching to optimize performance. If you encounter memory issues, consider reducing the dataset size
- **Slow loading**: First load may be slow as models and data are cached. Subsequent loads will be faster

## License

See LICENSE file for details.
