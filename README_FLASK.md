# Film Galore - Flask Application

A modern, interactive Flask web application for Movie Recommendations, optimized for Vercel deployment.

## Features

- **Modern UI**: Clean, responsive design with smooth interactions
- **Interactive Recommendations**: Get personalized movie recommendations
- **Similar Movies**: Find movies similar to your favorites
- **Business Insights**: Analytics dashboard with charts and statistics
- **API-First**: RESTful API endpoints for all functionality
- **Vercel Optimized**: Serverless deployment ready

## Project Structure

```
film-galore/
├── api/
│   └── index.py              # Flask app (Vercel entry point)
├── templates/                # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── recommendations.html
│   ├── similar.html
│   └── insights.html
├── static/                   # Static files
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
├── utils.py                  # Utility functions
├── vercel.json               # Vercel configuration
├── requirements.txt          # Python dependencies
└── streamlit/                # Data and models (unchanged)
```

## Local Development

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Flask app**
   ```bash
   python api/index.py
   ```

3. **Access the app**
   - Open http://localhost:5000 in your browser

## Vercel Deployment

### Prerequisites
- Vercel account
- Git repository with your code

### Deployment Steps

1. **Install Vercel CLI** (optional)
   ```bash
   npm i -g vercel
   ```

2. **Deploy to Vercel**
   ```bash
   vercel
   ```
   
   Or connect your GitHub repository to Vercel dashboard:
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your repository
   - Vercel will auto-detect Flask and deploy

3. **Environment Variables** (if needed)
   - Add any environment variables in Vercel dashboard
   - Project Settings → Environment Variables

### Vercel Configuration

The `vercel.json` file configures:
- Python 3.11 runtime
- Serverless function routing
- API route handling

## API Endpoints

### Statistics
- `GET /api/stats` - Get dataset statistics

### Movies
- `GET /api/movies` - Get list of movies
- `GET /api/movie/search?q=<query>` - Search movies
- `GET /api/movie/<id>/similar` - Get similar movies

### Users
- `GET /api/users` - Get list of user IDs
- `GET /api/user/<id>/ratings` - Get user's rated movies
- `GET /api/user/<id>/recommendations?type=<popular|content>` - Get recommendations

### Analytics
- `GET /api/analytics/genres` - Genre performance
- `GET /api/analytics/hidden-gems` - Hidden gems
- `GET /api/analytics/tags` - Tag analytics

## Pages

- `/` - Home page with statistics and trending movies
- `/recommendations` - Personalized recommendations
- `/similar` - Find similar movies
- `/insights` - Business insights dashboard

## Performance Optimizations

1. **Lazy Loading**: Models and feature matrices load only when needed
2. **Caching**: Using `@lru_cache` for data loading
3. **Serverless**: Optimized for Vercel's serverless functions
4. **Efficient Data Types**: Optimized pandas data types for memory

## Troubleshooting

### Vercel Deployment Issues

1. **Build Failures**
   - Check Python version (should be 3.11)
   - Verify all dependencies in `requirements.txt`
   - Check file paths are correct

2. **Import Errors**
   - Ensure `utils.py` is in root directory
   - Check `sys.path` modifications in `api/index.py`

3. **Static Files Not Loading**
   - Verify `static/` folder structure
   - Check Flask static folder configuration

4. **Data Files Not Found**
   - Ensure `streamlit/data/` and `streamlit/models/` are committed
   - Check file paths in `utils.py`

## Development Tips

- Use browser DevTools to debug API calls
- Check Vercel function logs for server-side errors
- Test API endpoints directly: `/api/stats`, etc.
- Use Flask's debug mode locally for better error messages

## Migration from Streamlit

The Flask app maintains all functionality from the Streamlit version:
- ✅ All recommendation algorithms
- ✅ All analytics features
- ✅ All data loading optimizations
- ✅ Lazy loading of models
- ✅ Error handling

Plus improvements:
- 🚀 Faster page loads (no full page reloads)
- 🎨 Modern, responsive UI
- 📱 Mobile-friendly design
- 🔄 Interactive without page refreshes
- 🌐 API-first architecture

