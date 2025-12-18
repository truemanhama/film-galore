
Make the Streamlit app stable on Streamlit Cloud by reducing startup crashes, memory usage, and runtime risk.

---

### ✅ Fix 2 — Force Python 3.11

* Create a file at:
  `.streamlit/config.toml`
* Add exactly:

  ```toml
  [server]
  pythonVersion = "3.11"
  ```

---

### ✅ Fix 3 — Lazy-load ML models

* Do **not** load ML models at app startup.
* Remove or avoid calling `load_models()` globally.
* Only load models when the current page is:

  * `"Get Recommendations"`
  * `"Find Similar Movies"`
* Example pattern:

  ```python
  models = None
  if page in ["Get Recommendations", "Find Similar Movies"]:
      models = load_models()
  ```
* All recommendation functions must safely handle `models is None`.

---

### ✅ Fix 4 — Lazy-load TF-IDF feature matrix

* Do **not** compute the TF-IDF feature matrix inside `load_models()`.
* Move TF-IDF transformation into a separate cached function:

  ```python
  @st.cache_resource
  def get_feature_matrix(tfidf, movies_cb):
      return tfidf.transform(movies_cb["features"])
  ```
* Only call this function **inside recommendation logic**, not during app startup.

---

### ✅ Fix 5 — Reduce NearestNeighbors load

* Reduce `n_neighbors` in all `kneighbors()` calls:

  * Change from `50` → `20`
* This applies to:

  * User-based content recommendations
  * Similar-movie recommendations

---

### ✅ Fix 6 — Add startup debug checkpoints

* Add lightweight startup logging using `st.write()`:

  * At the top of `app.py`:

    ```python
    st.write("✅ App starting...")
    ```
  * After each major load:

    ```python
    st.write("✅ Core data loaded")
    st.write("✅ Analytics loaded")
    st.write("✅ Models loaded")
    ```
* These must not interrupt execution.

---

### 🎯 Acceptance Criteria

* App passes Streamlit Cloud health check
* Home page loads without loading ML models
* Models load only when required
* No TF-IDF matrix built at startup
* Reduced memory usage and faster startup

