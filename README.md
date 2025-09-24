<div align="center">

# DCIT316 Semester Project – Ghana News Intelligence Platform
**Fake News Detection • News Aggregation • Personalized Recommendations**  
_Author: David Asamoa Berko (11245046)_

</div>

## 1. Overview
This project is a full-stack news intelligence platform focused on Ghanaian news. It:

* Aggregates news from multiple external sources (NewsAPI, APITube, optional SerpAPI)
* Detects potential fake news using a (logistic regression → ONNX) model with a heuristic fallback
* Provides personalized recommendations using a simplified recommendation engine (ONNX model or heuristic)
* Exposes a RESTful Go (Gin) backend with authentication (JWT), scheduling, and ML inference integration
* Offers a React frontend for browsing, searching, detection, and recommendations
* Includes evaluation tooling for ML models and datasets (LIAR + MIND)

If ML models (`lr_pipeline.onnx`, `recommender_scoring.onnx`) are missing, the system gracefully falls back to heuristic logic so development isn’t blocked.

---
## 2. High-Level Architecture
```
			   +-----------------------------+
			   |         React Frontend      |
			   |  (Auth, Browse, Detect,     |
			   |   Recommend, Fetch)         |
			   +--------------+--------------+
							  | (HTTP / JSON via proxy)
							  v
+-----------------------------+-----------------------------+
|                        Go Backend (Gin)                  |
|  Auth (JWT) | News CRUD | Fetchers | ML Services | Jobs  |
|             |           | (NewsAPI / APITube / SerpAPI)  |
|             |           |                                 |
|  SQLite (persistent store)    ML Models (.onnx / fallback) |
+------------------+------------------+---------------------+
				   |                                  |
				   | (Scheduled fetch & cleanup)      |
				   v                                  v
		  External APIs / Datasets        Evaluation Script & Data (LIAR / MIND)
```

---
## 3. Tech Stack
| Layer | Technology |
|-------|------------|
| Frontend | React 19, React Router, Axios, React Bootstrap |
| Backend | Go (Gin), SQLite, JWT, CORS, Scheduler |
| ML Inference | ONNX Runtime (Python during export/eval), Heuristic fallbacks in Go |
| Data / Datasets | LIAR (fake news), MIND (recommendation) |
| Tooling | Python 3.x (model evaluation), TQDM, scikit-learn, ONNX tooling |

---
## 4. Repository Structure (Simplified)
```
backend/                Go API + services + middleware + schedulers
frontend/               React application
models/                 Exported ONNX / Torch models
Data/                   Datasets (LIAR, MIND subsets)
scripts/                Training / helper scripts (e.g., logistic_regression.py)
evaluate_models.py      Unified evaluation for fake news & recommendation models
requirements.txt        Python dependencies for ML/evaluation
```

---
## 5. Environment Variables (.env)
Create a `.env` at the project root (or inside `backend/` – the backend loads it with `godotenv`). Example:
```
# Server
PORT=8080
DB_PATH=./backend.db
MODELS_DIR=./models
JWT_SECRET=change_this_in_production

# ML / Feature Toggles
FAKE_NEWS_THRESHOLD=0.6              # Probability >= threshold => fake
INCLUDE_SERPAPI_IN_FEED=false        # If true, SerpAPI results appended when space
PREFER_NEWSAPI=                      # If set (any value), prioritizes NewsAPI over APITube

# External API Keys (obtain below). Leave blank for demo fallback.
NEWS_API_KEY=                        # https://newsapi.org/ (Free tier limited; Ghana = country=gh)
APITUBE_API_KEY=                     # https://apitube.io/ (Create account, generate key)
SERPAPI_KEY=                         # https://serpapi.com/ (Optional; demo mock used if blank)
```

### Getting the Keys
| Variable | Source | Steps |
|----------|--------|-------|
| NEWS_API_KEY | NewsAPI.org | Sign up → Dashboard → Copy API key |
| APITUBE_API_KEY | APITube.io | Register → API Dashboard → Generate key |
| SERPAPI_KEY | SerpAPI.com (optional) | Sign up → Dashboard → Key (demo fallback prints warning) |

If keys are missing:
* `NEWS_API_KEY` / `APITUBE_API_KEY`: Fetchers operate in demo mode (limited or mock content)
* `SERPAPI_KEY`: SerpAPI service returns mock Ghana news (tagged with IDs like `serpapi-<n>`) if enabled in feed.

---
## 6. Prerequisites
| Component | Requirement |
|-----------|-------------|
| Go Backend | Go ≥ 1.23 (module sets toolchain 1.24.4) |
| Frontend | Node.js (LTS 18+ or 20+) & npm |
| Python ML | Python 3.10+ recommended (virtual environment suggested) |
| SQLite | Included via driver – no manual install required |

---
## 7. Setup & Run (Backend + Frontend + ML)  
Commands show PowerShell (Windows). Unix variants provided where useful.

### 7.1 Clone & Enter
```powershell
git clone <repo-url>
cd DCIT316-semester-project
```

### 7.2 Create `.env`
Copy the template above. Adjust thresholds / keys.

### 7.3 Python Environment (for evaluation / model work)
```powershell
python -m venv myenv
myenv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```
Unix:
```bash
python3 -m venv myenv
source myenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 7.4 Run the Go Backend
```powershell
cd backend
go mod download
go run cmd/main.go
```
Server starts on `http://localhost:8080` (override with `PORT`). A scheduler auto-fetches news (initial + every ~10 mins) and cleans up older articles.

Health check:
```
GET http://localhost:8080/health
```

### 7.5 Run the Frontend
In a new terminal:
```powershell
cd frontend
npm install
npm start
```
Opens `http://localhost:3000` with proxy & `setupProxy.js` redirecting `/api` → backend (`http://localhost:8080`).

### 7.6 (Optional) Stand‑alone Proxy Script
```powershell
cd frontend
npm run proxy
```

---
## 8. Using the Application
1. Register a user (`POST /api/auth/signup`) or via frontend signup form.
2. Login → JWT stored locally by frontend.
3. Browse news (fetched & stored). If feed toggle is enabled and capacity permits, SerpAPI items appear.
4. Use fake news detection (`/api/news/detect`) – backend returns probability & boolean.
5. View recommendations (`/api/user/recommendations`). Heuristic uses interests + recency + history penalty if model absent.

---
## 9. Key API Endpoints (Condensed)
| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| POST | /api/auth/signup | Create user | No |
| POST | /api/auth/login | Obtain JWT | No |
| GET | /api/news/ | List news (category optional) | No |
| GET | /api/news/:id | Get single article | No / dynamic |
| POST | /api/news/detect | Classify fake vs real | No |
| POST | /api/news/ | Create article | Yes |
| POST | /api/news/activity | Log user activity | Yes |
| GET | /api/user/profile | Get profile | Yes |
| PUT | /api/user/profile | Update profile | Yes |
| GET | /api/user/recommendations | Personalized recs | Yes |
| GET | /api/external/ghana | Fetch Ghana news (APIs) | No |
| GET | /health | Service health | No |

JWT: Include header `Authorization: Bearer <token>`.

---
## 10. ML Models & Evaluation
Models expected in `./models/`:
* `lr_pipeline.onnx` – Logistic regression (text features) for fake news detection
* `recommender_scoring.onnx` – Recommendation scoring model

If absent, backend uses heuristic fallbacks (keyword frequency for fake news; interest+recency for recommendations).

### Evaluate Both Models
```powershell
python evaluate_models.py
```
Outputs JSON & text reports under `evaluation_results/` with metrics: accuracy, precision, recall, f1, (AUC if available), confusion matrix, classification report.

### Datasets
* `Data/liar_dataset/` – TSV splits (test/train/valid)
* `Data/MINDsmall_*` – MIND small training & dev sets for recommendation behavior simulation

### (NEW) ML Development & Training Workflow
This section explains how to work on or regenerate models.

#### 1. Create / Activate Virtual Environment
PowerShell:
```powershell
python -m venv myenv
myenv\Scripts\Activate.ps1
```
Unix:
```bash
python3 -m venv myenv
source myenv/bin/activate
```

#### 2. Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

If you later add packages (e.g. transformers), append them to `requirements.txt` so others can reproduce.

#### 3. Inspect / Run Training Scripts
Scripts reside in `scripts/`:
* `logistic_regression.py` – Preprocess + train logistic regression → export ONNX pipeline.
* `neural_nets.py` – (Placeholder / extension) for potential neural model training.

Typical pattern (example – adjust flags if added later):
```powershell
python scripts/logistic_regression.py --export-model models/lr_pipeline.onnx
```

Check script source for available arguments (if argparse added). If not, modify directly to tune hyperparameters.

#### 4. Updating ONNX Models
After retraining, ensure exported filenames match those the backend expects:
```
models/
	lr_pipeline.onnx
	recommender_scoring.onnx   # (add when recommender export implemented)
```
Restart backend so it reloads models (or just stop/start the Go process).

#### 5. Verifying Model Loads
Backend log lines will warn if a model path is missing. You can force a quick manual check by hitting:
```
POST /api/news/detect {"title":"Test", "content":"This is a sample"}
```
If heuristic fallback is active, probabilities may look coarse (increments of 0.2 via keyword heuristic).

#### 6. Re-Running Evaluation
```powershell
python evaluate_models.py --output-dir evaluation_results
```
If you modify the evaluation script to accept parameters, document them here. The script currently auto-discovers datasets in `Data/`.

#### 7. Adding New Models
1. Train / export to ONNX (consistent input shapes / types).
2. Place file in `models/`.
3. Extend `backend/internal/services/ml/` with a loader + inference method (currently heuristic stubs).
4. Inject into `cmd/main.go` initialization similarly to existing models.

#### 8. Common Gotchas
| Problem | Fix |
|---------|-----|
| ONNX export fails due to opset | Upgrade `onnx`, set supported `opset_version` in exporter |
| Memory error on large TF-IDF | Reduce `max_features` / components in SVD (see `evaluate_models.py`) |
| Model reload not reflected | Restart Go backend (no hot-reload of models yet) |
| MIND parsing skips many lines | Check TSV formatting; adjust indices / fallback logic in dataset loader |

#### 9. Performance Tips
* Cache vectorizers / embeddings instead of recomputing every run during experimentation.
* For larger experiments, consider a notebook (not committed) or add a dedicated `experiments/` folder ignored by Git.
* Use `tqdm` progress bars already integrated in evaluation for visibility.

#### 10. Reproducibility Recommendations
Add a `MODELS_VERSION.md` documenting:
```
Model: lr_pipeline.onnx
Date: 2025-09-24
Training Data: liar_dataset train.tsv (hash ...)
Vectorizer Params: max_features=..., ngram=(1,2)
SVD Components: 128
Metrics (valid): accuracy=..., f1=...
```
This helps trace model provenance for future improvements.

---
## 11. Development Tips
* Modify CORS origins in `backend/internal/middleware/cors.go` for production.
* Adjust fetch intervals / cleanup in `NewsScheduler` (`backend/internal/services/news_scheduler.go`).
* Add new external fetchers in `backend/internal/services/fetcher/` and integrate via composite fetcher logic.
* Extend ML layer in `backend/internal/services/ml/` – replace heuristic with real ONNX inference (future enhancement).

---
## 12. Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| Backend prints warning about model file | ONNX missing | Place model in `models/` or ignore (fallback active) |
| Empty news feed | No API key / network | Add `NEWS_API_KEY` or `APITUBE_API_KEY`; check connectivity |
| SerpAPI mock results only | Missing `SERPAPI_KEY` | Obtain real key & set env var |
| CORS errors in browser | Origin not allowed | Update `NewCORSMiddleware()` origins list |
| 401 on protected endpoints | Missing / expired JWT | Re-login, include Authorization header |
| Database corruption / want reset | Stale `backend.db` | Stop server, delete file, restart (schema auto-migrates) |
| Scheduler not fetching | App just started | Wait first cycle or check logs; ensure keys configured |

Log statements provide context for fetch cycles and fallback usage.

---
## 13. Common Commands Cheat-Sheet
```powershell
# Backend run
cd backend; go run cmd/main.go

# Backend tests (if/when added)
go test ./...

# Frontend run
cd frontend; npm start

# Evaluation (activate venv first)
python evaluate_models.py

# Regenerate dependencies (Go)
go mod tidy
```

---
## 14. Security Notes
* Replace `JWT_SECRET` in production with a long random value (≥ 32 chars).
* Consider rate limiting and request validation for public endpoints.
* Do NOT commit real API keys or secrets – keep `.env` out of version control (add to `.gitignore`).

---
## 15. Roadmap / Possible Enhancements
* Real ONNX runtime inference embedded in Go (cgo / bindings)
* User feedback loop to refine recommendations
* Vector search (e.g., embedding-based similarity) for related articles
* Docker Compose for unified local spin-up
* CI pipeline (test + lint + build) and code coverage
* Pagination & caching layer (Redis) for scalability
* Admin dashboard for moderation / analytics

---
## 16. Contributing
1. Fork → Branch (`feature/improve-detection`)
2. Make changes + tests
3. PR with description & screenshots (if UI)

---
## 17. License
This project is licensed under the MIT License. See the `LICENSE` file for full text. While originated as an academic course project, the permissive MIT terms allow reuse, modification, and distribution—remember to retain the copyright notice.

---
## 18. Acknowledgements
* LIAR dataset – Fake news classification research
* MIND dataset – Microsoft News recommendation research
* NewsAPI / APITube / SerpAPI – External news data

---
## 19. Quick Start (TL;DR)
```powershell
git clone <repo>
cd DCIT316-semester-project
copy NUL .env  # then paste vars from section 5
python -m venv myenv; myenv\Scripts\Activate.ps1
pip install -r requirements.txt
cd backend; go run cmd/main.go
# new terminal
cd frontend; npm install; npm start
```

Open http://localhost:3000 and explore.


