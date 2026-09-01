# Adaptive Enterprise RAG

An Adaptive-K enterprise RAG research system. It chooses an initial retrieval depth from **query-only** features, verifies retrieved evidence, escalates K only when needed, then compares the resulting run with genuine Fixed K=3, K=5, and K=10 baselines.

The initial-K ML model is an experimental mechanism, not a validated optimal-K predictor: on the current 100-question proxy-labelled dataset, exact-K predictive performance was weak. The system therefore keeps the deterministic query-derived fallback and relies on verification-driven escalation.

## Prerequisites

- Windows PowerShell and Python 3.13 (the included `.venv` is supported)
- Ollama running locally with `qwen3:8b` available
- Existing `chroma_db/` and `models/adaptive_k_random_forest_*.joblib` artifacts (already present in this repository)

## Install

```powershell
cd 'C:\Users\Vaishnavi D\Documents\Projects\adaptive-enterprise-rag-second-attempt'
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = 'src'
```

If PowerShell blocks activation, run the commands with `.\.venv\Scripts\python.exe` instead of `python`.

## Run the website

Build the React frontend once (or after making UI changes):

```powershell
cd frontend
npm install
npm run build
cd ..
```

The WSGI app automatically serves `frontend/dist` when present. If it has not
yet been built, it falls back to the legacy static page so the API remains
usable.

In a separate terminal, start Ollama and ensure the configured generation model is available:

```powershell
ollama serve
ollama pull qwen3:8b
```

Then start the app:

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m api.app
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Development-only accounts are:

| Role | Email | Password |
| --- | --- | --- |
| Employee | `employee@example.com` | `change-me-employee` |
| HR | `hr@example.com` | `change-me-hr` |

For production, set `APP_USERS_JSON` to an externally managed JSON user map before starting the server. Never commit credentials or `.env` files.

## Run checks

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m compileall src
.\.venv\Scripts\python.exe -m unittest src.adaptive.test_core src.api.test_app -v
```

The RAGAS scripts are intentionally separate from ordinary tests because they invoke an evaluator and can be long-running. Do not run them as part of routine test execution.

## Final controlled experiment

This runs each question under Fixed K=3, Fixed K=5, Fixed K=10, and Adaptive K using identical generation configuration. It creates one telemetry row per question × mode and leaves quality fields as `pending` until validated evaluation results exist.

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m analysis.run_final_experiment --limit 3
```

Remove `--limit 3` for the full evaluation set. Outputs are saved to `results/final/final_experiment_results.csv` and `.json`.

## Configuration

The runtime reads optional environment variables: `AE_RAG_MIN_K`, `AE_RAG_MAX_K`, `AE_RAG_DEFAULT_K`, `AE_RAG_ESCALATION_STEP`, `AE_RAG_VERIFICATION_THRESHOLD`, `AE_RAG_CHROMA_PATH`, `AE_RAG_MODEL_PATH`, `AE_RAG_GENERATION_MODEL`, `AE_RAG_OLLAMA_URL`, and `AE_RAG_PORT`.

Research invariants: initial K never uses retrieval or answer outcomes; adaptive K never exceeds `MAX_K`; verification only escalates; fixed baselines never adapt; retrieved and generation-used chunks are recorded separately.
