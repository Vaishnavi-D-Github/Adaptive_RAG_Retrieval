# PROJECT INSPECTION REPORT

Inspection date: 2026-09-03  
Project root: `C:\Users\Vaishnavi D\Documents\Projects\adaptive-enterprise-rag-second-attempt`

This report is based on read-only inspection of the project directory and source code. Existing project files, databases, ChromaDB stores, research outputs, model artifacts, PDFs, and dependency folders were not modified.

## 1. Project Overview

The project is an Adaptive-K enterprise RAG application plus a substantial research/evaluation workspace.

The current application is a Python WSGI backend in `src/api/app.py` that serves a Vite/React frontend from `frontend/dist` when available. It supports session-based authentication, two roles in the current code (`employee`, `hr`), HR-only document upload, application-only ChromaDB indexing, adaptive retrieval, fixed-K retrieval, Ollama generation using `qwen3:8b`, telemetry, and file-backed user query history.

The research side contains static GAO PDFs, pre-extracted chunk JSON files, a research ChromaDB, RAGAS datasets, retrieval experiments, final experiment outputs, diagnostic CSV/JSON files, and trained Adaptive-K model artifacts.

## 2. Complete Directory Tree

Dependency/cache and generated internals are summarized rather than fully expanded, per the inspection constraints.

```text
.
├── .env
├── .env.example
├── .gitignore
├── .ragas_venv/                         DEPENDENCY/CACHE
│   ├── Include/
│   ├── Lib/
│   ├── Scripts/
│   └── share/
├── .venv/                               DEPENDENCY/CACHE
├── chroma_db/                           DATABASE, research ChromaDB
│   ├── chroma.sqlite3
│   └── 2b382e00-a8d7-4201-9e0c-7aa2b8d76b7e/
├── chroma_db.zip                        DATABASE archive
├── data/
│   ├── application_chroma_db/            DATABASE, app ChromaDB
│   │   ├── chroma.sqlite3
│   │   └── 1c249aca-4ddf-4e03-8fc2-a72bd2cab74f/
│   ├── chunks/                           RESEARCH DATA, research chunks
│   ├── documents/                        RESEARCH DATA, GAO PDFs
│   ├── evaluation/                       RESEARCH DATA, evaluation CSVs
│   ├── query_history/                    APPLICATION DATA, per-user JSONL history
│   └── uploads/
│       ├── chunks/                       APPLICATION DATA, uploaded-document chunks
│       ├── documents/                    APPLICATION DATA, uploaded originals
│       └── manifest.jsonl
├── frontend/
│   ├── dist/                             GENERATED BUILD FILE
│   ├── index.html
│   ├── node_modules/                     DEPENDENCY/CACHE
│   ├── package-lock.json
│   ├── package.json
│   ├── public/images/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   ├── index.css
│   │   ├── main.tsx
│   │   ├── services/api.ts
│   │   └── types.ts
│   ├── tsconfig*.json / *.tsbuildinfo
│   └── vite.config.*
├── models/                               EXPERIMENT, trained model artifacts
├── results/                              EXPERIMENT / RESEARCH DATA
│   ├── adaptive/
│   ├── audit_generation/
│   ├── evaluation/
│   ├── final/
│   ├── ragas/
│   ├── retrieval/
│   └── retrieval_v2/
├── scripts/                              UNKNOWN, directory present but no inspected files
├── sql/
│   └── create_rag_telemetry.sql
├── src/
│   ├── adaptive/                         SHARED application/research runtime
│   ├── analysis/                         EXPERIMENT
│   ├── api/                              BACKEND
│   ├── retrieval/                        SHARED retrieval adapter
│   ├── *.py                              research scripts, tests, runtime
│   └── __pycache__/                      DEPENDENCY/CACHE
├── README.md
├── requirements.txt
└── root-level experiment CSV/JSON files
```

## 3. Complete File Inventory

### Application / Backend

- `src/api/app.py` - BACKEND. WSGI API, static serving, auth/session endpoints, query endpoint, HR upload/list endpoints.
- `src/api/__init__.py` - BACKEND. API package marker.
- `src/runtime.py` - SHARED. Runtime wiring for Chroma, embeddings, K model, Ollama, pipelines.
- `src/adaptive/config.py` - CONFIGURATION. Environment loading and runtime constants.
- `src/adaptive/auth.py` - BACKEND. MySQL-backed users/auth service.
- `src/adaptive/document_store.py` - BACKEND. MySQL metadata store for uploaded app documents.
- `src/adaptive/upload_store.py` - BACKEND. File storage for uploaded originals and extracted chunks.
- `src/adaptive/history_store.py` - BACKEND. Per-user query history JSONL store.
- `src/adaptive/app_ingestion.py` - BACKEND. Application upload extraction/chunking/embedding validation/Chroma insertion helpers.
- `src/adaptive/mysql_store.py` - BACKEND. MySQL telemetry persistence.

### Shared RAG Runtime

- `src/adaptive/adaptive_pipeline.py` - SHARED. Adaptive-K pipeline.
- `src/adaptive/fixed_pipeline.py` - SHARED. Fixed K=3/5/10 pipeline.
- `src/adaptive/retrieval_controller.py` - SHARED. Retriever adapter.
- `src/retrieval/chroma_retriever.py` - SHARED. Chroma vector query wrapper.
- `src/adaptive/k_policy.py` - SHARED. Bootstrap and ML K policies.
- `src/adaptive/k_model.py` - SHARED. RandomForest K model wrapper.
- `src/adaptive/complexity_model.py` - EXPERIMENT/SHARED. RandomForest complexity model; not wired into production unless supplied.
- `src/adaptive/feature_extractor.py` - SHARED. Query-only feature extraction.
- `src/adaptive/query_optimizer.py` - SHARED. Query normalization and keyword expansion.
- `src/adaptive/context_optimizer.py` - SHARED. Deduplicates retrieved chunks.
- `src/adaptive/prompt_orchestrator.py` - SHARED. Plain-text context and prompt construction.
- `src/adaptive/verifier.py` - SHARED. Retrieval sufficiency check.
- `src/adaptive/telemetry.py` - SHARED. Runtime telemetry construction.
- `src/adaptive/schemas.py` - SHARED. Dataclasses for features, plans, verification, and results.
- `src/adaptive/__init__.py` - SHARED. Package marker/exports.

### Frontend

- `frontend/src/main.tsx` - FRONTEND. React root entry.
- `frontend/src/App.tsx` - FRONTEND. App shell, auth gate, sidebar, route state.
- `frontend/src/services/api.ts` - FRONTEND. Fetch client for `/api/*`.
- `frontend/src/types.ts` - FRONTEND. TypeScript domain types.
- `frontend/src/components/AuthPage.tsx` - FRONTEND. Login/signup UI.
- `frontend/src/components/ChatPage.tsx` - FRONTEND. Query UI, answer rendering, sources, telemetry details.
- `frontend/src/components/DocumentsPage.tsx` - FRONTEND. HR upload/list UI.
- `frontend/src/components/HistoryPage.tsx` - FRONTEND. Per-user query history UI.
- `frontend/src/components/AdaptiveCanvas.tsx` - FRONTEND. Chat welcome visual/prompt presets.
- `frontend/src/components/IntroGate.tsx` - FRONTEND. Intro animation.
- `frontend/src/index.css` and component CSS files - FRONTEND. Styling.
- `frontend/index.html` - FRONTEND. Vite HTML entry.
- `frontend/public/images/knowledge-network-intro.png` - FRONTEND asset.

### Configuration / Dependency Manifests

- `requirements.txt` - CONFIGURATION. Python runtime dependencies.
- `frontend/package.json` - CONFIGURATION. Node dependencies and scripts.
- `frontend/package-lock.json` - CONFIGURATION. Locked Node dependency tree.
- `frontend/vite.config.ts`, `frontend/vite.config.js`, `frontend/vite.config.d.ts` - CONFIGURATION. Vite build config and generated declarations.
- `frontend/tsconfig.json`, `frontend/tsconfig.node.json` - CONFIGURATION. TypeScript config.
- `.env` - CONFIGURATION. SECRET PRESENT - VALUES REDACTED.
- `.env.example` - CONFIGURATION. Example runtime env.
- `.gitignore` - CONFIGURATION.
- `sql/create_rag_telemetry.sql` - CONFIGURATION / DATABASE. SQL schema for `rag_runs`.

### Tests

- `src/api/test_app.py` - TEST. API/auth/upload/history behavior tests.
- `src/adaptive/test_core.py` - TEST. Feature extraction and pipeline basics.
- `src/adaptive/test_adaptive_pipeline.py` - TEST/diagnostic script. Prints policy output.
- `src/test_ollama_telemetry.py` - TEST. Ollama metadata telemetry tests.
- `src/test_qwen_json.py` - TEST. Qwen JSON-related test.
- `src/test_ragas_smoke.py` - TEST/EXPERIMENT. RAGAS smoke testing.

### Research / Experiment Code

- `src/ingest.py` - EXPERIMENT. Research PDF text extraction and chunk JSON generation.
- `src/build_vector_store.py` - EXPERIMENT. Builds research ChromaDB from `data/chunks`.
- `src/check_vector_store.py` - EXPERIMENT. Peeks research ChromaDB.
- `src/rag.py` - EXPERIMENT. Standalone research RAG demo.
- `src/retrieval_experiment.py` - EXPERIMENT. K retrieval source coverage experiment.
- `src/prepare_ragas_dataset.py` - EXPERIMENT. RAGAS dataset preparation.
- `src/evaluate_ragas_style.py`, `src/evaluate_ragas_style_revised.py` - EXPERIMENT. RAGAS-style evaluator scripts.
- `src/evaluation_worker.py` - EXPERIMENT. Evaluation worker.
- `src/audit_generation_experiment.py` - EXPERIMENT. Generation audit analysis.
- `src/analysis/*.py` - EXPERIMENT. Adaptive-K analysis, diagnostics, final experiments, model training.
- `src/analysis/adaptive_k_experiment/*.py` - EXPERIMENT. Engineered feature experiments.
- `src/analysis/adaptive_k_embedding_experiment/*.py` - EXPERIMENT. Embedding + engineered feature experiments.

### Research Data / Experiment Outputs

- `data/documents/*.pdf` - RESEARCH DATA. GAO source PDFs.
- `data/chunks/*.json` - RESEARCH DATA. Research chunk files.
- `data/evaluation/*.csv` - RESEARCH DATA. Evaluation question sets.
- `enterprise_query_evaluation_set_100.csv` - RESEARCH DATA.
- `candidate_k_vs_query_characteristics.csv`, `candidate_k_diagnostic_report.json`, `per_question_quality_vs_k_curve_summary.csv` - EXPERIMENT diagnostics.
- `results/**` - EXPERIMENT / RESEARCH DATA. RAGAS datasets, retrieval outputs, adaptive diagnostics, final experiments, audit results.
- `models/adaptive_k_random_forest_20260829T063710Z.joblib` - EXPERIMENT. Trained K model binary.
- `models/adaptive_k_random_forest_20260829T063710Z.json` - EXPERIMENT. Model report/metadata.

### Application Data / Databases / Generated / Cache

- `data/uploads/documents/*` - APPLICATION DATA. Uploaded originals.
- `data/uploads/chunks/*.json` - APPLICATION DATA. Uploaded-document extracted chunks.
- `data/uploads/manifest.jsonl` - APPLICATION DATA. Legacy/append-only upload manifest.
- `data/query_history/*.jsonl` - APPLICATION DATA. Per-user query history.
- `data/application_chroma_db/**` - DATABASE. Application ChromaDB.
- `chroma_db/**` - DATABASE. Research ChromaDB.
- `chroma_db.zip` - DATABASE archive.
- `frontend/dist/**` - GENERATED BUILD FILE. Built frontend served by backend.
- `frontend/node_modules/**`, `.venv/**`, `.ragas_venv/**`, `**/__pycache__/**`, `*.tsbuildinfo` - DEPENDENCY/CACHE or GENERATED BUILD FILE.

## 4. Application Entry Points

- Backend entry point: `src/api/app.py`, function `main()`.
- API entry point: `src/api/app.py`, function `create_app(...)`, nested WSGI `application(environ, start_response)`.
- Frontend entry point: `frontend/src/main.tsx`.
- Frontend app shell: `frontend/src/App.tsx`.
- Static frontend served by backend: `frontend/dist` if present, otherwise `src/api/static`.
- Application startup command from README: `$env:PYTHONPATH = 'src'`; `.\.venv\Scripts\python.exe -m api.app`.
- Development frontend command: `cd frontend`; `npm run dev`.
- Production frontend build command: `cd frontend`; `npm run build`.
- Production backend command shown by code/README: `python -m api.app` with `frontend/dist` present.
- Ollama prerequisite: `ollama serve`; `ollama pull qwen3:8b`.

## 5. Architecture

```text
React UI
  -> frontend/src/services/api.ts
  -> same-origin WSGI endpoints in src/api/app.py
  -> auth/session checks
  -> runtime pipelines from src/runtime.py
  -> ChromaRetriever over application ChromaDB
  -> AdaptivePipeline or FixedKPipeline
  -> prompt_orchestrator
  -> OllamaGenerator qwen3:8b
  -> telemetry/history response
```

Research code uses the same `adaptive` runtime modules in places but defaults to the research ChromaDB (`chroma_db`, collection `enterprise_documents`). The web application explicitly uses `load_application_runtime(...)`, which points to `data/application_chroma_db`, collection `application_uploaded_documents`.

## 6. Backend Architecture

FILE: `src/api/app.py`  
FUNCTION/CLASS: `create_app`  
PURPOSE: Builds the WSGI app, owns routing, JSON responses, static serving, auth checks, query flow, and HR upload flow.  
CALLED BY: `main()`, API tests.  
CALLS: `MySQLAuthService`, `SessionStore`, `FileHistoryStore`, `MySQLDocumentStore`, `UploadStore`, `adaptive_pipeline.run`, `fixed_pipeline.run`, `extract_pages`, `chunk_pages`, Chroma `upsert`.  
IMPORTANT DATA: session cookie `ae_rag_session`, upload limit 20 MB, allowed suffixes `.pdf`, `.docx`, `.txt`, app collection metadata.  
NOTES: Sessions are in-memory, so login sessions do not survive server restart.

FILE: `src/api/app.py`  
FUNCTION/CLASS: `main`  
PURPOSE: Loads production application runtime, starts `wsgiref.simple_server`.  
CALLED BY: `python -m api.app`.  
CALLS: `runtime.load_application_runtime`, `make_server`.  
IMPORTANT DATA: `AE_RAG_HOST`, `AE_RAG_PORT`, application Chroma path and collection.  
NOTES: Default server is development-grade WSGI, not a hardened production server.

## 7. Frontend Architecture

Pages currently present in React:

- Login/register: `AuthPage`.
- Intro animation: `IntroGate`.
- Chat/query interface: `ChatPage`.
- History: `HistoryPage`.
- HR document management: `DocumentsPage`.
- Settings: inline `SettingsPage` in `App.tsx`.

Routing/navigation is local React state, not React Router. `App.tsx` has `type Page = "chat" | "history" | "documents" | "settings"`. Documents are hidden unless `user.role === "hr"` and `navigate()` also blocks non-HR access to the documents page.

Important UI capabilities:

- Login/register form with role selection (`employee`, `hr`).
- Query composer with retrieval mode select (`adaptive`, `fixed_3`, `fixed_5`, `fixed_10`).
- Markdown answer rendering through `react-markdown`.
- Source cards.
- Retrieval diagnostics and expandable technical details.
- History list with selected-run detail.
- HR file upload with client-side file type/size checks.
- Loading and error states in auth, query, history, and upload workflows.

## 8. Database Architecture

### MySQL

Configured through environment variables:

- `AE_RAG_DB_HOST`
- `AE_RAG_DB_PORT`
- `AE_RAG_DB_USER`
- `AE_RAG_DB_PASSWORD` - SECRET PRESENT - VALUE REDACTED
- `AE_RAG_DB_NAME`

Tables shown in code:

- `users` - expected by `MySQLAuthService`; not created by `auth.py`. Stores `user_id`, `full_name`, `email`, `password_hash`, `role`, `is_active`, timestamps.
- `rag_runs` - created by `sql/create_rag_telemetry.sql`; inserted by `MySQLTelemetryStore.save_run`.
- `app_documents` - created automatically by `MySQLDocumentStore.ensure_schema`.

`rag_runs` stores question/run metadata, mode, model/config, K decisions, verification fields, chunk counts, token counts, latency fields, Ollama duration fields, generated answer, JSON query features, predicted K probabilities, retrieved sources, context/prompt lengths, fallback flag, and optional evaluation scores.

`app_documents` stores uploaded document metadata: document id, filenames, path, type, size, content hash, uploader user id, indexing status, page/word/character/chunk/embedding/Chroma counts, error message, metadata JSON, timestamps, and indexes.

## 9. ChromaDB Architecture

Research ChromaDB:

- Path: `chroma_db` or `AE_RAG_CHROMA_PATH`.
- Collection name: `enterprise_documents`.
- Created/used by `src/build_vector_store.py`, `src/rag.py`, `src/retrieval_experiment.py`, and `runtime.load_runtime()` default.
- Files observed: `chroma_db/chroma.sqlite3`, UUID index directory with `.bin` files.

Application ChromaDB:

- Path: `data/application_chroma_db` or `AE_RAG_APP_CHROMA_PATH`.
- Collection name: `application_uploaded_documents` or `AE_RAG_APP_COLLECTION`.
- Used by `runtime.load_application_runtime()` and production API.
- Files observed: `data/application_chroma_db/chroma.sqlite3`, UUID index directory with `.bin` files.

Application queries should not use the research corpus when started through `api.app.main()`, because `main()` calls `load_application_runtime()` and passes the app collection into both pipelines. The test `test_empty_application_knowledge_base_does_not_use_research_corpus` confirms intended behavior when app collection count is zero.

Risk: `runtime.load_runtime()` still defaults to research ChromaDB. Any future backend code that accidentally calls `load_runtime()` instead of `load_application_runtime()` could re-couple app queries to research data.

## 10. Document Ingestion Pipeline

Application upload flow:

1. `frontend/src/components/DocumentsPage.tsx` reads selected file as Base64.
2. `frontend/src/services/api.ts` sends `POST /api/hr/upload`.
3. `src/api/app.py` requires `role == "hr"`.
4. `UploadStore.stored_filename` derives a hash-prefixed stored filename.
5. `MySQLDocumentStore.create_pending` creates `app_documents` row with status `UPLOADED`.
6. `UploadStore.save_original` writes original bytes to `data/uploads/documents`.
7. `extract_pages` extracts text.
8. `chunk_pages` creates page-based chunks.
9. `UploadStore.save_chunks` writes chunk JSON under `data/uploads/chunks`.
10. `embedding_model.encode(...)` creates vectors.
11. `collection.upsert(...)` inserts documents, embeddings, and metadata.
12. `verify_chroma_records` confirms records exist.
13. `MySQLDocumentStore.update_status` updates counts and `INDEXED` status.

## 11. Chunking Implementation

Application:

- FILE: `src/adaptive/app_ingestion.py`
- FUNCTION/CLASS: `chunk_pages`
- PURPOSE: Creates word chunks per extracted page.
- IMPORTANT DATA: `CHUNK_SIZE_WORDS = 500`, `CHUNK_OVERLAP_WORDS = 100`, stride 400.
- CHUNK IDS: Sequential per document starting at 1.
- CHROMA IDS: `app_{document_id}_p{page_number}_c{chunk_id}`.
- METADATA: `scope`, `document_id`, `document`, `page`, `chunk_id`.

Research:

- FILE: `src/ingest.py`
- FUNCTION/CLASS: `create_chunks`
- PURPOSE: Creates word chunks per PDF page.
- IMPORTANT DATA: `CHUNK_SIZE = 500`, `CHUNK_OVERLAP = 100`.
- CHUNK IDS: Sequential per document starting at 1.
- METADATA IN CHUNK JSON: `chunk_id`, `page_number`, `text`.

## 12. Table Handling

Does the current implementation preserve PDF tables as actual table structures, or only extract their text?

Answer: It only extracts text.

Evidence in code:

- Application PDF extraction uses `page.get_text("text")` in `src/adaptive/app_ingestion.py`.
- Research PDF extraction uses `page.get_text("text")` in `src/ingest.py`.
- No inspected ingestion path calls PyMuPDF table APIs, `pdfplumber`, Camelot, Tabula, OCR layout analysis, HTML extraction, markdown table serialization, cell coordinates, row/column metadata, or table-specific chunk metadata.
- Chunk metadata contains page and document fields only, not table IDs, row/column coordinates, captions, or structured cells.

Implication: if a PDF contains tables, table content may be flattened into plain text in whatever order PyMuPDF emits. The system cannot reliably know that a source was tabular or reconstruct table structure from stored metadata.

## 13. Embedding Pipeline

FILE: `src/runtime.py`  
FUNCTION/CLASS: `load_runtime`, `load_application_runtime`  
PURPOSE: Instantiates `SentenceTransformer(EMBEDDING_MODEL)` and wires it to `ChromaRetriever`.  
IMPORTANT DATA: default embedding model `all-MiniLM-L6-v2`.

FILE: `src/retrieval/chroma_retriever.py`  
FUNCTION/CLASS: `ChromaRetriever.retrieve`  
PURPOSE: Encodes query with `embedding_model.encode([query])[0].tolist()` and queries Chroma.

FILE: `src/adaptive/app_ingestion.py`  
FUNCTION/CLASS: upload flow via `embedding_model.encode([chunk.text for chunk in chunks])`  
PURPOSE: Encodes each app-uploaded chunk before Chroma upsert.

## 14. Retrieval Pipeline

FILE: `src/retrieval/chroma_retriever.py`  
FUNCTION/CLASS: `ChromaRetriever.retrieve`  
PURPOSE: Query embedding -> Chroma `collection.query(query_embeddings=[embedding], n_results=k)` -> list of `{document, metadata, distance}`.  
CALLED BY: `RetrievalController.retrieve`.

FILE: `src/adaptive/retrieval_controller.py`  
FUNCTION/CLASS: `RetrievalController.retrieve`  
PURPOSE: Adapter around retrievers exposing `retrieve(query, k)` or `search(query, k)`.

Default K configuration:

- `DEFAULT_K = 5` in `src/adaptive/config.py`, but the production adaptive pipeline uses policy-selected K.
- Fixed K values: `(3, 5, 10)`.
- Maximum K: default `10`.

Metadata filtering: no filtering was found in `ChromaRetriever.retrieve`; it does not pass a Chroma `where` or `where_document` filter.

Role-based filtering: no role filtering exists in retrieval. All authenticated users query the same application collection.

## 15. Adaptive-K Pipeline

FILE: `src/adaptive/adaptive_pipeline.py`  
FUNCTION/CLASS: `AdaptivePipeline.run`  
PURPOSE: Extract query features, choose initial K, optimize query, retrieve, verify, escalate K if needed, optimize context, build prompt, generate answer, build telemetry, optionally persist telemetry.  
CALLED BY: `src/api/app.py` `/api/query` route when `mode == "adaptive"`.  
CALLS: `extract_features`, `k_policy.predict`, `optimize_query`, `RetrievalController.retrieve`, `verify_retrieval`, `optimize_context`, `build_context`, `build_prompt`, `_generate`, `build_telemetry`, `telemetry_store.save_run`.  
IMPORTANT DATA: `selected_k`, `initial_k`, `maximum_k`, retrieval attempts, generated answer, telemetry.

FILE: `src/adaptive/k_policy.py`  
FUNCTION/CLASS: `MLPredictiveKPolicy.predict`  
PURPOSE: Uses loaded `KModel` and optional complexity model to predict K from query-only features.  
NOTES: If no trained K model exists, `BootstrapKPolicy` is used.

FILE: `src/adaptive/k_model.py`  
FUNCTION/CLASS: `KModel`  
PURPOSE: RandomForest classifier over query-derived feature columns; can load/save model artifacts and predict with probabilities.

## 16. Fixed-K Pipeline

FILE: `src/adaptive/fixed_pipeline.py`  
FUNCTION/CLASS: `FixedKPipeline.run`  
PURPOSE: Runs a non-adaptive baseline at K=3, K=5, or K=10; retrieves once, optimizes context, builds prompt, optionally generates, builds telemetry.  
CALLED BY: `src/api/app.py` `/api/query` for `fixed_3`, `fixed_5`, `fixed_10`.

## 17. Verification

FILE: `src/adaptive/verifier.py`  
FUNCTION/CLASS: `verify_retrieval`  
PURPOSE: Determines if retrieved evidence is sufficient using result count and best Chroma distance converted to `1 / (1 + distance)`.  
IMPORTANT DATA: default `minimum_results = 1`, default `relevance_threshold = 0.35`.  
NOTES: The configured `VERIFICATION_THRESHOLD` in `config.py` is not passed into this function by `AdaptivePipeline.run`; the function default is used.

Retrieval escalation:

- In `AdaptivePipeline.run`, if verification is insufficient and `current_k < plan.maximum_k`, K increases by `plan.escalation_step` up to `maximum_k`.
- Default ML policy escalation step is 2.
- Bootstrap policy escalation step defaults to 2.

## 18. LLM Pipeline

FILE: `src/runtime.py`  
FUNCTION/CLASS: `OllamaGenerator.__call__`  
PURPOSE: Sends chat request to Ollama with model `GENERATION_MODEL`, default `qwen3:8b`.  
IMPORTANT DATA: `temperature = 0`, `num_predict = 512`, `think = False`, host from `OLLAMA_BASE_URL`.

FILE: `src/adaptive/prompt_orchestrator.py`  
FUNCTION/CLASS: `build_prompt`  
PURPOSE: Instructs model to answer only from supplied evidence and state insufficiency when evidence is insufficient.

## 19. Authentication

FILE: `src/adaptive/auth.py`  
FUNCTION/CLASS: `MySQLAuthService`  
PURPOSE: Creates users, authenticates users, fetches users by ID.  
IMPORTANT DATA: roles currently limited to `employee` and `hr`; passwords hashed with Argon2id.

FILE: `src/api/app.py`  
FUNCTION/CLASS: `SessionStore`, `_cookie_header`, `current_user`, `require_user`, `require_role`  
PURPOSE: In-memory session tokens, HttpOnly cookie, route authorization.

Frontend:

- `AuthPage` supports login/signup.
- Signup role select only has `Employee` and `HR`.

## 20. Current Authorization

Current roles: `employee`, `hr`.

Current access:

- Any authenticated user can call `/api/query`.
- Any authenticated user can call `/api/history` and only sees their file-backed history.
- Only `hr` can call `/api/hr/upload`.
- Only `hr` can call `/api/hr/documents`.
- Only `hr` sees Documents in the sidebar.

Missing for new requirements:

- No `office`, `teacher`, or `student` roles.
- No document-level role/scope metadata.
- No retrieval-time metadata filter by document access level.
- No upload UI/API field for document audience.
- No role hierarchy logic.

## 21. Telemetry / History

FILE: `src/adaptive/telemetry.py`  
FUNCTION/CLASS: `build_telemetry`  
PURPOSE: Produces telemetry dictionary for query features, K decisions, retrieval counts, latencies, tokens, verification, sources, context/prompt lengths.

FILE: `src/adaptive/mysql_store.py`  
FUNCTION/CLASS: `MySQLTelemetryStore.save_run`  
PURPOSE: Persists telemetry to MySQL `rag_runs`.

FILE: `src/api/app.py`  
FUNCTION/CLASS: `_history_record`, `_history_append`, `_history_list`  
PURPOSE: Normalizes result into per-user history records.

FILE: `src/adaptive/history_store.py`  
FUNCTION/CLASS: `FileHistoryStore`  
PURPOSE: Appends and reads per-user JSONL history from `data/query_history`.

## 22. Research / Application Separation

Research corpus:

- PDFs: `data/documents`.
- Chunks: `data/chunks`.
- Chroma: `chroma_db`.
- Collection: `enterprise_documents`.
- Scripts: `src/ingest.py`, `src/build_vector_store.py`, `src/rag.py`, `src/retrieval_experiment.py`, `src/analysis/**`, RAGAS/evaluation scripts.

Application corpus:

- Uploaded originals: `data/uploads/documents`.
- Uploaded chunks: `data/uploads/chunks`.
- Chroma: `data/application_chroma_db`.
- Collection: `application_uploaded_documents`.
- App ingestion: `src/adaptive/app_ingestion.py`.
- App runtime: `runtime.load_application_runtime()`.

Can application queries access the research corpus?

- Through the normal backend entry point `src/api/app.py:main`, no: it calls `load_application_runtime`, which points to application ChromaDB and `application_uploaded_documents`.
- Through accidental future wiring, yes: `runtime.load_runtime()` defaults to research `chroma_db` and `enterprise_documents`; using it in the web app would point queries to the research corpus.

## 23. Important Functions and File Paths

FILE: `src/adaptive/app_ingestion.py`  
FUNCTION/CLASS: `extract_pages`  
PURPOSE: Extracts text from TXT/PDF/DOCX uploads.  
CALLED BY: `src/api/app.py` upload route; `ingest_uploaded_document`.  
CALLS: PyMuPDF `page.get_text("text")`, `python-docx` paragraph extraction.  
IMPORTANT DATA: returns `ExtractedPage(page_number, text)`.  
NOTES: Tables are not preserved as structures.

FILE: `src/adaptive/app_ingestion.py`  
FUNCTION/CLASS: `chunk_pages`  
PURPOSE: Converts extracted pages to `AppChunk` records.  
CALLED BY: upload route.  
CALLS: none outside standard string operations.  
IMPORTANT DATA: 500-word chunks, 100-word overlap, page number retained.

FILE: `src/adaptive/app_ingestion.py`  
FUNCTION/CLASS: `AppChunk.metadata`  
PURPOSE: Creates Chroma metadata.  
IMPORTANT DATA: `scope=application`, `document_id`, `document`, `page`, `chunk_id`.

FILE: `src/retrieval/chroma_retriever.py`  
FUNCTION/CLASS: `ChromaRetriever.retrieve`  
PURPOSE: Embeds query and fetches top K chunks.  
IMPORTANT DATA: no metadata filter.

FILE: `src/adaptive/prompt_orchestrator.py`  
FUNCTION/CLASS: `build_context`, `build_prompt`  
PURPOSE: Plain-text context block and answer prompt.

FILE: `src/api/app.py`  
FUNCTION/CLASS: `/api/query` route inside `application`  
PURPOSE: Runs adaptive or fixed retrieval for authenticated users; blocks empty app corpus.

FILE: `src/api/app.py`  
FUNCTION/CLASS: `/api/hr/upload` route inside `application`  
PURPOSE: HR-only upload, storage, extraction, chunking, embedding, Chroma upsert, document metadata update.

FILE: `frontend/src/components/ChatPage.tsx`  
FUNCTION/CLASS: `ChatPage`, `Message`, `Sources`, `Retrieval`  
PURPOSE: Query UX, answer, source display, telemetry display.

FILE: `frontend/src/components/DocumentsPage.tsx`  
FUNCTION/CLASS: `DocumentsPage`, `readFile`  
PURPOSE: HR upload/list UX.

## 24. Dependencies

Python dependencies from `requirements.txt`:

- `chromadb==1.5.9`
- `sentence-transformers==6.0.0`
- `ollama==0.6.2`
- `pymupdf==1.28.2`
- `python-docx>=1.1,<2`
- `joblib==1.5.3`
- `pandas==3.0.5`
- `scikit-learn==1.9.0`
- `ragas==0.3.9`
- `langchain-ollama>=0.3,<1`
- `langchain-community==0.4.2`
- `pymysql>=1.1,<2`
- `argon2-cffi==25.1.0`

Node dependencies from `frontend/package.json`:

- Runtime: `@react-three/drei`, `@react-three/fiber`, `@vitejs/plugin-react`, `framer-motion`, `lucide-react`, `react`, `react-dom`, `react-markdown`, `three`.
- Dev: `@types/node`, `@types/react`, `@types/react-dom`, `typescript`, `vite`.

## 25. Configuration

Environment variables shown by `src/adaptive/config.py` and `.env.example`:

- K settings: `AE_RAG_MIN_K`, `AE_RAG_MAX_K`, `AE_RAG_DEFAULT_K`, `AE_RAG_ESCALATION_STEP`, `AE_RAG_VERIFICATION_THRESHOLD`.
- Chroma: `AE_RAG_CHROMA_PATH`, `AE_RAG_APP_CHROMA_PATH`, `AE_RAG_APP_COLLECTION`.
- Models: `AE_RAG_MODEL_PATH`, `AE_RAG_GENERATION_MODEL`, `AE_RAG_EMBEDDING_MODEL`.
- Ollama: `AE_RAG_OLLAMA_URL`.
- Storage: `AE_RAG_UPLOAD_ROOT`, `AE_RAG_HISTORY_ROOT`.
- Server: `AE_RAG_HOST`, `AE_RAG_PORT`.
- Database: `AE_RAG_DB_HOST`, `AE_RAG_DB_PORT`, `AE_RAG_DB_USER`, `AE_RAG_DB_PASSWORD`, `AE_RAG_DB_NAME`.
- Cookie security: `AE_RAG_COOKIE_SECURE`.

Ports/URLs:

- Backend default: `127.0.0.1:8000`.
- Ollama default: `http://localhost:11434`.
- Frontend dev: Vite default unless overridden externally.
- Frontend production: served by backend from `frontend/dist`.

Secrets:

- `.env` is present. SECRET PRESENT - VALUES REDACTED.
- `.env.example` contains sample/default values only.

## 26. Tests

Current test coverage includes:

- Signup/login/logout/session checks.
- Argon2 hash check.
- Role validation for `employee` and `hr`.
- HR-only upload/list authorization.
- Query modes and invalid fixed K rejection.
- Empty application corpus response.
- File-backed history persistence.
- TXT/PDF/DOCX upload indexing paths through mocks.
- Duplicate upload rejection by content hash.
- Feature extraction and fixed/adaptive baseline behavior.
- Ollama telemetry tests.
- RAGAS smoke tests.

Not covered:

- Structured table extraction.
- Table-aware answer formatting.
- Role hierarchy for Office/Teacher/Student.
- Retrieval metadata filtering.
- Document deletion.
- Document re-indexing.
- Live MySQL schema migration.
- Production server deployment behavior.

## 27. Current Problems / Risks

- Current roles are `employee` and `hr`, not `office`, `teacher`, `student`.
- Authorization is endpoint-level only; retrieval is not document-level access-controlled.
- Chroma query has no metadata filter, so future multi-role documents would leak unless filtering is added.
- Uploaded document metadata has no access/audience field.
- Chroma metadata has no access/audience field.
- PDF tables are flattened to plain text; no structured table preservation exists.
- Prompt does not instruct table output when source data is tabular.
- Stored chunks do not preserve table identity, rows, columns, captions, or bounding boxes.
- `AE_RAG_VERIFICATION_THRESHOLD` is configured but not passed into `verify_retrieval`.
- `SessionStore` is in-memory; sessions vanish on process restart.
- `users` table schema is assumed but not created or documented in SQL.
- `app_documents` schema is created inside runtime code rather than an explicit migration.
- No document deletion endpoint exists.
- No document re-indexing endpoint exists.
- No Chroma cleanup path for failed/deleted documents exists.
- Research and application runtime share `load_runtime`; accidental use of default runtime could point app to research ChromaDB.
- Frontend role types and labels are hard-coded to `employee`/`hr`.
- HR documents page is hidden client-side and protected server-side, but future role hierarchy needs a general authorization model.
- `frontend/dist` is checked into the workspace as generated build output.
- Large result CSV/JSON files and virtual environments live inside the project directory, increasing scan/build/tool noise.
- Root-level diagnostic CSV/JSON files duplicate files under `results/adaptive/diagnostics`.

## 28. Recommended Change Points

For three roles:

- `src/adaptive/auth.py`: replace `VALID_ROLES = {"employee", "hr"}` with `office`, `teacher`, `student`; update validation messages.
- `frontend/src/types.ts`: update `Role`.
- `frontend/src/components/AuthPage.tsx`: update signup role select and explanatory copy.
- `frontend/src/App.tsx`: update sidebar/profile labels and document-management access if Office should own uploads.
- `src/api/app.py`: replace `require_role(environ, "hr")` with hierarchy-aware policy functions.

For document access hierarchy:

- `src/adaptive/document_store.py`: add an access/audience column to `app_documents`, for example `document_role` or `access_level`.
- SQL/migrations: add explicit schema migration for `users`, `app_documents`, and future access columns.
- `src/adaptive/app_ingestion.py`: include access metadata in `AppChunk.metadata()`.
- `/api/hr/upload` or renamed upload endpoint: accept and validate document access level.
- `src/retrieval/chroma_retriever.py`: accept a metadata filter and pass Chroma `where`.
- `AdaptivePipeline.run` and `FixedKPipeline.run`: accept user/access context and pass filters through retrieval.
- `src/api/app.py` `/api/query`: derive allowed document scopes from current user role and pass them to pipeline.

For table-aware ingestion/questions:

- `src/adaptive/app_ingestion.py`: add table extraction for PDFs before/alongside text extraction.
- Store structured table chunks with metadata such as `content_type=table`, `table_id`, `page`, `caption`, columns, rows, and serialized markdown.
- `src/adaptive/prompt_orchestrator.py`: preserve table chunks in markdown table form in context.
- `src/adaptive/feature_extractor.py` or a new query-intent helper: detect table requests.
- `src/adaptive/prompt_orchestrator.py`: add conditional instruction to answer in a table when the question asks for a table and retrieved evidence contains table chunks.
- `frontend/src/components/ChatPage.tsx`: `react-markdown` can render markdown tables only with appropriate remark/rehype support if GFM is added; otherwise table markdown may not render as HTML tables.

For 50 college circulars:

- Keep application circulars under `data/uploads/documents` through the app upload pipeline or create a controlled bulk ingestion command that writes the same MySQL metadata, chunks, and application Chroma records.
- Add document metadata fields for circular type, department/source, date, audience/access level, and circular number.
- Add tests that verify role-scoped retrieval across Office, Teacher, and Student users.

## WHAT I NEED TO MODIFY FOR THE NEW REQUIREMENTS

1. Replace the current two-role auth model with `office`, `teacher`, and `student` in backend validation, frontend types, signup UI, labels, and tests.
2. Add document access metadata with the hierarchy: Office documents only Office; Teacher documents Office + Teacher; Student documents Office + Teacher + Student.
3. Add retrieval-time Chroma metadata filtering based on the current user role.
4. Add upload-time document audience/access selection and store it in MySQL, chunk JSON, and Chroma metadata.
5. Add or migrate database schema for the new role/access fields.
6. Implement table-aware PDF extraction because current code only extracts flat text via PyMuPDF `page.get_text("text")`.
7. Store table chunks as structured/markdown table evidence, not only flattened text.
8. Update prompt construction and frontend markdown rendering so table-request answers can be returned as visible tables.
9. Add ingestion/testing path for at least 50 college circulars, preferably through the same application corpus path rather than the research corpus.
10. Add tests for role hierarchy, role-filtered retrieval, table extraction, table answer formatting, and bulk circular ingestion.
