# Copilot Instructions for PageIndex

## Build, test, and lint commands

### Environment setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run the system
```powershell
# Windows helper scripts (recommended)
.\start.bat
.\stop.bat

# Direct service startup
python -m uvicorn pageindex_backend.main:app --host 0.0.0.0 --port 8000
python -m streamlit run streamlit_app\app.py
```

### Docker workflow
```powershell
docker-compose up -d
docker-compose logs -f
docker-compose down
```

### Tests
```powershell
# Documented full-suite commands
pytest -q
python -m unittest discover -s tests -p "test_*.py" -v

# Single test (unittest)
python -m unittest tests.test_module.TestClass.test_method -v
```

### Lint
No project lint command/config is currently defined in this repository (no ruff/flake8/pylint/mypy config files detected).

## High-level architecture

PageIndex is a two-runtime RAG system:

1. **Backend (`pageindex_backend/main.py`, FastAPI)** handles document ingestion (PDF/JSON/XLSX/CSV + URL ingestion), async background processing, OCR extraction, tree generation, and pgvector indexing.
2. **Frontend (`streamlit_app/app.py`, Streamlit)** handles chat UX, query orchestration, hierarchical retrieval, answer generation, and trace/journey-map rendering.

Core ingestion flow:

1. Upload/create document via backend (`/doc/`, `/url/`, `/urls/discover/`).
2. Backend queues work (`DOCUMENT_QUEUE`, worker tasks), builds:
   - LLM tree (`tree_llm_summary`, `tree_llm_no_summary`)
   - fallback structural tree
   - OCR pages / evidence units
3. Backend persists registry and artifacts:
   - `pageindex_backend\data\documents.json` (document registry)
   - `pageindex_backend\cache\*.json` (tree cache keyed by file hash)
4. If pgvector is enabled, backend upserts document + section embeddings (`retrieval_index.py`) into `document_index` and `section_index`.

Core query flow:

1. Streamlit builds session memory context.
2. `RetrievalService` routes docs (pgvector first, then local heuristic fallback) and optional section hints.
3. `RAGPipeline` performs hierarchical retrieval (sections -> child chunks), assembles evidence snippets, and calls answer generation.
4. UI renders answer + citations + retrieval journey map using `QueryTrace`.

## Key conventions for this codebase

### 1) Document lifecycle and readiness
- Document records track `status` (`queued`, `processing`, `completed`, `failed`) and `retrieval_ready`.
- Frontend readiness checks are intentionally permissive (`status == "completed"` OR `retrieval_ready == true`), so preserve both fields when changing backend status handling.

### 2) Registry + cache coupling
- Do not treat in-memory `DOCUMENTS` as the source of truth alone; persist via `_save_registry()` to `documents.json`.
- Cache regeneration behavior depends on `file_hash` and missing-cache requeue logic (`_maybe_requeue_missing_cache`).

### 3) Retrieval trace contract is UI-critical
- `streamlit_app/ui/journey_map.py` expects `trace.page_result.raw["selected_parent_nodes"]` as:
  - `Dict[doc_ref, List[Dict]]` (not string IDs),
  - each dict containing fields like `node_id`, `title`, `path`, `start_page`, `end_page`, `score`, `pages`.
- Keep `QueryTrace`, `PageContentResult.raw`, and citation metadata fields stable when modifying retrieval output.

### 4) Citation/evidence formatting conventions
- Evidence snippets use structured IDs like `citation_id = "{doc_ref}#p{page}:{type}:{element_id}"`.
- Answer prompts require inline citations in `<doc=DOC_REF;page=N;type=TYPE;element=ELEMENT_ID>` format; retrieval output must carry enough metadata (`doc_ref`, `page`, `content_type`, `element_id`) to support this.

### 5) Config loading behavior
- Both backend and frontend call `load_dotenv(..., override=True)` and then read environment variables, so `.env` values are loaded into process env at startup.
- Backend indexing/tuning config is split across:
  - `pageindex\config.yaml` (tree/index worker and parsing settings)
  - environment variables read in `pageindex_backend/main.py`.
- Frontend runtime settings are in `streamlit_app/config.py`.

### 6) pgvector is optional but first-class
- Retrieval/indexing should degrade gracefully when pgvector settings/embedding client are unavailable.
- Existing pattern: attempt pgvector route/rerank first, then fallback to local scoring without failing the request.

