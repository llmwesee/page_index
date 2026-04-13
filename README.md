# PageIndex 

Lightweight document indexing and RAG prototype using a backend indexer and a Streamlit UI.

## Overview

This repository contains a small system for indexing documents, caching index artifacts, and serving a Streamlit UI and a backend API for retrieval/QA workflows.

Key components:
- `pageindex/` — core indexing/parsing utilities and configuration.
- `pageindex_backend/` — backend service, caching, and main runner.
- `streamlit_app/` — Streamlit front-end app and RAG pipeline components.

Supported source formats:
- PDF documents
- JSON documents (Azure Document Intelligence layout JSON and generic page-text JSON)
- XLSX spreadsheets
- CSV files
- Web URLs with trafilatura extraction
- URL discovery via sitemaps, feeds, and focused crawling

## Features

- Document parsing and indexing
- Persistent cache of index artifacts (`pageindex_backend/cache/`)
- Streamlit UI for querying and exploring indexed docs
- Example pipeline and prompts (under `streamlit_app/rag/`)

## Prerequisites

- Python 3.10+ (recommended)
- Azure OpenAI credentials
- Docker Desktop with PostgreSQL container running

## Quick Start (Windows)

### Option 1: Setup Scripts (Recommended)

```batch
REM Step 1: Run setup
setup.bat

REM Step 2: Start application
start.bat

REM Step 3: Access at http://localhost:8501
```

### Option 2: Docker Compose

```batch
REM Create .env with your Azure credentials
copy .env_example .env

REM Start backend and frontend (connects to your existing PostgreSQL)
docker-compose up -d

REM Access at http://localhost:8501
```

**See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed Windows deployment guide.**

---

## What the Setup Does

The `setup.bat` script will:
✅ Check Python 3.10+ is installed  
✅ Create virtual environment (`.venv`)  
✅ Install all dependencies from `requirements.txt`  
✅ Prompt for Azure OpenAI credentials  
✅ Create `.env` file with configuration  
✅ Configure connection to your Docker PostgreSQL  
✅ Create necessary directories (data, cache, logs)

---

## Configuration

After setup, edit `.env` if needed:

```bash
# Azure OpenAI (Required)
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.cognitiveservices.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1

# PostgreSQL (Your Docker container)
PGVECTOR_DSN=postgresql://postgres:admin123@localhost:5432/vectordb
```

See configuration details:
- `pageindex/config.yaml` - Backend indexing and parsing configuration
- `streamlit_app/config.py` - Frontend and service settings
- `streamlit_app/CONFIG_TUNING_GUIDE.md` - Complete parameter tuning guide

---

## Running Services

**Start both services:**
```batch
start.bat
```

**Stop both services:**
```batch
stop.bat
```

**Check status:**
```batch
netstat -ano | findstr ":8000"  # Backend
netstat -ano | findstr ":8501"  # Frontend
docker ps                        # PostgreSQL
```

---

## Accessing the Application

- **Frontend UI**: http://localhost:8501  
- **Backend API**: http://localhost:8000  
- **API Documentation**: http://localhost:8000/docs

---

## Background Processing

- Document upload queues tree generation in background workers
- Check processing status: `GET /processing/stats/`
- View logs: `logs\backend.log` and `logs\frontend.log`

---

## Troubleshooting

**Backend won't start:**
- Check port 8000 is free: `netstat -ano | findstr :8000`
- Verify `.env` has correct Azure credentials
- Check PostgreSQL is running: `docker ps`

**Frontend won't start:**
- Check port 8501 is free: `netstat -ano | findstr :8501`
- Ensure backend is running first

**See [DEPLOYMENT.md](DEPLOYMENT.md) for comprehensive troubleshooting.**

---

## Development Notes

- Backend code changes require restart (`stop.bat` then `start.bat`)
- Frontend (Streamlit) auto-reloads on code changes
- Inspect `pageindex/page_index.py` for parsing/indexing logic
- `streamlit_app/rag/` contains the retrieval pipeline and prompt templates

---

## Documentation

- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Complete Windows deployment guide  
- **[ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md)** - System architecture  
- **[COPILOT_INSTRUCTIONS.md](COPILOT_INSTRUCTIONS.md)** - Development guide  
- **[streamlit_app/CONFIG_TUNING_GUIDE.md](streamlit_app/CONFIG_TUNING_GUIDE.md)** - Parameter tuning  

---

## Files You Can Delete (Linux/Mac only)

If you're only using Windows, you can delete:
- `setup.sh`
- `start.sh`
- `stop.sh`

---

## License

Add a license file if you intend to publish this project.


Run the backend service (example):

```bash
cd pageindex_backend
python main.py
```

Run the Streamlit UI:

```bash
cd streamlit_app
streamlit run app.py
```

Run tests:

```bash
pytest -q
```

## Configuration

- `pageindex/config.yaml` contains indexing and parsing configuration.
- `streamlit_app/config.py` contains front-end and service settings.

Important backend throughput knobs in `pageindex/config.yaml`:
- `tree_worker_count` controls how many documents can be processed in parallel.
- `llm_max_concurrency` caps concurrent LLM calls across all workers so throughput increases do not overwhelm the model provider.
- `pdf_page_worker_count` parallelizes page extraction inside a single document for OCR, fallback tree parsing, and page-token preparation.

## Data & Cache

- Place source documents under `data/` or feed them to the indexing utilities in `pageindex/`.
- Accepted inputs now include PDF, JSON, XLSX, and CSV document payloads.
- Generated index artifacts and summaries are stored in `pageindex_backend/cache/`.

## Background Processing

- Document upload now queues tree generation in background workers instead of blocking the upload request.
- Use `GET /processing/stats/` to inspect queue depth, document status counts, and average stage timings.
- Individual document metadata now includes a `timings` object with queue wait, OCR extraction, LLM tree generation, fallback tree generation, and total processing time.

## Benchmarking

Run the backend first, then benchmark a directory of PDFs:

```bash
python benchmark_tree_generation.py path/to/pdf_dir --backend-url http://localhost:8000 --recursive
```

Optional useful flags:

```bash
python benchmark_tree_generation.py path/to/pdf_dir --limit 100 --poll-interval 3 --output-json benchmark.json
```

This benchmark prints:
- end-to-end completion times
- average queue wait time
- average total processing time
- average LLM tree generation time
- final backend queue and worker stats

## Development notes

- Inspect `pageindex/page_index.py` and `pageindex/page_index_md.py` for parsing/indexing logic.
- `streamlit_app/rag/` contains the retrieval pipeline and prompt templates.
- `pageindex_client/client.py` shows example client usage against the backend API.

## Contributing

Contributions and bug reports welcome. Open an issue or submit a PR with a clear description and tests where applicable.

## License

Add a license file if you intend to publish this project.

---

If you'd like, I can add badges, expand any section (architecture diagram, env setup), or create a minimal `Makefile`/`dev` scripts next.
