# Streamlit RAG App

Production-focused Streamlit app for PageIndex backend with notebook-parity RAG workflow:

1. Upload document (PDF, JSON, XLSX, or CSV) (`/doc/`)
2. View simplified tree (`/doc/{doc_id}/?type=tree&summary=true`)
3. Ask question in chat
4. Show:
   - Tree search reasoning
   - Retrieved nodes
   - Get page content panel (`/doc/{doc_id}/pages/`)
   - Final answer with inline citations and cited page content

## Configuration Guide

For a complete explanation of all adjustable config parameters, defaults, and tuning recommendations, see `streamlit_app/CONFIG_TUNING_GUIDE.md`.

## Environment

Required:

- `PAGEINDEX_BACKEND_URL` (default: `http://localhost:8000`)
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION` (default: `2024-02-01`)
- `AZURE_OPENAI_DEPLOYMENT_NAME` (default: `gpt-4o-mini`)

Optional tuning:

- `MAX_TREE_NODES_FOR_PROMPT`
- `MAX_TREE_SUMMARY_CHARS`
- `MAX_PAGE_PREVIEW_CHARS`
- `MAX_PAGE_EVIDENCE_CHARS`
- `REQUEST_TIMEOUT_SEC`
- `MAX_RETRIES`
- `RETRY_BACKOFF_SEC`
- `RETRIEVAL_TOP_K`
- `HIERARCHICAL_CHILD_CHUNK_WORDS`
- `HIERARCHICAL_CHILD_CHUNK_OVERLAP_WORDS`
- `HIERARCHICAL_CANDIDATE_POOL_SIZE`
- `HIERARCHICAL_TOP_CHILD_CHUNKS`
- `HIERARCHICAL_NEIGHBOR_CHUNKS`
- `ANSWER_MAX_TOKENS`
- `MAX_UPLOAD_SIZE_MB`
- `PREPARED_DOC_CACHE_SIZE`
- `PREPARED_DOC_CACHE_TTL_SEC`
- `METADATA_CACHE_SIZE`
- `METADATA_CACHE_TTL_SEC`
- `SESSION_MEMORY_TURN_WINDOW`
- `SESSION_MEMORY_CHAR_BUDGET`
- `DOCUMENTS_PAGE_SIZE`
- `MAX_WORKSPACE_DOCUMENTS`
- `TREE_PREVIEW_DOC_LIMIT`
- `ENABLE_DOCUMENT_ROUTING`
- `DOCUMENT_ROUTING_TOP_K`
- `DOCUMENT_ROUTING_MIN_DOCS`
- `PGVECTOR_ENABLED`
- `PGVECTOR_DSN`
- `PGVECTOR_DOCUMENT_TABLE`
- `PGVECTOR_SECTION_TABLE`
- `PGVECTOR_EMBEDDING_DIMENSIONS`
- `ENABLE_SECTION_RERANKING`
- `SECTION_RERANK_TOP_K`
- `SECTION_RERANK_NODE_HINTS`
- `SECTION_RERANK_SCORE_BONUS`
- `AZURE_OPENAI_EMBEDDING_API_KEY`
- `AZURE_OPENAI_EMBEDDING_ENDPOINT`
- `AZURE_OPENAI_EMBEDDING_API_VERSION`
- `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT`
- `ENABLE_LANGGRAPH`
- `LANGFUSE_ENABLED`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_HOST`

## Run

Start backend:

```bash
uvicorn pageindex_backend.main:app --host 0.0.0.0 --port 8000
```

Start Streamlit app:

```bash
streamlit run streamlit_app/app.py
```

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

