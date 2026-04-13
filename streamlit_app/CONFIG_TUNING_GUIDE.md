# Streamlit App Configuration Tuning Guide

This guide explains all adjustable parameters in `streamlit_app/config.py` and how to tune them safely for different goals: quality, latency, and cost.

## How Config Resolution Works

- Environment variables are loaded from OS env first.
- If missing, values are read from `.env` in project root.
- If still missing, hardcoded defaults from `load_config()` are used.

## Quick Start Profiles

Use these as a starting point, then fine tune.

## `Fast / Low Cost`

- `RETRIEVAL_TOP_K=4`
- `HIERARCHICAL_TOP_CHILD_CHUNKS=6`
- `HIERARCHICAL_CANDIDATE_POOL_SIZE=24`
- `MAX_PAGE_EVIDENCE_CHARS=2500`
- `ANSWER_MAX_TOKENS=700`

Best for quick previews and high request volume.

## `Balanced` (close to current defaults)

- `RETRIEVAL_TOP_K=6`
- `HIERARCHICAL_TOP_CHILD_CHUNKS=10`
- `HIERARCHICAL_CANDIDATE_POOL_SIZE=36`
- `MAX_PAGE_EVIDENCE_CHARS=5000`
- `ANSWER_MAX_TOKENS=1200`

Best for most production workloads.

## `High Recall / Hard Legal Queries`

- `RETRIEVAL_TOP_K=8-10`
- `HIERARCHICAL_TOP_CHILD_CHUNKS=12-16`
- `HIERARCHICAL_CANDIDATE_POOL_SIZE=50-80`
- `MAX_PAGE_EVIDENCE_CHARS=7000-12000`
- `ANSWER_MAX_TOKENS=1400-2200`

Best for complex multi-clause questions, at higher latency and token cost.

## Parameter Reference

Each parameter below includes default, effect, and tuning guidance.

### Core Backend + LLM

| Env var | Default | What it controls | Increase when | Decrease when |
|---|---:|---|---|---|
| `PAGEINDEX_BACKEND_URL` | `http://localhost:8000` | PageIndex backend API endpoint | Backend moved to remote host | N/A |
| `AZURE_OPENAI_API_KEY` | none | Azure OpenAI auth | Required in all non-local mocks | N/A |
| `AZURE_OPENAI_ENDPOINT` | none | Azure OpenAI endpoint URL | Required in all non-local mocks | N/A |
| `AZURE_OPENAI_API_VERSION` | `2024-12-01-preview` | Azure API contract version | Need newer API capabilities | Stability issues appear |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | `gpt-4.1` | Chat model deployment | Need higher quality reasoning | Need lower cost/latency |

### Prompt / Answer Size Controls

| Env var | Default | What it controls | Increase when | Decrease when |
|---|---:|---|---|---|
| `MAX_TREE_NODES_FOR_PROMPT` | `240` | Max indexed sections considered in retrieval outline | Missing relevant sections | Prompt too large / slower |
| `MAX_TREE_SUMMARY_CHARS` | `1064` | Max chars kept per node summary | Summaries are too vague | Prompt bloat |
| `MAX_PAGE_PREVIEW_CHARS` | `532` | UI preview snippet length | Reviewers need more context in UI | UI feels noisy |
| `MAX_PAGE_EVIDENCE_CHARS` | `5000` | Evidence size sent into answer prompt | Answers miss key details | Token cost too high |
| `ANSWER_MAX_TOKENS` | `1200` | Max output tokens for final answer | Answers are getting truncated | Need concise responses/cost control |

### Reliability / Retry Controls

| Env var | Default | What it controls | Increase when | Decrease when |
|---|---:|---|---|---|
| `REQUEST_TIMEOUT_SEC` | `45` | HTTP/LLM timeout budget | Calls time out on large documents | Fail fast is preferred |
| `MAX_RETRIES` | `3` | Retry attempts for transient failures | Frequent temporary API/network issues | Need strict latency |
| `RETRY_BACKOFF_SEC` | `0.8` | Base backoff between retries | APIs are rate limited often | Retries are too slow |

### Hierarchical Retrieval Controls

| Env var | Default | What it controls | Increase when | Decrease when |
|---|---:|---|---|---|
| `RETRIEVAL_TOP_K` | `6` | Number of top parent sections selected | Query spans many clauses | Need lower latency |
| `HIERARCHICAL_CHILD_CHUNK_WORDS` | `180` | Child chunk size in words | Need broader local context | Need finer-grained matching |
| `HIERARCHICAL_CHILD_CHUNK_OVERLAP_WORDS` | `45` | Overlap between adjacent chunks | Boundary misses occur | Duplicate context is too high |
| `HIERARCHICAL_CANDIDATE_POOL_SIZE` | `36` | Initial scored chunk pool size | Relevant text is often missed | Retrieval too slow |
| `HIERARCHICAL_TOP_CHILD_CHUNKS` | `10` | Chunks promoted to evidence | Need stronger recall | Need less prompt size |
| `HIERARCHICAL_NEIGHBOR_CHUNKS` | `1` | Adjacent chunks pulled around winners | Explanations need nearby context | Redundant context appears |

### Caching / Throughput

| Env var | Default | What it controls | Increase when | Decrease when |
|---|---:|---|---|---|
| `PREPARED_DOC_CACHE_SIZE` | `128` | Number of prepared document hierarchies cached | Many repeated docs/sessions | Memory pressure |
| `PREPARED_DOC_CACHE_TTL_SEC` | `3600` | Prepared hierarchy cache TTL (seconds) | Documents are mostly static | Documents change often |
| `METADATA_CACHE_SIZE` | `512` | Cached metadata entries | Repeated metadata requests | Memory pressure |
| `METADATA_CACHE_TTL_SEC` | `1800` | Metadata cache TTL (seconds) | Metadata is stable | Metadata freshness needed |

### Session Memory (Follow-up Questions)

| Env var | Default | What it controls | Increase when | Decrease when |
|---|---:|---|---|---|
| `SESSION_MEMORY_TURN_WINDOW` | `6` | Number of previous turns considered | Follow-up references are ambiguous | Older context hurts precision |
| `SESSION_MEMORY_CHAR_BUDGET` | `900` | Max chars of compressed chat memory | Follow-up grounding is weak | Token budget is tight |

### Workspace / UI Scope

| Env var | Default | What it controls | Increase when | Decrease when |
|---|---:|---|---|---|
| `MAX_UPLOAD_SIZE_MB` | `500` | Max upload file size | Need large-file ingestion | Need stricter limits |
| `DOCUMENTS_PAGE_SIZE` | `250` | Batch size for listing docs from backend | Many docs and list calls are slow | Responses are too large |
| `MAX_WORKSPACE_DOCUMENTS` | `5000` | Cap on docs loaded in UI | Very large repositories | Keep UI responsive |
| `TREE_PREVIEW_DOC_LIMIT` | `6` | Max trees previewed in sidebar/UX areas | Need broader preview | UI clutter |

### Routing + PgVector Controls

| Env var | Default | What it controls | Increase when | Decrease when |
|---|---:|---|---|---|
| `ENABLE_DOCUMENT_ROUTING` | `true` | Enables pre-filtering documents before retrieval | Workspace has many docs | Small workspace |
| `DOCUMENT_ROUTING_TOP_K` | `8` | Number of docs kept by routing | Query touches many docs | Latency/cost too high |
| `DOCUMENT_ROUTING_MIN_DOCS` | `3` | Routing activates only above this doc count | Want routing always on | Want simple behavior |
| `PGVECTOR_ENABLED` | `true` if DSN exists | Enables vector retrieval path | Need semantic matching quality | Local/dev without pgvector |
| `PGVECTOR_DSN` | none | Postgres DSN for vector search | Enable semantic retrieval | Disable vector retrieval |
| `PGVECTOR_DOCUMENT_TABLE` | `document_index` | Doc-level vector index table | Custom schema names | N/A |
| `PGVECTOR_SECTION_TABLE` | `section_index` | Section-level vector index table | Custom schema names | N/A |
| `PGVECTOR_EMBEDDING_DIMENSIONS` | `3072` | Expected embedding vector size | Use a different embedding model | Never decrease unless model changes |

### Section Re-ranking Controls

| Env var | Default | What it controls | Increase when | Decrease when |
|---|---:|---|---|---|
| `ENABLE_SECTION_RERANKING` | `true` | Enables pgvector section hinting in retrieval | Need better section precision | No pgvector / simpler pipeline |
| `SECTION_RERANK_TOP_K` | `24` | Vector section hits fetched | Hints are too narrow | Latency too high |
| `SECTION_RERANK_NODE_HINTS` | `4` | Max hinted nodes per document | Retrieval misses target clause | Too many hints create noise |
| `SECTION_RERANK_SCORE_BONUS` | `2.25` | Score boost for hinted sections | Hints are ignored too often | Hints are over-dominating |

### Embedding Client Controls

| Env var | Default / Fallback | What it controls | Notes |
|---|---|---|---|
| `AZURE_OPENAI_EMBEDDING_API_KEY` | Falls back to `AZURE_OPENAI_KEY`, then `AZURE_OPENAI_API_KEY` | Embedding auth | Keep explicit if using separate key |
| `AZURE_OPENAI_EMBEDDING_ENDPOINT` | Falls back to `AZURE_OPENAI_ENDPOINT` | Embedding endpoint | Useful when chat + embeddings are split |
| `AZURE_OPENAI_EMBEDDING_API_VERSION` | Falls back to `AZURE_OPENAI_API_VERSION` | Embedding API version | Keep aligned with deployment support |
| `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT` / `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` | none | Embedding model deployment name | Required for pgvector semantic retrieval |

### Orchestration + Observability

| Env var | Default | What it controls | Increase / enable when | Disable when |
|---|---:|---|---|---|
| `ENABLE_LANGGRAPH` | `true` | Uses LangGraph orchestration path | Need explicit graph orchestration | Debugging simpler linear flow |
| `LANGFUSE_ENABLED` | `true` if keys exist | Enables Langfuse telemetry | Need trace/session/token observability | Local-only development |
| `LANGFUSE_PUBLIC_KEY` | none | Langfuse auth | Required for tracing | N/A |
| `LANGFUSE_SECRET_KEY` | none | Langfuse auth | Required for tracing | N/A |
| `LANGFUSE_HOST` / `LANGFUSE_BASE_URL` | none | Langfuse host override | Self-hosted Langfuse | Cloud default endpoint |

## Practical Tuning Playbook

### If answers are incomplete or miss key clauses

- Increase `RETRIEVAL_TOP_K` by +2.
- Increase `HIERARCHICAL_TOP_CHILD_CHUNKS` by +2 to +4.
- Increase `MAX_PAGE_EVIDENCE_CHARS` by +1500 to +3000.
- If pgvector is enabled, increase `SECTION_RERANK_TOP_K` modestly.

### If latency is too high

- Reduce `DOCUMENT_ROUTING_TOP_K` and `RETRIEVAL_TOP_K`.
- Reduce `HIERARCHICAL_CANDIDATE_POOL_SIZE`.
- Reduce `MAX_PAGE_EVIDENCE_CHARS`.
- Reduce `ANSWER_MAX_TOKENS`.

### If token cost is too high

- Reduce `MAX_PAGE_EVIDENCE_CHARS` first.
- Reduce `HIERARCHICAL_TOP_CHILD_CHUNKS`.
- Reduce `ANSWER_MAX_TOKENS`.
- Keep `SESSION_MEMORY_CHAR_BUDGET` conservative.

### If follow-up questions are misinterpreted

- Increase `SESSION_MEMORY_TURN_WINDOW` by +1 to +3.
- Increase `SESSION_MEMORY_CHAR_BUDGET` by +200 to +500.
- Keep an eye on prompt token growth in Langfuse.

## Safe Change Process

1. Change one parameter group at a time.
2. Run 20-50 representative queries.
3. Track in Langfuse:
   - answer quality
   - latency
   - input/output token usage
4. Keep a changelog of env values and outcomes.

## Minimal `.env` Template

```env
PAGEINDEX_BACKEND_URL=http://localhost:8000

AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1

RETRIEVAL_TOP_K=6
HIERARCHICAL_TOP_CHILD_CHUNKS=10
MAX_PAGE_EVIDENCE_CHARS=5000
ANSWER_MAX_TOKENS=1200

LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
```
