# PageIndex Architecture

This document presents the system at two levels:

- A high-level business architecture diagram for product, delivery, and leadership discussions.
- A low-level technical architecture diagram for engineering, platform, and support teams.

The current architecture reflects the implemented stack in this repository as of March 2026: Streamlit UI, FastAPI backend, Azure OpenAI for generation plus embeddings, pgvector for retrieval indexing and reranking, LangGraph for orchestration, Langfuse for tracing, and in-process caching plus session memory inside the app.

## 1. High-Level Architecture

### Purpose

Use this view when the audience cares about business capabilities, system responsibilities, and platform boundaries rather than function names or code paths.

```mermaid
flowchart LR
    user[Business User / Analyst / Legal Reviewer]

    subgraph experience[Experience Layer]
        ui[PageIndex Assistant UI\nStreamlit Web App]
    end

    subgraph intelligence[Intelligence Layer]
        orchestration[Query Orchestration\nLangGraph + Retrieval Logic]
        memory[Conversation Memory\nShort-Term Session Context]
        answering[Grounded Answer Generation\nAzure OpenAI]
    end

    subgraph knowledge[Knowledge and Retrieval Layer]
        routing[Document Routing\npgvector Document Index]
        sectioning[Section Narrowing\nPageIndex Hierarchy Search]
        rerank[Section / Snippet Reranking\npgvector Section Index]
        evidence[Evidence Assembly\nPage-Cited Snippets]
    end

    subgraph content[Content Processing Layer]
        upload[PDF Upload and Registry]
        processing[OCR + Tree Generation]
        indexing[Embedding and Vector Indexing]
    end

    subgraph storage[Storage and Observability Layer]
        files[Document and Tree Artifacts\nBackend Data + Cache]
        vectors[PostgreSQL + pgvector]
        telemetry[Tracing and Metrics\nLangfuse]
    end

    user --> ui
    ui --> orchestration
    orchestration --> memory
    orchestration --> routing
    routing --> sectioning
    sectioning --> rerank
    rerank --> evidence
    evidence --> answering
    answering --> ui

    ui --> upload
    upload --> processing
    processing --> indexing

    processing --> files
    indexing --> vectors
    orchestration --> telemetry
    answering --> telemetry
    routing --> telemetry
```

### Business Interpretation

- Users interact with a single assistant experience rather than multiple tools.
- Uploaded PDFs are processed into structured knowledge assets, not just stored as raw files.
- Query handling is staged: route relevant documents first, narrow to sections next, then answer from cited evidence.
- The model is not expected to search the whole corpus directly; retrieval reduces scope before generation.
- Tracing and observability are part of the product operating model, not an afterthought.

### Business Capability Map

| Capability | Business Value | Primary Implementation |
|---|---|---|
| Document onboarding | New PDFs become searchable and answerable | FastAPI backend + background workers |
| Structured retrieval | Better accuracy than naive full-text search | PageIndex tree + chunk hierarchy |
| Scalable corpus routing | Keeps large corpora queryable | pgvector document index |
| Follow-up handling | Supports conversational use cases | Session memory in Streamlit app |
| Grounded answers | Reduces hallucination risk | Evidence assembly + cited answer prompt |
| Operational visibility | Enables debugging, tuning, and governance | Langfuse traces and metrics |

## 2. Low-Level Technical Architecture

### Purpose

Use this view when the audience needs concrete data flow, component interactions, and deployment/runtime boundaries.

```mermaid
flowchart TD
    user[User]
    ui[streamlit_app/app.py\nStreamlit UI]
    config[streamlit_app/config.py\nRuntime Config]

    subgraph app_runtime[Application Runtime]
        memory[rag/memory.py\nSession Memory Builder]
        retrieval[services/retrieval.py\nDocument Routing + Section Hints]
        graph[rag/langgraph_orchestrator.py\nLangGraph Query Graph]
        pipeline[rag/pipeline.py\nHierarchical Retrieval + Answer Path]
        prompts[rag/prompts.py\nPrompt Builders]
        observer[services/observer.py\nLangfuse Adapter]
        apiClient[services/pageindex_api.py\nBackend API Client]
        llm[services/azure_llm.py\nAzure Chat Client]
        cache[cache.py\nLRU + TTL Caches]
        embedUtil[embedding_utils.py\nAzure Embedding Client]
    end

    subgraph backend_runtime[Backend Runtime]
        fastapi[pageindex_backend/main.py\nFastAPI + Workers]
        worker[Background Worker Queue]
        tree[Tree Generation\nLLM Tree + Fallback Tree]
        indexer[pageindex_backend/retrieval_index.py\npgvector Indexer]
    end

    subgraph persistence[Persistence]
        data[(pageindex_backend/data\ndocuments.json + PDFs)]
        artifactCache[(pageindex_backend/cache\nTree Cache Files)]
        pg[(PostgreSQL + pgvector\ndocument_index + section_index)]
        lf[(Langfuse Cloud)]
    end

    user --> ui
    ui --> config
    ui --> memory
    ui --> graph
    graph --> retrieval
    graph --> pipeline
    retrieval --> embedUtil
    retrieval --> apiClient
    retrieval --> pg
    pipeline --> apiClient
    pipeline --> prompts
    pipeline --> llm
    pipeline --> cache
    pipeline --> observer
    observer --> lf

    ui --> apiClient
    apiClient --> fastapi

    fastapi --> worker
    worker --> tree
    tree --> artifactCache
    tree --> data
    worker --> indexer
    indexer --> embedUtil
    indexer --> pg
    fastapi --> data
```

### Technical Interpretation

- The Streamlit app owns query orchestration, session memory, answer generation, and trace rendering.
- The FastAPI backend owns document persistence, background processing, OCR, tree creation, and vector indexing.
- pgvector is split into two retrieval roles:
  - `document_index` for document-level routing.
  - `section_index` for section-level hints and reranking.
- LangGraph currently orchestrates the query path but still delegates actual retrieval and answering to the existing retrieval service and RAG pipeline.
- Langfuse is attached through an observer adapter, which lets the app keep running if tracing is disabled.

## 3. Runtime Query Flow

This is the practical query path used by the current implementation.

```mermaid
sequenceDiagram
    participant U as User
    participant S as Streamlit App
    participant M as Session Memory
    participant G as LangGraph
    participant R as Retrieval Service
    participant P as pgvector
    participant B as Backend API
    participant H as Hierarchical Pipeline
    participant A as Azure OpenAI
    participant L as Langfuse

    U->>S: Ask question
    S->>M: Build compact conversation memory
    S->>G: Start query orchestration
    G->>R: Prepare query documents
    R->>P: Route docs and fetch section hints
    R->>B: Load trees for routed docs
    R-->>G: Routed docs + preferred node hints
    G->>H: Run hierarchical retrieval
    H->>B: Fetch metadata / OCR pages
    H->>H: Build sections, chunks, scores, evidence
    H->>A: Generate grounded answer
    A-->>H: Final answer text
    H->>L: Emit trace, events, generation metadata
    H-->>S: Trace + answer + citations
    S-->>U: Render answer and retrieval trace
```

## 4. Runtime Ingestion Flow

This is the path that converts a PDF into retrieval-ready assets.

```mermaid
flowchart TD
    upload[Upload PDF] --> submit[POST /doc/]
    submit --> save[Save PDF + registry entry]
    save --> queue[Async worker queue]
    queue --> ocr[Extract OCR pages]
    ocr --> llmTree[Generate summarized tree]
    llmTree --> fallback[Generate fallback structural tree]
    fallback --> summaries[Build retrieval summary + section records]
    summaries --> embed[Generate embeddings]
    embed --> vectorWrite[Upsert document_index + section_index]
    vectorWrite --> ready[Mark retrieval_ready = true]
    llmTree --> treeCache[Write cached tree artifacts]
    ready --> docs[(documents.json)]
    treeCache --> cache[(cache/*.json)]
```

## 5. Alignment Summary

This table is useful for stakeholder reviews where people need a quick statement of what each architecture concern maps to in the implementation.

| Concern | Current State | Key Files |
|---|---|---|
| UI and user workflow | Implemented | `streamlit_app/app.py` |
| Short-term memory | Implemented | `streamlit_app/rag/memory.py` |
| In-process caching | Implemented | `streamlit_app/cache.py`, `streamlit_app/services/pageindex_api.py`, `streamlit_app/rag/pipeline.py` |
| LangGraph orchestration | Implemented | `streamlit_app/rag/langgraph_orchestrator.py` |
| Structured section retrieval | Implemented | `streamlit_app/rag/pipeline.py` |
| pgvector document routing | Implemented, active when thresholds/config permit | `streamlit_app/services/retrieval.py` |
| pgvector section reranking | Implemented, active when section index exists | `streamlit_app/services/retrieval.py` |
| Embedding generation | Implemented | `embedding_utils.py`, `pageindex_backend/retrieval_index.py` |
| Vector indexing on ingest | Implemented | `pageindex_backend/main.py`, `pageindex_backend/retrieval_index.py` |
| Trace and metrics export | Implemented | `streamlit_app/services/observer.py` |

## 6. Audience Guidance

- Use the high-level diagram in product reviews, design sign-off meetings, and stakeholder updates.
- Use the low-level diagram in engineering handoff, operations review, and debugging discussions.
- Use the runtime query and ingestion diagrams when analyzing performance, quality, or failure points.
