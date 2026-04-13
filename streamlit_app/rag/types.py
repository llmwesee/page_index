"""
Data types for the PageIndex legal RAG pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TreeSearchResult:
    thinking: str
    node_list: List[str]
    query_decomposition: List[str] = field(default_factory=list)
    raw: str = ""


@dataclass
class RetrievedNode:
    node_id: str
    title: str
    page: Optional[int]
    doc_id: Optional[str] = None
    doc_name: Optional[str] = None
    doc_ref: Optional[str] = None
    section_path: Optional[str] = None
    score: Optional[float] = None


@dataclass
class PageContentResult:
    requested_pages: str
    returned_pages: str
    content: List[Dict[str, Any]]
    raw: Dict[str, Any]


@dataclass
class AnswerResult:
    thinking: str              # Structured markdown evidence chain
    answer_markdown: str       # Rich markdown answer for display
    cited_pages: List[int]
    cited_sources: List[Dict[str, Any]] = field(default_factory=list)
    confidence: str = "MEDIUM"        # HIGH | MEDIUM | LOW
    answer_type: str = "DIRECT"       # DIRECT | PARTIAL | INSUFFICIENT
    raw: str = ""


@dataclass
class QueryTrace:
    trace_id: str
    query: str
    doc_id: str
    doc_name: str
    selected_doc_ids: List[str] = field(default_factory=list)
    selected_doc_names: List[str] = field(default_factory=list)
    tree_search: Optional[TreeSearchResult] = None
    retrieved_nodes: List[RetrievedNode] = field(default_factory=list)
    page_result: Optional[PageContentResult] = None
    answer_result: Optional[AnswerResult] = None
    timings: Dict[str, float] = field(default_factory=dict)
    token_usage: Dict[str, int] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    routing: Dict[str, Any] = field(default_factory=dict)
    cache_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
