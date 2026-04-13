from __future__ import annotations

from typing import Any, Dict, Optional, TypedDict

from langgraph.graph import END, StateGraph


class QueryGraphState(TypedDict, total=False):
    query: str
    ready_documents: list[dict]
    tree_cache: dict
    session_memory: dict
    query_documents: list[dict]
    route_info: dict
    trace: Any
    stage_callback: Any
    trace_handle: Any
    langfuse_session_id: str


class LangGraphQueryOrchestrator:
    def __init__(self, *, retrieval_service: Any, pipeline: Any) -> None:
        self.retrieval_service = retrieval_service
        self.pipeline = pipeline
        graph = StateGraph(QueryGraphState)
        graph.add_node("route_documents", self._route_documents)
        graph.add_node("run_query", self._run_query)
        graph.set_entry_point("route_documents")
        graph.add_edge("route_documents", "run_query")
        graph.add_edge("run_query", END)
        self.graph = graph.compile()

    def invoke(
        self,
        *,
        query: str,
        ready_documents: list[dict],
        tree_cache: dict,
        session_memory: Optional[dict],
        stage_callback: Any,
        trace_handle: Any = None,
        langfuse_session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.graph.invoke(
            {
                "query": query,
                "ready_documents": ready_documents,
                "tree_cache": tree_cache,
                "session_memory": session_memory or {},
                "stage_callback": stage_callback,
                "trace_handle": trace_handle,
                "langfuse_session_id": langfuse_session_id,
            }
        )

    def _route_documents(self, state: QueryGraphState) -> QueryGraphState:
        query_documents, route_info = self.retrieval_service.prepare_query_documents(
            query=state.get("query", ""),
            documents=state.get("ready_documents", []) or [],
            tree_cache=state.get("tree_cache", {}) or {},
            session_memory=state.get("session_memory", {}) or {},
            trace_handle=state.get("trace_handle"),
        )
        return {
            "query_documents": query_documents,
            "route_info": route_info,
        }

    def _run_query(self, state: QueryGraphState) -> QueryGraphState:
        query_documents = state.get("query_documents", []) or []
        if not query_documents:
            return {"trace": None}
        trace = self.pipeline.run_query_multi(
            query=state.get("query", ""),
            documents=query_documents,
            session_memory=state.get("session_memory", {}) or {},
            langfuse_session_id=state.get("langfuse_session_id"),
            stage_callback=state.get("stage_callback"),
            parent_trace_handle=state.get("trace_handle"),
            route_info=state.get("route_info", {}) or {},
        )
        trace.routing = state.get("route_info", {}) or {}
        return {"trace": trace}
