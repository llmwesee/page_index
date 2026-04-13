from __future__ import annotations

from typing import Any, Dict, Optional


class BaseObserver:
    def start_trace(
        self,
        name: str,
        *,
        trace_id: Optional[str] = None,
        parent_observation_id: Optional[str] = None,
        session_id: Optional[str] = None,
        input_payload: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        return None

    def event(
        self,
        trace_handle: Optional[Any],
        name: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        return None

    def generation(
        self,
        trace_handle: Optional[Any],
        name: str,
        *,
        model: str,
        input_text: str,
        output_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        usage: Optional[Dict[str, int]] = None,
    ) -> None:
        return None

    def end_trace(
        self,
        trace_handle: Optional[Any],
        *,
        output_payload: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        return None


class NullObserver(BaseObserver):
    pass


class LangfuseObserver(BaseObserver):
    def __init__(
        self,
        *,
        public_key: str,
        secret_key: str,
        host: Optional[str] = None,
    ) -> None:
        from langfuse import Langfuse, propagate_attributes  # type: ignore

        kwargs: Dict[str, Any] = {
            "public_key": public_key,
            "secret_key": secret_key,
        }
        if host:
            kwargs["host"] = host
        self.client = Langfuse(**kwargs)
        self._propagate_attributes = propagate_attributes

    def _trace_context(
        self,
        *,
        trace_id: Optional[str] = None,
        parent_observation_id: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        context: Dict[str, str] = {}
        if trace_id:
            context["trace_id"] = trace_id
        if parent_observation_id:
            context["parent_observation_id"] = parent_observation_id
        return context or None

    def start_trace(
        self,
        name: str,
        *,
        trace_id: Optional[str] = None,
        parent_observation_id: Optional[str] = None,
        session_id: Optional[str] = None,
        input_payload: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        try:
            safe_session_id = str(session_id or "").strip()[:200] or None
            kwargs: Dict[str, Any] = {
                "name": name,
                "as_type": "chain",
            }
            trace_context = self._trace_context(
                trace_id=trace_id,
                parent_observation_id=parent_observation_id,
            )
            if trace_context is not None:
                kwargs["trace_context"] = trace_context
            if input_payload is not None:
                kwargs["input"] = input_payload
            if metadata is not None:
                kwargs["metadata"] = metadata

            if safe_session_id:
                with self._propagate_attributes(session_id=safe_session_id):
                    handle = self.client.start_observation(**kwargs)
            else:
                handle = self.client.start_observation(**kwargs)

            if handle is not None and safe_session_id:
                try:
                    setattr(handle, "session_id", safe_session_id)
                except Exception:
                    pass
            return handle
        except Exception:
            return None

    def event(
        self,
        trace_handle: Optional[Any],
        name: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        if trace_handle is None:
            return
        try:
            safe_session_id = str(getattr(trace_handle, "session_id", "") or "").strip()[:200] or None
            if safe_session_id:
                with self._propagate_attributes(session_id=safe_session_id):
                    self.client.create_event(
                        trace_context=self._trace_context(
                            trace_id=getattr(trace_handle, "trace_id", None),
                            parent_observation_id=getattr(trace_handle, "id", None),
                        ),
                        name=name,
                        metadata=payload or {},
                    )
            else:
                self.client.create_event(
                    trace_context=self._trace_context(
                        trace_id=getattr(trace_handle, "trace_id", None),
                        parent_observation_id=getattr(trace_handle, "id", None),
                    ),
                    name=name,
                    metadata=payload or {},
                )
        except Exception:
            return None

    def generation(
        self,
        trace_handle: Optional[Any],
        name: str,
        *,
        model: str,
        input_text: str,
        output_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        usage: Optional[Dict[str, int]] = None,
    ) -> None:
        if trace_handle is None:
            return
        try:
            safe_session_id = str(getattr(trace_handle, "session_id", "") or "").strip()[:200] or None
            generation_kwargs: Dict[str, Any] = {
                "trace_context": self._trace_context(
                    trace_id=getattr(trace_handle, "trace_id", None),
                    parent_observation_id=getattr(trace_handle, "id", None),
                ),
                "name": name,
                "as_type": "generation",
                "input": input_text,
                "model": model,
                "metadata": metadata or {},
            }
            if safe_session_id:
                with self._propagate_attributes(session_id=safe_session_id):
                    generation = self.client.start_observation(**generation_kwargs)
            else:
                generation = self.client.start_observation(**generation_kwargs)
            update_kwargs: Dict[str, Any] = {"output": output_text}
            if usage:
                update_kwargs["usage"] = usage
            try:
                generation.update(**update_kwargs)
            except TypeError:
                update_kwargs.pop("usage", None)
                generation.update(**update_kwargs)
            end = getattr(generation, "end", None)
            if callable(end):
                end()
        except Exception:
            return None

    def end_trace(
        self,
        trace_handle: Optional[Any],
        *,
        output_payload: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        if trace_handle is None:
            return
        try:
            update = getattr(trace_handle, "update", None)
            if callable(update):
                kwargs: Dict[str, Any] = {}
                if output_payload is not None:
                    kwargs["output"] = output_payload
                if metadata is not None:
                    kwargs["metadata"] = metadata
                if error:
                    kwargs["status_message"] = error
                if kwargs:
                    update(**kwargs)
            end = getattr(trace_handle, "end", None)
            if callable(end):
                end()
            flush = getattr(self.client, "flush", None)
            if callable(flush):
                flush()
        except Exception:
            return None


def build_observer(
    *,
    enabled: bool,
    public_key: Optional[str],
    secret_key: Optional[str],
    host: Optional[str],
) -> BaseObserver:
    if not enabled or not public_key or not secret_key:
        return NullObserver()
    try:
        return LangfuseObserver(public_key=public_key, secret_key=secret_key, host=host)
    except Exception:
        return NullObserver()
