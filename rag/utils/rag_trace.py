#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""Lightweight Phase 0 RAG diagnostics trace support.

The collector is intentionally local and best-effort: it records compact,
JSON-serializable metadata for one chat request, avoids vectors and full chunk
bodies, and writes JSONL only when diagnostics are explicitly enabled.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

_TRACE_SCHEMA_VERSION = "rag_trace.v1"
_SENSITIVE_KEY_PARTS = ("api_key", "apikey", "secret", "password", "authorization", "credential")
_SENSITIVE_EXACT_KEYS = {"token", "access_token", "refresh_token", "id_token", "auth_token", "api_token", "bearer_token"}
_VECTOR_KEY_PARTS = ("vector", "embedding")
_TEXT_KEYS_TO_TRUNCATE = {"question", "query", "keywords", "model_name", "model_provider", "error"}
_MAX_TEXT_LEN = 512
_MAX_LIST_ITEMS = 32
_OMIT = object()


def _now_ms() -> float:
    return time.time() * 1000.0


def _duration_ms(start_ms: float | None) -> float | None:
    if start_ms is None:
        return None
    return round(max(0.0, _now_ms() - start_ms), 3)


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in _SENSITIVE_EXACT_KEYS or any(part in lowered for part in _SENSITIVE_KEY_PARTS)


def _is_vector_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in _VECTOR_KEY_PARTS)


def sanitize_for_trace(value: Any, *, key: str | None = None, depth: int = 0) -> Any:
    """Return a JSON-safe, redacted, bounded representation for traces."""
    if key and _is_sensitive_key(key):
        return "<redacted>"
    if key and _is_vector_key(key) and not (value is None or isinstance(value, (bool, int, float, str))):
        return _OMIT
    if depth > 6:
        return "<omitted:depth>"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, str):
        if key not in _TEXT_KEYS_TO_TRUNCATE and len(value) > _MAX_TEXT_LEN:
            return value[:_MAX_TEXT_LEN] + "...<truncated>"
        return value[:_MAX_TEXT_LEN] + "...<truncated>" if len(value) > _MAX_TEXT_LEN else value
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for k, v in list(value.items())[:_MAX_LIST_ITEMS]:
            item = sanitize_for_trace(v, key=str(k), depth=depth + 1)
            if item is not _OMIT:
                sanitized[str(k)] = item
        return sanitized
    if isinstance(value, (list, tuple, set)):
        sanitized_list = [sanitize_for_trace(v, depth=depth + 1) for v in list(value)[:_MAX_LIST_ITEMS]]
        return [v for v in sanitized_list if v is not _OMIT]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)[:_MAX_TEXT_LEN]


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping.get(key)
    return None


def _score_fields(chunk: dict[str, Any]) -> dict[str, Any]:
    similarity = chunk.get("similarity")
    final_score = chunk.get("final_score", chunk.get("score", similarity))
    return {
        "score": final_score,
        "similarity": similarity,
        "term_similarity": chunk.get("term_similarity"),
        "vector_similarity": chunk.get("vector_similarity"),
        "rank_feature_score": _first_present(chunk, ("rank_feature_score", "pagerank_fea", "rank_feature")),
        "final_score": final_score,
    }


def summarize_chunks(chunks: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Summarize chunks without full content or vectors."""
    summarized = []
    for chunk in chunks or []:
        raw_agentic = chunk.get("_ragflow_agentic_retrieval")
        agentic: dict[str, Any] = raw_agentic if isinstance(raw_agentic, dict) else {}
        item = {
            "chunk_id": chunk.get("chunk_id") or chunk.get("id"),
            "doc_id": chunk.get("doc_id"),
            "doc_title": chunk.get("docnm_kwd") or chunk.get("document_name"),
            "source_uri": chunk.get("url"),
            "plan_id": chunk.get("plan_id") or agentic.get("plan_id"),
            "facet_id": chunk.get("facet_id") or agentic.get("facet_id"),
            "subquery_id": chunk.get("subquery_id") or agentic.get("subquery_id"),
            "retrieval_call_id": chunk.get("retrieval_call_id") or agentic.get("retrieval_call_id"),
            "lineage_rank": chunk.get("lineage_rank") or agentic.get("lineage_rank"),
            "merge_rank": chunk.get("merge_rank") or agentic.get("merge_rank"),
            "selected_for_context": chunk.get("selected_for_context") if chunk.get("selected_for_context") is not None else agentic.get("selected_for_context"),
            "rejection_reason": chunk.get("rejection_reason") or agentic.get("rejection_reason"),
        }
        for key in ("iteration_id", "followup_id"):
            value = chunk.get(key) or agentic.get(key)
            if value is not None:
                item[key] = value
        item.update(_score_fields(chunk))
        summarized.append(item)
    return sanitize_for_trace(summarized)


def _chunk_scores(chunk: dict[str, Any]) -> dict[str, Any]:
    return sanitize_for_trace(_score_fields(chunk))


def _chunk_position(chunk: dict[str, Any]) -> dict[str, Any]:
    return sanitize_for_trace(
        {
            "page": chunk.get("page") or chunk.get("page_num") or chunk.get("page_idx"),
            "order": chunk.get("position_int") or chunk.get("position") or chunk.get("chunk_order"),
            "section": chunk.get("section") or chunk.get("section_name"),
        }
    )


def trace_enabled_from_env() -> bool:
    return str(os.getenv("RAG_TRACE_ENABLED", "")).strip().lower() in {"1", "true", "yes", "on"}


class RagTraceCollector:
    """In-memory diagnostic trace for one normal or agentic chat request."""

    def __init__(
        self,
        *,
        path: str,
        question: str | None = None,
        session_id: str | None = None,
        conversation_id: str | None = None,
        enabled: bool = False,
        output_path: str | None = None,
        include_in_response: bool = False,
    ):
        self.enabled = bool(enabled or output_path or include_in_response)
        self.output_path = output_path
        self.include_in_response = include_in_response
        self._started_ms = _now_ms()
        self._tool_seq = 0
        self._retrieval_seq = 0
        self._evidence_seq = 0
        self._tool_index: dict[str, dict[str, Any]] = {}
        self.data: dict[str, Any] = {
            "schema_version": _TRACE_SCHEMA_VERSION,
            "trace_id": uuid.uuid4().hex,
            "path": path,
            "question": sanitize_for_trace(question or "", key="question"),
            "session_id": sanitize_for_trace(session_id),
            "conversation_id": sanitize_for_trace(conversation_id),
            "tool_calls": [],
            "retrieval_calls": [],
            "evidence_ledger": [],
            "citation_mappings": [],
            "context_builder": [],
            "agentic_retrieval": [],
            "agentic_planning": [],
            "agentic_refinement": [],
            "llm": {},
            "latency_ms": {},
            "token_usage": {},
            "errors": [],
        }

    @classmethod
    def from_kwargs(cls, *, path: str, dialog: Any = None, messages: list[dict[str, Any]] | None = None, kwargs: dict[str, Any] | None = None) -> "RagTraceCollector | None":
        kwargs = kwargs or {}
        explicit_enabled = bool(kwargs.get("rag_trace_enabled") or kwargs.get("rag_trace") or trace_enabled_from_env())
        # JSONL trace files are a server-side diagnostic sink only. Do not accept
        # request-controlled file paths from kwargs; callers may use the env var
        # to configure a deployment-owned output path.
        output_path = os.getenv("RAG_TRACE_PATH")
        include_in_response = bool(kwargs.get("include_rag_trace") or kwargs.get("rag_trace_include_response"))
        if not (explicit_enabled or output_path or include_in_response or kwargs.get("rag_trace_collector")):
            return None
        if kwargs.get("rag_trace_collector") is not None:
            return kwargs["rag_trace_collector"]
        question = ""
        for message in reversed(messages or []):
            if message.get("role") == "user":
                question = message.get("content") or ""
                break
        return cls(
            path=path,
            question=question,
            session_id=kwargs.get("session_id"),
            conversation_id=getattr(dialog, "id", None),
            enabled=explicit_enabled,
            output_path=output_path,
            include_in_response=include_in_response,
        )

    def set_path(self, path: str) -> None:
        self.data["path"] = path

    def set_llm_info(self, **fields: Any) -> None:
        if not self.enabled:
            return
        self.data["llm"].update(sanitize_for_trace(fields))

    def start_tool(self, name: str, args: dict[str, Any] | None = None, *, retrieval_mode: str | None = None) -> str | None:
        if not self.enabled:
            return None
        self._tool_seq += 1
        tool_call_id = f"tool-{self._tool_seq:04d}"
        event = {
            "tool_call_id": tool_call_id,
            "name": name,
            "arguments": sanitize_for_trace(args or {}),
            "retrieval_mode": retrieval_mode,
            "start_time_ms": round(_now_ms(), 3),
            "end_time_ms": None,
            "latency_ms": None,
            "success": None,
            "result_summary": {},
            "error": None,
        }
        self.data["tool_calls"].append(event)
        self._tool_index[tool_call_id] = event
        return tool_call_id

    def end_tool(self, tool_call_id: str | None, *, success: bool = True, result: Any = None, result_summary: dict[str, Any] | None = None, error: Exception | str | None = None) -> None:
        if not (self.enabled and tool_call_id and tool_call_id in self._tool_index):
            return
        event = self._tool_index[tool_call_id]
        event["end_time_ms"] = round(_now_ms(), 3)
        event["latency_ms"] = _duration_ms(event.get("start_time_ms"))
        event["success"] = bool(success)
        summary = result_summary if result_summary is not None else self.summarize_result(result)
        event["result_summary"] = sanitize_for_trace(summary)
        if error is not None:
            event["error"] = sanitize_for_trace(str(error), key="error")
            self.record_error(str(error), source=event.get("name"))

    def summarize_result(self, result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            chunks = result.get("chunks") or []
            return {
                "type": "dict",
                "keys": list(result.keys())[:_MAX_LIST_ITEMS],
                "chunk_count": len(chunks) if isinstance(chunks, list) else None,
                "doc_agg_count": len(result.get("doc_aggs") or []) if isinstance(result.get("doc_aggs"), list) else None,
                "total": result.get("total"),
            }
        if isinstance(result, list):
            return {"type": "list", "count": len(result)}
        if isinstance(result, str):
            return {"type": "str", "length": len(result)}
        return {"type": type(result).__name__}

    def add_retrieval_call(
        self,
        *,
        mode: str,
        query_text: str | None = None,
        keywords: str | None = None,
        docid_scope: list[str] | None = None,
        metadata_filters: Any = None,
        chunks: list[dict[str, Any]] | None = None,
        doc_aggs: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
        used_embedding: bool | None = None,
        used_web: bool | None = None,
        started_ms: float | None = None,
        error: Exception | str | None = None,
        retrieval_variant: str | None = None,
        similarity_threshold: float | None = None,
        vector_similarity_weight: float | None = None,
        top_k: int | None = None,
        page_size: int | None = None,
        doc_scope_enabled: bool | None = None,
        metadata_filter_enabled: bool | None = None,
        diagnostics: dict[str, Any] | None = None,
        plan_id: str | None = None,
        facet_id: str | None = None,
        subquery_id: str | None = None,
        iteration_id: str | None = None,
        followup_id: str | None = None,
    ) -> str | None:
        if not self.enabled:
            return None
        self._retrieval_seq += 1
        retrieval_call_id = f"retrieval-{self._retrieval_seq:04d}"
        record = {
            "retrieval_call_id": retrieval_call_id,
            "tool_call_id": tool_call_id,
            "mode": mode,
            "query_text": sanitize_for_trace(query_text or "", key="query"),
            "keywords": sanitize_for_trace(keywords or "", key="keywords"),
            "docid_scope": sanitize_for_trace(docid_scope or []),
            "metadata_filters": sanitize_for_trace(metadata_filters or {}),
            "chunk_count": len(chunks or []),
            "doc_agg_count": len(doc_aggs or []),
            "chunks": summarize_chunks(chunks),
            "used_embedding": used_embedding,
            "used_web": used_web,
            "retrieval_variant": retrieval_variant,
            "similarity_threshold": similarity_threshold,
            "vector_similarity_weight": vector_similarity_weight,
            "top_k": top_k,
            "page_size": page_size,
            "doc_scope_enabled": doc_scope_enabled,
            "metadata_filter_enabled": metadata_filter_enabled,
            "diagnostics": sanitize_for_trace(diagnostics or {}),
            "plan_id": plan_id,
            "facet_id": facet_id,
            "subquery_id": subquery_id,
            "iteration_id": iteration_id,
            "latency_ms": _duration_ms(started_ms),
            "error": sanitize_for_trace(str(error), key="error") if error else None,
        }
        if followup_id is not None:
            record["followup_id"] = followup_id
        self.data["retrieval_calls"].append(record)
        if error:
            self.record_error(str(error), source=mode)
        return retrieval_call_id

    def add_evidence_from_chunks(
        self,
        chunks: list[dict[str, Any]] | None,
        *,
        source_type: str,
        retrieval_call_id: str | None = None,
        tool_call_id: str | None = None,
        start_citation_index: int = 0,
        selected_for_context: str = "unknown",
        support_status: str = "unverified",
    ) -> None:
        if not self.enabled:
            return
        for offset, chunk in enumerate(chunks or []):
            raw_agentic = chunk.get("_ragflow_agentic_retrieval")
            agentic: dict[str, Any] = raw_agentic if isinstance(raw_agentic, dict) else {}
            self._evidence_seq += 1
            evidence_id = f"evidence-{self._evidence_seq:05d}"
            citation_index = start_citation_index + offset
            entry = {
                "evidence_id": evidence_id,
                "tool_call_id": tool_call_id,
                "retrieval_call_id": retrieval_call_id,
                "plan_id": chunk.get("plan_id") or agentic.get("plan_id"),
                "facet_id": chunk.get("facet_id") or agentic.get("facet_id"),
                "subquery_id": chunk.get("subquery_id") or agentic.get("subquery_id"),
                "iteration_id": chunk.get("iteration_id") or agentic.get("iteration_id"),
                "lineage_rank": chunk.get("lineage_rank") or agentic.get("lineage_rank"),
                "merge_rank": chunk.get("merge_rank") or agentic.get("merge_rank"),
                "source_type": source_type,
                "chunk_id": chunk.get("chunk_id") or chunk.get("id"),
                "doc_id": chunk.get("doc_id"),
                "doc_title": chunk.get("docnm_kwd") or chunk.get("document_name"),
                "source_uri": chunk.get("url"),
                "section_page_order": _chunk_position(chunk),
                "scores": _chunk_scores(chunk),
                "citation_index": citation_index,
                "selected_for_context": chunk.get("selected_for_context") if chunk.get("selected_for_context") is not None else agentic.get("selected_for_context", selected_for_context),
                "rejection_reason": chunk.get("rejection_reason") or agentic.get("rejection_reason"),
                "support_status": chunk.get("support_status") or agentic.get("support_status", support_status),
            }
            followup_id = chunk.get("followup_id") or agentic.get("followup_id")
            if followup_id is not None:
                entry["followup_id"] = followup_id
            self.data["evidence_ledger"].append(sanitize_for_trace(entry))
            self.data["citation_mappings"].append(
                sanitize_for_trace(
                    {
                        "citation_index": citation_index,
                        "evidence_id": evidence_id,
                        "chunk_id": entry["chunk_id"],
                        "doc_id": entry["doc_id"],
                        "final_reference_position": None,
                    }
                )
            )

    def add_context_builder_summary(self, summary: dict[str, Any] | None, *, stage: str | None = None) -> None:
        if not self.enabled:
            return
        record = sanitize_for_trace(dict(summary or {}))
        if stage is not None:
            record["stage"] = sanitize_for_trace(stage, key="stage")
        self.data["context_builder"].append(record)

    def add_agentic_retrieval_event(self, stage: str, payload: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        record = dict(payload or {})
        record["stage"] = stage
        self.data["agentic_retrieval"].append(sanitize_for_trace(record))

    def add_agentic_planner_event(self, **fields: Any) -> None:
        if not self.enabled:
            return
        record = {"event": "planner"}
        record.update(fields)
        plan_id = str(record.get("plan_id") or "")
        if _is_sensitive_key(plan_id):
            record["plan_id"] = "<redacted-plan-id>"
        self.data["agentic_planning"].append(sanitize_for_trace(record))

    def add_agentic_subquery_event(self, **fields: Any) -> None:
        if not self.enabled:
            return
        record = {"event": "subquery"}
        record.update(fields)
        self.data["agentic_planning"].append(sanitize_for_trace(record))

    def add_agentic_refinement_event(self, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        record = {"event": event}
        record.update(fields)
        self.data["agentic_refinement"].append(sanitize_for_trace(record))

    def add_agentic_fallback(self, **fields: Any) -> None:
        if not self.enabled:
            return
        record = {"event": "fallback"}
        if "reason" in record and "fallback_reason" not in fields:
            record["fallback_reason"] = record.pop("reason")
        record.update(fields)
        if "reason" in record and "fallback_reason" not in record:
            record["fallback_reason"] = record.pop("reason")
        self.data["agentic_planning"].append(sanitize_for_trace(record))

    def record_error(self, message: str, *, source: str | None = None) -> None:
        if not self.enabled:
            return
        self.data["errors"].append(sanitize_for_trace({"source": source, "message": message}, key="error"))

    def finish(
        self,
        *,
        final_reference_count: int | None = None,
        latency_timings: dict[str, Any] | None = None,
        token_usage: dict[str, Any] | None = None,
        citation_ids: list[int] | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        self.data["final_reference_count"] = final_reference_count
        self.data["latency_ms"].update(sanitize_for_trace(latency_timings or {}))
        self.data["latency_ms"]["total_trace"] = _duration_ms(self._started_ms)
        self.data["token_usage"].update(sanitize_for_trace(token_usage or {}))
        if citation_ids is not None:
            self.data["citation_ids"] = sanitize_for_trace(sorted(citation_ids))
        out = self.to_dict()
        self.write_jsonl()
        return out if self.include_in_response else None

    def to_dict(self) -> dict[str, Any]:
        payload = sanitize_for_trace(deepcopy(self.data))
        for sanitized, recorded in zip(payload.get("context_builder", []), self.data.get("context_builder", [])):
            if isinstance(sanitized, dict) and isinstance(recorded, dict) and "stage" in recorded:
                sanitized["stage"] = recorded["stage"]
        return payload

    def write_jsonl(self) -> None:
        if not (self.enabled and self.output_path):
            return
        path = Path(self.output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
