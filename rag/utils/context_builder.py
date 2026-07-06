#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#
"""Shared Phase 3 context builder / evidence bundle helpers.

The builder is deliberately a thin prompt-boundary adapter over RAGFlow's
existing ``kbinfos`` dict contract.  Disabled mode returns the original object
unchanged; enabled mode returns a selected view that can still be consumed by
``kb_prompt`` and the final reference path.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import Counter, defaultdict
from copy import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

try:  # keep this module importable in small unit-test environments
    from common.token_utils import num_tokens_from_string
except ImportError:  # pragma: no cover - optional in small unit-test environments
    num_tokens_from_string = None  # type: ignore[assignment]

EvidenceSourceType = Literal["kb", "sql", "web", "summary"]
ContextDiversityStrategy = Literal["rank", "source_round_robin"]
_SOURCE_PRIORITY = {"kb": 0, "sql": 1, "web": 2, "summary": 3}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_MAX_TRACE_IDS = 128
_TRACE_CONTENT_PREVIEW = 120


@dataclass(frozen=True)
class EvidenceBundleConfig:
    enabled: bool = False
    max_chunks: int | None = None
    max_chunks_per_doc: int | None = None
    max_context_tokens: int | None = None
    diversity_enabled: bool = False
    diversity_strategy: ContextDiversityStrategy = "rank"
    max_chunks_per_source: int | None = None
    max_source_fraction: float | None = None

    @classmethod
    def from_env(cls, overrides: dict[str, Any] | None = None) -> "EvidenceBundleConfig":
        overrides = overrides or {}
        enabled = _bool_setting(overrides.get("context_builder_enabled"), "CONTEXT_BUILDER_ENABLED", False)
        strategy = str(overrides.get("context_diversity_strategy") or os.getenv("CONTEXT_DIVERSITY_STRATEGY") or "rank").strip().lower()
        if strategy not in {"rank", "source_round_robin"}:
            strategy = "rank"
        return cls(
            enabled=enabled,
            max_chunks=_int_setting(overrides.get("context_max_chunks"), "CONTEXT_MAX_CHUNKS"),
            max_chunks_per_doc=_int_setting(overrides.get("context_max_chunks_per_doc"), "CONTEXT_MAX_CHUNKS_PER_DOC"),
            max_context_tokens=_int_setting(overrides.get("context_max_tokens"), "CONTEXT_MAX_TOKENS"),
            diversity_enabled=_bool_setting(overrides.get("context_diversity_enabled"), "CONTEXT_DIVERSITY_ENABLED", False),
            diversity_strategy=strategy,  # type: ignore[arg-type]
            max_chunks_per_source=_int_setting(overrides.get("context_max_chunks_per_source"), "CONTEXT_MAX_CHUNKS_PER_SOURCE"),
            max_source_fraction=_float_setting(overrides.get("context_max_source_fraction"), "CONTEXT_MAX_SOURCE_FRACTION"),
        )


@dataclass
class EvidenceRecord:
    evidence_id: str
    source_type: EvidenceSourceType
    chunk_id: str | None = None
    doc_id: str | None = None
    doc_title: str | None = None
    source_uri: str | None = None
    content: str = ""
    section_page_order: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, Any] = field(default_factory=dict)
    retrieval_call_id: str | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    rank: int = 0
    selected_for_context: bool = False
    rejection_reason: str | None = None
    citation_index: int | None = None
    chunk: dict[str, Any] | None = field(default=None, repr=False, compare=False)

    def to_dict(self, *, include_content: bool = True) -> dict[str, Any]:
        data = asdict(self)
        data.pop("chunk", None)
        if not include_content:
            data.pop("content", None)
        return data

    def to_trace_dict(self) -> dict[str, Any]:
        data = self.to_dict(include_content=False)
        data["content_preview"] = self.content[:_TRACE_CONTENT_PREVIEW] if self.content else ""
        return data


@dataclass
class EvidenceBundle:
    records: list[EvidenceRecord]
    selected: list[EvidenceRecord]
    rejected: list[EvidenceRecord]
    config: EvidenceBundleConfig

    def summary(self) -> dict[str, Any]:
        rejection_counts = Counter(r.rejection_reason or "unknown" for r in self.rejected)
        source_counts = Counter(r.source_type for r in self.records)
        selected_source_counts = Counter(r.source_type for r in self.selected)
        per_doc_selected = Counter(r.doc_id for r in self.selected if r.doc_id)
        per_source_selected = Counter(_source_group_key(r) for r in self.selected if r.source_type == "kb")
        citation_mapping = [
            {
                "citation_index": r.citation_index,
                "evidence_id": r.evidence_id,
                "chunk_id": r.chunk_id,
                "doc_id": r.doc_id,
                "source_type": r.source_type,
            }
            for r in self.selected[:_MAX_TRACE_IDS]
        ]
        estimated_tokens = sum(_estimate_tokens(r.content) for r in self.selected)
        return {
            "enabled": self.config.enabled,
            "candidate_evidence_count": len(self.records),
            "selected_evidence_count": len(self.selected),
            "rejected_evidence_count": len(self.rejected),
            "rejection_reason_counts": dict(sorted(rejection_counts.items())),
            "duplicate_count": rejection_counts.get("duplicate", 0),
            "max_chunks": self.config.max_chunks,
            "max_chunks_per_doc": self.config.max_chunks_per_doc,
            "max_context_tokens": self.config.max_context_tokens,
            "diversity_enabled": self.config.diversity_enabled,
            "diversity_strategy": self.config.diversity_strategy,
            "max_chunks_per_source": self.config.max_chunks_per_source,
            "max_source_fraction": self.config.max_source_fraction,
            "selected_evidence_ids": [r.evidence_id for r in self.selected[:_MAX_TRACE_IDS]],
            "citation_index_mapping": citation_mapping,
            "source_type_counts": dict(sorted(source_counts.items())),
            "selected_source_type_counts": dict(sorted(selected_source_counts.items())),
            "per_document_selected_counts": dict(sorted((str(k), v) for k, v in per_doc_selected.items())),
            "per_source_selected_counts": dict(sorted((str(k), v) for k, v in per_source_selected.items())),
            "estimated_context_tokens": estimated_tokens,
        }


@dataclass(frozen=True)
class ContextBuilderResult:
    kbinfos: dict[str, Any]
    bundle: EvidenceBundle | None
    original_unchanged: bool


def _bool_setting(value: Any, env_name: str, default: bool) -> bool:
    if value is None:
        value = os.getenv(env_name)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
    return default


def _int_setting(value: Any, env_name: str, default: int | None = None) -> int | None:
    if value is None:
        value = os.getenv(env_name)
    if value in (None, ""):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _float_setting(value: Any, env_name: str, default: float | None = None) -> float | None:
    if value is None:
        value = os.getenv(env_name)
    if value in (None, ""):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def normalize_source_type(source_type: str | None, chunk: dict[str, Any] | None = None) -> EvidenceSourceType:
    normalized = (source_type or "").strip().lower()
    if normalized == "summarize":
        normalized = "summary"
    if normalized in _SOURCE_PRIORITY:
        return normalized  # type: ignore[return-value]
    chunk = chunk or {}
    marked = chunk.get("_ragflow_source_type") or chunk.get("source_type")
    if marked:
        return normalize_source_type(str(marked), None)
    if chunk.get("url") and chunk.get("chunk_id") and chunk.get("doc_id") == chunk.get("chunk_id"):
        return "web"
    return "kb"


def mark_chunks_source_type(chunks: list[dict[str, Any]] | None, source_type: str) -> None:
    canonical = normalize_source_type(source_type)
    for chunk in chunks or []:
        if isinstance(chunk, dict):
            chunk["_ragflow_source_type"] = canonical


def _content_from_chunk(chunk: dict[str, Any]) -> str:
    content = chunk.get("content")
    if content is None:
        content = chunk.get("content_with_weight")
    return "" if content is None else str(content)


def _position_from_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "page": chunk.get("page") or chunk.get("page_num") or chunk.get("page_idx"),
        "order": chunk.get("position_int") or chunk.get("position") or chunk.get("chunk_order"),
        "section": chunk.get("section") or chunk.get("section_name"),
        "positions": chunk.get("positions"),
    }


def _scores_from_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    final_score = chunk.get("final_score", chunk.get("score", chunk.get("similarity")))
    return {
        "score": final_score,
        "similarity": chunk.get("similarity"),
        "term_similarity": chunk.get("term_similarity"),
        "vector_similarity": chunk.get("vector_similarity"),
        "rank_feature_score": chunk.get("rank_feature_score") or chunk.get("pagerank_fea") or chunk.get("rank_feature"),
        "final_score": final_score,
    }


def evidence_records_from_kbinfos(
    kbinfos: dict[str, Any],
    *,
    source_type: str | None = None,
    retrieval_call_id: str | None = None,
    tool_call_id: str | None = None,
) -> list[EvidenceRecord]:
    records: list[EvidenceRecord] = []
    for rank, chunk in enumerate(kbinfos.get("chunks") or []):
        if not isinstance(chunk, dict):
            continue
        canonical_source = normalize_source_type(source_type, chunk)
        evidence_id = _evidence_id(canonical_source, chunk, rank)
        metadata = dict(chunk.get("document_metadata") or {})
        if chunk.get("_ragflow_source_type"):
            metadata["source_type"] = chunk.get("_ragflow_source_type")
        records.append(
            EvidenceRecord(
                evidence_id=evidence_id,
                source_type=canonical_source,
                chunk_id=chunk.get("chunk_id") or chunk.get("id"),
                doc_id=chunk.get("doc_id"),
                doc_title=chunk.get("docnm_kwd") or chunk.get("document_name") or chunk.get("title"),
                source_uri=chunk.get("url") or chunk.get("source_uri"),
                content=_content_from_chunk(chunk),
                section_page_order=_position_from_chunk(chunk),
                scores=_scores_from_chunk(chunk),
                retrieval_call_id=retrieval_call_id or chunk.get("retrieval_call_id"),
                tool_call_id=tool_call_id or chunk.get("tool_call_id"),
                metadata=metadata,
                rank=rank,
                chunk=chunk,
            )
        )
    return records


def build_context_bundle(
    records: list[EvidenceRecord],
    config: EvidenceBundleConfig,
    *,
    start_citation_index: int = 0,
) -> EvidenceBundle:
    selected: list[EvidenceRecord] = []
    rejected: list[EvidenceRecord] = []
    seen: dict[tuple[str, str], EvidenceRecord] = {}
    per_doc: defaultdict[str, int] = defaultdict(int)
    used_tokens = 0

    ordered = sorted(records, key=lambda record: _record_sort_key(record, config))
    if config.diversity_enabled and config.diversity_strategy == "source_round_robin":
        ordered = _round_robin_by_source(ordered)

    source_limit = _source_limit(config)
    per_source: defaultdict[str, int] = defaultdict(int)

    for record in ordered:
        dup_key = _dedupe_key(record)
        source_key = _source_group_key(record)
        if dup_key in seen:
            _reject(record, "duplicate", rejected)
            continue
        if config.diversity_enabled and source_limit and record.source_type == "kb" and per_source[source_key] >= source_limit:
            _reject(record, "max_chunks_per_source", rejected)
            continue
        if config.max_chunks is not None and len(selected) >= config.max_chunks:
            _reject(record, "max_chunks", rejected)
            continue
        if record.source_type == "kb" and config.max_chunks_per_doc and record.doc_id:
            if per_doc[record.doc_id] >= config.max_chunks_per_doc:
                _reject(record, "max_chunks_per_doc", rejected)
                continue
        estimated = _estimate_tokens(record.content)
        if config.max_context_tokens is not None and selected and used_tokens + estimated > config.max_context_tokens:
            _reject(record, "token_budget", rejected)
            continue
        if config.max_context_tokens is not None and not selected and estimated > config.max_context_tokens:
            _reject(record, "token_budget", rejected)
            continue
        record.selected_for_context = True
        record.rejection_reason = None
        record.citation_index = start_citation_index + len(selected)
        selected.append(record)
        seen[dup_key] = record
        used_tokens += estimated
        if record.source_type == "kb" and record.doc_id:
            per_doc[record.doc_id] += 1
        if config.diversity_enabled and record.source_type == "kb":
            per_source[source_key] += 1
    return EvidenceBundle(records=ordered, selected=selected, rejected=rejected, config=config)


def apply_context_builder_to_kbinfos(
    kbinfos: dict[str, Any],
    config: EvidenceBundleConfig,
    *,
    start_citation_index: int = 0,
    source_type: str | None = None,
    retrieval_call_id: str | None = None,
    tool_call_id: str | None = None,
) -> ContextBuilderResult:
    if not config.enabled:
        return ContextBuilderResult(kbinfos=kbinfos, bundle=None, original_unchanged=True)
    records = evidence_records_from_kbinfos(kbinfos, source_type=source_type, retrieval_call_id=retrieval_call_id, tool_call_id=tool_call_id)
    bundle = build_context_bundle(records, config, start_citation_index=start_citation_index)
    return ContextBuilderResult(kbinfos=bundle_to_kbinfos(bundle, kbinfos), bundle=bundle, original_unchanged=False)


def bundle_to_kbinfos(bundle: EvidenceBundle, original_kbinfos: dict[str, Any]) -> dict[str, Any]:
    selected_chunks = [r.chunk for r in bundle.selected if r.chunk is not None]
    selected_doc_ids = {c.get("doc_id") for c in selected_chunks if isinstance(c, dict) and c.get("doc_id")}
    new_kbinfos = copy(original_kbinfos)
    new_kbinfos["chunks"] = selected_chunks
    doc_aggs = original_kbinfos.get("doc_aggs") or []
    if selected_doc_ids and isinstance(doc_aggs, list):
        filtered_aggs = [d for d in doc_aggs if not isinstance(d, dict) or d.get("doc_id") in selected_doc_ids]
        new_kbinfos["doc_aggs"] = filtered_aggs or doc_aggs
    else:
        new_kbinfos["doc_aggs"] = doc_aggs
    if "total" in original_kbinfos:
        new_kbinfos["total"] = original_kbinfos["total"]
    return new_kbinfos


def _reject(record: EvidenceRecord, reason: str, rejected: list[EvidenceRecord]) -> None:
    record.selected_for_context = False
    record.rejection_reason = reason
    record.citation_index = None
    rejected.append(record)


def _record_sort_key(record: EvidenceRecord, _config: EvidenceBundleConfig) -> tuple[Any, ...]:
    return (record.rank, record.evidence_id)


def _round_robin_by_source(records: list[EvidenceRecord]) -> list[EvidenceRecord]:
    buckets: dict[str, list[EvidenceRecord]] = defaultdict(list)
    source_order: list[str] = []
    for record in records:
        source_key = _source_group_key(record)
        if source_key not in buckets:
            source_order.append(source_key)
        buckets[source_key].append(record)
    for bucket in buckets.values():
        bucket.sort(key=lambda record: _record_sort_key(record, EvidenceBundleConfig(enabled=True)))

    ordered: list[EvidenceRecord] = []
    while any(buckets.values()):
        for source_key in source_order:
            if buckets[source_key]:
                ordered.append(buckets[source_key].pop(0))
    return ordered


def _source_limit(config: EvidenceBundleConfig) -> int | None:
    limits: list[int] = []
    if config.max_chunks_per_source:
        limits.append(config.max_chunks_per_source)
    if config.max_chunks and config.max_source_fraction:
        limits.append(max(1, int(config.max_chunks * config.max_source_fraction + 0.999999)))
    return min(limits) if limits else None


def _dedupe_key(record: EvidenceRecord) -> tuple[str, str]:
    if record.source_type == "web":
        if record.source_uri:
            return record.source_type, f"uri-content:{_normalized_uri(record.source_uri)}:{_content_hash(record.content)}"
        return record.source_type, "content:" + _content_hash(record.content)
    if record.chunk_id:
        return record.source_type, f"chunk:{record.chunk_id}"
    section_key = _section_key(record.section_page_order)
    if record.doc_id and section_key:
        return record.source_type, f"doc-section:{record.doc_id}:{section_key}"
    return record.source_type, "content:" + _content_hash(record.content)


def _section_key(section_page_order: dict[str, Any]) -> str:
    parts = []
    for key in ("section", "page", "order"):
        value = section_page_order.get(key)
        if value not in (None, ""):
            parts.append(str(value))
    return ":".join(parts)


def _normalized_uri(uri: str) -> str:
    return (uri or "").strip().lower().rstrip("/")


def _normalized_doc_title(title: str | None) -> str:
    value = (title or "").strip().lower()
    if not value:
        return ""
    value = re.sub(r"\(\d+\)(?=\.[^.]+$|$)", "", value)
    value = re.sub(r"\.[a-z0-9]{1,8}$", "", value)
    value = re.sub(r"[_\s]+", " ", value)
    return value.strip()


def _source_group_key(record: EvidenceRecord) -> str:
    title = _normalized_doc_title(record.doc_title)
    if title:
        return f"title:{title}"
    if record.source_uri:
        return f"uri:{_normalized_uri(record.source_uri)}"
    if record.doc_id:
        return f"doc:{record.doc_id}"
    return f"evidence:{record.evidence_id}"


def _content_hash(content: str) -> str:
    normalized = re.sub(r"\s+", " ", (content or "").strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _evidence_id(source_type: EvidenceSourceType, chunk: dict[str, Any], rank: int) -> str:
    content_hash = _content_hash(_content_from_chunk(chunk))
    if source_type == "web":
        uri = chunk.get("url") or chunk.get("source_uri")
        if uri:
            return f"web:{_content_hash(_normalized_uri(str(uri)))}:{content_hash}"
    chunk_id = chunk.get("chunk_id") or chunk.get("id")
    if chunk_id:
        return f"{source_type}:{chunk_id}"
    doc_id = chunk.get("doc_id") or "no-doc"
    return f"{source_type}:{doc_id}:{rank}:{content_hash}"


def _estimate_tokens(content: str) -> int:
    if not content:
        return 0
    if num_tokens_from_string is not None:
        return int(num_tokens_from_string(content))
    return max(1, (len(content) + 3) // 4)
