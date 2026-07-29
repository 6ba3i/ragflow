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
_MAX_TRACE_CANDIDATE_OBSERVATIONS = 48
CONTEXT_TRACE_POLICY_VERSION = "2"
_STRONG_RELEVANCE_THRESHOLD = 0.5
_SCORE_FAMILY_BY_FIELD = {
    "vector_similarity": "dense_similarity",
    "term_similarity": "lexical_similarity",
}
_COMPARABLE_SCORE_SCALES = {"unit_interval", "cosine_similarity"}


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
    relevance_filter_enabled: bool = False
    min_vector_similarity: float | None = None
    min_term_similarity: float | None = None
    preserve_top_ranked: int = 3
    preserve_query_token_overlap: bool = True
    primary_doc_extra_chunks: int = 0
    primary_doc_min_rank_hits: int = 2
    primary_doc_max_chunks: int | None = None

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
            relevance_filter_enabled=_bool_setting(overrides.get("context_relevance_filter_enabled"), "CONTEXT_RELEVANCE_FILTER_ENABLED", False),
            min_vector_similarity=_float_setting(overrides.get("context_min_vector_similarity"), "CONTEXT_MIN_VECTOR_SIMILARITY"),
            min_term_similarity=_float_setting(overrides.get("context_min_term_similarity"), "CONTEXT_MIN_TERM_SIMILARITY"),
            preserve_top_ranked=_int_setting(overrides.get("context_preserve_top_ranked"), "CONTEXT_PRESERVE_TOP_RANKED", 3) or 3,
            preserve_query_token_overlap=_bool_setting(overrides.get("context_preserve_query_token_overlap"), "CONTEXT_PRESERVE_QUERY_TOKEN_OVERLAP", True),
            primary_doc_extra_chunks=_int_setting(overrides.get("context_primary_doc_extra_chunks"), "CONTEXT_PRIMARY_DOC_EXTRA_CHUNKS", 0) or 0,
            primary_doc_min_rank_hits=_int_setting(overrides.get("context_primary_doc_min_rank_hits"), "CONTEXT_PRIMARY_DOC_MIN_RANK_HITS", 2) or 2,
            primary_doc_max_chunks=_int_setting(overrides.get("context_primary_doc_max_chunks"), "CONTEXT_PRIMARY_DOC_MAX_CHUNKS"),
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
    protection_reason: str | None = None
    selection_reason: str | None = None
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
    query_feature_flags: dict[str, bool] = field(default_factory=dict)
    priority_boosted_evidence_ids: list[str] = field(default_factory=list)
    primary_document_ids: list[str] = field(default_factory=list)
    per_document_effective_limits: dict[str, int] = field(default_factory=dict)
    primary_doc_extra_selected_count: int = 0
    priority_boost_strength_counts: dict[str, int] = field(default_factory=dict)
    low_relevance_preserved_count: int = 0
    low_relevance_bypass_reason_counts: dict[str, int] = field(default_factory=dict)
    relevance_decision_counts: dict[str, int] = field(default_factory=dict)
    per_source_base_limits: dict[str, int] = field(default_factory=dict)
    per_source_effective_limits: dict[str, int] = field(default_factory=dict)
    source_cap_expansion_reasons: dict[str, list[str]] = field(default_factory=dict)

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
        agentic_lineage = _agentic_lineage_summary(self.records, self.selected, self.rejected)
        compilation_survival = _compilation_survival_summary(self.records, self.selected)
        provenance_counts = _score_provenance_counts(self.records)
        candidate_observations = _candidate_trace_observations(self.records, self.selected, self.rejected)
        candidate_observation_overflow_count = max(0, len(self.records) - len(candidate_observations))
        protected_records = [record for record in self.records if record.protection_reason]
        selected_by_reason = Counter(record.selection_reason or "rank" for record in self.selected)
        summary = {
            "policy_version": CONTEXT_TRACE_POLICY_VERSION,
            "enabled": self.config.enabled,
            "candidate_evidence_count": len(self.records),
            "input_candidate_count": len(self.records),
            "unique_candidate_count": len({_dedupe_key(record) for record in self.records}),
            "selected_evidence_count": len(self.selected),
            "rejected_evidence_count": len(self.rejected),
            "protected_count": len(protected_records),
            "selected_protected_count": sum(record in self.selected for record in protected_records),
            "selected_by_reason": dict(sorted(selected_by_reason.items())),
            "rejection_reason_counts": dict(sorted(rejection_counts.items())),
            "source_cap_rejection_count": rejection_counts.get("source_cap", 0),
            "document_cap_rejection_count": rejection_counts.get("document_cap", 0),
            "context_limit_rejection_count": rejection_counts.get("context_limit", 0),
            "duplicate_count": rejection_counts.get("duplicate", 0),
            "max_chunks": self.config.max_chunks,
            "max_chunks_per_doc": self.config.max_chunks_per_doc,
            "max_context_tokens": self.config.max_context_tokens,
            "diversity_enabled": self.config.diversity_enabled,
            "diversity_strategy": self.config.diversity_strategy,
            "max_chunks_per_source": self.config.max_chunks_per_source,
            "max_source_fraction": self.config.max_source_fraction,
            "relevance_filter_enabled": self.config.relevance_filter_enabled,
            **(
                {
                    "min_vector_similarity": self.config.min_vector_similarity,
                    "min_term_similarity": self.config.min_term_similarity,
                    "low_relevance_rejected_count": rejection_counts.get("low_relevance", 0),
                }
                if self.config.relevance_filter_enabled
                else {"low_relevance_rejected_count": rejection_counts.get("low_relevance", 0)}
            ),
            "low_relevance_preserved_count": self.low_relevance_preserved_count,
            "low_relevance_bypass_reason_counts": dict(sorted(self.low_relevance_bypass_reason_counts.items())),
            "selected_evidence_ids": [r.evidence_id for r in self.selected[:_MAX_TRACE_IDS]],
            "per_source_selected_counts": dict(sorted((str(k), v) for k, v in per_source_selected.items())),
            "relevance_decision_counts": dict(sorted(self.relevance_decision_counts.items())),
            "score_provenance_counts": provenance_counts,
            "source_cap_policy": {
                "base_limits": dict(sorted(self.per_source_base_limits.items())),
                "effective_limits": dict(sorted(self.per_source_effective_limits.items())),
                "expansion_reasons": {key: list(reasons) for key, reasons in sorted(self.source_cap_expansion_reasons.items())},
            },
            "query_feature_flags": dict(sorted(self.query_feature_flags.items())),
            "priority_boosted_count": len(self.priority_boosted_evidence_ids),
            "priority_boosted_evidence_ids": self.priority_boosted_evidence_ids[:_MAX_TRACE_IDS],
            "priority_boost_strength_counts": dict(sorted(self.priority_boost_strength_counts.items())),
            "primary_document_ids": self.primary_document_ids[:_MAX_TRACE_IDS],
            "per_document_effective_limits": dict(sorted((str(k), v) for k, v in self.per_document_effective_limits.items())),
            "primary_doc_extra_selected_count": self.primary_doc_extra_selected_count,
            "citation_index_mapping": citation_mapping,
            "source_type_counts": dict(sorted(source_counts.items())),
            "selected_source_type_counts": dict(sorted(selected_source_counts.items())),
            "per_document_selected_counts": dict(sorted((str(k), v) for k, v in per_doc_selected.items())),
            "per_source_base_limits": dict(sorted(self.per_source_base_limits.items())),
            "per_source_effective_limits": dict(sorted(self.per_source_effective_limits.items())),
            "source_cap_expansion_reasons": {key: list(reasons) for key, reasons in sorted(self.source_cap_expansion_reasons.items())},
            # Compatibility alias for trace consumers that adopted the plan name.
            "per_source_expansion_reasons": {key: list(reasons) for key, reasons in sorted(self.source_cap_expansion_reasons.items())},
            "estimated_context_tokens": estimated_tokens,
            "represented_source_count": len(per_source_selected),
            "represented_document_count": len(per_doc_selected),
            "max_chunks_from_one_source": max(per_source_selected.values(), default=0),
            "max_chunks_from_one_document": max(per_doc_selected.values(), default=0),
            "candidate_observations": candidate_observations,
            "candidate_observation_overflow_count": candidate_observation_overflow_count,
        }
        if agentic_lineage:
            summary["agentic_lineage"] = agentic_lineage
        if compilation_survival:
            summary["compilation_context_survival"] = compilation_survival
        return summary


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


def _score_provenance_from_chunk(chunk: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = chunk.get("_ragflow_score_provenance")
    if isinstance(raw, dict):
        return {str(score_name): _normalize_score_provenance(score_name, provenance) for score_name, provenance in raw.items() if isinstance(score_name, str)}

    # Untagged legacy scores have no trustworthy family or scale contract.
    inferred: dict[str, dict[str, Any]] = {}
    for score_name in (*_SCORE_FAMILY_BY_FIELD, "similarity", "score", "final_score", "fusion_score", "reranker_score"):
        if chunk.get(score_name) is None:
            continue
        inferred[score_name] = {
            "family": "unknown",
            "scale": "unknown",
            "calibrated": False,
            "source": "legacy_untyped_score",
        }
    return inferred


def _normalize_score_provenance(score_name: str, provenance: Any) -> dict[str, Any]:
    if not isinstance(provenance, dict):
        return {
            "family": "unknown",
            "scale": "unknown",
            "calibrated": False,
            "source": "invalid_provenance",
        }
    normalized = {
        "family": str(provenance.get("family") or "unknown"),
        "scale": str(provenance.get("scale") or "unknown"),
        "calibrated": provenance.get("calibrated") is True,
        "source": str(provenance.get("source") or "unknown"),
    }
    rank = provenance.get("rank")
    if isinstance(rank, int) and rank > 0:
        normalized["rank"] = rank
    return normalized


def _scores_from_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    final_score = chunk.get("final_score", chunk.get("score", chunk.get("similarity")))
    return {
        "score": final_score,
        "similarity": chunk.get("similarity"),
        "term_similarity": chunk.get("term_similarity"),
        "vector_similarity": chunk.get("vector_similarity"),
        "rank_feature_score": chunk.get("rank_feature_score") or chunk.get("pagerank_fea") or chunk.get("rank_feature"),
        "final_score": final_score,
        "fusion_score": chunk.get("fusion_score"),
        "reranker_score": chunk.get("reranker_score"),
        "score_provenance": _score_provenance_from_chunk(chunk),
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
        agentic_metadata = chunk.get("_ragflow_agentic_retrieval")
        if isinstance(agentic_metadata, dict):
            metadata["agentic_retrieval"] = dict(agentic_metadata)
        compilation_metadata = chunk.get("_ragflow_compilation")
        if isinstance(compilation_metadata, dict):
            metadata["compilation"] = dict(compilation_metadata)
        for lineage_key in (
            "plan_id",
            "facet_id",
            "subquery_id",
            "iteration_id",
            "followup_id",
            "retrieval_call_id",
            "lineage_rank",
            "merge_rank",
            "retrieval_variant",
            "selected_for_context",
            "rejection_reason",
        ):
            if lineage_key in chunk and chunk.get(lineage_key) is not None:
                metadata[lineage_key] = chunk.get(lineage_key)
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
    query: str | None = None,
) -> EvidenceBundle:
    selected: list[EvidenceRecord] = []
    rejected: list[EvidenceRecord] = []
    seen: dict[tuple[str, str], EvidenceRecord] = {}
    per_doc: defaultdict[str, int] = defaultdict(int)
    used_tokens = 0

    query_features = _query_features(query)
    ordered = sorted(records, key=lambda record: _record_sort_key(record, config))
    all_ordered = list(ordered)

    low_relevance_preserved_count = 0
    low_relevance_bypass_reason_counts: Counter[str] = Counter()
    relevance_decision_counts: Counter[str] = Counter()
    relevance_kept: list[EvidenceRecord] = []
    for record in ordered:
        is_low_relevance, relevance_decision = _relevance_assessment(record, config)
        relevance_decision_counts[relevance_decision] += 1
        bypass_reasons = _low_relevance_bypass_reasons(record, record.rank, config, query_features) if is_low_relevance else []
        if is_low_relevance and not bypass_reasons:
            _reject(record, "low_relevance", rejected)
        else:
            if is_low_relevance:
                low_relevance_preserved_count += 1
                low_relevance_bypass_reason_counts.update(bypass_reasons)
                record.protection_reason = ";".join(bypass_reasons) or None
            relevance_kept.append(record)
    ordered = relevance_kept

    priority_boosts_by_object_id = {id(record): _priority_boost_details(record, query_features) for record in all_ordered}
    priority_boosted_ids = [r.evidence_id for r in ordered if _priority_boost_score(priority_boosts_by_object_id.get(id(r))) > 0]
    if priority_boosted_ids:
        ordered = sorted(ordered, key=lambda record: (-_priority_boost_score(priority_boosts_by_object_id.get(id(record))), record.rank, record.evidence_id))
        for record in ordered:
            if _priority_boost_score(priority_boosts_by_object_id.get(id(record))) > 0:
                record.protection_reason = record.protection_reason or "priority_evidence"

    primary_doc_reasons = _primary_doc_candidate_reasons(ordered, config, query_features)
    primary_doc_ids = _primary_doc_ids(primary_doc_reasons)
    primary_source_reasons = _primary_source_candidate_reasons(ordered, config, query_features, primary_doc_reasons)
    effective_doc_limits = _effective_doc_limits(ordered, config, primary_doc_ids)

    source_limit = _source_limit(config)
    source_base_limits, source_effective_limits, source_expansion_reasons = _effective_source_limits(
        ordered,
        config,
        primary_doc_ids,
        primary_doc_reasons,
        primary_source_reasons,
        source_limit,
        query_features,
    )

    if config.diversity_enabled and config.diversity_strategy == "source_round_robin":
        pre_diversity_order = list(ordered)
        ordered = _round_robin_by_source(ordered)
        for record in pre_diversity_order[: config.max_chunks or len(pre_diversity_order)]:
            if _has_protected_retrieval_lineage(record):
                record.protection_reason = record.protection_reason or "protected_retrieval_lineage"
        ordered = _preserve_protected_lineage_prefix(pre_diversity_order, ordered, config.max_chunks)

    per_source: defaultdict[str, int] = defaultdict(int)
    primary_doc_extra_selected_count = 0

    for record in ordered:
        dup_key = _dedupe_key(record)
        source_key = _source_group_key(record)
        protected_baseline = _has_protected_retrieval_lineage(record)
        if dup_key in seen:
            _reject(record, "duplicate", rejected)
            continue
        record_source_limit = _source_limit_for_record(
            record,
            config,
            primary_doc_ids,
            primary_doc_reasons,
            primary_source_reasons,
            source_limit,
            query_features,
        )
        if not protected_baseline and record_source_limit is not None and record.source_type == "kb" and per_source[source_key] >= record_source_limit:
            _reject(record, "max_chunks_per_source", rejected)
            continue
        if config.max_chunks is not None and len(selected) >= config.max_chunks:
            _reject(record, "max_chunks", rejected)
            continue
        if not protected_baseline and record.source_type == "kb" and config.max_chunks_per_doc and record.doc_id:
            doc_limit = _doc_limit_for_record(record, config, primary_doc_ids)
            if per_doc[record.doc_id] >= doc_limit:
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
        record.selection_reason = "protection" if record.protection_reason else "rank"
        if config.diversity_enabled and config.diversity_strategy == "source_round_robin":
            record.selection_reason = "diversity" if not record.protection_reason else "protection"
        record.citation_index = start_citation_index + len(selected)
        _sync_record_chunk_context_metadata(record)
        selected.append(record)
        seen[dup_key] = record
        used_tokens += estimated
        if record.source_type == "kb" and record.doc_id:
            per_doc[record.doc_id] += 1
            if record.doc_id in primary_doc_ids and config.max_chunks_per_doc and per_doc[record.doc_id] > config.max_chunks_per_doc:
                primary_doc_extra_selected_count += 1
        if record.source_type == "kb":
            per_source[source_key] += 1
    return EvidenceBundle(
        records=all_ordered,
        selected=selected,
        rejected=rejected,
        config=config,
        query_feature_flags={k: v for k, v in query_features.items() if isinstance(v, bool)},
        priority_boosted_evidence_ids=priority_boosted_ids,
        primary_document_ids=sorted(primary_doc_ids),
        per_document_effective_limits=effective_doc_limits,
        primary_doc_extra_selected_count=primary_doc_extra_selected_count,
        priority_boost_strength_counts=_priority_boost_strength_counts(priority_boosts_by_object_id.values()),
        low_relevance_preserved_count=low_relevance_preserved_count,
        low_relevance_bypass_reason_counts=dict(low_relevance_bypass_reason_counts),
        relevance_decision_counts=dict(relevance_decision_counts),
        per_source_base_limits=source_base_limits,
        per_source_effective_limits=source_effective_limits,
        source_cap_expansion_reasons=source_expansion_reasons,
    )


def apply_context_builder_to_kbinfos(
    kbinfos: dict[str, Any],
    config: EvidenceBundleConfig,
    *,
    start_citation_index: int = 0,
    source_type: str | None = None,
    retrieval_call_id: str | None = None,
    tool_call_id: str | None = None,
    query: str | None = None,
) -> ContextBuilderResult:
    if not config.enabled:
        return ContextBuilderResult(kbinfos=kbinfos, bundle=None, original_unchanged=True)
    records = evidence_records_from_kbinfos(kbinfos, source_type=source_type, retrieval_call_id=retrieval_call_id, tool_call_id=tool_call_id)
    bundle = build_context_bundle(records, config, start_citation_index=start_citation_index, query=query)
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


_YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|2100)\b")
_NUMERIC_VALUE_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_TEMPORAL_TERMS = {"as", "of", "time", "first", "official", "current", "recent", "year", "when", "date", "founded", "independence", "incorporated"}
_NUMERIC_TERMS = {"population", "census", "rank", "ranked", "most", "populous", "many", "total", "number", "count", "largest", "smallest"}
_BIO_TERMS = {"born", "birthplace", "died", "death", "title", "won", "winner", "award"}
_ATTRIBUTE_TERMS = {
    "area",
    "award",
    "birthplace",
    "born",
    "capital",
    "census",
    "currency",
    "death",
    "died",
    "established",
    "founded",
    "incorporated",
    "independence",
    "language",
    "official",
    "opened",
    "population",
    "title",
    "won",
    "winner",
}
_TABLE_RANK_TERMS = {"rank", "ranked", "ranking", "row", "rows", "table"}


def _query_features(query: str | None) -> dict[str, Any]:
    text = (query or "").lower()
    tokens = set(_TOKEN_RE.findall(text))
    years = set(_YEAR_RE.findall(text))
    numbers = {_normalize_number_token(match) for match in _NUMERIC_VALUE_RE.findall(text)}
    return {
        "text": text,
        "temporal": bool(years or tokens.intersection(_TEMPORAL_TERMS) or "as of" in text or "at the time" in text),
        "numeric_table": bool(tokens.intersection(_NUMERIC_TERMS) or "how many" in text),
        "biography": bool(tokens.intersection(_BIO_TERMS)),
        "years": years,
        "numbers": numbers,
        "numeric_claims": numbers.difference(years),
        "tokens": {t for t in tokens if len(t) > 2},
    }


def _record_relevance_signals(record: EvidenceRecord) -> dict[str, float | None]:
    return {
        "vector_similarity": _float_or_none(record.scores.get("vector_similarity")),
        "term_similarity": _float_or_none(record.scores.get("term_similarity")),
        "similarity": _float_or_none(record.scores.get("similarity")),
        "final_score": _float_or_none(record.scores.get("final_score")),
    }


def _score_provenance(record: EvidenceRecord, score_name: str) -> dict[str, Any]:
    provenance = record.scores.get("score_provenance")
    if not isinstance(provenance, dict):
        return {}
    value = provenance.get(score_name)
    return value if isinstance(value, dict) else {}


def _is_compatible_calibrated_score(record: EvidenceRecord, score_name: str) -> bool:
    provenance = _score_provenance(record, score_name)
    return bool(provenance.get("calibrated") is True and provenance.get("family") == _SCORE_FAMILY_BY_FIELD.get(score_name) and provenance.get("scale") in _COMPARABLE_SCORE_SCALES)


def _relevance_assessment(record: EvidenceRecord, config: EvidenceBundleConfig) -> tuple[bool, str]:
    if not config.relevance_filter_enabled:
        return False, "filter_disabled"
    signals = _record_relevance_signals(record)
    checks: list[bool] = []
    configured = (
        ("vector_similarity", config.min_vector_similarity),
        ("term_similarity", config.min_term_similarity),
    )
    for score_name, threshold in configured:
        if threshold is None or signals[score_name] is None:
            continue
        if _is_compatible_calibrated_score(record, score_name):
            checks.append(bool(signals[score_name] < threshold))
    if not checks:
        return False, "kept_unscored_or_uncalibrated"
    if all(checks):
        return True, "rejected_below_calibrated_threshold"
    return False, "kept_above_calibrated_threshold"


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_low_relevance(record: EvidenceRecord, config: EvidenceBundleConfig) -> bool:
    return _relevance_assessment(record, config)[0]


def _missing_configured_relevance_signal(record: EvidenceRecord, config: EvidenceBundleConfig) -> bool:
    signals = _record_relevance_signals(record)
    return bool(
        (config.min_vector_similarity is not None and (signals["vector_similarity"] is None or not _is_compatible_calibrated_score(record, "vector_similarity")))
        or (config.min_term_similarity is not None and (signals["term_similarity"] is None or not _is_compatible_calibrated_score(record, "term_similarity")))
    )


def _strong_low_relevance_evidence_reasons(record: EvidenceRecord, query_features: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    score_reasons = _record_score_support_reasons(record)
    if "strong_term_score" in score_reasons:
        reasons.append("strong_term_score")
    if _exact_title_match(record, query_features):
        reasons.append("exact_title_match")
    if _numeric_claim_matches(record, query_features):
        reasons.append("numeric_claim_match")

    attribute_terms = _attribute_specific_terms(record, query_features)
    has_stronger_anchor = bool(reasons or attribute_terms)
    if attribute_terms and (_exact_year_match(record, query_features) or _exact_title_match(record, query_features) or _numeric_claim_matches(record, query_features)):
        reasons.append("attribute_specific_match")
    if _exact_year_match(record, query_features) and has_stronger_anchor:
        reasons.append("exact_year_with_stronger_anchor")
    return _unique_reasons(reasons)


def _low_relevance_bypass_reasons(record: EvidenceRecord, rank: int, config: EvidenceBundleConfig, query_features: dict[str, Any]) -> list[str]:
    strong_reasons = _strong_low_relevance_evidence_reasons(record, query_features)
    reasons: list[str] = []
    if _has_protected_retrieval_lineage(record):
        reasons.append("protected_retrieval_lineage")
    if rank < max(0, config.preserve_top_ranked):
        if _missing_configured_relevance_signal(record, config):
            reasons.append("top_ranked_with_missing_relevance_signal")
        if strong_reasons:
            reasons.append("top_ranked_with_strong_evidence")
    reasons.extend(strong_reasons)
    return _unique_reasons(reasons)


def _has_protected_retrieval_lineage(record: EvidenceRecord) -> bool:
    chunk = record.chunk if isinstance(record.chunk, dict) else {}
    compilation = record.metadata.get("compilation")
    if isinstance(compilation, dict) and compilation.get("baseline_protected") is True:
        return True
    fusion = chunk.get("_ragflow_fusion")
    if isinstance(fusion, dict):
        lane_ranks = fusion.get("lane_ranks")
        if isinstance(lane_ranks, dict) and any(isinstance(rank, int) and rank == 1 for rank in lane_ranks.values()):
            return True
        if fusion.get("lane_reservations"):
            return True
    lineage_rank = record.metadata.get("lineage_rank")
    agentic = record.metadata.get("agentic_retrieval")
    if isinstance(agentic, dict) and agentic.get("lineage_rank") is not None:
        lineage_rank = agentic.get("lineage_rank")
    return isinstance(lineage_rank, int) and lineage_rank == 1


def _query_token_overlap(record: EvidenceRecord, query_features: dict[str, Any]) -> bool:
    tokens = query_features.get("tokens") or set()
    if not tokens:
        return False
    content_tokens = set(_TOKEN_RE.findall((record.content or "").lower()))
    return bool(tokens.intersection(content_tokens))


def _priority_boost_score(priority_boost: dict[str, Any] | None) -> int:
    if not priority_boost:
        return 0
    try:
        return int(priority_boost.get("score") or 0)
    except (TypeError, ValueError):
        return 0


def _priority_boost_strength_counts(priority_boosts: Any) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for priority_boost in priority_boosts:
        if not isinstance(priority_boost, dict):
            continue
        strength = str(priority_boost.get("strength") or "none")
        if strength != "none":
            counts[strength] += 1
    return dict(sorted(counts.items()))


def _record_text(record: EvidenceRecord) -> str:
    return f"{record.doc_title or ''}\n{record.content or ''}".lower()


def _record_tokens(record: EvidenceRecord) -> set[str]:
    return set(_TOKEN_RE.findall(_record_text(record)))


def _record_numbers(record: EvidenceRecord) -> set[str]:
    return {_normalize_number_token(match) for match in _NUMERIC_VALUE_RE.findall(_record_text(record))}


def _normalize_number_token(value: Any) -> str:
    return str(value).replace(",", "").strip().lower()


def _attribute_specific_terms(record: EvidenceRecord, query_features: dict[str, Any]) -> set[str]:
    query_terms = (query_features.get("tokens") or set()).intersection(_ATTRIBUTE_TERMS)
    if not query_terms:
        return set()
    return query_terms.intersection(_record_tokens(record))


def _table_or_rank_indicator_terms(record: EvidenceRecord, query_features: dict[str, Any]) -> set[str]:
    if not query_features.get("numeric_table"):
        return set()
    return (query_features.get("tokens") or set()).union(_TABLE_RANK_TERMS).intersection(_TABLE_RANK_TERMS).intersection(_record_tokens(record))


def _numeric_claim_matches(record: EvidenceRecord, query_features: dict[str, Any]) -> set[str]:
    query_numbers = query_features.get("numeric_claims") or set()
    if not query_numbers:
        return set()
    return query_numbers.intersection(_record_numbers(record))


def _record_score_support_reasons(record: EvidenceRecord) -> list[str]:
    signals = _record_relevance_signals(record)
    reasons: list[str] = []
    vector_similarity = signals.get("vector_similarity")
    if vector_similarity is not None and _is_compatible_calibrated_score(record, "vector_similarity") and vector_similarity >= _STRONG_RELEVANCE_THRESHOLD:
        reasons.append("strong_dense_score")
    term_similarity = signals.get("term_similarity")
    if term_similarity is not None and _is_compatible_calibrated_score(record, "term_similarity") and term_similarity >= _STRONG_RELEVANCE_THRESHOLD:
        reasons.append("strong_term_score")
    return reasons


def _record_anchor_support_reasons(record: EvidenceRecord, query_features: dict[str, Any]) -> list[str]:
    reasons = _record_score_support_reasons(record)
    if _exact_title_match(record, query_features):
        reasons.append("exact_title_match")
    if (
        _exact_year_match(record, query_features)
        and (_attribute_specific_terms(record, query_features) or _table_or_rank_indicator_terms(record, query_features))
        and not _has_calibrated_weak_anchor_score(record)
    ):
        reasons.append("typed_query_anchor")
    return _unique_reasons(reasons)


def _has_calibrated_weak_anchor_score(record: EvidenceRecord) -> bool:
    signals = _record_relevance_signals(record)
    return any(
        signals.get(score_name) is not None and _is_compatible_calibrated_score(record, score_name) and (signals.get(score_name) or 0.0) < _STRONG_RELEVANCE_THRESHOLD
        for score_name in _SCORE_FAMILY_BY_FIELD
    )


def _exact_year_match(record: EvidenceRecord, query_features: dict[str, Any]) -> bool:
    years = query_features.get("years") or set()
    if not years:
        return False
    text = f"{record.doc_title or ''}\n{record.content or ''}".lower()
    return any(re.search(rf"\b{re.escape(str(year))}\b", text) for year in years)


def _exact_title_match(record: EvidenceRecord, query_features: dict[str, Any]) -> bool:
    title = _normalized_doc_title(record.doc_title)
    if not title:
        return False
    title_tokens = {token for token in _TOKEN_RE.findall(title) if len(token) > 2}
    if not title_tokens:
        return False
    query_text = str(query_features.get("text") or "")
    if len(title) >= 4 and re.search(rf"(?<!\w){re.escape(title)}(?!\w)", query_text):
        return True
    query_tokens = query_features.get("tokens") or set()
    return title_tokens.issubset(query_tokens)


def _unique_reasons(reasons: list[str]) -> list[str]:
    unique: list[str] = []
    for reason in reasons:
        if reason and reason not in unique:
            unique.append(reason)
    return unique


def _primary_doc_ids(primary_doc_reasons: dict[str, list[dict[str, Any]]]) -> set[str]:
    return {str(doc_id) for doc_id, reasons in primary_doc_reasons.items() if reasons}


def _primary_doc_candidate_reasons(records: list[EvidenceRecord], config: EvidenceBundleConfig, query_features: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    if not (config.max_chunks_per_doc and config.primary_doc_extra_chunks > 0):
        return {}
    reasons: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    required_rank_hit_count = max(2, config.primary_doc_min_rank_hits)
    top_window_size = max(required_rank_hit_count * 3, config.preserve_top_ranked, 6)
    top_window = records[:top_window_size]
    supported_top_hits = [record for record in top_window if record.source_type == "kb" and record.doc_id and _record_anchor_support_reasons(record, query_features)]
    counts = Counter(record.doc_id for record in supported_top_hits)
    for doc_id, count in counts.items():
        if count >= required_rank_hit_count:
            reasons[str(doc_id)].append(
                {
                    "reason": "repeated_top_rank_hits",
                    "rank_hit_count": count,
                    "required_rank_hit_count": required_rank_hit_count,
                    "top_window_size": top_window_size,
                    "support_reasons": ",".join(
                        _unique_reasons([reason for record in supported_top_hits if record.doc_id == doc_id for reason in _record_anchor_support_reasons(record, query_features)])
                    ),
                }
            )
    for record in records:
        if record.source_type != "kb" or not record.doc_id:
            continue
        if _exact_title_match(record, query_features):
            reasons[str(record.doc_id)].append({"reason": "exact_title_match", "evidence_id": record.evidence_id})
    return {doc_id: _dedupe_reason_details(reason_details) for doc_id, reason_details in reasons.items()}


def _primary_source_candidate_reasons(
    records: list[EvidenceRecord],
    config: EvidenceBundleConfig,
    query_features: dict[str, Any],
    primary_doc_reasons: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    if config.primary_doc_extra_chunks <= 0:
        return {}
    reasons: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    required_rank_hit_count = max(2, config.primary_doc_min_rank_hits)
    top_window_size = max(required_rank_hit_count * 3, config.preserve_top_ranked, 6)
    top_window = records[:top_window_size]
    supported_top_hits = [record for record in top_window if record.source_type == "kb" and _record_anchor_support_reasons(record, query_features)]
    source_counts = Counter(_source_group_key(record) for record in supported_top_hits)
    for source_key, count in source_counts.items():
        if count >= required_rank_hit_count:
            reasons[str(source_key)].append(
                {
                    "reason": "repeated_top_rank_hits",
                    "rank_hit_count": count,
                    "required_rank_hit_count": required_rank_hit_count,
                    "top_window_size": top_window_size,
                    "support_reasons": ",".join(
                        _unique_reasons([reason for record in supported_top_hits if _source_group_key(record) == source_key for reason in _record_anchor_support_reasons(record, query_features)])
                    ),
                }
            )
    for record in records:
        if record.source_type != "kb":
            continue
        source_key = _source_group_key(record)
        if record.doc_id:
            for detail in primary_doc_reasons.get(str(record.doc_id), []):
                reason = detail.get("reason") if isinstance(detail, dict) else None
                if reason in {"repeated_top_rank_hits", "exact_title_match"}:
                    reasons[str(source_key)].append({"reason": str(reason), "evidence_id": record.evidence_id})
        if _exact_title_match(record, query_features):
            reasons[str(source_key)].append({"reason": "exact_title_match", "evidence_id": record.evidence_id})
    return {source_key: _dedupe_reason_details(reason_details) for source_key, reason_details in reasons.items()}


def _doc_limit_for_record(record: EvidenceRecord, config: EvidenceBundleConfig, primary_doc_ids: set[str]) -> int:
    base = config.max_chunks_per_doc or 0
    if not base:
        return 0
    if record.doc_id in primary_doc_ids:
        return _expanded_primary_cap(base, config)
    return base


def _effective_doc_limits(records: list[EvidenceRecord], config: EvidenceBundleConfig, primary_doc_ids: set[str]) -> dict[str, int]:
    if not config.max_chunks_per_doc:
        return {}
    doc_ids = {r.doc_id for r in records if r.source_type == "kb" and r.doc_id}
    limits = {}
    for doc_id in doc_ids:
        limit = config.max_chunks_per_doc
        if doc_id in primary_doc_ids:
            limit = _expanded_primary_cap(config.max_chunks_per_doc, config)
        limits[str(doc_id)] = limit
    return limits


def _expanded_primary_cap(base: int, config: EvidenceBundleConfig) -> int:
    expanded = base + max(0, config.primary_doc_extra_chunks)
    if config.primary_doc_max_chunks is not None:
        expanded = min(expanded, config.primary_doc_max_chunks)
    return max(base, expanded)


def _source_limit_for_record(
    record: EvidenceRecord,
    config: EvidenceBundleConfig,
    primary_doc_ids: set[str],
    primary_doc_reasons: dict[str, list[dict[str, Any]]],
    primary_source_reasons: dict[str, list[dict[str, Any]]],
    base_source_limit: int | None,
    query_features: dict[str, Any],
) -> int | None:
    if base_source_limit is None:
        return None
    reasons = _source_cap_expansion_reasons(record, primary_doc_ids, primary_doc_reasons, primary_source_reasons, query_features)
    if not reasons or config.primary_doc_extra_chunks <= 0:
        return base_source_limit
    return _expanded_primary_cap(base_source_limit, config)


def _effective_source_limits(
    records: list[EvidenceRecord],
    config: EvidenceBundleConfig,
    primary_doc_ids: set[str],
    primary_doc_reasons: dict[str, list[dict[str, Any]]],
    primary_source_reasons: dict[str, list[dict[str, Any]]],
    base_source_limit: int | None,
    query_features: dict[str, Any],
) -> tuple[dict[str, int], dict[str, int], dict[str, list[str]]]:
    if base_source_limit is None:
        return {}, {}, {}
    base_limits: dict[str, int] = {}
    effective_limits: dict[str, int] = {}
    expansion_reasons: defaultdict[str, list[str]] = defaultdict(list)
    for record in records:
        if record.source_type != "kb":
            continue
        source_key = _source_group_key(record)
        base_limits[source_key] = base_source_limit
        effective_limits[source_key] = max(
            effective_limits.get(source_key, base_source_limit),
            _source_limit_for_record(
                record,
                config,
                primary_doc_ids,
                primary_doc_reasons,
                primary_source_reasons,
                base_source_limit,
                query_features,
            )
            or base_source_limit,
        )
        expansion_reasons[source_key].extend(
            _source_cap_expansion_reasons(
                record,
                primary_doc_ids,
                primary_doc_reasons,
                primary_source_reasons,
                query_features,
            )
        )
    return (
        base_limits,
        effective_limits,
        {source_key: _unique_reasons(reasons) for source_key, reasons in expansion_reasons.items() if effective_limits.get(source_key, base_source_limit) > base_source_limit},
    )


def _source_cap_expansion_reasons(
    record: EvidenceRecord,
    primary_doc_ids: set[str],
    primary_doc_reasons: dict[str, list[dict[str, Any]]],
    primary_source_reasons: dict[str, list[dict[str, Any]]],
    query_features: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    doc_id = str(record.doc_id) if record.doc_id else ""
    if doc_id and doc_id in primary_doc_ids:
        reasons.extend(
            str(reason_detail.get("reason"))
            for reason_detail in primary_doc_reasons.get(doc_id, [])
            if isinstance(reason_detail, dict) and reason_detail.get("reason") in {"repeated_top_rank_hits", "exact_title_match"}
        )
    source_key = _source_group_key(record)
    reasons.extend(
        str(reason_detail.get("reason"))
        for reason_detail in primary_source_reasons.get(source_key, [])
        if isinstance(reason_detail, dict) and reason_detail.get("reason") in {"repeated_top_rank_hits", "exact_title_match"}
    )
    if _exact_title_match(record, query_features):
        reasons.append("exact_title_match")
    return _unique_reasons(reasons)


def _dedupe_reason_details(reason_details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str | None]] = set()
    deduped: list[dict[str, Any]] = []
    for detail in reason_details:
        reason = str(detail.get("reason") or "")
        evidence_id = str(detail.get("evidence_id")) if detail.get("evidence_id") is not None else None
        key = (reason, evidence_id)
        if not reason or key in seen:
            continue
        seen.add(key)
        deduped.append(detail)
    return deduped


def _priority_boost_details(record: EvidenceRecord, query_features: dict[str, Any]) -> dict[str, Any]:
    if not query_features:
        return {"score": 0, "strength": "none"}
    score = 0
    strong_reason_count = 0
    weak_reason_count = 0
    has_strong_typed_anchor = False

    if _exact_title_match(record, query_features):
        score += 6
        has_strong_typed_anchor = True
        strong_reason_count += 1
    if _exact_year_match(record, query_features):
        score += 4
        has_strong_typed_anchor = True
        strong_reason_count += 1

    numeric_matches = _numeric_claim_matches(record, query_features)
    if numeric_matches:
        score += 4
        has_strong_typed_anchor = True
        strong_reason_count += 1

    attribute_terms = _attribute_specific_terms(record, query_features)
    if attribute_terms and has_strong_typed_anchor:
        score += 2
        strong_reason_count += 1
    elif attribute_terms:
        weak_reason_count += 1

    table_rank_terms = _table_or_rank_indicator_terms(record, query_features)
    if table_rank_terms:
        score += 1
        weak_reason_count += 1

    if _query_token_overlap(record, query_features):
        weak_reason_count += 1

    strength = "strong" if strong_reason_count else "weak" if weak_reason_count else "none"
    return {"score": score, "strength": strength}


def _reject(record: EvidenceRecord, reason: str, rejected: list[EvidenceRecord]) -> None:
    record.selected_for_context = False
    record.rejection_reason = reason
    record.citation_index = None
    _sync_record_chunk_context_metadata(record)
    rejected.append(record)


def _candidate_trace_observations(
    records: list[EvidenceRecord],
    selected: list[EvidenceRecord],
    rejected: list[EvidenceRecord],
) -> list[dict[str, Any]]:
    """Return a bounded decision projection without serializing chunk text."""
    selected_ids = {id(record) for record in selected}
    selected_records = list(selected)[:_MAX_TRACE_CANDIDATE_OBSERVATIONS]
    remaining = _MAX_TRACE_CANDIDATE_OBSERVATIONS - len(selected_records)
    protected_rejected = [record for record in rejected if record.protection_reason][:remaining]
    remaining -= len(protected_rejected)
    rejected_records = sorted(
        (record for record in rejected if record not in protected_rejected),
        key=lambda record: (record.rank, record.evidence_id),
    )[: max(0, remaining)]
    included: list[EvidenceRecord] = []
    for record in [*selected_records, *protected_rejected, *rejected_records]:
        if record not in included:
            included.append(record)

    def score_provenance(record: EvidenceRecord) -> dict[str, Any]:
        value = record.scores.get("score_provenance", {})
        return value if isinstance(value, dict) else {}

    observations = []
    for record in included:
        observations.append(
            {
                "evidence_id": record.evidence_id,
                "source_id": hashlib.sha256(str(record.source_uri or record.doc_title or record.source_type).encode()).hexdigest()[:16],
                "document_id": record.doc_id,
                "retrieval_call_id": record.retrieval_call_id,
                "rank": record.rank,
                "scores": {key: value for key, value in record.scores.items() if key != "score_provenance"},
                "score_provenance": score_provenance(record),
                "lineage": {
                    key: record.metadata.get(key)
                    for key in ("plan_id", "facet_id", "subquery_id", "iteration_id", "followup_id", "lineage_rank", "merge_rank", "retrieval_variant")
                    if record.metadata.get(key) is not None
                },
                "protected": bool(record.protection_reason),
                "protection_reason": record.protection_reason,
                "selected": id(record) in selected_ids,
                "selection_reason": record.selection_reason,
                "rejection_reason": record.rejection_reason,
            }
        )
    return observations


def _sync_record_chunk_context_metadata(record: EvidenceRecord) -> None:
    if not isinstance(record.chunk, dict):
        return
    record.chunk["selected_for_context"] = record.selected_for_context
    record.chunk["rejection_reason"] = record.rejection_reason
    if record.citation_index is not None:
        record.chunk["citation_index"] = record.citation_index
    raw_agentic = record.metadata.get("agentic_retrieval")
    agentic: dict[str, Any] = dict(raw_agentic) if isinstance(raw_agentic, dict) else {}
    for key in ("plan_id", "facet_id", "subquery_id", "iteration_id", "followup_id", "retrieval_call_id", "lineage_rank", "merge_rank", "retrieval_variant"):
        if key in record.metadata and record.metadata.get(key) is not None:
            record.chunk[key] = record.metadata.get(key)
            agentic[key] = record.metadata.get(key)
    if agentic:
        agentic["selected_for_context"] = record.selected_for_context
        agentic["rejection_reason"] = record.rejection_reason
        if record.citation_index is not None:
            agentic["citation_index"] = record.citation_index
        record.chunk["_ragflow_agentic_retrieval"] = agentic


def _agentic_lineage_summary(records: list[EvidenceRecord], selected: list[EvidenceRecord], rejected: list[EvidenceRecord]) -> list[dict[str, Any]]:
    def lineage_value(record: EvidenceRecord, key: str) -> Any:
        agentic = record.metadata.get("agentic_retrieval")
        if isinstance(agentic, dict) and agentic.get(key) is not None:
            return agentic.get(key)
        return record.metadata.get(key)

    lineage: list[dict[str, Any]] = []
    selected_ids = {id(record) for record in selected}
    for record in records[:_MAX_TRACE_IDS]:
        if not lineage_value(record, "plan_id"):
            continue
        selected_for_context = id(record) in selected_ids
        item = {
            "evidence_id": record.evidence_id,
            "chunk_id": record.chunk_id,
            "plan_id": lineage_value(record, "plan_id"),
            "subquery_id": lineage_value(record, "subquery_id"),
            "facet_id": lineage_value(record, "facet_id"),
            "retrieval_call_id": record.retrieval_call_id or lineage_value(record, "retrieval_call_id"),
            "lineage_rank": lineage_value(record, "lineage_rank"),
            "merge_rank": lineage_value(record, "merge_rank"),
            "selected_for_context": selected_for_context,
            "rejection_reason": None if selected_for_context else record.rejection_reason,
        }
        for key in ("iteration_id", "followup_id"):
            if lineage_value(record, key) is not None:
                item[key] = lineage_value(record, key)
        lineage.append(item)
    return lineage


def _compilation_survival_summary(records: list[EvidenceRecord], selected: list[EvidenceRecord]) -> list[dict[str, Any]]:
    selected_ids = {id(record) for record in selected}
    items: list[dict[str, Any]] = []
    for record in records:
        compilation = record.metadata.get("compilation")
        if not isinstance(compilation, dict):
            continue
        items.append(
            {
                "chunk_id": record.chunk_id,
                "doc_id": record.doc_id,
                "selected_for_context": id(record) in selected_ids,
                "rejection_reason": None if id(record) in selected_ids else record.rejection_reason,
                "baseline_protected": bool(compilation.get("baseline_protected") or record.protection_reason),
                "route_occurrences": compilation.get("route_occurrences") or [],
            }
        )
        if len(items) >= _MAX_TRACE_IDS:
            break
    return items


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
    ordered: list[EvidenceRecord] = []
    while any(buckets.values()):
        for source_key in source_order:
            if buckets[source_key]:
                ordered.append(buckets[source_key].pop(0))
    return ordered


def _preserve_protected_lineage_prefix(
    ranked: list[EvidenceRecord],
    diversified: list[EvidenceRecord],
    max_chunks: int | None,
) -> list[EvidenceRecord]:
    """Keep protected members of the bounded rank prefix inside that prefix."""
    if not max_chunks or max_chunks <= 0 or len(diversified) <= max_chunks:
        return diversified
    protected = [record for record in ranked[:max_chunks] if _has_protected_retrieval_lineage(record)]
    if not protected:
        return diversified
    ordered = list(diversified)
    for record in protected:
        current = ordered.index(record)
        if current < max_chunks:
            continue
        replacement = next(
            (index for index in range(max_chunks - 1, -1, -1) if not _has_protected_retrieval_lineage(ordered[index])),
            None,
        )
        if replacement is None:
            break
        ordered[replacement], ordered[current] = ordered[current], ordered[replacement]
    return ordered


def _source_limit(config: EvidenceBundleConfig) -> int | None:
    limits: list[int] = []
    if config.max_chunks_per_source:
        limits.append(config.max_chunks_per_source)
    if config.max_chunks and config.max_source_fraction:
        limits.append(max(1, int(config.max_chunks * config.max_source_fraction + 0.999999)))
    return min(limits) if limits else None


def _score_provenance_counts(records: list[EvidenceRecord]) -> dict[str, Any]:
    families: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    calibrated = 0
    uncalibrated = 0
    for record in records:
        provenance = record.scores.get("score_provenance")
        if not isinstance(provenance, dict):
            continue
        for value in provenance.values():
            if not isinstance(value, dict):
                continue
            families[str(value.get("family") or "unknown")] += 1
            sources[str(value.get("source") or "unknown")] += 1
            if value.get("calibrated") is True:
                calibrated += 1
            else:
                uncalibrated += 1
    return {
        "families": dict(sorted(families.items())),
        "sources": dict(sorted(sources.items())),
        "calibrated": calibrated,
        "uncalibrated": uncalibrated,
    }


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
