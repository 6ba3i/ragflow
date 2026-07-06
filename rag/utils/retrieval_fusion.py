#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#
"""Phase 2 retrieval fusion helpers.

This module is deliberately pure/mostly pure so sparse+dense fusion can be
unit-tested without a doc-store.  Production integration lives in
``rag.nlp.search.Dealer``; this file owns config parsing, candidate identity,
score normalization, deterministic fusion, lightweight diversity metrics, and
trace-safe summaries.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal, cast

FusionStrategy = Literal["current_weighted", "linear", "rrf"]
LaneName = Literal["sparse", "dense"]

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_SUPPORTED_STRATEGIES = {"current_weighted", "linear", "rrf"}
_MAX_TRACE_CANDIDATES = 16
_MAX_TRACE_LIST = 16
_MAX_BODY_PREVIEW = 160


@dataclass(frozen=True)
class FusionConfig:
    enabled: bool = False
    strategy: FusionStrategy = "rrf"
    window: int = 60
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    doc_level: bool = False
    diversity_enabled: bool = False

    @classmethod
    def from_env(cls, overrides: dict[str, Any] | None = None) -> "FusionConfig":
        overrides = overrides or {}
        enabled = _bool_setting(overrides.get("fusion_enabled"), "FUSION_ENABLED", False, legacy_env="RAGFLOW_FUSION_ENABLED")
        strategy = str(_setting(overrides.get("fusion_strategy"), "FUSION_STRATEGY", "rrf", legacy_env="RAGFLOW_FUSION_STRATEGY") or "rrf").strip().lower()
        if strategy not in _SUPPORTED_STRATEGIES:
            strategy = "rrf"
        return cls(
            enabled=enabled,
            strategy=strategy,  # type: ignore[arg-type]
            window=_int_setting(overrides.get("fusion_window"), "FUSION_WINDOW", 60, legacy_env="RAGFLOW_FUSION_WINDOW"),
            rrf_k=_int_setting(overrides.get("fusion_rrf_k"), "FUSION_RRF_K", 60, legacy_env="RAGFLOW_FUSION_RRF_K"),
            dense_weight=_float_setting(overrides.get("fusion_dense_weight"), "FUSION_DENSE_WEIGHT", 1.0, legacy_env="RAGFLOW_FUSION_DENSE_WEIGHT"),
            sparse_weight=_float_setting(overrides.get("fusion_sparse_weight"), "FUSION_SPARSE_WEIGHT", 1.0, legacy_env="RAGFLOW_FUSION_SPARSE_WEIGHT"),
            doc_level=_bool_setting(overrides.get("fusion_doc_level"), "FUSION_DOC_LEVEL", False, legacy_env="RAGFLOW_FUSION_DOC_LEVEL"),
            diversity_enabled=_bool_setting(overrides.get("fusion_diversity_enabled"), "FUSION_DIVERSITY_ENABLED", False, legacy_env="RAGFLOW_FUSION_DIVERSITY_ENABLED"),
        )

    def lane_weight(self, lane: str) -> float:
        return self.sparse_weight if normalize_lane_name(lane) == "sparse" else self.dense_weight

    def to_trace_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalLaneCandidate:
    candidate_id: str
    identity_basis: str
    lane: LaneName
    requested_lane: str
    effective_lane: str
    rank: int
    raw_score: float = 0.0
    normalized_score: float = 0.0
    sparse_score: float | None = None
    dense_score: float | None = None
    chunk_id: str | None = None
    doc_id: str | None = None
    doc_title: str | None = None
    content: str = ""
    section_page_order: dict[str, Any] = field(default_factory=dict)
    original_index: int = 0
    chunk: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    provenance: dict[str, Any] = field(default_factory=dict)
    fallback_note: str | None = None

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "identity_basis": self.identity_basis,
            "lane": self.lane,
            "requested_lane": self.requested_lane,
            "effective_lane": self.effective_lane,
            "rank": self.rank,
            "score": _finite_or_zero(self.raw_score),
            "normalized_score": _finite_or_zero(self.normalized_score),
            "sparse_score": _safe_number(self.sparse_score),
            "dense_score": _safe_number(self.dense_score),
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "fallback_note": self.fallback_note,
        }


@dataclass
class FusedCandidate:
    candidate_id: str
    identity_basis: str
    fused_rank: int
    fused_score: float
    best_lane_rank: int
    lane_ranks: dict[str, int]
    lane_scores: dict[str, float]
    normalized_lane_scores: dict[str, float]
    chunk: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    chunks_by_lane: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False, compare=False)
    doc_id: str | None = None
    doc_title: str | None = None
    section_page_order: dict[str, Any] = field(default_factory=dict)
    original_index: int = 0
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "chunk_id": self.chunk.get("chunk_id") or self.chunk.get("id"),
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "fused_rank": self.fused_rank,
            "fused_score": _finite_or_zero(self.fused_score),
            "best_lane_rank": self.best_lane_rank,
            "lane_ranks": dict(sorted(self.lane_ranks.items())),
            "lane_scores": {k: _finite_or_zero(v) for k, v in sorted(self.lane_scores.items())},
            "normalized_lane_scores": {k: _finite_or_zero(v) for k, v in sorted(self.normalized_lane_scores.items())},
        }


@dataclass
class FusionResult:
    candidates: list[FusedCandidate]
    metrics: dict[str, Any]
    diagnostics: dict[str, Any]
    fallback_notes: list[str] = field(default_factory=list)

    def page(self, *, offset: int, limit: int) -> list[FusedCandidate]:
        return self.candidates[offset : offset + limit]

    def to_kbinfos(self, original_kbinfos: dict[str, Any] | None = None, *, offset: int = 0, limit: int | None = None) -> dict[str, Any]:
        original_kbinfos = original_kbinfos or {}
        selected = self.candidates[offset : offset + limit if limit is not None else None]
        chunks = [candidate.chunk for candidate in selected]
        doc_aggs = _doc_aggs_from_chunks(chunks, original_kbinfos.get("doc_aggs"))
        out = dict(original_kbinfos)
        out["chunks"] = chunks
        out["doc_aggs"] = doc_aggs
        out["total"] = len(self.candidates)
        out["diagnostics"] = dict(out.get("diagnostics") or {})
        out["diagnostics"]["fusion"] = self.diagnostics
        out["fusion"] = self.diagnostics
        return out


def normalize_lane_name(lane: Any) -> LaneName:
    value = str(lane or "").strip().lower()
    if value in {"dense", "vector", "embedding", "semantic"}:
        return "dense"
    return "sparse"


def lane_candidates_from_chunks(
    chunks: Iterable[dict[str, Any]],
    *,
    lane: str,
    requested_lane: str | None = None,
    effective_lane: str | None = None,
    fallback_note: str | None = None,
) -> list[RetrievalLaneCandidate]:
    lane_name = normalize_lane_name(lane)
    candidates: list[RetrievalLaneCandidate] = []
    for original_index, chunk in enumerate(chunks or []):
        if not isinstance(chunk, dict):
            continue
        score = _score_for_lane(chunk, lane_name)
        candidate_id, basis = stable_candidate_identity(chunk)
        section = _section_page_order(chunk)
        candidates.append(
            RetrievalLaneCandidate(
                candidate_id=candidate_id,
                identity_basis=basis,
                lane=lane_name,
                requested_lane=requested_lane or lane_name,
                effective_lane=effective_lane or lane_name,
                rank=original_index + 1,
                raw_score=score,
                sparse_score=_safe_number(chunk.get("term_similarity")),
                dense_score=_safe_number(chunk.get("vector_similarity")),
                chunk_id=chunk.get("chunk_id") or chunk.get("id"),
                doc_id=chunk.get("doc_id"),
                doc_title=chunk.get("docnm_kwd") or chunk.get("document_name") or chunk.get("title"),
                content=_content_from_chunk(chunk),
                section_page_order=section,
                original_index=original_index,
                chunk=chunk,
                provenance=dict(chunk.get("_ragflow_fusion") or {}),
                fallback_note=fallback_note,
            )
        )
    return normalize_candidate_scores(candidates)


def normalize_candidate_scores(candidates: list[RetrievalLaneCandidate]) -> list[RetrievalLaneCandidate]:
    by_lane: dict[str, list[RetrievalLaneCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_lane[candidate.lane].append(candidate)
    for lane_candidates in by_lane.values():
        finite = [_finite_or_none(c.raw_score) for c in lane_candidates]
        finite = [v for v in finite if v is not None]
        if not finite:
            for candidate in lane_candidates:
                candidate.normalized_score = 0.0
            continue
        lo, hi = min(finite), max(finite)
        if hi == lo:
            for candidate in lane_candidates:
                candidate.normalized_score = 1.0 if _finite_or_none(candidate.raw_score) is not None else 0.0
            continue
        span = hi - lo
        for candidate in lane_candidates:
            value = _finite_or_none(candidate.raw_score)
            candidate.normalized_score = 0.0 if value is None else (value - lo) / span
    return candidates


def dedupe_lane_candidates(candidates: list[RetrievalLaneCandidate]) -> list[RetrievalLaneCandidate]:
    seen: set[tuple[str, str]] = set()
    out: list[RetrievalLaneCandidate] = []
    for candidate in sorted(candidates, key=lambda c: (c.lane, c.rank, c.original_index, c.candidate_id)):
        key = (candidate.lane, candidate.candidate_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def linear_fusion(candidates: list[RetrievalLaneCandidate], config: FusionConfig) -> list[FusedCandidate]:
    grouped = _group_candidates(candidates)
    fused = [_make_fused_candidate(group, _linear_score(group, config)) for group in grouped.values()]
    return _rank_fused(fused)


def rrf_fusion(candidates: list[RetrievalLaneCandidate], config: FusionConfig) -> list[FusedCandidate]:
    grouped = _group_candidates(candidates)
    fused = [_make_fused_candidate(group, _rrf_score(group, config)) for group in grouped.values()]
    return _rank_fused(fused)


def current_weighted_fusion(candidates: list[RetrievalLaneCandidate], _config: FusionConfig) -> list[FusedCandidate]:
    grouped = _group_candidates(candidates)
    fused = [_make_fused_candidate(group, max(_finite_or_zero(c.raw_score) for c in group)) for group in grouped.values()]
    return _rank_fused(fused)


def fuse_retrieval_lanes(
    lanes: dict[str, Iterable[dict[str, Any]] | list[RetrievalLaneCandidate]],
    config: FusionConfig,
    *,
    fallback_notes: list[str] | None = None,
) -> FusionResult:
    started_notes = list(fallback_notes or [])
    candidates: list[RetrievalLaneCandidate] = []
    lane_counts: dict[str, int] = {}
    for lane, items in lanes.items():
        items_list = list(items or [])
        if all(isinstance(item, RetrievalLaneCandidate) for item in items_list):
            lane_candidates = cast(list[RetrievalLaneCandidate], items_list)
        else:
            chunk_items = [item for item in items_list if isinstance(item, dict)]
            lane_candidates = lane_candidates_from_chunks(chunk_items, lane=lane)
        candidates.extend(lane_candidates)
        lane_counts[normalize_lane_name(lane)] = len(lane_candidates)
        for candidate in lane_candidates:
            if candidate.fallback_note and candidate.fallback_note not in started_notes:
                started_notes.append(candidate.fallback_note)
    candidates = dedupe_lane_candidates(normalize_candidate_scores(candidates))

    fusion_level = "chunk"
    if config.doc_level:
        if any(not c.doc_id for c in candidates):
            fusion_level = "chunk"
            started_notes.append("doc_level_fallback:missing_doc_id")
        else:
            fusion_level = "doc"
            candidates = _doc_level_representatives(candidates, config)

    if config.strategy == "linear":
        fused = linear_fusion(candidates, config)
    elif config.strategy == "current_weighted":
        fused = current_weighted_fusion(candidates, config)
    else:
        fused = rrf_fusion(candidates, config)

    if config.diversity_enabled:
        fused = _apply_conservative_diversity(fused)
        fused = _rank_fused(fused)

    for candidate in fused:
        attach_fusion_metadata(candidate, fallback_notes=started_notes)
    metrics = compute_fusion_metrics(candidates, fused)
    diagnostics = make_fusion_trace_summary(
        config=config,
        fusion_level=fusion_level,
        lane_counts=lane_counts,
        fused=fused,
        metrics=metrics,
        fallback_notes=started_notes,
    )
    return FusionResult(candidates=fused, metrics=metrics, diagnostics=diagnostics, fallback_notes=started_notes)


def compute_fusion_metrics(candidates: list[RetrievalLaneCandidate], fused: list[FusedCandidate] | None = None) -> dict[str, Any]:
    fused = fused or []
    by_lane: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        by_lane[candidate.lane].add(candidate.candidate_id)
    sparse = by_lane.get("sparse", set())
    dense = by_lane.get("dense", set())
    union = sparse | dense
    overlap = (len(sparse & dense) / len(union)) if union else 0.0
    candidate_ids = [c.candidate_id for c in candidates]
    duplicate_count = len(candidate_ids) - len(set(candidate_ids))
    chunks = [fc.chunk for fc in fused] if fused else [c.chunk for c in candidates]
    doc_ids = [c.get("doc_id") for c in chunks if isinstance(c, dict) and c.get("doc_id")]
    sections = [_section_key(_section_page_order(c)) for c in chunks if isinstance(c, dict)]
    sources = [str(c.get("_ragflow_source_type") or c.get("source_type") or "kb") for c in chunks if isinstance(c, dict)]
    return {
        "sparse_dense_overlap": round(overlap, 6),
        "duplicate_count": duplicate_count,
        "duplicate_ratio": round((duplicate_count / len(candidate_ids)) if candidate_ids else 0.0, 6),
        "unique_document_count": len(set(doc_ids)),
        "document_diversity": len(set(doc_ids)),
        "section_diversity": len({s for s in sections if s}),
        "source_diversity": len(set(sources)),
        "candidate_count": len(candidates),
        "fused_candidate_count": len(fused) if fused else len(set(candidate_ids)),
    }


def make_fusion_trace_summary(
    *,
    config: FusionConfig,
    fusion_level: str,
    lane_counts: dict[str, int],
    fused: list[FusedCandidate],
    metrics: dict[str, Any],
    fallback_notes: list[str] | None = None,
    latency_ms: dict[str, float] | None = None,
) -> dict[str, Any]:
    return _sanitize_trace(
        {
            "enabled": config.enabled,
            "strategy": config.strategy,
            "fusion_level": fusion_level,
            "window": config.window,
            "rrf_k": config.rrf_k,
            "dense_weight": config.dense_weight,
            "sparse_weight": config.sparse_weight,
            "doc_level": config.doc_level,
            "diversity_enabled": config.diversity_enabled,
            "lane_counts": dict(sorted(lane_counts.items())),
            "sparse_candidate_count": lane_counts.get("sparse", 0),
            "dense_candidate_count": lane_counts.get("dense", 0),
            "fused_count": len(fused),
            "duplicate_count": metrics.get("duplicate_count", 0),
            "metrics": metrics,
            "fallback_notes": list(dict.fromkeys(fallback_notes or []))[:_MAX_TRACE_LIST],
            "latency_ms": latency_ms or {},
            "candidates": [c.to_trace_dict() for c in fused[:_MAX_TRACE_CANDIDATES]],
        }
    )


def attach_fusion_metadata(candidate: FusedCandidate, *, fallback_notes: list[str] | None = None) -> None:
    metadata = {
        "candidate_id": candidate.candidate_id,
        "fused_rank": candidate.fused_rank,
        "fused_score": round(_finite_or_zero(candidate.fused_score), 12),
        "lane_ranks": dict(sorted(candidate.lane_ranks.items())),
        "lane_scores": {k: _finite_or_zero(v) for k, v in sorted(candidate.lane_scores.items())},
        "requested_lanes": sorted({str(v.get("requested_lane") or k) for k, v in candidate.provenance.get("lanes", {}).items()}),
        "effective_lanes": sorted({str(v.get("effective_lane") or k) for k, v in candidate.provenance.get("lanes", {}).items()}),
        "fallback_notes": list(dict.fromkeys(fallback_notes or []))[:_MAX_TRACE_LIST],
    }
    candidate.chunk["_ragflow_fusion"] = _sanitize_trace(metadata)


def merge_child_fusion_metadata(children: list[dict[str, Any]]) -> dict[str, Any] | None:
    metas: list[dict[str, Any]] = []
    for child in children:
        if not isinstance(child, dict):
            continue
        meta = child.get("_ragflow_fusion")
        if isinstance(meta, dict):
            metas.append(meta)
    if not metas:
        return None

    def rank_of(meta: dict[str, Any]) -> int:
        rank = meta.get("fused_rank")
        return rank if isinstance(rank, int) else 10**9
    best = min(metas, key=rank_of)
    child_ids = [str(m.get("candidate_id")) for m in metas if m.get("candidate_id")]
    child_ranks: list[int] = [rank for m in metas if isinstance((rank := m.get("fused_rank")), int)]
    merged = {
        "parent_aggregated": True,
        "child_candidate_ids": child_ids[:_MAX_TRACE_LIST],
        "child_fused_ranks": child_ranks[:_MAX_TRACE_LIST],
        "best_child_fused_rank": min(child_ranks) if child_ranks else None,
        "lane_ranks": best.get("lane_ranks", {}),
        "lane_scores": best.get("lane_scores", {}),
        "fallback_notes": list(dict.fromkeys(note for m in metas for note in (m.get("fallback_notes") or [])))[:_MAX_TRACE_LIST],
    }
    return _sanitize_trace(merged)


def stable_candidate_identity(chunk: dict[str, Any]) -> tuple[str, str]:
    chunk_id = chunk.get("chunk_id") or chunk.get("id")
    if chunk_id:
        return f"chunk:{chunk_id}", "chunk_id"
    doc_id = chunk.get("doc_id")
    section = _section_key(_section_page_order(chunk))
    if doc_id and section:
        return f"doc-section:{doc_id}:{section}", "doc_section"
    return f"content:{_content_hash(_content_from_chunk(chunk))}", "content_hash"


def _group_candidates(candidates: list[RetrievalLaneCandidate]) -> dict[str, list[RetrievalLaneCandidate]]:
    grouped: dict[str, list[RetrievalLaneCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.candidate_id].append(candidate)
    return grouped


def _make_fused_candidate(group: list[RetrievalLaneCandidate], fused_score: float) -> FusedCandidate:
    ordered = sorted(group, key=lambda c: (c.rank, c.original_index, c.lane, c.candidate_id))
    representative = _representative_candidate(ordered)
    lane_ranks = {c.lane: c.rank for c in ordered}
    lane_scores = {c.lane: _finite_or_zero(c.raw_score) for c in ordered}
    normalized_lane_scores = {c.lane: _finite_or_zero(c.normalized_score) for c in ordered}
    provenance = {
        "lanes": {
            c.lane: {
                "requested_lane": c.requested_lane,
                "effective_lane": c.effective_lane,
                "rank": c.rank,
                "fallback_note": c.fallback_note,
            }
            for c in ordered
        }
    }
    return FusedCandidate(
        candidate_id=representative.candidate_id,
        identity_basis=representative.identity_basis,
        fused_rank=0,
        fused_score=_finite_or_zero(fused_score),
        best_lane_rank=min(lane_ranks.values()) if lane_ranks else 10**9,
        lane_ranks=lane_ranks,
        lane_scores=lane_scores,
        normalized_lane_scores=normalized_lane_scores,
        chunk=deepcopy(representative.chunk),
        chunks_by_lane={c.lane: c.chunk for c in ordered},
        doc_id=representative.doc_id,
        doc_title=representative.doc_title,
        section_page_order=representative.section_page_order,
        original_index=min(c.original_index for c in ordered),
        provenance=provenance,
    )


def _representative_candidate(group: list[RetrievalLaneCandidate]) -> RetrievalLaneCandidate:
    # Prefer dense when it is the best-ranked evidence for richer semantic score
    # fields; otherwise preserve best rank/original order deterministically.
    return sorted(group, key=lambda c: (c.rank, c.original_index, 0 if c.lane == "dense" else 1, c.candidate_id))[0]


def _linear_score(group: list[RetrievalLaneCandidate], config: FusionConfig) -> float:
    score = 0.0
    for candidate in group:
        score += config.lane_weight(candidate.lane) * _finite_or_zero(candidate.normalized_score)
    return score


def _rrf_score(group: list[RetrievalLaneCandidate], config: FusionConfig) -> float:
    total = 0.0
    for candidate in group:
        total += config.lane_weight(candidate.lane) / (config.rrf_k + max(1, int(candidate.rank)))
    return total


def _rank_fused(fused: list[FusedCandidate]) -> list[FusedCandidate]:
    ordered = sorted(
        fused,
        key=lambda c: (
            -_finite_or_zero(c.fused_score),
            c.best_lane_rank,
            -len(c.lane_ranks),
            str(c.doc_id or ""),
            c.original_index,
            c.candidate_id,
        ),
    )
    for rank, candidate in enumerate(ordered, 1):
        candidate.fused_rank = rank
    return ordered


def _doc_level_representatives(candidates: list[RetrievalLaneCandidate], config: FusionConfig) -> list[RetrievalLaneCandidate]:
    by_doc: dict[str, list[RetrievalLaneCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_doc[str(candidate.doc_id)].append(candidate)
    doc_candidates: list[RetrievalLaneCandidate] = []
    for doc_id, items in by_doc.items():
        score = max(_finite_or_zero(c.normalized_score) for c in items)
        best = sorted(items, key=lambda c: (c.rank, c.original_index, c.candidate_id))[0]
        pseudo = RetrievalLaneCandidate(
            candidate_id=f"doc:{doc_id}:{best.candidate_id}",
            identity_basis="doc_id",
            lane=best.lane,
            requested_lane=best.requested_lane,
            effective_lane=best.effective_lane,
            rank=min(c.rank for c in items),
            raw_score=score,
            normalized_score=score,
            sparse_score=best.sparse_score,
            dense_score=best.dense_score,
            chunk_id=best.chunk_id,
            doc_id=best.doc_id,
            doc_title=best.doc_title,
            content=best.content,
            section_page_order=best.section_page_order,
            original_index=min(c.original_index for c in items),
            chunk=best.chunk,
            provenance={"doc_level": True, "selected_chunk_ids": [c.chunk_id for c in items[:_MAX_TRACE_LIST]]},
            fallback_note=best.fallback_note,
        )
        doc_candidates.append(pseudo)
    return normalize_candidate_scores(doc_candidates)


def _apply_conservative_diversity(fused: list[FusedCandidate]) -> list[FusedCandidate]:
    if len(fused) <= 2:
        return fused
    buckets: dict[str, list[FusedCandidate]] = defaultdict(list)
    for candidate in fused:
        buckets[str(candidate.doc_id or candidate.candidate_id)].append(candidate)
    for values in buckets.values():
        values.sort(key=lambda c: c.fused_rank)
    out: list[FusedCandidate] = []
    while any(buckets.values()):
        for doc_id in sorted(buckets):
            if buckets[doc_id]:
                out.append(buckets[doc_id].pop(0))
    return out


def _score_for_lane(chunk: dict[str, Any], lane: LaneName) -> float:
    if lane == "sparse":
        return _finite_or_zero(chunk.get("term_similarity", chunk.get("similarity", chunk.get("score", 0.0))))
    return _finite_or_zero(chunk.get("vector_similarity", chunk.get("similarity", chunk.get("score", 0.0))))


def _content_from_chunk(chunk: dict[str, Any]) -> str:
    content = chunk.get("content")
    if content is None:
        content = chunk.get("content_with_weight")
    return "" if content is None else str(content)


def _section_page_order(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "section": chunk.get("section") or chunk.get("section_name"),
        "page": chunk.get("page") or chunk.get("page_num") or chunk.get("page_idx"),
        "order": chunk.get("position_int") or chunk.get("position") or chunk.get("chunk_order"),
    }


def _section_key(section: dict[str, Any]) -> str:
    return ":".join(str(section[k]) for k in ("section", "page", "order") if section.get(k) not in (None, "", []))


def _content_hash(content: str) -> str:
    normalized = re.sub(r"\s+", " ", (content or "").strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _finite_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _finite_or_zero(value: Any) -> float:
    parsed = _finite_or_none(value)
    return 0.0 if parsed is None else parsed


def _safe_number(value: Any) -> float | None:
    return _finite_or_none(value)


def _setting(value: Any, env_name: str, default: Any, *, legacy_env: str | None = None) -> Any:
    if value is not None:
        return value
    env_value = os.getenv(env_name)
    if env_value is not None:
        return env_value
    if legacy_env:
        legacy = os.getenv(legacy_env)
        if legacy is not None:
            return legacy
    return default


def _bool_setting(value: Any, env_name: str, default: bool, *, legacy_env: str | None = None) -> bool:
    value = _setting(value, env_name, default, legacy_env=legacy_env)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_VALUES:
            return True
        if lowered in _FALSE_VALUES:
            return False
    return default


def _int_setting(value: Any, env_name: str, default: int, *, legacy_env: str | None = None) -> int:
    value = _setting(value, env_name, default, legacy_env=legacy_env)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _float_setting(value: Any, env_name: str, default: float, *, legacy_env: str | None = None) -> float:
    value = _setting(value, env_name, default, legacy_env=legacy_env)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed) or parsed < 0:
        return default
    return parsed


def _sanitize_trace(value: Any, *, key: str | None = None, depth: int = 0) -> Any:
    if depth > 6:
        return "<omitted:depth>"
    if key and any(part in key.lower() for part in ("vector", "embedding")):
        return None
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return _safe_number(value)
    if isinstance(value, str):
        return value[:_MAX_BODY_PREVIEW] + "...<truncated>" if len(value) > _MAX_BODY_PREVIEW else value
    if isinstance(value, dict):
        out = {}
        for k, v in list(value.items())[:64]:
            item = _sanitize_trace(v, key=str(k), depth=depth + 1)
            if item is not None:
                out[str(k)] = item
        return out
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_trace(v, depth=depth + 1) for v in list(value)[:_MAX_TRACE_LIST]]
    return repr(value)[:_MAX_BODY_PREVIEW]


def _doc_aggs_from_chunks(chunks: list[dict[str, Any]], original_doc_aggs: Any) -> list[dict[str, Any]]:
    counts: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        doc_name = chunk.get("docnm_kwd") or chunk.get("document_name") or ""
        doc_id = chunk.get("doc_id") or ""
        key = doc_name or doc_id
        if not key:
            continue
        counts.setdefault(key, {"doc_name": doc_name, "doc_id": doc_id, "count": 0})["count"] += 1
    if counts:
        return sorted(counts.values(), key=lambda row: (-row["count"], row.get("doc_name") or "", row.get("doc_id") or ""))
    return original_doc_aggs if isinstance(original_doc_aggs, list) else []
