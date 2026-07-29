#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#
"""Bounded query-time retrieval planning and sufficiency refinement."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from collections import defaultdict
from collections.abc import Mapping
from copy import copy, deepcopy
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

from rag.advanced_rag.structured_output import (
    PlannerContentV2,
    STRUCTURED_OUTPUT_SCHEMA_VERSION,
    StructuredOutputError,
    StructuredOutputFailureReason,
    StructuredOutputMode,
    StructuredOutputResult,
    SufficiencyDecisionV2,
    canonical_json_schema,
    canonical_output_instructions,
    decode_structured_output,
)
from rag.utils.retrieval_diagnostics import make_retrieval_variant, resolve_variant_knobs

AgenticRetrievalMode = Literal["off", "diagnostic", "shadow", "active"]
AgenticRetrievalPlannerMode = Literal["llm", "deterministic"]
AgenticRefinementJudgeMode = Literal["heuristic", "llm", "hybrid", "deterministic"]
RetrievalVariant = Literal["hybrid_default", "keyword_first", "embedding_retry"]

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|2100)\b")
_QUOTED_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][\w-]*(?:\s+[A-Z][\w-]*){0,5}")
_NATIVE_LLM_ERROR_RE = re.compile(r"\*\*ERROR\*\*: (?P<code>[A-Z][A-Z0-9_]*) - .+", re.DOTALL)
_SMALL_TALK = {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay"}
_LEADING_QUERY_COMMANDS = {"find", "give", "identify", "locate", "look", "retrieve", "search", "show", "tell"}
_STOPWORDS = {
    "about",
    "after",
    "also",
    "among",
    "and",
    "are",
    "before",
    "between",
    "can",
    "compare",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "into",
    "list",
    "many",
    "more",
    "most",
    "of",
    "one",
    "rank",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "these",
    "this",
    "those",
    "through",
    "versus",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
_COMPARISON_TERMS = {"compare", "compared", "versus", "vs", "difference", "higher", "lower", "largest", "smallest", "more", "less", "between"}
_LIST_TERMS = {"list", "rank", "ranking", "top", "first", "last", "all", "which"}
_TEMPORAL_TERMS = {"after", "before", "during", "when", "date", "year", "current", "recent", "as", "of"}
_ALLOWED_COMPLEXITIES = {"simple", "multi_facet", "multi_hop", "comparison", "temporal", "list", "unknown"}
_ALLOWED_EVIDENCE_TYPES = {"definition", "numeric", "date", "comparison", "quote", "list_item"}
_ALLOWED_RETRIEVAL_VARIANTS = {"hybrid_default", "keyword_first", "embedding_retry"}
_REQUIRED_JUDGE_KEYS = {
    "sufficient",
    "confidence",
    "covered_facets",
    "missing_facets",
    "contradictions",
    "exact_fact_gaps",
    "refusal_justified",
    "recommended_followups",
}
_REQUIRED_JUDGE_V2_KEYS = (_REQUIRED_JUDGE_KEYS - {"sufficient"}) | {"action"}
_SUPPORTED_FACET_SUPPORT = {"strong", "weak"}
_SUPPORTED_EXACT_FACT_TYPES = {"date", "number", "name"}
_MAX_REFINEMENT_FOLLOWUP_TOP_N = 20


class AgenticPlannerError(Exception):
    """Raised when the LLM planner cannot produce a valid bounded plan."""

    def __init__(
        self,
        reason: str,
        detail: str | None = None,
        *,
        failure_class: str = "unknown",
        retryable: bool = False,
        status_code: int | None = None,
        bounded_retry_after_ms: int | None = None,
        safe_category: str | None = None,
    ):
        self.reason = reason
        self.detail = detail
        self.failure_class = failure_class
        self.retryable = retryable
        self.status_code = status_code
        self.bounded_retry_after_ms = bounded_retry_after_ms
        self.safe_category = safe_category or failure_class
        super().__init__(reason if detail is None else f"{reason}: {detail}")


class AgenticRefinementError(Exception):
    """Raised when Phase 6 cannot safely refine the Phase 5 evidence."""

    def __init__(self, reason: str, detail: str | None = None):
        self.reason = reason
        self.detail = detail
        super().__init__(reason if detail is None else f"{reason}: {detail}")


@dataclass(frozen=True)
class AgenticRetrievalConfig:
    enabled: bool = False
    mode: AgenticRetrievalMode = "off"
    planner_mode: AgenticRetrievalPlannerMode = "llm"
    max_subqueries: int = 3
    subquery_top_n: int = 6
    planner_timeout_ms: int = 25000
    planner_max_attempts_per_key: int = 2
    planner_max_calls_per_turn: int = 4
    planner_total_budget_ms: int = 45000
    plan_cache_enabled: bool = False
    retrieval_timeout_ms: int = 6000
    max_extra_retrieval_calls: int = 3
    min_anchor_overlap: float = 0.5
    simple_query_bypass: bool = True
    latency_budget_ms: int = 12000
    rrf_k: int = 60

    @classmethod
    def from_env(cls, overrides: dict[str, Any] | None = None) -> "AgenticRetrievalConfig":
        overrides = overrides or {}
        enabled = _bool_setting(overrides.get("agentic_retrieval_enabled"), "AGENTIC_RETRIEVAL_ENABLED", False)
        mode = str(overrides.get("agentic_retrieval_mode") or os.getenv("AGENTIC_RETRIEVAL_MODE") or "off").strip().lower()
        if mode not in {"off", "diagnostic", "shadow", "active"}:
            mode = "off"
        if not enabled:
            mode = "off"
        planner_mode = str(overrides.get("agentic_retrieval_planner_mode") or os.getenv("AGENTIC_RETRIEVAL_PLANNER_MODE") or "llm").strip().lower()
        if planner_mode not in {"llm", "deterministic"}:
            raise ValueError("AGENTIC_RETRIEVAL_PLANNER_MODE must be one of: llm, deterministic")
        return cls(
            enabled=enabled,
            mode=mode,  # type: ignore[arg-type]
            planner_mode=planner_mode,  # type: ignore[arg-type]
            max_subqueries=_int_setting(overrides.get("agentic_retrieval_max_subqueries"), "AGENTIC_RETRIEVAL_MAX_SUBQUERIES", 3) or 3,
            subquery_top_n=_int_setting(overrides.get("agentic_retrieval_subquery_top_n"), "AGENTIC_RETRIEVAL_SUBQUERY_TOP_N", 6) or 6,
            planner_timeout_ms=_bounded_int_setting(overrides.get("agentic_retrieval_planner_timeout_ms"), "AGENTIC_RETRIEVAL_PLANNER_TIMEOUT_MS", 25000, 1, 60000),
            planner_max_attempts_per_key=_bounded_int_setting(overrides.get("agentic_retrieval_planner_max_attempts_per_key"), "AGENTIC_RETRIEVAL_PLANNER_MAX_ATTEMPTS_PER_KEY", 2, 1, 4),
            planner_max_calls_per_turn=_bounded_int_setting(overrides.get("agentic_retrieval_planner_max_calls_per_turn"), "AGENTIC_RETRIEVAL_PLANNER_MAX_CALLS_PER_TURN", 4, 1, 16),
            planner_total_budget_ms=_bounded_int_setting(overrides.get("agentic_retrieval_planner_total_budget_ms"), "AGENTIC_RETRIEVAL_PLANNER_TOTAL_BUDGET_MS", 45000, 1, 120000),
            plan_cache_enabled=_bool_setting(overrides.get("agentic_retrieval_plan_cache_enabled"), "AGENTIC_RETRIEVAL_PLAN_CACHE_ENABLED", False),
            retrieval_timeout_ms=_bounded_int_setting(overrides.get("agentic_retrieval_retrieval_timeout_ms"), "AGENTIC_RETRIEVAL_RETRIEVAL_TIMEOUT_MS", 6000, 1, 30000),
            max_extra_retrieval_calls=_int_setting(overrides.get("agentic_retrieval_max_extra_retrieval_calls"), "AGENTIC_RETRIEVAL_MAX_EXTRA_RETRIEVAL_CALLS", 3) or 3,
            min_anchor_overlap=_float_setting(overrides.get("agentic_retrieval_drift_min_anchor_overlap"), "AGENTIC_RETRIEVAL_DRIFT_MIN_ANCHOR_OVERLAP", 0.5) or 0.5,
            simple_query_bypass=_bool_setting(overrides.get("agentic_retrieval_simple_query_bypass"), "AGENTIC_RETRIEVAL_SIMPLE_QUERY_BYPASS", True),
            latency_budget_ms=_bounded_int_setting(overrides.get("agentic_retrieval_latency_budget_ms"), "AGENTIC_RETRIEVAL_LATENCY_BUDGET_MS", 12000, 1, 60000),
            rrf_k=_int_setting(overrides.get("agentic_retrieval_rrf_k"), "AGENTIC_RETRIEVAL_RRF_K", 60) or 60,
        )


@dataclass(frozen=True)
class AgenticRefinementConfig:
    enabled: bool = False
    mode: AgenticRetrievalMode = "off"
    max_iterations: int = 2
    max_followup_queries: int = 4
    min_new_evidence: int = 1
    judge: AgenticRefinementJudgeMode = "hybrid"
    judge_timeout_ms: int = 15000
    latency_budget_ms: int = 50000
    max_drift_rate: float = 0.05
    confidence_threshold: float = 0.72

    @classmethod
    def from_env(cls, overrides: dict[str, Any] | None = None) -> "AgenticRefinementConfig":
        overrides = overrides or {}
        enabled = _bool_setting(overrides.get("agentic_refinement_enabled"), "AGENTIC_REFINEMENT_ENABLED", False)
        mode = str(overrides.get("agentic_refinement_mode") or os.getenv("AGENTIC_REFINEMENT_MODE") or "off").strip().lower()
        if mode not in {"off", "diagnostic", "shadow", "active"} or not enabled:
            mode = "off"
        judge = str(overrides.get("agentic_refinement_judge") or os.getenv("AGENTIC_REFINEMENT_JUDGE") or "hybrid").strip().lower()
        if judge not in {"heuristic", "llm", "hybrid", "deterministic"}:
            judge = "hybrid"
        return cls(
            enabled=enabled,
            mode=mode,  # type: ignore[arg-type]
            max_iterations=_bounded_int_setting(overrides.get("agentic_refinement_max_iterations"), "AGENTIC_REFINEMENT_MAX_ITERATIONS", 2, 1, 4),
            max_followup_queries=_bounded_int_setting(overrides.get("agentic_refinement_max_followup_queries"), "AGENTIC_REFINEMENT_MAX_FOLLOWUP_QUERIES", 4, 1, 4),
            min_new_evidence=_bounded_int_setting(overrides.get("agentic_refinement_min_new_evidence"), "AGENTIC_REFINEMENT_MIN_NEW_EVIDENCE", 1, 1, 20),
            judge=judge,  # type: ignore[arg-type]
            judge_timeout_ms=_bounded_int_setting(overrides.get("agentic_refinement_judge_timeout_ms"), "AGENTIC_REFINEMENT_JUDGE_TIMEOUT_MS", 15000, 1, 30000),
            latency_budget_ms=_bounded_int_setting(overrides.get("agentic_refinement_latency_budget_ms"), "AGENTIC_REFINEMENT_LATENCY_BUDGET_MS", 50000, 1, 120000),
            max_drift_rate=_nonnegative_float_setting(overrides.get("agentic_refinement_max_drift_rate"), "AGENTIC_REFINEMENT_MAX_DRIFT_RATE", 0.05),
            confidence_threshold=_bounded_float_setting(overrides.get("agentic_refinement_confidence_threshold"), "AGENTIC_REFINEMENT_CONFIDENCE_THRESHOLD", 0.72),
        )

    def normalized(self) -> "AgenticRefinementConfig":
        mode = self.mode if self.enabled and self.mode in {"off", "diagnostic", "shadow", "active"} else "off"
        judge = self.judge if self.judge in {"heuristic", "llm", "hybrid", "deterministic"} else "hybrid"
        return replace(
            self,
            mode=mode,
            judge=judge,
            max_iterations=max(1, min(int(self.max_iterations), 4)),
            max_followup_queries=max(1, min(int(self.max_followup_queries), 4)),
            min_new_evidence=max(1, min(int(self.min_new_evidence), 20)),
            judge_timeout_ms=max(1, min(int(self.judge_timeout_ms), 30000)),
            latency_budget_ms=max(1, min(int(self.latency_budget_ms), 120000)),
            max_drift_rate=max(0.0, min(float(self.max_drift_rate), 1.0)),
            confidence_threshold=max(0.0, min(float(self.confidence_threshold), 1.0)),
        )


@dataclass(frozen=True)
class PlanningTrigger:
    enabled: bool
    mode: AgenticRetrievalMode
    should_plan: bool
    reasons: tuple[str, ...] = ()
    bypass_reason: str | None = None
    features: dict[str, Any] = field(default_factory=dict)

    def to_trace_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RequiredFacet:
    facet_id: str
    description: str
    anchors: tuple[str, ...]
    evidence_type: str = "quote"


@dataclass(frozen=True)
class SubquerySpec:
    plan_id: str
    subquery_id: str
    facet_id: str
    query: str
    keywords: str | None = None
    docid_scope: list[str] | None = None
    top_n: int = 6
    retrieval_variant: RetrievalVariant = "hybrid_default"
    must_have_terms: tuple[str, ...] = ()
    forbidden_new_entities: tuple[str, ...] = ()
    rationale: str = ""
    iteration_id: str | None = None
    followup_id: str | None = None


@dataclass(frozen=True)
class BoundedRetrievalPlan:
    plan_id: str
    original_question: str
    complexity: str
    trigger_reasons: tuple[str, ...]
    required_facets: tuple[RequiredFacet, ...]
    subqueries: tuple[SubquerySpec, ...]
    merge_policy: dict[str, Any] = field(default_factory=dict)
    drift_controls: dict[str, Any] = field(default_factory=dict)
    plan_origin: Literal["llm_native", "llm_text", "deterministic_fallback", "deterministic_primary"] = "llm_text"

    def to_trace_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BoundedRetrievalResult:
    kbinfos: dict[str, Any]
    plan: BoundedRetrievalPlan
    accepted_subqueries: int
    rejected_subqueries: int
    fallback_to_baseline: bool = False
    fallback_reason: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FollowupQuerySpec:
    plan_id: str
    facet_id: str
    query: str
    iteration_id: str = ""
    followup_id: str = ""
    keywords: str | None = None
    top_n: int = 5
    retrieval_variant: RetrievalVariant = "hybrid_default"


@dataclass(frozen=True)
class SufficiencyJudge:
    sufficient: bool
    confidence: float
    covered_facets: tuple[dict[str, Any], ...]
    missing_facets: tuple[dict[str, Any], ...]
    contradictions: tuple[dict[str, Any], ...]
    exact_fact_gaps: tuple[dict[str, Any], ...]
    refusal_justified: bool
    recommended_followups: tuple[FollowupQuerySpec, ...]

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "confidence": self.confidence,
            "covered_facets": [dict(item) for item in self.covered_facets],
            "missing_facets": [dict(item) for item in self.missing_facets],
            "contradictions": [dict(item) for item in self.contradictions],
            "exact_fact_gaps": [dict(item) for item in self.exact_fact_gaps],
            "refusal_justified": self.refusal_justified,
            "recommended_followups": [asdict(item) for item in self.recommended_followups],
        }


@dataclass(frozen=True)
class RefinementIteration:
    iteration_id: str
    judge: SufficiencyJudge | None
    followups: tuple[FollowupQuerySpec, ...]
    accepted_new_evidence_count: int
    rejected_evidence_count: int
    coverage_before: float
    coverage_after: float
    marginal_gain: float
    latency_ms: float
    stop_reason: str | None = None


@dataclass(frozen=True)
class RefinementResult:
    kbinfos: dict[str, Any]
    iterations: tuple[RefinementIteration, ...]
    candidate_kbinfos: dict[str, Any] | None = None
    accepted_chunks: tuple[dict[str, Any], ...] = ()
    rejected_chunks: tuple[dict[str, Any], ...] = ()
    changed: bool = False
    fallback_to_previous_context: bool = False
    fallback_reason: str | None = None
    stop_reason: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def selected_new_evidence_count(self) -> int:
        return len(self.accepted_chunks)


@dataclass(frozen=True)
class EvidenceAcceptanceResult:
    kbinfos: dict[str, Any]
    accepted_chunks: tuple[dict[str, Any], ...]
    rejected_chunks: tuple[dict[str, Any], ...]
    candidate_kbinfos: dict[str, Any] | None = None

    @property
    def selected_new_evidence_count(self) -> int:
        return len(self.accepted_chunks)


@dataclass(frozen=True)
class PlanValidationResult:
    valid: bool
    errors: list[str]
    plan: BoundedRetrievalPlan
    fallback_reason: str | None = None

    def __iter__(self):
        yield self.valid
        yield self.errors

    def __bool__(self) -> bool:
        return self.valid


def should_plan(
    question: str,
    history: list[dict[str, Any]] | None,
    dialog: Any,
    attachments: list[str] | None,
    cfg: AgenticRetrievalConfig,
    cheap_features: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> PlanningTrigger:
    text = (question or "").strip()
    if not cfg.enabled or cfg.mode == "off":
        return PlanningTrigger(enabled=cfg.enabled, mode=cfg.mode, should_plan=False, bypass_reason="disabled")
    if not text or text.lower() in _SMALL_TALK:
        return PlanningTrigger(enabled=True, mode=cfg.mode, should_plan=False, bypass_reason="empty_or_small_talk")
    if dialog is not None and not getattr(dialog, "kb_ids", None) and not (cheap_features or {}).get("kb_scope_available"):
        return PlanningTrigger(enabled=True, mode=cfg.mode, should_plan=False, bypass_reason="no_kb_scope")
    if cfg.max_subqueries < 2 or cfg.max_extra_retrieval_calls < 2:
        return PlanningTrigger(enabled=True, mode=cfg.mode, should_plan=False, bypass_reason="insufficient_retrieval_budget")

    features = _question_features(text)
    if cheap_features:
        features.update(cheap_features)
    reasons: list[str] = []
    if features["entity_count"] >= 2:
        reasons.append("multiple_entities")
    if features["comparison"]:
        reasons.append("comparison_or_ranking")
    if features["temporal"]:
        reasons.append("temporal_constraint")
    if features["clause_count"] >= 2:
        reasons.append("multiple_independent_clauses")
    if features["indirect_description"]:
        reasons.append("indirect_entity_description")
    if diagnostics and diagnostics.get("duplicate_heavy"):
        reasons.append("duplicate_heavy_first_pass")

    if attachments and cfg.simple_query_bypass and not reasons:
        return PlanningTrigger(enabled=True, mode=cfg.mode, should_plan=False, bypass_reason="attachment_scoped_simple_query", features=features)
    if cfg.simple_query_bypass and not reasons:
        return PlanningTrigger(enabled=True, mode=cfg.mode, should_plan=False, bypass_reason="simple_query_bypass", features=features)
    return PlanningTrigger(enabled=True, mode=cfg.mode, should_plan=bool(reasons), reasons=tuple(reasons), features=features)


def build_planner_input(
    question: str,
    history: list[dict[str, Any]] | None = None,
    dialog: Any | None = None,
    attachments: list[str] | None = None,
    cfg: AgenticRetrievalConfig | None = None,
    trigger: PlanningTrigger | None = None,
    tenant_ids: list[str] | None = None,
    metadata_filters: Any = None,
    capability_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = cfg or AgenticRetrievalConfig()
    doc_ids = list(attachments or [])
    kb_ids = list(getattr(dialog, "kb_ids", []) or [])
    history_summary = _summarize_history(history or [])
    plan_id = f"plan-{uuid.uuid4().hex[:12]}"
    result = {
        "plan_id": plan_id,
        "original_question": question,
        "question": question,
        "history_summary": history_summary,
        "history_turns": len(history or []),
        "trigger": trigger.to_trace_dict() if trigger else {"should_plan": False, "reasons": []},
        "limits": {
            "max_subqueries": cfg.max_subqueries,
            "subquery_top_n": cfg.subquery_top_n,
            "allowed_retrieval_variants": sorted(_ALLOWED_RETRIEVAL_VARIANTS),
            "allowed_evidence_types": sorted(_ALLOWED_EVIDENCE_TYPES),
        },
        "scope": {
            "tenant_ids": list(tenant_ids or []),
            "kb_ids": kb_ids,
            "doc_ids": doc_ids,
            "metadata_filters": metadata_filters if isinstance(metadata_filters, dict) else {},
            "metadata_filters_enabled": bool(metadata_filters),
        },
        "output_schema": canonical_json_schema(PlannerContentV2),
        # Backward-compatible introspection fields. These are not used to synthesize plans.
        "kb_ids": kb_ids,
        "tenant_ids": list(tenant_ids or []),
        "doc_ids": doc_ids,
        "metadata_filters_enabled": bool(metadata_filters),
        "config": asdict(cfg),
        "anchors": _anchors_for_text(question),
        "clauses": _split_clauses(question),
        "features": _question_features(question),
    }
    if capability_manifest is not None:
        result["capability_manifest"] = capability_manifest
    return result


def build_deterministic_plan(
    planner_input: dict[str, Any],
    cfg: AgenticRetrievalConfig,
    *,
    plan_origin: Literal["deterministic_fallback", "deterministic_primary"],
) -> BoundedRetrievalPlan:
    """Build one validated, bounded plan from application-owned request data."""
    question = str(planner_input.get("original_question") or planner_input.get("question") or "").strip()
    if not question:
        raise ValueError("empty_original_question")

    plan_id = str(planner_input.get("plan_id") or f"plan-{uuid.uuid4().hex[:12]}")
    anchors = tuple(_anchors_for_text(question)[:6]) or tuple(token for token in _TOKEN_RE.findall(question) if len(token) > 2)[:6]
    if not anchors:
        raise ValueError("original_question_has_no_retrieval_terms")

    features = _question_features(question)
    if features.get("comparison"):
        complexity = "comparison"
    elif features.get("temporal"):
        complexity = "temporal"
    elif features.get("list_or_ranking"):
        complexity = "list"
    else:
        complexity = "unknown"
    facet = RequiredFacet(
        facet_id="primary",
        description=question[:240],
        anchors=anchors,
        evidence_type="quote",
    )
    subquery = SubquerySpec(
        plan_id=plan_id,
        subquery_id="original_query",
        facet_id=facet.facet_id,
        query=question,
        keywords=question,
        top_n=max(1, min(cfg.subquery_top_n, 30)),
        retrieval_variant="hybrid_default",
        must_have_terms=anchors[:8],
        rationale="Use the original request without model-generated expansion.",
    )
    return BoundedRetrievalPlan(
        plan_id=plan_id,
        original_question=question,
        complexity=complexity,
        trigger_reasons=("planner_fallback" if plan_origin == "deterministic_fallback" else "deterministic_primary",),
        required_facets=(facet,),
        subqueries=(subquery,),
        merge_policy={"strategy": "subquery_rrf_then_context_builder", "max_chunks_per_facet": min(4, cfg.subquery_top_n)},
        drift_controls={"anchor_entities": list(anchors), "min_anchor_overlap": cfg.min_anchor_overlap, "allow_new_entities": False},
        plan_origin=plan_origin,
    )


def build_deterministic_fallback_plan(
    planner_input: dict[str, Any],
    cfg: AgenticRetrievalConfig,
) -> BoundedRetrievalPlan:
    """Build the deterministic fallback plan with failure-path provenance."""
    return build_deterministic_plan(planner_input, cfg, plan_origin="deterministic_fallback")


def resolve_deterministic_primary_plan(
    planner_input: dict[str, Any],
    cfg: AgenticRetrievalConfig,
    *,
    rag_trace: Any = None,
) -> PlanValidationResult:
    """Build and validate an opt-in deterministic Phase 5 plan without LLM work."""
    plan_id = str(planner_input.get("plan_id") or "")
    lifecycle_id = f"planner.deterministic.{plan_id or uuid.uuid4().hex[:12]}"
    try:
        candidate = build_deterministic_plan(planner_input, cfg, plan_origin="deterministic_primary")
        validation = validate_plan(candidate, cfg, original_question=candidate.original_question)
    except (TypeError, ValueError) as exc:
        _trace_planner_event(
            rag_trace,
            "planner_deterministic_primary",
            {
                "lifecycle_id": lifecycle_id,
                "planner_mode": cfg.planner_mode,
                "plan_origin": "deterministic_primary",
                "planner_llm_skipped": True,
                "skip_reason": "deterministic_primary",
                "provider_call_count": 0,
                "repair_call_count": 0,
                "cache_status": "skipped",
                "validation_status": "failed",
                "validation_errors": [str(exc)],
            },
        )
        raise
    _trace_planner_event(
        rag_trace,
        "planner_deterministic_primary",
        {
            "lifecycle_id": lifecycle_id,
            "planner_mode": cfg.planner_mode,
            "plan_origin": "deterministic_primary",
            "planner_llm_skipped": True,
            "skip_reason": "deterministic_primary",
            "provider_call_count": 0,
            "repair_call_count": 0,
            "cache_status": "skipped",
            "plan_id": validation.plan.plan_id,
            "validation_status": "valid" if validation.valid else "failed",
            "validation_errors": _safe_planner_validation_errors(validation.errors),
        },
    )
    return validation


def build_llm_planner_prompt(
    planner_input: dict[str, Any],
    cfg: AgenticRetrievalConfig,
    *,
    repair_errors: tuple[str, ...] = (),
) -> tuple[str, list[dict[str, str]]]:
    """Build a server-owned planner envelope plus canonical fillable output."""
    plan_id = str(planner_input.get("plan_id") or f"plan-{uuid.uuid4().hex[:12]}")
    prompt_input = {
        "plan_id": plan_id,
        "original_question": str(planner_input.get("original_question") or planner_input.get("question") or ""),
        "history_summary": str(planner_input.get("history_summary") or ""),
        "trigger": planner_input.get("trigger") or {"should_plan": True, "reasons": []},
        "limits": planner_input.get("limits")
        or {
            "max_subqueries": cfg.max_subqueries,
            "subquery_top_n": cfg.subquery_top_n,
            "allowed_retrieval_variants": sorted(_ALLOWED_RETRIEVAL_VARIANTS),
            "allowed_evidence_types": sorted(_ALLOWED_EVIDENCE_TYPES),
        },
        "scope": planner_input.get("scope")
        or {
            "tenant_ids": planner_input.get("tenant_ids") or [],
            "kb_ids": planner_input.get("kb_ids") or [],
            "doc_ids": planner_input.get("doc_ids") or [],
            "metadata_filters": {},
        },
        "output_schema": canonical_json_schema(PlannerContentV2),
    }
    if "capability_manifest" in planner_input:
        # Availability is advisory. Capability actions use a separate fixed
        # application schema and never replace ordinary subqueries.
        prompt_input["capability_manifest"] = planner_input["capability_manifest"]
    system_prompt = (
        "You are a retrieval planning helper for RAGFlow.\n"
        "You do not answer the user.\n"
        "You create one bounded retrieval plan for the existing retriever.\n"
        "The plan must help retrieval find evidence chunks for the original question.\n"
        "Do not call tools or choose execution policy.\n" + canonical_output_instructions(PlannerContentV2)
    )
    user_prompt = (
        "Fill the canonical planner object using the server-owned envelope below.\n"
        "The envelope fields, including plan_id, question, scope, limits, and execution policy, must not be copied into the returned object.\n"
        f"Create at most {cfg.max_subqueries} subqueries.\n"
        "Preserve original and facet anchors, do not invent unrelated new entities, and keep subqueries bounded.\n"
        "Use only allowed retrieval_variant and evidence_type values.\n"
        "Set docid_scope to null unless the provided document scope is explicitly required.\n"
        "Set top_n at or below the envelope subquery_top_n.\n\n" + json.dumps(prompt_input, ensure_ascii=False, sort_keys=True)
    )
    if repair_errors:
        user_prompt += (
            "\n\nThe previous response failed strict parsing or validation. "
            "Return one complete replacement JSON object from scratch using the canonical schema; do not return a patch.\n"
            "Application failure codes: " + json.dumps(list(repair_errors), ensure_ascii=True, sort_keys=True)
        )
    return system_prompt, [{"role": "user", "content": user_prompt}]


def _bounded_retry_after_ms(value: Any) -> int | None:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if seconds < 0:
        return None
    return min(int(seconds * 1000), 60000)


def _status_code_from(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        status = int(value)
    except (TypeError, ValueError):
        return None
    return status if 100 <= status <= 599 else None


def _provider_failure(
    *,
    status_code: int | None,
    retry_after_ms: int | None = None,
    timeout: bool = False,
) -> dict[str, Any]:
    if timeout:
        return {
            "failure_class": "timeout",
            "retryable": True,
            "status_code": status_code,
            "bounded_retry_after_ms": retry_after_ms,
            "safe_category": "provider_timeout",
        }
    if status_code == 429:
        category = "rate_limit"
        retryable = True
    elif status_code is not None and 500 <= status_code <= 599:
        category = "transient_server"
        retryable = True
    elif status_code in {401, 403}:
        category = "auth"
        retryable = False
    else:
        category = "unknown"
        retryable = False
    return {
        "failure_class": category,
        "retryable": retryable,
        "status_code": status_code,
        "bounded_retry_after_ms": retry_after_ms,
        "safe_category": f"provider_{category}",
    }


def classify_planner_exception(exc: BaseException) -> dict[str, Any]:
    """Classify a provider exception chain without retaining exception text."""
    seen: set[int] = set()
    current: BaseException | None = exc
    status_code: int | None = None
    retry_after_ms: int | None = None
    timeout = False
    safe_class_category: str | None = None
    auth_status_code: int | None = None
    auth_detected = False
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        class_name = type(current).__name__.lower()
        timeout = (
            timeout
            or isinstance(current, (asyncio.TimeoutError, TimeoutError))
            or class_name
            in {
                "apitimeouterror",
                "connecttimeout",
                "pooltimeout",
                "readtimeout",
                "writetimeout",
            }
        )
        if class_name in {"ratelimiterror", "toomanyrequestserror"}:
            safe_class_category = safe_class_category or "rate_limit"
        elif class_name in {"authenticationerror", "permissiondeniederror", "unauthorizederror", "forbiddenerror"}:
            auth_detected = True
            safe_class_category = safe_class_category or "auth"
        elif class_name in {"internalservererror", "serviceunavailableerror", "badgatewayerror"}:
            safe_class_category = safe_class_category or "transient_server"
        current_status_code = _status_code_from(getattr(current, "status_code", None))
        if current_status_code in {401, 403}:
            auth_detected = True
            auth_status_code = auth_status_code or current_status_code
        status_code = status_code or current_status_code
        response = getattr(current, "response", None)
        if response is not None:
            response_status_code = _status_code_from(getattr(response, "status_code", None))
            if response_status_code in {401, 403}:
                auth_detected = True
                auth_status_code = auth_status_code or response_status_code
            status_code = status_code or response_status_code
            headers = getattr(response, "headers", None)
            if isinstance(headers, Mapping):
                retry_after_ms = retry_after_ms or _bounded_retry_after_ms(headers.get("retry-after") or headers.get("Retry-After"))
        retry_after_ms = retry_after_ms or _bounded_retry_after_ms(getattr(current, "retry_after", None))
        current = current.__cause__ or current.__context__
    if auth_detected:
        failure = _provider_failure(status_code=auth_status_code or 401, retry_after_ms=retry_after_ms)
        if auth_status_code is None:
            failure["status_code"] = None
        return failure
    failure = _provider_failure(status_code=status_code, retry_after_ms=retry_after_ms, timeout=timeout)
    if status_code is None and not timeout and safe_class_category is not None:
        synthetic_status = {"rate_limit": 429, "auth": 401, "transient_server": 500}[safe_class_category]
        failure = _provider_failure(status_code=synthetic_status, retry_after_ms=retry_after_ms)
        failure["status_code"] = None
    return failure


def _provider_error_envelope(raw: Any) -> dict[str, Any] | None:
    """Recognize allowlisted provider error envelopes before planner parsing."""
    parsed = raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text.startswith("{") or not text.endswith("}"):
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, Mapping):
        return None
    error = parsed.get("error")
    has_error_shape = isinstance(error, Mapping) or ("errors" in parsed and isinstance(parsed.get("errors"), list))
    if not has_error_shape:
        return None
    error_map = error if isinstance(error, Mapping) else {}
    status_code = (
        _status_code_from(parsed.get("status_code"))
        or _status_code_from(parsed.get("status"))
        or _status_code_from(error_map.get("status_code"))
        or _status_code_from(error_map.get("status"))
        or _status_code_from(error_map.get("code"))
    )
    retry_after_ms = _bounded_retry_after_ms(parsed.get("retry_after") or error_map.get("retry_after"))
    failure = _provider_failure(status_code=status_code, retry_after_ms=retry_after_ms)
    if status_code is None:
        safe_type = str(error_map.get("type") or error_map.get("code") or "").strip().lower()
        if safe_type in {"rate_limit", "rate_limit_error", "too_many_requests"}:
            failure = _provider_failure(status_code=429, retry_after_ms=retry_after_ms)
        elif safe_type in {"authentication_error", "permission_error", "unauthorized", "forbidden"}:
            failure = _provider_failure(status_code=401, retry_after_ms=retry_after_ms)
        elif safe_type in {"server_error", "service_unavailable", "internal_server_error"}:
            failure = _provider_failure(status_code=500, retry_after_ms=retry_after_ms)
    return failure


def _native_provider_error_sentinel(raw: Any) -> dict[str, Any] | None:
    """Recognize RAGFlow native returned errors without retaining their suffix."""
    if not isinstance(raw, str):
        return None
    match = _NATIVE_LLM_ERROR_RE.fullmatch(raw)
    if match is None:
        return None
    code = match.group("code")
    if code == "RATE_LIMIT_EXCEEDED":
        return _provider_failure(status_code=429)
    if code == "SERVER_ERROR":
        return _provider_failure(status_code=500)
    if code == "AUTH_ERROR":
        return _provider_failure(status_code=401)
    if code == "TIMEOUT":
        return _provider_failure(status_code=None, timeout=True)
    if code == "CONNECTION_ERROR":
        failure = _provider_failure(status_code=500)
        failure["status_code"] = None
        return failure
    return _provider_failure(status_code=None)


def _safe_planner_validation_errors(errors: list[str]) -> list[str]:
    allowlisted = (
        "schema_invalid",
        "empty_original_question",
        "unsupported_complexity",
        "no_subqueries",
        "unsupported_evidence_type",
        "unknown_facet",
        "empty_query",
        "unsupported_variant",
        "anchor_drift",
        "new_entities",
        "no_valid_subqueries",
    )
    sanitized = []
    for error in errors:
        code = next((candidate for candidate in allowlisted if candidate in str(error)), "validation_failed")
        if code not in sanitized:
            sanitized.append(code)
    return sanitized or ["validation_failed"]


def parse_llm_planner_json(raw: Any) -> dict[str, Any]:
    """Read historical planner JSON; live provider output uses the strict decoder."""
    try:
        response = StructuredOutputResult(StructuredOutputMode.TEXT_ONLY, None, raw, 0)
        try:
            return _decode_planner_response(response).model_dump(mode="python")
        except StructuredOutputError as exc:
            if exc.reason is not StructuredOutputFailureReason.UNKNOWN_FIELD:
                raise
            legacy = _plain_json_object(raw)
            if legacy is None:
                raise
            extras = set(legacy).difference(PlannerContentV2.model_fields)
            if not extras or not extras.issubset({"plan_id", "original_question"}):
                raise
            for field_name in extras:
                legacy.pop(field_name, None)
            return decode_structured_output(PlannerContentV2, structured_payload=legacy).model_dump(mode="python")
    except StructuredOutputError as exc:
        raise AgenticPlannerError("planner_invalid_json", exc.reason.value) from exc


def _plain_json_object(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        stripped = stripped[7:-3].strip()
    elif stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped[3:-3].strip()
    try:
        value = json.loads(stripped)
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _decode_planner_response(response: StructuredOutputResult) -> PlannerContentV2:
    return decode_structured_output(
        PlannerContentV2,
        structured_payload=response.structured_payload,
        display_text=response.display_text,
        require_structured_payload=response.mode is not StructuredOutputMode.TEXT_ONLY,
    )


def _decode_judge_response(response: StructuredOutputResult) -> SufficiencyDecisionV2:
    return decode_structured_output(
        SufficiencyDecisionV2,
        structured_payload=response.structured_payload,
        display_text=response.display_text,
        require_structured_payload=response.mode is not StructuredOutputMode.TEXT_ONLY,
    )


def _planner_envelope_semantic_errors(content: PlannerContentV2, planner_input: dict[str, Any], cfg: AgenticRetrievalConfig) -> list[str]:
    errors: list[str] = []
    if len(content.subqueries) > cfg.max_subqueries:
        errors.append("subquery_limit_exceeded")
    allowed_doc_ids = {str(doc_id) for doc_id in ((planner_input.get("scope") or {}).get("doc_ids") or planner_input.get("doc_ids") or []) if str(doc_id)}
    for subquery in content.subqueries:
        if subquery.top_n > cfg.subquery_top_n:
            errors.append(f"{subquery.subquery_id}:top_n_out_of_bounds")
        if subquery.docid_scope is not None and (not allowed_doc_ids or any(doc_id not in allowed_doc_ids for doc_id in subquery.docid_scope)):
            errors.append(f"{subquery.subquery_id}:invalid_scope")
    return errors


async def _request_structured_output(
    chat_mdl: Any,
    system_prompt: str,
    messages: list[dict[str, str]],
    model: type[PlannerContentV2] | type[SufficiencyDecisionV2],
    *,
    gen_conf: dict[str, Any],
    tool_name: str,
    force_text: bool = False,
) -> StructuredOutputResult:
    native = getattr(chat_mdl, "async_structured_output", None)
    if callable(native):
        return await native(
            system_prompt,
            messages,
            canonical_json_schema(model),
            mode=StructuredOutputMode.TEXT_ONLY if force_text else None,
            gen_conf=gen_conf,
            tool_name=tool_name,
        )
    raw = await chat_mdl.async_chat(system_prompt, messages, gen_conf)
    if isinstance(raw, tuple):
        display_text, used_tokens = raw
    else:
        display_text, used_tokens = raw, 0
    return StructuredOutputResult(
        mode=StructuredOutputMode.TEXT_ONLY,
        structured_payload=None,
        display_text=display_text,
        used_tokens=int(used_tokens or 0),
    )


async def generate_llm_plan(
    planner_input: dict[str, Any],
    cfg: AgenticRetrievalConfig,
    *,
    chat_mdl: Any,
    rag_trace: Any = None,
    repair_errors: tuple[str, ...] = (),
    attempt_timeout_ms: int | None = None,
    operation_id: str | None = None,
    lifecycle_id: str | None = None,
    cleanup_callback: Any = None,
) -> BoundedRetrievalPlan:
    if chat_mdl is None or not any(callable(getattr(chat_mdl, name, None)) for name in ("async_structured_output", "async_chat")):
        _trace_planner_event(rag_trace, "planner_missing_chat_model", {"plan_id": planner_input.get("plan_id"), "mode": cfg.mode})
        raise AgenticPlannerError(
            "planner_missing_chat_model",
            "chat_model_unavailable",
            failure_class="configuration",
            safe_category="missing_chat_model",
        )
    plan_id = str(planner_input.get("plan_id") or f"plan-{uuid.uuid4().hex[:12]}")
    lifecycle_id = str(lifecycle_id or operation_id or f"planner.lifecycle.{uuid.uuid4().hex[:12]}")
    planner_input = {**planner_input, "plan_id": plan_id}
    system_prompt, messages = build_llm_planner_prompt(planner_input, cfg, repair_errors=repair_errors)
    gen_conf = {"temperature": 0.0, "top_p": 0.1}
    effective_timeout_ms = max(1, min(int(attempt_timeout_ms or cfg.planner_timeout_ms), int(cfg.planner_timeout_ms)))
    timeout_s = effective_timeout_ms / 1000.0
    model_name = str(getattr(chat_mdl, "llm_name", None) or getattr(chat_mdl, "model_name", None) or getattr(chat_mdl, "model_config", {}).get("llm_name", ""))
    started = time.monotonic()
    base_trace = {
        "plan_id": plan_id,
        "mode": cfg.mode,
        "model": model_name,
        "operation_id": operation_id,
        "lifecycle_id": lifecycle_id,
        "repair": bool(repair_errors),
        "attempt_timeout_ms": effective_timeout_ms,
    }
    _trace_planner_event(rag_trace, "planner_llm_start", base_trace)
    chat_task = asyncio.create_task(
        _request_structured_output(
            chat_mdl,
            system_prompt,
            messages,
            PlannerContentV2,
            gen_conf=gen_conf,
            tool_name="ragflow_retrieval_plan",
            force_text=bool(repair_errors),
        )
    )
    try:
        done, _ = await asyncio.wait({chat_task}, timeout=timeout_s)
    except asyncio.CancelledError:
        cleanup_ms = await _cancel_and_drain_tasks([chat_task])
        if callable(cleanup_callback):
            cleanup_callback(cleanup_ms, "upstream_cancellation")
        _trace_planner_event(
            rag_trace,
            "planner_llm_cleanup",
            {**base_trace, "cleanup_latency_ms": cleanup_ms, "timeout_owner": "upstream_cancellation", "admitted_new_work": False},
        )
        raise
    if not done:
        cleanup_ms = await _cancel_and_drain_tasks([chat_task])
        if callable(cleanup_callback):
            cleanup_callback(cleanup_ms, "planner_attempt")
        latency_ms = round((time.monotonic() - started) * 1000.0, 3)
        failure = classify_planner_exception(asyncio.TimeoutError())
        _trace_planner_event(
            rag_trace,
            "planner_llm_cleanup",
            {**base_trace, "cleanup_latency_ms": cleanup_ms, "timeout_owner": "planner_attempt", "admitted_new_work": False},
        )
        _trace_planner_event(
            rag_trace,
            "planner_llm_timeout",
            {**base_trace, "latency_ms": latency_ms, "cleanup_latency_ms": cleanup_ms, **failure, "timeout_owner": "planner_attempt"},
        )
        raise AgenticPlannerError("planner_timeout", "attempt_deadline_exceeded", **failure)
    try:
        response = await chat_task
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        latency_ms = round((time.monotonic() - started) * 1000.0, 3)
        failure = classify_planner_exception(exc)
        _trace_planner_event(rag_trace, "planner_llm_error", {**base_trace, "latency_ms": latency_ms, **failure})
        raise AgenticPlannerError(
            "planner_provider_error",
            failure_class=failure["failure_class"],
            retryable=failure["retryable"],
            status_code=failure["status_code"],
            bounded_retry_after_ms=failure["bounded_retry_after_ms"],
            safe_category=failure["safe_category"],
        ) from exc

    raw = response.display_text
    returned_failure = _native_provider_error_sentinel(raw)
    returned_error_shape = "native_sentinel"
    if returned_failure is None:
        returned_failure = _provider_error_envelope(raw)
        returned_error_shape = "provider_envelope"
    if returned_failure is not None:
        latency_ms = round((time.monotonic() - started) * 1000.0, 3)
        _trace_planner_event(
            rag_trace,
            "planner_llm_error",
            {**base_trace, "latency_ms": latency_ms, **returned_failure, "returned_error_shape": returned_error_shape},
        )
        raise AgenticPlannerError(
            "planner_provider_error",
            failure_class=returned_failure["failure_class"],
            retryable=returned_failure["retryable"],
            status_code=returned_failure["status_code"],
            bounded_retry_after_ms=returned_failure["bounded_retry_after_ms"],
            safe_category=returned_failure["safe_category"],
        )

    try:
        decoded = _decode_planner_response(response)
        raw_plan = decoded.model_dump(mode="python")
    except StructuredOutputError as exc:
        latency_ms = round((time.monotonic() - started) * 1000.0, 3)
        schema_reasons = {
            StructuredOutputFailureReason.UNKNOWN_FIELD,
            StructuredOutputFailureReason.MISSING_REQUIRED_FIELD,
            StructuredOutputFailureReason.WRONG_TYPE,
            StructuredOutputFailureReason.INVALID_ENUM,
            StructuredOutputFailureReason.VALUE_OUT_OF_BOUNDS,
            StructuredOutputFailureReason.ARRAY_TOO_LONG,
            StructuredOutputFailureReason.DUPLICATE_ITEM,
            StructuredOutputFailureReason.SEMANTIC_VALIDATION_FAILED,
            StructuredOutputFailureReason.UNSUPPORTED_ACTION,
        }
        failure_class = "validation" if exc.reason in schema_reasons else "invalid_json"
        event_name = "planner_llm_validation_failed" if failure_class == "validation" else "planner_llm_invalid_json"
        _trace_planner_event(
            rag_trace,
            event_name,
            {
                **base_trace,
                "latency_ms": latency_ms,
                "raw_response_chars": len(str(raw or "")),
                "fallback_reason": exc.reason.value,
                "validation_errors": [exc.reason.value] if failure_class == "validation" else [],
                "structured_output_mode": response.mode.value,
                "failure_class": failure_class,
                "retryable": True,
            },
        )
        raise AgenticPlannerError(
            "planner_invalid_json" if failure_class == "invalid_json" else "planner_validation_failed",
            exc.reason.value,
            failure_class=failure_class,
            retryable=True,
            safe_category=f"planner_{exc.reason.value}",
        ) from exc

    envelope_errors = _planner_envelope_semantic_errors(decoded, planner_input, cfg)
    if envelope_errors:
        latency_ms = round((time.monotonic() - started) * 1000.0, 3)
        _trace_planner_event(
            rag_trace,
            "planner_llm_validation_failed",
            {
                **base_trace,
                "latency_ms": latency_ms,
                "validation_errors": envelope_errors,
                "fallback_reason": ";".join(envelope_errors),
                "structured_output_mode": response.mode.value,
                "failure_class": "validation",
                "retryable": True,
            },
        )
        raise AgenticPlannerError(
            "planner_validation_failed",
            ";".join(envelope_errors),
            failure_class="validation",
            retryable=True,
            safe_category="planner_semantic_validation",
        )

    raw_plan["plan_id"] = plan_id
    raw_plan["original_question"] = str(planner_input.get("original_question") or planner_input.get("question") or raw_plan.get("original_question") or "")
    raw_plan["plan_origin"] = "llm_text" if response.mode is StructuredOutputMode.TEXT_ONLY else "llm_native"
    for subquery in raw_plan.get("subqueries") or []:
        if isinstance(subquery, dict):
            subquery["plan_id"] = plan_id
    validation = validate_plan(raw_plan, cfg, original_question=raw_plan["original_question"])
    latency_ms = round((time.monotonic() - started) * 1000.0, 3)
    if not validation.valid or validation.errors:
        safe_validation_errors = _safe_planner_validation_errors(validation.errors)
        policy_drift = any(error in {"anchor_drift", "new_entities"} for error in safe_validation_errors)
        failure_class = "policy_drift" if policy_drift else "validation"
        retryable = not policy_drift
        safe_category = "planner_policy_drift" if policy_drift else "planner_validation"
        _trace_planner_event(
            rag_trace,
            "planner_llm_validation_failed",
            {
                **base_trace,
                "latency_ms": latency_ms,
                "validation_errors": safe_validation_errors,
                "fallback_reason": ";".join(safe_validation_errors),
                "failure_class": failure_class,
                "retryable": retryable,
            },
        )
        raise AgenticPlannerError(
            "planner_validation_failed",
            ";".join(safe_validation_errors),
            failure_class=failure_class,
            retryable=retryable,
            safe_category=safe_category,
        )
    _trace_planner_event(
        rag_trace,
        "planner_llm_success",
        {
            **base_trace,
            "latency_ms": latency_ms,
            "model": model_name,
            "raw_response_chars": len(str(raw or "")),
            "structured_output_mode": response.mode.value,
            "structured_payload_present": response.structured_payload is not None,
            "schema_version": STRUCTURED_OUTPUT_SCHEMA_VERSION,
            "plan_origin": validation.plan.plan_origin,
            "subquery_count": len(validation.plan.subqueries),
            "facet_count": len(validation.plan.required_facets),
            "complexity": validation.plan.complexity,
        },
    )
    return validation.plan


def validate_plan(plan: BoundedRetrievalPlan | dict[str, Any], cfg: AgenticRetrievalConfig, original_question: str | None = None) -> PlanValidationResult:
    errors: list[str] = []
    try:
        plan = _coerce_plan(plan, cfg, original_question=original_question)
    except (TypeError, ValueError) as exc:
        fallback = f"schema_invalid:{exc}"
        empty = BoundedRetrievalPlan(
            plan_id="invalid-plan",
            original_question=original_question or "",
            complexity="unknown",
            trigger_reasons=(),
            required_facets=(),
            subqueries=(),
            merge_policy={},
            drift_controls={"anchor_entities": [], "min_anchor_overlap": cfg.min_anchor_overlap, "allow_new_entities": False},
            plan_origin="deterministic_fallback",
        )
        return PlanValidationResult(False, [fallback], empty, fallback_reason=fallback)
    if not plan.original_question.strip():
        errors.append("empty_original_question")
    if plan.complexity not in _ALLOWED_COMPLEXITIES:
        errors.append("unsupported_complexity")
    if not plan.subqueries:
        errors.append("no_subqueries")
    facets = {facet.facet_id: facet for facet in plan.required_facets}
    for facet in plan.required_facets:
        if facet.evidence_type not in _ALLOWED_EVIDENCE_TYPES:
            errors.append(f"{facet.facet_id}:unsupported_evidence_type")
    original_entities = set(_proper_entities(plan.original_question))
    original_anchor_keys = {anchor.lower() for anchor in _anchors_for_text(plan.original_question)}
    kept_subqueries: list[SubquerySpec] = []
    for subquery in plan.subqueries:
        facet = facets.get(subquery.facet_id)
        if not facet:
            errors.append(f"{subquery.subquery_id}:unknown_facet")
            continue
        if not subquery.query.strip():
            errors.append(f"{subquery.subquery_id}:empty_query")
            continue
        if subquery.top_n < 1 or subquery.top_n > max(cfg.subquery_top_n, 1):
            subquery = _replace_subquery(subquery, top_n=max(1, min(subquery.top_n, cfg.subquery_top_n)))
        if subquery.retrieval_variant not in {"hybrid_default", "keyword_first", "embedding_retry"}:
            errors.append(f"{subquery.subquery_id}:unsupported_variant")
            continue
        overlap = _anchor_overlap(subquery.query, facet.anchors or tuple(_anchors_for_text(plan.original_question)[:3]))
        if overlap < cfg.min_anchor_overlap:
            errors.append(f"{subquery.subquery_id}:anchor_drift")
            continue
        new_entities = {entity for entity in set(_proper_entities(subquery.query)).difference(original_entities) if entity.lower() not in original_anchor_keys}
        if new_entities:
            errors.append(f"{subquery.subquery_id}:new_entities:{','.join(sorted(new_entities))}")
            continue
        kept_subqueries.append(subquery)
        if len(kept_subqueries) >= cfg.max_subqueries:
            break
    if not kept_subqueries:
        errors.append("no_valid_subqueries")
    normalized = BoundedRetrievalPlan(
        plan_id=plan.plan_id,
        original_question=plan.original_question,
        complexity=plan.complexity,
        trigger_reasons=plan.trigger_reasons,
        required_facets=plan.required_facets,
        subqueries=tuple(kept_subqueries),
        merge_policy=plan.merge_policy,
        drift_controls=plan.drift_controls,
        plan_origin=plan.plan_origin,
    )
    fallback_reason = ";".join(errors) if errors else None
    return PlanValidationResult(fallback_reason is None and bool(kept_subqueries), errors, normalized, fallback_reason=fallback_reason)


def detect_missing_facets(question: str, plan: BoundedRetrievalPlan, kbinfos: dict[str, Any]) -> dict[str, Any]:
    """Return cheap coverage diagnostics without generating semantic followups."""
    chunks = [chunk for chunk in ((kbinfos or {}).get("chunks") or []) if isinstance(chunk, dict)]
    covered: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    exact_fact_gaps: list[dict[str, str]] = []
    contradictions: list[dict[str, Any]] = []
    for facet in plan.required_facets:
        supporting: list[dict[str, Any]] = []
        for chunk in chunks:
            metadata = chunk.get("_ragflow_agentic_retrieval") or {}
            facet_ids = set(str(value) for value in metadata.get("facet_ids") or [])
            facet_id = str(chunk.get("facet_id") or metadata.get("facet_id") or "")
            if facet.facet_id == facet_id or facet.facet_id in facet_ids or _text_has_any_anchor(_chunk_text(chunk), facet.anchors):
                supporting.append(chunk)
        combined = "\n".join(_chunk_text(chunk) for chunk in supporting)
        missing_anchors = [anchor for anchor in facet.anchors if not _phrase_in_text(anchor, combined)]
        evidence_ids = [_evidence_id(chunk) for chunk in supporting]
        unique_sources = {_chunk_source_id(chunk) for chunk in supporting if _chunk_source_id(chunk)}
        duplicate_count = len(supporting) - len({_refinement_chunk_identity(chunk) for chunk in supporting})
        support = "strong" if supporting and not missing_anchors and len(unique_sources) >= 2 and not duplicate_count else "weak"
        if not supporting:
            missing.append({"facet_id": facet.facet_id, "reason": "no_selected_evidence", "required_anchors": list(facet.anchors)})
        elif missing_anchors:
            missing.append({"facet_id": facet.facet_id, "reason": "required_anchors_absent", "required_anchors": missing_anchors})
        elif len(unique_sources) < 2:
            missing.append({"facet_id": facet.facet_id, "reason": "weak_source_diversity", "required_anchors": list(facet.anchors)})
        elif duplicate_count:
            missing.append({"facet_id": facet.facet_id, "reason": "redundant_selected_evidence", "required_anchors": list(facet.anchors)})
        else:
            covered.append({"facet_id": facet.facet_id, "evidence_ids": evidence_ids, "support": support})

        gap = _exact_fact_gap(question, facet, combined)
        if gap:
            exact_fact_gaps.append(gap)
            missing = [item for item in missing if item["facet_id"] != facet.facet_id or item["reason"] not in {"weak_source_diversity", "redundant_selected_evidence"}]
            if not any(item["facet_id"] == facet.facet_id and item["reason"] == "exact_fact_absent" for item in missing):
                missing.append({"facet_id": facet.facet_id, "reason": "exact_fact_absent", "required_anchors": list(facet.anchors)})

        conflicting_ids = [_evidence_id(chunk) for chunk in supporting if _chunk_has_contradiction_signal(chunk)]
        if conflicting_ids:
            contradictions.append(
                {
                    "facet_id": facet.facet_id,
                    "evidence_ids": conflicting_ids,
                    "description": "metadata_conflict_signal",
                }
            )
    coverage = len(covered) / max(1, len(plan.required_facets))
    return {
        "covered_facets": covered,
        "missing_facets": missing,
        "contradictions": contradictions,
        "exact_fact_gaps": exact_fact_gaps,
        "coverage": round(coverage, 6),
        "obviously_sufficient": bool(plan.required_facets) and not missing and not contradictions and not exact_fact_gaps and all(item["support"] == "strong" for item in covered),
    }


def parse_sufficiency_judge_json(raw: Any) -> dict[str, Any]:
    """Read historical judge JSON; live provider output uses the strict decoder."""
    if not isinstance(raw, str):
        raise AgenticRefinementError("refinement_judge_invalid_json", f"non_string_response:{type(raw).__name__}")
    text = raw.strip()
    if not text:
        raise AgenticRefinementError("refinement_judge_invalid_json", "empty_response")
    if "```" in text:
        raise AgenticRefinementError("refinement_judge_invalid_json", "markdown_fence")
    if not text.startswith("{") or not text.endswith("}"):
        raise AgenticRefinementError("refinement_judge_invalid_json", "response_not_single_json_object")
    try:
        parsed, end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError as exc:
        raise AgenticRefinementError("refinement_judge_invalid_json", str(exc)) from exc
    if text[end:].strip():
        raise AgenticRefinementError("refinement_judge_invalid_json", "multiple_json_or_trailing_text")
    if not isinstance(parsed, dict) or not parsed:
        raise AgenticRefinementError("refinement_judge_invalid_json", "json_not_nonempty_object")
    if "action" in parsed:
        missing = sorted(_REQUIRED_JUDGE_V2_KEYS.difference(parsed))
    else:
        missing = sorted(_REQUIRED_JUDGE_KEYS.difference(parsed))
    if missing:
        raise AgenticRefinementError("refinement_judge_validation_failed", "missing_keys:" + ",".join(missing))
    return parsed


def validate_sufficiency_judge(
    raw: dict[str, Any],
    plan: BoundedRetrievalPlan,
    cfg: AgenticRefinementConfig,
    iteration_id: str = "",
    selected_evidence_ids: set[str] | None = None,
) -> SufficiencyJudge:
    if not isinstance(raw, dict) or not raw:
        raise AgenticRefinementError("refinement_judge_validation_failed", "judge_not_nonempty_object")
    raw = dict(raw)
    if "action" in raw:
        if "sufficient" in raw:
            raise AgenticRefinementError("refinement_judge_validation_failed", "action_and_sufficient_are_mutually_exclusive")
        action = raw.pop("action")
        if action not in {"sufficient", "refine"}:
            raise AgenticRefinementError("refinement_judge_unsupported_action", str(action))
        raw["sufficient"] = action == "sufficient"
    missing_keys = sorted(_REQUIRED_JUDGE_KEYS.difference(raw))
    if missing_keys:
        raise AgenticRefinementError("refinement_judge_validation_failed", "missing_keys:" + ",".join(missing_keys))
    unknown_keys = sorted(set(raw).difference(_REQUIRED_JUDGE_KEYS))
    if unknown_keys:
        raise AgenticRefinementError("refinement_judge_validation_failed", "unknown_keys:" + ",".join(unknown_keys))
    if not isinstance(raw["sufficient"], bool) or not isinstance(raw["refusal_justified"], bool):
        raise AgenticRefinementError("refinement_judge_validation_failed", "boolean_fields_invalid")
    confidence = raw["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        raise AgenticRefinementError("refinement_judge_validation_failed", "confidence_out_of_range")
    for key in ("covered_facets", "missing_facets", "contradictions", "exact_fact_gaps", "recommended_followups"):
        if not isinstance(raw[key], list):
            raise AgenticRefinementError("refinement_judge_validation_failed", f"{key}_not_array")

    facet_ids = {facet.facet_id for facet in plan.required_facets}
    covered: list[dict[str, Any]] = []
    for index, item in enumerate(raw["covered_facets"], start=1):
        if not isinstance(item, dict):
            raise AgenticRefinementError("refinement_judge_validation_failed", f"covered_facet_{index}_not_object")
        if set(item) != {"facet_id", "evidence_ids", "support"}:
            raise AgenticRefinementError("refinement_judge_validation_failed", f"covered_facet_{index}_schema_invalid")
        facet_id = str(item.get("facet_id") or "")
        support = str(item.get("support") or "")
        evidence_ids = item.get("evidence_ids")
        if facet_id not in facet_ids:
            raise AgenticRefinementError("refinement_judge_validation_failed", f"unknown_facet:{facet_id}")
        if support not in _SUPPORTED_FACET_SUPPORT:
            raise AgenticRefinementError("refinement_judge_validation_failed", f"unsupported_support:{support}")
        if not isinstance(evidence_ids, list) or not evidence_ids or any(not isinstance(value, str) or not value.strip() for value in evidence_ids):
            raise AgenticRefinementError("refinement_judge_validation_failed", f"invalid_evidence_ids:{facet_id}")
        if selected_evidence_ids is not None and any(value not in selected_evidence_ids for value in evidence_ids):
            raise AgenticRefinementError("refinement_judge_validation_failed", f"unknown_evidence_id:{facet_id}")
        covered.append({"facet_id": facet_id, "evidence_ids": list(evidence_ids), "support": support})

    missing: list[dict[str, Any]] = []
    for index, item in enumerate(raw["missing_facets"], start=1):
        if not isinstance(item, dict):
            raise AgenticRefinementError("refinement_judge_validation_failed", f"missing_facet_{index}_not_object")
        if set(item) != {"facet_id", "reason", "required_anchors"}:
            raise AgenticRefinementError("refinement_judge_validation_failed", f"missing_facet_{index}_schema_invalid")
        facet_id = str(item.get("facet_id") or "")
        anchors = item.get("required_anchors")
        if facet_id not in facet_ids:
            raise AgenticRefinementError("refinement_judge_validation_failed", f"unknown_facet:{facet_id}")
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            raise AgenticRefinementError("refinement_judge_validation_failed", f"missing_reason:{facet_id}")
        if not isinstance(anchors, list) or any(not isinstance(value, str) or not value.strip() for value in anchors):
            raise AgenticRefinementError("refinement_judge_validation_failed", f"invalid_required_anchors:{facet_id}")
        facet = next(facet for facet in plan.required_facets if facet.facet_id == facet_id)
        grounding_text = "\n".join((plan.original_question, facet.description, *facet.anchors))
        if any(not _phrase_in_text(anchor, grounding_text) for anchor in anchors):
            raise AgenticRefinementError("refinement_judge_validation_failed", f"ungrounded_required_anchor:{facet_id}")
        missing.append({"facet_id": facet_id, "reason": item["reason"].strip(), "required_anchors": list(anchors)})
    missing_ids = {item["facet_id"] for item in missing}
    covered_ids = {item["facet_id"] for item in covered}
    if covered_ids.intersection(missing_ids):
        raise AgenticRefinementError("refinement_judge_validation_failed", "facet_both_covered_and_missing")

    contradictions: list[dict[str, Any]] = []
    for index, item in enumerate(raw["contradictions"], start=1):
        if not isinstance(item, dict):
            raise AgenticRefinementError("refinement_judge_validation_failed", f"contradiction_{index}_not_object")
        if set(item) != {"facet_id", "evidence_ids", "description"}:
            raise AgenticRefinementError("refinement_judge_validation_failed", f"contradiction_{index}_schema_invalid")
        facet_id = str(item.get("facet_id") or "")
        evidence_ids = item.get("evidence_ids")
        if facet_id not in facet_ids:
            raise AgenticRefinementError("refinement_judge_validation_failed", f"unknown_facet:{facet_id}")
        if not isinstance(evidence_ids, list) or len(evidence_ids) < 2 or any(not isinstance(value, str) or not value.strip() for value in evidence_ids):
            raise AgenticRefinementError("refinement_judge_validation_failed", f"invalid_contradiction_evidence:{facet_id}")
        if selected_evidence_ids is not None and any(value not in selected_evidence_ids for value in evidence_ids):
            raise AgenticRefinementError("refinement_judge_validation_failed", f"unknown_contradiction_evidence_id:{facet_id}")
        if not isinstance(item.get("description"), str) or not item["description"].strip():
            raise AgenticRefinementError("refinement_judge_validation_failed", f"invalid_contradiction_description:{facet_id}")
        contradictions.append({"facet_id": facet_id, "evidence_ids": list(evidence_ids), "description": item["description"].strip()})

    gaps: list[dict[str, str]] = []
    for index, item in enumerate(raw["exact_fact_gaps"], start=1):
        if not isinstance(item, dict):
            raise AgenticRefinementError("refinement_judge_validation_failed", f"exact_fact_gap_{index}_not_object")
        if set(item) != {"type", "description"}:
            raise AgenticRefinementError("refinement_judge_validation_failed", f"exact_fact_gap_{index}_schema_invalid")
        gap_type = str(item.get("type") or "")
        description = str(item.get("description") or "").strip()
        if gap_type not in _SUPPORTED_EXACT_FACT_TYPES or not description:
            raise AgenticRefinementError("refinement_judge_validation_failed", f"invalid_exact_fact_gap:{index}")
        gaps.append({"type": gap_type, "description": description})

    if len(raw["recommended_followups"]) > cfg.normalized().max_followup_queries:
        raise AgenticRefinementError("refinement_judge_validation_failed", "followup_limit_exceeded")
    followups: list[FollowupQuerySpec] = []
    for index, item in enumerate(raw["recommended_followups"], start=1):
        if not isinstance(item, dict):
            raise AgenticRefinementError("refinement_judge_validation_failed", f"followup_{index}_not_object")
        if set(item) != {"facet_id", "query", "keywords", "top_n"}:
            raise AgenticRefinementError("refinement_judge_validation_failed", f"followup_{index}_schema_invalid")
        facet_id = str(item.get("facet_id") or "")
        query = str(item.get("query") or "").strip()
        keywords = item.get("keywords")
        top_n = item.get("top_n")
        if facet_id not in missing_ids:
            raise AgenticRefinementError("refinement_judge_validation_failed", f"followup_facet_not_missing:{facet_id}")
        if not query:
            raise AgenticRefinementError("refinement_judge_validation_failed", f"followup_empty_query:{index}")
        if keywords is not None and not isinstance(keywords, str):
            raise AgenticRefinementError("refinement_judge_validation_failed", f"followup_keywords_invalid:{index}")
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1 or top_n > _MAX_REFINEMENT_FOLLOWUP_TOP_N:
            raise AgenticRefinementError("refinement_judge_validation_failed", f"followup_top_n_invalid:{index}")
        followups.append(
            FollowupQuerySpec(
                plan_id=plan.plan_id,
                iteration_id=iteration_id,
                followup_id=f"{iteration_id or 'iteration'}.followup.{index}",
                facet_id=facet_id,
                query=query,
                keywords=keywords,
                top_n=top_n,
            )
        )
    if raw["sufficient"] and (missing or contradictions or gaps or followups):
        raise AgenticRefinementError("refinement_judge_validation_failed", "sufficient_with_unresolved_gaps")
    return SufficiencyJudge(
        sufficient=raw["sufficient"],
        confidence=float(confidence),
        covered_facets=tuple(covered),
        missing_facets=tuple(missing),
        contradictions=tuple(contradictions),
        exact_fact_gaps=tuple(gaps),
        refusal_justified=raw["refusal_justified"],
        recommended_followups=tuple(followups),
    )


def build_sufficiency_judge_prompt(
    question: str,
    plan: BoundedRetrievalPlan,
    kbinfos: dict[str, Any],
    cfg: AgenticRefinementConfig,
    *,
    repair_errors: tuple[str, ...] = (),
) -> tuple[str, list[dict[str, str]]]:
    facets = [asdict(facet) for facet in plan.required_facets]
    evidence = [
        {
            "evidence_id": _evidence_id(chunk),
            "document": str(chunk.get("docnm_kwd") or chunk.get("document_name") or chunk.get("title") or "")[:200],
            "text": re.sub(r"\s+", " ", str(chunk.get("content_with_weight") or chunk.get("content") or "")).strip()[:800],
            "facet_ids": _chunk_facet_ids(chunk),
            "capability": {key: value for key, value in (chunk.get("_ragflow_compilation") or {}).items() if key in {"source_grounded", "citation_mode", "route_occurrences"}},
        }
        for chunk in ((kbinfos or {}).get("chunks") or [])[:20]
        if isinstance(chunk, dict)
    ]
    system_prompt = (
        "You are a sufficiency judge for RAGFlow retrieval evidence.\n"
        "Evaluate evidence only; do not answer the user, retrieve documents, or call tools.\n"
        "Recommended followups may target only missing facet IDs from the supplied Phase 5 plan.\n" + canonical_output_instructions(SufficiencyDecisionV2)
    )
    payload = {
        "original_question": question,
        "plan_id": plan.plan_id,
        "required_facets": facets,
        "selected_evidence": evidence,
        "limits": {"max_followup_queries": cfg.normalized().max_followup_queries, "max_top_n": _MAX_REFINEMENT_FOLLOWUP_TOP_N},
        "output_schema": canonical_json_schema(SufficiencyDecisionV2),
    }
    user_content = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if repair_errors:
        user_content += "\nThe previous response failed. Return one complete replacement canonical object, not a patch.\nApplication failure codes: " + json.dumps(
            list(repair_errors), ensure_ascii=True, sort_keys=True
        )
    return system_prompt, [{"role": "user", "content": user_content}]


async def generate_sufficiency_judge(
    *,
    question: str,
    plan: BoundedRetrievalPlan,
    kbinfos: dict[str, Any],
    cfg: AgenticRefinementConfig,
    chat_mdl: Any,
    rag_trace: Any = None,
    iteration_id: str = "",
    lifecycle_id: str | None = None,
    timeout_ms: int | None = None,
    remaining_budget_ms: float | None = None,
    cleanup_callback: Any = None,
    _repair_errors: tuple[str, ...] = (),
) -> SufficiencyJudge:
    if chat_mdl is None or not any(callable(getattr(chat_mdl, name, None)) for name in ("async_structured_output", "async_chat")):
        raise AgenticRefinementError("refinement_judge_missing_chat_model")
    system_prompt, messages = build_sufficiency_judge_prompt(question, plan, kbinfos, cfg, repair_errors=_repair_errors)
    lifecycle_id = str(lifecycle_id or f"{plan.plan_id}:refinement")
    started = time.monotonic()
    effective_timeout_ms = max(1, min(int(timeout_ms or cfg.normalized().judge_timeout_ms), cfg.normalized().judge_timeout_ms))
    _trace_refinement_event(
        rag_trace,
        "refinement_judge_start",
        {
            "plan_id": plan.plan_id,
            "iteration_id": iteration_id,
            "lifecycle_id": lifecycle_id,
            "mode": cfg.mode,
            "judge_timeout_ms": effective_timeout_ms,
            "remaining_refinement_budget_ms": remaining_budget_ms,
            "repair": bool(_repair_errors),
        },
    )
    judge_task = asyncio.create_task(
        _request_structured_output(
            chat_mdl,
            system_prompt,
            messages,
            SufficiencyDecisionV2,
            gen_conf={"temperature": 0.0, "top_p": 0.1},
            tool_name="ragflow_sufficiency_decision",
            force_text=bool(_repair_errors),
        )
    )
    try:
        done, _ = await asyncio.wait(
            {judge_task},
            timeout=effective_timeout_ms / 1000.0,
        )
    except asyncio.CancelledError:
        cleanup_ms = await _cancel_and_drain_tasks([judge_task])
        if callable(cleanup_callback):
            cleanup_callback(cleanup_ms, "upstream_cancellation")
        _trace_refinement_event(
            rag_trace,
            "refinement_cleanup",
            {
                "plan_id": plan.plan_id,
                "iteration_id": iteration_id,
                "lifecycle_id": lifecycle_id,
                "mode": cfg.mode,
                "cleanup_latency_ms": cleanup_ms,
                "timeout_owner": "upstream_cancellation",
                "admitted_new_work": False,
            },
        )
        raise
    if not done:
        cleanup_ms = await _cancel_and_drain_tasks([judge_task])
        if callable(cleanup_callback):
            cleanup_callback(cleanup_ms, "phase6_judge")
        latency_ms = round((time.monotonic() - started) * 1000.0, 3)
        _trace_refinement_event(
            rag_trace,
            "refinement_cleanup",
            {
                "plan_id": plan.plan_id,
                "iteration_id": iteration_id,
                "lifecycle_id": lifecycle_id,
                "mode": cfg.mode,
                "cleanup_latency_ms": cleanup_ms,
                "timeout_owner": "phase6_judge",
                "admitted_new_work": False,
            },
        )
        _trace_refinement_event(
            rag_trace,
            "refinement_judge_timeout",
            {
                "plan_id": plan.plan_id,
                "iteration_id": iteration_id,
                "lifecycle_id": lifecycle_id,
                "mode": cfg.mode,
                "latency_ms": latency_ms,
                "cleanup_latency_ms": cleanup_ms,
                "judge_timeout_ms": effective_timeout_ms,
                "judge_outcome": "timeout",
                "remaining_refinement_budget_ms": remaining_budget_ms,
                "timeout_owner": "phase6_judge",
            },
        )
        raise AgenticRefinementError("refinement_judge_timeout")
    try:
        response = await judge_task
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _trace_refinement_event(
            rag_trace, "refinement_judge_validation_failed", {"plan_id": plan.plan_id, "iteration_id": iteration_id, "lifecycle_id": lifecycle_id, "mode": cfg.mode, "fallback_reason": "judge_error"}
        )
        raise AgenticRefinementError("refinement_judge_error", "provider_error") from exc
    try:
        decoded = _decode_judge_response(response)
        parsed = decoded.model_dump(mode="python")
        selected_evidence_ids = {_evidence_id(chunk) for chunk in ((kbinfos or {}).get("chunks") or []) if isinstance(chunk, dict)}
        judge = validate_sufficiency_judge(
            parsed,
            plan,
            cfg,
            iteration_id=iteration_id,
            selected_evidence_ids=selected_evidence_ids,
        )
    except (AgenticRefinementError, StructuredOutputError) as exc:
        latency_ms = round((time.monotonic() - started) * 1000.0, 3)
        if isinstance(exc, StructuredOutputError):
            detail = exc.reason.value
            syntax_reasons = {
                StructuredOutputFailureReason.EMPTY_RESPONSE,
                StructuredOutputFailureReason.PROVIDER_STRUCTURED_PAYLOAD_MISSING,
                StructuredOutputFailureReason.UNEXPECTED_TEXT,
                StructuredOutputFailureReason.JSON_OBJECT_NOT_FOUND,
                StructuredOutputFailureReason.JSON_DECODE_ERROR,
                StructuredOutputFailureReason.TRUNCATED_OUTPUT,
            }
            reason = "refinement_judge_invalid_json" if exc.reason in syntax_reasons else "refinement_judge_validation_failed"
        else:
            detail = exc.detail or exc.reason
            reason = exc.reason
        event = "refinement_judge_invalid_json" if reason == "refinement_judge_invalid_json" else "refinement_judge_validation_failed"
        _trace_refinement_event(
            rag_trace,
            event,
            {
                "plan_id": plan.plan_id,
                "iteration_id": iteration_id,
                "lifecycle_id": lifecycle_id,
                "mode": cfg.mode,
                "latency_ms": latency_ms,
                "fallback_reason": reason,
                "failure_class": detail,
                "structured_output_mode": response.mode.value,
                "repair": bool(_repair_errors),
            },
        )
        if not _repair_errors:
            remaining_ms = max(1, effective_timeout_ms - int(latency_ms))
            try:
                return await generate_sufficiency_judge(
                    question=question,
                    plan=plan,
                    kbinfos=kbinfos,
                    cfg=cfg,
                    chat_mdl=chat_mdl,
                    rag_trace=rag_trace,
                    iteration_id=iteration_id,
                    lifecycle_id=lifecycle_id,
                    timeout_ms=remaining_ms,
                    remaining_budget_ms=remaining_budget_ms,
                    cleanup_callback=cleanup_callback,
                    _repair_errors=(detail,),
                )
            except AgenticRefinementError as repair_exc:
                _trace_refinement_event(
                    rag_trace,
                    "refinement_judge_repair_failed",
                    {
                        "plan_id": plan.plan_id,
                        "iteration_id": iteration_id,
                        "lifecycle_id": lifecycle_id,
                        "mode": cfg.mode,
                        "fallback_reason": "repair_failed",
                        "failure_class": StructuredOutputFailureReason.REPAIR_FAILED.value,
                        "initial_failure": detail,
                        "repair_failure": repair_exc.detail or repair_exc.reason,
                        "structured_output_mode": response.mode.value,
                        "repair": True,
                    },
                )
                raise AgenticRefinementError(reason, f"repair_failed:{repair_exc.detail or repair_exc.reason}") from repair_exc
        if isinstance(exc, AgenticRefinementError):
            raise
        raise AgenticRefinementError(reason, detail) from exc
    latency_ms = round((time.monotonic() - started) * 1000.0, 3)
    _trace_refinement_event(
        rag_trace,
        "refinement_judge_success",
        {
            "plan_id": plan.plan_id,
            "iteration_id": iteration_id,
            "lifecycle_id": lifecycle_id,
            "mode": cfg.mode,
            "latency_ms": latency_ms,
            "judge_timeout_ms": effective_timeout_ms,
            "judge_outcome": "success",
            "remaining_refinement_budget_ms": remaining_budget_ms,
            "confidence": judge.confidence,
            "sufficient": judge.sufficient,
            "missing_facet_count": len(judge.missing_facets),
            "contradiction_count": len(judge.contradictions),
            "followup_count": len(judge.recommended_followups),
            "structured_output_mode": response.mode.value,
            "structured_payload_present": response.structured_payload is not None,
            "schema_version": STRUCTURED_OUTPUT_SCHEMA_VERSION,
            "repair": bool(_repair_errors),
        },
    )
    return judge


def validate_followup_queries(
    judge: SufficiencyJudge,
    plan: BoundedRetrievalPlan,
    cfg: AgenticRefinementConfig,
    iteration_id: str = "",
) -> tuple[tuple[FollowupQuerySpec, ...], tuple[dict[str, Any], ...]]:
    config = cfg.normalized()
    missing_ids = {item["facet_id"] for item in judge.missing_facets}
    missing_required_anchors = {item["facet_id"]: tuple(item.get("required_anchors") or ()) for item in judge.missing_facets}
    facets = {facet.facet_id: facet for facet in plan.required_facets}
    accepted: list[FollowupQuerySpec] = []
    rejected: list[dict[str, Any]] = []
    for index, followup in enumerate(judge.recommended_followups[: config.max_followup_queries], start=1):
        reason = None
        facet = facets.get(followup.facet_id)
        if facet is None:
            reason = "unknown_facet"
        elif followup.facet_id not in missing_ids:
            reason = "facet_not_missing"
        elif not followup.query.strip():
            reason = "empty_query"
        else:
            allowed_anchors = _required_drift_anchors(
                facet,
                plan.original_question,
                missing_required_anchors.get(followup.facet_id, ()),
            )
            if not _text_satisfies_required_anchors(followup.query, allowed_anchors):
                reason = "anchor_drift"
            allowed_entity_text = "\n".join((plan.original_question, facet.description, *facet.anchors))
            new_entities = [entity for entity in _proper_entities_for_drift(followup.query) if not _phrase_in_text(entity, allowed_entity_text)]
            if new_entities:
                reason = "unrelated_proper_noun"
        if reason:
            rejected.append({"followup_id": followup.followup_id, "facet_id": followup.facet_id, "rejection_reason": reason})
            continue
        accepted.append(
            replace(
                followup,
                plan_id=plan.plan_id,
                iteration_id=iteration_id or followup.iteration_id,
                followup_id=followup.followup_id or f"{iteration_id or 'iteration'}.followup.{index}",
                top_n=max(1, min(followup.top_n, _MAX_REFINEMENT_FOLLOWUP_TOP_N)),
            )
        )
    return tuple(accepted), tuple(rejected)


def accept_followup_evidence(
    *,
    current_kbinfos: dict[str, Any],
    candidate_kbinfos: dict[str, Any] | None = None,
    lane_results: list[dict[str, Any]],
    plan: BoundedRetrievalPlan,
    missing_facet_ids: set[str],
    missing_required_anchors: dict[str, tuple[str, ...]] | None = None,
    question: str,
    context_builder_config: Any = None,
    rag_trace: Any = None,
) -> EvidenceAcceptanceResult:
    current = deepcopy(current_kbinfos or {"total": 0, "chunks": [], "doc_aggs": []})
    candidates = deepcopy(candidate_kbinfos if candidate_kbinfos is not None else current)
    selected_chunks = [chunk for chunk in (current.get("chunks") or []) if isinstance(chunk, dict)]
    candidate_chunks = [chunk for chunk in (candidates.get("chunks") or []) if isinstance(chunk, dict)]
    existing_keys = {_refinement_chunk_identity(chunk) for chunk in candidate_chunks}
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    facets = {facet.facet_id: facet for facet in plan.required_facets}
    for lane in lane_results:
        followup = lane.get("followup")
        if not isinstance(followup, FollowupQuerySpec):
            spec = lane.get("subquery")
            if isinstance(spec, SubquerySpec):
                followup = FollowupQuerySpec(
                    plan_id=plan.plan_id, facet_id=spec.facet_id, query=spec.query, followup_id=spec.subquery_id, keywords=spec.keywords, top_n=spec.top_n, retrieval_variant=spec.retrieval_variant
                )
        chunks = [chunk for chunk in ((lane.get("kbinfos") or {}).get("chunks") or []) if isinstance(chunk, dict)]
        if not lane.get("accepted") or not isinstance(followup, FollowupQuerySpec):
            for chunk in chunks:
                rejected.append(_mark_refinement_chunk(chunk, followup, selected=False, reason=str(lane.get("rejection_reason") or "retrieval_rejected")))
            continue
        facet = facets.get(followup.facet_id)
        allowed_anchors = (
            _required_drift_anchors(
                facet,
                question,
                (missing_required_anchors or {}).get(followup.facet_id, ()),
            )
            if facet
            else tuple(_anchors_for_text(question))
        )
        existing_facet_quality = max(
            (
                quality
                for chunk in selected_chunks
                if (followup.facet_id in _chunk_facet_ids(chunk) or _text_satisfies_required_anchors(_chunk_text(chunk), allowed_anchors))
                for quality in [_chunk_quality(chunk)]
                if quality is not None
            ),
            default=None,
        )
        group_text = "\n".join(_chunk_text(chunk) for chunk in chunks[: min(3, len(chunks))])
        if followup.facet_id not in missing_facet_ids or not _text_satisfies_required_anchors(group_text, allowed_anchors):
            for chunk in chunks:
                rejected.append(_mark_refinement_chunk(chunk, followup, selected=False, reason="result_facet_or_title_drift"))
            continue
        for chunk in chunks:
            key = _refinement_chunk_identity(chunk)
            if key in existing_keys:
                rejected.append(_mark_refinement_chunk(chunk, followup, selected=False, reason="duplicate_evidence"))
                continue
            chunk_text = _chunk_text(chunk)
            if not _text_satisfies_required_anchors(chunk_text, allowed_anchors):
                rejected.append(_mark_refinement_chunk(chunk, followup, selected=False, reason="evidence_anchor_drift"))
                continue
            quality = _chunk_quality(chunk)
            if quality is not None and existing_facet_quality is not None and quality < existing_facet_quality:
                rejected.append(_mark_refinement_chunk(chunk, followup, selected=False, reason="lower_quality_than_selected_evidence"))
                continue
            marked = _mark_refinement_chunk(chunk, followup, selected=None, reason=None)
            eligible.append(marked)
            existing_keys.add(key)

    if not eligible:
        return EvidenceAcceptanceResult(kbinfos=current, accepted_chunks=(), rejected_chunks=tuple(rejected), candidate_kbinfos=candidates)
    staged_candidate = deepcopy(candidates)
    staged_candidate["chunks"] = candidate_chunks + eligible
    staged_candidate["total"] = len(staged_candidate["chunks"])
    staged_candidate["doc_aggs"] = _merge_doc_aggs([], staged_candidate["chunks"])
    selected = deepcopy(staged_candidate)
    if context_builder_config is not None and getattr(context_builder_config, "enabled", False):
        from rag.utils.context_builder import apply_context_builder_to_kbinfos

        bundle = apply_context_builder_to_kbinfos(staged_candidate, context_builder_config, query=question)
        selected = bundle.kbinfos
        if getattr(bundle, "bundle", None) is not None:
            _trace_refinement_event(
                rag_trace,
                "refinement_context_builder_summary",
                {"plan_id": plan.plan_id, "context_summary": bundle.bundle.summary()},
            )
    selected_keys = {_refinement_chunk_identity(chunk) for chunk in selected.get("chunks", []) if isinstance(chunk, dict)}
    accepted: list[dict[str, Any]] = []
    for chunk in eligible:
        if _refinement_chunk_identity(chunk) in selected_keys:
            accepted.append(_mark_refinement_chunk(chunk, _followup_from_chunk(chunk), selected=True, reason=None))
        else:
            rejected.append(_mark_refinement_chunk(chunk, _followup_from_chunk(chunk), selected=False, reason="context_builder_rejected"))
    if not accepted:
        return EvidenceAcceptanceResult(kbinfos=current, accepted_chunks=(), rejected_chunks=tuple(rejected), candidate_kbinfos=candidates)
    accepted_by_key = {_refinement_chunk_identity(chunk): chunk for chunk in accepted}
    selected["chunks"] = [accepted_by_key.get(_refinement_chunk_identity(chunk), chunk) for chunk in selected.get("chunks", []) if isinstance(chunk, dict)]
    selected["total"] = len(selected["chunks"])
    selected["doc_aggs"] = _merge_doc_aggs([], selected["chunks"])
    committed_candidate = deepcopy(candidates)
    committed_candidate["chunks"] = candidate_chunks + accepted
    committed_candidate["total"] = len(committed_candidate["chunks"])
    committed_candidate["doc_aggs"] = _merge_doc_aggs([], committed_candidate["chunks"])
    return EvidenceAcceptanceResult(kbinfos=selected, accepted_chunks=tuple(accepted), rejected_chunks=tuple(rejected), candidate_kbinfos=committed_candidate)


async def _cancel_and_drain_tasks(tasks: list[asyncio.Task[Any]]) -> float:
    cleanup_started = time.monotonic()
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    return round((time.monotonic() - cleanup_started) * 1000.0, 3)


async def _wait_for_ordered_task_results(tasks: list[asyncio.Task[Any]], timeout_s: float) -> list[Any]:
    if not tasks:
        return []
    done, pending = await asyncio.wait(tasks, timeout=timeout_s)
    # A child cancellation is an upstream cancellation signal even when a
    # sibling is still pending; callers will cancel and drain the full set.
    if any(task.cancelled() for task in done):
        raise asyncio.CancelledError
    if pending:
        raise asyncio.TimeoutError
    results: list[Any] = []
    for task in tasks:
        if task.cancelled():
            raise asyncio.CancelledError
        exception = task.exception()
        results.append(exception if exception is not None else task.result())
    return results


async def run_refinement_loop(
    *,
    question: str,
    plan: BoundedRetrievalPlan,
    kbinfos: dict[str, Any],
    candidate_kbinfos: dict[str, Any] | None = None,
    retriever: Any,
    chat_mdl: Any,
    embd_mdl: Any,
    tenant_ids: list[str],
    kb_ids: list[str],
    doc_ids: list[str] | None,
    similarity_threshold: float,
    vector_similarity_weight: float,
    top_k: int,
    rank_feature: dict[str, Any] | None,
    refinement_cfg: AgenticRefinementConfig = AgenticRefinementConfig(),
    retrieval_cfg: AgenticRetrievalConfig = AgenticRetrievalConfig(),
    context_builder_config: Any = None,
    rag_trace: Any = None,
    metadata_filters: Any = None,
    rerank_mdl: Any = None,
    apply_children: bool = True,
    apply_toc: bool = False,
    cleanup_callback: Any = None,
) -> RefinementResult:
    """Run the bounded Phase 6 loop without mutating the caller's Phase 5 state."""
    config = refinement_cfg.normalized()
    previous = deepcopy(kbinfos or {"total": 0, "chunks": [], "doc_aggs": []})
    previous_candidates = deepcopy(candidate_kbinfos if candidate_kbinfos is not None else previous)
    if not config.enabled or config.mode == "off":
        _trace_refinement_event(rag_trace, "refinement_skip", {"mode": config.mode, "plan_id": getattr(plan, "plan_id", None), "stop_reason": "disabled"})
        return RefinementResult(kbinfos=previous, candidate_kbinfos=previous_candidates, iterations=(), stop_reason="disabled")
    if not isinstance(plan, BoundedRetrievalPlan) or not plan.required_facets or not question.strip():
        _trace_refinement_event(rag_trace, "refinement_skip", {"mode": config.mode, "plan_id": getattr(plan, "plan_id", None), "stop_reason": "phase5_plan_unavailable"})
        return RefinementResult(
            kbinfos=previous, candidate_kbinfos=previous_candidates, iterations=(), fallback_to_previous_context=True, fallback_reason="phase5_plan_unavailable", stop_reason="phase5_plan_unavailable"
        )

    refinement_lifecycle_id = f"{plan.plan_id}:refinement"
    initial_diagnostics = detect_missing_facets(question, plan, previous)
    _trace_refinement_event(
        rag_trace,
        "refinement_start",
        {
            "mode": config.mode,
            "plan_id": plan.plan_id,
            "facet_ids": [facet.facet_id for facet in plan.required_facets],
            "coverage_before": initial_diagnostics["coverage"],
            "missing_facet_count": len(initial_diagnostics["missing_facets"]),
        },
    )
    if config.mode == "diagnostic":
        diagnostic_payload = {
            "mode": config.mode,
            "plan_id": plan.plan_id,
            "stop_reason": "diagnostic_only",
            "coverage_before": initial_diagnostics["coverage"],
            "coverage_after": initial_diagnostics["coverage"],
            "marginal_gain": 0.0,
            "followup_count": 0,
            "rejected_followup_count": 0,
        }
        _trace_refinement_event(rag_trace, "refinement_stop", diagnostic_payload)
        return RefinementResult(
            kbinfos=previous, candidate_kbinfos=previous_candidates, iterations=(), stop_reason="diagnostic_only", diagnostics={"heuristic": initial_diagnostics, **diagnostic_payload}
        )

    current = previous
    current_candidates = previous_candidates
    iterations: list[RefinementIteration] = []
    accepted_all: list[dict[str, Any]] = []
    rejected_all: list[dict[str, Any]] = []
    calls_used = 0
    loop_started = time.monotonic()
    previous_contradictions: tuple[tuple[str, tuple[str, ...]], ...] = ()
    initial_coverage = float(initial_diagnostics["coverage"])
    current_coverage = initial_coverage
    followup_count_total = 0
    rejected_followup_count_total = 0
    cleanup_ms_total = 0.0

    def stop_result(
        stop_reason: str,
        latency_ms: float,
        *,
        fallback: bool = False,
        fallback_reason: str | None = None,
        timeout_owner: str | None = None,
    ) -> RefinementResult:
        if latency_ms >= config.latency_budget_ms:
            stop_reason = "latency_budget_exceeded"
            fallback = True
            fallback_reason = "latency_budget_exceeded"
            timeout_owner = timeout_owner or "phase6_total"
        result_current = previous if fallback else current
        result_candidates = previous_candidates if fallback else current_candidates
        result_accepted = [] if fallback else accepted_all
        result_coverage = initial_coverage if fallback else current_coverage
        marginal_gain = (result_coverage - initial_coverage) / max(1, followup_count_total)
        return _refinement_stop_result(
            result_current,
            result_candidates,
            iterations,
            result_accepted,
            rejected_all,
            plan,
            config,
            rag_trace,
            stop_reason,
            latency_ms,
            fallback=fallback,
            fallback_reason=fallback_reason,
            coverage_before=initial_coverage,
            coverage_after=result_coverage,
            marginal_gain=marginal_gain,
            followup_count=followup_count_total,
            rejected_followup_count=rejected_followup_count_total,
            cleanup_latency_ms=cleanup_ms_total,
            timeout_owner=timeout_owner,
        )

    for iteration_number in range(1, config.max_iterations + 1):
        iteration_started = time.monotonic()
        iteration_id = f"refinement.iteration.{iteration_number}"
        elapsed_ms = (iteration_started - loop_started) * 1000.0
        remaining_refinement_ms = config.latency_budget_ms - elapsed_ms
        if remaining_refinement_ms <= 0:
            return stop_result("latency_budget_exceeded", elapsed_ms, fallback=True)
        diagnostics_before = detect_missing_facets(question, plan, current)
        coverage_before = float(diagnostics_before["coverage"])
        if config.judge == "deterministic":
            judge = build_deterministic_sufficiency_judge(diagnostics_before, plan, config, iteration_id=iteration_id)
            _trace_refinement_event(
                rag_trace,
                "refinement_judge_success",
                {
                    "mode": config.mode,
                    "plan_id": plan.plan_id,
                    "iteration_id": iteration_id,
                    "judge_mode": "deterministic",
                    "llm_skipped": True,
                    "repair_call_count": 0,
                    "required_facets": [facet.facet_id for facet in plan.required_facets],
                    "covered_facets": [item.get("facet_id") for item in judge.covered_facets],
                    "missing_facets": [item.get("facet_id") for item in judge.missing_facets],
                    "confidence": judge.confidence,
                    "sufficient": judge.sufficient,
                    "followup_count": len(judge.recommended_followups),
                    "remaining_refinement_budget_ms": round(remaining_refinement_ms, 3),
                    "judge_outcome": "deterministic",
                },
            )
        elif config.judge == "heuristic" or (config.judge == "hybrid" and diagnostics_before["obviously_sufficient"]):
            judge = _heuristic_judge(diagnostics_before, plan, sufficient=bool(diagnostics_before["obviously_sufficient"]))
            _trace_refinement_event(
                rag_trace,
                "refinement_judge_success",
                {
                    "mode": config.mode,
                    "plan_id": plan.plan_id,
                    "iteration_id": iteration_id,
                    "judge_mode": "heuristic_guardrail",
                    "confidence": judge.confidence,
                    "sufficient": judge.sufficient,
                    "missing_facet_count": len(judge.missing_facets),
                    "followup_count": 0,
                    "remaining_refinement_budget_ms": round(remaining_refinement_ms, 3),
                    "judge_outcome": "heuristic",
                },
            )
        else:
            judge_timeout_ms = max(1, min(config.judge_timeout_ms, int(remaining_refinement_ms)))

            def account_judge_cleanup(cleanup_ms: float, timeout_owner: str) -> None:
                nonlocal cleanup_ms_total
                cleanup_ms_total += cleanup_ms
                if callable(cleanup_callback):
                    cleanup_callback(cleanup_ms, timeout_owner)

            try:
                judge = await generate_sufficiency_judge(
                    question=question,
                    plan=plan,
                    kbinfos=current,
                    cfg=config,
                    chat_mdl=chat_mdl,
                    rag_trace=rag_trace,
                    iteration_id=iteration_id,
                    lifecycle_id=refinement_lifecycle_id,
                    timeout_ms=judge_timeout_ms,
                    remaining_budget_ms=remaining_refinement_ms,
                    cleanup_callback=account_judge_cleanup,
                )
            except asyncio.CancelledError:
                raise
            except AgenticRefinementError as exc:
                return stop_result(exc.reason, (time.monotonic() - loop_started) * 1000.0, fallback=True, fallback_reason=exc.reason)

        if (time.monotonic() - loop_started) * 1000.0 >= config.latency_budget_ms:
            return stop_result("latency_budget_exceeded", (time.monotonic() - loop_started) * 1000.0, fallback=True)
        if judge.confidence < config.confidence_threshold:
            return stop_result("judge_confidence_below_threshold", (time.monotonic() - loop_started) * 1000.0, fallback=True)
        if judge.sufficient:
            iterations.append(
                RefinementIteration(
                    iteration_id=iteration_id,
                    judge=judge,
                    followups=(),
                    accepted_new_evidence_count=0,
                    rejected_evidence_count=0,
                    coverage_before=coverage_before,
                    coverage_after=coverage_before,
                    marginal_gain=0.0,
                    latency_ms=round((time.monotonic() - iteration_started) * 1000.0, 3),
                    stop_reason="sufficient",
                )
            )
            return stop_result("sufficient", (time.monotonic() - loop_started) * 1000.0)

        contradiction_key = tuple(sorted((item["facet_id"], tuple(sorted(item["evidence_ids"]))) for item in judge.contradictions))
        if contradiction_key and contradiction_key == previous_contradictions:
            return stop_result("persistent_contradiction", (time.monotonic() - loop_started) * 1000.0, fallback=True)
        previous_contradictions = contradiction_key

        followups, rejected_followups = validate_followup_queries(judge, plan, config, iteration_id=iteration_id)
        elapsed_ms = (time.monotonic() - loop_started) * 1000.0
        if elapsed_ms >= config.latency_budget_ms:
            return stop_result("latency_budget_exceeded", elapsed_ms, fallback=True)
        _trace_refinement_event(
            rag_trace,
            "refinement_followup_validation",
            {
                "mode": config.mode,
                "plan_id": plan.plan_id,
                "iteration_id": iteration_id,
                "proposed_followup_count": len(judge.recommended_followups),
                "validated_followup_count": len(followups),
                "rejected_followup_count": len(rejected_followups),
                "remaining_refinement_budget_ms": round(max(0.0, config.latency_budget_ms - (time.monotonic() - loop_started) * 1000.0), 3),
            },
        )
        rejected_followup_count_total += len(rejected_followups)
        for rejected in rejected_followups:
            _trace_refinement_event(rag_trace, "refinement_followup_rejected", {"mode": config.mode, "plan_id": plan.plan_id, "iteration_id": iteration_id, **rejected})
        if rejected_followups:
            drift_rate = len(rejected_followups) / max(1, len(judge.recommended_followups))
            if drift_rate > config.max_drift_rate:
                return stop_result("drift_rejection", (time.monotonic() - loop_started) * 1000.0, fallback=True)
        if not followups:
            iterations.append(
                RefinementIteration(
                    iteration_id=iteration_id,
                    judge=judge,
                    followups=(),
                    accepted_new_evidence_count=0,
                    rejected_evidence_count=0,
                    coverage_before=coverage_before,
                    coverage_after=coverage_before,
                    marginal_gain=0.0,
                    latency_ms=round((time.monotonic() - iteration_started) * 1000.0, 3),
                    stop_reason="no_followups",
                )
            )
            return stop_result("no_followups", (time.monotonic() - loop_started) * 1000.0)
        remaining_calls = config.max_followup_queries - calls_used
        if remaining_calls <= 0:
            return stop_result("max_followup_calls", (time.monotonic() - loop_started) * 1000.0)
        followups = followups[:remaining_calls]
        calls_used += len(followups)
        followup_count_total += len(followups)
        for followup in followups:
            _trace_refinement_event(
                rag_trace,
                "refinement_followup_built",
                {"mode": config.mode, "plan_id": plan.plan_id, "iteration_id": iteration_id, "followup_id": followup.followup_id, "facet_id": followup.facet_id, "top_n": followup.top_n},
            )
        _trace_refinement_event(rag_trace, "refinement_followup_execute_start", {"mode": config.mode, "plan_id": plan.plan_id, "iteration_id": iteration_id, "followup_count": len(followups)})
        tasks = []
        for followup in followups:
            facet = next(facet for facet in plan.required_facets if facet.facet_id == followup.facet_id)
            spec = SubquerySpec(
                plan_id=plan.plan_id,
                subquery_id=followup.followup_id,
                facet_id=followup.facet_id,
                query=followup.query,
                keywords=followup.keywords,
                docid_scope=doc_ids,
                top_n=min(followup.top_n, retrieval_cfg.subquery_top_n),
                retrieval_variant=followup.retrieval_variant,
                must_have_terms=facet.anchors,
                iteration_id=iteration_id,
                followup_id=followup.followup_id,
            )
            tasks.append(
                asyncio.create_task(
                    _execute_subquery(
                        spec,
                        plan,
                        retriever=retriever,
                        embd_mdl=embd_mdl,
                        tenant_ids=tenant_ids,
                        kb_ids=kb_ids,
                        doc_ids=doc_ids,
                        similarity_threshold=similarity_threshold,
                        vector_similarity_weight=vector_similarity_weight,
                        top_k=top_k,
                        rank_feature=rank_feature,
                        rerank_mdl=rerank_mdl,
                        cfg=retrieval_cfg,
                        rag_trace=rag_trace,
                        metadata_filters=metadata_filters,
                        apply_children=apply_children,
                        apply_toc=apply_toc,
                        chat_mdl=chat_mdl,
                        enforce_result_anchor_drift=True,
                    )
                )
            )
        remaining_latency_s = max(0.001, (config.latency_budget_ms - (time.monotonic() - loop_started) * 1000.0) / 1000.0)
        try:
            lane_results = await _wait_for_ordered_task_results(tasks, remaining_latency_s)
        except asyncio.CancelledError:
            cleanup_ms = await _cancel_and_drain_tasks(tasks)
            cleanup_ms_total += cleanup_ms
            if callable(cleanup_callback):
                cleanup_callback(cleanup_ms, "upstream_cancellation")
            _trace_refinement_event(
                rag_trace,
                "refinement_cleanup",
                {"mode": config.mode, "plan_id": plan.plan_id, "iteration_id": iteration_id, "cleanup_latency_ms": cleanup_ms, "timeout_owner": "upstream_cancellation", "admitted_new_work": False},
            )
            raise
        except asyncio.TimeoutError:
            cleanup_ms = await _cancel_and_drain_tasks(tasks)
            cleanup_ms_total += cleanup_ms
            if callable(cleanup_callback):
                cleanup_callback(cleanup_ms, "phase6_total")
            _trace_refinement_event(
                rag_trace,
                "refinement_cleanup",
                {"mode": config.mode, "plan_id": plan.plan_id, "iteration_id": iteration_id, "cleanup_latency_ms": cleanup_ms, "timeout_owner": "phase6_total", "admitted_new_work": False},
            )
            return stop_result("latency_budget_exceeded", (time.monotonic() - loop_started) * 1000.0, fallback=True)
        normalized_lanes: list[dict[str, Any]] = []
        for followup, result in zip(followups, lane_results):
            lane = _normalize_lane_result(result)
            lane["followup"] = followup
            normalized_lanes.append(lane)
        if any(not lane.get("accepted") for lane in normalized_lanes):
            for lane in normalized_lanes:
                if not lane.get("accepted"):
                    for chunk in (lane.get("kbinfos") or {}).get("chunks", []):
                        rejected_all.append(_mark_refinement_chunk(chunk, lane.get("followup"), selected=False, reason=str(lane.get("rejection_reason") or "bounded_retrieval_failure")))
            return stop_result("bounded_retrieval_failure", (time.monotonic() - loop_started) * 1000.0, fallback=True)
        _trace_refinement_event(rag_trace, "refinement_followup_execute_success", {"mode": config.mode, "plan_id": plan.plan_id, "iteration_id": iteration_id, "followup_count": len(followups)})

        acceptance = accept_followup_evidence(
            current_kbinfos=current,
            candidate_kbinfos=current_candidates,
            lane_results=normalized_lanes,
            plan=plan,
            missing_facet_ids={item["facet_id"] for item in judge.missing_facets},
            missing_required_anchors={item["facet_id"]: tuple(item.get("required_anchors") or ()) for item in judge.missing_facets},
            question=question,
            context_builder_config=context_builder_config,
            rag_trace=rag_trace,
        )
        elapsed_ms = (time.monotonic() - loop_started) * 1000.0
        if elapsed_ms >= config.latency_budget_ms:
            return stop_result("latency_budget_exceeded", elapsed_ms, fallback=True, timeout_owner="phase6_total")
        for chunk in acceptance.rejected_chunks:
            _trace_refinement_event(
                rag_trace,
                "refinement_evidence_rejected",
                {
                    "mode": config.mode,
                    "plan_id": plan.plan_id,
                    "iteration_id": iteration_id,
                    "evidence_id": _evidence_id(chunk),
                    "facet_id": chunk.get("facet_id"),
                    "rejection_reason": chunk.get("rejection_reason"),
                },
            )
        if acceptance.selected_new_evidence_count < config.min_new_evidence:
            threshold_rejected = [_mark_refinement_chunk(chunk, _followup_from_chunk(chunk), selected=False, reason="min_new_evidence_not_met") for chunk in acceptance.accepted_chunks]
            for chunk in threshold_rejected:
                _trace_refinement_event(
                    rag_trace,
                    "refinement_evidence_rejected",
                    {
                        "mode": config.mode,
                        "plan_id": plan.plan_id,
                        "iteration_id": iteration_id,
                        "evidence_id": _evidence_id(chunk),
                        "facet_id": chunk.get("facet_id"),
                        "rejection_reason": "min_new_evidence_not_met",
                    },
                )
            rejected_all.extend(acceptance.rejected_chunks)
            rejected_all.extend(threshold_rejected)
            iterations.append(
                RefinementIteration(
                    iteration_id=iteration_id,
                    judge=judge,
                    followups=followups,
                    accepted_new_evidence_count=0,
                    rejected_evidence_count=len(acceptance.rejected_chunks) + len(threshold_rejected),
                    coverage_before=coverage_before,
                    coverage_after=coverage_before,
                    marginal_gain=0.0,
                    latency_ms=round((time.monotonic() - iteration_started) * 1000.0, 3),
                    stop_reason="no_selected_new_evidence",
                )
            )
            return stop_result("no_selected_new_evidence", (time.monotonic() - loop_started) * 1000.0)

        for chunk in acceptance.accepted_chunks:
            _trace_refinement_event(
                rag_trace,
                "refinement_evidence_accepted",
                {"mode": config.mode, "plan_id": plan.plan_id, "iteration_id": iteration_id, "evidence_id": _evidence_id(chunk), "facet_id": chunk.get("facet_id")},
            )
        current = acceptance.kbinfos
        current_candidates = acceptance.candidate_kbinfos or current_candidates
        accepted_all.extend(acceptance.accepted_chunks)
        rejected_all.extend(acceptance.rejected_chunks)
        diagnostics_after = detect_missing_facets(question, plan, current)
        coverage_after = float(diagnostics_after["coverage"])
        current_coverage = coverage_after
        marginal_gain = (coverage_after - coverage_before) / max(1, len(followups))
        iteration_latency_ms = round((time.monotonic() - iteration_started) * 1000.0, 3)
        iterations.append(
            RefinementIteration(
                iteration_id=iteration_id,
                judge=judge,
                followups=followups,
                accepted_new_evidence_count=acceptance.selected_new_evidence_count,
                rejected_evidence_count=len(acceptance.rejected_chunks),
                coverage_before=coverage_before,
                coverage_after=coverage_after,
                marginal_gain=round(marginal_gain, 6),
                latency_ms=iteration_latency_ms,
            )
        )
        _trace_refinement_event(
            rag_trace,
            "refinement_context_recomputed",
            {
                "mode": config.mode,
                "plan_id": plan.plan_id,
                "iteration_id": iteration_id,
                "accepted_new_evidence_count": acceptance.selected_new_evidence_count,
                "rejected_evidence_count": len(acceptance.rejected_chunks),
                "retrieved_evidence_count": sum(len((lane.get("kbinfos") or {}).get("chunks", [])) for lane in normalized_lanes),
                "eligible_evidence_count": len(acceptance.accepted_chunks) + len(acceptance.rejected_chunks),
                "selected_evidence_count": len(acceptance.kbinfos.get("chunks", [])),
                "coverage_before": coverage_before,
                "coverage_after": coverage_after,
                "marginal_gain": round(marginal_gain, 6),
                "marginal_gain_formula": "(coverage_after - coverage_before) / max(1, extra_calls)",
                "latency_ms": iteration_latency_ms,
            },
        )

    return stop_result("max_iterations", (time.monotonic() - loop_started) * 1000.0)


async def execute_bounded_plan(
    plan: BoundedRetrievalPlan,
    *,
    retriever: Any,
    embd_mdl: Any,
    tenant_ids: list[str],
    kb_ids: list[str],
    doc_ids: list[str] | None,
    similarity_threshold: float,
    vector_similarity_weight: float,
    top_k: int,
    rank_feature: dict[str, Any] | None,
    rerank_mdl: Any = None,
    cfg: AgenticRetrievalConfig = AgenticRetrievalConfig(),
    rag_trace: Any = None,
    metadata_filters: Any = None,
    apply_children: bool = True,
    apply_toc: bool = False,
    chat_mdl: Any = None,
    enforce_result_anchor_drift: bool = True,
    cleanup_callback: Any = None,
) -> BoundedRetrievalResult:
    started = time.monotonic()
    plan_trace = {"plan_origin": plan.plan_origin, "planner_mode": cfg.planner_mode}
    tasks = [
        asyncio.create_task(
            _execute_subquery(
                spec,
                plan,
                retriever=retriever,
                embd_mdl=embd_mdl,
                tenant_ids=tenant_ids,
                kb_ids=kb_ids,
                doc_ids=doc_ids,
                similarity_threshold=similarity_threshold,
                vector_similarity_weight=vector_similarity_weight,
                top_k=top_k,
                rank_feature=rank_feature,
                rerank_mdl=rerank_mdl,
                cfg=cfg,
                rag_trace=rag_trace,
                metadata_filters=metadata_filters,
                apply_children=apply_children,
                apply_toc=apply_toc,
                chat_mdl=chat_mdl,
                enforce_result_anchor_drift=enforce_result_anchor_drift,
            )
        )
        for spec in plan.subqueries[: min(cfg.max_subqueries, cfg.max_extra_retrieval_calls)]
    ]
    overall_timeout = max(0.001, cfg.latency_budget_ms / 1000.0)
    try:
        lane_results = await _wait_for_ordered_task_results(tasks, overall_timeout)
    except BaseException as exc:
        cleanup_ms = await _cancel_and_drain_tasks(tasks)
        if callable(cleanup_callback):
            cleanup_callback(cleanup_ms, "upstream_cancellation" if isinstance(exc, asyncio.CancelledError) else "phase5_retrieval")
        _trace_agentic_event(
            rag_trace,
            "cleanup",
            {
                "plan_id": plan.plan_id,
                "cleanup_latency_ms": cleanup_ms,
                "timeout_owner": "upstream_cancellation" if isinstance(exc, asyncio.CancelledError) else "phase5_retrieval",
                "admitted_new_work": False,
            },
        )
        if isinstance(exc, asyncio.CancelledError):
            raise
        diagnostics = {
            "accepted_subqueries": 0,
            "rejected_subqueries": len(tasks),
            "rejections": [{"rejection_reason": "overall_timeout_or_exception", "failure_class": type(exc).__name__}],
            "cleanup_latency_ms": cleanup_ms,
        }
        _trace_agentic_event(rag_trace, "fallback", {"plan_id": plan.plan_id, **plan_trace, "reason": "overall_timeout_or_exception", **diagnostics})
        return BoundedRetrievalResult(
            kbinfos={"total": 0, "chunks": [], "doc_aggs": []},
            plan=plan,
            accepted_subqueries=0,
            rejected_subqueries=len(tasks),
            fallback_to_baseline=True,
            fallback_reason="overall_timeout_or_exception",
            diagnostics=diagnostics,
        )

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for result in lane_results:
        normalized = _normalize_lane_result(result)
        if normalized.get("accepted"):
            accepted.append(normalized)
        else:
            rejected.append(normalized)

    required_facet_ids = {facet.facet_id for facet in plan.required_facets}
    required_subquery_ids = {spec.subquery_id for spec in plan.subqueries}
    accepted_specs: list[SubquerySpec] = []
    for lane in accepted:
        spec = lane.get("subquery")
        if isinstance(spec, SubquerySpec):
            accepted_specs.append(spec)
    accepted_facet_ids = {spec.facet_id for spec in accepted_specs}
    accepted_subquery_ids = {spec.subquery_id for spec in accepted_specs}
    missing_facet_ids = sorted(required_facet_ids.difference(accepted_facet_ids))
    missing_subquery_ids = sorted(required_subquery_ids.difference(accepted_subquery_ids))
    if rejected or missing_facet_ids or missing_subquery_ids:
        diagnostics = {
            "accepted_subqueries": len(accepted),
            "rejected_subqueries": len(rejected) + len(missing_subquery_ids),
            "rejections": rejected,
            "missing_required_facets": missing_facet_ids,
            "missing_required_subqueries": missing_subquery_ids,
        }
        reason = "required_subquery_rejected" if rejected or missing_subquery_ids else "required_facet_coverage_incomplete"
        _trace_agentic_event(rag_trace, "fallback", {"plan_id": plan.plan_id, **plan_trace, "reason": reason, **diagnostics})
        return BoundedRetrievalResult(
            kbinfos={"total": 0, "chunks": [], "doc_aggs": []},
            plan=plan,
            accepted_subqueries=len(accepted),
            rejected_subqueries=len(rejected),
            fallback_to_baseline=True,
            fallback_reason=reason,
            diagnostics=diagnostics,
        )

    if not accepted:
        diagnostics = {"accepted_subqueries": 0, "rejected_subqueries": len(rejected), "rejections": rejected}
        _trace_agentic_event(rag_trace, "fallback", {"plan_id": plan.plan_id, **plan_trace, "reason": "all_subqueries_rejected", **diagnostics})
        return BoundedRetrievalResult(
            kbinfos={"total": 0, "chunks": [], "doc_aggs": []},
            plan=plan,
            accepted_subqueries=0,
            rejected_subqueries=len(rejected),
            fallback_to_baseline=True,
            fallback_reason="all_subqueries_rejected",
            diagnostics=diagnostics,
        )
    merged = merge_subquery_kbinfos(plan, accepted, cfg, original_question=plan.original_question)
    latency_ms = round((time.monotonic() - started) * 1000.0, 3)
    diagnostics = {
        "latency_ms": latency_ms,
        "accepted_subqueries": len(accepted),
        "rejected_subqueries": len(rejected),
        "rejections": rejected,
        "merged_chunk_count": len(merged.get("chunks", [])),
    }
    if latency_ms > cfg.latency_budget_ms:
        _trace_agentic_event(rag_trace, "fallback", {"plan_id": plan.plan_id, **plan_trace, "reason": "latency_budget_exceeded", **diagnostics})
        return BoundedRetrievalResult(
            kbinfos={"total": 0, "chunks": [], "doc_aggs": []},
            plan=plan,
            accepted_subqueries=len(accepted),
            rejected_subqueries=len(rejected),
            fallback_to_baseline=True,
            fallback_reason="latency_budget_exceeded",
            diagnostics=diagnostics,
        )
    diagnostics.update(plan_trace)
    _trace_agentic_event(rag_trace, "execute_bounded_plan", {"plan_id": plan.plan_id, **diagnostics})
    _trace_agentic_event(rag_trace, "planner_execute_bounded_plan", {"plan_id": plan.plan_id, **diagnostics})
    _trace_planner_event(rag_trace, "planner_execute_bounded_plan", {"plan_id": plan.plan_id, **diagnostics})
    return BoundedRetrievalResult(
        kbinfos=merged,
        plan=plan,
        accepted_subqueries=len(accepted),
        rejected_subqueries=len(rejected),
        diagnostics=diagnostics,
    )


def _normalize_lane_result(result: Any) -> dict[str, Any]:
    if isinstance(result, BaseException):
        return {
            "accepted": False,
            "rejection_reason": "lane_exception",
            "failure_class": type(result).__name__,
        }
    if not isinstance(result, dict):
        return {"accepted": False, "rejection_reason": "invalid_lane_result", "error": type(result).__name__}
    normalized = dict(result)
    normalized["accepted"] = bool(normalized.get("accepted"))
    if not normalized["accepted"] and not normalized.get("rejection_reason"):
        normalized["rejection_reason"] = "lane_rejected"
    return normalized


def merge_subquery_kbinfos(
    plan: BoundedRetrievalPlan,
    lane_results: list[dict[str, Any]] | None = None,
    cfg: AgenticRetrievalConfig | None = None,
    *,
    original_question: str,
    rrf_k: int | None = None,
) -> dict[str, Any]:
    if not isinstance(plan, BoundedRetrievalPlan):
        raise TypeError("merge_subquery_kbinfos requires a validated BoundedRetrievalPlan; bare lane lists are not accepted")
    cfg = cfg or AgenticRetrievalConfig(rrf_k=rrf_k or 60)
    if rrf_k is not None:
        cfg = replace(cfg, rrf_k=rrf_k)
    validation = validate_plan(plan, cfg=cfg, original_question=original_question)
    if not validation.valid:
        raise AgenticPlannerError("planner_invalid_plan", validation.fallback_reason or ";".join(validation.errors))
    plan = validation.plan
    lane_results = lane_results or []
    groups: dict[str, dict[str, Any]] = {}
    original_anchors = _anchors_for_text(original_question)
    for lane in lane_results:
        spec: SubquerySpec = lane["subquery"]
        chunks = (lane.get("kbinfos") or {}).get("chunks") or []
        for rank, chunk in enumerate(chunks, start=1):
            if not isinstance(chunk, dict):
                continue
            key = _chunk_identity(chunk)
            if key not in groups:
                groups[key] = {
                    "chunk": copy(chunk),
                    "rrf_score": 0.0,
                    "best_rank": rank,
                    "facets": set(),
                    "subqueries": set(),
                    "lineage": [],
                    "occurrences": 0,
                }
            group = groups[key]
            group["rrf_score"] += 1.0 / (max(1, cfg.rrf_k) + rank)
            group["best_rank"] = min(group["best_rank"], rank)
            group["facets"].add(spec.facet_id)
            group["subqueries"].add(spec.subquery_id)
            group["lineage"].append(
                {
                    "plan_id": spec.plan_id,
                    "facet_id": spec.facet_id,
                    "subquery_id": spec.subquery_id,
                    "retrieval_call_id": lane.get("retrieval_call_id"),
                    "lineage_rank": rank,
                    "retrieval_variant": spec.retrieval_variant,
                }
            )
            group["occurrences"] += 1
    ranked: list[tuple[float, int, str, dict[str, Any]]] = []
    for key, group in groups.items():
        chunk = group["chunk"]
        facet_bonus = 0.03 * len(group["facets"])
        original_bonus = 0.02 if _text_has_any_anchor(_chunk_text(chunk), original_anchors) else 0.0
        duplicate_penalty = 0.005 * max(0, int(group["occurrences"]) - 1)
        score = float(group["rrf_score"]) + facet_bonus + original_bonus - duplicate_penalty
        ranked.append((score, int(group["best_rank"]), key, group))
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    chunks: list[dict[str, Any]] = []
    for merge_rank, (score, _best_rank, _key, group) in enumerate(ranked, start=1):
        chunk = group["chunk"]
        chunk["merge_rank"] = merge_rank
        chunk["agentic_rrf_score"] = round(score, 8)
        metadata = dict(chunk.get("_ragflow_agentic_retrieval") or {})
        metadata.update(
            {
                "plan_id": plan.plan_id,
                "merge_rank": merge_rank,
                "rrf_score": round(score, 8),
                "facet_ids": sorted(group["facets"]),
                "subquery_ids": sorted(group["subqueries"]),
                "agentic_lineage": group["lineage"],
                "duplicate_occurrences": group["occurrences"],
            }
        )
        first_lineage = group["lineage"][0] if group["lineage"] else {}
        metadata.setdefault("facet_id", first_lineage.get("facet_id"))
        metadata.setdefault("subquery_id", first_lineage.get("subquery_id"))
        metadata.setdefault("retrieval_call_id", first_lineage.get("retrieval_call_id"))
        metadata.setdefault("lineage_rank", first_lineage.get("lineage_rank"))
        metadata.setdefault("retrieval_variant", first_lineage.get("retrieval_variant"))
        chunk["plan_id"] = plan.plan_id
        chunk["merge_rank"] = merge_rank
        for key_name in ("facet_id", "subquery_id", "retrieval_call_id", "lineage_rank", "retrieval_variant"):
            if metadata.get(key_name) is not None:
                chunk[key_name] = metadata.get(key_name)
        chunk["_ragflow_agentic_retrieval"] = metadata
        document_metadata = dict(chunk.get("document_metadata") or chunk.get("metadata") or {})
        document_metadata.update(metadata)
        chunk["document_metadata"] = document_metadata
        chunks.append(chunk)
    doc_aggs = _merge_doc_aggs(lane_results, chunks)
    return {
        "total": len(chunks),
        "chunks": chunks,
        "doc_aggs": doc_aggs,
        "diagnostics": {
            "agentic_retrieval": {
                "plan_id": plan.plan_id,
                "strategy": "subquery_rrf_then_context_builder",
                "rrf_k": cfg.rrf_k,
                "lane_count": len(lane_results),
                "merged_chunk_count": len(chunks),
            }
        },
    }


def attach_planning_metadata(
    chunks: list[dict[str, Any]] | None,
    *,
    plan_id: str,
    facet_id: str,
    subquery_id: str,
    retrieval_call_id: str | None = None,
    lineage_rank_start: int = 1,
    retrieval_variant: str | None = None,
    selected_for_context: bool | None = None,
    rejection_reason: str | None = None,
    iteration_id: str | None = None,
    followup_id: str | None = None,
) -> None:
    for offset, chunk in enumerate(chunks or []):
        if not isinstance(chunk, dict):
            continue
        lineage_rank = lineage_rank_start + offset
        chunk["plan_id"] = plan_id
        chunk["facet_id"] = facet_id
        chunk["subquery_id"] = subquery_id
        chunk["retrieval_call_id"] = retrieval_call_id
        chunk["lineage_rank"] = lineage_rank
        if retrieval_variant is not None:
            chunk["retrieval_variant"] = retrieval_variant
        if selected_for_context is not None:
            chunk["selected_for_context"] = selected_for_context
        if rejection_reason is not None:
            chunk["rejection_reason"] = rejection_reason
        if iteration_id is not None:
            chunk["iteration_id"] = iteration_id
        if followup_id is not None:
            chunk["followup_id"] = followup_id
        metadata = dict(chunk.get("_ragflow_agentic_retrieval") or {})
        metadata.update(
            {
                "plan_id": plan_id,
                "facet_id": facet_id,
                "subquery_id": subquery_id,
                "retrieval_call_id": retrieval_call_id,
                "lineage_rank": lineage_rank,
                "retrieval_variant": retrieval_variant,
                "selected_for_context": selected_for_context,
                "rejection_reason": rejection_reason,
                "iteration_id": iteration_id,
                "followup_id": followup_id,
            }
        )
        chunk["_ragflow_agentic_retrieval"] = metadata
        document_metadata = dict(chunk.get("document_metadata") or chunk.get("metadata") or {})
        document_metadata.update({k: v for k, v in metadata.items() if v is not None})
        chunk["document_metadata"] = document_metadata


async def execute_plan(
    plan: BoundedRetrievalPlan | dict[str, Any],
    *,
    retriever: Any,
    cfg: AgenticRetrievalConfig,
    tenant_ids: list[str],
    kb_ids: list[str],
    embed_mdl: Any = None,
    embd_mdl: Any = None,
    doc_ids: list[str] | None = None,
    similarity_threshold: float = 0.2,
    vector_similarity_weight: float = 0.3,
    top_k: int = 1024,
    rank_feature: dict[str, Any] | None = None,
    rag_trace: Any = None,
    metadata_filters: Any = None,
    rerank_mdl: Any = None,
    apply_children: bool = False,
    apply_toc: bool = False,
    chat_mdl: Any = None,
    enforce_result_anchor_drift: bool = True,
) -> BoundedRetrievalResult:
    validation = validate_plan(plan, cfg=cfg, original_question=(plan.get("original_question") if isinstance(plan, dict) else None))
    if validation.fallback_reason:
        return BoundedRetrievalResult(
            kbinfos={"total": 0, "chunks": [], "doc_aggs": []},
            plan=validation.plan,
            accepted_subqueries=0,
            rejected_subqueries=len(validation.errors),
            fallback_to_baseline=True,
            fallback_reason=validation.fallback_reason,
            diagnostics={"validation_errors": validation.errors},
        )
    return await execute_bounded_plan(
        validation.plan,
        retriever=retriever,
        embd_mdl=embd_mdl if embd_mdl is not None else embed_mdl,
        tenant_ids=tenant_ids,
        kb_ids=kb_ids,
        doc_ids=doc_ids,
        similarity_threshold=similarity_threshold,
        vector_similarity_weight=vector_similarity_weight,
        top_k=top_k,
        rank_feature=rank_feature,
        rerank_mdl=rerank_mdl,
        cfg=cfg,
        rag_trace=rag_trace,
        metadata_filters=metadata_filters,
        apply_children=apply_children,
        apply_toc=apply_toc,
        chat_mdl=chat_mdl,
        enforce_result_anchor_drift=enforce_result_anchor_drift,
    )


execute_agentic_plan = execute_plan


async def run_agentic_retrieval(
    *,
    question: str,
    retriever: Any,
    cfg: AgenticRetrievalConfig,
    tenant_ids: list[str],
    kb_ids: list[str],
    planner: Any = None,
    chat_mdl: Any = None,
    embed_mdl: Any = None,
    embd_mdl: Any = None,
    context_builder_config: Any = None,
    doc_ids: list[str] | None = None,
    similarity_threshold: float = 0.2,
    vector_similarity_weight: float = 0.3,
    top_k: int = 1024,
    rank_feature: dict[str, Any] | None = None,
    rag_trace: Any = None,
    metadata_filters: Any = None,
    rerank_mdl: Any = None,
) -> dict[str, Any]:
    baseline = await _baseline_retrieval(
        retriever,
        question=question,
        embd_mdl=embd_mdl if embd_mdl is not None else embed_mdl,
        tenant_ids=tenant_ids,
        kb_ids=kb_ids,
        doc_ids=doc_ids,
        similarity_threshold=similarity_threshold,
        vector_similarity_weight=vector_similarity_weight,
        top_k=top_k,
        rank_feature=rank_feature,
        rerank_mdl=rerank_mdl,
        page_size=cfg.subquery_top_n,
    )
    trigger = should_plan(
        question=question,
        history=[],
        dialog=type("AgenticScope", (), {"kb_ids": kb_ids})(),
        attachments=doc_ids or [],
        cfg=cfg,
        diagnostics=None,
    )
    _trace_agentic_event(rag_trace, "trigger", trigger.to_trace_dict())
    if not trigger.should_plan:
        return {"kbinfos": baseline, **baseline, "fallback_to_baseline": True, "fallback_reason": trigger.bypass_reason or "planning_not_triggered"}
    if cfg.mode == "diagnostic":
        return {"kbinfos": baseline, **baseline, "mode": cfg.mode, "fallback_to_baseline": True, "fallback_reason": "diagnostic_trigger_only"}
    planner_input = build_planner_input(
        question=question,
        history=[],
        dialog=type("AgenticScope", (), {"kb_ids": kb_ids})(),
        attachments=doc_ids or [],
        cfg=cfg,
        trigger=trigger,
        tenant_ids=tenant_ids,
        metadata_filters=metadata_filters,
    )
    if cfg.planner_mode == "deterministic":
        try:
            validation = resolve_deterministic_primary_plan(planner_input, cfg, rag_trace=rag_trace)
            if not validation.valid:
                return {"kbinfos": baseline, **baseline, "fallback_reason": validation.fallback_reason, "fallback_to_baseline": True}
            plan = validation.plan
        except (TypeError, ValueError) as exc:
            return {"kbinfos": baseline, **baseline, "fallback_reason": f"deterministic_plan_invalid:{exc}", "fallback_to_baseline": True}
    elif chat_mdl is None:
        _trace_agentic_event(rag_trace, "planner_llm_fallback_to_baseline", {"reason": "planner_missing_chat_model", "mode": cfg.mode})
        _trace_planner_event(rag_trace, "planner_llm_fallback_to_baseline", {"fallback_reason": "planner_missing_chat_model", "mode": cfg.mode})
        return {"kbinfos": baseline, **baseline, "fallback_reason": "planner_missing_chat_model", "fallback_to_baseline": True}
    try:
        if cfg.planner_mode == "llm":
            plan = await generate_llm_plan(planner_input, cfg, chat_mdl=chat_mdl, rag_trace=rag_trace)
        agentic_result = await execute_plan(
            plan,
            retriever=retriever,
            cfg=cfg,
            tenant_ids=tenant_ids,
            kb_ids=kb_ids,
            embed_mdl=embed_mdl,
            embd_mdl=embd_mdl,
            doc_ids=doc_ids,
            similarity_threshold=similarity_threshold,
            vector_similarity_weight=vector_similarity_weight,
            top_k=top_k,
            rank_feature=rank_feature,
            rag_trace=rag_trace,
            metadata_filters=metadata_filters,
            rerank_mdl=rerank_mdl,
            enforce_result_anchor_drift=True,
        )
    except AgenticPlannerError as exc:
        _trace_agentic_event(rag_trace, "planner_llm_fallback_to_baseline", {"reason": exc.reason, "detail": exc.detail, "mode": cfg.mode})
        _trace_planner_event(rag_trace, "planner_llm_fallback_to_baseline", {"fallback_reason": exc.reason, "detail": exc.detail, "mode": cfg.mode})
        return {"kbinfos": baseline, **baseline, "fallback_reason": exc.reason, "fallback_to_baseline": True}
    except Exception:
        _trace_agentic_event(rag_trace, "planner_llm_fallback_to_baseline", {"reason": "planner_error", "failure_class": "unknown", "mode": cfg.mode})
        _trace_planner_event(rag_trace, "planner_llm_fallback_to_baseline", {"fallback_reason": "planner_error", "failure_class": "unknown", "mode": cfg.mode})
        return {"kbinfos": baseline, **baseline, "fallback_reason": "planner_error", "fallback_to_baseline": True}

    if cfg.mode == "shadow":
        return {"kbinfos": baseline, **baseline, "mode": cfg.mode, "shadow_agentic": agentic_result.diagnostics}
    if agentic_result.fallback_to_baseline:
        return {"kbinfos": baseline, **baseline, "fallback_reason": agentic_result.fallback_reason, "fallback_to_baseline": True}
    kbinfos = agentic_result.kbinfos
    if context_builder_config is not None and getattr(context_builder_config, "enabled", False):
        from rag.utils.context_builder import apply_context_builder_to_kbinfos

        kbinfos = apply_context_builder_to_kbinfos(kbinfos, context_builder_config, query=question).kbinfos
    return {"kbinfos": kbinfos, **kbinfos, "diagnostics": agentic_result.diagnostics}


agentic_retrieval = run_agentic_retrieval
retrieve_with_agentic_planning = run_agentic_retrieval


async def _baseline_retrieval(
    retriever: Any,
    *,
    question: str,
    embd_mdl: Any,
    tenant_ids: list[str],
    kb_ids: list[str],
    doc_ids: list[str] | None,
    similarity_threshold: float,
    vector_similarity_weight: float,
    top_k: int,
    rank_feature: dict[str, Any] | None,
    rerank_mdl: Any = None,
    page_size: int = 6,
) -> dict[str, Any]:
    return await retriever.retrieval(
        question,
        embd_mdl,
        tenant_ids,
        kb_ids,
        1,
        page_size,
        similarity_threshold,
        vector_similarity_weight=vector_similarity_weight,
        doc_ids=doc_ids,
        top=top_k,
        aggs=True,
        rerank_mdl=rerank_mdl,
        rank_feature=rank_feature,
    )


async def _execute_subquery(
    spec: SubquerySpec,
    plan: BoundedRetrievalPlan,
    **kwargs: Any,
) -> dict[str, Any]:
    rag_trace = kwargs.get("rag_trace")
    trace_fields = {"plan_id": plan.plan_id, "subquery_id": spec.subquery_id, "plan_origin": plan.plan_origin, "planner_mode": kwargs["cfg"].planner_mode}
    _trace_agentic_event(rag_trace, "subquery_start", trace_fields)
    try:
        result = await asyncio.wait_for(_execute_subquery_work(spec, plan, **kwargs), timeout=max(0.001, kwargs["cfg"].retrieval_timeout_ms / 1000.0))
        _trace_agentic_event(rag_trace, "subquery_result", {**trace_fields, "accepted": bool(result.get("accepted")), "rejection_reason": result.get("rejection_reason")})
        return result
    except BaseException as exc:
        if isinstance(exc, asyncio.CancelledError):
            raise
        failure_class = type(exc).__name__
        _trace_agentic_event(rag_trace, "subquery_rejected", {**trace_fields, "reason": "lane_timeout_or_exception", "failure_class": failure_class})
        return {"subquery": spec, "accepted": False, "rejection_reason": "lane_timeout_or_exception", "failure_class": failure_class}


async def _execute_subquery_work(
    spec: SubquerySpec,
    plan: BoundedRetrievalPlan,
    *,
    retriever: Any,
    embd_mdl: Any,
    tenant_ids: list[str],
    kb_ids: list[str],
    doc_ids: list[str] | None,
    similarity_threshold: float,
    vector_similarity_weight: float,
    top_k: int,
    rank_feature: dict[str, Any] | None,
    rerank_mdl: Any,
    cfg: AgenticRetrievalConfig,
    rag_trace: Any,
    metadata_filters: Any,
    apply_children: bool,
    apply_toc: bool,
    chat_mdl: Any,
    enforce_result_anchor_drift: bool,
) -> dict[str, Any]:
    started_ms = time.time() * 1000.0
    variant = make_retrieval_variant(spec.retrieval_variant)
    knobs = resolve_variant_knobs(
        variant,
        default_vector_similarity_weight=vector_similarity_weight,
        default_similarity_threshold=similarity_threshold,
        embedding_available=embd_mdl is not None,
    )
    used_embedding = bool(knobs["using_embedding"] and embd_mdl is not None)
    lane_embd_mdl = embd_mdl if used_embedding else None
    lane_doc_ids = spec.docid_scope if spec.docid_scope is not None else doc_ids
    top_n = max(1, min(spec.top_n, cfg.subquery_top_n))
    retrieval = retriever.retrieval(
        spec.query,
        lane_embd_mdl,
        tenant_ids,
        kb_ids,
        1,
        top_n,
        knobs["similarity_threshold"],
        vector_similarity_weight=knobs["vector_similarity_weight"],
        doc_ids=lane_doc_ids,
        top=top_k,
        aggs=True,
        rerank_mdl=rerank_mdl,
        rank_feature=rank_feature,
    )
    try:
        kbinfos = await asyncio.wait_for(retrieval, timeout=max(0.001, cfg.retrieval_timeout_ms / 1000.0))
    except Exception as exc:
        failure_class = type(exc).__name__
        _trace_agentic_event(rag_trace, "subquery_rejected", {"plan_id": plan.plan_id, "subquery_id": spec.subquery_id, "reason": "retrieval_exception", "failure_class": failure_class})
        return {"subquery": spec, "accepted": False, "rejection_reason": "retrieval_exception", "failure_class": failure_class}
    if apply_toc and chat_mdl:
        cks = await retriever.retrieval_by_toc(spec.query, kbinfos.get("chunks", []), tenant_ids, chat_mdl, top_n)
        if cks:
            kbinfos["chunks"] = cks
    if apply_children:
        kbinfos["chunks"] = retriever.retrieval_by_children(kbinfos.get("chunks", []), tenant_ids)

    facet = next((f for f in plan.required_facets if f.facet_id == spec.facet_id), None)
    anchors = tuple(dict.fromkeys(list(spec.must_have_terms) + list(facet.anchors if facet else ()) + _anchors_for_text(plan.original_question)[:3]))
    result_anchor_drift = enforce_result_anchor_drift and not _result_group_has_anchor(kbinfos.get("chunks", []), anchors)
    retrieval_call_id = None
    attach_planning_metadata(
        kbinfos.get("chunks", []),
        plan_id=plan.plan_id,
        facet_id=spec.facet_id,
        subquery_id=spec.subquery_id,
        retrieval_variant=knobs["retrieval_variant"],
        selected_for_context=False if result_anchor_drift else None,
        rejection_reason="result_anchor_drift" if result_anchor_drift else None,
        iteration_id=spec.iteration_id,
        followup_id=spec.followup_id,
    )
    if rag_trace:
        retrieval_trace_fields = dict(
            mode="agentic_subquery",
            query_text=spec.query,
            keywords=spec.keywords,
            docid_scope=lane_doc_ids,
            metadata_filters=metadata_filters,
            chunks=kbinfos.get("chunks", []),
            doc_aggs=kbinfos.get("doc_aggs", []),
            used_embedding=used_embedding,
            used_web=False,
            started_ms=started_ms,
            retrieval_variant=knobs["retrieval_variant"],
            similarity_threshold=knobs["similarity_threshold"],
            vector_similarity_weight=knobs["vector_similarity_weight"],
            top_k=top_k,
            page_size=top_n,
            doc_scope_enabled=bool(lane_doc_ids),
            metadata_filter_enabled=bool(metadata_filters),
            diagnostics=kbinfos.get("diagnostics", {}),
            plan_id=plan.plan_id,
            facet_id=spec.facet_id,
            subquery_id=spec.subquery_id,
        )
        if spec.iteration_id is not None:
            retrieval_trace_fields["iteration_id"] = spec.iteration_id
        if spec.followup_id is not None:
            retrieval_trace_fields["followup_id"] = spec.followup_id
        try:
            retrieval_call_id = rag_trace.add_retrieval_call(**retrieval_trace_fields)
        except TypeError:
            retrieval_trace_fields.pop("followup_id", None)
            retrieval_call_id = rag_trace.add_retrieval_call(**retrieval_trace_fields)
    attach_planning_metadata(
        kbinfos.get("chunks", []),
        plan_id=plan.plan_id,
        facet_id=spec.facet_id,
        subquery_id=spec.subquery_id,
        retrieval_call_id=retrieval_call_id,
        retrieval_variant=knobs["retrieval_variant"],
        selected_for_context=False if result_anchor_drift else None,
        rejection_reason="result_anchor_drift" if result_anchor_drift else None,
        iteration_id=spec.iteration_id,
        followup_id=spec.followup_id,
    )
    if rag_trace:
        rag_trace.add_evidence_from_chunks(kbinfos.get("chunks", []), source_type="kb", retrieval_call_id=retrieval_call_id)

    if result_anchor_drift:
        _trace_agentic_event(rag_trace, "subquery_rejected", {"plan_id": plan.plan_id, "subquery_id": spec.subquery_id, "reason": "result_anchor_drift"})
        return {
            "subquery": spec,
            "accepted": False,
            "kbinfos": kbinfos,
            "rejection_reason": "result_anchor_drift",
            "retrieval_call_id": retrieval_call_id,
        }
    return {"subquery": spec, "accepted": True, "kbinfos": kbinfos, "retrieval_call_id": retrieval_call_id}


def _replace_subquery(subquery: SubquerySpec, **changes: Any) -> SubquerySpec:
    return replace(subquery, **changes)


def _coerce_plan(plan: BoundedRetrievalPlan | dict[str, Any], cfg: AgenticRetrievalConfig, *, original_question: str | None = None) -> BoundedRetrievalPlan:
    if isinstance(plan, BoundedRetrievalPlan):
        normalized_subqueries = tuple(_replace_subquery(sq, top_n=max(1, min(int(sq.top_n or cfg.subquery_top_n), cfg.subquery_top_n))) for sq in plan.subqueries[: cfg.max_subqueries])
        return replace(plan, subqueries=normalized_subqueries)
    if not isinstance(plan, dict):
        raise TypeError("plan must be BoundedRetrievalPlan or dict")
    question = str(original_question or plan.get("original_question") or "").strip()
    plan_id = str(plan.get("plan_id") or f"plan-{uuid.uuid4().hex[:12]}")
    facets: list[RequiredFacet] = []
    for idx, raw in enumerate(plan.get("required_facets") or [], start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"facet_{idx}_not_object")
        facet_id = str(raw.get("facet_id") or "").strip()
        if not facet_id:
            raise ValueError(f"facet_{idx}_missing_id")
        anchors = tuple(str(anchor).strip() for anchor in raw.get("anchors") or [] if str(anchor).strip())
        facets.append(
            RequiredFacet(
                facet_id=facet_id,
                description=str(raw.get("description") or facet_id),
                anchors=anchors,
                evidence_type=str(raw.get("evidence_type") or "quote"),
            )
        )
    subqueries: list[SubquerySpec] = []
    for idx, raw in enumerate(plan.get("subqueries") or [], start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"subquery_{idx}_not_object")
        variant = str(raw.get("retrieval_variant") or "hybrid_default")
        top_n = max(1, min(_safe_int(raw.get("top_n"), cfg.subquery_top_n), cfg.subquery_top_n))
        subqueries.append(
            SubquerySpec(
                plan_id=str(raw.get("plan_id") or plan_id),
                subquery_id=str(raw.get("subquery_id") or f"sq{idx}"),
                facet_id=str(raw.get("facet_id") or ""),
                query=str(raw.get("query") or ""),
                keywords=raw.get("keywords"),
                docid_scope=list(raw.get("docid_scope") or []) if raw.get("docid_scope") else None,
                top_n=top_n,
                retrieval_variant=variant,  # type: ignore[arg-type]
                must_have_terms=tuple(str(term) for term in raw.get("must_have_terms") or []),
                forbidden_new_entities=tuple(str(term) for term in raw.get("forbidden_new_entities") or []),
                rationale=str(raw.get("rationale") or ""),
                iteration_id=str(raw.get("iteration_id")) if raw.get("iteration_id") else None,
                followup_id=str(raw.get("followup_id")) if raw.get("followup_id") else None,
            )
        )
    return BoundedRetrievalPlan(
        plan_id=plan_id,
        original_question=question,
        complexity=str(plan.get("complexity") or "unknown"),
        trigger_reasons=tuple(str(reason) for reason in plan.get("trigger_reasons") or []),
        required_facets=tuple(facets),
        subqueries=tuple(subqueries[: cfg.max_subqueries]),
        merge_policy=dict(plan.get("merge_policy") or {}),
        drift_controls=dict(plan.get("drift_controls") or {"anchor_entities": _anchors_for_text(question), "min_anchor_overlap": cfg.min_anchor_overlap, "allow_new_entities": False}),
        plan_origin=str(plan.get("plan_origin") or "llm_text"),  # type: ignore[arg-type]
    )


def _summarize_history(history: list[dict[str, Any]]) -> str:
    snippets: list[str] = []
    for turn in history[-4:]:
        if not isinstance(turn, dict):
            continue
        role = re.sub(r"[^a-zA-Z_-]", "", str(turn.get("role") or "message"))[:20] or "message"
        content = re.sub(r"\s+", " ", str(turn.get("content") or "")).strip()
        if content:
            snippets.append(f"{role}: {content[:500]}")
    return "\n".join(snippets)[:2000]


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bounded_int_setting(value: Any, env_name: str, default: int, lower: int, upper: int) -> int:
    if value is None:
        value = os.getenv(env_name)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(lower, min(parsed, upper))


def _nonnegative_float_setting(value: Any, env_name: str, default: float) -> float:
    if value is None:
        value = os.getenv(env_name)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(parsed, 1.0))


def _bounded_float_setting(value: Any, env_name: str, default: float) -> float:
    return _nonnegative_float_setting(value, env_name, default)


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


def _question_features(question: str) -> dict[str, Any]:
    text = (question or "").lower()
    tokens = set(_TOKEN_RE.findall(text))
    entities = _proper_entities(question)
    clauses = _split_clauses(question)
    return {
        "entity_count": len(entities),
        "comparison": bool(tokens.intersection(_COMPARISON_TERMS) or " vs " in f" {text} "),
        "temporal": bool(_YEAR_RE.search(question or "") or tokens.intersection(_TEMPORAL_TERMS) or "as of" in text),
        "clause_count": len(clauses),
        "indirect_description": bool(re.search(r"\b(the|which|who|that)\b.+\b(that|which|who)\b", text)),
        "list_or_ranking": bool(tokens.intersection(_LIST_TERMS)),
    }


def _split_clauses(question: str) -> list[str]:
    parts = re.split(r";|\?|,(?=\s*(?:and|which|who|where|when)\b)|\s+\band\b\s+", question or "")
    clauses = [part.strip(" ,;?") for part in parts if len(part.strip(" ,;?")) >= 12]
    return clauses[:4] or ([question.strip()] if question.strip() else [])


def _anchors_for_text(text: str) -> list[str]:
    anchors: list[str] = []
    for match in _QUOTED_RE.findall(text or ""):
        value = (match[0] or match[1]).strip()
        if value:
            anchors.append(value)
    anchors.extend(_proper_entities(text))
    anchors.extend(_YEAR_RE.findall(text or ""))
    for token in _TOKEN_RE.findall((text or "").lower()):
        if len(token) > 3 and token not in _STOPWORDS:
            anchors.append(token)
    return _dedupe_preserve_order(anchors)[:12]


def _proper_entities(text: str) -> list[str]:
    entities: list[str] = []
    for match in _PROPER_NOUN_RE.findall(text or ""):
        value = match.strip()
        if value.lower() in _STOPWORDS or len(value) < 3:
            continue
        entities.append(value)
    return _dedupe_preserve_order(entities)


def _proper_entities_for_drift(text: str) -> list[str]:
    entities: list[str] = []
    for entity in _proper_entities(text):
        parts = entity.split()
        if parts and parts[0].lower() in _LEADING_QUERY_COMMANDS:
            entity = " ".join(parts[1:]).strip()
        if entity:
            entities.append(entity)
    return _dedupe_preserve_order(entities)


def _anchor_overlap(text: str, anchors: tuple[str, ...] | list[str]) -> float:
    anchors = [anchor for anchor in anchors if anchor]
    if not anchors:
        return 1.0
    matched = sum(1 for anchor in anchors if _phrase_in_text(anchor, text))
    return matched / len(anchors)


def _required_drift_anchors(
    facet: RequiredFacet,
    original_question: str,
    additional_anchors: tuple[str, ...] = (),
) -> tuple[str, ...]:
    facet_anchors = tuple(anchor for anchor in facet.anchors if anchor)
    if facet_anchors or additional_anchors:
        return tuple(_dedupe_preserve_order([*facet_anchors, *additional_anchors]))
    return tuple(_anchors_for_text(original_question))


def _text_satisfies_required_anchors(text: str, anchors: tuple[str, ...]) -> bool:
    return not anchors or all(_phrase_in_text(anchor, text) for anchor in anchors)


def _phrase_in_text(phrase: str, text: str) -> bool:
    phrase_norm = re.sub(r"\s+", " ", (phrase or "").strip().lower())
    text_norm = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not phrase_norm:
        return False
    if phrase_norm in text_norm:
        return True
    phrase_tokens = {t for t in _TOKEN_RE.findall(phrase_norm) if t not in _STOPWORDS}
    text_tokens = set(_TOKEN_RE.findall(text_norm))
    return bool(phrase_tokens and phrase_tokens.issubset(text_tokens))


def _result_group_has_anchor(chunks: list[dict[str, Any]], anchors: tuple[str, ...]) -> bool:
    if not chunks:
        return False
    if not anchors:
        return True
    top_text = "\n".join(_chunk_text(chunk) for chunk in chunks[: min(3, len(chunks))])
    return _text_has_any_anchor(top_text, anchors)


def _text_has_any_anchor(text: str, anchors: tuple[str, ...] | list[str]) -> bool:
    return any(_phrase_in_text(anchor, text) for anchor in anchors)


def _chunk_text(chunk: dict[str, Any]) -> str:
    return "\n".join(
        str(value or "")
        for value in (
            chunk.get("docnm_kwd"),
            chunk.get("document_name"),
            chunk.get("title"),
            chunk.get("content_with_weight"),
            chunk.get("content"),
        )
    )


def _chunk_identity(chunk: dict[str, Any]) -> str:
    if chunk.get("chunk_id"):
        return f"chunk:{chunk['chunk_id']}"
    section = ":".join(str(chunk.get(key) or "") for key in ("doc_id", "page", "position_int", "chunk_order"))
    if section.strip(":"):
        return f"section:{section}"
    content = chunk.get("content_with_weight") or chunk.get("content") or ""
    return "content:" + hashlib.sha256(str(content).encode("utf-8")).hexdigest()[:16]


def _refinement_chunk_identity(chunk: dict[str, Any]) -> str:
    if chunk.get("chunk_id"):
        return f"chunk:{chunk['chunk_id']}"
    doc_id = str(chunk.get("doc_id") or "")
    content = str(chunk.get("content_with_weight") or chunk.get("content") or "")
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"doc_content:{doc_id}:{content_hash}"


def _evidence_id(chunk: dict[str, Any]) -> str:
    return str(chunk.get("evidence_id") or chunk.get("chunk_id") or _refinement_chunk_identity(chunk))


def _chunk_source_id(chunk: dict[str, Any]) -> str:
    return str(chunk.get("doc_id") or chunk.get("document_id") or chunk.get("docnm_kwd") or chunk.get("document_name") or "")


def _chunk_quality(chunk: dict[str, Any]) -> float | None:
    for key in ("similarity", "similarity_score", "relevance_score", "score", "vector_similarity"):
        value = chunk.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _chunk_facet_ids(chunk: dict[str, Any]) -> list[str]:
    metadata = chunk.get("_ragflow_agentic_retrieval") or {}
    values = list(metadata.get("facet_ids") or [])
    value = chunk.get("facet_id") or metadata.get("facet_id")
    if value:
        values.append(value)
    return _dedupe_preserve_order([str(item) for item in values])


def _chunk_has_contradiction_signal(chunk: dict[str, Any]) -> bool:
    metadata = chunk.get("_ragflow_agentic_retrieval") or chunk.get("document_metadata") or chunk.get("metadata") or {}
    return bool(chunk.get("contradiction") or chunk.get("conflicting") or metadata.get("contradiction") or metadata.get("conflicting") or metadata.get("contradiction_with"))


def _exact_fact_gap(question: str, facet: RequiredFacet, evidence_text: str) -> dict[str, str] | None:
    evidence_type = facet.evidence_type.lower()
    description = facet.description.lower()
    if evidence_type == "date" or any(term in description for term in ("date", "year", "when")):
        required_years = _YEAR_RE.findall(" ".join((question, facet.description, *facet.anchors)))
        if required_years and not any(year in evidence_text for year in required_years):
            return {"type": "date", "description": f"exact date evidence missing for {facet.facet_id}"}
        if not required_years and not _YEAR_RE.search(evidence_text):
            return {"type": "date", "description": f"date evidence missing for {facet.facet_id}"}
    if evidence_type == "numeric" or any(term in description for term in ("number", "count", "amount", "revenue", "population")):
        if not re.search(r"\b\d[\d,]*(?:\.\d+)?\b", evidence_text):
            return {"type": "number", "description": f"numeric evidence missing for {facet.facet_id}"}
    if any(term in description for term in ("name", "who", "person")) and not _proper_entities(evidence_text):
        return {"type": "name", "description": f"name evidence missing for {facet.facet_id}"}
    return None


def _mark_refinement_chunk(
    chunk: dict[str, Any],
    followup: FollowupQuerySpec | None,
    *,
    selected: bool | None,
    reason: str | None,
) -> dict[str, Any]:
    marked = deepcopy(chunk)
    if followup is None:
        return marked
    metadata = dict(marked.get("_ragflow_agentic_retrieval") or {})
    metadata.update(
        {
            "plan_id": followup.plan_id,
            "facet_id": followup.facet_id,
            "subquery_id": followup.followup_id,
            "iteration_id": followup.iteration_id,
            "followup_id": followup.followup_id,
            "retrieval_variant": followup.retrieval_variant,
            "selected_for_context": selected,
            "rejection_reason": reason,
        }
    )
    for key, value in metadata.items():
        if value is not None:
            marked[key] = value
    marked["_ragflow_agentic_retrieval"] = metadata
    document_metadata = dict(marked.get("document_metadata") or marked.get("metadata") or {})
    document_metadata.update({key: value for key, value in metadata.items() if value is not None})
    marked["document_metadata"] = document_metadata
    return marked


def _followup_from_chunk(chunk: dict[str, Any]) -> FollowupQuerySpec | None:
    metadata = chunk.get("_ragflow_agentic_retrieval") or {}
    followup_id = str(chunk.get("followup_id") or metadata.get("followup_id") or "")
    facet_id = str(chunk.get("facet_id") or metadata.get("facet_id") or "")
    plan_id = str(chunk.get("plan_id") or metadata.get("plan_id") or "")
    if not followup_id or not facet_id:
        return None
    return FollowupQuerySpec(
        plan_id=plan_id,
        facet_id=facet_id,
        query="",
        iteration_id=str(chunk.get("iteration_id") or metadata.get("iteration_id") or ""),
        followup_id=followup_id,
        retrieval_variant=str(chunk.get("retrieval_variant") or metadata.get("retrieval_variant") or "hybrid_default"),  # type: ignore[arg-type]
    )


def _heuristic_judge(diagnostics: dict[str, Any], plan: BoundedRetrievalPlan, *, sufficient: bool) -> SufficiencyJudge:
    return SufficiencyJudge(
        sufficient=sufficient,
        confidence=1.0 if sufficient else 0.0,
        covered_facets=tuple(diagnostics.get("covered_facets") or []),
        missing_facets=tuple(diagnostics.get("missing_facets") or []),
        contradictions=tuple(diagnostics.get("contradictions") or []),
        exact_fact_gaps=tuple(diagnostics.get("exact_fact_gaps") or []),
        refusal_justified=False,
        recommended_followups=(),
    )


def build_deterministic_sufficiency_judge(
    diagnostics: dict[str, Any],
    plan: BoundedRetrievalPlan,
    cfg: AgenticRefinementConfig,
    *,
    iteration_id: str,
) -> SufficiencyJudge:
    """Build a conservative, application-owned Phase 6 decision.

    Existing heuristic mode remains diagnostic-only. This opt-in mode creates
    focused follow-ups only for explicit missing facets whose focused query is
    not a replay of a Phase 5 subquery.
    """
    if diagnostics.get("obviously_sufficient"):
        return _heuristic_judge(diagnostics, plan, sufficient=True)

    facets = {facet.facet_id: facet for facet in plan.required_facets}
    planned_queries = {_normalize_query_key(spec.query) for spec in plan.subqueries}
    followups: list[FollowupQuerySpec] = []
    for index, missing in enumerate(diagnostics.get("missing_facets") or [], start=1):
        facet_id = str(missing.get("facet_id") or "")
        facet = facets.get(facet_id)
        required_anchors = tuple(str(anchor) for anchor in missing.get("required_anchors") or () if str(anchor).strip())
        if facet is None or not required_anchors:
            continue
        query = " ".join(dict.fromkeys((facet.description, *required_anchors))).strip()
        if not query or _normalize_query_key(query) in planned_queries:
            continue
        planned_queries.add(_normalize_query_key(query))
        followups.append(
            FollowupQuerySpec(
                plan_id=plan.plan_id,
                facet_id=facet_id,
                query=query,
                keywords=" ".join(required_anchors),
                top_n=5,
                iteration_id=iteration_id,
                followup_id=f"{iteration_id}.deterministic.{index}",
            )
        )
        if len(followups) >= cfg.normalized().max_followup_queries:
            break
    return SufficiencyJudge(
        sufficient=False,
        confidence=1.0,
        covered_facets=tuple(diagnostics.get("covered_facets") or ()),
        missing_facets=tuple(diagnostics.get("missing_facets") or ()),
        contradictions=tuple(diagnostics.get("contradictions") or ()),
        exact_fact_gaps=tuple(diagnostics.get("exact_fact_gaps") or ()),
        refusal_justified=False,
        recommended_followups=tuple(followups),
    )


def _refinement_stop_result(
    current: dict[str, Any],
    candidate_current: dict[str, Any],
    iterations: list[RefinementIteration],
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    plan: BoundedRetrievalPlan,
    cfg: AgenticRefinementConfig,
    rag_trace: Any,
    stop_reason: str,
    latency_ms: float,
    *,
    fallback: bool = False,
    fallback_reason: str | None = None,
    coverage_before: float = 0.0,
    coverage_after: float = 0.0,
    marginal_gain: float = 0.0,
    followup_count: int = 0,
    rejected_followup_count: int = 0,
    cleanup_latency_ms: float = 0.0,
    timeout_owner: str | None = None,
) -> RefinementResult:
    payload = {
        "mode": cfg.mode,
        "plan_id": plan.plan_id,
        "lifecycle_id": f"{plan.plan_id}:refinement",
        "stop_reason": stop_reason,
        "fallback_reason": fallback_reason or (stop_reason if fallback else None),
        "accepted_new_evidence_count": len(accepted),
        "rejected_evidence_count": len(rejected),
        "latency_ms": round(latency_ms, 3),
        "iteration_count": len(iterations),
        "l_added_ms_estimate": round(latency_ms, 3),
        "l_added_formula": "N_iter * (L_judge + max_parallel(L_followups))",
        "coverage_before": round(coverage_before, 6),
        "coverage_after": round(coverage_after, 6),
        "marginal_gain": round(marginal_gain, 6),
        "followup_count": followup_count,
        "rejected_followup_count": rejected_followup_count,
        "cleanup_latency_ms": round(cleanup_latency_ms, 3),
        "timeout_owner": timeout_owner,
    }
    if fallback:
        _trace_refinement_event(rag_trace, "refinement_fallback_to_previous_context", payload)
    _trace_refinement_event(rag_trace, "refinement_stop", payload)
    return RefinementResult(
        kbinfos=deepcopy(current),
        candidate_kbinfos=deepcopy(candidate_current),
        iterations=tuple(iterations),
        accepted_chunks=tuple(deepcopy(accepted)),
        rejected_chunks=tuple(deepcopy(rejected)),
        changed=bool(accepted),
        fallback_to_previous_context=fallback,
        fallback_reason=fallback_reason or (stop_reason if fallback else None),
        stop_reason=stop_reason,
        diagnostics=payload,
    )


def _merge_doc_aggs(lane_results: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for lane in lane_results:
        for doc in (lane.get("kbinfos") or {}).get("doc_aggs") or []:
            if not isinstance(doc, dict):
                continue
            key = (str(doc.get("doc_id") or ""), str(doc.get("doc_name") or doc.get("docnm_kwd") or ""))
            counts[key] += int(doc.get("count") or 1)
    if not counts:
        for chunk in chunks:
            key = (str(chunk.get("doc_id") or ""), str(chunk.get("docnm_kwd") or chunk.get("document_name") or ""))
            counts[key] += 1
    return [{"doc_id": doc_id, "doc_name": doc_name, "count": count} for (doc_id, doc_name), count in sorted(counts.items(), key=lambda item: (-item[1], item[0][1], item[0][0])) if doc_id or doc_name]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = re.sub(r"\s+", " ", str(value or "").strip())
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _normalize_query_key(query: str) -> str:
    return re.sub(r"\s+", " ", str(query or "").strip()).lower()


def _trace_planner_event(trace: Any, event_name: str, payload: dict[str, Any]) -> None:
    if not trace:
        return
    payload = dict(payload or {})
    add_planner_event = getattr(trace, "add_agentic_planner_event", None)
    if callable(add_planner_event):
        try:
            add_planner_event(event_name, payload)
        except TypeError:
            add_planner_event(event_name=event_name, **payload)
    add_retrieval_event = getattr(trace, "add_agentic_retrieval_event", None)
    if callable(add_retrieval_event):
        add_retrieval_event(event_name, payload)


def _trace_agentic_event(trace: Any, stage: str, payload: dict[str, Any]) -> None:
    if not trace:
        return
    add_event = getattr(trace, "add_agentic_retrieval_event", None)
    if callable(add_event):
        add_event(stage, payload)


def _trace_refinement_event(trace: Any, event_name: str, payload: dict[str, Any]) -> None:
    if not trace:
        return
    payload = dict(payload or {})
    if payload.get("plan_id") and not payload.get("lifecycle_id"):
        payload["lifecycle_id"] = f"{payload['plan_id']}:refinement"
    add_event = getattr(trace, "add_agentic_refinement_event", None)
    if callable(add_event):
        try:
            add_event(event_name, **payload)
        except TypeError:
            add_event(event_name, payload)
        return
    add_retrieval_event = getattr(trace, "add_agentic_retrieval_event", None)
    if callable(add_retrieval_event):
        add_retrieval_event(event_name, payload)
