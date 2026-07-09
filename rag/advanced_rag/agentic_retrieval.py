#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#
"""Phase 5 bounded query-time retrieval planning.

The planner is an LLM helper that emits one bounded retrieval plan. Planner
failures fall back directly to baseline retrieval; this module does not
synthesize deterministic retrieval plans.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from collections import defaultdict
from copy import copy
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

from rag.utils.retrieval_diagnostics import make_retrieval_variant, resolve_variant_knobs

AgenticRetrievalMode = Literal["off", "diagnostic", "shadow", "active"]
RetrievalVariant = Literal["hybrid_default", "keyword_first", "embedding_retry"]

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|2100)\b")
_QUOTED_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][\w-]*(?:\s+[A-Z][\w-]*){0,5}")
_SMALL_TALK = {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay"}
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
_REQUIRED_PLANNER_KEYS = {
    "plan_id",
    "original_question",
    "complexity",
    "trigger_reasons",
    "required_facets",
    "subqueries",
    "merge_policy",
    "drift_controls",
}


class AgenticPlannerError(Exception):
    """Raised when the LLM planner cannot produce a valid bounded plan."""

    def __init__(self, reason: str, detail: str | None = None):
        self.reason = reason
        self.detail = detail
        super().__init__(reason if detail is None else f"{reason}: {detail}")


@dataclass(frozen=True)
class AgenticRetrievalConfig:
    enabled: bool = False
    mode: AgenticRetrievalMode = "off"
    max_subqueries: int = 3
    subquery_top_n: int = 6
    planner_timeout_ms: int = 1200
    retrieval_timeout_ms: int = 1500
    max_extra_retrieval_calls: int = 3
    min_anchor_overlap: float = 0.5
    simple_query_bypass: bool = True
    latency_budget_ms: int = 2500
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
        return cls(
            enabled=enabled,
            mode=mode,  # type: ignore[arg-type]
            max_subqueries=_int_setting(overrides.get("agentic_retrieval_max_subqueries"), "AGENTIC_RETRIEVAL_MAX_SUBQUERIES", 3) or 3,
            subquery_top_n=_int_setting(overrides.get("agentic_retrieval_subquery_top_n"), "AGENTIC_RETRIEVAL_SUBQUERY_TOP_N", 6) or 6,
            planner_timeout_ms=_int_setting(overrides.get("agentic_retrieval_planner_timeout_ms"), "AGENTIC_RETRIEVAL_PLANNER_TIMEOUT_MS", 1200) or 1200,
            retrieval_timeout_ms=_int_setting(overrides.get("agentic_retrieval_retrieval_timeout_ms"), "AGENTIC_RETRIEVAL_RETRIEVAL_TIMEOUT_MS", 1500) or 1500,
            max_extra_retrieval_calls=_int_setting(overrides.get("agentic_retrieval_max_extra_retrieval_calls"), "AGENTIC_RETRIEVAL_MAX_EXTRA_RETRIEVAL_CALLS", 3) or 3,
            min_anchor_overlap=_float_setting(overrides.get("agentic_retrieval_drift_min_anchor_overlap"), "AGENTIC_RETRIEVAL_DRIFT_MIN_ANCHOR_OVERLAP", 0.5) or 0.5,
            simple_query_bypass=_bool_setting(overrides.get("agentic_retrieval_simple_query_bypass"), "AGENTIC_RETRIEVAL_SIMPLE_QUERY_BYPASS", True),
            latency_budget_ms=_int_setting(overrides.get("agentic_retrieval_latency_budget_ms"), "AGENTIC_RETRIEVAL_LATENCY_BUDGET_MS", 2500) or 2500,
            rrf_k=_int_setting(overrides.get("agentic_retrieval_rrf_k"), "AGENTIC_RETRIEVAL_RRF_K", 60) or 60,
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
) -> dict[str, Any]:
    cfg = cfg or AgenticRetrievalConfig()
    doc_ids = list(attachments or [])
    kb_ids = list(getattr(dialog, "kb_ids", []) or [])
    history_summary = _summarize_history(history or [])
    plan_id = f"plan-{uuid.uuid4().hex[:12]}"
    return {
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
        "output_schema": _planner_output_schema(),
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


def build_llm_planner_prompt(planner_input: dict[str, Any], cfg: AgenticRetrievalConfig) -> tuple[str, list[dict[str, str]]]:
    """Build the strict JSON-only prompt for the LLM retrieval planner."""
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
        "output_schema": planner_input.get("output_schema") or _planner_output_schema(),
    }
    system_prompt = (
        "You are a retrieval planning helper for RAGFlow.\n"
        "You do not answer the user.\n"
        "You create one bounded retrieval plan for the existing retriever.\n"
        "Return strict JSON only.\n"
        "The plan must help retrieval find evidence chunks for the original question.\n"
        "Do not include markdown fences, prose, explanations, citations, or tool calls."
    )
    user_prompt = (
        "Create exactly one bounded retrieval plan as a JSON object.\n"
        f"Use plan_id exactly: {plan_id}.\n"
        f"Create 2 to {max(2, cfg.max_subqueries)} subqueries when planning is triggered, but never exceed max_subqueries.\n"
        "Preserve original and facet anchors, do not invent unrelated new entities, and keep subqueries bounded.\n"
        "Use only allowed retrieval_variant and evidence_type values.\n"
        "Set docid_scope to null unless the provided document scope is explicitly required.\n"
        "Set top_n at or below subquery_top_n.\n"
        "Return only JSON matching the output_schema.\n\n"
        + json.dumps(prompt_input, ensure_ascii=False, sort_keys=True)
    )
    return system_prompt, [{"role": "user", "content": user_prompt}]


def parse_llm_planner_json(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise AgenticPlannerError("planner_invalid_json", f"non_string_response:{type(raw).__name__}")
    text = raw.strip()
    if not text:
        raise AgenticPlannerError("planner_invalid_json", "empty_response")
    if "```" in text:
        raise AgenticPlannerError("planner_invalid_json", "markdown_fence")
    if not text.startswith("{"):
        raise AgenticPlannerError("planner_invalid_json", "response_not_json_object")
    if not text.endswith("}"):
        raise AgenticPlannerError("planner_invalid_json", "trailing_or_incomplete_json")
    decoder = json.JSONDecoder()
    try:
        parsed, end = decoder.raw_decode(text)
    except json.JSONDecodeError as exc:
        raise AgenticPlannerError("planner_invalid_json", str(exc)) from exc
    if text[end:].strip():
        raise AgenticPlannerError("planner_invalid_json", "multiple_json_or_trailing_text")
    if not isinstance(parsed, dict) or not parsed:
        raise AgenticPlannerError("planner_invalid_json", "json_not_nonempty_object")
    missing = sorted(_REQUIRED_PLANNER_KEYS.difference(parsed))
    if missing:
        raise AgenticPlannerError("planner_invalid_json", "missing_keys:" + ",".join(missing))
    return parsed


async def generate_llm_plan(
    planner_input: dict[str, Any],
    cfg: AgenticRetrievalConfig,
    *,
    chat_mdl: Any,
    rag_trace: Any = None,
) -> BoundedRetrievalPlan:
    if chat_mdl is None or not callable(getattr(chat_mdl, "async_chat", None)):
        _trace_planner_event(rag_trace, "planner_missing_chat_model", {"plan_id": planner_input.get("plan_id"), "mode": cfg.mode})
        raise AgenticPlannerError("planner_missing_chat_model", "chat_mdl.async_chat is required")
    plan_id = str(planner_input.get("plan_id") or f"plan-{uuid.uuid4().hex[:12]}")
    planner_input = {**planner_input, "plan_id": plan_id}
    system_prompt, messages = build_llm_planner_prompt(planner_input, cfg)
    gen_conf = {"temperature": 0.0, "top_p": 0.1}
    timeout_s = max(0.001, cfg.planner_timeout_ms / 1000.0)
    model_name = str(getattr(chat_mdl, "llm_name", None) or getattr(chat_mdl, "model_name", None) or getattr(chat_mdl, "model_config", {}).get("llm_name", ""))
    started = time.monotonic()
    _trace_planner_event(rag_trace, "planner_llm_start", {"plan_id": plan_id, "mode": cfg.mode, "model": model_name})
    try:
        raw = await asyncio.wait_for(chat_mdl.async_chat(system_prompt, messages, gen_conf), timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        latency_ms = round((time.monotonic() - started) * 1000.0, 3)
        _trace_planner_event(rag_trace, "planner_llm_timeout", {"plan_id": plan_id, "mode": cfg.mode, "latency_ms": latency_ms, "model": model_name})
        raise AgenticPlannerError("planner_timeout", f"timeout_ms={cfg.planner_timeout_ms}") from exc
    except Exception as exc:
        latency_ms = round((time.monotonic() - started) * 1000.0, 3)
        _trace_planner_event(rag_trace, "planner_llm_error", {"plan_id": plan_id, "mode": cfg.mode, "latency_ms": latency_ms, "model": model_name, "error": str(exc)})
        raise AgenticPlannerError("planner_error", str(exc)) from exc

    try:
        raw_plan = parse_llm_planner_json(raw)
    except AgenticPlannerError as exc:
        latency_ms = round((time.monotonic() - started) * 1000.0, 3)
        _trace_planner_event(
            rag_trace,
            "planner_llm_invalid_json",
            {"plan_id": plan_id, "mode": cfg.mode, "latency_ms": latency_ms, "model": model_name, "raw_response_chars": len(str(raw or "")), "fallback_reason": exc.reason, "detail": exc.detail},
        )
        raise

    raw_plan["plan_id"] = plan_id
    raw_plan["original_question"] = str(planner_input.get("original_question") or planner_input.get("question") or raw_plan.get("original_question") or "")
    for subquery in raw_plan.get("subqueries") or []:
        if isinstance(subquery, dict):
            subquery["plan_id"] = plan_id
    validation = validate_plan(raw_plan, cfg, original_question=raw_plan["original_question"])
    latency_ms = round((time.monotonic() - started) * 1000.0, 3)
    if not validation.valid or validation.errors:
        _trace_planner_event(
            rag_trace,
            "planner_llm_validation_failed",
            {"plan_id": plan_id, "mode": cfg.mode, "latency_ms": latency_ms, "model": model_name, "validation_errors": validation.errors, "fallback_reason": validation.fallback_reason},
        )
        raise AgenticPlannerError("planner_validation_failed", ";".join(validation.errors))
    _trace_planner_event(
        rag_trace,
        "planner_llm_success",
        {
            "plan_id": plan_id,
            "mode": cfg.mode,
            "latency_ms": latency_ms,
            "model": model_name,
            "raw_response_chars": len(str(raw or "")),
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
        new_entities = {
            entity
            for entity in set(_proper_entities(subquery.query)).difference(original_entities)
            if entity.lower() not in original_anchor_keys
        }
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
    )
    fallback_reason = ";".join(errors) if errors else None
    return PlanValidationResult(fallback_reason is None and bool(kept_subqueries), errors, normalized, fallback_reason=fallback_reason)


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
) -> BoundedRetrievalResult:
    started = time.monotonic()
    tasks = [
        asyncio.create_task(_execute_subquery(
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
        ))
        for spec in plan.subqueries[: min(cfg.max_subqueries, cfg.max_extra_retrieval_calls)]
    ]
    overall_timeout = max(0.001, cfg.latency_budget_ms / 1000.0)
    try:
        lane_results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=overall_timeout)
    except BaseException as exc:
        if isinstance(exc, asyncio.CancelledError):
            raise
        for task in tasks:
            task.cancel()
        diagnostics = {"accepted_subqueries": 0, "rejected_subqueries": len(tasks), "rejections": [{"rejection_reason": "overall_timeout_or_exception", "error": str(exc)}]}
        _trace_agentic_event(rag_trace, "fallback", {"plan_id": plan.plan_id, "reason": "overall_timeout_or_exception", **diagnostics})
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
        _trace_agentic_event(rag_trace, "fallback", {"plan_id": plan.plan_id, "reason": reason, **diagnostics})
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
        _trace_agentic_event(rag_trace, "fallback", {"plan_id": plan.plan_id, "reason": "all_subqueries_rejected", **diagnostics})
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
        _trace_agentic_event(rag_trace, "fallback", {"plan_id": plan.plan_id, "reason": "latency_budget_exceeded", **diagnostics})
        return BoundedRetrievalResult(
            kbinfos={"total": 0, "chunks": [], "doc_aggs": []},
            plan=plan,
            accepted_subqueries=len(accepted),
            rejected_subqueries=len(rejected),
            fallback_to_baseline=True,
            fallback_reason="latency_budget_exceeded",
            diagnostics=diagnostics,
        )
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
        return {"accepted": False, "rejection_reason": "lane_exception", "error": str(result)}
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
    if chat_mdl is None:
        _trace_agentic_event(rag_trace, "planner_llm_fallback_to_baseline", {"reason": "planner_missing_chat_model", "mode": cfg.mode})
        _trace_planner_event(rag_trace, "planner_llm_fallback_to_baseline", {"fallback_reason": "planner_missing_chat_model", "mode": cfg.mode})
        return {"kbinfos": baseline, **baseline, "fallback_reason": "planner_missing_chat_model", "fallback_to_baseline": True}

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
    try:
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
    except Exception as exc:
        _trace_agentic_event(rag_trace, "planner_llm_fallback_to_baseline", {"reason": "planner_error", "detail": str(exc), "mode": cfg.mode})
        _trace_planner_event(rag_trace, "planner_llm_fallback_to_baseline", {"fallback_reason": "planner_error", "detail": str(exc), "mode": cfg.mode})
        return {"kbinfos": baseline, **baseline, "fallback_reason": f"planner_error:{exc}", "fallback_to_baseline": True}

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
    try:
        return await asyncio.wait_for(_execute_subquery_work(spec, plan, **kwargs), timeout=max(0.001, kwargs["cfg"].retrieval_timeout_ms / 1000.0))
    except BaseException as exc:
        if isinstance(exc, asyncio.CancelledError):
            raise
        rag_trace = kwargs.get("rag_trace")
        _trace_agentic_event(rag_trace, "subquery_rejected", {"plan_id": plan.plan_id, "subquery_id": spec.subquery_id, "reason": "lane_timeout_or_exception", "error": str(exc)})
        return {"subquery": spec, "accepted": False, "rejection_reason": "lane_timeout_or_exception", "error": str(exc)}


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
        _trace_agentic_event(rag_trace, "subquery_rejected", {"plan_id": plan.plan_id, "subquery_id": spec.subquery_id, "reason": "retrieval_exception", "error": str(exc)})
        return {"subquery": spec, "accepted": False, "rejection_reason": "retrieval_exception", "error": str(exc)}
    if apply_toc and chat_mdl:
        cks = await retriever.retrieval_by_toc(spec.query, kbinfos.get("chunks", []), tenant_ids, chat_mdl, top_n)
        if cks:
            kbinfos["chunks"] = cks
    if apply_children:
        kbinfos["chunks"] = retriever.retrieval_by_children(kbinfos.get("chunks", []), tenant_ids)

    retrieval_call_id = None
    if rag_trace:
        retrieval_call_id = rag_trace.add_retrieval_call(
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
    attach_planning_metadata(
        kbinfos.get("chunks", []),
        plan_id=plan.plan_id,
        facet_id=spec.facet_id,
        subquery_id=spec.subquery_id,
        retrieval_call_id=retrieval_call_id,
        retrieval_variant=knobs["retrieval_variant"],
    )
    if rag_trace:
        rag_trace.add_evidence_from_chunks(kbinfos.get("chunks", []), source_type="kb", retrieval_call_id=retrieval_call_id)

    facet = next((f for f in plan.required_facets if f.facet_id == spec.facet_id), None)
    anchors = tuple(dict.fromkeys(list(spec.must_have_terms) + list(facet.anchors if facet else ()) + _anchors_for_text(plan.original_question)[:3]))
    if enforce_result_anchor_drift and not _result_group_has_anchor(kbinfos.get("chunks", []), anchors):
        attach_planning_metadata(
            kbinfos.get("chunks", []),
            plan_id=plan.plan_id,
            facet_id=spec.facet_id,
            subquery_id=spec.subquery_id,
            retrieval_call_id=retrieval_call_id,
            retrieval_variant=knobs["retrieval_variant"],
            selected_for_context=False,
            rejection_reason="result_anchor_drift",
        )
        _trace_agentic_event(rag_trace, "subquery_rejected", {"plan_id": plan.plan_id, "subquery_id": spec.subquery_id, "reason": "result_anchor_drift"})
        return {"subquery": spec, "accepted": False, "rejection_reason": "result_anchor_drift", "retrieval_call_id": retrieval_call_id}
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
    )



def _planner_output_schema() -> dict[str, Any]:
    return {
        "plan_id": "string; copy the provided plan_id exactly",
        "original_question": "string",
        "complexity": "simple|multi_facet|multi_hop|comparison|temporal|list|unknown",
        "trigger_reasons": ["string"],
        "required_facets": [
            {"facet_id": "f1", "description": "string", "anchors": ["term"], "evidence_type": "definition|numeric|date|comparison|quote|list_item"}
        ],
        "subqueries": [
            {
                "subquery_id": "sq1",
                "facet_id": "f1",
                "query": "string",
                "keywords": "string|null",
                "docid_scope": None,
                "top_n": "integer <= subquery_top_n",
                "retrieval_variant": "hybrid_default|keyword_first|embedding_retry",
                "must_have_terms": ["string"],
                "forbidden_new_entities": ["string"],
                "rationale": "short string",
            }
        ],
        "merge_policy": {"strategy": "subquery_rrf_then_context_builder", "max_chunks_per_facet": 4},
        "drift_controls": {"anchor_entities": ["string"], "min_anchor_overlap": 0.5, "allow_new_entities": False},
    }


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


def _anchor_overlap(text: str, anchors: tuple[str, ...] | list[str]) -> float:
    anchors = [anchor for anchor in anchors if anchor]
    if not anchors:
        return 1.0
    matched = sum(1 for anchor in anchors if _phrase_in_text(anchor, text))
    return matched / len(anchors)


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
    return [
        {"doc_id": doc_id, "doc_name": doc_name, "count": count}
        for (doc_id, doc_name), count in sorted(counts.items(), key=lambda item: (-item[1], item[0][1], item[0][0]))
        if doc_id or doc_name
    ]


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
