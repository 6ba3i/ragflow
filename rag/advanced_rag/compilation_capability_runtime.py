"""Discovery and execution adapters for RAG Agent compilation capabilities."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from common import settings
from common.doc_store.doc_store_base import OrderByExpr
from common.misc_utils import thread_pool_exec
from rag.advanced_rag.compilation_capabilities import (
    CapabilityActionSpec,
    CapabilityEvidence,
    CapabilityExecutionResult,
    CapabilityManifest,
    CapabilityManifestEntry,
    make_attempt,
)
from rag.nlp import search


_LOG = logging.getLogger(__name__)
_STRUCTURAL_KINDS = {
    "tree": {"tree", "toc", "raptor"},
    "page_index": {"page_index", "timeline"},
    "mind_map": {"mindmap", "mind_map"},
}
_BLOCKED_KINDS = {"artifacts", "wiki", "session_graph", "session_essence", "timeline", "list", "set", "hypergraph", "skill", "skill_all"}


def _normalize_kind(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized == "pageindex":
        return "page_index"
    if normalized == "mindmap":
        return "mind_map"
    return normalized


def _configured_templates(kb: Any) -> dict[str, set[str]]:
    configured: dict[str, set[str]] = {}
    parser_config = getattr(kb, "parser_config", None) or {}
    for raw_kind in ("tree", "page_index", "mind_map", "mindmap", "knowledge_graph", "artifacts", "wiki", "session_graph", "session_essence"):
        if parser_config.get(raw_kind):
            configured.setdefault(_normalize_kind(raw_kind), set()).add("parser_config")
    try:
        from api.db.services.compilation_template_service import CompilationTemplateService
        from rag.svr.task_executor_refactor.chunk_post_processor import _parser_config_compilation_template_ids

        template_ids = _parser_config_compilation_template_ids(parser_config, getattr(kb, "tenant_id", ""))
        for template_id in template_ids:
            template = CompilationTemplateService.get_saved(template_id, getattr(kb, "tenant_id", ""))
            config = (template or {}).get("config") or {}
            kind = _normalize_kind(config.get("kind"))
            if kind:
                configured.setdefault(kind, set()).add(str(template_id))
    except Exception:
        _LOG.exception("capability template discovery failed for kb=%s", getattr(kb, "id", ""))
    return configured


async def _read_rows(kb: Any, condition: dict[str, Any], fields: list[str], limit: int = 128) -> list[dict[str, Any]]:
    tenant_id = str(getattr(kb, "tenant_id", "") or "")
    kb_id = str(getattr(kb, "id", "") or "")
    if not tenant_id or not kb_id:
        return []
    index_name = search.index_name(tenant_id)
    try:
        exists = await thread_pool_exec(settings.docStoreConn.index_exist, index_name, kb_id)
        if not exists:
            return []
        result = await thread_pool_exec(
            settings.docStoreConn.search,
            fields,
            [],
            condition,
            [],
            OrderByExpr(),
            0,
            limit,
            index_name,
            [kb_id],
        )
        values = settings.docStoreConn.get_fields(result, fields) or {}
        return [dict(row) for row in values.values() if isinstance(row, dict)]
    except Exception:
        _LOG.exception("capability row discovery failed for kb=%s condition=%s", kb_id, condition)
        return []


def _graph_row_details(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    details: dict[str, dict[str, Any]] = {}
    for row in rows:
        raw_kind = row.get("compilation_template_kind_kwd") or row.get("compile_kwd")
        kind = _normalize_kind(raw_kind)
        if kind == "mindmap":
            kind = "mind_map"
        if kind == "timeline" and _normalize_kind(row.get("compilation_template_kind_kwd")) == "page_index":
            kind = "page_index"
        try:
            graph = json.loads(row.get("content_with_weight") or "{}")
        except Exception:
            graph = {}
        entities = graph.get("entities") or [] if isinstance(graph, dict) else []
        relations = graph.get("relations") or [] if isinstance(graph, dict) else []
        source_ids = []
        for item in [*entities, *relations]:
            if isinstance(item, dict):
                source_ids.extend(str(source_id) for source_id in item.get("source_chunk_ids") or [] if source_id)
        item = details.setdefault(kind, {"template_ids": set(), "source_ids": set(), "rows": 0})
        item["rows"] += 1
        item["source_ids"].update(source_ids)
        item["template_ids"].update(str(template_id) for template_id in row.get("compilation_template_ids") or [] if template_id)
    return details


async def discover_capabilities(kbs: list[Any]) -> CapabilityManifest:
    entries: list[CapabilityManifestEntry] = []
    for kb in kbs:
        kb_id = str(getattr(kb, "id", "") or "")
        configured = await thread_pool_exec(_configured_templates, kb)
        graph_rows = await _read_rows(
            kb,
            {"knowledge_graph_kwd": ["graph"]},
            ["content_with_weight", "compile_kwd", "compilation_template_ids", "compilation_template_kind_kwd"],
        )
        graph_details = _graph_row_details(graph_rows)
        nav_rows = await _read_rows(
            kb,
            {"compile_kwd": ["dataset_nav"]},
            ["id", "name", "content_with_weight", "doc_id", "doc_ids_kwd", "type_kwd"],
        )
        nav_text_ready = False
        nav_scope_ready = False
        for row in nav_rows:
            try:
                payload = json.loads(row.get("content_with_weight") or "{}")
            except Exception:
                payload = {}
            if str(row.get("name") or payload.get("description") or "").strip():
                nav_text_ready = True
            if row.get("doc_id") or any(row.get("doc_ids_kwd") or []):
                nav_scope_ready = True
        if nav_rows or configured.get("tree") or configured.get("page_index"):
            configured_nav = bool(configured.get("tree") or configured.get("page_index"))
            blocked = None
            if not nav_rows:
                blocked = "representation_rows_missing"
            elif not nav_scope_ready:
                blocked = "document_scope_unavailable"
            elif not nav_text_ready:
                blocked = "lexical_fallback_unusable"
            entries.append(
                CapabilityManifestEntry(
                    capability_id=f"{kb_id}:dataset_nav",
                    representation_kind="dataset_nav",
                    kb_id=kb_id,
                    scope="dataset",
                    template_ids=tuple(sorted((configured.get("tree") or set()) | (configured.get("page_index") or set()))),
                    evidence_mode="document_scope",
                    source_grounded=True,
                    reader_ready=True,
                    availability_id=f"dataset_nav:{len(nav_rows)}:{int(nav_text_ready)}:{int(nav_scope_ready)}",
                    estimated_cost_class="index_probe",
                    configured=configured_nav,
                    stored=bool(nav_rows),
                    blocked_reason=blocked,
                )
            )

        for representation_kind, accepted_kinds in _STRUCTURAL_KINDS.items():
            stored_detail = next((detail for kind, detail in graph_details.items() if kind in accepted_kinds or kind == representation_kind), None)
            configured_ids = configured.get(representation_kind) or set()
            if not stored_detail and not configured_ids:
                continue
            stored = bool(stored_detail and stored_detail["rows"])
            source_grounded = bool(stored_detail and stored_detail["source_ids"])
            blocked = None
            if not stored:
                blocked = "representation_rows_missing"
            elif not source_grounded:
                blocked = "source_lineage_missing"
            entries.append(
                CapabilityManifestEntry(
                    capability_id=f"{kb_id}:{representation_kind}",
                    representation_kind=representation_kind,
                    kb_id=kb_id,
                    scope="document",
                    template_ids=tuple(sorted(configured_ids | (stored_detail["template_ids"] if stored_detail else set()))),
                    evidence_mode="source_backprojection",
                    source_grounded=source_grounded,
                    reader_ready=True,
                    availability_id=f"{representation_kind}:{stored_detail['rows'] if stored_detail else 0}:{len(stored_detail['source_ids']) if stored_detail else 0}",
                    estimated_cost_class="structural_model_selection",
                    configured=bool(configured_ids),
                    stored=stored,
                    blocked_reason=blocked,
                )
            )

        kg_rows = await _read_rows(
            kb,
            {"compilation_template_kind_kwd": ["knowledge_graph"], "knowledge_graph_kwd": ["entity", "relation"]},
            ["id", "content_with_weight", "source_chunk_ids", "doc_id", "compilation_template_ids", "knowledge_graph_kwd"],
        )
        configured_kg = configured.get("knowledge_graph") or set()
        if kg_rows or configured_kg:
            source_ids = {str(source_id) for row in kg_rows for source_id in row.get("source_chunk_ids") or [] if source_id}
            template_ids = configured_kg | {str(template_id) for row in kg_rows for template_id in row.get("compilation_template_ids") or [] if template_id}
            blocked = None if kg_rows and source_ids else ("representation_rows_missing" if not kg_rows else "source_lineage_missing")
            entries.append(
                CapabilityManifestEntry(
                    capability_id=f"{kb_id}:knowledge_graph",
                    representation_kind="knowledge_graph",
                    kb_id=kb_id,
                    scope="dataset",
                    template_ids=tuple(sorted(template_ids)),
                    source_grounded=bool(source_ids),
                    reader_ready=True,
                    availability_id=f"knowledge_graph:{len(kg_rows)}:{len(source_ids)}",
                    estimated_cost_class="graph_traversal",
                    configured=bool(configured_kg),
                    stored=bool(kg_rows),
                    blocked_reason=blocked,
                )
            )

        for kind in sorted(_BLOCKED_KINDS.intersection(configured)):
            entries.append(
                CapabilityManifestEntry(
                    capability_id=f"{kb_id}:{kind}",
                    representation_kind=kind,
                    kb_id=kb_id,
                    scope="dataset" if kind in {"artifacts", "wiki", "session_graph", "session_essence", "skill", "skill_all"} else "document",
                    template_ids=tuple(sorted(configured[kind])),
                    evidence_mode="synthetic" if kind in {"artifacts", "wiki", "session_graph", "session_essence"} else "source_backprojection",
                    source_grounded=False,
                    citation_mode="blocked",
                    reader_ready=False,
                    availability_id=f"blocked:{kind}",
                    estimated_cost_class="unsupported",
                    configured=True,
                    stored=False,
                    blocked_reason="reader_or_citation_contract_incomplete",
                )
            )
    return CapabilityManifest.build(entries)


async def execute_capability_action(
    tools: Any,
    entry: CapabilityManifestEntry,
    action: CapabilityActionSpec,
    *,
    ordinary_retrieve: Callable[[str, list[str], int], Awaitable[dict[str, Any]]],
) -> CapabilityExecutionResult:
    from rag.advanced_rag.harness.tools.navigation import navigate_compiled_sources, navigate_graph_sources, route_dataset_documents

    started = time.monotonic()
    if not entry.usable:
        return CapabilityExecutionResult(make_attempt(action, "unavailable", started_at=started, fallback=entry.blocked_reason))
    try:
        if entry.representation_kind == "dataset_nav":
            routed = await route_dataset_documents(tools, action.query, top_k=max(1, action.source_budget), kb_ids={entry.kb_id})
            doc_ids = [str(doc_id) for doc_id in routed.get("doc_ids") or [] if doc_id]
            if not doc_ids:
                return CapabilityExecutionResult(
                    make_attempt(
                        action,
                        "empty",
                        started_at=started,
                        selected_node_ids=tuple(routed.get("selected_node_ids") or ()),
                        structural_score_family="dataset_nav_local",
                    )
                )
            scoped = await ordinary_retrieve(action.query, doc_ids, action.source_budget)
            chunks = [chunk for chunk in scoped.get("chunks") or [] if isinstance(chunk, dict)]
            attempt = make_attempt(
                action,
                "success" if chunks else "fallback",
                started_at=started,
                selected_node_ids=tuple(routed.get("selected_node_ids") or ()),
                selected_doc_ids=tuple(doc_ids),
                structural_score_family="dataset_nav_local",
                source_ids_requested=tuple(str(chunk.get("chunk_id") or chunk.get("id") or "") for chunk in chunks),
                source_ids_loaded=tuple(str(chunk.get("chunk_id") or chunk.get("id") or "") for chunk in chunks),
                source_lineage_complete=bool(chunks),
            )
            evidence = [
                CapabilityEvidence(
                    source_chunk=chunk,
                    capability_id=entry.capability_id,
                    attempt_id=attempt.attempt_id,
                    evidence_mode="scoped_ordinary",
                    source_grounded=True,
                    source_chunk_ids=(str(chunk.get("chunk_id") or chunk.get("id") or ""),),
                    source_doc_ids=(str(chunk.get("doc_id") or ""),),
                    local_rank=rank,
                    score_family="ordinary_scoped",
                    facet_id=action.facet_id,
                )
                for rank, chunk in enumerate(chunks, start=1)
            ]
            return CapabilityExecutionResult(attempt, evidence)

        if entry.representation_kind in _STRUCTURAL_KINDS:
            result = await navigate_compiled_sources(
                tools,
                action.query,
                doc_scope=list(action.optional_scope),
                kinds=_STRUCTURAL_KINDS[entry.representation_kind],
                label=f"{entry.representation_kind} navigation",
                noun=entry.representation_kind.replace("_", " "),
                source_budget=action.source_budget,
            )
            score_family = f"{entry.representation_kind}_local"
        elif entry.representation_kind == "knowledge_graph":
            result = await navigate_graph_sources(
                tools,
                action.query,
                doc_scope=list(action.optional_scope) or None,
                kb_ids={entry.kb_id},
                source_budget=action.source_budget,
            )
            score_family = "graph_local"
        else:
            return CapabilityExecutionResult(make_attempt(action, "unavailable", started_at=started, fallback="reader_not_registered"))

        chunks = [chunk for chunk in result.get("chunks") or [] if isinstance(chunk, dict)]
        requested = tuple(str(source_id) for source_id in result.get("source_ids_requested") or [] if source_id)
        loaded = tuple(str(source_id) for source_id in result.get("source_ids_loaded") or [] if source_id)
        outcome = result.get("outcome") or ("success" if chunks else "empty")
        attempt = make_attempt(
            action,
            outcome,
            started_at=started,
            selected_node_ids=tuple(str(node_id) for node_id in result.get("selected_node_ids") or [] if node_id),
            selected_doc_ids=tuple(action.optional_scope),
            structural_path_or_hops=tuple(result.get("path_or_hops") or ()),
            structural_score_family=score_family,
            source_ids_requested=requested,
            source_ids_loaded=loaded,
            source_lineage_complete=bool(requested) and len(loaded) == len(requested),
        )
        evidence = [
            CapabilityEvidence(
                source_chunk=chunk,
                capability_id=entry.capability_id,
                attempt_id=attempt.attempt_id,
                evidence_mode="source_backprojection",
                source_grounded=True,
                source_chunk_ids=(str(chunk.get("chunk_id") or chunk.get("id") or ""),),
                source_doc_ids=(str(chunk.get("doc_id") or ""),),
                structural_path=tuple(result.get("path_or_hops") or ()),
                local_rank=rank,
                score_family="source_backprojection",
                facet_id=action.facet_id,
            )
            for rank, chunk in enumerate(chunks, start=1)
        ]
        return CapabilityExecutionResult(attempt, evidence)
    except Exception as exc:
        _LOG.exception("capability execution failed capability=%s", entry.capability_id)
        return CapabilityExecutionResult(make_attempt(action, "error", started_at=started, fallback="ordinary_baseline", error_code=type(exc).__name__))
