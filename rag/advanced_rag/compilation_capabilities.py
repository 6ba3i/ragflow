"""Source-grounded knowledge-compilation capabilities for the RAG Agent.

Compiled representations are navigation aids.  This module owns the stable
contracts and additive merge rules used after a navigator has resolved a
representation back to ordinary source chunks.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


CapabilityMode = Literal["off", "diagnostic", "shadow", "active"]
CapabilityOutcome = Literal[
    "unavailable",
    "skipped",
    "empty",
    "partial_lineage",
    "fallback",
    "success",
    "error",
    "cancelled",
]
EvidenceMode = Literal["document_scope", "scoped_ordinary", "source_backprojection", "synthetic"]

_OUTCOMES = {
    "unavailable",
    "skipped",
    "empty",
    "partial_lineage",
    "fallback",
    "success",
    "error",
    "cancelled",
}
_NORMALIZE_QUERY_RE = re.compile(r"\s+")


def _bounded_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(parsed, high))


@dataclass(frozen=True)
class CapabilityConfig:
    mode: CapabilityMode = "off"
    max_actions_per_turn: int = 2
    max_concurrent_actions: int = 2
    action_latency_budget_ms: int = 8000
    supplemental_evidence_units: int = 4
    protected_baseline_floor: int = 3

    @classmethod
    def from_env(cls, overrides: dict[str, Any] | None = None) -> "CapabilityConfig":
        overrides = overrides or {}

        def setting(key: str, env_name: str) -> Any:
            return overrides[key] if key in overrides else os.getenv(env_name)

        mode = str(setting("mode", "RAG_AGENT_COMPILATION_MODE") or "off").strip().lower()
        if mode not in {"off", "diagnostic", "shadow", "active"}:
            mode = "off"
        return cls(
            mode=mode,  # type: ignore[arg-type]
            max_actions_per_turn=_bounded_int(setting("max_actions_per_turn", "RAG_AGENT_COMPILATION_MAX_ACTIONS"), 2, 0, 8),
            max_concurrent_actions=_bounded_int(setting("max_concurrent_actions", "RAG_AGENT_COMPILATION_MAX_CONCURRENCY"), 2, 1, 4),
            action_latency_budget_ms=_bounded_int(setting("action_latency_budget_ms", "RAG_AGENT_COMPILATION_ACTION_TIMEOUT_MS"), 8000, 100, 60000),
            supplemental_evidence_units=_bounded_int(setting("supplemental_evidence_units", "RAG_AGENT_COMPILATION_SUPPLEMENTAL_UNITS"), 4, 0, 32),
            protected_baseline_floor=_bounded_int(setting("protected_baseline_floor", "RAG_AGENT_COMPILATION_BASELINE_FLOOR"), 3, 0, 32),
        )


@dataclass(frozen=True)
class CapabilityManifestEntry:
    capability_id: str
    representation_kind: str
    kb_id: str
    scope: Literal["dataset", "document"]
    template_ids: tuple[str, ...] = ()
    evidence_mode: EvidenceMode = "source_backprojection"
    source_grounded: bool = False
    citation_mode: str = "original_source"
    reader_ready: bool = False
    availability_id: str = ""
    estimated_cost_class: str = "bounded"
    configured: bool = False
    stored: bool = False
    blocked_reason: str | None = None

    @property
    def usable(self) -> bool:
        return self.stored and self.reader_ready and self.source_grounded and not self.blocked_reason

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "template_ids": list(self.template_ids), "usable": self.usable}


@dataclass(frozen=True)
class CapabilityManifest:
    entries: tuple[CapabilityManifestEntry, ...] = ()
    availability_fingerprint: str = ""

    @classmethod
    def build(cls, entries: list[CapabilityManifestEntry]) -> "CapabilityManifest":
        ordered = tuple(sorted(entries, key=lambda item: (item.kb_id, item.capability_id, item.availability_id)))
        payload = [entry.to_dict() for entry in ordered]
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return cls(entries=ordered, availability_fingerprint=fingerprint)

    def usable_entries(self) -> tuple[CapabilityManifestEntry, ...]:
        return tuple(entry for entry in self.entries if entry.usable)

    def to_dict(self) -> dict[str, Any]:
        return {
            "availability_fingerprint": self.availability_fingerprint,
            "entries": [entry.to_dict() for entry in self.entries],
        }


@dataclass(frozen=True)
class CapabilityActionSpec:
    action_id: str
    capability_id: str
    query: str
    facet_id: str = "primary"
    optional_scope: tuple[str, ...] = ()
    latency_budget_ms: int = 8000
    source_budget: int = 4

    def dedupe_key(self) -> tuple[str, str, tuple[str, ...]]:
        normalized = _NORMALIZE_QUERY_RE.sub(" ", self.query.strip().lower())
        return self.capability_id, normalized, tuple(sorted(self.optional_scope))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "optional_scope": list(self.optional_scope)}


@dataclass(frozen=True)
class CapabilityAttempt:
    attempt_id: str
    action_id: str
    capability_id: str
    query: str
    outcome: CapabilityOutcome
    fallback: str | None = None
    latency_ms: float = 0.0
    selected_node_ids: tuple[str, ...] = ()
    selected_doc_ids: tuple[str, ...] = ()
    structural_path_or_hops: tuple[Any, ...] = ()
    structural_score_family: str | None = None
    source_ids_requested: tuple[str, ...] = ()
    source_ids_loaded: tuple[str, ...] = ()
    source_lineage_complete: bool = False
    error_code: str | None = None

    def __post_init__(self) -> None:
        if self.outcome not in _OUTCOMES:
            raise ValueError(f"unsupported capability outcome: {self.outcome}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in (
            "selected_node_ids",
            "selected_doc_ids",
            "structural_path_or_hops",
            "source_ids_requested",
            "source_ids_loaded",
        ):
            data[key] = list(data[key])
        return data


@dataclass(frozen=True)
class CapabilityEvidence:
    source_chunk: dict[str, Any]
    capability_id: str
    attempt_id: str
    evidence_mode: EvidenceMode
    source_grounded: bool
    source_chunk_ids: tuple[str, ...] = ()
    source_doc_ids: tuple[str, ...] = ()
    structural_path: tuple[Any, ...] = ()
    local_rank: int = 0
    citation_mode: str = "original_source"
    score_family: str = "source_backprojection"
    facet_id: str = "primary"

    def occurrence(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "attempt_id": self.attempt_id,
            "evidence_mode": self.evidence_mode,
            "source_grounded": self.source_grounded,
            "source_chunk_ids": list(self.source_chunk_ids),
            "source_doc_ids": list(self.source_doc_ids),
            "structural_path": list(self.structural_path),
            "local_rank": self.local_rank,
            "citation_mode": self.citation_mode,
            "score_family": self.score_family,
            "facet_id": self.facet_id,
        }


@dataclass
class CapabilityExecutionResult:
    attempt: CapabilityAttempt
    evidence: list[CapabilityEvidence] = field(default_factory=list)


def new_action(
    entry: CapabilityManifestEntry,
    query: str,
    *,
    facet_id: str = "primary",
    scope: list[str] | tuple[str, ...] | None = None,
    latency_budget_ms: int = 8000,
    source_budget: int = 4,
) -> CapabilityActionSpec:
    return CapabilityActionSpec(
        action_id=f"cap-action-{uuid.uuid4().hex[:12]}",
        capability_id=entry.capability_id,
        query=query,
        facet_id=facet_id,
        optional_scope=tuple(scope or ()),
        latency_budget_ms=latency_budget_ms,
        source_budget=source_budget,
    )


def select_capability_actions(
    manifest: CapabilityManifest,
    query: str,
    *,
    config: CapabilityConfig,
    facet_id: str = "primary",
    doc_scope: list[str] | None = None,
    explicit_document_scope: bool = False,
) -> tuple[list[CapabilityActionSpec], list[dict[str, Any]]]:
    """Select bounded deterministic actions without treating availability as execution."""
    if config.mode == "off" or config.max_actions_per_turn <= 0:
        return [], []
    priority = {"dataset_nav": 0, "tree": 1, "page_index": 2, "mind_map": 3, "knowledge_graph": 4}
    selected: list[CapabilityActionSpec] = []
    skipped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for entry in sorted(manifest.entries, key=lambda item: (priority.get(item.representation_kind, 99), item.capability_id)):
        if not entry.usable:
            skipped.append({"capability_id": entry.capability_id, "reason": entry.blocked_reason or "unavailable"})
            continue
        scope = doc_scope or []
        if entry.representation_kind == "dataset_nav" and explicit_document_scope:
            skipped.append({"capability_id": entry.capability_id, "reason": "existing_document_scope"})
            continue
        if entry.scope == "document" and not scope:
            skipped.append({"capability_id": entry.capability_id, "reason": "document_scope_unavailable"})
            continue
        action = new_action(
            entry,
            query,
            facet_id=facet_id,
            scope=scope,
            latency_budget_ms=config.action_latency_budget_ms,
            source_budget=config.supplemental_evidence_units,
        )
        if action.dedupe_key() in seen:
            skipped.append({"capability_id": entry.capability_id, "reason": "duplicate_action"})
            continue
        if len(selected) >= config.max_actions_per_turn:
            skipped.append({"capability_id": entry.capability_id, "reason": "action_budget"})
            continue
        seen.add(action.dedupe_key())
        selected.append(action)
    return selected, skipped


def source_identity(chunk: dict[str, Any]) -> tuple[str, str]:
    chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "").strip()
    if chunk_id:
        return "chunk", chunk_id
    doc_id = str(chunk.get("doc_id") or "").strip()
    content = " ".join(str(chunk.get("content_with_weight") or chunk.get("content") or "").split())
    digest = hashlib.sha256(f"{doc_id}\0{content}".encode()).hexdigest()
    return "fallback", digest


def merge_ordinary_evidence(baseline: dict[str, Any], supplemental: dict[str, Any] | None) -> dict[str, Any]:
    """Merge ordinary lanes without replacing baseline rank or scores."""
    result = {key: value for key, value in baseline.items() if key not in {"chunks", "doc_aggs", "total"}}
    chunks = [dict(chunk) for chunk in baseline.get("chunks") or [] if isinstance(chunk, dict)]
    by_identity = {source_identity(chunk): chunk for chunk in chunks}
    for chunk in (supplemental or {}).get("chunks") or []:
        if not isinstance(chunk, dict):
            continue
        existing = by_identity.get(source_identity(chunk))
        if existing is not None:
            incoming_agentic = chunk.get("_ragflow_agentic_retrieval")
            if isinstance(incoming_agentic, dict):
                existing_agentic = existing.get("_ragflow_agentic_retrieval")
                if not isinstance(existing_agentic, dict):
                    existing_agentic = {}
                    existing["_ragflow_agentic_retrieval"] = existing_agentic
                occurrences = existing_agentic.setdefault("ordinary_route_occurrences", [])
                occurrence = {
                    key: incoming_agentic.get(key) for key in ("plan_id", "facet_id", "subquery_id", "retrieval_call_id", "retrieval_variant", "lineage_rank") if incoming_agentic.get(key) is not None
                }
                if occurrence and occurrence not in occurrences:
                    occurrences.append(occurrence)
            continue
        copied = dict(chunk)
        chunks.append(copied)
        by_identity[source_identity(copied)] = copied
    doc_aggs: list[Any] = []
    seen_aggs: set[str] = set()
    for aggregate in [*(baseline.get("doc_aggs") or []), *((supplemental or {}).get("doc_aggs") or [])]:
        key = json.dumps(aggregate, sort_keys=True, default=str)
        if key in seen_aggs:
            continue
        seen_aggs.add(key)
        doc_aggs.append(aggregate)
    result.update({"total": len(chunks), "chunks": chunks, "doc_aggs": doc_aggs})
    return result


def _merge_occurrence(chunk: dict[str, Any], occurrence: dict[str, Any]) -> None:
    metadata = chunk.get("_ragflow_compilation")
    if not isinstance(metadata, dict):
        metadata = {}
        chunk["_ragflow_compilation"] = metadata
    occurrences = metadata.get("route_occurrences")
    if not isinstance(occurrences, list):
        occurrences = []
        metadata["route_occurrences"] = occurrences
    key = (occurrence.get("capability_id"), occurrence.get("attempt_id"), occurrence.get("local_rank"))
    existing = {(item.get("capability_id"), item.get("attempt_id"), item.get("local_rank")) for item in occurrences if isinstance(item, dict)}
    if key not in existing:
        occurrences.append(occurrence)
    metadata["source_grounded"] = all(item.get("source_grounded") is True for item in occurrences if isinstance(item, dict))
    metadata["citation_mode"] = "original_source"


def merge_capability_routes(target: dict[str, Any], source: dict[str, Any]) -> bool:
    """Union capability routes onto an already accumulated source chunk."""
    metadata = source.get("_ragflow_compilation")
    if not isinstance(metadata, dict):
        return False
    occurrences = metadata.get("route_occurrences")
    if not isinstance(occurrences, list) or not occurrences:
        return False
    for occurrence in occurrences:
        if isinstance(occurrence, dict):
            _merge_occurrence(target, occurrence)
    return True


@dataclass
class AdditiveEvidenceAccumulator:
    baseline_chunks: list[dict[str, Any]]
    supplemental_limit: int
    baseline_floor: int = 0
    supplemental_admitted: int = 0
    decisions: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._chunks = list(self.baseline_chunks)
        self._baseline_floor_protected = False
        self._by_identity = {source_identity(chunk): chunk for chunk in self._chunks if isinstance(chunk, dict)}

    def _protect_baseline_floor(self) -> None:
        if self._baseline_floor_protected:
            return
        for rank, chunk in enumerate(self._chunks, start=1):
            if rank > self.baseline_floor or not isinstance(chunk, dict):
                continue
            metadata = chunk.get("_ragflow_compilation")
            if not isinstance(metadata, dict):
                metadata = {}
                chunk["_ragflow_compilation"] = metadata
            metadata.update(
                {
                    "evidence_mode": "ordinary_baseline",
                    "baseline_rank": rank,
                    "baseline_protected": True,
                    "source_grounded": True,
                    "citation_mode": "original_source",
                }
            )
        self._baseline_floor_protected = True

    @property
    def chunks(self) -> list[dict[str, Any]]:
        return self._chunks

    def add(self, evidence: CapabilityEvidence) -> bool:
        occurrence = evidence.occurrence()
        identity = source_identity(evidence.source_chunk)
        if not evidence.source_grounded or evidence.citation_mode != "original_source":
            self.decisions.append({"identity": list(identity), "admitted": False, "reason": "not_source_grounded", "baseline_protected": False})
            return False
        existing = self._by_identity.get(identity)
        if existing is not None:
            self._protect_baseline_floor()
            _merge_occurrence(existing, occurrence)
            self.decisions.append({"identity": list(identity), "admitted": True, "reason": "duplicate_route_union", "baseline_protected": existing in self.baseline_chunks})
            return True
        if self.supplemental_admitted >= self.supplemental_limit:
            self.decisions.append({"identity": list(identity), "admitted": False, "reason": "global_supplemental_budget", "baseline_protected": False})
            return False
        self._protect_baseline_floor()
        chunk = dict(evidence.source_chunk)
        _merge_occurrence(chunk, occurrence)
        self._chunks.append(chunk)
        self._by_identity[identity] = chunk
        self.supplemental_admitted += 1
        self.decisions.append({"identity": list(identity), "admitted": True, "reason": "supplemental_budget", "baseline_protected": False})
        return True


def make_attempt(
    action: CapabilityActionSpec,
    outcome: CapabilityOutcome,
    *,
    started_at: float,
    **fields: Any,
) -> CapabilityAttempt:
    return CapabilityAttempt(
        attempt_id=str(fields.pop("attempt_id", f"cap-attempt-{uuid.uuid4().hex[:12]}")),
        action_id=action.action_id,
        capability_id=action.capability_id,
        query=action.query,
        outcome=outcome,
        latency_ms=round((time.monotonic() - started_at) * 1000.0, 3),
        **fields,
    )
