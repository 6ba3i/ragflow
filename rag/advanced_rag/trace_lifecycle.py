"""Bounded lifecycle reconstruction for agentic trace diagnostics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LifecycleRecord:
    lifecycle_id: str
    population: str = "primary"
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def terminal_events(self) -> list[dict[str, Any]]:
        return [event for event in self.events if event.get("terminal") is True or event.get("event_name") in {"planner_llm_success", "refinement_judge_success"} or event.get("event") in {"planner_llm_success", "refinement_judge_success"}]


def reconstruct_lifecycles(trace: dict[str, Any], *, population: str = "primary") -> list[LifecycleRecord]:
    grouped: dict[str, LifecycleRecord] = {}
    for section in (
        trace.get("agentic_planning") or [],
        trace.get("agentic_refinement") or [],
        trace.get("agentic_retrieval") or [],
    ):
        for raw in section:
            if not isinstance(raw, dict):
                continue
            lifecycle_id = raw.get("lifecycle_id") or raw.get("operation_id") or raw.get("iteration_id")
            if not lifecycle_id and raw.get("stage") == "planner_cache":
                lifecycle_id = f"planner.cache.{raw.get('scope_hash') or 'unknown'}"
            lifecycle_id = lifecycle_id or "unknown"
            record = grouped.setdefault(str(lifecycle_id), LifecycleRecord(str(lifecycle_id), population))
            record.events.append(dict(raw))
    return list(grouped.values())


def lifecycle_counts(records: list[LifecycleRecord]) -> dict[str, int]:
    counts = Counter()
    for record in records:
        names = {event.get("event_name") or event.get("event") or event.get("stage") for event in record.events}
        if "planner_cache" in names:
            statuses = {event.get("cache_status") for event in record.events}
            if "hit_valid" in statuses:
                counts["cache_valid_reuse"] += 1
                continue
        if "planner_llm_fallback_to_deterministic" in names:
            counts["deterministic_fallback"] += 1
            continue
        names.discard(None)
        if "planner_cache_hit_valid" in names:
            counts["cache_valid_reuse"] += 1
            continue
        if "planner_deterministic_fallback" in names or "refinement_fallback_to_previous_context" in names:
            counts["deterministic_fallback"] += 1
            continue
        if "cancelled" in names or any("cancel" in str(name) for name in names):
            counts["cancelled"] += 1
            continue
        has_success = "planner_llm_success" in names or "refinement_judge_success" in names
        has_repair = any("repair" in str(name) for name in names) or any(event.get("repair") is True for event in record.events)
        if has_success:
            counts["valid"] += 1
            counts["repaired_valid" if has_repair else "first_attempt_valid"] += 1
        elif "planner_llm_timeout" in names or "refinement_judge_timeout" in names:
            counts["timeout"] += 1
        elif "planner_llm_validation_failed" in names or "refinement_judge_validation_failed" in names:
            counts["validation_failed"] += 1
        elif "planner_llm_repair_failed" in names or "refinement_judge_repair_failed" in names:
            counts["repair_failed"] += 1
        else:
            counts["terminal_unknown"] += 1
    if not counts:
        return {}
    return dict(counts)
