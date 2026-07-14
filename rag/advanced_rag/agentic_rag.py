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

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import asdict, is_dataclass, replace
from typing import Any, List

import json_repair
from copy import deepcopy
from rag.advanced_rag.agentic_retrieval import (
    AgenticPlannerError,
    AgenticRefinementConfig,
    AgenticRefinementError,
    AgenticRetrievalConfig,
    build_planner_input,
    execute_bounded_plan,
    generate_llm_plan,
    run_refinement_loop,
    should_plan,
)
from api.db.services.doc_metadata_service import DocMetadataService
from api.db.services.document_service import DocumentService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.llm_service import LLMBundle
from common import settings
from common.misc_utils import thread_pool_exec
from rag.app.tag import label_question
from rag.llm.tool_decorator import tool
from rag.prompts.generator import citation_prompt, gen_meta_filter, kb_prompt
from api.db.db_models import Document, Knowledgebase
from rag.utils.tavily_conn import Tavily
from common.token_utils import num_tokens_from_string
from rag.utils.rag_trace import RagTraceCollector, _now_ms
from rag.utils.context_builder import EvidenceBundleConfig, apply_context_builder_to_kbinfos, mark_chunks_source_type
from rag.utils.retrieval_diagnostics import make_retrieval_variant, resolve_variant_knobs
from rag.utils.retrieval_fusion import FusionConfig


def _refinement_storage_identity(chunk: dict[str, Any]) -> str:
    if chunk.get("chunk_id"):
        return f"chunk:{chunk['chunk_id']}"
    doc_id = str(chunk.get("doc_id") or "")
    content = str(chunk.get("content_with_weight") or chunk.get("content") or "")
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"doc_content:{doc_id}:{content_hash}"


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _rag_agent_cleanup_margin_s(timeout_ceiling_s: float) -> float:
    return min(3.0, max(0.0, timeout_ceiling_s) * 0.5)


def _canonical_agentic_value(value: Any, *, key: str | None = None) -> Any:
    if key == "plan_id":
        return "<volatile-plan-id>"
    if key == "api_key_payload" and isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return "<redacted-credential-payload>"
        if not isinstance(value, dict):
            return "<redacted-credential-payload>"
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        canonical = {}
        credential_keys = {
            "api_key",
            "access_token",
            "authorization",
            "auth_token",
            "bearer_token",
            "client_secret",
            "credentials",
            "password",
            "private_key",
            "refresh_token",
            "secret",
            "secret_key",
            "token",
        }
        credential_suffixes = (
            "_ak",
            "_api_key",
            "_account_key",
            "_access_key",
            "_credential",
            "_credentials",
            "_password",
            "_private_key",
            "_secret",
            "_secret_key",
            "_sid",
            "_sk",
            "_token",
        )
        for item_key, item_value in sorted(value.items(), key=lambda item: str(item[0])):
            key_name = str(item_key).lower()
            if key_name in credential_keys or key_name.endswith(credential_suffixes):
                continue
            canonical[str(item_key)] = _canonical_agentic_value(item_value, key=str(item_key))
        return canonical
    if isinstance(value, (list, tuple)):
        return [_canonical_agentic_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_canonical_agentic_value(item) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True, default=str),
        )
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _agentic_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        _canonical_agentic_value(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class RAGTools:
    def __init__(self, 
                 tenant_ids: list[str],
                 chat_mdl: LLMBundle, 
                 embed_mdl: LLMBundle | None = None, 
                 kb_ids: List[str] | None = None,
                 kbs: list[Knowledgebase] | None = [], 
                 tav: Tavily | None = None,
                 meta_data_filter: dict | None = None,
                 user_defined_prompts: dict | None = None,
                 trace: RagTraceCollector | None = None,
                 context_builder_config: EvidenceBundleConfig | None = None,
                 agentic_retrieval_config: AgenticRetrievalConfig | None = None,
                 agentic_refinement_config: AgenticRefinementConfig | None = None,
                 ):
        self.tenant_ids = tenant_ids
        self.chat_mdl = deepcopy(chat_mdl)
        self.embed_mdl = embed_mdl
        self.field_map = {}
        self.sql_kbs = []
        self.kbs = []
        self.kb_ids = []
        def _exclude_sql_kb(kb):
            if kb.parser_config and "field_map" in kb.parser_config:
                self.field_map.update(kb.parser_config["field_map"])
                self.sql_kbs.append(kb)
            else:
                self.kbs.append(kb)
                self.kb_ids.append(kb.id)
        if kb_ids:
            for kb in KnowledgebaseService.get_by_ids(kb_ids):
                _exclude_sql_kb(kb)
        elif kbs:
            for kb in kbs:
                _exclude_sql_kb(kb)

        self.tav = tav
        self.meta_data_filter = meta_data_filter
        self.user_defined_prompts = user_defined_prompts or {}
        self.trace = trace
        self.context_builder_config = context_builder_config or EvidenceBundleConfig.from_env()
        self.agentic_retrieval_config = agentic_retrieval_config or AgenticRetrievalConfig.from_env()
        self.agentic_refinement_config = agentic_refinement_config or AgenticRefinementConfig.from_env()
        # Accumulator for chunks/doc_aggs across tool calls within a turn —
        # populated by ``search_knowledge_bases`` and ``search_structured_data``
        # so the final answer can cite everything retrieved so far.
        self.kbinfos: dict[str, list] = {"chunks": [], "doc_aggs": []}
        # Set to True after the first retrieval tool has stamped the citation
        # rules onto its output, so subsequent retrieval calls don't repeat.
        self._citations_injected: bool = False
        self._keyword_first_chunk_ids: dict[tuple[str, str, tuple[str, ...]], set[str]] = {}
        self._last_context_query: str | None = None
        self._planner_elapsed_ms = 0.0
        self._planner_calls_used = 0
        self._planner_attempts_by_fingerprint: dict[str, int] = {}
        self._planner_terminal_ledger: dict[str, dict[str, Any]] = {}
        self._validated_plan_cache: dict[str, Any] = {}
        self._completed_search_ledger: dict[str, dict[str, Any]] = {}
        self._agentic_search_lock = asyncio.Lock()
        self._rag_agent_tool_timeout_s = 120.0
        self._rag_agent_tool_expired_tasks: set[asyncio.Task[Any]] = set()
        self._agentic_turn_budget_ms = _bounded_env_int("AGENTIC_RAG_TURN_AGENTIC_BUDGET_MS", 120000, 1, 120000)
        self._agentic_elapsed_ms = 0.0
        self._agentic_tool_elapsed_ms = 0.0
        self._agentic_stage_ms = {"planner": 0.0, "phase5": 0.0, "phase6": 0.0, "cleanup": 0.0}
        self._planner_operation_counter = 0

        tools = [
            self.formalize_question,
            self.select_documents,
            self.search_knowledge_bases,
        ]
        if self.tav:
            tools.append(self.web_search)
        if meta_data_filter:
            tools.append(self.filter_docs_by_metadata)
        if self.sql_kbs:
            tools.append(self.search_structured_data)
        if self.kb_ids:
            tools.append(self.summarize_document)
        chat_mdl.bind_tools(None, tools)

    def context_kbinfos(self, *, stage: str | None = None, start_citation_index: int = 0, query: str | None = None) -> dict[str, Any]:
        result = apply_context_builder_to_kbinfos(
            self.kbinfos,
            self.context_builder_config,
            start_citation_index=start_citation_index,
            query=query or self._last_context_query,
        )
        if self.trace:
            if result.bundle:
                self.trace.add_context_builder_summary(result.bundle.summary(), stage=stage)
            else:
                self.trace.add_context_builder_summary(
                    {"enabled": False, "candidate_evidence_count": len(self.kbinfos.get("chunks", []))},
                    stage=stage,
                )
        return result.kbinfos

    def _prompt_start_idx(self, start_idx: int) -> int:
        return 0 if self.context_builder_config.enabled else start_idx

    def _selected_phase5_kbinfos(self, kbinfos: dict[str, Any], *, question: str) -> dict[str, Any]:
        return apply_context_builder_to_kbinfos(
            deepcopy(kbinfos),
            self.context_builder_config,
            query=question,
        ).kbinfos

    def _model_identity(self) -> dict[str, Any]:
        model_config = getattr(self.chat_mdl, "model_config", {})
        return {
            "class": f"{type(self.chat_mdl).__module__}.{type(self.chat_mdl).__qualname__}",
            "llm_name": getattr(self.chat_mdl, "llm_name", None),
            "model_name": getattr(self.chat_mdl, "model_name", None),
            "model_config": model_config if isinstance(model_config, dict) else {},
        }

    def _embedding_identity(self) -> dict[str, Any] | None:
        if self.embed_mdl is None:
            return None
        model_config = getattr(self.embed_mdl, "model_config", {})
        return {
            "class": f"{type(self.embed_mdl).__module__}.{type(self.embed_mdl).__qualname__}",
            "llm_name": getattr(self.embed_mdl, "llm_name", None),
            "model_name": getattr(self.embed_mdl, "model_name", None),
            "model_config": model_config if isinstance(model_config, dict) else {},
        }

    def _planner_fingerprint(self, planner_input: dict[str, Any]) -> str:
        return _agentic_fingerprint(
            {
                "planner_input": planner_input,
                "model": self._model_identity(),
                "planner_config": self.agentic_retrieval_config,
            }
        )

    def _search_fingerprint(
        self,
        *,
        planner_input: dict[str, Any],
        question: str,
        keywords: str,
        using_embedding: bool,
        top_n: int,
        similarity_threshold: float,
        docid_scope: list[str] | None,
    ) -> str:
        return _agentic_fingerprint(
            {
                "planner_input": planner_input,
                "search": {
                    "question": question,
                    "keywords": keywords,
                    "using_embedding": using_embedding,
                    "top_n": top_n,
                    "similarity_threshold": similarity_threshold,
                    "docid_scope": sorted(docid_scope or []),
                },
                "scope": {
                    "tenant_ids": sorted(self.tenant_ids),
                    "kb_ids": sorted(self.kb_ids),
                    "metadata_filters": self.meta_data_filter or {},
                },
                "model": self._model_identity(),
                "embedding_model": self._embedding_identity(),
                "retriever": f"{type(settings.retriever).__module__}.{type(settings.retriever).__qualname__}",
                "retrieval_config": self.agentic_retrieval_config,
                "refinement_config": self.agentic_refinement_config,
                "context_builder_config": self.context_builder_config,
                "fusion_config": FusionConfig.from_env(),
                "output_settings": {
                    "chat_max_length": getattr(self.chat_mdl, "max_length", None),
                    "user_defined_prompts": self.user_defined_prompts,
                },
            }
        )

    def _remaining_agentic_budget_ms(self) -> float:
        return max(0.0, self._agentic_turn_budget_ms - self._agentic_elapsed_ms)

    def _account_agentic_stage(
        self,
        stage: str,
        elapsed_ms: float,
        *,
        cleanup_ms: float = 0.0,
        timeout_owner: str | None = None,
    ) -> None:
        cleanup_ms = max(0.0, min(cleanup_ms, elapsed_ms))
        measured_work_ms = max(0.0, elapsed_ms - cleanup_ms)
        admitted_ms = min(measured_work_ms, self._remaining_agentic_budget_ms())
        budget_overrun_ms = max(0.0, measured_work_ms - admitted_ms)
        self._agentic_stage_ms[stage] += admitted_ms
        self._agentic_stage_ms["cleanup"] += cleanup_ms
        self._agentic_elapsed_ms += admitted_ms
        if self.trace:
            self.trace.add_agentic_retrieval_event(
                "stage_accounting",
                {
                    "stage_name": stage,
                    "stage_measured_latency_ms": round(measured_work_ms, 3),
                    "stage_latency_ms": round(admitted_ms, 3),
                    "stage_budget_overrun_ms": round(budget_overrun_ms, 3),
                    "cleanup_latency_ms": round(cleanup_ms, 3),
                    "cumulative_planner_ms": round(self._agentic_stage_ms["planner"], 3),
                    "cumulative_phase5_ms": round(self._agentic_stage_ms["phase5"], 3),
                    "cumulative_phase6_ms": round(self._agentic_stage_ms["phase6"], 3),
                    "cumulative_cleanup_ms": round(self._agentic_stage_ms["cleanup"], 3),
                    "cumulative_agentic_ms": round(self._agentic_elapsed_ms, 3),
                    "remaining_turn_agentic_budget_ms": round(self._remaining_agentic_budget_ms(), 3),
                    "timeout_owner": timeout_owner,
                },
            )

    def _trace_turn_budget_exhausted(self, operation_id: str | None = None) -> None:
        if self.trace:
            self.trace.add_agentic_retrieval_event(
                "agentic_turn_budget_exhausted",
                {
                    "operation_id": operation_id,
                    "terminal": True,
                    "fallback": "baseline_retrieval",
                    "remaining_turn_agentic_budget_ms": round(self._remaining_agentic_budget_ms(), 3),
                    "timeout_owner": "turn_agentic_budget",
                },
            )

    async def _coordinated_plan(
        self,
        planner_input: dict[str, Any],
        *,
        tool_deadline: float,
    ) -> Any | None:
        cfg = self.agentic_retrieval_config
        if self._remaining_agentic_budget_ms() <= 0:
            self._trace_turn_budget_exhausted()
            return None
        fingerprint = self._planner_fingerprint(planner_input)
        safe_hash = fingerprint[:16]
        if fingerprint in self._planner_terminal_ledger:
            terminal = self._planner_terminal_ledger[fingerprint]
            if self.trace:
                self.trace.add_agentic_retrieval_event(
                    "planner_cache",
                    {"scope_hash": safe_hash, "cache_status": "hit_terminal", "terminal": True, **terminal},
                )
            return None
        if cfg.plan_cache_enabled and fingerprint in self._validated_plan_cache:
            if self.trace:
                self.trace.add_agentic_retrieval_event(
                    "planner_cache",
                    {"scope_hash": safe_hash, "cache_status": "hit_valid", "terminal": False},
                )
            return deepcopy(self._validated_plan_cache[fingerprint])
        if self.trace:
            self.trace.add_agentic_retrieval_event(
                "planner_cache",
                {"scope_hash": safe_hash, "cache_status": "miss", "terminal": False},
            )

        max_attempts = max(1, min(int(cfg.planner_max_attempts_per_key), 4))
        max_calls = max(1, min(int(cfg.planner_max_calls_per_turn), 16))
        total_budget_ms = max(1, min(int(cfg.planner_total_budget_ms), 120000))
        repair_errors: tuple[str, ...] = ()
        while self._planner_attempts_by_fingerprint.get(fingerprint, 0) < max_attempts:
            remaining_planner_ms = total_budget_ms - self._planner_elapsed_ms
            remaining_turn_ms = self._remaining_agentic_budget_ms()
            remaining_tool_ms = max(0.0, (tool_deadline - time.monotonic()) * 1000.0)
            if self._planner_calls_used >= max_calls:
                terminal = {"failure_class": "call_ceiling", "retryable": False, "fallback_reason": "planner_turn_call_ceiling"}
                self._planner_terminal_ledger[fingerprint] = terminal
                break
            if remaining_tool_ms <= 0:
                if self.trace:
                    self.trace.add_agentic_retrieval_event(
                        "rag_agent_tool_timeout",
                        {
                            "scope_hash": safe_hash,
                            "outcome": "timeout",
                            "timeout_owner": "rag_agent_tool",
                            "terminal": True,
                            "admitted_new_work": False,
                        },
                    )
                return None
            if min(remaining_planner_ms, remaining_turn_ms) <= 0:
                reason = "agentic_turn_budget_exhausted" if remaining_turn_ms <= 0 else "planner_total_budget_exhausted"
                terminal = {"failure_class": "budget", "retryable": False, "fallback_reason": reason}
                self._planner_terminal_ledger[fingerprint] = terminal
                if remaining_turn_ms <= 0:
                    self._trace_turn_budget_exhausted()
                break

            attempt = self._planner_attempts_by_fingerprint.get(fingerprint, 0) + 1
            self._planner_attempts_by_fingerprint[fingerprint] = attempt
            self._planner_calls_used += 1
            self._planner_operation_counter += 1
            operation_id = f"planner-{self._planner_operation_counter:03d}-{safe_hash[:8]}"
            attempt_timeout_ms = max(1, int(min(cfg.planner_timeout_ms, remaining_planner_ms, remaining_turn_ms, remaining_tool_ms)))
            attempt_started = time.monotonic()
            attempt_cleanup_ms = 0.0
            attempt_timeout_owner = None

            def account_planner_cleanup(cleanup_ms: float, timeout_owner: str) -> None:
                nonlocal attempt_cleanup_ms, attempt_timeout_owner
                attempt_cleanup_ms += cleanup_ms
                attempt_timeout_owner = timeout_owner

            try:
                plan = await generate_llm_plan(
                    planner_input,
                    cfg,
                    chat_mdl=self.chat_mdl,
                    rag_trace=self.trace,
                    repair_errors=repair_errors,
                    attempt_timeout_ms=attempt_timeout_ms,
                    operation_id=operation_id,
                    cleanup_callback=account_planner_cleanup,
                )
            except asyncio.CancelledError:
                elapsed_ms = (time.monotonic() - attempt_started) * 1000.0
                self._planner_elapsed_ms += max(0.0, elapsed_ms - attempt_cleanup_ms)
                timeout_owner = "rag_agent_tool" if asyncio.current_task() in self._rag_agent_tool_expired_tasks else "upstream_cancellation"
                self._account_agentic_stage("planner", elapsed_ms, cleanup_ms=attempt_cleanup_ms, timeout_owner=timeout_owner)
                raise
            except AgenticPlannerError as exc:
                elapsed_ms = (time.monotonic() - attempt_started) * 1000.0
                attempt_work_ms = max(0.0, elapsed_ms - attempt_cleanup_ms)
                self._planner_elapsed_ms += attempt_work_ms
                timeout_owner = attempt_timeout_owner or ("planner_attempt" if exc.failure_class == "timeout" else None)
                self._account_agentic_stage("planner", elapsed_ms, cleanup_ms=attempt_cleanup_ms, timeout_owner=timeout_owner)
                if self.trace:
                    self.trace.add_agentic_retrieval_event(
                        "planner_attempt",
                        {
                            "operation_id": operation_id,
                            "scope_hash": safe_hash,
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "calls_used": self._planner_calls_used,
                            "max_calls": max_calls,
                            "failure_class": exc.failure_class,
                            "retryable": exc.retryable,
                            "repair": bool(repair_errors),
                            "attempt_timeout_ms": attempt_timeout_ms,
                            "attempt_latency_ms": round(attempt_work_ms, 3),
                            "cleanup_latency_ms": round(attempt_cleanup_ms, 3),
                            "remaining_planner_budget_ms": round(max(0.0, total_budget_ms - self._planner_elapsed_ms), 3),
                            "status_code": exc.status_code,
                            "bounded_retry_after_ms": exc.bounded_retry_after_ms,
                            "safe_category": exc.safe_category,
                        },
                    )
                if not exc.retryable or attempt >= max_attempts or self._planner_calls_used >= max_calls:
                    self._planner_terminal_ledger[fingerprint] = {
                        "failure_class": exc.failure_class,
                        "retryable": False,
                        "fallback_reason": exc.reason,
                    }
                    break
                if exc.failure_class in {"invalid_json", "validation"}:
                    repair_errors = (f"{exc.reason}:{exc.detail or 'invalid'}",)
                else:
                    repair_errors = ()
                remaining_tool_ms = max(0.0, (tool_deadline - time.monotonic()) * 1000.0)
                retry_delay_ms = min(
                    exc.bounded_retry_after_ms or 0,
                    total_budget_ms - self._planner_elapsed_ms,
                    self._remaining_agentic_budget_ms(),
                    remaining_tool_ms,
                )
                if retry_delay_ms > 0:
                    delay_started = time.monotonic()
                    delay_timeout_owner = None
                    try:
                        await asyncio.sleep(retry_delay_ms / 1000.0)
                    except asyncio.CancelledError:
                        delay_timeout_owner = "rag_agent_tool" if asyncio.current_task() in self._rag_agent_tool_expired_tasks else "upstream_cancellation"
                        raise
                    finally:
                        delay_elapsed_ms = (time.monotonic() - delay_started) * 1000.0
                        self._planner_elapsed_ms += delay_elapsed_ms
                        self._account_agentic_stage("planner", delay_elapsed_ms, timeout_owner=delay_timeout_owner)
                continue
            else:
                elapsed_ms = (time.monotonic() - attempt_started) * 1000.0
                attempt_work_ms = max(0.0, elapsed_ms - attempt_cleanup_ms)
                self._planner_elapsed_ms += attempt_work_ms
                self._account_agentic_stage("planner", elapsed_ms, cleanup_ms=attempt_cleanup_ms, timeout_owner=attempt_timeout_owner)
                remaining_planner_ms = total_budget_ms - self._planner_elapsed_ms
                remaining_turn_ms = self._remaining_agentic_budget_ms()
                remaining_tool_ms = max(0.0, (tool_deadline - time.monotonic()) * 1000.0)
                if remaining_tool_ms <= 0:
                    if self.trace:
                        self.trace.add_agentic_retrieval_event(
                            "rag_agent_tool_timeout",
                            {
                                "scope_hash": safe_hash,
                                "outcome": "timeout",
                                "timeout_owner": "rag_agent_tool",
                                "terminal": True,
                                "admitted_new_work": False,
                            },
                        )
                    return None
                if min(remaining_planner_ms, remaining_turn_ms) <= 0:
                    reason = "agentic_turn_budget_exhausted" if remaining_turn_ms <= 0 else "planner_total_budget_exhausted"
                    self._planner_terminal_ledger[fingerprint] = {
                        "failure_class": "budget",
                        "retryable": False,
                        "fallback_reason": reason,
                    }
                    if remaining_turn_ms <= 0:
                        self._trace_turn_budget_exhausted(operation_id)
                    break
                if cfg.plan_cache_enabled:
                    self._validated_plan_cache[fingerprint] = deepcopy(plan)
                if self.trace:
                    self.trace.add_agentic_retrieval_event(
                        "planner_attempt",
                        {
                            "operation_id": operation_id,
                            "scope_hash": safe_hash,
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "calls_used": self._planner_calls_used,
                            "max_calls": max_calls,
                            "failure_class": None,
                            "retryable": False,
                            "repair": bool(repair_errors),
                            "attempt_timeout_ms": attempt_timeout_ms,
                            "attempt_latency_ms": round(attempt_work_ms, 3),
                            "cleanup_latency_ms": round(attempt_cleanup_ms, 3),
                            "remaining_planner_budget_ms": round(max(0.0, total_budget_ms - self._planner_elapsed_ms), 3),
                            "terminal": False,
                        },
                    )
                return plan

        terminal = self._planner_terminal_ledger.get(fingerprint, {"failure_class": "attempt_ceiling", "retryable": False, "fallback_reason": "planner_attempts_exhausted"})
        self._planner_terminal_ledger[fingerprint] = terminal
        if self.trace:
            self.trace.add_agentic_retrieval_event(
                "planner_terminal",
                {
                    "scope_hash": safe_hash,
                    "cache_status": "hit_terminal",
                    "attempt": self._planner_attempts_by_fingerprint.get(fingerprint, 0),
                    "max_attempts": max_attempts,
                    "calls_used": self._planner_calls_used,
                    "max_calls": max_calls,
                    "terminal": True,
                    "fallback": "baseline_retrieval",
                    **terminal,
                },
            )
            self.trace.add_agentic_retrieval_event(
                "planner_llm_fallback_to_baseline",
                {
                    "scope_hash": safe_hash,
                    "terminal": True,
                    "fallback": "baseline_retrieval",
                    **terminal,
                },
            )
        return None

    def _trace_tool_start(self, name: str, args: dict[str, Any] | None = None, *, retrieval_mode: str | None = None) -> str | None:
        if not self.trace:
            return None
        return self.trace.start_tool(name, args, retrieval_mode=retrieval_mode)

    def _trace_tool_end(self, tool_call_id: str | None, *, success: bool = True, result: Any = None, result_summary: dict[str, Any] | None = None, error: Exception | str | None = None) -> None:
        if self.trace:
            self.trace.end_tool(tool_call_id, success=success, result=result, result_summary=result_summary, error=error)

    def sys_prompt(self) -> str:
        """Return the system instruction the chat model should be initialised with.

        The workflow encoded here mentions ONLY the tools that this
        ``RAGTools`` instance actually registered — optional tools
        (``filter_docs_by_metadata``, ``search_structured_data``,
        ``web_search``) are described only when they are bound, so the
        model is never told to call something it does not have.
        """
        has_meta = bool(self.meta_data_filter)
        has_sql = bool(self.sql_kbs)
        has_web = self.tav is not None
        has_unstructured = bool(self.kb_ids)
        has_embedding = self.embed_mdl is not None
        has_summarize = has_unstructured  # tool gated on kb_ids in __init__

        # Step 2 — document-scope narrowing bullets
        narrow_bullets = [
            "- Call `select_documents` when the question names or strongly "
            "implies a particular document or a small subset by title or topic."
        ]
        if has_meta:
            narrow_bullets.append(
                "- Call `filter_docs_by_metadata` when the question references "
                "structured attributes (year, author, department, product, ...) "
                "that are carried as document metadata."
            )

        # Step 3 — retrieval paragraph, depending on which KB shapes are bound
        summarize_special_case = (
            " SPECIAL CASE — summarisation: if the user EXPLICITLY asked you "
            "to summarise a specific document (phrasings like 'summarise the "
            "security audit', 'give me a summary of doc X', 'tldr the "
            "onboarding guide'), call `summarize_document` with the doc ID "
            "obtained in step 2 INSTEAD OF `search_knowledge_bases`. Use this "
            "tool ONLY for explicit summarisation requests — not for general "
            "Q&A about a document's contents, which still goes through "
            "`search_knowledge_bases`."
            if has_summarize
            else ""
        )

        embedding_retry = (
            " Inspect the chunks `search_knowledge_bases` returns. If they are "
            "not fully relevant — they only hit on incidental keyword overlap, "
            "or the keywords clearly miss the concept the user is after — call "
            "`search_knowledge_bases` ONCE MORE with the SAME `question` and "
            "`keywords` but with `using_embedding=True`. Do this at most one "
            "extra time per turn; if neither keyword nor embedding mode finds "
            "relevant content, the KB likely does not cover this question."
            if has_embedding
            else ""
        )

        if has_unstructured and has_sql:
            retrieval_para = (
                "If the question is naturally an aggregate or filter over tabular "
                "data ('how many ...', 'list the top N by ...', 'sum of ... grouped "
                "by ...'), call `search_structured_data` with the formalized "
                "question — the structured knowledge base has a typed schema and "
                "the tool translates your question into SQL. Otherwise call "
                "`search_knowledge_bases` with the formalized question AND a short "
                "keyword string (3-8 keywords plus 1-2 close synonyms for ambiguous "
                "terms, in the same language as the question). Pass any doc IDs "
                "collected in step 2 as `docid_scope`." + embedding_retry + summarize_special_case
            )
        elif has_sql and not has_unstructured:
            retrieval_para = (
                "Call `search_structured_data` with the formalized question. The "
                "knowledge base has a typed schema (`field_map`), so the tool will "
                "translate your question into SQL and execute it."
            )
        else:
            retrieval_para = (
                "Call `search_knowledge_bases` with the formalized question AND a "
                "short keyword string (3-8 keywords plus 1-2 close synonyms for "
                "ambiguous terms, in the same language as the question). Pass any "
                "doc IDs collected in step 2 as `docid_scope`." + embedding_retry
            )

        steps: list[str] = []
        steps.append(
            "**Formalize the question.** If the latest user message is a "
            "follow-up that depends on earlier turns (pronouns, 'and X?', "
            "'what about ...'), call `formalize_question` to produce a "
            "self-contained question and use that for every subsequent step. "
            "Otherwise use the latest user message as-is."
        )
        steps.append(
            "**Narrow the document scope.** Before retrieving, try to limit "
            "which documents you'll search:\n   "
            + "\n   ".join(narrow_bullets)
            + "\n   Collect the doc IDs these tools return VERBATIM and pass "
            "them to the next step as `docid_scope`. You DO NOT know any doc "
            "IDs on your own — they are 32-character hex strings (e.g. "
            "`41a5271858ca11f1bbb9047c16ec874f`) that only these tools can "
            "produce. Skip this step entirely when the question gives you no "
            "signal to narrow down, and in that case pass `null` (NOT an "
            "invented list) for `docid_scope` in the next step."
        )
        steps.append("**Retrieve evidence from the knowledge bases.** " + retrieval_para)
        if has_web:
            steps.append(
                "**Fall back to web search.** If `search_knowledge_bases` "
                "returned no relevant chunks AND the question is about "
                "generally public information, call `web_search`. Skip this "
                "step whenever the KB retrieval succeeded — prefer "
                "KB-grounded answers."
            )
        steps.append(
            "**Compose the final answer with citations.** Citation rules will "
            "be delivered to you inline with the FIRST retrieval result of "
            "this turn (look for a `# Citation rules` block at the top of the "
            "tool output). Apply those rules VERBATIM. Do NOT invent your own "
            "citation style, and NEVER cite a source you did not actually "
            "retrieve in this turn."
        )

        numbered = "\n\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1))

        return (
            "You are a Retrieval-Augmented-Generation (RAG) agent. Answer the "
            "user's question using ONLY evidence you retrieved through the "
            "tools available to you. Do not invent facts: if the evidence "
            "cannot support a claim, say so plainly instead of guessing.\n\n"
            "# Workflow\n\n"
            "Work through the following steps in order. Skip a step when it "
            "is obviously inapplicable.\n\n"
            f"{numbered}\n\n"
            "# Hard rules\n\n"
            "- DO NOT make anything up. If the retrieved evidence does not "
            "answer the question, reply with an explicit \"I don't have "
            "enough information based on the available sources\" (in the "
            "user's language). If you identify an intermediate entity but "
            "the retrieved chunks lack the requested attribute, issue one "
            "focused `search_knowledge_bases` query for that entity plus the "
            "missing attribute/year/token before refusing.\n"
            "- For numeric, date, rank, birthplace, title-year, population, "
            "or city/state claims, cite a retrieved chunk containing the "
            "exact fact or state that the exact fact was not found. Do not "
            "use uncited world knowledge to complete a missing arithmetic or "
            "multi-hop chain. Before refusing a multi-hop question, say which "
            "hop is missing and whether targeted retrieval was attempted.\n"
            "- DO NOT cite sources that were not returned by your tool calls "
            "in this turn.\n"
            "- DO NOT invent identifiers. Every doc ID you pass to a tool "
            "MUST be a value some other tool returned earlier IN THIS SAME "
            "TURN. If you have no IDs from a prior tool, pass `null` — never "
            "a fabricated 32-character string.\n"
            "- **Answer in the user's language.** The prose of the final "
            "answer MUST be in the SAME language as the user's question. If "
            "the user wrote in Chinese, you answer in Chinese; if Japanese, "
            "Japanese; and so on for every other non-English language. "
            "Answering in English when the user did NOT write in English is "
            "FORBIDDEN — translate retrieved evidence into the user's "
            "language as part of composing the answer. The single exception "
            "is verbatim quoted snippets from the knowledge base that you "
            "cite as evidence: those may stay in the source's original "
            "language so the citation remains faithful. Everything OUTSIDE "
            "those quoted snippets — your prose, your headings, your "
            "summaries, the \"I don't have enough information\" fallback — "
            "must be in the user's language."
        )

    @tool
    async def formalize_question(self, messages: List[str]) -> str:
        """Rewrite the latest user message if it's not suitable for searching into a complete, standalone question
        by resolving pronouns and elliptical references against earlier turns.

        Args:
            messages: the conversation so far, oldest first. Each item should be
                prefixed with the speaker, e.g.
                  ["User: what's the population of Beijing?",
                   "Assistant: About 50 million.",
                   "User: New york?"]

        Returns:
            A single self-contained question, e.g.
              "What's the population of New York?"
            If the latest user message is already standalone, it's returned unchanged.
        """
        tool_call_id = self._trace_tool_start("formalize_question", {"messages": messages})
        if not messages:
            self._trace_tool_end(tool_call_id, result="")
            return ""

        try:
            transcript = "\n".join(messages)
            system = (
                "You rewrite the LAST user message into a single, self-contained question "
                "that can be understood without seeing the prior conversation. "
                "Resolve pronouns, ellipses, and follow-up shortcuts using earlier turns. "
                "Preserve the original language of the last user message. "
                "Output ONLY the rewritten question — no preamble, no quotes, no explanation. "
                "If the last user message is already a complete standalone question, return it unchanged."
            )
            user = (
                f"Conversation:\n{transcript}\n\n"
                "Rewritten standalone question:"
            )

            ans = await self.chat_mdl.async_chat(
                    system=system,
                    history=[{"role": "user", "content": user}],
                    gen_conf={"temperature": 0.1},
                )

            result = ans.strip().strip('"').strip("'")
            self._trace_tool_end(tool_call_id, result=result)
            return result
        except Exception as e:
            self._trace_tool_end(tool_call_id, success=False, error=e)
            raise

    async def _get_cached_metas(self) -> dict:
        """Lazy-load the flattened metadata map for the bound KBs and cache it
        on the instance so repeat tool calls in the same session don't re-hit
        the DB. Returns an empty dict when no KBs are bound.
        """
        cached = getattr(self, "_metas_cache", None)
        if cached is not None:
            return cached
        if not self.kb_ids:
            self._metas_cache = {}
            return self._metas_cache
        self._metas_cache = await thread_pool_exec(
            DocMetadataService.get_flatted_meta_by_kbs, self.kb_ids
        )
        return self._metas_cache or {}

    @tool(timeout=60)
    async def filter_docs_by_metadata(self, question: str) -> List[str]:
        """Narrow the search to a smaller document set using structured metadata.

        If the bound knowledge bases carry document-level metadata (e.g.
        ``author``, ``year``, ``department``, ``product``), this tool asks an
        LLM to translate the question into a metadata filter and runs it
        against the index, returning the matching document IDs.

        Call this BEFORE ``search_knowledge_bases`` whenever the user's
        question references such structured attributes ("documents from 2024",
        "papers by Alice on X"), then pass the returned IDs to
        ``search_knowledge_bases`` via its ``attachments`` parameter so
        retrieval is restricted to those docs.

        Skip this tool when the question is pure free-text with no obvious
        metadata predicate — running it then wastes an LLM call and may
        produce an over-restrictive filter.

        :param question: the self-contained natural-language question.

        :returns: list of document IDs matching the metadata filter. An
            empty list means one of: no metadata is defined on the KBs, no
            filter could be generated from the question, no docs matched,
            or the filter couldn't be pushed down to the index. In any of
            those cases the caller should fall back to unfiltered retrieval
            (i.e. call ``search_knowledge_bases`` without ``attachments``).
        """
        tool_call_id = self._trace_tool_start("filter_docs_by_metadata", {"question": question}, retrieval_mode="metadata")
        try:
            if not self.kb_ids:
                self._trace_tool_end(tool_call_id, result=[])
                return []

            metas = await self._get_cached_metas()
            if not metas:
                self._trace_tool_end(tool_call_id, result=[])
                return []

            filters = await gen_meta_filter(self.chat_mdl, metas, question)
            logging.debug(f"Metadata filter(auto) generated: {filters}")

            conditions = filters.get("conditions") or []
            if not conditions:
                self._trace_tool_end(tool_call_id, result=[], result_summary={"docid_count": 0, "metadata_filters": filters})
                return []

            logic = filters.get("logic", "and")
            try:
                doc_ids = await thread_pool_exec(
                    DocMetadataService.filter_doc_ids_by_meta_pushdown,
                    self.kb_ids,
                    conditions,
                    logic,
                )
            except Exception as e:
                logging.error(f"Metadata filter push down errored: {e}")
                self._trace_tool_end(tool_call_id, success=False, result=[], result_summary={"docid_count": 0, "metadata_filters": filters}, error=e)
                return []
            logging.debug(f"Doc ids filtered by metadata: {doc_ids}")
            # ``filter_doc_ids_by_meta_pushdown`` returns None when push-down isn't
            # viable; treat that as "no filter applied" and surface as empty list
            # so the caller falls back to unfiltered retrieval.
            result = doc_ids or []
            self._trace_tool_end(tool_call_id, result=result, result_summary={"docid_count": len(result), "metadata_filters": filters})
            return result
        except Exception as e:
            self._trace_tool_end(tool_call_id, success=False, error=e)
            raise

    def _collect_doc_titles(self, max_docs:int = 512) -> list[tuple[str, str]]:
        """Return ``[(doc_id, name)]`` for every document in the bound KBs.

        Lightweight sync DB read — meant to be wrapped in ``thread_pool_exec``
        by the tool entry-point so the event loop isn't blocked.
        """
        result: list[tuple[str, str]] = []
        for kb_id in self.kb_ids:
            for doc in DocumentService.query(kb_id=kb_id):
                result.append((doc.id, doc.name))
                if len(result) >= max_docs:
                    return None
        return result

    async def _try_agentic_refinement(
        self,
        *,
        question: str,
        agentic_result: Any,
        docid_scope: list[str] | None,
        top_n: int,
        similarity_threshold: float,
        tool_deadline: float,
    ) -> Any | None:
        """Run Phase 6 only on a successful Phase 5 bounded retrieval result."""
        cfg = self.agentic_refinement_config
        if not cfg.enabled or cfg.mode == "off":
            return None

        plan = getattr(agentic_result, "plan", None)
        plan_id = getattr(plan, "plan_id", None)
        if plan is None or getattr(agentic_result, "fallback_to_baseline", True):
            if self.trace:
                self.trace.add_agentic_refinement_event(
                    "refinement_skip",
                    mode=cfg.mode,
                    plan_id=plan_id,
                    stop_reason="phase5_unavailable",
                )
            return None
        if cfg.mode == "active" and self.agentic_retrieval_config.mode != "active":
            if self.trace:
                self.trace.add_agentic_refinement_event(
                    "refinement_skip",
                    mode=cfg.mode,
                    plan_id=plan_id,
                    stop_reason="phase5_not_active",
                )
            return None

        candidate_kbinfos = deepcopy(agentic_result.kbinfos)
        selected_kbinfos = self._selected_phase5_kbinfos(candidate_kbinfos, question=question)
        remaining_turn_ms = self._remaining_agentic_budget_ms()
        remaining_tool_ms = max(0.0, (tool_deadline - time.monotonic()) * 1000.0)
        if min(remaining_turn_ms, remaining_tool_ms) <= 0:
            self._trace_turn_budget_exhausted()
            return None
        phase6_cfg = replace(cfg, latency_budget_ms=max(1, int(min(cfg.latency_budget_ms, remaining_turn_ms, remaining_tool_ms))))
        phase6_started = time.monotonic()
        phase6_cleanup_ms = 0.0
        phase6_timeout_owner = None

        def account_phase6_cleanup(cleanup_ms: float, timeout_owner: str) -> None:
            nonlocal phase6_cleanup_ms, phase6_timeout_owner
            phase6_cleanup_ms += cleanup_ms
            phase6_timeout_owner = timeout_owner

        try:
            result = await run_refinement_loop(
                question=question,
                plan=plan,
                kbinfos=selected_kbinfos,
                candidate_kbinfos=candidate_kbinfos,
                retriever=settings.retriever,
                chat_mdl=self.chat_mdl,
                embd_mdl=self.embed_mdl,
                tenant_ids=self.tenant_ids,
                kb_ids=self.kb_ids,
                doc_ids=docid_scope,
                similarity_threshold=similarity_threshold,
                vector_similarity_weight=0.7 if self.embed_mdl else 0.0,
                top_k=max(1, top_n),
                rank_feature=label_question(question, self.kbs),
                refinement_cfg=phase6_cfg,
                retrieval_cfg=self.agentic_retrieval_config,
                context_builder_config=self.context_builder_config,
                rag_trace=self.trace,
                metadata_filters=self.meta_data_filter,
                rerank_mdl=None,
                apply_children=True,
                apply_toc=False,
                cleanup_callback=account_phase6_cleanup,
            )
            if phase6_timeout_owner is None:
                phase6_timeout_owner = result.diagnostics.get("timeout_owner")
        except asyncio.CancelledError:
            raise
        except AgenticRefinementError as exc:
            if self.trace:
                self.trace.add_agentic_refinement_event(
                    "refinement_fallback_to_previous_context",
                    mode=cfg.mode,
                    plan_id=plan_id,
                    fallback_reason=exc.reason,
                    failure_class=exc.reason,
                )
            return None
        except Exception:
            if self.trace:
                self.trace.add_agentic_refinement_event(
                    "refinement_fallback_to_previous_context",
                    mode=cfg.mode,
                    plan_id=plan_id,
                    fallback_reason="refinement_error",
                    failure_class="unknown",
                )
            return None
        finally:
            elapsed_ms = (time.monotonic() - phase6_started) * 1000.0
            if asyncio.current_task() in self._rag_agent_tool_expired_tasks:
                phase6_timeout_owner = "rag_agent_tool"
            self._account_agentic_stage("phase6", elapsed_ms, cleanup_ms=phase6_cleanup_ms, timeout_owner=phase6_timeout_owner)

        if tool_deadline <= time.monotonic():
            if self.trace:
                self.trace.add_agentic_refinement_event(
                    "refinement_fallback_to_previous_context",
                    mode=cfg.mode,
                    plan_id=plan_id,
                    fallback_reason="rag_agent_tool_timeout",
                    failure_class="timeout",
                    timeout_owner="rag_agent_tool",
                )
            return None
        if self._remaining_agentic_budget_ms() <= 0:
            self._trace_turn_budget_exhausted()
            return None
        return result

    def _with_citation_guidelines(self, output: Any) -> Any:
        """Stamp the citation rules onto the FIRST retrieval-tool output of the
        turn, then short-circuit on subsequent calls.

        The citation policy is static and applies to every final answer, but
        the model only needs to see it once — and only when retrieval has
        actually happened (otherwise there's nothing to cite). Injecting
        inline with the tool result keeps the system prompt small and avoids
        a separate tool round-trip that the model would routinely skip.

        Accepts either ``str`` or ``list[str]`` (the two shapes the retrieval
        tools currently return) and prepends a ``# Citation rules`` block.
        """
        if self._citations_injected:
            return output
        self._citations_injected = True
        rules = citation_prompt(self.user_defined_prompts).strip()
        header = (
            "# Citation rules\n"
            "Apply the following rules VERBATIM to your final answer. "
            "They are stated here in full and apply for the rest of this "
            "turn.\n\n"
            f"{rules}\n\n"
            "----\n\n"
        )
        if isinstance(output, list):
            return [header] + output
        return header + str(output)

    def _filter_known_doc_ids(self, candidate_ids: list[str]) -> set[str]:
        """Return the subset of ``candidate_ids`` that actually belong to a
        bound unstructured KB.

        Single targeted ``WHERE id IN (...) AND kb_id IN (...)`` query —
        size bounded by ``len(candidate_ids)`` (which is at most the
        LLM-supplied ``docid_scope``), not by the KB size. Used to catch
        hallucinated 32-char hex IDs before they reach the retriever, which
        would silently return zero chunks for unknown IDs.

        Sync DB call — wrap in ``thread_pool_exec`` at the call site so the
        event loop isn't blocked.
        """
        if not candidate_ids or not self.kb_ids:
            return set()
        rows = Document.select(Document.id).where(
            (Document.id.in_(list(candidate_ids)))
            & (Document.kb_id.in_(self.kb_ids))
        )
        return {row.id for row in rows}

    async def _try_agentic_bounded_retrieval(
        self,
        *,
        question: str,
        docid_scope: list[str] | None,
        top_n: int,
        similarity_threshold: float,
        tool_deadline: float,
    ) -> Any | None:
        """Run Phase 5's LLM helper planner inside the rag_agent retrieval tool.

        This is intentionally a bounded helper around ``search_knowledge_bases``:
        diagnostic mode records only the trigger, shadow mode runs the plan but
        leaves the tool output on the baseline retrieval path, and active mode
        returns bounded retrieval only after LLM planning, validation, and lane
        execution all succeed.
        """
        cfg = self.agentic_retrieval_config
        if not cfg.enabled or cfg.mode == "off" or not self.embed_mdl or not self.kb_ids:
            return None

        scope = type("RAGAgentPlannerScope", (), {"kb_ids": self.kb_ids})()
        trigger = should_plan(
            question,
            history=[],
            dialog=scope,
            attachments=docid_scope or [],
            cfg=cfg,
            cheap_features={"kb_scope_available": bool(self.kb_ids)},
            diagnostics=None,
        )
        trigger_payload = {
            **trigger.to_trace_dict(),
            "source": "rag_agent.search_knowledge_bases",
            "tool_top_n": top_n,
        }
        if self.trace:
            self.trace.add_agentic_retrieval_event("trigger", trigger_payload)
        if not trigger.should_plan:
            if self.trace:
                self.trace.add_agentic_retrieval_event("bypass", trigger_payload)
            return None
        if cfg.mode == "diagnostic":
            if self.trace:
                self.trace.add_agentic_retrieval_event("diagnostic_trigger_only", trigger_payload)
            return None

        planner_input = build_planner_input(
            question,
            history=[],
            dialog=scope,
            attachments=docid_scope or [],
            cfg=cfg,
            trigger=trigger,
            tenant_ids=self.tenant_ids,
            metadata_filters=self.meta_data_filter,
        )
        plan = await self._coordinated_plan(planner_input, tool_deadline=tool_deadline)
        if plan is None:
            return None
        remaining_turn_ms = self._remaining_agentic_budget_ms()
        remaining_tool_ms = max(0.0, (tool_deadline - time.monotonic()) * 1000.0)
        if min(remaining_turn_ms, remaining_tool_ms) <= 0:
            self._trace_turn_budget_exhausted()
            return None
        phase5_cfg = replace(
            cfg,
            latency_budget_ms=max(1, int(min(cfg.latency_budget_ms, remaining_turn_ms, remaining_tool_ms))),
            retrieval_timeout_ms=max(1, int(min(cfg.retrieval_timeout_ms, remaining_turn_ms, remaining_tool_ms))),
        )
        phase5_started = time.monotonic()
        phase5_cleanup_ms = 0.0
        phase5_timeout_owner = None

        def account_phase5_cleanup(cleanup_ms: float, timeout_owner: str) -> None:
            nonlocal phase5_cleanup_ms, phase5_timeout_owner
            phase5_cleanup_ms += cleanup_ms
            phase5_timeout_owner = timeout_owner

        try:
            result = await execute_bounded_plan(
                plan,
                retriever=settings.retriever,
                embd_mdl=self.embed_mdl,
                tenant_ids=self.tenant_ids,
                kb_ids=self.kb_ids,
                doc_ids=docid_scope,
                similarity_threshold=similarity_threshold,
                vector_similarity_weight=0.7 if self.embed_mdl else 0.0,
                top_k=max(1, top_n),
                rank_feature=label_question(question, self.kbs),
                rerank_mdl=None,
                cfg=phase5_cfg,
                rag_trace=self.trace,
                metadata_filters=self.meta_data_filter,
                apply_children=True,
                apply_toc=False,
                chat_mdl=self.chat_mdl,
                cleanup_callback=account_phase5_cleanup,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            if self.trace:
                self.trace.add_agentic_retrieval_event(
                    "phase5_fallback_to_baseline",
                    {"reason": "bounded_retrieval_error", "mode": cfg.mode, "source": "rag_agent.search_knowledge_bases", "failure_class": "unknown"},
                )
            return None
        finally:
            elapsed_ms = (time.monotonic() - phase5_started) * 1000.0
            if asyncio.current_task() in self._rag_agent_tool_expired_tasks:
                phase5_timeout_owner = "rag_agent_tool"
            self._account_agentic_stage("phase5", elapsed_ms, cleanup_ms=phase5_cleanup_ms, timeout_owner=phase5_timeout_owner)

        if self.trace:
            self.trace.add_agentic_retrieval_event(
                "result",
                {
                    "plan_id": plan.plan_id,
                    "mode": cfg.mode,
                    "source": "rag_agent.search_knowledge_bases",
                    "fallback_to_baseline": result.fallback_to_baseline,
                    "fallback_reason": result.fallback_reason,
                    "diagnostics": result.diagnostics,
                },
            )
        if result.fallback_to_baseline and self.trace:
            self.trace.add_agentic_retrieval_event(
                "planner_llm_fallback_to_baseline",
                {
                    "reason": result.fallback_reason or "bounded_retrieval_failed",
                    "mode": cfg.mode,
                    "source": "rag_agent.search_knowledge_bases",
                },
            )
        return result

    @tool(timeout=60)
    async def select_documents(self, question: str, max_docs:int=512) -> List[str]:
        """Ask an LLM to pick the document IDs whose titles look relevant to the question.

        Every document in the bound knowledge bases is listed to the LLM in
        the format ``docID: <id>, title: <title>`` and the LLM returns a JSON
        array of the IDs it considers relevant.

        Use this BEFORE ``search_knowledge_bases`` when the user's question
        names a specific document or refers to a small subset by topic
        ("summarize the security audit", "what does the 2024 onboarding guide
        say about X"). Pass the returned IDs to ``search_knowledge_bases`` via
        its ``docid_scope`` parameter so retrieval is restricted to those
        documents.

        Skip this tool for broad free-form questions — running it then wastes
        an LLM round-trip on a list the model would mostly discard.

        :param question: the self-contained natural-language question.

        :returns: list of document IDs the LLM judged relevant, or an empty
            list when no KBs are bound, no documents exist, or none look
            relevant title. IDs returned by the LLM that are not in the catalogue
            are filtered out defensively.
        """
        tool_call_id = self._trace_tool_start("select_documents", {"question": question, "max_docs": max_docs}, retrieval_mode="doc_select")
        try:
            if not self.kb_ids:
                self._trace_tool_end(tool_call_id, result=[])
                return []

            docs = await thread_pool_exec(self._collect_doc_titles)
            if docs is None:
                result = "Too much documents for LLM to judge."
                self._trace_tool_end(tool_call_id, result=result, result_summary={"docid_count": 0, "catalogue_truncated": True})
                return result
            if not docs:
                self._trace_tool_end(tool_call_id, result=[])
                return []

            catalogue = "\n".join(f"docID: {doc_id}, title: {title}" for doc_id, title in docs)
            system = (
                "You filter a document catalogue to find which documents are relevant "
                "to a user's question. Use ONLY the titles in the catalogue — do not "
                "invent docIDs. "
                "Output ONLY a JSON array of the docIDs you consider relevant, e.g. "
                '["abc123", "def456"]. If no document is clearly relevant, output []. '
                "No explanations, no Markdown, no code fences, no prose around the array."
            )
            user = (
                f"Question:\n{question}\n\n"
                f"Documents:\n{catalogue}\n\n"
                "Relevant docIDs (JSON array):"
            )

            ans = await self.chat_mdl.async_chat(
                system=system,
                history=[{"role": "user", "content": user}],
                gen_conf={"temperature": 0.1},
            )
            if isinstance(ans, tuple):
                ans = ans[0]

            # Strip <think> reasoning prefixes and ```json fences before parsing.
            cleaned = re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)
            cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", cleaned).strip()
            try:
                ids = json_repair.loads(cleaned)
            except Exception as e:
                logging.warning(f"select_documents could not parse LLM output: {e!r} raw={ans[:200]!r}")
                self._trace_tool_end(tool_call_id, success=False, result=[], result_summary={"docid_count": 0, "catalogue_count": len(docs)}, error=e)
                return []
            if not isinstance(ids, list):
                self._trace_tool_end(tool_call_id, result=[], result_summary={"docid_count": 0, "catalogue_count": len(docs)})
                return []

            # Defensive: drop anything the LLM hallucinated outside our catalogue.
            known = {doc_id for doc_id, _ in docs}
            res = [doc_id for doc_id in ids if isinstance(doc_id, str) and doc_id in known]
            if not res:
                result = "Fail to pick the document IDs. Try other methods."
                self._trace_tool_end(tool_call_id, result=result, result_summary={"docid_count": 0, "catalogue_count": len(docs), "dropped_doc_ids": len(ids)})
                return result
            self._trace_tool_end(tool_call_id, result=res, result_summary={"docid_count": len(res), "catalogue_count": len(docs), "dropped_doc_ids": len(ids) - len(res)})
            return res
        except Exception as e:
            self._trace_tool_end(tool_call_id, success=False, error=e)
            raise

    @tool(timeout=120)
    async def search_knowledge_bases(
        self,
        question: str,
        keywords: str,
        top_n: int = 6,
        similarity_threshold: float = 0.2,
        docid_scope: List[str] | None = None,
        using_embedding : bool = False
    ) -> dict[str, Any]:
        """Search the user's knowledge bases for chunks relevant to a question.

        You (the calling LLM) must supply BOTH the natural-language question
        and the keyword string used for retrieval — the tool does NOT run a
        second LLM pass to extract keywords.

        Two retrieval modes, controlled by ``using_embedding``:

        - **Keyword (default)**: pure sparse / BM25-style search driven by
          the ``keywords`` argument. Fast, cheap, exact-term match. Result
          quality is bounded by the keywords you provide.
        - **Embedding**: dense semantic retrieval over the full question
          (the ``keywords`` argument is ignored in this mode). Better at
          catching paraphrases and concepts the keywords miss, at the cost
          of an extra embedding pass.

        Recommended call pattern: call ONCE with ``using_embedding=False``.
        Inspect the returned chunks. If they are not fully relevant — e.g.
        they only hit on incidental keyword overlap, or the keywords clearly
        miss the concept the user is after — call AGAIN with
        ``using_embedding=True`` (same question, same keywords). Do NOT
        retry more than twice; if neither mode finds relevant content, the
        KB likely does not cover this question and you should consider
        ``web_search`` (when available) or admit you cannot answer.

        Prefer this tool for grounded answers; only fall back to
        ``web_search`` when both retrieval modes return nothing useful.

        :param question: the self-contained natural-language question (run
            ``formalize_question`` first if the latest user message is a
            follow-up that depends on earlier turns).
        :param keywords: 3-8 of the most important content terms from the
            question, plus 1-2 close synonyms or alternative phrasings for
            any ambiguous or polysemous term. Single words or short noun
            phrases, space-separated. The language MUST match the language of
            ``question``. If you cannot produce better keywords, repeat the
            question verbatim. Ignored when ``using_embedding=True``.
        :param top_n: maximum number of chunks to return (default 6).
        :param similarity_threshold: minimum similarity score for a chunk
            to be returned (default 0.2).
        :param docid_scope: OPTIONAL list of document IDs. Each ID is a
            32-character lowercase hex string such as
            ``41a5271858ca11f1bbb9047c16ec874f``. You DO NOT know any doc IDs
            on your own and you MUST NOT invent, guess, modify, or
            reconstruct one — not even if a 32-character string in your
            context happens to look like a doc ID. The ONLY acceptable
            sources for values here are doc IDs returned VERBATIM from a
            previous call to ``select_documents`` or
            ``filter_docs_by_metadata`` IN THIS SAME TURN. If neither tool
            has returned any IDs yet, pass null (the default) — that
            searches across all documents in the bound knowledge bases.
            Passing an invented ID will silently return zero chunks.
        :param using_embedding: set to ``True`` to switch from keyword
            search to dense embedding search. Default ``False``. Use this
            ONLY on a follow-up call when the previous keyword-only call
            returned chunks that were not fully relevant — not on the first
            attempt. Has no effect when this agent was constructed without
            an embedding model.

        :returns: a ``SearchResult``-shaped dict with the matched chunks,
            ``doc_aggs``, and the keywords actually used for retrieval. An
            empty result is returned when no knowledge bases are bound.
        """
        mode = "embedding" if using_embedding else "keyword"
        tool_call_id = self._trace_tool_start(
            "search_knowledge_bases",
            {
                "question": question,
                "keywords": keywords,
                "top_n": top_n,
                "similarity_threshold": similarity_threshold,
                "docid_scope": docid_scope,
                "using_embedding": using_embedding,
            },
            retrieval_mode=mode,
        )
        tool_started = time.monotonic()
        timeout_ceiling_s = max(0.001, float(self._rag_agent_tool_timeout_s))
        cleanup_margin_s = _rag_agent_cleanup_margin_s(timeout_ceiling_s)
        work_timeout_s = max(0.001, timeout_ceiling_s - cleanup_margin_s)
        tool_deadline = tool_started + work_timeout_s
        admission_guard_s = min(0.01, work_timeout_s * 0.1)
        work_task = asyncio.current_task()
        assert work_task is not None

        async def expire_local_work() -> None:
            await asyncio.sleep(work_timeout_s)
            self._rag_agent_tool_expired_tasks.add(work_task)
            work_task.cancel()

        def raise_if_local_work_deadline_exhausted() -> None:
            if tool_deadline - time.monotonic() <= admission_guard_s:
                self._rag_agent_tool_expired_tasks.add(work_task)
                raise asyncio.CancelledError

        watchdog_task = asyncio.create_task(expire_local_work())
        search_fingerprint = None
        lock_acquired = False
        planner_terminal_keys_before = None
        validated_plan_keys_before = None
        completed_search_keys_before = None
        kbinfos_before = None
        last_context_query_before = None
        citations_injected_before = None
        try:
            await self._agentic_search_lock.acquire()
            lock_acquired = True
            planner_terminal_keys_before = set(self._planner_terminal_ledger)
            validated_plan_keys_before = set(self._validated_plan_cache)
            completed_search_keys_before = set(self._completed_search_ledger)
            kbinfos_before = deepcopy(self.kbinfos)
            last_context_query_before = self._last_context_query
            citations_injected_before = self._citations_injected
            if not self.kb_ids:
                result = {"total": 0, "chunks": [], "doc_aggs": {}}
                self._trace_tool_end(tool_call_id, result=result)
                return result

            # Validate docid_scope against the actual catalogue — models often
            # hallucinate 32-char hex strings, and the retriever silently returns
            # zero chunks for unknown IDs (which then looks like "KB has nothing"
            # to the calling LLM, when really the filter was bogus).
            if docid_scope:
                candidates = [d for d in docid_scope if isinstance(d, str)]
                known = await thread_pool_exec(self._filter_known_doc_ids, candidates)
                valid = [d for d in candidates if d in known]
                if len(valid) != len(docid_scope):
                    dropped = [d for d in docid_scope if d not in known]
                    logging.warning(
                        f"search_knowledge_bases: dropping {len(dropped)}/{len(docid_scope)} "
                        f"unknown doc IDs from docid_scope (samples: {dropped[:3]})"
                    )
                if valid:
                    docid_scope = valid
                else:
                    # Every supplied ID was bogus. Falling back to unfiltered
                    # retrieval is safer than returning zero chunks and forcing
                    # the LLM into a retry loop.
                    logging.warning(
                        "search_knowledge_bases: every supplied doc ID was unknown; "
                        "falling back to unfiltered retrieval"
                    )
                    docid_scope = None

            fingerprint_scope = type("RAGAgentPlannerScope", (), {"kb_ids": self.kb_ids})()
            fingerprint_trigger = should_plan(
                question,
                history=[],
                dialog=fingerprint_scope,
                attachments=docid_scope or [],
                cfg=self.agentic_retrieval_config,
                cheap_features={"kb_scope_available": bool(self.kb_ids)},
                diagnostics=None,
            )
            fingerprint_input = build_planner_input(
                question,
                history=[],
                dialog=fingerprint_scope,
                attachments=docid_scope or [],
                cfg=self.agentic_retrieval_config,
                trigger=fingerprint_trigger,
                tenant_ids=self.tenant_ids,
                metadata_filters=self.meta_data_filter,
            )
            search_fingerprint = self._search_fingerprint(
                planner_input=fingerprint_input,
                question=question,
                keywords=keywords,
                using_embedding=using_embedding,
                top_n=top_n,
                similarity_threshold=similarity_threshold,
                docid_scope=docid_scope,
            )
            completed = self._completed_search_ledger.get(search_fingerprint)
            if completed is not None:
                raise_if_local_work_deadline_exhausted()
                if self.trace:
                    self.trace.add_agentic_retrieval_event(
                        "exact_result_cache",
                        {
                            "operation_hash": search_fingerprint[:16],
                            "cache_status": "hit_completed",
                            "plan_id": completed.get("plan_id"),
                            "terminal": True,
                        },
                    )
                result = self._with_citation_guidelines(deepcopy(completed["output"]))
                raise_if_local_work_deadline_exhausted()
                self._trace_tool_end(
                    tool_call_id,
                    result=result,
                    result_summary={
                        "cache_status": "hit_completed",
                        "plan_id": completed.get("plan_id"),
                        "replayed": True,
                    },
                )
                raise_if_local_work_deadline_exhausted()
                return result
            if self.trace:
                self.trace.add_agentic_retrieval_event(
                    "exact_result_cache",
                    {"operation_hash": search_fingerprint[:16], "cache_status": "miss", "terminal": False},
                )

            agentic_result = await self._try_agentic_bounded_retrieval(
                question=question,
                docid_scope=docid_scope,
                top_n=top_n,
                similarity_threshold=similarity_threshold,
                tool_deadline=tool_deadline,
            )
            raise_if_local_work_deadline_exhausted()
            use_agentic_result = (
                agentic_result is not None
                and self.agentic_retrieval_config.mode == "active"
                and not agentic_result.fallback_to_baseline
            )
            refinement_result = None
            if self.agentic_refinement_config.enabled and self.agentic_refinement_config.mode != "off":
                refinement_result = await self._try_agentic_refinement(
                    question=question,
                    agentic_result=agentic_result,
                    docid_scope=docid_scope,
                    top_n=top_n,
                    similarity_threshold=similarity_threshold,
                    tool_deadline=tool_deadline,
                )
                raise_if_local_work_deadline_exhausted()
            use_refinement_result = (
                use_agentic_result
                and refinement_result is not None
                and self.agentic_refinement_config.mode == "active"
                and refinement_result.changed
                and not refinement_result.fallback_to_previous_context
                and refinement_result.selected_new_evidence_count >= self.agentic_refinement_config.min_new_evidence
            )

            search_terms = keywords.strip() if keywords else ""
            if not search_terms or using_embedding:
                search_terms = question

            variant_name = "embedding_retry" if using_embedding else "keyword_first"
            variant = make_retrieval_variant(variant_name)
            variant_knobs = resolve_variant_knobs(
                variant,
                default_vector_similarity_weight=0.7 if self.embed_mdl else 0.0,
                default_similarity_threshold=similarity_threshold,
                embedding_available=self.embed_mdl is not None,
            )
            fusion_config = FusionConfig.from_env()
            vector_weight = variant_knobs["vector_similarity_weight"]
            start_idx = len(self.kbinfos.get("chunks", []))
            retrieval_call_id = None
            embd_mdl = None
            if use_refinement_result:
                kbinfos = deepcopy(refinement_result.kbinfos)
                allowed_refinement_keys = {
                    _refinement_storage_identity(chunk)
                    for chunk in agentic_result.kbinfos.get("chunks", [])
                    if isinstance(chunk, dict)
                }
                allowed_refinement_keys.update(
                    _refinement_storage_identity(chunk)
                    for chunk in refinement_result.accepted_chunks
                    if isinstance(chunk, dict)
                )
                kbinfos["chunks"] = [
                    chunk
                    for chunk in kbinfos.get("chunks", [])
                    if isinstance(chunk, dict) and _refinement_storage_identity(chunk) in allowed_refinement_keys
                ]
                selected_doc_ids = {chunk.get("doc_id") for chunk in kbinfos["chunks"] if chunk.get("doc_id")}
                selected_doc_names = {
                    chunk.get("docnm_kwd") or chunk.get("document_name") or chunk.get("title")
                    for chunk in kbinfos["chunks"]
                    if chunk.get("docnm_kwd") or chunk.get("document_name") or chunk.get("title")
                }
                if isinstance(kbinfos.get("doc_aggs"), list):
                    kbinfos["doc_aggs"] = [
                        doc_agg
                        for doc_agg in kbinfos["doc_aggs"]
                        if isinstance(doc_agg, dict)
                        and (
                            doc_agg.get("doc_id") in selected_doc_ids
                            or (doc_agg.get("doc_name") or doc_agg.get("docnm_kwd") or doc_agg.get("document_name")) in selected_doc_names
                        )
                    ]
                kbinfos["total"] = len(kbinfos["chunks"])
            elif use_agentic_result:
                assert agentic_result is not None
                kbinfos = agentic_result.kbinfos
            else:
                raise_if_local_work_deadline_exhausted()
                embd_mdl = self.embed_mdl if (variant_knobs["using_embedding"] or fusion_config.enabled) else None
                retrieval_started_ms = _now_ms()
                kbinfos = await settings.retriever.retrieval(
                    search_terms,
                    embd_mdl,
                    self.tenant_ids,
                    self.kb_ids,
                    1,
                    top_n,
                    similarity_threshold,
                    vector_similarity_weight=vector_weight,
                    aggs=True,
                    doc_ids=docid_scope,
                    rank_feature=label_question(question, self.kbs),
                )
                raise_if_local_work_deadline_exhausted()
                kbinfos["chunks"] = settings.retriever.retrieval_by_children(kbinfos["chunks"], self.tenant_ids)
                if self.trace:
                    retrieval_call_id = self.trace.add_retrieval_call(
                        mode=mode,
                        query_text=search_terms,
                        keywords=keywords,
                        docid_scope=docid_scope,
                        metadata_filters=self.meta_data_filter,
                        chunks=kbinfos.get("chunks", []),
                        doc_aggs=kbinfos.get("doc_aggs", []),
                        tool_call_id=tool_call_id,
                        used_embedding=bool(embd_mdl),
                        used_web=False,
                        started_ms=retrieval_started_ms,
                        retrieval_variant=variant_knobs["retrieval_variant"],
                        similarity_threshold=variant_knobs["similarity_threshold"],
                        vector_similarity_weight=vector_weight,
                        top_k=top_n,
                        page_size=top_n,
                        doc_scope_enabled=bool(docid_scope),
                        metadata_filter_enabled=bool(self.meta_data_filter),
                        diagnostics=self._retrieval_diagnostics(
                            question,
                            keywords,
                            docid_scope,
                            using_embedding,
                            kbinfos.get("chunks", []),
                            kbinfos,
                            fusion_config,
                            bool(embd_mdl),
                        ),
                    )
                    self.trace.add_evidence_from_chunks(
                        kbinfos.get("chunks", []),
                        source_type="kb",
                        retrieval_call_id=retrieval_call_id,
                        tool_call_id=tool_call_id,
                        start_citation_index=start_idx,
                    )

            appended_refinement_evidence_count = 0
            raise_if_local_work_deadline_exhausted()
            if kbinfos:
                if self.context_builder_config.enabled:
                    mark_chunks_source_type(kbinfos.get("chunks", []), "kb")
                chunks_to_append = list(kbinfos.get("chunks", []))
                if use_refinement_result:
                    existing_keys = {
                        _refinement_storage_identity(chunk)
                        for chunk in self.kbinfos.get("chunks", [])
                        if isinstance(chunk, dict)
                    }
                    deduped_chunks = []
                    accepted_keys = {
                        _refinement_storage_identity(chunk)
                        for chunk in refinement_result.accepted_chunks
                        if isinstance(chunk, dict)
                    }
                    for chunk in chunks_to_append:
                        identity = _refinement_storage_identity(chunk)
                        if identity in existing_keys:
                            raw_agentic = chunk.get("_ragflow_agentic_retrieval")
                            lineage = raw_agentic if isinstance(raw_agentic, dict) else {}
                            followup_id = chunk.get("followup_id") or lineage.get("followup_id")
                            if followup_id and self.trace:
                                self.trace.add_agentic_refinement_event(
                                    "refinement_evidence_rejected",
                                    mode=self.agentic_refinement_config.mode,
                                    plan_id=chunk.get("plan_id") or lineage.get("plan_id"),
                                    iteration_id=chunk.get("iteration_id") or lineage.get("iteration_id"),
                                    followup_id=followup_id,
                                    facet_id=chunk.get("facet_id") or lineage.get("facet_id"),
                                    evidence_id=chunk.get("chunk_id") or chunk.get("id"),
                                    rejection_reason="duplicate_accumulated_evidence",
                                )
                            continue
                        existing_keys.add(identity)
                        deduped_chunks.append(chunk)
                        if identity in accepted_keys:
                            appended_refinement_evidence_count += 1
                    chunks_to_append = deduped_chunks
                self.kbinfos["chunks"].extend(chunks_to_append)
                self.kbinfos["doc_aggs"].extend(kbinfos.get("doc_aggs", []))
            self._last_context_query = question
            context_stage = "rag_agent.search_knowledge_bases"
            if use_refinement_result:
                accepted_iterations = [
                    index
                    for index, iteration in enumerate(refinement_result.iterations, start=1)
                    if iteration.accepted_new_evidence_count > 0
                ]
                if accepted_iterations:
                    context_stage = f"refinement.iteration.{accepted_iterations[-1]}"
            prompt_kbinfos = self.context_kbinfos(stage=context_stage, start_citation_index=self._prompt_start_idx(start_idx), query=question)
            raw_result = kb_prompt(prompt_kbinfos, self.chat_mdl.max_length, self._prompt_start_idx(start_idx))
            result = self._with_citation_guidelines(raw_result)
            raise_if_local_work_deadline_exhausted()
            if use_refinement_result:
                retrieval_mode = "agentic_refined"
                retrieval_variant = "llm_refined_plan"
                effective_mode = "agentic_refined"
            elif use_agentic_result:
                retrieval_mode = "agentic"
                retrieval_variant = "llm_bounded_plan"
                effective_mode = "agentic_bounded"
            else:
                retrieval_mode = mode
                retrieval_variant = variant_knobs["retrieval_variant"]
                fusion_enabled = bool((kbinfos.get("diagnostics", {}).get("fusion") or {}).get("enabled"))
                effective_mode = "fusion" if fusion_enabled else mode
            result_summary = {
                "retrieval_call_id": retrieval_call_id,
                "chunk_count": len(kbinfos.get("chunks", [])),
                "doc_agg_count": len(kbinfos.get("doc_aggs", [])),
                "retrieval_mode": retrieval_mode,
                "retrieval_variant": retrieval_variant,
                "similarity_threshold": variant_knobs["similarity_threshold"],
                "vector_similarity_weight": vector_weight,
                "doc_scope_enabled": bool(docid_scope),
                "metadata_filter_enabled": bool(self.meta_data_filter),
                "embedding_retry_used": bool(embd_mdl),
                "effective_mode": effective_mode,
                "refinement_selected_new_evidence_count": refinement_result.selected_new_evidence_count if use_refinement_result else 0,
                "refinement_appended_new_evidence_count": appended_refinement_evidence_count,
            }
            if search_fingerprint is not None:
                plan = getattr(agentic_result, "plan", None)
                phase5_fallback = bool(getattr(agentic_result, "fallback_to_baseline", False)) if agentic_result is not None else None
                phase6_fallback = bool(getattr(refinement_result, "fallback_to_previous_context", False)) if refinement_result is not None else None
                if agentic_result is None:
                    phase5_outcome = "not_run"
                elif phase5_fallback:
                    phase5_outcome = "fallback"
                else:
                    phase5_outcome = "completed"
                if refinement_result is None:
                    phase6_outcome = "not_run"
                elif phase6_fallback:
                    phase6_outcome = "fallback"
                else:
                    phase6_outcome = "completed"
                current_tool_elapsed_ms = (time.monotonic() - tool_started) * 1000.0
                completed_record = {
                    "output": deepcopy(raw_result),
                    "kbinfos": deepcopy(kbinfos),
                    "plan_id": getattr(plan, "plan_id", None),
                    "phase5_committed": bool(use_agentic_result),
                    "phase6_terminal": getattr(refinement_result, "stop_reason", None),
                    "safety_version": "phase6.1",
                    "phase5": {
                        "outcome": phase5_outcome,
                        "plan_id": getattr(plan, "plan_id", None),
                        "committed": bool(use_agentic_result),
                        "fallback_to_baseline": phase5_fallback,
                        "fallback_reason": getattr(agentic_result, "fallback_reason", None),
                        "kbinfos": deepcopy(getattr(agentic_result, "kbinfos", None)),
                    },
                    "phase6": {
                        "outcome": phase6_outcome,
                        "stop_reason": getattr(refinement_result, "stop_reason", None),
                        "committed": bool(use_refinement_result),
                        "fallback_to_previous_context": phase6_fallback,
                        "fallback_reason": getattr(refinement_result, "fallback_reason", None),
                        "changed": bool(getattr(refinement_result, "changed", False)),
                        "selected_new_evidence_count": getattr(refinement_result, "selected_new_evidence_count", 0),
                        "kbinfos": deepcopy(getattr(refinement_result, "kbinfos", None)),
                    },
                    "safety": {
                        "version": "phase6.1",
                        "fingerprint_version": 1,
                        "search_fingerprint": search_fingerprint,
                        "config_fingerprint": _agentic_fingerprint(
                            {
                                "agentic_retrieval_config": self.agentic_retrieval_config,
                                "agentic_refinement_config": self.agentic_refinement_config,
                                "context_builder_config": self.context_builder_config,
                                "fusion_config": FusionConfig.from_env(),
                            }
                        ),
                    },
                    "accounting": {
                        "planner_elapsed_ms": round(self._planner_elapsed_ms, 3),
                        "planner_calls_used": self._planner_calls_used,
                        "agentic_elapsed_ms": round(self._agentic_elapsed_ms, 3),
                        "tool_elapsed_ms": round(current_tool_elapsed_ms, 3),
                        "cumulative_tool_elapsed_ms": round(self._agentic_tool_elapsed_ms + current_tool_elapsed_ms, 3),
                        "remaining_turn_agentic_budget_ms": round(self._remaining_agentic_budget_ms(), 3),
                        "stage_ms": deepcopy(self._agentic_stage_ms),
                    },
                }
                raise_if_local_work_deadline_exhausted()
                self._completed_search_ledger[search_fingerprint] = deepcopy(completed_record)
                if self.trace:
                    self.trace.add_agentic_retrieval_event(
                        "exact_result_cache",
                        {
                            "operation_hash": search_fingerprint[:16],
                            "cache_status": "stored_completed",
                            "plan_id": getattr(getattr(agentic_result, "plan", None), "plan_id", None),
                            "terminal": True,
                        },
                    )
            raise_if_local_work_deadline_exhausted()
            self._trace_tool_end(tool_call_id, result=result, result_summary=result_summary)
            raise_if_local_work_deadline_exhausted()
            return result
        except asyncio.CancelledError:
            local_timeout = work_task in self._rag_agent_tool_expired_tasks
            if planner_terminal_keys_before is not None:
                for key in set(self._planner_terminal_ledger) - planner_terminal_keys_before:
                    self._planner_terminal_ledger.pop(key, None)
            if validated_plan_keys_before is not None:
                for key in set(self._validated_plan_cache) - validated_plan_keys_before:
                    self._validated_plan_cache.pop(key, None)
            if completed_search_keys_before is not None:
                for key in set(self._completed_search_ledger) - completed_search_keys_before:
                    self._completed_search_ledger.pop(key, None)
            if kbinfos_before is not None:
                self.kbinfos = kbinfos_before
                self._last_context_query = last_context_query_before
                self._citations_injected = citations_injected_before
            if self.trace:
                self.trace.add_agentic_retrieval_event(
                    "rag_agent_tool_timeout" if local_timeout else "cancellation",
                    {
                        "operation_hash": search_fingerprint[:16] if search_fingerprint else None,
                        "outcome": "timeout" if local_timeout else "cancelled",
                        "timeout_owner": "rag_agent_tool" if local_timeout else "upstream_cancellation",
                        "terminal": True,
                        "admitted_new_work": False,
                    },
                )
            self._trace_tool_end(
                tool_call_id,
                success=False,
                result_summary={
                    "cancelled": not local_timeout,
                    "outcome": "timeout" if local_timeout else "cancelled",
                    "timeout_owner": "rag_agent_tool" if local_timeout else "upstream_cancellation",
                },
                error="rag_agent_tool_timeout" if local_timeout else "upstream_cancellation",
            )
            if local_timeout:
                raise asyncio.TimeoutError("rag agent tool work deadline exceeded") from None
            raise
        except Exception as e:
            self._trace_tool_end(tool_call_id, success=False, error=e)
            raise
        finally:
            if not watchdog_task.done():
                watchdog_task.cancel()
            elapsed_ms = (time.monotonic() - tool_started) * 1000.0
            self._agentic_tool_elapsed_ms += elapsed_ms
            if self.trace:
                self.trace.add_agentic_retrieval_event(
                    "tool_accounting",
                    {
                        "operation_hash": search_fingerprint[:16] if search_fingerprint else None,
                        "tool_latency_ms": round(elapsed_ms, 3),
                        "cumulative_tool_latency_ms": round(self._agentic_tool_elapsed_ms, 3),
                        "cumulative_agentic_ms": round(self._agentic_elapsed_ms, 3),
                        "remaining_turn_agentic_budget_ms": round(self._remaining_agentic_budget_ms(), 3),
                    },
                )
            if lock_acquired:
                self._agentic_search_lock.release()
            if not watchdog_task.done():
                try:
                    await watchdog_task
                except asyncio.CancelledError:
                    pass
            self._rag_agent_tool_expired_tasks.discard(work_task)


    def _retrieval_diagnostics(
        self,
        question: str,
        keywords: str,
        docid_scope: list[str] | None,
        using_embedding: bool,
        chunks: list[dict[str, Any]],
        kbinfos: dict[str, Any],
        fusion_config: FusionConfig,
        used_embedding_for_fusion: bool,
    ) -> dict[str, Any]:
        diagnostics = self._retry_diagnostics(question, keywords, docid_scope, using_embedding, chunks)
        fusion = (kbinfos.get("diagnostics") or {}).get("fusion") or kbinfos.get("fusion")
        requested_mode = "embedding" if using_embedding else "keyword"
        diagnostics["retrieval"] = {
            "requested_mode": requested_mode,
            "effective_mode": "fusion" if fusion and fusion.get("enabled") else requested_mode,
            "used_embedding_for_fusion": bool(fusion_config.enabled and used_embedding_for_fusion),
        }
        if fusion:
            diagnostics["fusion"] = fusion
            notes = fusion.get("fallback_notes") or []
            if notes:
                diagnostics["retrieval"]["fallback_note"] = notes[0]
        return diagnostics

    def _retry_diagnostics(self, question: str, keywords: str, docid_scope: list[str] | None, using_embedding: bool, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """Track keyword-first vs embedding-retry evidence deltas for traces."""
        scope_key = tuple(sorted(docid_scope or []))
        key = (question or "", keywords or "", scope_key)
        chunk_ids = {str(c.get("chunk_id") or c.get("id")) for c in chunks if c.get("chunk_id") or c.get("id")}
        if not using_embedding:
            self._keyword_first_chunk_ids[key] = chunk_ids
            return {
                "keyword_first_chunk_count": len(chunk_ids),
                "embedding_retry_zero_chunks": None,
                "new_evidence_count": None,
                "new_chunk_ids_after_retry": [],
            }
        previous = self._keyword_first_chunk_ids.get(key, set())
        new_ids = sorted(chunk_ids - previous)
        return {
            "keyword_first_chunk_count": len(previous),
            "embedding_retry_zero_chunks": len(chunk_ids) == 0,
            "new_evidence_count": len(new_ids),
            "new_chunk_ids_after_retry": new_ids[:32],
        }

    def _resolve_doc_tenant(self, doc_id: str) -> tuple[str, str] | None:
        """Return ``(kb_id, tenant_id)`` for ``doc_id`` if and only if the
        document belongs to one of the agent's bound unstructured KBs.

        Returns ``None`` otherwise — used by ``summarize_document`` both as
        a hallucination guard against fabricated 32-char hex IDs and as the
        tenant-resolution step needed to query the doc store.

        Sync DB call — wrap in ``thread_pool_exec`` at the call site.
        """
        rows = list(
            Document.select(Document.kb_id).where(
                (Document.id == doc_id) & (Document.kb_id.in_(self.kb_ids))
            )
        )
        if not rows:
            return None
        kb_id = rows[0].kb_id
        for kb in self.kbs:
            if kb.id == kb_id:
                return kb_id, kb.tenant_id
        return None

    @tool
    async def summarize_document(self, doc_id: str) -> list[str]:
        """Return a single document's content, position-ordered, ready to be summarised.

        Call this tool ONLY when the user EXPLICITLY asks for a summary of a
        specific document — phrasings like "summarize the security audit",
        "give me a summary of doc X", "tldr the onboarding guide". Do NOT
        call it for general Q&A: use ``search_knowledge_bases`` for that.

        The tool fetches every chunk of the named document from the doc
        store, sorted by page / position so reading order is preserved, and
        formats them with ``kb_prompt`` so the result already respects the
        chat model's context-length budget (chunks past the budget are
        dropped with a warning). The output is the full chunk-formatted
        text that you, the calling LLM, should then turn into a natural-
        language summary in the user's language — applying the citation
        rules from the system prompt to attribute claims to chunk IDs.

        :param doc_id: a 32-character lowercase hex string (e.g.
            ``41a5271858ca11f1bbb9047c16ec874f``). You DO NOT know any doc
            IDs on your own and you MUST NOT invent one. Acceptable sources
            are doc IDs returned VERBATIM from a previous
            ``select_documents`` or ``filter_docs_by_metadata`` call in
            this same turn — typically you will call ``select_documents``
            first to map the user's spoken document title to an ID.

        :returns: a list of formatted chunk blocks (one per chunk, in
            document order, each carrying its ID / title / content) that
            collectively fit within the chat model's context budget. An
            empty list is returned when the doc ID is unknown to the bound
            KBs or the document has no chunks indexed.
        """
        tool_call_id = self._trace_tool_start("summarize_document", {"doc_id": doc_id}, retrieval_mode="summarize")
        try:
            if not self.kb_ids:
                self._trace_tool_end(tool_call_id, result=[])
                return []

            resolved = await thread_pool_exec(self._resolve_doc_tenant, doc_id)
            if resolved is None:
                logging.warning(
                    f"summarize_document: doc_id {doc_id!r} is not in any bound "
                    "knowledge base — refusing to fetch (likely an LLM hallucination)"
                )
                self._trace_tool_end(tool_call_id, result=[], result_summary={"chunk_count": 0, "doc_id": doc_id})
                return []
            kb_id, tenant_id = resolved

            cks = []
            tokens = 0
            retrieval_started_ms = _now_ms()
            for offset in range(0, 10000, 128):
                chunks = await thread_pool_exec(
                    settings.retriever.chunk_list,
                    doc_id,
                    tenant_id,
                    [kb_id],
                    max_count=offset+128,
                    offset=offset,
                    fields=["content_with_weight", "docnm_kwd", "doc_id"],
                    sort_by_position=True,
                    retrieve_all=False,
                )
                for ck in chunks:
                    num = num_tokens_from_string(str(ck["content_with_weight"]))
                    if tokens + num > self.chat_mdl.max_length:
                        break
                    tokens += num
                    cks.append(ck)

            if not cks:
                self._trace_tool_end(tool_call_id, result=[], result_summary={"chunk_count": 0, "doc_id": doc_id})
                return []
            doc_name = next(
                (c.get("docnm_kwd") or "" for c in cks if c.get("docnm_kwd")),
                "",
            )
            kbinfos = {
                "chunks": cks,
                "doc_aggs": [
                    {
                        "doc_name": doc_name,
                        "doc_id": doc_id,
                        "count": len(cks),
                    }
                ],
            }
            start_idx = len(self.kbinfos.get("chunks", []))
            retrieval_call_id = None
            if self.trace:
                retrieval_call_id = self.trace.add_retrieval_call(
                    mode="summarize",
                    query_text=doc_id,
                    docid_scope=[doc_id],
                    chunks=kbinfos.get("chunks", []),
                    doc_aggs=kbinfos.get("doc_aggs", []),
                    tool_call_id=tool_call_id,
                    started_ms=retrieval_started_ms,
                )
                self.trace.add_evidence_from_chunks(
                    kbinfos.get("chunks", []),
                    source_type="summarize",
                    retrieval_call_id=retrieval_call_id,
                    tool_call_id=tool_call_id,
                    start_citation_index=start_idx,
                )
            if kbinfos:
                if self.context_builder_config.enabled:
                    mark_chunks_source_type(kbinfos.get("chunks", []), "summary")
                self.kbinfos["chunks"].extend(kbinfos.get("chunks", []))
                self.kbinfos["doc_aggs"].extend(kbinfos.get("doc_aggs", []))
            prompt_kbinfos = self.context_kbinfos(stage="rag_agent.summarize_document", start_citation_index=self._prompt_start_idx(start_idx))
            result = self._with_citation_guidelines(
                kb_prompt(prompt_kbinfos, self.chat_mdl.max_length, self._prompt_start_idx(start_idx))
            )
            self._trace_tool_end(tool_call_id, result=result, result_summary={"retrieval_call_id": retrieval_call_id, "chunk_count": len(cks), "doc_id": doc_id})
            return result
        except Exception as e:
            self._trace_tool_end(tool_call_id, success=False, error=e)
            raise

    @tool(timeout=60)
    async def search_structured_data(self, question: str) -> str:
        """Query the structured (tabular) knowledge bases by translating the
        question into SQL and executing it.

        This tool is only registered when at least one bound knowledge base
        carries a ``field_map`` (i.e. its documents are spreadsheets / tables
        with a typed schema). It asks an LLM to generate SQL against that
        schema, executes it via the document engine (Elasticsearch / Infinity
        / OceanBase), and returns the formatted answer with citation markers.

        Use this tool when the question is naturally answered by an aggregate
        or filter over tabular data ("how many orders in 2024?", "list the
        top-5 vendors by spend"). For free-text questions, prefer
        ``search_knowledge_bases``.

        Matching chunks and doc aggregations are appended to the running
        ``self.kbinfos`` accumulator, so any subsequent retrieval tool can
        share the citation pool.

        :param question: the self-contained natural-language question (run
            ``formalize_question`` first if the latest user message is a
            follow-up that depends on earlier turns).

        :returns: the natural-language answer produced from the SQL result,
            already including the citation markers required by the citation
            rules in the system prompt. An empty string is returned when no
            structured KB is bound or SQL generation/execution fails.
        """
        tool_call_id = self._trace_tool_start("search_structured_data", {"question": question}, retrieval_mode="sql")
        try:
            if not self.sql_kbs or not self.field_map:
                self._trace_tool_end(tool_call_id, result="")
                return ""

            # Imported lazily to avoid a circular import:
            # dialog_service constructs ``RAGTools``.
            from api.db.services.dialog_service import use_sql

            sql_kb_ids = [kb.id for kb in self.sql_kbs]
            tenant_id = self.sql_kbs[0].tenant_id
            started_ms = _now_ms()
            try:
                ans = await use_sql(
                    question,
                    self.field_map,
                    tenant_id,
                    self.chat_mdl,
                    quota=True,
                    kb_ids=sql_kb_ids,
                )
            except Exception as e:
                logging.exception(f"search_structured_data: use_sql failed: {e}")
                if self.trace:
                    self.trace.add_retrieval_call(mode="sql", query_text=question, docid_scope=sql_kb_ids, tool_call_id=tool_call_id, started_ms=started_ms, error=e)
                self._trace_tool_end(tool_call_id, success=False, result="", error=e)
                return ""

            if not ans:
                self._trace_tool_end(tool_call_id, result="", result_summary={"chunk_count": 0, "doc_agg_count": 0})
                return ""

            reference = ans.get("reference") or {}
            new_chunks = reference.get("chunks") or []
            new_doc_aggs = reference.get("doc_aggs") or []
            start_idx = len(self.kbinfos.get("chunks", []))
            retrieval_call_id = None
            if self.trace:
                retrieval_call_id = self.trace.add_retrieval_call(
                    mode="sql",
                    query_text=question,
                    docid_scope=sql_kb_ids,
                    chunks=new_chunks,
                    doc_aggs=new_doc_aggs,
                    tool_call_id=tool_call_id,
                    started_ms=started_ms,
                )
                self.trace.add_evidence_from_chunks(
                    new_chunks,
                    source_type="sql",
                    retrieval_call_id=retrieval_call_id,
                    tool_call_id=tool_call_id,
                    start_citation_index=start_idx,
                )
            if new_chunks:
                if self.context_builder_config.enabled:
                    mark_chunks_source_type(new_chunks, "sql")
                self.kbinfos["chunks"].extend(new_chunks)
            if new_doc_aggs:
                self.kbinfos["doc_aggs"].extend(new_doc_aggs)

            result = self._with_citation_guidelines(ans.get("answer", "") or "")
            self._trace_tool_end(tool_call_id, result=result, result_summary={"retrieval_call_id": retrieval_call_id, "chunk_count": len(new_chunks), "doc_agg_count": len(new_doc_aggs)})
            return result
        except Exception as e:
            self._trace_tool_end(tool_call_id, success=False, error=e)
            raise

    @tool
    async def web_search(self, query: str) -> List[dict[str, Any]]:
        """Search the public web for information not available in the knowledge base.

        Use this tool ONLY as a fallback when the knowledge-base retrieval
        tool returned no relevant chunks for the user's question. Prefer
        KB-grounded answers whenever the KB has the information.

        :param query: a self-contained natural-language search query
            (resolve pronouns / follow-up references before calling)

        :returns: a list of search results, each shaped as
            ``{"url": str, "title": str, "content": str, "score": float}``,
            or an empty list when web search is not configured or fails.
        """
        tool_call_id = self._trace_tool_start("web_search", {"query": query}, retrieval_mode="web")
        try:
            if self.tav is None:
                self._trace_tool_end(tool_call_id, result=[])
                return []
            started_ms = _now_ms()
            tav_res = await thread_pool_exec(self.tav.retrieve_chunks, query)
            start_idx = len(self.kbinfos.get("chunks", []))
            retrieval_call_id = None
            if self.trace:
                retrieval_call_id = self.trace.add_retrieval_call(
                    mode="web",
                    query_text=query,
                    chunks=tav_res.get("chunks", []),
                    doc_aggs=tav_res.get("doc_aggs", []),
                    tool_call_id=tool_call_id,
                    used_web=True,
                    started_ms=started_ms,
                )
                self.trace.add_evidence_from_chunks(
                    tav_res.get("chunks", []),
                    source_type="web",
                    retrieval_call_id=retrieval_call_id,
                    tool_call_id=tool_call_id,
                    start_citation_index=start_idx,
                )
            if self.context_builder_config.enabled:
                mark_chunks_source_type(tav_res.get("chunks", []), "web")
            self.kbinfos["chunks"].extend(tav_res["chunks"])
            self.kbinfos["doc_aggs"].extend(tav_res["doc_aggs"])
            prompt_kbinfos = self.context_kbinfos(stage="rag_agent.web_search", start_citation_index=self._prompt_start_idx(start_idx))
            result = self._with_citation_guidelines(
                kb_prompt(prompt_kbinfos, self.chat_mdl.max_length, self._prompt_start_idx(start_idx))
            )
            self._trace_tool_end(tool_call_id, result=result, result_summary={"retrieval_call_id": retrieval_call_id, "chunk_count": len(tav_res.get("chunks", [])), "doc_agg_count": len(tav_res.get("doc_aggs", [])), "used_web": True})
            return result
        except Exception as e:
            self._trace_tool_end(tool_call_id, success=False, error=e)
            raise

    def get_citation_guidelines(self) -> str:
        """Return the citation guidelines this agent uses.

        Plain method (NOT registered as a tool): the guidelines are static
        and are embedded directly in ``sys_prompt()`` so the chat model sees
        them from token zero. Letting the model decide whether to fetch them
        via a tool call was unreliable — the call was routinely skipped.
        Kept as a public helper so callers can introspect / override the
        text from outside the class.
        """
        return citation_prompt(self.user_defined_prompts)
