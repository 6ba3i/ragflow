#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#
"""Phase 1 retrieval diagnostics and ablation helpers.

These helpers are intentionally diagnostic-only.  They describe retrieval
variants in terms of the knobs already exposed by ``Dealer.retrieval`` and
compute offline, unlabeled ablation metrics without changing production
retrieval defaults.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

from rag.utils.context_builder import EvidenceBundleConfig, apply_context_builder_to_kbinfos

RetrievalVariantName = Literal[
    "hybrid_default",
    "bm25_only",
    "dense_only",
    "hybrid_weighted",
    "keyword_first",
    "embedding_retry",
]

DEFAULT_RETRIEVAL_VARIANT: RetrievalVariantName = "hybrid_default"
SUPPORTED_RETRIEVAL_VARIANTS: tuple[RetrievalVariantName, ...] = (
    "hybrid_default",
    "bm25_only",
    "dense_only",
    "hybrid_weighted",
    "keyword_first",
    "embedding_retry",
)


@dataclass(frozen=True)
class RetrievalVariant:
    """Explicit diagnostic representation of a retrieval variant.

    ``dense_only`` is a best-effort diagnostic on the current ES path: it sets
    the final local fusion weight to vector-only and asks for embeddings, but
    ``Dealer.search`` still builds the ES candidate pool with the existing
    lexical+dense query and filters.  This avoids a deeper ES behavior fork in
    Phase 1 while making the limitation explicit in traces/reports.
    """

    name: RetrievalVariantName = DEFAULT_RETRIEVAL_VARIANT
    using_embedding: bool | None = None
    vector_similarity_weight: float | None = None
    similarity_threshold: float | None = None
    note: str | None = None

    def to_trace_fields(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


def _clamp_weight(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        raise ValueError("vector_similarity_weight must be finite")
    return max(0.0, min(1.0, float(value)))


def make_retrieval_variant(
    name: str | None = None,
    *,
    vector_similarity_weight: float | None = None,
    similarity_threshold: float | None = None,
) -> RetrievalVariant:
    """Return a safe diagnostic variant mapped to existing retrieval knobs."""

    variant_name = (name or DEFAULT_RETRIEVAL_VARIANT).strip().lower()
    if variant_name not in SUPPORTED_RETRIEVAL_VARIANTS:
        raise ValueError(f"Unsupported retrieval variant: {name!r}")

    if variant_name in {"bm25_only", "keyword_first"}:
        return RetrievalVariant(
            name=variant_name,  # type: ignore[arg-type]
            using_embedding=False,
            vector_similarity_weight=0.0,
            similarity_threshold=similarity_threshold,
        )
    if variant_name == "dense_only":
        return RetrievalVariant(
            name="dense_only",
            using_embedding=True,
            vector_similarity_weight=1.0,
            similarity_threshold=similarity_threshold,
            note="Dense-only is diagnostic best-effort: final score uses vector weight 1.0, while existing ES candidate generation may still include lexical clauses and filters.",
        )
    if variant_name == "embedding_retry":
        return RetrievalVariant(
            name="embedding_retry",
            using_embedding=True,
            vector_similarity_weight=_clamp_weight(0.7 if vector_similarity_weight is None else vector_similarity_weight),
            similarity_threshold=similarity_threshold,
        )
    if variant_name == "hybrid_weighted":
        if vector_similarity_weight is None:
            raise ValueError("hybrid_weighted requires vector_similarity_weight")
        return RetrievalVariant(
            name="hybrid_weighted",
            using_embedding=True,
            vector_similarity_weight=_clamp_weight(vector_similarity_weight),
            similarity_threshold=similarity_threshold,
        )
    return RetrievalVariant(
        name="hybrid_default",
        using_embedding=True,
        vector_similarity_weight=None if vector_similarity_weight is None else _clamp_weight(vector_similarity_weight),
        similarity_threshold=similarity_threshold,
    )


def resolve_variant_knobs(
    variant: RetrievalVariant | str | None,
    *,
    default_vector_similarity_weight: float,
    default_similarity_threshold: float,
    embedding_available: bool = True,
) -> dict[str, Any]:
    """Resolve a variant to concrete knobs without mutating caller defaults."""

    variant_obj = make_retrieval_variant(variant) if not isinstance(variant, RetrievalVariant) else variant
    requested_embedding = bool(variant_obj.using_embedding)
    using_embedding = embedding_available if variant_obj.using_embedding is None else (requested_embedding and embedding_available)
    vector_weight = default_vector_similarity_weight if variant_obj.vector_similarity_weight is None else variant_obj.vector_similarity_weight
    note = variant_obj.note
    effective_variant = variant_obj.name
    if requested_embedding and not embedding_available:
        vector_weight = 0.0
        effective_variant = "bm25_only_no_embedding"
        missing_note = "Embedding model unavailable; diagnostic run fell back to lexical-only retrieval while preserving the requested variant label."
        note = f"{note} {missing_note}" if note else missing_note
    elif not using_embedding:
        vector_weight = 0.0
    threshold = default_similarity_threshold if variant_obj.similarity_threshold is None else variant_obj.similarity_threshold
    return {
        "retrieval_variant": variant_obj.name,
        "effective_retrieval_variant": effective_variant,
        "using_embedding": using_embedding,
        "vector_similarity_weight": vector_weight,
        "similarity_threshold": threshold,
        "note": note,
    }


def generate_sweep_variants(
    *,
    base_variants: Iterable[str] = ("hybrid_default", "bm25_only", "dense_only"),
    vector_weights: Iterable[float] = (),
    similarity_thresholds: Iterable[float] = (),
) -> list[RetrievalVariant]:
    variants = [make_retrieval_variant(name) for name in base_variants]
    for weight in vector_weights:
        variants.append(make_retrieval_variant("hybrid_weighted", vector_similarity_weight=weight))
    if similarity_thresholds:
        expanded: list[RetrievalVariant] = []
        for variant in variants:
            for threshold in similarity_thresholds:
                expanded.append(replace(variant, similarity_threshold=float(threshold)))
        variants = expanded
    return variants


def duplicate_doc_ratio(chunks: list[dict[str, Any]]) -> float:
    doc_ids = [c.get("doc_id") for c in chunks if c.get("doc_id")]
    if not doc_ids:
        return 0.0
    return round(1.0 - (len(set(doc_ids)) / len(doc_ids)), 6)


def score_summary(chunks: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    fields = ("similarity", "term_similarity", "vector_similarity", "rank_feature_score", "final_score")
    summary: dict[str, dict[str, float]] = {}
    for field in fields:
        vals = [float(c[field]) for c in chunks if isinstance(c.get(field), (int, float))]
        if vals:
            ordered = sorted(vals)
            summary[field] = {
                "min": ordered[0],
                "avg": statistics.fmean(vals),
                "p50": _percentile(ordered, 50),
                "p95": _percentile(ordered, 95),
                "max": ordered[-1],
            }
    return summary


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_values[int(rank)]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (rank - lo)


def make_ablation_row(
    *,
    question_id: str,
    question: str,
    variant: str,
    similarity_threshold: float | None,
    vector_similarity_weight: float | None,
    top_k: int | None,
    page_size: int | None,
    retrieval_latency_ms: float | None,
    chunks: list[dict[str, Any]] | None,
    errors: list[str] | None = None,
    effective_retrieval_variant: str | None = None,
    diagnostic_note: str | None = None,
) -> dict[str, Any]:
    chunks = chunks or []
    doc_ids = [c.get("doc_id") for c in chunks if c.get("doc_id")]
    slim_chunks = [
        {
            "chunk_id": c.get("chunk_id") or c.get("id"),
            "doc_id": c.get("doc_id"),
            "similarity": c.get("similarity"),
            "term_similarity": c.get("term_similarity"),
            "vector_similarity": c.get("vector_similarity"),
            "rank_feature_score": c.get("rank_feature_score"),
            "final_score": c.get("final_score", c["score"] if "score" in c else c.get("similarity")),
        }
        for c in chunks
    ]
    return {
        "question_id": question_id,
        "question": question,
        "variant": variant,
        "effective_retrieval_variant": effective_retrieval_variant or variant,
        "diagnostic_note": diagnostic_note,
        "similarity_threshold": similarity_threshold,
        "vector_similarity_weight": vector_similarity_weight,
        "top_k": top_k,
        "page_size": page_size,
        "retrieval_latency_ms": retrieval_latency_ms,
        "chunks": slim_chunks,
        "doc_ids": doc_ids,
        "duplicate_doc_ratio": duplicate_doc_ratio(slim_chunks),
        "context_builder_metrics": make_context_builder_metrics(chunks),
        "errors": errors or [],
    }


def make_context_builder_metrics(chunks: list[dict[str, Any]] | dict[str, Any] | None) -> dict[str, Any]:
    """Unlabeled same-candidate context-builder metrics for offline ablations."""

    if isinstance(chunks, dict):
        chunks = chunks.get("chunks", [])
    chunks = chunks or []
    current = apply_context_builder_to_kbinfos({"chunks": chunks, "doc_aggs": []}, EvidenceBundleConfig(enabled=False))
    deduped = apply_context_builder_to_kbinfos({"chunks": chunks, "doc_aggs": []}, EvidenceBundleConfig(enabled=True))
    max_per_doc = apply_context_builder_to_kbinfos({"chunks": chunks, "doc_aggs": []}, EvidenceBundleConfig(enabled=True, max_chunks_per_doc=1))
    source_types = {str(c.get("_ragflow_source_type") or c.get("source_type") or ("web" if c.get("url") and c.get("doc_id") == c.get("chunk_id") else "kb")) for c in chunks if isinstance(c, dict)}
    doc_ids = {c.get("doc_id") for c in chunks if isinstance(c, dict) and c.get("doc_id")}
    token_estimate = deduped.bundle.summary()["estimated_context_tokens"] if deduped.bundle else 0
    return {
        "variants": [
            "current_top_n",
            "accumulated_kbinfos_as_is",
            "doc_dedup",
            "max_chunks_per_doc",
            "evidence_bundle_builder",
            "larger_top_n_only",
        ],
        "candidate_count": len(chunks),
        "current_selected_count": len(current.kbinfos.get("chunks", [])),
        "dedup_selected_count": len(deduped.kbinfos.get("chunks", [])),
        "max_chunks_per_doc_selected_count": len(max_per_doc.kbinfos.get("chunks", [])),
        "rejected_count": deduped.bundle.summary()["rejected_evidence_count"] if deduped.bundle else 0,
        "duplicate_ratio": duplicate_doc_ratio(chunks),
        "source_diversity": len(source_types),
        "document_diversity": len(doc_ids),
        "estimated_context_tokens": token_estimate,
        "citation_mapping_completeness": 1.0 if not deduped.bundle or all(r.citation_index is not None for r in deduped.bundle.selected) else 0.0,
        "labeled_metrics": {"status": "not_computed", "reason": "No labels supplied; context-builder ablation reports unlabeled metrics only."},
    }


def read_questions(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".jsonl":
        rows = []
        for i, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            data = json.loads(line)
            question = str(data.get("question") or data.get("query") or data.get("text") or "")
            rows.append({"question_id": str(data.get("question_id") or data.get("id") or i), "question": question})
        return rows
    if p.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, list):
            return [
                {"question_id": str(row.get("question_id") or row.get("id") or i + 1), "question": str(row.get("question") or row.get("query") or row.get("text") or "")}
                for i, row in enumerate(data)
            ]
        if isinstance(data, dict) and isinstance(data.get("questions"), list):
            return [
                {"question_id": str(row.get("question_id") or row.get("id") or i + 1), "question": str(row.get("question") or row.get("query") or row.get("text") or "")}
                for i, row in enumerate(data["questions"])
            ]
    return [{"question_id": str(i), "question": line.strip()} for i, line in enumerate(text.splitlines(), 1) if line.strip()]


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_ablation_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    by_variant: dict[str, list[dict[str, Any]]] = {}
    by_question: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        by_variant.setdefault(str(row.get("variant")), []).append(row)
        qid = str(row.get("question_id"))
        by_question.setdefault(qid, {})[str(row.get("variant"))] = {str(c.get("chunk_id")) for c in row.get("chunks", []) if c.get("chunk_id")}

    variants: dict[str, Any] = {}
    for variant, items in sorted(by_variant.items()):
        latencies = [float(r["retrieval_latency_ms"]) for r in items if isinstance(r.get("retrieval_latency_ms"), (int, float))]
        chunk_counts = [len(r.get("chunks", [])) for r in items]
        unique_doc_counts = [len(set(r.get("doc_ids", []))) for r in items]
        dup_ratios = [float(r.get("duplicate_doc_ratio", 0.0) or 0.0) for r in items]
        all_chunks = [c for r in items for c in r.get("chunks", [])]
        variants[variant] = {
            "query_count": len(items),
            "empty_result_rate": _avg([1.0 if not r.get("chunks") else 0.0 for r in items]),
            "latency_ms": _latency_summary(latencies),
            "avg_returned_chunks": _avg(chunk_counts),
            "avg_unique_docs": _avg(unique_doc_counts),
            "avg_duplicate_doc_ratio": _avg(dup_ratios),
            "avg_context_builder_dedup_selected": _avg([float((r.get("context_builder_metrics") or {}).get("dedup_selected_count", 0)) for r in items]),
            "avg_context_builder_rejected": _avg([float((r.get("context_builder_metrics") or {}).get("rejected_count", 0)) for r in items]),
            "avg_context_builder_tokens": _avg([float((r.get("context_builder_metrics") or {}).get("estimated_context_tokens", 0)) for r in items]),
            "score_components": score_summary(all_chunks),
        }

    variant_names = sorted(by_variant)
    overlap: dict[str, dict[str, float]] = {v: {} for v in variant_names}
    disagreements: list[dict[str, Any]] = []
    for qid, variant_chunks in by_question.items():
        for left in variant_names:
            left_set = variant_chunks.get(left, set())
            for right in variant_names:
                right_set = variant_chunks.get(right, set())
                union = left_set | right_set
                overlap[left][right] = overlap[left].get(right, 0.0) + ((len(left_set & right_set) / len(union)) if union else 1.0)
        if len(variant_chunks) >= 2:
            all_sets = list(variant_chunks.values())
            union = set().union(*all_sets)
            intersection = set.intersection(*all_sets) if all_sets else set()
            disagreement = 1.0 - ((len(intersection) / len(union)) if union else 1.0)
            disagreements.append({"question_id": qid, "disagreement": round(disagreement, 6), "variants": {k: len(v) for k, v in variant_chunks.items()}})
    q_count = max(1, len(by_question))
    for left in variant_names:
        for right in variant_names:
            overlap[left][right] = round(overlap[left].get(right, 0.0) / q_count, 6)

    return {
        "query_count": len(by_question),
        "variant_count": len(variant_names),
        "variants": variants,
        "overlap_matrix": overlap,
        "top_disagreements": sorted(disagreements, key=lambda r: r["disagreement"], reverse=True)[:10],
        "labeled_metrics": {"status": "not_computed", "reason": "No labels supplied; Phase 1 does not fake labeled metrics."},
    }


def _avg(values: Iterable[float]) -> float:
    values = list(values)
    return round(statistics.fmean(values), 6) if values else 0.0


def _latency_summary(values: list[float]) -> dict[str, float | None]:
    ordered = sorted(values)
    if not ordered:
        return {"avg": None, "p50": None, "p95": None}
    return {"avg": round(statistics.fmean(ordered), 3), "p50": round(_percentile(ordered, 50), 3), "p95": round(_percentile(ordered, 95), 3)}


def write_markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Retrieval Ablation Summary",
        "",
        f"- Query count: {summary['query_count']}",
        f"- Variant count: {summary['variant_count']}",
        f"- Labeled metrics: {summary['labeled_metrics']['status']} ({summary['labeled_metrics']['reason']})",
        "",
        "## Variants",
        "",
        "| Variant | Empty rate | Avg latency ms | P50 | P95 | Avg chunks | Avg unique docs | Avg dup doc ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, stats in summary["variants"].items():
        latency = stats["latency_ms"]
        lines.append(
            f"| {variant} | {stats['empty_result_rate']:.3f} | {latency['avg']} | {latency['p50']} | {latency['p95']} | "
            f"{stats['avg_returned_chunks']:.3f} | {stats['avg_unique_docs']:.3f} | {stats['avg_duplicate_doc_ratio']:.3f} |"
        )
    lines.extend(["", "## Overlap matrix", ""])
    variants = sorted(summary["overlap_matrix"])
    lines.append("| Variant | " + " | ".join(variants) + " |")
    lines.append("|---" + "|---:" * len(variants) + "|")
    for left in variants:
        lines.append("| " + left + " | " + " | ".join(f"{summary['overlap_matrix'][left][right]:.3f}" for right in variants) + " |")
    lines.extend(["", "## Top disagreements", ""])
    for row in summary["top_disagreements"]:
        lines.append(f"- {row['question_id']}: disagreement={row['disagreement']}, chunk_counts={row['variants']}")
    return "\n".join(lines) + "\n"


async def run_retrieval_variant(
    *,
    retriever: Any,
    question: str,
    tenant_ids: list[str],
    kb_ids: list[str],
    embd_mdl: Any = None,
    variant: RetrievalVariant,
    page_size: int = 20,
    top_k: int = 128,
    default_similarity_threshold: float = 0.2,
    default_vector_similarity_weight: float = 0.3,
    doc_ids: list[str] | None = None,
    rank_feature: dict[str, Any] | None = None,
) -> dict[str, Any]:
    knobs = resolve_variant_knobs(
        variant,
        default_vector_similarity_weight=default_vector_similarity_weight,
        default_similarity_threshold=default_similarity_threshold,
        embedding_available=embd_mdl is not None,
    )
    model = embd_mdl if knobs["using_embedding"] else None
    start = time.perf_counter()
    errors: list[str] = []
    chunks: list[dict[str, Any]] = []
    try:
        result = await retriever.retrieval(
            question,
            model,
            tenant_ids,
            kb_ids,
            1,
            page_size,
            knobs["similarity_threshold"],
            vector_similarity_weight=knobs["vector_similarity_weight"],
            top=top_k,
            doc_ids=doc_ids,
            aggs=True,
            rank_feature=rank_feature,
        )
        chunks = result.get("chunks", []) if isinstance(result, dict) else []
    except Exception as exc:  # diagnostics should report paired failures
        errors.append(str(exc))
    return make_ablation_row(
        question_id="",
        question=question,
        variant=variant.name,
        effective_retrieval_variant=knobs["effective_retrieval_variant"],
        diagnostic_note=knobs.get("note"),
        similarity_threshold=knobs["similarity_threshold"],
        vector_similarity_weight=knobs["vector_similarity_weight"],
        top_k=top_k,
        page_size=page_size,
        retrieval_latency_ms=round((time.perf_counter() - start) * 1000, 3),
        chunks=chunks,
        errors=errors,
    )
