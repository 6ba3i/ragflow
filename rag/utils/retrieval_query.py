#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#
"""Immutable query roles shared by dataset and agent retrieval."""

from __future__ import annotations

import json
import re
import unicodedata
from csv import Error as CsvError
from csv import reader as csv_reader
from dataclasses import dataclass
from typing import Iterable, Literal

_MAX_EXACT_PHRASE_CHARS = 160
_MAX_EXACT_PHRASES = 3
_MAX_EXACT_PHRASE_TOKENS = 12
_MAX_EXPANSION_TERM_CHARS = 80
_MAX_EXPANSION_TERM_TOKENS = 8
_MAX_LEGACY_CSV_TERM_TOKENS = 3
_MAX_LEGACY_CSV_TERMS = 8
_QUERY_SYNTAX = frozenset('"*?~^:(){}[]\\/')
_TOKEN_RE = re.compile(r"[^\W_]+(?:['\N{RIGHT SINGLE QUOTATION MARK}-][^\W_]+)*", re.UNICODE)
_ENTITY_PHRASE_RE = re.compile(
    r"[^\W_]+(?:['\N{RIGHT SINGLE QUOTATION MARK}-][^\W_]+)*(?:\s+[^\W_]+(?:['\N{RIGHT SINGLE QUOTATION MARK}-][^\W_]+)*)+",
    re.UNICODE,
)
_QUOTED_SPAN_RE = re.compile(r'"([^"\r\n]+)"')
# Legacy CSV fields are noun-like keywords, not instructions. Reject fields
# whose first token belongs to this control-verb class before applying the
# lowercase-keyword or redundant-entity production below.
_LEGACY_CONTROL_VERBS = frozenset(
    {
        "analyze",
        "compare",
        "consider",
        "describe",
        "explain",
        "explore",
        "find",
        "identify",
        "list",
        "provide",
        "recommend",
        "retrieve",
        "return",
        "search",
        "select",
        "suggest",
        "use",
    }
)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "were",
        "with",
    }
)


def normalize_retrieval_query(value: str) -> str:
    """Return a stable Unicode and whitespace representation of a query."""

    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def _bounded_exact_phrase(value: str) -> tuple[str, list[str]] | None:
    if not isinstance(value, str) or not value or any(unicodedata.category(char) == "Cc" for char in value):
        return None
    phrase = normalize_retrieval_query(value)
    if not phrase or len(phrase) > _MAX_EXACT_PHRASE_CHARS or any(char in _QUERY_SYNTAX for char in phrase):
        return None
    tokens = _TOKEN_RE.findall(phrase)
    if not 2 <= len(tokens) <= _MAX_EXACT_PHRASE_TOKENS or _ENTITY_PHRASE_RE.fullmatch(phrase) is None:
        return None
    meaningful = [token for token in tokens if len(token) > 1 and token.casefold() not in _STOPWORDS]
    return (phrase, meaningful) if len(meaningful) >= 2 else None


def _has_entity_token_form(token: str) -> bool:
    cased_characters = [char for char in token if char.isalpha() and char.lower() != char.upper()]
    return not cased_characters or cased_characters[0].isupper()


def is_exact_phrase_eligible(value: str) -> bool:
    """Return whether *value* is conservative enough for a phrase lane."""

    bounded = _bounded_exact_phrase(value)
    return bounded is not None and all(_has_entity_token_form(token) for token in bounded[1])


def eligible_exact_phrases(value: str) -> tuple[str, ...]:
    """Return bounded exact phrases from an entity query or quoted spans."""

    if not isinstance(value, str) or any(unicodedata.category(char) == "Cc" for char in value):
        return ()
    if is_exact_phrase_eligible(value):
        return (normalize_retrieval_query(value),)

    phrases: list[str] = []
    seen: set[str] = set()
    for match in _QUOTED_SPAN_RE.finditer(value):
        if re.search(r":\s*$", value[: match.start()]):
            continue
        bounded = _bounded_exact_phrase(match.group(1))
        if bounded is None:
            continue
        phrase = bounded[0]
        key = phrase.casefold()
        if key in seen:
            continue
        phrases.append(phrase)
        seen.add(key)
        if len(phrases) >= _MAX_EXACT_PHRASES:
            break
    return tuple(phrases)


def _normalize_terms(terms: Iterable[str], base_query: str) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    base_tokens = {token.casefold() for token in _TOKEN_RE.findall(base_query)}
    for term in terms:
        value = normalize_retrieval_query(term)
        key = value.casefold()
        term_tokens = {token.casefold() for token in _TOKEN_RE.findall(value)}
        if value and key not in seen and term_tokens and not term_tokens.issubset(base_tokens):
            normalized.append(value)
            seen.add(key)
    return tuple(normalized)


ExpansionStatus = Literal["disabled", "accepted", "empty", "rejected", "failed"]
CrossLanguageStatus = Literal["disabled", "applied", "failed"]


@dataclass(frozen=True)
class KeywordRejection:
    term: str
    reason: str


@dataclass(frozen=True)
class KeywordExpansionResult:
    status: ExpansionStatus
    terms: tuple[str, ...] = ()
    rejected: tuple[KeywordRejection, ...] = ()
    reason: str | None = None


def parse_keyword_expansion(
    raw: str,
    original_query: str,
    *,
    max_terms: int = 3,
) -> KeywordExpansionResult:
    """Validate JSON keyword output, with a bounded legacy comma fallback."""

    if not isinstance(raw, str):
        return KeywordExpansionResult(status="rejected", reason="non_string_output")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = _legacy_keyword_payload(raw, original_query)
        if payload is None:
            return KeywordExpansionResult(status="rejected", reason="invalid_json")
    if not isinstance(payload, dict) or set(payload) != {"keywords"} or not isinstance(payload["keywords"], list):
        return KeywordExpansionResult(status="rejected", reason="invalid_shape")

    limit = max(0, min(int(max_terms), 8))
    original_tokens = {token.casefold() for token in _TOKEN_RE.findall(normalize_retrieval_query(original_query))}
    accepted: list[str] = []
    rejected: list[KeywordRejection] = []
    seen: set[str] = set()
    for index, candidate in enumerate(payload["keywords"]):
        if not isinstance(candidate, str):
            rejected.append(KeywordRejection(term="", reason="invalid_term_type"))
            continue
        if any(unicodedata.category(char) == "Cc" for char in candidate):
            rejected.append(KeywordRejection(term="", reason="control_characters"))
            continue
        term = normalize_retrieval_query(candidate)
        if not term:
            rejected.append(KeywordRejection(term="", reason="empty_term"))
            continue
        if index >= limit:
            rejected.append(KeywordRejection(term=term, reason="too_many_terms"))
            continue
        if len(term) > _MAX_EXPANSION_TERM_CHARS:
            rejected.append(KeywordRejection(term=term[:_MAX_EXPANSION_TERM_CHARS], reason="term_too_long"))
            continue
        tokens = [token.casefold() for token in _TOKEN_RE.findall(term)]
        if not tokens or len(tokens) > _MAX_EXPANSION_TERM_TOKENS:
            rejected.append(KeywordRejection(term=term, reason="invalid_token_count"))
            continue
        key = term.casefold()
        if key in seen:
            rejected.append(KeywordRejection(term=term, reason="duplicate"))
            continue
        seen.add(key)
        if set(tokens).issubset(original_tokens):
            rejected.append(KeywordRejection(term=term, reason="no_novel_tokens"))
            continue
        accepted.append(term)

    if accepted:
        return KeywordExpansionResult(status="accepted", terms=tuple(accepted), rejected=tuple(rejected))
    if rejected:
        return KeywordExpansionResult(status="rejected", rejected=tuple(rejected), reason="all_terms_rejected")
    return KeywordExpansionResult(status="empty")


def _legacy_keyword_payload(raw: str, original_query: str) -> dict[str, list[str]] | None:
    """Parse only unambiguously comma-delimited legacy model output."""

    stripped = raw.strip()
    if "," not in stripped or stripped.startswith(("{", "[")) or any(unicodedata.category(char) == "Cc" for char in stripped):
        return None
    try:
        rows = list(csv_reader([stripped], skipinitialspace=True, strict=True))
    except CsvError:
        return None
    if len(rows) != 1 or not 2 <= len(rows[0]) <= _MAX_LEGACY_CSV_TERMS:
        return None
    original_tokens = {token.casefold() for token in _TOKEN_RE.findall(normalize_retrieval_query(original_query))}
    terms: list[str] = []
    for field in rows[0]:
        term = normalize_retrieval_query(field)
        raw_tokens = _TOKEN_RE.findall(term)
        tokens = [token.casefold() for token in raw_tokens]
        is_lowercase_keyword = len(tokens) <= 2 and all(token == token.casefold() for token in raw_tokens)
        is_redundant_entity = (
            len(tokens) <= 3
            and all(_has_entity_token_form(token) for token in raw_tokens)
            and set(tokens).issubset(original_tokens)
        )
        if (
            not term
            or len(term) > _MAX_EXPANSION_TERM_CHARS
            or not 1 <= len(tokens) <= _MAX_LEGACY_CSV_TERM_TOKENS
            or " ".join(_TOKEN_RE.findall(term)) != term
            or any(token in _STOPWORDS for token in tokens)
            or tokens[0] in _LEGACY_CONTROL_VERBS
            or not (is_lowercase_keyword or is_redundant_entity)
        ):
            return None
        terms.append(term)
    return {"keywords": terms}


@dataclass(frozen=True)
class RetrievalQueryBundle:
    """Explicit immutable inputs for each base-retrieval query role."""

    raw_original: str
    normalized_original: str
    effective_base: str
    cross_language_status: CrossLanguageStatus
    cross_language_identity: str | None
    exact_phrases: tuple[str, ...]
    original_sparse_query: str
    original_dense_query: str
    validated_terms: tuple[str, ...]
    expanded_sparse_query: str
    expansion_status: ExpansionStatus
    expansion_reason: str | None
    expansion_rejection_reasons: tuple[str, ...]

    @property
    def exact_phrase(self) -> str | None:
        return self.exact_phrases[0] if self.exact_phrases else None

    @property
    def expanded_terms(self) -> tuple[str, ...]:
        return self.validated_terms

    @classmethod
    def build(
        cls,
        original: str,
        *,
        effective_base: str | None = None,
        cross_language_status: CrossLanguageStatus = "disabled",
        cross_language_identity: str | None = None,
        expansion: KeywordExpansionResult | None = None,
    ) -> "RetrievalQueryBundle":
        expansion = expansion or KeywordExpansionResult(status="disabled")
        return build_retrieval_query_bundle(
            original,
            effective_base=effective_base,
            validated_terms=expansion.terms,
            cross_language_status=cross_language_status,
            cross_language_identity=cross_language_identity,
            expansion_status=expansion.status,
            expansion_reason=expansion.reason,
            expansion_rejection_reasons=(rejection.reason for rejection in expansion.rejected),
        )


def build_retrieval_query_bundle(
    original: str,
    *,
    effective_base: str | None = None,
    validated_terms: Iterable[str] = (),
    cross_language_status: CrossLanguageStatus = "disabled",
    cross_language_identity: str | None = None,
    expansion_status: ExpansionStatus = "disabled",
    expansion_reason: str | None = None,
    expansion_rejection_reasons: Iterable[str] = (),
) -> RetrievalQueryBundle:
    """Build immutable lane inputs without adding expansion to original lanes."""

    raw_original = str(original or "")
    normalized_original = normalize_retrieval_query(raw_original)
    effective_source = raw_original if effective_base is None else str(effective_base or "")
    normalized_effective = normalize_retrieval_query(effective_source)
    terms = _normalize_terms(validated_terms, normalized_effective)
    rejection_reasons = tuple(dict.fromkeys(str(reason) for reason in expansion_rejection_reasons if reason))[:8]
    expanded_sparse_query = ", ".join((normalized_effective, *terms)) if terms else normalized_effective
    return RetrievalQueryBundle(
        raw_original=raw_original,
        normalized_original=normalized_original,
        effective_base=normalized_effective,
        cross_language_status=cross_language_status,
        cross_language_identity=cross_language_identity,
        exact_phrases=eligible_exact_phrases(effective_source),
        original_sparse_query=normalized_effective,
        original_dense_query=normalized_effective,
        validated_terms=terms,
        expanded_sparse_query=expanded_sparse_query,
        expansion_status=expansion_status,
        expansion_reason=expansion_reason,
        expansion_rejection_reasons=rejection_reasons,
    )
