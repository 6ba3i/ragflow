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

"""Canonical Phase 5/6 structured-output contracts and strict decoding."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic_core import PydanticCustomError


STRUCTURED_OUTPUT_SCHEMA_VERSION = "2"
MAX_STRUCTURED_OUTPUT_CALLS = 2


class StructuredOutputMode(StrEnum):
    JSON_SCHEMA = "json_schema"
    TOOL_ARGUMENTS = "tool_arguments"
    TEXT_ONLY = "text_only"


@dataclass(frozen=True, slots=True)
class StructuredOutputResult:
    mode: StructuredOutputMode
    structured_payload: dict[str, Any] | None
    display_text: str | None
    used_tokens: int


class StructuredOutputFailureReason(StrEnum):
    EMPTY_RESPONSE = "empty_response"
    PROVIDER_STRUCTURED_PAYLOAD_MISSING = "provider_structured_payload_missing"
    UNEXPECTED_TEXT = "unexpected_text"
    JSON_OBJECT_NOT_FOUND = "json_object_not_found"
    JSON_DECODE_ERROR = "json_decode_error"
    TRUNCATED_OUTPUT = "truncated_output"
    UNKNOWN_FIELD = "unknown_field"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    WRONG_TYPE = "wrong_type"
    INVALID_ENUM = "invalid_enum"
    VALUE_OUT_OF_BOUNDS = "value_out_of_bounds"
    ARRAY_TOO_LONG = "array_too_long"
    DUPLICATE_ITEM = "duplicate_item"
    SEMANTIC_VALIDATION_FAILED = "semantic_validation_failed"
    UNSUPPORTED_ACTION = "unsupported_action"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    REPAIR_FAILED = "repair_failed"


class StructuredOutputError(ValueError):
    def __init__(
        self,
        reason: StructuredOutputFailureReason,
        message: str,
        *,
        location: tuple[str | int, ...] = (),
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.location = location


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


ShortText = Annotated[str, Field(min_length=1, max_length=96)]
Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,31}$")]
Description = Annotated[str, Field(min_length=1, max_length=240)]
QueryText = Annotated[str, Field(min_length=1, max_length=512)]
DocumentId = Annotated[str, Field(min_length=1, max_length=128)]
EvidenceId = Annotated[str, Field(min_length=1, max_length=128)]


def _normalized(value: str) -> str:
    return " ".join(value.split()).casefold()


def _require_unique(values: list[str], *, normalized: bool = False) -> None:
    keys = [_normalized(value) if normalized else value for value in values]
    if len(keys) != len(set(keys)):
        raise PydanticCustomError("duplicate_item", "items must be unique")


class PlannerFacetV2(_StrictModel):
    facet_id: Identifier
    description: Description
    anchors: Annotated[list[ShortText], Field(min_length=1, max_length=6)]
    evidence_type: Literal["definition", "numeric", "date", "comparison", "quote", "list_item"]

    @model_validator(mode="after")
    def validate_uniqueness(self) -> "PlannerFacetV2":
        _require_unique(self.anchors, normalized=True)
        return self


class PlannerSubqueryV2(_StrictModel):
    subquery_id: Identifier
    facet_id: Identifier
    query: QueryText
    keywords: QueryText | None
    docid_scope: Annotated[list[DocumentId], Field(min_length=1, max_length=64)] | None
    top_n: Annotated[int, Field(strict=True, ge=1, le=30)]
    retrieval_variant: Literal["hybrid_default", "keyword_first", "embedding_retry"]
    must_have_terms: Annotated[list[ShortText], Field(max_length=8)]
    forbidden_new_entities: Annotated[list[ShortText], Field(max_length=8)]
    rationale: Description

    @model_validator(mode="after")
    def validate_values(self) -> "PlannerSubqueryV2":
        if not self.query.strip():
            raise PydanticCustomError("semantic_validation_failed", "query must not be blank")
        if self.keywords is not None and not self.keywords.strip():
            raise PydanticCustomError("semantic_validation_failed", "keywords must not be blank")
        if self.docid_scope is not None:
            _require_unique(self.docid_scope)
        _require_unique(self.must_have_terms, normalized=True)
        _require_unique(self.forbidden_new_entities, normalized=True)
        return self


class PlannerMergePolicyV2(_StrictModel):
    strategy: Literal["subquery_rrf_then_context_builder"]
    max_chunks_per_facet: Annotated[int, Field(strict=True, ge=1, le=24)]


class PlannerDriftControlsV2(_StrictModel):
    anchor_entities: Annotated[list[ShortText], Field(max_length=16)]
    min_anchor_overlap: Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
    allow_new_entities: Literal[False]

    @model_validator(mode="after")
    def validate_uniqueness(self) -> "PlannerDriftControlsV2":
        _require_unique(self.anchor_entities, normalized=True)
        return self


class PlannerContentV2(_StrictModel):
    complexity: Literal["simple", "multi_facet", "multi_hop", "comparison", "temporal", "list", "unknown"]
    trigger_reasons: Annotated[list[ShortText], Field(max_length=8)]
    required_facets: Annotated[list[PlannerFacetV2], Field(min_length=1, max_length=4)]
    subqueries: Annotated[list[PlannerSubqueryV2], Field(min_length=1, max_length=4)]
    merge_policy: PlannerMergePolicyV2
    drift_controls: PlannerDriftControlsV2

    @model_validator(mode="after")
    def validate_plan(self) -> "PlannerContentV2":
        _require_unique(self.trigger_reasons, normalized=True)
        facet_ids = [facet.facet_id for facet in self.required_facets]
        _require_unique(facet_ids)
        subquery_ids = [subquery.subquery_id for subquery in self.subqueries]
        _require_unique(subquery_ids)
        _require_unique([subquery.query for subquery in self.subqueries], normalized=True)
        known_facets = set(facet_ids)
        if any(subquery.facet_id not in known_facets for subquery in self.subqueries):
            raise PydanticCustomError("semantic_validation_failed", "subquery references an unknown facet")
        return self


class CoveredFacetV2(_StrictModel):
    facet_id: Identifier
    evidence_ids: Annotated[list[EvidenceId], Field(min_length=1, max_length=20)]
    support: Literal["weak", "strong"]

    @model_validator(mode="after")
    def validate_uniqueness(self) -> "CoveredFacetV2":
        _require_unique(self.evidence_ids)
        return self


class MissingFacetV2(_StrictModel):
    facet_id: Identifier
    reason: Description
    required_anchors: Annotated[list[ShortText], Field(max_length=6)]

    @model_validator(mode="after")
    def validate_uniqueness(self) -> "MissingFacetV2":
        _require_unique(self.required_anchors, normalized=True)
        return self


class ContradictionV2(_StrictModel):
    facet_id: Identifier
    evidence_ids: Annotated[list[EvidenceId], Field(min_length=2, max_length=20)]
    description: Description

    @model_validator(mode="after")
    def validate_uniqueness(self) -> "ContradictionV2":
        _require_unique(self.evidence_ids)
        return self


class ExactFactGapV2(_StrictModel):
    type: Literal["date", "number", "name"]
    description: Description


class FollowupQueryV2(_StrictModel):
    facet_id: Identifier
    query: QueryText
    keywords: QueryText | None
    top_n: Annotated[int, Field(strict=True, ge=1, le=10)]

    @model_validator(mode="after")
    def validate_values(self) -> "FollowupQueryV2":
        if not self.query.strip():
            raise PydanticCustomError("semantic_validation_failed", "query must not be blank")
        if self.keywords is not None and not self.keywords.strip():
            raise PydanticCustomError("semantic_validation_failed", "keywords must not be blank")
        return self


class SufficiencyDecisionV2(_StrictModel):
    action: Literal["sufficient", "refine"]
    confidence: Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
    covered_facets: Annotated[list[CoveredFacetV2], Field(max_length=4)]
    missing_facets: Annotated[list[MissingFacetV2], Field(max_length=4)]
    contradictions: Annotated[list[ContradictionV2], Field(max_length=4)]
    exact_fact_gaps: Annotated[list[ExactFactGapV2], Field(max_length=4)]
    refusal_justified: Annotated[bool, Field(strict=True)]
    recommended_followups: Annotated[list[FollowupQueryV2], Field(max_length=4)]

    @model_validator(mode="after")
    def validate_decision(self) -> "SufficiencyDecisionV2":
        _require_unique([item.facet_id for item in self.covered_facets])
        _require_unique([item.facet_id for item in self.missing_facets])
        _require_unique([item.query for item in self.recommended_followups], normalized=True)
        missing = {item.facet_id for item in self.missing_facets}
        if self.action == "sufficient" and (self.missing_facets or self.contradictions or self.exact_fact_gaps or self.recommended_followups):
            raise PydanticCustomError("semantic_validation_failed", "sufficient action cannot contain unresolved work")
        if self.action == "refine" and any(item.facet_id not in missing for item in self.recommended_followups):
            raise PydanticCustomError("semantic_validation_failed", "followups must target missing facets")
        return self


_TEMPLATES: dict[type[BaseModel], dict[str, Any]] = {
    PlannerContentV2: {
        "complexity": "unknown",
        "trigger_reasons": [],
        "required_facets": [
            {
                "facet_id": "primary",
                "description": "Primary fact required to answer the question",
                "anchors": ["primary"],
                "evidence_type": "definition",
            }
        ],
        "subqueries": [
            {
                "subquery_id": "primary_query",
                "facet_id": "primary",
                "query": "Primary fact required to answer the question",
                "keywords": None,
                "docid_scope": None,
                "top_n": 5,
                "retrieval_variant": "hybrid_default",
                "must_have_terms": [],
                "forbidden_new_entities": [],
                "rationale": "Retrieve evidence for the primary facet.",
            }
        ],
        "merge_policy": {"strategy": "subquery_rrf_then_context_builder", "max_chunks_per_facet": 4},
        "drift_controls": {"anchor_entities": [], "min_anchor_overlap": 0.0, "allow_new_entities": False},
    },
    SufficiencyDecisionV2: {
        "action": "sufficient",
        "confidence": 1.0,
        "covered_facets": [],
        "missing_facets": [],
        "contradictions": [],
        "exact_fact_gaps": [],
        "refusal_justified": False,
        "recommended_followups": [],
    },
}


ModelT = TypeVar("ModelT", bound=BaseModel)


def canonical_json_schema(model: type[ModelT]) -> dict[str, Any]:
    return deepcopy(model.model_json_schema(mode="validation"))


def canonical_schema_fingerprint(model: type[ModelT]) -> str:
    payload = json.dumps(
        {
            "schema": canonical_json_schema(model),
            "schema_version": STRUCTURED_OUTPUT_SCHEMA_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_json_template(model: type[ModelT]) -> str:
    try:
        template = deepcopy(_TEMPLATES[model])
    except KeyError as exc:
        raise TypeError(f"No canonical template registered for {model.__name__}") from exc
    model.model_validate(template)
    return json.dumps(template, ensure_ascii=False, indent=2)


def canonical_output_instructions(model: type[ModelT]) -> str:
    return (
        "Return exactly one JSON object matching the supplied schema.\n"
        "Use exactly the field names shown. Do not rename fields. Do not add fields. "
        "Do not remove required fields.\n"
        "Do not wrap the object in markdown. Do not include prose before or after it.\n"
        "Use the listed enum values exactly. Use empty arrays when no items apply. "
        "Use null only where the schema explicitly permits null.\n"
        f"Canonical fillable object:\n{canonical_json_template(model)}"
    )


def _remove_trailing_commas(text: str) -> str:
    output: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue
        if char == ",":
            cursor = index + 1
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor < len(text) and text[cursor] in "}]":
                index += 1
                continue
        output.append(char)
        index += 1
    return "".join(output)


def _balanced_json_objects(text: str) -> tuple[list[str], bool]:
    objects: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : index + 1])
                start = None
    return objects, depth > 0


def _reason_from_validation(error: dict[str, Any]) -> StructuredOutputFailureReason:
    error_type = str(error.get("type", ""))
    message = str(error.get("msg", ""))
    value = error.get("input")
    location = tuple(error.get("loc", ()))
    if error_type == "extra_forbidden":
        return StructuredOutputFailureReason.UNKNOWN_FIELD
    if error_type == "missing":
        return StructuredOutputFailureReason.MISSING_REQUIRED_FIELD
    if error_type == "duplicate_item" or "duplicate_item" in message:
        return StructuredOutputFailureReason.DUPLICATE_ITEM
    if error_type == "semantic_validation_failed" or "semantic_validation_failed" in message:
        return StructuredOutputFailureReason.SEMANTIC_VALIDATION_FAILED
    if error_type == "literal_error":
        if not isinstance(value, str):
            return StructuredOutputFailureReason.WRONG_TYPE
        if location and location[-1] == "action":
            return StructuredOutputFailureReason.UNSUPPORTED_ACTION
        return StructuredOutputFailureReason.INVALID_ENUM
    if error_type in {"too_long", "list_too_long"} and isinstance(value, list):
        return StructuredOutputFailureReason.ARRAY_TOO_LONG
    if error_type in {
        "greater_than",
        "greater_than_equal",
        "less_than",
        "less_than_equal",
        "string_too_long",
        "string_too_short",
        "string_pattern_mismatch",
        "too_short",
    }:
        return StructuredOutputFailureReason.VALUE_OUT_OF_BOUNDS
    if error_type.endswith("_type") or error_type in {"int_parsing", "float_parsing", "bool_parsing", "string_unicode"}:
        return StructuredOutputFailureReason.WRONG_TYPE
    return StructuredOutputFailureReason.SEMANTIC_VALIDATION_FAILED


def _validate(model: type[ModelT], payload: Any) -> ModelT:
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        first = exc.errors(include_url=False)[0]
        reason = _reason_from_validation(first)
        raise StructuredOutputError(
            reason,
            first.get("msg", "structured output validation failed"),
            location=tuple(first.get("loc", ())),
        ) from exc


def decode_structured_output(
    model: type[ModelT],
    *,
    structured_payload: dict[str, Any] | None = None,
    display_text: str | None = None,
    require_structured_payload: bool = False,
) -> ModelT:
    if structured_payload is not None:
        return _validate(model, structured_payload)
    if require_structured_payload:
        raise StructuredOutputError(
            StructuredOutputFailureReason.PROVIDER_STRUCTURED_PAYLOAD_MISSING,
            "provider did not return a structured payload",
        )
    if display_text is None or not display_text.strip():
        raise StructuredOutputError(StructuredOutputFailureReason.EMPTY_RESPONSE, "structured output response is empty")

    text = display_text.strip()
    try:
        direct = json.loads(_remove_trailing_commas(text))
    except json.JSONDecodeError:
        direct = None
    if isinstance(direct, dict):
        return _validate(model, direct)

    objects, truncated = _balanced_json_objects(text)
    if len(objects) > 1:
        raise StructuredOutputError(StructuredOutputFailureReason.UNEXPECTED_TEXT, "response contains multiple JSON objects")
    if not objects:
        if truncated:
            raise StructuredOutputError(StructuredOutputFailureReason.TRUNCATED_OUTPUT, "JSON object is truncated")
        if "{" not in text:
            raise StructuredOutputError(StructuredOutputFailureReason.JSON_OBJECT_NOT_FOUND, "response contains no JSON object")
        raise StructuredOutputError(StructuredOutputFailureReason.JSON_DECODE_ERROR, "JSON object could not be decoded")

    candidate = _remove_trailing_commas(objects[0])
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(StructuredOutputFailureReason.JSON_DECODE_ERROR, "JSON object could not be decoded") from exc
    if not isinstance(payload, dict):
        raise StructuredOutputError(StructuredOutputFailureReason.JSON_OBJECT_NOT_FOUND, "structured output must be a JSON object")
    return _validate(model, payload)
