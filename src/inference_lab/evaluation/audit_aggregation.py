"""Pure deterministic aggregation of one validated corpus-audit request."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil

from inference_lab.evaluation.audit_capabilities import ParserCapabilityProfile
from inference_lab.evaluation.audit_contracts import (
    AcquisitionStatus,
    AuditCoverage,
    AuditDistribution,
    AuditMissingReason,
    AuditParsedDocumentFacts,
    AuditParseFailure,
    AuditParseObservation,
    AuditRate,
    AuditSourceFailure,
    AuditSourceFormatCoverage,
    AuditStructureKind,
    CorpusAuditRequest,
)
from inference_lab.ingest.discovery import SourceSupport
from inference_lab.provenance import DocumentFamilyMembership, DuplicateRelationship

_LANGUAGE_TAG = re.compile(r"[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*")
_QUANTILES: tuple[tuple[str, int], ...] = (
    ("p50", 50),
    ("p90", 90),
    ("p95", 95),
    ("p99", 99),
)
_FIELD_NAMES: tuple[str, ...] = (
    "language",
    "page_count",
    "character_count",
    "token_count",
)


@dataclass(frozen=True, slots=True)
class _IntegerSummary:
    """Summarise exact non-negative integer observations."""

    distribution: AuditDistribution
    quantiles: tuple[tuple[str, int | None], ...]


@dataclass(frozen=True, slots=True)
class _CoverageSummary:
    """Pair reconciled coverage with deterministic reason counts."""

    coverage: AuditCoverage
    reasons: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class _SourceAggregates:
    files_seen: int
    classified_files: int
    traversal_failures: int
    formats: tuple[AuditSourceFormatCoverage, ...]
    size_summary: _IntegerSummary
    failure_counts: tuple[tuple[str, int], ...]
    reconciles: bool


@dataclass(frozen=True, slots=True)
class _ParseAggregates:
    status_counts: tuple[tuple[str, int], ...]
    parser_counts: tuple[tuple[str, int], ...]
    media_type_counts: tuple[tuple[str, int], ...]
    failure_counts: tuple[tuple[str, int], ...]
    field_coverage: tuple[tuple[str, _CoverageSummary], ...]
    language_counts: tuple[tuple[str, int], ...]
    numeric_summaries: tuple[tuple[str, _IntegerSummary], ...]
    token_summaries: tuple[tuple[str, _IntegerSummary], ...]


@dataclass(frozen=True, slots=True)
class _AcquisitionAggregates:
    status_counts: tuple[tuple[str, int], ...]
    coverage: _CoverageSummary
    known_only_rate: AuditRate
    population_rate: AuditRate


@dataclass(frozen=True, slots=True)
class _StructureAggregate:
    kind: AuditStructureKind
    coverage: _CoverageSummary
    present_documents: int
    absent_documents: int
    unit_summary: _IntegerSummary


@dataclass(frozen=True, slots=True)
class _SpanAggregates:
    block_summary: _IntegerSummary
    source_span_count: int
    page_number_count: int
    bounding_box_count: int
    document_coverage: _CoverageSummary


@dataclass(frozen=True, slots=True)
class _OcrAggregates:
    coverage: _CoverageSummary
    zero_text_page_count: int
    page_character_summary: _IntegerSummary
    mixed_text_documents: int
    pages_with_layout: int
    pages_without_layout: int
    pages_with_boxes: int
    pages_without_boxes: int


@dataclass(frozen=True, slots=True)
class _RelationshipAggregates:
    duplicate_edge_count: int
    duplicate_participating_members: int
    duplicate_group_count: int
    duplicate_group_summary: _IntegerSummary
    duplicate_excess_copies: int
    duplicate_participating_rate: AuditRate
    duplicate_excess_rate: AuditRate
    family_count: int
    family_document_count: int
    family_version_count: int
    unassigned_document_count: int
    family_size_summary: _IntegerSummary
    document_level_memberships: int
    version_level_memberships: int
    family_candidate_count: int
    family_candidate_family_count: int


@dataclass(frozen=True, slots=True)
class _CorpusAuditAggregates:
    """Private complete aggregate bundle consumed by the report PR."""

    population: int
    source: _SourceAggregates
    parse: _ParseAggregates
    acquisition: _AcquisitionAggregates
    structures: tuple[_StructureAggregate, ...]
    spans: _SpanAggregates
    ocr: _OcrAggregates
    relationships: _RelationshipAggregates
    limitations: tuple[str, ...]


def aggregate_corpus_audit(request: CorpusAuditRequest) -> _CorpusAuditAggregates:
    """Calculate all currently specified audit aggregates without side effects."""
    if not isinstance(request, CorpusAuditRequest):
        raise TypeError("request must be a CorpusAuditRequest.")
    profiles = {
        profile.parser_version.value: profile for profile in request.capability_profiles
    }
    source = _aggregate_source(request)
    parse = _aggregate_parse(request, profiles)
    acquisition = _aggregate_acquisition(request)
    structures = _aggregate_structures(request, profiles)
    spans = _aggregate_spans(request, profiles)
    ocr = _aggregate_ocr(request, profiles)
    relationships = _aggregate_relationships(request)
    limitations = _limitations(request, parse, acquisition, structures, ocr)
    return _CorpusAuditAggregates(
        population=len(request.scope.memberships),
        source=source,
        parse=parse,
        acquisition=acquisition,
        structures=structures,
        spans=spans,
        ocr=ocr,
        relationships=relationships,
        limitations=limitations,
    )


def _aggregate_source(request: CorpusAuditRequest) -> _SourceAggregates:
    inventory = request.source_inventory
    classified_files = sum(item.file_count for item in inventory.format_coverage)
    traversal_failures = inventory.discovery_counts.files_seen - classified_files
    failure_counts = _counts(_source_failure_key(item) for item in inventory.failures)
    return _SourceAggregates(
        files_seen=inventory.discovery_counts.files_seen,
        classified_files=classified_files,
        traversal_failures=traversal_failures,
        formats=inventory.format_coverage,
        size_summary=_integer_summary(inventory.known_source_byte_counts),
        failure_counts=failure_counts,
        reconciles=traversal_failures
        == sum(
            item.category.value
            not in {
                "recognised_unsupported_format",
                "unknown_format",
            }
            and item.stage.value == "traversal"
            for item in inventory.failures
        ),
    )


def _aggregate_parse(
    request: CorpusAuditRequest,
    profiles: dict[str, ParserCapabilityProfile],
) -> _ParseAggregates:
    observations = request.parse_observations
    status_counts = _counts(
        (observation.evidence.status if observation.evidence is not None else "missing")
        for observation in observations
    )
    parser_counts = _counts(
        observation.parser_version.value
        for observation in observations
        if observation.parser_version is not None
    )
    media_type_counts = _counts(
        observation.media_type
        for observation in observations
        if observation.media_type is not None
    )
    failure_counts = _counts(
        _parse_failure_key(failure)
        for observation in observations
        if observation.evidence is not None
        for failure in observation.evidence.failures
    )
    field_coverage = tuple(
        (field_name, _aggregate_field_coverage(observations, profiles, field_name))
        for field_name in _FIELD_NAMES
    )
    language_counts = _counts(
        document.language
        for observation in observations
        for document in _documents(observation)
        if document.language is not None and _valid_language(document.language)
    )
    numeric_summaries = tuple(
        (
            field_name,
            _integer_summary(
                value
                for observation in observations
                for document in _documents(observation)
                for value in (_numeric_value(document, field_name),)
                if value is not None
                and _field_is_capable(observation, profiles, field_name)
            ),
        )
        for field_name in ("page_count", "character_count", "token_count")
    )
    token_values: dict[str, list[int]] = {}
    for observation in observations:
        if observation.parser_version is None or not _field_is_capable(
            observation, profiles, "token_count"
        ):
            continue
        for document in _documents(observation):
            if document.token_count is not None:
                token_values.setdefault(observation.parser_version.value, []).append(
                    document.token_count
                )
    token_summaries = tuple(
        (key, _integer_summary(values)) for key, values in sorted(token_values.items())
    )
    return _ParseAggregates(
        status_counts=status_counts,
        parser_counts=parser_counts,
        media_type_counts=media_type_counts,
        failure_counts=failure_counts,
        field_coverage=field_coverage,
        language_counts=language_counts,
        numeric_summaries=numeric_summaries,
        token_summaries=token_summaries,
    )


def _aggregate_field_coverage(
    observations: tuple[AuditParseObservation, ...],
    profiles: dict[str, ParserCapabilityProfile],
    field_name: str,
) -> _CoverageSummary:
    states: list[str] = []
    for observation in observations:
        state = _field_state(observation, profiles, field_name)
        states.append(state)
    return _coverage_summary(states)


def _field_state(
    observation: AuditParseObservation,
    profiles: dict[str, ParserCapabilityProfile],
    field_name: str,
) -> str:
    if observation.evidence is None:
        return _missing_state(observation.missing_reason)
    if observation.evidence.status == "failed" or observation.evidence.document is None:
        return "failed"
    if not _field_is_capable(observation, profiles, field_name):
        return "unavailable"
    value = _numeric_or_language_value(observation.evidence.document, field_name)
    if field_name == "language":
        if value is not None and (
            not isinstance(value, str) or not _valid_language(value)
        ):
            return "invalid"
        return "observed" if value is not None else "observed_absent"
    return "observed" if value is not None else "observed_absent"


def _field_is_capable(
    observation: AuditParseObservation,
    profiles: dict[str, ParserCapabilityProfile],
    field_name: str,
) -> bool:
    if observation.parser_version is None:
        return False
    profile = _profile_for(observation, profiles)
    if field_name == "language":
        return profile.language
    if field_name == "page_count":
        return profile.page_count
    if field_name == "character_count":
        return profile.character_count
    if field_name == "token_count":
        return profile.token_count
    raise ValueError("unsupported aggregate field.")


def _profile_for(
    observation: AuditParseObservation,
    profiles: dict[str, ParserCapabilityProfile],
) -> ParserCapabilityProfile:
    if observation.parser_version is None:
        raise ValueError("a parser profile is required for observed evidence.")
    return profiles[observation.parser_version.value]


def _numeric_or_language_value(
    document: AuditParsedDocumentFacts,
    field_name: str,
) -> int | str | None:
    if field_name == "language":
        return document.language
    return _numeric_value(document, field_name)


def _numeric_value(document: AuditParsedDocumentFacts, field_name: str) -> int | None:
    return getattr(document, field_name)


def _aggregate_acquisition(request: CorpusAuditRequest) -> _AcquisitionAggregates:
    by_version = {
        item.membership.document_version_id: item
        for item in request.acquisition_observations
    }
    statuses = (
        by_version.get(membership.document_version_id)
        for membership in request.scope.memberships
    )
    status_values = tuple(
        item.status.value if item is not None else "missing" for item in statuses
    )
    known = sum(
        value
        in {
            AcquisitionStatus.TEXT_NATIVE.value,
            AcquisitionStatus.SCANNED.value,
            AcquisitionStatus.MIXED.value,
        }
        for value in status_values
    )
    assessed = sum(value != "missing" for value in status_values)
    coverage = _coverage_summary(
        "observed"
        if value
        in {
            AcquisitionStatus.TEXT_NATIVE.value,
            AcquisitionStatus.SCANNED.value,
            AcquisitionStatus.MIXED.value,
        }
        else "unavailable"
        if value == AcquisitionStatus.UNKNOWN.value
        else "missing"
        for value in status_values
    )
    return _AcquisitionAggregates(
        status_counts=_counts(status_values),
        coverage=coverage,
        known_only_rate=AuditRate(known, assessed),
        population_rate=AuditRate(assessed, len(status_values)),
    )


def _aggregate_structures(
    request: CorpusAuditRequest,
    profiles: dict[str, ParserCapabilityProfile],
) -> tuple[_StructureAggregate, ...]:
    result: list[_StructureAggregate] = []
    for kind in AuditStructureKind:
        states: list[str] = []
        counts: list[int] = []
        for observation in request.parse_observations:
            if observation.evidence is None:
                states.append(_missing_state(observation.missing_reason))
                continue
            if (
                observation.evidence.status == "failed"
                or observation.evidence.document is None
            ):
                states.append("failed")
                continue
            profile = _profile_for(observation, profiles)
            if kind.value not in profile.structure_kinds:
                states.append("unavailable")
                continue
            count = _structure_count(observation.evidence.document, kind)
            states.append("observed" if count else "observed_absent")
            counts.append(count)
        result.append(
            _StructureAggregate(
                kind=kind,
                coverage=_coverage_summary(states),
                present_documents=sum(value > 0 for value in counts),
                absent_documents=sum(value == 0 for value in counts),
                unit_summary=_integer_summary(counts),
            )
        )
    return tuple(result)


def _aggregate_spans(
    request: CorpusAuditRequest,
    profiles: dict[str, ParserCapabilityProfile],
) -> _SpanAggregates:
    documents = tuple(
        (observation, document)
        for observation in request.parse_observations
        for document in _documents(observation)
    )
    capable = tuple(
        document
        for observation, document in documents
        if observation.parser_version is not None
        and _profile_for(observation, profiles).source_spans
    )
    states = [
        _span_state(observation, profiles) for observation in request.parse_observations
    ]
    return _SpanAggregates(
        block_summary=_integer_summary(item.source_span_count for item in capable),
        source_span_count=sum(item.source_span_count for item in capable),
        page_number_count=sum(item.page_number_count for item in capable),
        bounding_box_count=sum(item.bounding_box_count for item in capable),
        document_coverage=_coverage_summary(states),
    )


def _span_state(
    observation: AuditParseObservation,
    profiles: dict[str, ParserCapabilityProfile],
) -> str:
    if observation.evidence is None:
        return _missing_state(observation.missing_reason)
    if observation.evidence.status == "failed" or observation.evidence.document is None:
        return "failed"
    return (
        "observed"
        if _profile_for(observation, profiles).source_spans
        else "unavailable"
    )


def _aggregate_ocr(
    request: CorpusAuditRequest,
    profiles: dict[str, ParserCapabilityProfile],
) -> _OcrAggregates:
    page_counts: list[int] = []
    capable_documents = 0
    mixed_documents = 0
    pages_with_layout = 0
    pages_with_boxes = 0
    for observation in request.parse_observations:
        for document in _documents(observation):
            if (
                observation.parser_version is None
                or not _profile_for(observation, profiles).page_text
                or document.page_character_counts is None
            ):
                continue
            capable_documents += 1
            counts = document.page_character_counts
            page_counts.extend(counts)
            pages_with_layout += min(document.page_layout_page_count, len(counts))
            pages_with_boxes += min(document.bounding_box_page_count, len(counts))
            if any(value == 0 for value in counts) and any(
                value > 0 for value in counts
            ):
                mixed_documents += 1
    states = [
        "observed"
        if _has_page_text(observation, profiles)
        else _field_state(observation, profiles, "page_count")
        for observation in request.parse_observations
    ]
    page_total = len(page_counts)
    return _OcrAggregates(
        coverage=_coverage_summary(states),
        zero_text_page_count=sum(value == 0 for value in page_counts),
        page_character_summary=_integer_summary(page_counts),
        mixed_text_documents=mixed_documents,
        pages_with_layout=pages_with_layout,
        pages_without_layout=(
            page_total - pages_with_layout if capable_documents else 0
        ),
        pages_with_boxes=pages_with_boxes,
        pages_without_boxes=(page_total - pages_with_boxes if capable_documents else 0),
    )


def _has_page_text(
    observation: AuditParseObservation,
    profiles: dict[str, ParserCapabilityProfile],
) -> bool:
    return (
        observation.evidence is not None
        and observation.evidence.document is not None
        and observation.parser_version is not None
        and _profile_for(observation, profiles).page_text
        and observation.evidence.document.page_character_counts is not None
    )


def _aggregate_relationships(request: CorpusAuditRequest) -> _RelationshipAggregates:
    relationships = request.relationships
    edges = relationships.duplicate_relationships
    components = _duplicate_components(edges)
    group_sizes = tuple(len(component) for component in components)
    participating = len(
        {
            endpoint
            for edge in edges
            for endpoint in (
                edge.first_version_id.as_str(),
                edge.second_version_id.as_str(),
            )
        }
    )
    excess = sum(size - 1 for size in group_sizes)
    family_sizes = _counts_by_family(relationships.family_memberships)
    assigned_documents = {
        item.document_id.as_str() for item in relationships.family_memberships
    }
    return _RelationshipAggregates(
        duplicate_edge_count=len(edges),
        duplicate_participating_members=participating,
        duplicate_group_count=len(components),
        duplicate_group_summary=_integer_summary(group_sizes),
        duplicate_excess_copies=excess,
        duplicate_participating_rate=AuditRate(
            participating, len(request.scope.memberships)
        ),
        duplicate_excess_rate=AuditRate(excess, len(request.scope.memberships)),
        family_count=len(family_sizes),
        family_document_count=len(assigned_documents),
        family_version_count=sum(
            item.document_version_id is not None
            for item in relationships.family_memberships
        ),
        unassigned_document_count=len(request.scope.memberships)
        - len(assigned_documents),
        family_size_summary=_integer_summary(tuple(family_sizes.values())),
        document_level_memberships=sum(
            item.document_version_id is None
            for item in relationships.family_memberships
        ),
        version_level_memberships=sum(
            item.document_version_id is not None
            for item in relationships.family_memberships
        ),
        family_candidate_count=len(relationships.family_candidates),
        family_candidate_family_count=len(
            {
                item.document_family_id.as_str()
                for item in relationships.family_candidates
            }
        ),
    )


def _duplicate_components(
    edges: tuple[DuplicateRelationship, ...],
) -> tuple[frozenset[str], ...]:
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        first = edge.first_version_id.as_str()
        second = edge.second_version_id.as_str()
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)
    components: list[frozenset[str]] = []
    unseen = set(adjacency)
    while unseen:
        root = min(unseen)
        stack = [root]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            unseen.discard(current)
            stack.extend(adjacency[current] - component)
        components.append(frozenset(component))
    return tuple(sorted(components, key=lambda item: min(item)))


def _counts_by_family(
    memberships: tuple[DocumentFamilyMembership, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for membership in memberships:
        family_id = membership.family_id.as_str()
        counts[family_id] = counts.get(family_id, 0) + 1
    return counts


def _limitations(
    request: CorpusAuditRequest,
    parse: _ParseAggregates,
    acquisition: _AcquisitionAggregates,
    structures: tuple[_StructureAggregate, ...],
    ocr: _OcrAggregates,
) -> tuple[str, ...]:
    limitations: set[str] = set()
    if any(
        item.support is not SourceSupport.SUPPORTED
        for item in request.source_inventory.format_coverage
    ):
        limitations.add("source_format_not_supported")
    if request.source_inventory.failures:
        limitations.add("source_upstream_failures")
    if any(
        summary.coverage.unavailable
        or summary.coverage.missing
        or summary.coverage.failed
        for _, summary in parse.field_coverage
    ):
        limitations.add("parse_field_coverage_incomplete")
    if any(item.coverage.coverage.unavailable for item in structures):
        limitations.add("structure_capability_unavailable")
    if (
        ocr.coverage.coverage.unavailable
        or ocr.coverage.coverage.missing
        or ocr.coverage.coverage.failed
    ):
        limitations.add("ocr_indicators_unavailable")
    if (
        acquisition.coverage.coverage.unavailable
        or acquisition.coverage.coverage.missing
    ):
        limitations.add("acquisition_status_incomplete")
    if any(
        observation.evidence is not None
        and observation.evidence.document is not None
        and observation.evidence.document.token_count is not None
        for observation in request.parse_observations
    ):
        limitations.add("token_semantics_are_parser_versioned")
    if request.relationships.family_candidates:
        limitations.add("family_candidates_are_advisory")
    return tuple(sorted(limitations))


def _documents(
    observation: AuditParseObservation,
) -> tuple[AuditParsedDocumentFacts, ...]:
    return (
        (observation.evidence.document,)
        if observation.evidence is not None
        and observation.evidence.document is not None
        else ()
    )


def _structure_count(
    document: AuditParsedDocumentFacts, kind: AuditStructureKind
) -> int:
    return next(
        (item.count for item in document.structure_counts if item.kind is kind),
        0,
    )


def _missing_state(reason: AuditMissingReason | None) -> str:
    if reason in {
        AuditMissingReason.PARSER_FIELD_UNAVAILABLE,
        AuditMissingReason.UNSUPPORTED_FORMAT,
    }:
        return "unavailable"
    if reason in {AuditMissingReason.PARSE_FAILED, AuditMissingReason.SOURCE_FAILURE}:
        return "failed"
    return "missing"


def _coverage_summary(states: Iterable[str]) -> _CoverageSummary:
    state_values = tuple(states)
    counts = {
        name: state_values.count(name)
        for name in (
            "observed",
            "observed_absent",
            "unavailable",
            "missing",
            "failed",
            "invalid",
        )
    }
    return _CoverageSummary(
        coverage=AuditCoverage(
            observed=counts["observed"],
            observed_absent=counts["observed_absent"],
            unavailable=counts["unavailable"],
            missing=counts["missing"],
            failed=counts["failed"],
            invalid=counts["invalid"],
            population=len(state_values),
        ),
        reasons=tuple((key, counts[key]) for key in sorted(counts) if counts[key]),
    )


def _integer_summary(values: Iterable[int]) -> _IntegerSummary:
    numbers = tuple(values)
    ordered = tuple(sorted(numbers))
    distribution = AuditDistribution(
        count=len(ordered),
        total=sum(ordered),
        minimum=ordered[0] if ordered else None,
        maximum=ordered[-1] if ordered else None,
        zero_count=ordered.count(0),
    )
    quantiles = tuple(
        (label, ordered[ceil(percentile * len(ordered) / 100) - 1] if ordered else None)
        for label, percentile in _QUANTILES
    )
    return _IntegerSummary(distribution=distribution, quantiles=quantiles)


def _counts(values: Iterable[object]) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return tuple(sorted(counts.items()))


def _source_failure_key(failure: AuditSourceFailure) -> str:
    return "|".join(
        (
            failure.category.value,
            failure.scope.value,
            failure.stage.value,
            str(failure.retryable).lower(),
        )
    )


def _parse_failure_key(failure: AuditParseFailure) -> str:
    return "|".join(
        (
            failure.category,
            failure.severity,
            failure.stage,
            failure.unit_kind or "none",
        )
    )


def _valid_language(value: str) -> bool:
    return _LANGUAGE_TAG.fullmatch(value) is not None
