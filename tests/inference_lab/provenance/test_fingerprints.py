"""Tests for deterministic derived-record fingerprint construction."""

from collections.abc import Callable

import pytest

from inference_lab.provenance.derived import (
    DerivedRecordKind,
    OutputDigest,
    build_derivation_fingerprint,
)
from inference_lab.provenance.identifiers import (
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.provenance.lineage import (
    InputRole,
    LineageInputs,
    ParentRecordReference,
    SourceAnchor,
)
from inference_lab.provenance.operations import (
    ComponentRole,
    ConfigurationIdentity,
    DigestAlgorithm,
    OperationKind,
    ProducingComponent,
    ProducingOperation,
)
from inference_lab.provenance.versioning import CURRENT_LINEAGE_SCHEMA_VERSION

_DOCUMENT_VERSION_A = DocumentVersionId.from_str("11111111-1111-4111-8111-111111111111")
_DOCUMENT_VERSION_B = DocumentVersionId.from_str("22222222-2222-4222-8222-222222222222")
_SOURCE_SPAN_A = SourceSpanId.from_str("33333333-3333-4333-8333-333333333333")
_SOURCE_SPAN_B = SourceSpanId.from_str("44444444-4444-4444-8444-444444444444")
_PARENT_A = RecordId.from_str("55555555-5555-4555-8555-555555555555")
_PARENT_B = RecordId.from_str("66666666-6666-4666-8666-666666666666")
_CONFIGURATION_DIGEST = "a" * 64
_OUTPUT_DIGEST = "b" * 64
_COMPONENT_ROLES = (
    "generator-model",
    "grouping",
    "normaliser",
    "parser",
    "prompt",
)


def _source_a() -> SourceAnchor:
    return SourceAnchor(
        document_version_id=_DOCUMENT_VERSION_A,
        source_span_id=_SOURCE_SPAN_A,
        input_role=InputRole("primary-source"),
        input_ordinal=0,
    )


def _source_b() -> SourceAnchor:
    return SourceAnchor(
        document_version_id=_DOCUMENT_VERSION_B,
        source_span_id=_SOURCE_SPAN_B,
        input_role=InputRole("secondary-source"),
        input_ordinal=1,
    )


def _inputs(
    *,
    sources: tuple[SourceAnchor, ...] | None = None,
    parent_record_id: RecordId = _PARENT_A,
) -> LineageInputs:
    return LineageInputs(
        source_anchors=sources or (_source_a(), _source_b()),
        parent_records=(
            ParentRecordReference(
                record_id=parent_record_id,
                input_role=InputRole("prior-summary"),
            ),
        ),
    )


def _components(
    *,
    changed_role: str | None = None,
    model_revision: str = "revision-a",
) -> tuple[ProducingComponent, ...]:
    components = []
    for role in _COMPONENT_ROLES:
        is_model = role == "generator-model"
        components.append(
            ProducingComponent(
                role=ComponentRole(role),
                implementation=f"{role}-implementation",
                contract_version="1.0.1" if role == changed_role else "1.0.0",
                artefact_identity="organisation/model" if is_model else None,
                artefact_revision=model_revision if is_model else None,
            )
        )
    return tuple(components)


def _operation(
    *,
    kind: str = "summarisation",
    implementation: str = "topic-summariser",
    contract_version: str = "1.0.0",
    configuration_digest: str = _CONFIGURATION_DIGEST,
    components: tuple[ProducingComponent, ...] | None = None,
) -> ProducingOperation:
    return ProducingOperation(
        kind=OperationKind(kind),
        implementation=implementation,
        contract_version=contract_version,
        configuration_identity=ConfigurationIdentity(
            algorithm=DigestAlgorithm.SHA256,
            digest=configuration_digest,
        ),
        components=components or _components(),
    )


def _fingerprint(
    *,
    record_kind: str = "topic-summary",
    inputs: LineageInputs | None = None,
    operation: ProducingOperation | None = None,
    output_digest: str = _OUTPUT_DIGEST,
):
    return build_derivation_fingerprint(
        record_kind=DerivedRecordKind(record_kind),
        lineage_schema_version=CURRENT_LINEAGE_SCHEMA_VERSION,
        inputs=inputs or _inputs(),
        operation=operation or _operation(),
        output_digest=OutputDigest(DigestAlgorithm.SHA256, output_digest),
    )


def test_repeated_derivation_produces_byte_stable_fingerprint() -> None:
    first = _fingerprint()
    second = _fingerprint()

    assert first == second
    assert first.algorithm is DigestAlgorithm.SHA256
    assert first.digest == (
        "7c834ed7f7060dd2c4ec3c6144db1f05dfa69d597f37525643e8b4abb4822dde"
    )


def test_fingerprint_is_independent_of_caller_input_order() -> None:
    forward = _inputs(sources=(_source_a(), _source_b()))
    reverse = _inputs(sources=(_source_b(), _source_a()))

    assert _fingerprint(inputs=forward) == _fingerprint(inputs=reverse)


@pytest.mark.parametrize(
    "changed_inputs",
    [
        lambda: _inputs(
            sources=(
                SourceAnchor(
                    document_version_id=_DOCUMENT_VERSION_B,
                    source_span_id=_SOURCE_SPAN_A,
                    input_role=InputRole("primary-source"),
                    input_ordinal=0,
                ),
                _source_b(),
            )
        ),
        lambda: _inputs(parent_record_id=_PARENT_B),
        lambda: _inputs(
            sources=(
                SourceAnchor(
                    document_version_id=_DOCUMENT_VERSION_A,
                    source_span_id=_SOURCE_SPAN_A,
                    input_role=InputRole("changed-role"),
                    input_ordinal=0,
                ),
                _source_b(),
            )
        ),
    ],
    ids=["source", "parent", "role"],
)
def test_changed_input_metadata_changes_fingerprint(
    changed_inputs: Callable[[], LineageInputs],
) -> None:
    assert _fingerprint(inputs=changed_inputs()) != _fingerprint()


@pytest.mark.parametrize(
    "changed_operation",
    [
        lambda: _operation(kind="extraction"),
        lambda: _operation(implementation="document-summariser"),
        lambda: _operation(contract_version="1.0.1"),
        lambda: _operation(configuration_digest="c" * 64),
        lambda: _operation(components=_components(model_revision="revision-b")),
    ],
    ids=["kind", "implementation", "contract", "configuration", "model"],
)
def test_changed_operation_metadata_changes_fingerprint(
    changed_operation: Callable[[], ProducingOperation],
) -> None:
    assert _fingerprint(operation=changed_operation()) != _fingerprint()


@pytest.mark.parametrize("role", _COMPONENT_ROLES)
def test_changed_component_version_changes_fingerprint(role: str) -> None:
    changed = _operation(components=_components(changed_role=role))

    assert _fingerprint(operation=changed) != _fingerprint()


def test_changed_record_kind_or_output_changes_fingerprint() -> None:
    baseline = _fingerprint()

    assert _fingerprint(record_kind="corpus-summary") != baseline
    assert _fingerprint(output_digest="d" * 64) != baseline


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("record_kind", "topic-summary", "record_kind"),
        ("lineage_schema_version", 1, "lineage_schema_version"),
        ("inputs", (), "inputs"),
        ("operation", "summarisation", "operation"),
        ("output_digest", _OUTPUT_DIGEST, "output_digest"),
    ],
)
def test_fingerprint_builder_rejects_untyped_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    valid = {
        "record_kind": DerivedRecordKind("topic-summary"),
        "lineage_schema_version": CURRENT_LINEAGE_SCHEMA_VERSION,
        "inputs": _inputs(),
        "operation": _operation(),
        "output_digest": OutputDigest(DigestAlgorithm.SHA256, _OUTPUT_DIGEST),
    }

    with pytest.raises(TypeError, match=message):
        build_derivation_fingerprint(
            **(valid | {field: value})  # type: ignore[arg-type]
        )
