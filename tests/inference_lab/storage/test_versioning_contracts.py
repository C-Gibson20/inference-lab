"""Tests for immutable corpus-version and manifest contracts."""

from dataclasses import FrozenInstanceError, replace

import pytest

from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
)
from inference_lab.storage import (
    CORPUS_MANIFEST_CONTRACT_VERSION,
    ComponentGenerationReference,
    ComponentGenerationState,
    ComponentKind,
    CorpusVersionManifest,
    InvalidCorpusVersionContractError,
)

CORPUS_ID = CorpusId.from_str("00000000-0000-4000-8000-000000000001")
OTHER_CORPUS_ID = CorpusId.from_str("00000000-0000-4000-8000-000000000002")
VERSION_ID = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000003")
PARENT_VERSION_ID = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000004")
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def _membership(
    suffix: int,
    *,
    corpus_id: CorpusId = CORPUS_ID,
    document_id: DocumentId | None = None,
) -> CorpusDocumentVersionMembership:
    return CorpusDocumentVersionMembership(
        corpus_id=corpus_id,
        document_id=document_id
        or DocumentId.from_str(f"00000000-0000-4000-8000-{suffix:012d}"),
        document_version_id=DocumentVersionId.from_str(
            f"10000000-0000-4000-8000-{suffix:012d}"
        ),
    )


def _component(
    kind: str = "lexical-index",
    *,
    corpus_id: CorpusId = CORPUS_ID,
    corpus_version: CorpusVersionId = VERSION_ID,
    state: ComponentGenerationState = ComponentGenerationState.READY,
    generation_id: str | None = None,
) -> ComponentGenerationReference:
    return ComponentGenerationReference(
        component_kind=ComponentKind(kind),
        generation_id=(
            f"{kind}:generation-1" if generation_id is None else generation_id
        ),
        implementation="fake-index",
        format_version="1.0.0",
        contract_version="2.1.0-rc.1",
        configuration_digest=DIGEST_A,
        source_manifest_digest=DIGEST_B,
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        state=state,
    )


def _manifest(
    *,
    corpus_id: CorpusId = CORPUS_ID,
    corpus_version: CorpusVersionId = VERSION_ID,
    memberships: tuple[CorpusDocumentVersionMembership, ...] = (),
    components: tuple[ComponentGenerationReference, ...] = (),
) -> CorpusVersionManifest:
    return CorpusVersionManifest(
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        document_memberships=memberships,
        component_generations=components,
    )


def test_component_kind_is_typed_immutable_and_round_trips() -> None:
    kind = ComponentKind.from_str("lexical-index")

    assert kind.as_str() == "lexical-index"
    with pytest.raises(FrozenInstanceError):
        kind.value = "dense-index"  # type: ignore[misc]


@pytest.mark.parametrize("value", [None, 1])
def test_component_kind_rejects_wrong_types(value: object) -> None:
    with pytest.raises(TypeError, match="component_kind"):
        ComponentKind(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    ["", "Lexical", "lexical_index", "lexical--index", "-lexical"],
)
def test_component_kind_rejects_noncanonical_values(value: str) -> None:
    with pytest.raises(InvalidCorpusVersionContractError, match="component_kind"):
        ComponentKind(value)


def test_component_generation_reference_retains_complete_compatibility() -> None:
    component = _component()

    assert component.component_kind == ComponentKind("lexical-index")
    assert component.generation_id == "lexical-index:generation-1"
    assert component.key() == ("lexical-index", "lexical-index:generation-1")
    assert component.state is ComponentGenerationState.READY
    with pytest.raises(FrozenInstanceError):
        component.state = ComponentGenerationState.PREPARING  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("component_kind", "lexical-index", "component_kind"),
        ("generation_id", 1, "generation_id"),
        ("implementation", 1, "implementation"),
        ("format_version", 1, "format_version"),
        ("contract_version", 1, "contract_version"),
        ("configuration_digest", 1, "configuration_digest"),
        ("source_manifest_digest", 1, "source_manifest_digest"),
        ("corpus_id", VERSION_ID, "corpus_id"),
        ("corpus_version", CORPUS_ID, "corpus_version"),
        ("state", "ready", "state"),
    ],
)
def test_component_generation_rejects_wrong_field_types(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        replace(_component(), **{field: value})


@pytest.mark.parametrize("generation_id", ["", " generation", "../index", "a/b"])
def test_component_generation_rejects_unsafe_identifier(
    generation_id: str,
) -> None:
    with pytest.raises(InvalidCorpusVersionContractError, match="generation_id"):
        _component(generation_id=generation_id)


@pytest.mark.parametrize("field", ["format_version", "contract_version"])
def test_component_generation_requires_canonical_semantic_versions(
    field: str,
) -> None:
    with pytest.raises(InvalidCorpusVersionContractError, match=field):
        replace(_component(), **{field: "v1"})


@pytest.mark.parametrize("field", ["configuration_digest", "source_manifest_digest"])
def test_component_generation_requires_canonical_digests(field: str) -> None:
    with pytest.raises(InvalidCorpusVersionContractError, match=field):
        replace(_component(), **{field: "A" * 64})


def test_manifest_canonicalises_membership_and_components() -> None:
    first_membership = _membership(10)
    second_membership = _membership(20)
    lexical = _component("lexical-index")
    metadata = _component("metadata")

    manifest = _manifest(
        memberships=(second_membership, first_membership),
        components=(metadata, lexical),
    )

    assert manifest.document_memberships == (first_membership, second_membership)
    assert manifest.component_generations == (lexical, metadata)
    assert len(manifest.manifest_digest) == 64
    assert CORPUS_MANIFEST_CONTRACT_VERSION == "1.0.0"
    with pytest.raises(FrozenInstanceError):
        manifest.document_memberships = ()  # type: ignore[misc]


def test_manifest_digest_is_deterministic_across_input_order() -> None:
    memberships = (_membership(10), _membership(20))
    components = (_component("lexical-index"), _component("metadata"))

    first = _manifest(memberships=memberships, components=components)
    second = _manifest(
        memberships=tuple(reversed(memberships)),
        components=tuple(reversed(components)),
    )

    assert first == second
    assert first.manifest_digest == second.manifest_digest


def test_manifest_digest_excludes_proposed_version_identity() -> None:
    other_version = CorpusVersionId.generate()
    first = _manifest(components=(_component(),))
    second = _manifest(
        corpus_version=other_version,
        components=(_component(corpus_version=other_version),),
    )

    assert first.corpus_version != second.corpus_version
    assert first.manifest_digest == second.manifest_digest


def test_manifest_semantic_state_ignores_generation_identity() -> None:
    other_version = CorpusVersionId.generate()
    first = _manifest(components=(_component(generation_id="generation-1"),))
    second = _manifest(
        corpus_version=other_version,
        components=(
            _component(
                corpus_version=other_version,
                generation_id="generation-2",
            ),
        ),
    )

    assert first.manifest_digest != second.manifest_digest
    assert first.has_same_semantic_state(second)


def test_manifest_semantic_state_detects_component_changes() -> None:
    baseline_component = _component()
    baseline = _manifest(components=(baseline_component,))
    changed = _manifest(
        components=(replace(baseline_component, configuration_digest="c" * 64),)
    )

    assert not baseline.has_same_semantic_state(changed)
    with pytest.raises(TypeError, match="other"):
        baseline.has_same_semantic_state(object())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("corpus_id", VERSION_ID),
        ("corpus_version", CORPUS_ID),
        ("document_memberships", []),
        ("document_memberships", ("membership",)),
        ("component_generations", []),
        ("component_generations", ("component",)),
    ],
)
def test_manifest_rejects_wrong_field_types(field: str, value: object) -> None:
    with pytest.raises(TypeError, match=field):
        replace(_manifest(), **{field: value})


def test_manifest_digest_changes_with_semantic_state() -> None:
    baseline = _manifest(memberships=(_membership(10),))
    changed_membership = _manifest(memberships=(_membership(20),))
    changed_component = _manifest(components=(_component(),))
    other_corpus = _manifest(corpus_id=OTHER_CORPUS_ID)

    assert (
        len(
            {
                baseline.manifest_digest,
                changed_membership.manifest_digest,
                changed_component.manifest_digest,
                other_corpus.manifest_digest,
            }
        )
        == 4
    )


def test_manifest_rejects_multiple_visible_versions_of_one_document() -> None:
    document_id = DocumentId.generate()

    with pytest.raises(InvalidCorpusVersionContractError, match="one document"):
        _manifest(
            memberships=(
                _membership(10, document_id=document_id),
                _membership(20, document_id=document_id),
            )
        )


def test_manifest_rejects_foreign_document_membership() -> None:
    with pytest.raises(InvalidCorpusVersionContractError, match="manifest corpus"):
        _manifest(memberships=(_membership(10, corpus_id=OTHER_CORPUS_ID),))


@pytest.mark.parametrize(
    "component",
    [
        _component(corpus_id=OTHER_CORPUS_ID),
        _component(corpus_version=PARENT_VERSION_ID),
        _component(state=ComponentGenerationState.PREPARING),
    ],
)
def test_manifest_rejects_foreign_or_unready_component(
    component: ComponentGenerationReference,
) -> None:
    with pytest.raises(InvalidCorpusVersionContractError):
        _manifest(components=(component,))


def test_manifest_rejects_conflicting_component_kinds() -> None:
    with pytest.raises(InvalidCorpusVersionContractError, match="component kinds"):
        _manifest(
            components=(
                _component(generation_id="generation-1"),
                _component(generation_id="generation-2"),
            )
        )


def test_manifest_checks_exact_declared_component_set() -> None:
    manifest = _manifest(
        components=(_component("lexical-index"), _component("metadata"))
    )

    assert manifest.has_required_components(
        (ComponentKind("metadata"), ComponentKind("lexical-index"))
    )
    assert not manifest.has_required_components((ComponentKind("metadata"),))


@pytest.mark.parametrize("kinds", [[], ("metadata",)])
def test_manifest_component_check_requires_typed_tuple(kinds: object) -> None:
    with pytest.raises(TypeError, match="required_component_kinds"):
        _manifest().has_required_components(kinds)  # type: ignore[arg-type]


def test_manifest_component_check_rejects_duplicate_kinds() -> None:
    kind = ComponentKind("metadata")

    with pytest.raises(InvalidCorpusVersionContractError, match="duplicates"):
        _manifest().has_required_components((kind, kind))
