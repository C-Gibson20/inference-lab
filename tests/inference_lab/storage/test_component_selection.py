"""Shared component-generation selection and reference comparison."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.storage import (
    ComponentGenerationReference,
    ComponentGenerationState,
    ComponentKind,
    generation_matches_reference,
    select_ready_component_generation,
)

CORPUS = CorpusId.from_str("00000000-0000-4000-8000-000000000001")
VERSION = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000002")
OTHER_CORPUS = CorpusId.from_str("00000000-0000-4000-8000-000000000003")
OTHER_VERSION = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000004")
KIND = ComponentKind("lexical-index")
OTHER_KIND = ComponentKind("dense-index")


def _reference(**overrides: object) -> ComponentGenerationReference:
    """Build a ready reference owned by the pinned corpus and version."""
    fields: dict[str, object] = {
        "component_kind": KIND,
        "generation_id": "generation-1",
        "implementation": "sqlite-fts5",
        "format_version": "1.0.0",
        "contract_version": "1.0.0",
        "configuration_digest": "a" * 64,
        "source_manifest_digest": "b" * 64,
        "corpus_id": CORPUS,
        "corpus_version": VERSION,
        "state": ComponentGenerationState.READY,
    }
    return ComponentGenerationReference(**(fields | overrides))  # type: ignore[arg-type]


def _manifest(*references: ComponentGenerationReference) -> object:
    return SimpleNamespace(component_generations=references)


def _select(*references: ComponentGenerationReference):
    return select_ready_component_generation(
        _manifest(*references),  # type: ignore[arg-type]
        KIND,
        corpus_id=CORPUS,
        corpus_version=VERSION,
    )


def test_selects_the_single_ready_generation_of_the_requested_kind() -> None:
    wanted = _reference()
    other = _reference(component_kind=OTHER_KIND, generation_id="generation-2")
    assert _select(other, wanted) is wanted


@pytest.mark.parametrize(
    "overrides",
    [
        {"state": ComponentGenerationState.PREPARING},
        {"corpus_id": OTHER_CORPUS},
        {"corpus_version": OTHER_VERSION},
    ],
    ids=["unready", "foreign-corpus", "foreign-version"],
)
def test_rejects_a_generation_that_is_not_ready_and_owned(
    overrides: dict[str, object],
) -> None:
    # A committed manifest should never contain these, so the published view
    # must fail closed rather than serve an unpublished or foreign artefact.
    assert _select(_reference(**overrides)) is None


def test_rejects_absent_and_duplicated_kinds() -> None:
    assert _select() is None
    assert _select(_reference(), _reference(generation_id="generation-2")) is None


def test_matches_a_reference_regardless_of_lifecycle_state() -> None:
    reference = _reference()
    candidate = replace(reference, state=ComponentGenerationState.PREPARING)
    assert generation_matches_reference(reference, candidate)


@pytest.mark.parametrize(
    "field",
    [
        "component_kind",
        "generation_id",
        "implementation",
        "format_version",
        "contract_version",
        "configuration_digest",
        "source_manifest_digest",
        "corpus_id",
        "corpus_version",
    ],
)
def test_every_identity_field_participates_in_the_comparison(field: str) -> None:
    # Comparing whole references means a new identity field is covered without
    # revisiting each call site; this asserts none is silently ignored.
    changed = {
        "component_kind": OTHER_KIND,
        "generation_id": "generation-2",
        "implementation": "sqlite-flat",
        "format_version": "2.0.0",
        "contract_version": "2.0.0",
        "configuration_digest": "c" * 64,
        "source_manifest_digest": "d" * 64,
        "corpus_id": OTHER_CORPUS,
        "corpus_version": OTHER_VERSION,
    }[field]
    reference = _reference()
    assert not generation_matches_reference(
        reference, replace(reference, **{field: changed})
    )
