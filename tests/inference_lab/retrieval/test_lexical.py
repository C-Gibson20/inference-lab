"""Tests for RET-02 lexical contracts and analysis policy."""

import hashlib
import json

import pytest

from inference_lab.provenance import CorpusId, CorpusVersionId, RecordId
from inference_lab.retrieval import (
    ExactFilter,
    InvalidLexicalContractError,
    InvalidLexicalSerialisationError,
    LexicalIndexConfiguration,
    LexicalIndexManifest,
    LexicalMatchMode,
    LexicalQuery,
    RetrievalLimit,
    RetrievalQuery,
    analyse_terms,
    build_fts5_query,
    deserialise_lexical_manifest,
    escape_fts5_literal,
    expose_bm25_score,
    extract_identifiers,
    identifier_score,
    lexical_order_key,
    lexical_score_label,
    serialise_lexical_manifest,
)


def _manifest(**overrides: object) -> LexicalIndexManifest:
    """Build a deterministic manifest fixture."""
    configuration = LexicalIndexConfiguration()
    values: dict[str, object] = {
        "corpus_id": CorpusId.from_str("00000000-0000-4000-8000-000000000001"),
        "corpus_version": CorpusVersionId.from_str(
            "00000000-0000-4000-8000-000000000002"
        ),
        "generation_id": "lexical-0002",
        "configuration": configuration,
        "configuration_digest": configuration.configuration_identity.digest,
        "source_manifest_digest": "a" * 64,
        "database_checksum": "b" * 64,
        "member_count": 2,
        "chunk_count": 4,
        "identifier_count": 3,
    }
    values.update(overrides)
    return LexicalIndexManifest(**values)  # type: ignore[arg-type]


def test_configuration_identity_covers_every_result_affecting_choice() -> None:
    baseline = LexicalIndexConfiguration()
    changed = LexicalIndexConfiguration(column_weights=(2.0,))

    assert baseline.configuration_identity != changed.configuration_identity
    assert (
        baseline.configuration_identity.digest
        == hashlib.sha256(baseline.canonical_bytes()).hexdigest()
    )
    assert "sqlite-fts5" in baseline.canonical_bytes().decode()


def test_configuration_rejects_unsupported_baseline_policies() -> None:
    with pytest.raises(InvalidLexicalContractError):
        LexicalIndexConfiguration(unicode_normalisation="NFKC")
    with pytest.raises(InvalidLexicalContractError):
        LexicalIndexConfiguration(identifier_case_policy="folded")
    with pytest.raises(InvalidLexicalContractError):
        LexicalIndexConfiguration(column_weights=(0.0,))
    with pytest.raises(InvalidLexicalContractError):
        LexicalIndexConfiguration(column_weights=(1.0, 2.0))
    with pytest.raises(InvalidLexicalContractError):
        LexicalIndexConfiguration(bm25_k1=1.5)
    with pytest.raises(InvalidLexicalContractError):
        LexicalIndexConfiguration(bm25_b=0.5)


def test_analyser_normalises_unicode_and_preserves_identifier_punctuation() -> None:
    assert analyse_terms("cafe\u0301 10.1234/ABC-7") == ("café", "10.1234/ABC-7")
    assert extract_identifiers("A paper cites 10.1234/ABC-7 and 2024") == (
        "10.1234/ABC-7",
    )
    assert extract_identifiers("alpha beta 123") == ()


def test_fts_queries_are_literal_and_mode_specific() -> None:
    assert escape_fts5_literal('alpha"beta') == '"alpha""beta"'
    assert build_fts5_query('alpha OR "beta"') == '"alpha" AND "OR" AND "beta"'
    assert build_fts5_query("alpha beta", LexicalMatchMode.EXACT_PHRASE) == (
        '"alpha beta"'
    )
    with pytest.raises(InvalidLexicalContractError):
        build_fts5_query("10.1234/ABC", LexicalMatchMode.IDENTIFIER)


def test_lexical_query_rejects_filters_and_requires_one_identifier() -> None:
    version = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000002")
    query = RetrievalQuery("10.1234/ABC", version, RetrievalLimit(5))
    request = LexicalQuery(query, LexicalMatchMode.IDENTIFIER)

    assert request.identifier == "10.1234/ABC"
    assert request.engine_query() is None
    assert LexicalQuery(query).engine_query() == '"10.1234/ABC"'

    with pytest.raises(InvalidLexicalContractError):
        LexicalQuery(
            RetrievalQuery(
                "alpha",
                version,
                RetrievalLimit(1),
                filter=ExactFilter("document-type", "paper"),
            )
        )
    with pytest.raises(InvalidLexicalContractError):
        LexicalQuery(
            RetrievalQuery(
                "10.1234/ABC DOI-123",
                version,
                RetrievalLimit(1),
            ),
            LexicalMatchMode.IDENTIFIER,
        )


def test_manifest_serialisation_is_canonical_and_payload_free() -> None:
    manifest = _manifest()
    encoded = serialise_lexical_manifest(manifest)

    assert encoded == serialise_lexical_manifest(manifest)
    assert encoded.endswith(b"\n")
    assert b"/private" not in encoded
    assert deserialise_lexical_manifest(encoded) == manifest


def test_manifest_serialisation_rejects_unknown_fields_and_digest_mismatch() -> None:
    payload = json.loads(serialise_lexical_manifest(_manifest()))
    payload["unexpected"] = "private-path-marker"
    with pytest.raises(InvalidLexicalSerialisationError):
        deserialise_lexical_manifest(json.dumps(payload).encode())

    with pytest.raises(InvalidLexicalContractError):
        _manifest(configuration_digest="c" * 64)


def test_manifest_rejects_path_like_generation_identity_without_echoing_it() -> None:
    marker = "/private/index/marker"
    with pytest.raises(InvalidLexicalContractError) as error:
        _manifest(generation_id=marker)
    assert marker not in str(error.value)


def test_score_semantics_and_stable_ordering_are_explicit() -> None:
    record_id = RecordId.from_str("00000000-0000-4000-8000-000000000010")
    assert expose_bm25_score(-2.5) == 2.5
    assert identifier_score(4) == 4
    assert lexical_score_label(LexicalMatchMode.TERMS) == (
        "sqlite_fts5_negated_bm25_v2"
    )
    assert lexical_score_label(LexicalMatchMode.IDENTIFIER) == (
        "sqlite_exact_identifier_occurrences_v1"
    )
    assert lexical_order_key(2.0, record_id) == (
        -2.0,
        "00000000-0000-4000-8000-000000000010",
    )
