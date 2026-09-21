"""Tests for the first ING-04 tokenisation and source-boundary contract."""

from pathlib import Path

import pytest

from inference_lab.ingest import (
    CURRENT_CHUNKER_VERSION,
    CURRENT_PARSER_VERSION,
    ChunkingPolicy,
    ChunkTokeniserError,
    FakeChunkTokeniser,
    LocalFastChunkTokeniser,
    NormalisedText,
    UnsafeSourceBoundaryError,
    build_canonical_source_boundary_map,
    derive_chunk_configuration_identity,
    validate_chunk_tokeniser,
)
from inference_lab.ingest.tokenisation import TokenisedText, TokenOffset

LOCAL_TOKENISER_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "backends"
    / "runtime"
    / "local_mechanics"
)


def test_baseline_policy_is_immutable_and_uses_closed_values() -> None:
    policy = ChunkingPolicy()

    assert policy.min_tokens == 256
    assert policy.max_tokens == 512
    assert policy.overlap_numerator == 15
    assert policy.overlap_denominator == 100
    with pytest.raises(TypeError):
        ChunkingPolicy(min_tokens=True)  # type: ignore[arg-type]


def test_fake_tokeniser_is_stable_and_uses_code_point_offsets() -> None:
    tokeniser = validate_chunk_tokeniser(FakeChunkTokeniser())

    result = tokeniser.tokenise("café  😀")

    assert result == tokeniser.tokenise("café  😀")
    assert result.offsets == (TokenOffset(0, 4), TokenOffset(6, 7))
    assert tokeniser.count_tokens("café  😀") == 2


def test_boundary_map_rejects_crlf_and_nfc_internal_boundaries() -> None:
    text = NormalisedText("A\r\ne\u0301 repeated e\u0301")
    boundary_map = build_canonical_source_boundary_map(text)

    assert (
        boundary_map.canonical_text_for_source_range(0, len(text.source_text))
        == text.text
    )
    assert boundary_map.canonical_offset_for_source(1) == 1
    with pytest.raises(UnsafeSourceBoundaryError):
        boundary_map.canonical_offset_for_source(2)
    with pytest.raises(UnsafeSourceBoundaryError):
        boundary_map.canonical_offset_for_source(4)
    assert boundary_map.source_range_for_canonical_range(0, len(text.text)) == (
        0,
        len(text.source_text),
    )


def test_boundary_map_handles_empty_and_lone_cr_source() -> None:
    empty = build_canonical_source_boundary_map(NormalisedText(""))
    lone_cr = build_canonical_source_boundary_map(NormalisedText("left\rright"))

    assert empty.boundaries[0].source_offset == 0
    assert lone_cr.canonical_text == "left\nright"
    assert lone_cr.canonical_offset_for_source(5) == 5


def test_local_fast_tokeniser_loads_offline_and_is_path_free() -> None:
    tokeniser = LocalFastChunkTokeniser.from_directory(LOCAL_TOKENISER_FIXTURE)

    result = tokeniser.tokenise("café astral 😀")

    assert tokeniser.identity.implementation == "transformers-fast"
    assert str(LOCAL_TOKENISER_FIXTURE) not in tokeniser.identity.as_str()
    assert result.offsets[-1].end == len("café astral 😀")
    assert tokeniser.count_tokens("café astral 😀") == len(result.token_ids)


def test_public_chunk_configuration_identity_is_stable_for_local_tokeniser() -> None:
    """Expose the exact ingest request identity without private imports."""
    tokeniser = LocalFastChunkTokeniser.from_directory(LOCAL_TOKENISER_FIXTURE)
    policy = ChunkingPolicy()
    first = derive_chunk_configuration_identity(
        parser_version=CURRENT_PARSER_VERSION,
        chunker_version=CURRENT_CHUNKER_VERSION,
        tokeniser_identity=tokeniser.identity,
        policy=policy,
    )
    assert first == derive_chunk_configuration_identity(
        parser_version=CURRENT_PARSER_VERSION,
        chunker_version=CURRENT_CHUNKER_VERSION,
        tokeniser_identity=tokeniser.identity,
        policy=policy,
    )
    assert str(LOCAL_TOKENISER_FIXTURE) not in first.as_str()
    with pytest.raises(TypeError, match="policy"):
        derive_chunk_configuration_identity(
            parser_version=CURRENT_PARSER_VERSION,
            chunker_version=CURRENT_CHUNKER_VERSION,
            tokeniser_identity=tokeniser.identity,
            policy=None,  # type: ignore[arg-type]
        )


def test_local_fast_tokeniser_rejects_symlinked_inventory(tmp_path: Path) -> None:
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    link = tmp_path / "linked.json"
    link.symlink_to(tmp_path / "tokenizer.json")

    with pytest.raises(ChunkTokeniserError) as error:
        LocalFastChunkTokeniser.from_directory(tmp_path)

    assert error.value.category == "unsafe_artefact"
    assert str(tmp_path) not in str(error.value)


def test_invalid_tokeniser_offsets_fail_without_retaining_payload() -> None:
    class InvalidTokeniser:
        identity = FakeChunkTokeniser().identity

        def tokenise(self, text: str) -> TokenisedText:
            return TokenisedText((1,), (TokenOffset(0, len(text) + 1),))

        def count_tokens(self, text: str) -> int:
            return 1

    with pytest.raises(ChunkTokeniserError) as error:
        validate_chunk_tokeniser(InvalidTokeniser())  # type: ignore[arg-type]

    assert error.value.category == "invalid_offsets"
    assert "private" not in str(error.value)
