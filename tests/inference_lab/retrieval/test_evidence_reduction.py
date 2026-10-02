"""RET-07 deterministic evidence reduction and provenance contracts."""

from dataclasses import dataclass, replace

import pytest

from inference_lab.provenance import (
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.retrieval.evidence import (
    CandidateKind,
    ChannelScore,
    EvidenceHit,
    EvidenceSource,
    RetrievalCandidate,
)
from inference_lab.retrieval.reduction import (
    EVIDENCE_SEPARATOR,
    EvidenceReductionAction,
    EvidenceReductionDecision,
    EvidenceReductionRequest,
    EvidenceTextSourceRange,
    EvidenceTokenisation,
    EvidenceTokeniserIdentity,
    EvidenceTokenOffset,
    TextSourceBoundary,
    reduce_evidence,
)
from inference_lab.retrieval.results import EvidenceResult, ResultStatus
from inference_lab.retrieval.values import RetrievalChannel, RetrievalLimit


@dataclass(frozen=True)
class _WhitespaceTokeniser:
    """Stable offset-bearing fake for evidence budget fixtures."""

    identity: EvidenceTokeniserIdentity = EvidenceTokeniserIdentity(
        implementation="test-whitespace",
        revision="fixture-v1",
        contract_version="1.0.0",
        options=(("special-tokens", "false"),),
    )

    def tokenise(self, text: str) -> EvidenceTokenisation:
        offsets: list[EvidenceTokenOffset] = []
        start: int | None = None
        for index, character in enumerate(text):
            if character.isspace():
                if start is not None:
                    offsets.append(EvidenceTokenOffset(start, index))
                    start = None
            elif start is None:
                start = index
        if start is not None:
            offsets.append(EvidenceTokenOffset(start, len(text)))
        return EvidenceTokenisation(tuple(offsets))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenise(text).offsets)


def _source(number: int = 1) -> EvidenceSource:
    return EvidenceSource(
        document_id=DocumentId.from_str(f"00000000-0000-4000-8000-{number:012d}"),
        document_version_id=DocumentVersionId.from_str(
            f"00000000-0000-4000-8000-{number + 10:012d}"
        ),
        source_span_id=SourceSpanId.from_str(
            f"00000000-0000-4000-8000-{number + 20:012d}"
        ),
    )


def _hit(number: int, text: str) -> EvidenceHit:
    return EvidenceHit(
        candidate=RetrievalCandidate(
            record_id=RecordId.from_str(f"00000000-0000-4000-8000-{number + 100:012d}"),
            kind=CandidateKind.ORIGINAL_PASSAGE,
            content=text,
            sources=(_source(number),),
        ),
        contributions=(ChannelScore(RetrievalChannel.LEXICAL, number),),
        rank=number,
    )


def _result(*hits: EvidenceHit) -> EvidenceResult:
    return EvidenceResult(
        corpus_version=CorpusVersionId.from_str("00000000-0000-4000-8000-000000000001"),
        requested_limit=RetrievalLimit(10),
        hits=tuple(hits),
        status=ResultStatus.COMPLETED,
        failures=(),
    )


def _map(hit: EvidenceHit) -> tuple[EvidenceTextSourceRange, ...]:
    text = hit.candidate.content
    return (
        EvidenceTextSourceRange(
            source=hit.candidate.sources[0],
            text_start=0,
            text_end=len(text),
            source_start=40,
            source_end=40 + len(text),
            boundaries=tuple(
                TextSourceBoundary(index, 40 + index) for index in range(len(text) + 1)
            ),
        ),
    )


def _request(
    hits: tuple[EvidenceHit, ...],
    *,
    token_budget: int,
    item_budget: int = 10,
    mapped: tuple[EvidenceHit, ...] | None = None,
) -> EvidenceReductionRequest:
    corpus_version = CorpusVersionId.from_str("00000000-0000-4000-8000-000000000001")
    return EvidenceReductionRequest(
        corpus_id=CorpusId.from_str("00000000-0000-4000-8000-000000000002"),
        corpus_version=corpus_version,
        result=_result(*hits),
        tokeniser=_WhitespaceTokeniser(),
        token_budget=token_budget,
        item_budget=item_budget,
        source_ranges={hit.candidate.record_id: _map(hit) for hit in (mapped or ())},
    )


def test_reduction_keeps_exact_joined_evidence_under_budget() -> None:
    first = _hit(1, "αβ🙂 café")
    second = _hit(2, "naïve résumé")

    outcome = reduce_evidence(_request((first, second), token_budget=4))

    assert tuple(item.hit for item in outcome.items) == (first, second)
    assert outcome.token_count == 4
    assert outcome.dropped_count == 0
    assert outcome.identity.separator == EVIDENCE_SEPARATOR
    assert all(item.action is EvidenceReductionAction.KEPT for item in outcome.items)
    assert tuple(decision.action for decision in outcome.decisions) == (
        EvidenceReductionAction.KEPT,
        EvidenceReductionAction.KEPT,
    )


def test_reduction_identity_changes_with_tokeniser_revision() -> None:
    hit = _hit(1, "stable evidence")
    original = reduce_evidence(_request((hit,), token_budget=2))
    alternate_tokeniser = _WhitespaceTokeniser(
        identity=replace(_WhitespaceTokeniser().identity, revision="fixture-v2")
    )
    alternate_request = EvidenceReductionRequest(
        corpus_id=original.corpus_id,
        corpus_version=original.corpus_version,
        result=_result(hit),
        tokeniser=alternate_tokeniser,
        token_budget=2,
        item_budget=10,
        source_ranges={},
    )

    alternate = reduce_evidence(alternate_request)

    assert alternate.items == original.items
    assert alternate.identity.tokeniser.revision == "fixture-v2"
    assert alternate.identity != original.identity


def test_oversized_evidence_truncates_at_reversible_source_boundary() -> None:
    hit = _hit(1, "alpha beta gamma delta")

    outcome = reduce_evidence(_request((hit,), token_budget=2, mapped=(hit,)))

    item = outcome.items[0]
    assert item.text == "alpha beta"
    assert item.action is EvidenceReductionAction.TRUNCATED
    assert item.source_ranges[0].text_end == len(item.text)
    assert item.source_ranges[0].source_start == 40
    assert item.source_ranges[0].source_end == 40 + len(item.text)
    assert outcome.token_count == 2


def test_unmapped_oversize_item_is_reported_and_does_not_break_budget() -> None:
    hit = _hit(1, "alpha beta gamma")

    outcome = reduce_evidence(_request((hit,), token_budget=1))

    assert outcome.items == ()
    assert outcome.dropped_count == 1
    assert outcome.failures[0].record_id == hit.candidate.record_id
    assert outcome.failures[0].category == "source_map_unavailable"
    assert outcome.token_count == 0


def test_item_ceiling_and_deterministic_order_follow_ret06_rank() -> None:
    hits = (_hit(1, "first"), _hit(2, "second"), _hit(3, "third"))

    first = reduce_evidence(_request(hits, token_budget=10, item_budget=2))
    second = reduce_evidence(_request(hits, token_budget=10, item_budget=2))

    assert tuple(item.text for item in first.items) == ("first", "second")
    assert first.items == second.items
    assert first.dropped_count == 1
    assert first.decisions[-1] == EvidenceReductionDecision(
        hits[-1].candidate.record_id,
        EvidenceReductionAction.DROPPED_ITEM_LIMIT,
    )


def test_empty_result_is_a_valid_zero_token_reduction() -> None:
    outcome = reduce_evidence(_request((), token_budget=1))

    assert outcome.items == ()
    assert outcome.input_count == 0
    assert outcome.dropped_count == 0
    assert outcome.token_count == 0


def test_request_rejects_a_result_from_another_corpus_version() -> None:
    hit = _hit(1, "passage")
    with pytest.raises(ValueError, match="selected corpus version"):
        EvidenceReductionRequest(
            corpus_id=CorpusId.from_str("00000000-0000-4000-8000-000000000002"),
            corpus_version=CorpusVersionId.from_str(
                "00000000-0000-4000-8000-000000000099"
            ),
            result=_result(hit),
            tokeniser=_WhitespaceTokeniser(),
            token_budget=2,
            item_budget=10,
            source_ranges={},
        )


def test_unmapped_normalisation_boundary_is_not_used_for_truncation() -> None:
    hit = _hit(1, "e\u0301clair café")
    source_map = list(_map(hit)[0].boundaries)
    source_map.remove(TextSourceBoundary(len("e\u0301clair"), 40 + len("e\u0301clair")))
    invalid_for_cut = EvidenceTextSourceRange(
        source=hit.candidate.sources[0],
        text_start=0,
        text_end=len(hit.candidate.content),
        source_start=40,
        source_end=40 + len(hit.candidate.content),
        boundaries=tuple(source_map),
    )
    request = EvidenceReductionRequest(
        corpus_id=CorpusId.from_str("00000000-0000-4000-8000-000000000002"),
        corpus_version=CorpusVersionId.from_str("00000000-0000-4000-8000-000000000001"),
        result=_result(hit),
        tokeniser=_WhitespaceTokeniser(),
        token_budget=1,
        item_budget=10,
        source_ranges={hit.candidate.record_id: (invalid_for_cut,)},
    )

    outcome = reduce_evidence(request)

    assert outcome.items == ()
    assert outcome.failures[0].category == "source_map_invalid"
