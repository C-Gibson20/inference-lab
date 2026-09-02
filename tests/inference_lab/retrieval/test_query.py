"""Tests for the retrieval query contract."""

import pytest

from inference_lab.provenance import CorpusVersionId
from inference_lab.retrieval import (
    ExactFilter,
    InvalidRetrievalContractError,
    RetrievalChannel,
    RetrievalLimit,
    RetrievalQuery,
)


def test_query_and_channel_values_are_strict_and_immutable() -> None:
    """Construct valid values, preserve text, and reject mutation."""
    corpus_version = CorpusVersionId.generate()
    expression = ExactFilter("document-type", "paper")
    query = RetrievalQuery(
        "  exact text  ", corpus_version, RetrievalLimit(10), expression
    )

    assert query.text == "  exact text  "
    assert query.corpus_version is corpus_version
    assert query.filter == expression
    assert RetrievalChannel.LEXICAL.value == "lexical"
    assert hash(RetrievalLimit(10)) == hash(RetrievalLimit(10))
    with pytest.raises(AttributeError):
        query.text = "changed"  # type: ignore[misc]


def test_query_requires_text_corpus_version_limit_and_filter_types() -> None:
    """Reject blank text and values from the wrong contract domains."""
    corpus_version = CorpusVersionId.generate()
    with pytest.raises(InvalidRetrievalContractError):
        RetrievalQuery(" \n", corpus_version, RetrievalLimit(1))
    with pytest.raises(TypeError):
        RetrievalQuery("question", str(corpus_version), RetrievalLimit(1))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        RetrievalQuery("question", corpus_version, 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        RetrievalQuery("question", corpus_version, RetrievalLimit(1), object())  # type: ignore[arg-type]
