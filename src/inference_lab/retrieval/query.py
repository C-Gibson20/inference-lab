"""The immutable retrieval query envelope."""

from dataclasses import dataclass

from inference_lab.provenance import CorpusVersionId
from inference_lab.retrieval.errors import InvalidRetrievalContractError
from inference_lab.retrieval.filters import _FILTER_TYPES, FilterExpression
from inference_lab.retrieval.values import RetrievalLimit


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """A private query text and one corpus-scoped retrieval limit."""

    text: str
    corpus_version: CorpusVersionId
    limit: RetrievalLimit
    filter: FilterExpression | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the query's text, corpus version, limit, and filter."""
        if not isinstance(self.text, str):
            raise TypeError("text must be a string.")
        if not self.text.strip():
            raise InvalidRetrievalContractError("text must not be blank.")
        if not isinstance(self.corpus_version, CorpusVersionId):
            raise TypeError("corpus_version must be a CorpusVersionId.")
        if not isinstance(self.limit, RetrievalLimit):
            raise TypeError("limit must be a RetrievalLimit.")
        if self.filter is not None and not isinstance(self.filter, _FILTER_TYPES):
            raise TypeError("filter must be a filter expression or None.")
