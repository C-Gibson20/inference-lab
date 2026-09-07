"""Contracts for persisting one immutable parser attempt."""

from dataclasses import dataclass, field

from inference_lab.ingest.outcomes import ParseOutcome
from inference_lab.ingest.parser import ParserInput
from inference_lab.ingest.versioning import ParserVersion
from inference_lab.provenance import CorpusId, RecordId


@dataclass(frozen=True, slots=True, kw_only=True)
class ParsePersistenceRequest:
    """Carry one corpus-authorised parser attempt to persistence."""

    corpus_id: CorpusId
    parser_input: ParserInput = field(repr=False)
    parser_version: ParserVersion
    outcome: ParseOutcome = field(repr=False)
    parse_id: RecordId
    parsed_record_id: RecordId | None
    supersedes_record_id: RecordId | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Reject inconsistent source, parser, outcome, and record identities."""
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(self.parser_input, ParserInput):
            raise TypeError("parser_input must be a ParserInput.")
        if not isinstance(self.parser_version, ParserVersion):
            raise TypeError("parser_version must be a ParserVersion.")
        if not isinstance(self.outcome, ParseOutcome):
            raise TypeError("outcome must be a ParseOutcome.")
        if not isinstance(self.parse_id, RecordId):
            raise TypeError("parse_id must be a RecordId.")
        if self.parsed_record_id is not None and not isinstance(
            self.parsed_record_id, RecordId
        ):
            raise TypeError("parsed_record_id must be a RecordId or None.")
        if self.supersedes_record_id is not None and not isinstance(
            self.supersedes_record_id, RecordId
        ):
            raise TypeError("supersedes_record_id must be a RecordId or None.")

        document = self.outcome.document
        if document is None:
            if self.parsed_record_id is not None:
                raise ValueError("a failed parse must not name a parsed record.")
            if self.supersedes_record_id is not None:
                raise ValueError("a failed parse must not supersede a representation.")
        else:
            if self.parsed_record_id is None:
                raise ValueError("a represented parse requires a parsed record ID.")
            if document.document_id != self.parser_input.document_id:
                raise ValueError("parsed document uses the wrong document identity.")
            if document.document_version_id != self.parser_input.document_version_id:
                raise ValueError("parsed document uses the wrong document version.")
            if document.parser_version != self.parser_version:
                raise ValueError("parsed document uses the wrong parser version.")

        record_ids = tuple(
            record_id
            for record_id in (
                self.parse_id,
                self.parsed_record_id,
                self.supersedes_record_id,
            )
            if record_id is not None
        )
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("parse record identities must be distinct.")
