"""Contracts and deterministic analysis policy for lexical retrieval."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from inference_lab._json import JsonObject, canonical_json_bytes
from inference_lab._validation import is_canonical_semantic_version, is_lowercase_slug
from inference_lab.provenance import (
    ConfigurationIdentity,
    CorpusId,
    CorpusVersionId,
    DigestAlgorithm,
    RecordId,
)
from inference_lab.retrieval.errors import InvalidRetrievalContractError
from inference_lab.retrieval.query import RetrievalQuery

LEXICAL_COMPONENT_KIND: Final[str] = "lexical-index"
LEXICAL_IMPLEMENTATION: Final[str] = "sqlite-fts5"
LEXICAL_FORMAT_VERSION: Final[str] = "1.0.0"
LEXICAL_CONTRACT_VERSION: Final[str] = "1.1.0"
LEXICAL_ENGINE_REVISION: Final[str] = "sqlite-fts5-weighted-bm25"
_LEGACY_ENGINE_REVISION: Final[str] = "sqlite-fts5-baseline"
LEXICAL_MANIFEST_SCHEMA_VERSION: Final[str] = "1.0.0"
LEXICAL_SCORE_LABEL: Final[str] = "sqlite_fts5_negated_bm25_v2"
LEXICAL_IDENTIFIER_SCORE_LABEL: Final[str] = "sqlite_exact_identifier_occurrences_v1"

_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_GENERATION_ID_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_IDENTIFIER_PUNCTUATION: Final[frozenset[str]] = frozenset("-._:/+")
_CONFIGURATION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "engine",
        "engine_revision",
        "index_schema_version",
        "fts_tokenizer",
        "unicode_normalisation",
        "term_case_policy",
        "identifier_grammar_version",
        "identifier_case_policy",
        "query_escaping_version",
        "bm25_k1",
        "bm25_b",
        "column_weights",
        "score_label",
        "identifier_score_label",
        "tie_break_policy",
    }
)


class LexicalMatchMode(StrEnum):
    """Closed lexical matching modes supported by the baseline."""

    TERMS = "terms"
    EXACT_PHRASE = "exact_phrase"
    IDENTIFIER = "identifier"


class LexicalFailureCategory(StrEnum):
    """Payload-free categories for lexical operational failures."""

    INVALID_REQUEST = "invalid_request"
    GENERATION_UNAVAILABLE = "generation_unavailable"
    CORRUPT_GENERATION = "corrupt_generation"
    INCOMPATIBLE_GENERATION = "incompatible_generation"
    EXECUTION_FAILURE = "execution_failure"
    CANCELLED = "cancelled"


class LexicalError(Exception):
    """A lexical operational failure containing only a bounded category."""

    def __init__(self, category: LexicalFailureCategory) -> None:
        if not isinstance(category, LexicalFailureCategory):
            raise TypeError("category must be a LexicalFailureCategory.")
        super().__init__(category.value)
        self.category = category


class InvalidLexicalContractError(InvalidRetrievalContractError):
    """Raised when correctly typed lexical data violates its contract."""


class InvalidLexicalSerialisationError(InvalidLexicalContractError):
    """Raised when a lexical manifest is not valid canonical data."""


def _validate_slug(value: object, field_name: str) -> str:
    """Validate one project-owned lowercase slug without exposing its value."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not is_lowercase_slug(value):
        raise InvalidLexicalContractError(
            f"{field_name} must be a lowercase project-owned slug."
        )
    return value


def _validate_semantic_version(value: object, field_name: str) -> str:
    """Validate one canonical semantic version."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not is_canonical_semantic_version(value):
        raise InvalidLexicalContractError(
            f"{field_name} must be a canonical semantic version."
        )
    return value


def _validate_sha256(value: object, field_name: str) -> str:
    """Validate one lowercase SHA-256 digest."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise InvalidLexicalContractError(
            f"{field_name} must be a lowercase SHA-256 digest."
        )
    return value


def _validate_non_negative_integer(value: object, field_name: str) -> int:
    """Validate one strict non-negative integer."""
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if value < 0:
        raise InvalidLexicalContractError(f"{field_name} must not be negative.")
    return value


@dataclass(frozen=True, slots=True)
class LexicalIndexConfiguration:
    """Versioned lexical choices; the current FTS5 revision fixes k1/b.

    ``column_weights`` contains the one indexed content-column weight. The
    unindexed record ID occupies FTS5 column zero and always receives zero.
    Legacy revision values remain decodable for published v1 generations but
    cannot be used to build a new generation.
    """

    engine: str = LEXICAL_IMPLEMENTATION
    engine_revision: str = LEXICAL_ENGINE_REVISION
    index_schema_version: str = LEXICAL_FORMAT_VERSION
    fts_tokenizer: str = "unicode61 remove_diacritics 0 tokenchars '-._:/+'"
    unicode_normalisation: str = "NFC"
    term_case_policy: str = "fts5-unicode61"
    identifier_grammar_version: str = "bounded-alnum-internal-punctuation-v1"
    identifier_case_policy: str = "binary"
    query_escaping_version: str = "literal-quoted-v1"
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    column_weights: tuple[float, ...] = (1.0,)
    score_label: str = LEXICAL_SCORE_LABEL
    identifier_score_label: str = LEXICAL_IDENTIFIER_SCORE_LABEL
    tie_break_policy: str = "record-id-ascending"

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate all result-affecting lexical choices."""
        _validate_slug(self.engine, "engine")
        if not isinstance(self.engine_revision, str) or not self.engine_revision:
            raise InvalidLexicalContractError("engine_revision must be non-empty.")
        if self.engine_revision not in {
            _LEGACY_ENGINE_REVISION,
            LEXICAL_ENGINE_REVISION,
        }:
            raise InvalidLexicalContractError("unsupported lexical engine revision.")
        _validate_semantic_version(self.index_schema_version, "index_schema_version")
        if not isinstance(self.fts_tokenizer, str) or not self.fts_tokenizer:
            raise InvalidLexicalContractError("fts_tokenizer must be non-empty.")
        if "\x00" in self.fts_tokenizer:
            raise InvalidLexicalContractError("fts_tokenizer must not contain NUL.")
        for field_name in (
            "unicode_normalisation",
            "term_case_policy",
            "identifier_grammar_version",
            "identifier_case_policy",
            "query_escaping_version",
            "tie_break_policy",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise InvalidLexicalContractError(f"{field_name} must be non-empty.")
        if self.unicode_normalisation != "NFC":
            raise InvalidLexicalContractError(
                "unicode_normalisation must use the baseline NFC policy."
            )
        if self.identifier_case_policy != "binary":
            raise InvalidLexicalContractError(
                "identifier_case_policy must use the baseline binary policy."
            )
        if not isinstance(self.bm25_k1, float) or not math.isfinite(self.bm25_k1):
            raise TypeError("bm25_k1 must be a finite float.")
        if self.engine_revision == LEXICAL_ENGINE_REVISION:
            if self.bm25_k1 != 1.2:
                raise InvalidLexicalContractError("SQLite FTS5 fixes bm25_k1 at 1.2.")
        elif self.bm25_k1 <= 0:
            raise InvalidLexicalContractError("bm25_k1 must be positive.")
        if not isinstance(self.bm25_b, float) or not math.isfinite(self.bm25_b):
            raise TypeError("bm25_b must be a finite float.")
        if self.engine_revision == LEXICAL_ENGINE_REVISION:
            if self.bm25_b != 0.75:
                raise InvalidLexicalContractError("SQLite FTS5 fixes bm25_b at 0.75.")
        elif not 0 <= self.bm25_b <= 1:
            raise InvalidLexicalContractError("bm25_b must be between zero and one.")
        if not isinstance(self.column_weights, tuple):
            raise TypeError("column_weights must be a tuple.")
        if self.engine_revision == LEXICAL_ENGINE_REVISION:
            if len(self.column_weights) != 1:
                raise InvalidLexicalContractError(
                    "column_weights must contain the single indexed content weight."
                )
        elif not self.column_weights:
            raise InvalidLexicalContractError("column_weights must not be empty.")
        for weight in self.column_weights:
            if not isinstance(weight, float) or not math.isfinite(weight):
                raise TypeError("column_weights must contain finite floats.")
            if weight <= 0:
                raise InvalidLexicalContractError(
                    "column_weights must contain positive values."
                )
        for field_name in ("score_label", "identifier_score_label"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise InvalidLexicalContractError(f"{field_name} must be non-empty.")

    def as_payload(self) -> JsonObject:
        """Return the complete canonical JSON-compatible configuration."""
        return {
            "engine": self.engine,
            "engine_revision": self.engine_revision,
            "index_schema_version": self.index_schema_version,
            "fts_tokenizer": self.fts_tokenizer,
            "unicode_normalisation": self.unicode_normalisation,
            "term_case_policy": self.term_case_policy,
            "identifier_grammar_version": self.identifier_grammar_version,
            "identifier_case_policy": self.identifier_case_policy,
            "query_escaping_version": self.query_escaping_version,
            "bm25_k1": self.bm25_k1,
            "bm25_b": self.bm25_b,
            "column_weights": list(self.column_weights),
            "score_label": self.score_label,
            "identifier_score_label": self.identifier_score_label,
            "tie_break_policy": self.tie_break_policy,
        }

    def canonical_bytes(self) -> bytes:
        """Return deterministic bytes used for the configuration identity."""
        return canonical_json_bytes(self.as_payload())

    @property
    def configuration_identity(self) -> ConfigurationIdentity:
        """Return the digest identity of this complete configuration."""
        return ConfigurationIdentity(
            algorithm=DigestAlgorithm.SHA256,
            digest=hashlib.sha256(self.canonical_bytes()).hexdigest(),
        )


def _is_token_character(character: str) -> bool:
    """Return whether a character may occur in an analysed token."""
    return character.isalnum() or character == "_"


def _scan_tokens(text: str) -> tuple[str, ...]:
    """Scan NFC text while retaining approved punctuation only internally."""
    normalised = unicodedata.normalize("NFC", text)
    tokens: list[str] = []
    current: list[str] = []
    for index, character in enumerate(normalised):
        previous = normalised[index - 1] if index else ""
        following = normalised[index + 1] if index + 1 < len(normalised) else ""
        internal_punctuation = (
            character in _IDENTIFIER_PUNCTUATION
            and bool(current)
            and _is_token_character(previous)
            and _is_token_character(following)
        )
        if _is_token_character(character) or internal_punctuation:
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tuple(tokens)


def analyse_terms(text: str) -> tuple[str, ...]:
    """Return deterministic NFC terms for the baseline lexical analyser."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    return _scan_tokens(text)


def extract_identifiers(text: str) -> tuple[str, ...]:
    """Extract NFC identifier lexemes using the versioned bounded grammar."""
    terms = analyse_terms(text)
    identifiers: list[str] = []
    for term in terms:
        has_letters = any(character.isalpha() for character in term)
        has_digits = any(character.isdigit() for character in term)
        has_internal_punctuation = any(
            character in _IDENTIFIER_PUNCTUATION for character in term
        )
        if (has_letters and has_digits) or has_internal_punctuation:
            identifiers.append(term)
    return tuple(identifiers)


def escape_fts5_literal(value: str) -> str:
    """Quote one FTS5 value so its contents cannot become FTS syntax."""
    if not isinstance(value, str):
        raise TypeError("value must be a string.")
    if not value:
        raise InvalidLexicalContractError("FTS5 literal must not be empty.")
    return '"' + value.replace('"', '""') + '"'


def build_fts5_query(
    text: str,
    mode: LexicalMatchMode = LexicalMatchMode.TERMS,
) -> str:
    """Build a literal FTS5 expression for terms or exact-phrase matching."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if not isinstance(mode, LexicalMatchMode):
        raise TypeError("mode must be a LexicalMatchMode.")
    if mode is LexicalMatchMode.IDENTIFIER:
        raise InvalidLexicalContractError(
            "identifier mode does not construct an FTS5 expression."
        )
    terms = analyse_terms(text)
    if not terms:
        raise InvalidLexicalContractError("text must contain an analysable term.")
    if mode is LexicalMatchMode.EXACT_PHRASE:
        return escape_fts5_literal(" ".join(terms))
    return " AND ".join(escape_fts5_literal(term) for term in terms)


@dataclass(frozen=True, slots=True)
class LexicalQuery:
    """A RET-01 query plus one closed lexical matching mode."""

    query: RetrievalQuery
    mode: LexicalMatchMode = LexicalMatchMode.TERMS

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the lexical request without executing an engine query."""
        if not isinstance(self.query, RetrievalQuery):
            raise TypeError("query must be a RetrievalQuery.")
        if not isinstance(self.mode, LexicalMatchMode):
            raise TypeError("mode must be a LexicalMatchMode.")
        if self.query.filter is not None:
            raise InvalidLexicalContractError(
                "lexical requests do not support filters; RET-03 owns filter execution."
            )
        if not analyse_terms(self.query.text):
            raise InvalidLexicalContractError(
                "query text must contain an analysable lexical term."
            )
        if self.mode is LexicalMatchMode.IDENTIFIER:
            identifiers = extract_identifiers(self.query.text)
            if len(identifiers) != 1:
                raise InvalidLexicalContractError(
                    "identifier mode requires exactly one identifier lexeme."
                )

    @property
    def analysed_terms(self) -> tuple[str, ...]:
        """Return the deterministic terms used by this request."""
        return analyse_terms(self.query.text)

    @property
    def identifier(self) -> str | None:
        """Return the exact identifier value for identifier mode."""
        identifiers = extract_identifiers(self.query.text)
        return identifiers[0] if self.mode is LexicalMatchMode.IDENTIFIER else None

    def engine_query(self) -> str | None:
        """Return a safe FTS5 expression, or ``None`` for identifier mode."""
        if self.mode is LexicalMatchMode.IDENTIFIER:
            return None
        return build_fts5_query(self.query.text, self.mode)


# The longer name is useful at integration boundaries while the short name is
# convenient for callers constructing a request.
LexicalRetrievalRequest = LexicalQuery


@dataclass(frozen=True, slots=True)
class LexicalIndexManifest:
    """Strict, path-free metadata for one immutable lexical generation."""

    corpus_id: CorpusId
    corpus_version: CorpusVersionId
    generation_id: str
    configuration: LexicalIndexConfiguration
    configuration_digest: str
    source_manifest_digest: str
    database_checksum: str
    member_count: int
    chunk_count: int
    identifier_count: int
    schema_version: str = LEXICAL_MANIFEST_SCHEMA_VERSION
    component_kind: str = LEXICAL_COMPONENT_KIND
    implementation: str = LEXICAL_IMPLEMENTATION
    format_version: str = LEXICAL_FORMAT_VERSION
    contract_version: str = LEXICAL_CONTRACT_VERSION

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate ownership, compatibility, digests, and bounded counts."""
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        if not isinstance(self.corpus_version, CorpusVersionId):
            raise TypeError("corpus_version must be a CorpusVersionId.")
        if not isinstance(self.generation_id, str):
            raise TypeError("generation_id must be a string.")
        if _GENERATION_ID_PATTERN.fullmatch(self.generation_id) is None:
            raise InvalidLexicalContractError(
                "generation_id must be a canonical non-path identifier."
            )
        _validate_slug(self.component_kind, "component_kind")
        _validate_slug(self.implementation, "implementation")
        _validate_semantic_version(self.schema_version, "schema_version")
        _validate_semantic_version(self.format_version, "format_version")
        _validate_semantic_version(self.contract_version, "contract_version")
        if not isinstance(self.configuration, LexicalIndexConfiguration):
            raise TypeError("configuration must be a LexicalIndexConfiguration.")
        if self.schema_version != LEXICAL_MANIFEST_SCHEMA_VERSION:
            raise InvalidLexicalContractError("unsupported lexical manifest version.")
        if self.component_kind != LEXICAL_COMPONENT_KIND:
            raise InvalidLexicalContractError("unsupported lexical component kind.")
        if self.implementation != self.configuration.engine:
            raise InvalidLexicalContractError(
                "manifest implementation must match its configuration engine."
            )
        expected_revision = {
            "1.0.0": _LEGACY_ENGINE_REVISION,
            LEXICAL_CONTRACT_VERSION: LEXICAL_ENGINE_REVISION,
        }.get(self.contract_version)
        if self.configuration.engine_revision != expected_revision:
            raise InvalidLexicalContractError(
                "manifest contract and engine revision are incompatible."
            )
        if (
            self.configuration_digest
            != self.configuration.configuration_identity.digest
        ):
            raise InvalidLexicalContractError(
                "configuration_digest must match the complete configuration."
            )
        _validate_sha256(self.configuration_digest, "configuration_digest")
        _validate_sha256(self.source_manifest_digest, "source_manifest_digest")
        _validate_sha256(self.database_checksum, "database_checksum")
        for field_name in (
            "member_count",
            "chunk_count",
            "identifier_count",
        ):
            _validate_non_negative_integer(getattr(self, field_name), field_name)

    def as_payload(self) -> JsonObject:
        """Return the complete manifest payload without filesystem details."""
        return {
            "component_kind": self.component_kind,
            "configuration": self.configuration.as_payload(),
            "configuration_digest": self.configuration_digest,
            "contract_version": self.contract_version,
            "corpus_id": self.corpus_id.as_str(),
            "corpus_version": self.corpus_version.as_str(),
            "database_checksum": self.database_checksum,
            "format_version": self.format_version,
            "generation_id": self.generation_id,
            "identifier_count": self.identifier_count,
            "implementation": self.implementation,
            "member_count": self.member_count,
            "schema_version": self.schema_version,
            "source_manifest_digest": self.source_manifest_digest,
            "chunk_count": self.chunk_count,
        }

    def canonical_bytes(self) -> bytes:
        """Return deterministic UTF-8 manifest bytes."""
        return canonical_json_bytes(self.as_payload())


@dataclass(frozen=True, slots=True)
class LexicalIndexBuildStatistics:
    """Bounded statistics for one lexical generation build or reuse."""

    members_processed: int
    chunks_processed: int
    identifiers_emitted: int
    reused_chunks: int = 0
    inserted_chunks: int = 0
    replaced_chunks: int = 0
    removed_chunks: int = 0
    database_reused: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "members_processed",
            "chunks_processed",
            "identifiers_emitted",
            "reused_chunks",
            "inserted_chunks",
            "replaced_chunks",
            "removed_chunks",
        ):
            _validate_non_negative_integer(getattr(self, field_name), field_name)
        if type(self.database_reused) is not bool:
            raise TypeError("database_reused must be a boolean.")


@dataclass(frozen=True, slots=True)
class LexicalIndexBuildOutcome:
    """Successful result of a lexical build or compatible payload reuse."""

    manifest: LexicalIndexManifest
    statistics: LexicalIndexBuildStatistics

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, LexicalIndexManifest):
            raise TypeError("manifest must be a LexicalIndexManifest.")
        if not isinstance(self.statistics, LexicalIndexBuildStatistics):
            raise TypeError("statistics must be LexicalIndexBuildStatistics.")


def lexical_score_label(mode: LexicalMatchMode) -> str:
    """Return the channel-local score semantics for one match mode."""
    if not isinstance(mode, LexicalMatchMode):
        raise TypeError("mode must be a LexicalMatchMode.")
    return (
        LEXICAL_IDENTIFIER_SCORE_LABEL
        if mode is LexicalMatchMode.IDENTIFIER
        else LEXICAL_SCORE_LABEL
    )


def expose_bm25_score(raw_score: float) -> float:
    """Convert SQLite's lower-is-better BM25 value to higher-is-better."""
    if not isinstance(raw_score, float) or not math.isfinite(raw_score):
        raise TypeError("raw_score must be a finite float.")
    return -raw_score


def identifier_score(occurrence_count: int) -> int:
    """Return the higher-is-better exact-identifier occurrence score."""
    if type(occurrence_count) is not int:
        raise TypeError("occurrence_count must be an integer.")
    if occurrence_count < 0:
        raise InvalidLexicalContractError("occurrence_count must not be negative.")
    return occurrence_count


def lexical_order_key(score: int | float, record_id: RecordId) -> tuple[float, str]:
    """Return the deterministic descending-score, ascending-ID sort key."""
    if isinstance(score, bool) or type(score) not in (int, float):
        raise TypeError("score must be a finite number.")
    if not math.isfinite(score):
        raise InvalidLexicalContractError("score must be finite.")
    if not isinstance(record_id, RecordId):
        raise TypeError("record_id must be a RecordId.")
    return (-float(score), record_id.as_str())


def derive_lexical_generation_id(
    corpus_id: CorpusId,
    corpus_version: CorpusVersionId,
    configuration: LexicalIndexConfiguration,
    source_manifest_digest: str,
) -> str:
    """Derive the stable, path-free identity of one lexical generation."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(corpus_version, CorpusVersionId):
        raise TypeError("corpus_version must be a CorpusVersionId.")
    if not isinstance(configuration, LexicalIndexConfiguration):
        raise TypeError("configuration must be a LexicalIndexConfiguration.")
    _validate_sha256(source_manifest_digest, "source_manifest_digest")
    payload: JsonObject = {
        "component_kind": LEXICAL_COMPONENT_KIND,
        "configuration_digest": configuration.configuration_identity.digest,
        "contract_version": LEXICAL_CONTRACT_VERSION,
        "corpus_id": corpus_id.as_str(),
        "corpus_version": corpus_version.as_str(),
        "format_version": LEXICAL_FORMAT_VERSION,
        "source_manifest_digest": source_manifest_digest,
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return f"lexical-{digest}"
