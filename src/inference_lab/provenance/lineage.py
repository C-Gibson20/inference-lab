"""Immutable input references and directed edges for derived-record lineage."""

from dataclasses import dataclass
from typing import Self, TypeAlias, TypeVar

from inference_lab.provenance._validation import is_lowercase_slug
from inference_lab.provenance.errors import InvalidLineageReferenceError
from inference_lab.provenance.identifiers import (
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)


@dataclass(frozen=True, slots=True)
class InputRole:
    """Validated semantic role played by one derivation input."""

    value: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the input role value against the slug pattern."""
        if not isinstance(self.value, str):
            raise TypeError("input_role must be a string.")
        if not is_lowercase_slug(self.value):
            raise InvalidLineageReferenceError("input_role must be a lowercase slug.")

    @classmethod
    def from_str(cls, value: str) -> Self:
        """Reconstruct an input role from its canonical string value."""
        return cls(value)

    def as_str(self) -> str:
        """Return the canonical input-role value."""
        return self.value


def _validate_annotation(
    input_role: InputRole | None,
    input_ordinal: int | None,
) -> None:
    """Validate optional semantic role and order annotations."""
    if input_role is not None and not isinstance(input_role, InputRole):
        raise TypeError("input_role must be an InputRole or None.")
    if input_ordinal is not None and (
        isinstance(input_ordinal, bool) or not isinstance(input_ordinal, int)
    ):
        raise TypeError("input_ordinal must be an integer or None.")
    if input_ordinal is not None and input_ordinal < 0:
        raise InvalidLineageReferenceError(
            "input_ordinal must be greater than or equal to zero."
        )


def _annotation_key(
    input_role: InputRole | None,
    input_ordinal: int | None,
) -> tuple[str, int]:
    """Return a canonical key for optional input annotations."""
    return (
        input_role.as_str() if input_role is not None else "",
        input_ordinal if input_ordinal is not None else -1,
    )


@dataclass(frozen=True, slots=True)
class SourceAnchor:
    """Reference one immutable source span used by a derivation."""

    document_version_id: DocumentVersionId
    source_span_id: SourceSpanId
    input_role: InputRole | None = None
    input_ordinal: int | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the source anchor fields and optional annotations."""
        if not isinstance(self.document_version_id, DocumentVersionId):
            raise TypeError("document_version_id must be a DocumentVersionId.")
        if not isinstance(self.source_span_id, SourceSpanId):
            raise TypeError("source_span_id must be a SourceSpanId.")
        _validate_annotation(self.input_role, self.input_ordinal)

    def key(self) -> tuple[str, str, str, str, int]:
        """Return the canonical source-input key."""
        role, ordinal = _annotation_key(self.input_role, self.input_ordinal)
        return (
            "source",
            self.document_version_id.as_str(),
            self.source_span_id.as_str(),
            role,
            ordinal,
        )

    def identity_key(self) -> tuple[str, str]:
        """Return the immutable source identity independently of annotations."""
        return self.document_version_id.as_str(), self.source_span_id.as_str()


@dataclass(frozen=True, slots=True)
class ParentRecordReference:
    """Reference one immutable parent record used by a derivation."""

    record_id: RecordId
    input_role: InputRole | None = None
    input_ordinal: int | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the parent record reference fields and optional annotations."""
        if not isinstance(self.record_id, RecordId):
            raise TypeError("record_id must be a RecordId.")
        _validate_annotation(self.input_role, self.input_ordinal)

    def key(self) -> tuple[str, str, str, int]:
        """Return the canonical parent-input key."""
        role, ordinal = _annotation_key(self.input_role, self.input_ordinal)
        return ("parent", self.record_id.as_str(), role, ordinal)

    def identity_key(self) -> tuple[str]:
        """Return the immutable parent identity independently of annotations."""
        return (self.record_id.as_str(),)


LineageInput: TypeAlias = SourceAnchor | ParentRecordReference
_LineageInputT = TypeVar(
    "_LineageInputT",
    SourceAnchor,
    ParentRecordReference,
)


@dataclass(frozen=True, slots=True)
class LineageEdge:
    """Direct one immutable source or parent input towards a derived child."""

    input_reference: LineageInput
    child_record_id: RecordId

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the directed-edge fields."""
        if not isinstance(
            self.input_reference,
            (SourceAnchor, ParentRecordReference),
        ):
            raise TypeError(
                "input_reference must be a SourceAnchor or ParentRecordReference."
            )
        if not isinstance(self.child_record_id, RecordId):
            raise TypeError("child_record_id must be a RecordId.")

    def key(self) -> tuple[str, ...]:
        """Return the canonical directed-edge key."""
        input_key = tuple(str(value) for value in self.input_reference.key())
        return (*input_key, self.child_record_id.as_str())


@dataclass(frozen=True, slots=True)
class LineageInputs:
    """Immutable, unique, canonically ordered direct derivation inputs."""

    source_anchors: tuple[SourceAnchor, ...] = ()
    parent_records: tuple[ParentRecordReference, ...] = ()

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the input collections and their contents."""
        self._validate_collection_types()
        if not self.source_anchors and not self.parent_records:
            raise InvalidLineageReferenceError(
                "lineage inputs require at least one source anchor or parent record."
            )
        object.__setattr__(
            self,
            "source_anchors",
            self._canonicalise(self.source_anchors, "source_anchors"),
        )
        object.__setattr__(
            self,
            "parent_records",
            self._canonicalise(self.parent_records, "parent_records"),
        )

    def _validate_collection_types(self) -> None:
        """Require immutable collections containing the correct reference types."""
        if not isinstance(self.source_anchors, tuple):
            raise TypeError("source_anchors must be a tuple.")
        if any(not isinstance(anchor, SourceAnchor) for anchor in self.source_anchors):
            raise TypeError("source_anchors must contain only SourceAnchor values.")
        if not isinstance(self.parent_records, tuple):
            raise TypeError("parent_records must be a tuple.")
        if any(
            not isinstance(parent, ParentRecordReference)
            for parent in self.parent_records
        ):
            raise TypeError(
                "parent_records must contain only ParentRecordReference values."
            )

    @staticmethod
    def _canonicalise(
        references: tuple[_LineageInputT, ...],
        field_name: str,
    ) -> tuple[_LineageInputT, ...]:
        """Reject duplicates and return references in canonical key order."""
        identity_keys = tuple(reference.identity_key() for reference in references)
        if len(identity_keys) != len(set(identity_keys)):
            raise InvalidLineageReferenceError(
                f"{field_name} must not contain duplicates."
            )
        return tuple(sorted(references, key=lambda reference: reference.key()))

    def edges_to(self, child_record_id: RecordId) -> tuple[LineageEdge, ...]:
        """Return canonical input-to-child edges for one derived record."""
        if not isinstance(child_record_id, RecordId):
            raise TypeError("child_record_id must be a RecordId.")
        edges = tuple(
            LineageEdge(input_reference=reference, child_record_id=child_record_id)
            for reference in (*self.source_anchors, *self.parent_records)
        )
        return tuple(sorted(edges, key=lambda edge: edge.key()))
