"""Pure validation for complete, storage-neutral lineage bundles."""

from dataclasses import dataclass
from typing import Self

from inference_lab.provenance.derived import DerivedRecordMetadata
from inference_lab.provenance.errors import InvalidLineageGraphError
from inference_lab.provenance.identifiers import (
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.provenance.lineage import (
    LineageEdge,
    ParentRecordReference,
    SourceAnchor,
)
from inference_lab.provenance.versioning import validate_lineage_schema_version


@dataclass(frozen=True, slots=True)
class KnownSourceSpan:
    """Catalogue the document version that owns one immutable source span."""

    document_version_id: DocumentVersionId
    source_span_id: SourceSpanId

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the source span is paired with a document version."""
        if not isinstance(self.document_version_id, DocumentVersionId):
            raise TypeError("document_version_id must be a DocumentVersionId.")
        if not isinstance(self.source_span_id, SourceSpanId):
            raise TypeError("source_span_id must be a SourceSpanId.")

    def key(self) -> tuple[str, str]:
        """Return the canonical source-catalogue key."""
        return self.source_span_id.as_str(), self.document_version_id.as_str()


@dataclass(frozen=True, slots=True)
class LineageBundle:
    """Complete records, source catalogue, and edges for lineage validation."""

    source_spans: tuple[KnownSourceSpan, ...] = ()
    records: tuple[DerivedRecordMetadata, ...] = ()
    edges: tuple[LineageEdge, ...] = ()

    def __post_init__(self) -> None:
        self._validate()
        object.__setattr__(
            self,
            "source_spans",
            tuple(sorted(self.source_spans, key=lambda source: source.key())),
        )
        object.__setattr__(
            self,
            "records",
            tuple(
                sorted(
                    self.records,
                    key=lambda record: record.record_id.as_str(),
                )
            ),
        )
        object.__setattr__(
            self,
            "edges",
            tuple(sorted(self.edges, key=lambda edge: edge.key())),
        )

    def _validate(self) -> None:
        """Validate the bundle's immutable collection boundary."""
        self._validate_collection_types()

    def _validate_collection_types(self) -> None:
        """Require immutable collections with strict member types."""
        if not isinstance(self.source_spans, tuple):
            raise TypeError("source_spans must be a tuple.")
        if any(not isinstance(source, KnownSourceSpan) for source in self.source_spans):
            raise TypeError("source_spans must contain only KnownSourceSpan values.")
        if not isinstance(self.records, tuple):
            raise TypeError("records must be a tuple.")
        if any(
            not isinstance(record, DerivedRecordMetadata) for record in self.records
        ):
            raise TypeError("records must contain only DerivedRecordMetadata values.")
        if not isinstance(self.edges, tuple):
            raise TypeError("edges must be a tuple.")
        if any(not isinstance(edge, LineageEdge) for edge in self.edges):
            raise TypeError("edges must contain only LineageEdge values.")

    @classmethod
    def from_records(
        cls,
        *,
        source_spans: tuple[KnownSourceSpan, ...],
        records: tuple[DerivedRecordMetadata, ...],
    ) -> Self:
        """Build a bundle whose edges exactly reflect each record's inputs."""
        if not isinstance(records, tuple):
            raise TypeError("records must be a tuple.")
        if any(not isinstance(record, DerivedRecordMetadata) for record in records):
            raise TypeError("records must contain only DerivedRecordMetadata values.")
        edges = tuple(
            edge
            for record in records
            for edge in record.inputs.edges_to(record.record_id)
        )
        return cls(source_spans=source_spans, records=records, edges=edges)


def validate_lineage_bundle(bundle: LineageBundle) -> LineageBundle:
    """Validate a complete lineage graph and return it unchanged."""
    if not isinstance(bundle, LineageBundle):
        raise TypeError("bundle must be a LineageBundle.")

    _reject_duplicate_catalogue_entries(bundle)
    records_by_id = {record.record_id: record for record in bundle.records}
    sources_by_id = {
        source.source_span_id: source.document_version_id
        for source in bundle.source_spans
    }
    _validate_schema_versions(bundle.records)
    _validate_record_references(bundle.records, records_by_id, sources_by_id)
    _validate_edges(bundle, records_by_id, sources_by_id)
    _validate_parent_cycles(bundle.records, records_by_id)
    _validate_supersession(bundle.records, records_by_id)
    _validate_source_termination(bundle.records, records_by_id)
    return bundle


def _reject_duplicate_catalogue_entries(bundle: LineageBundle) -> None:
    """Reject duplicate record, source-span, and directed-edge identities."""
    record_ids = tuple(record.record_id for record in bundle.records)
    if len(record_ids) != len(set(record_ids)):
        raise InvalidLineageGraphError("lineage bundle contains duplicate records.")

    source_ids = tuple(source.source_span_id for source in bundle.source_spans)
    if len(source_ids) != len(set(source_ids)):
        raise InvalidLineageGraphError(
            "lineage bundle contains duplicate source-span entries."
        )

    edge_keys = tuple(edge.key() for edge in bundle.edges)
    if len(edge_keys) != len(set(edge_keys)):
        raise InvalidLineageGraphError("lineage bundle contains duplicate edges.")


def _validate_schema_versions(records: tuple[DerivedRecordMetadata, ...]) -> None:
    """Require every record to use a schema supported by this build."""
    for record in records:
        validate_lineage_schema_version(record.lineage_schema_version)


def _validate_record_references(
    records: tuple[DerivedRecordMetadata, ...],
    records_by_id: dict[RecordId, DerivedRecordMetadata],
    sources_by_id: dict[SourceSpanId, DocumentVersionId],
) -> None:
    """Validate every direct source and parent reference from record metadata."""
    for record in records:
        for anchor in record.inputs.source_anchors:
            _validate_source_anchor(anchor, sources_by_id)
        for parent in record.inputs.parent_records:
            _validate_parent_reference(parent, record.record_id, records_by_id)


def _validate_source_anchor(
    anchor: SourceAnchor,
    sources_by_id: dict[SourceSpanId, DocumentVersionId],
) -> None:
    """Require a known source span paired with its owning document version."""
    document_version_id = sources_by_id.get(anchor.source_span_id)
    if document_version_id is None:
        raise InvalidLineageGraphError("lineage references a missing source span.")
    if document_version_id != anchor.document_version_id:
        raise InvalidLineageGraphError(
            "source span is paired with the wrong document version."
        )


def _validate_parent_reference(
    parent: ParentRecordReference,
    child_record_id: RecordId,
    records_by_id: dict[RecordId, DerivedRecordMetadata],
) -> None:
    """Require a known, non-self parent record."""
    if parent.record_id not in records_by_id:
        raise InvalidLineageGraphError("lineage references a missing parent record.")
    if parent.record_id == child_record_id:
        raise InvalidLineageGraphError("a derived record cannot be its own parent.")


def _validate_edges(
    bundle: LineageBundle,
    records_by_id: dict[RecordId, DerivedRecordMetadata],
    sources_by_id: dict[SourceSpanId, DocumentVersionId],
) -> None:
    """Validate supplied edges and require an exact match to record inputs."""
    for edge in bundle.edges:
        if edge.child_record_id not in records_by_id:
            raise InvalidLineageGraphError("lineage edge references a missing child.")
        if isinstance(edge.input_reference, SourceAnchor):
            _validate_source_anchor(edge.input_reference, sources_by_id)
        else:
            _validate_parent_reference(
                edge.input_reference,
                edge.child_record_id,
                records_by_id,
            )

    expected_keys = {
        edge.key()
        for record in bundle.records
        for edge in record.inputs.edges_to(record.record_id)
    }
    supplied_keys = {edge.key() for edge in bundle.edges}
    if supplied_keys != expected_keys:
        raise InvalidLineageGraphError(
            "lineage edges do not exactly match derived-record inputs."
        )


def _validate_parent_cycles(
    records: tuple[DerivedRecordMetadata, ...],
    records_by_id: dict[RecordId, DerivedRecordMetadata],
) -> None:
    """Reject direct or indirect cycles in parent-record lineage."""
    completed: set[RecordId] = set()
    active: set[RecordId] = set()

    def visit(record_id: RecordId) -> None:
        if record_id in active:
            raise InvalidLineageGraphError("derived-record lineage contains a cycle.")
        if record_id in completed:
            return
        active.add(record_id)
        for parent in records_by_id[record_id].inputs.parent_records:
            visit(parent.record_id)
        active.remove(record_id)
        completed.add(record_id)

    for record in records:
        visit(record.record_id)


def _validate_supersession(
    records: tuple[DerivedRecordMetadata, ...],
    records_by_id: dict[RecordId, DerivedRecordMetadata],
) -> None:
    """Require compatible, existing, acyclic supersession chains."""
    successors: dict[RecordId, RecordId] = {}
    for record in records:
        predecessor_id = record.supersedes_record_id
        if predecessor_id is None:
            continue
        predecessor = records_by_id.get(predecessor_id)
        if predecessor is None:
            raise InvalidLineageGraphError(
                "supersession references a missing predecessor record."
            )
        if predecessor.record_kind != record.record_kind:
            raise InvalidLineageGraphError(
                "supersession must retain the derived record kind."
            )
        if predecessor_id in successors:
            raise InvalidLineageGraphError(
                "a superseded record cannot have multiple replacements."
            )
        successors[predecessor_id] = record.record_id

    completed: set[RecordId] = set()
    active: set[RecordId] = set()

    def visit(record_id: RecordId) -> None:
        if record_id in active:
            raise InvalidLineageGraphError("supersession lineage contains a cycle.")
        if record_id in completed:
            return
        active.add(record_id)
        successor_id = successors.get(record_id)
        if successor_id is not None:
            visit(successor_id)
        active.remove(record_id)
        completed.add(record_id)

    for record in records:
        visit(record.record_id)


def _validate_source_termination(
    records: tuple[DerivedRecordMetadata, ...],
    records_by_id: dict[RecordId, DerivedRecordMetadata],
) -> None:
    """Require every parent branch to terminate at original source evidence."""
    terminated: dict[RecordId, bool] = {}

    def terminates(record_id: RecordId) -> bool:
        cached = terminated.get(record_id)
        if cached is not None:
            return cached
        record = records_by_id[record_id]
        if record.inputs.parent_records:
            result = all(
                terminates(parent.record_id) for parent in record.inputs.parent_records
            )
        else:
            result = bool(record.inputs.source_anchors)
        terminated[record_id] = result
        return result

    if any(not terminates(record.record_id) for record in records):
        raise InvalidLineageGraphError(
            "derived-record lineage does not terminate at original source evidence."
        )
