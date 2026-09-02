"""Channel-neutral, provenance-preserving retrieval value contracts."""

from inference_lab.retrieval.errors import (
    InvalidRetrievalContractError,
    InvalidRetrievalResultError,
    InvalidRetrievalSerialisationError,
    UnsupportedRetrievalSchemaVersionError,
)
from inference_lab.retrieval.evidence import (
    CandidateKind,
    ChannelScore,
    EvidenceHit,
    EvidenceSource,
    RetrievalCandidate,
)
from inference_lab.retrieval.filters import (
    AllFilter,
    AnyFilter,
    ExactFilter,
    ExistsFilter,
    FilterExpression,
    FilterScalar,
    MissingFilter,
    NotFilter,
    OneOfFilter,
    RangeFilter,
)
from inference_lab.retrieval.query import RetrievalQuery
from inference_lab.retrieval.results import (
    EvidenceResult,
    FailureCategory,
    ResultStatus,
    RetrievalFailure,
)
from inference_lab.retrieval.serialisation import (
    deserialise_evidence_result,
    deserialise_query,
    serialise_evidence_result,
    serialise_query,
)
from inference_lab.retrieval.values import RetrievalChannel, RetrievalLimit
from inference_lab.retrieval.versioning import (
    CURRENT_RETRIEVAL_SCHEMA_VERSION,
    SUPPORTED_RETRIEVAL_SCHEMA_VERSIONS,
)

__all__ = [
    "CURRENT_RETRIEVAL_SCHEMA_VERSION",
    "SUPPORTED_RETRIEVAL_SCHEMA_VERSIONS",
    "AllFilter",
    "AnyFilter",
    "CandidateKind",
    "ChannelScore",
    "EvidenceHit",
    "EvidenceResult",
    "EvidenceSource",
    "ExactFilter",
    "ExistsFilter",
    "FailureCategory",
    "FilterExpression",
    "FilterScalar",
    "InvalidRetrievalContractError",
    "InvalidRetrievalResultError",
    "InvalidRetrievalSerialisationError",
    "MissingFilter",
    "NotFilter",
    "OneOfFilter",
    "RangeFilter",
    "ResultStatus",
    "RetrievalCandidate",
    "RetrievalChannel",
    "RetrievalFailure",
    "RetrievalLimit",
    "RetrievalQuery",
    "UnsupportedRetrievalSchemaVersionError",
    "deserialise_evidence_result",
    "deserialise_query",
    "serialise_evidence_result",
    "serialise_query",
]
