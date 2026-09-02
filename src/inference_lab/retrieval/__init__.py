"""Channel-neutral, provenance-preserving retrieval value contracts."""

from inference_lab.retrieval.errors import (
    InvalidRetrievalContractError,
    InvalidRetrievalResultError,
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
from inference_lab.retrieval.values import RetrievalChannel, RetrievalLimit

__all__ = [
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
]
