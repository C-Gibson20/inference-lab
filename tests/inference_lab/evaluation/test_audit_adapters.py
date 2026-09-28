"""Focused EVAL-02 missing-parse observation behavior."""

from inference_lab.evaluation import (
    AuditMissingReason,
    AuditParseObservation,
    CorpusAuditScope,
    complete_parse_observations,
)
from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
)


def test_missing_parse_member_is_retained_in_canonical_scope_order() -> None:
    corpus_id = CorpusId("11111111-1111-4111-8111-111111111111")
    scope = CorpusAuditScope(
        corpus_id=corpus_id,
        corpus_version=CorpusVersionId("33333333-3333-4333-8333-333333333333"),
        memberships=(
            CorpusDocumentVersionMembership(
                corpus_id,
                DocumentId("44444444-4444-4444-8444-444444444444"),
                DocumentVersionId("55555555-5555-4555-8555-555555555555"),
            ),
            CorpusDocumentVersionMembership(
                corpus_id,
                DocumentId("77777777-7777-4777-8777-777777777777"),
                DocumentVersionId("88888888-8888-4888-8888-888888888888"),
            ),
        ),
    )
    observed = AuditParseObservation(
        membership=scope.memberships[1],
        missing_reason=AuditMissingReason.SOURCE_FAILURE,
    )

    completed = complete_parse_observations(scope, (observed,))

    assert tuple(item.membership for item in completed) == scope.memberships
    assert completed[0].missing_reason is AuditMissingReason.PARSE_NOT_ATTEMPTED
    assert completed[1] == observed
