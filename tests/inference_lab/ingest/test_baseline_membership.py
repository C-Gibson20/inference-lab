"""Tests for deterministic baseline membership-change contracts."""

from typing import cast

import pytest

from inference_lab.ingest import (
    BaselineDocumentOutcome,
    BaselineDocumentOutcomeKind,
    BaselineFailure,
    BaselineFailureCategory,
    BaselineFailureScope,
    BaselineFailureSeverity,
    BaselineMembershipApplicationError,
    BaselineMembershipChange,
    BaselineMembershipChangeKind,
    BaselineMembershipPlanningError,
    BaselineStage,
    BaselineStageDisposition,
    apply_baseline_membership_plan,
    plan_baseline_membership,
)
from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
)
from inference_lab.storage import MetadataTransaction, StorageConflictError


def _membership(
    *,
    corpus_id: CorpusId | None = None,
    document_id: DocumentId | None = None,
    document_version_id: DocumentVersionId | None = None,
) -> CorpusDocumentVersionMembership:
    """Build one valid corpus membership fixture."""
    return CorpusDocumentVersionMembership(
        corpus_id=corpus_id or CorpusId.generate(),
        document_id=document_id or DocumentId.generate(),
        document_version_id=document_version_id or DocumentVersionId.generate(),
    )


def test_membership_change_accepts_add() -> None:
    membership = _membership()

    change = BaselineMembershipChange(
        document_id=membership.document_id,
        kind=BaselineMembershipChangeKind.ADD,
        before=None,
        after=membership,
    )

    assert change.after == membership


def test_membership_change_accepts_retain() -> None:
    membership = _membership()

    change = BaselineMembershipChange(
        document_id=membership.document_id,
        kind=BaselineMembershipChangeKind.RETAIN,
        before=membership,
        after=membership,
    )

    assert change.before == change.after


def test_membership_change_accepts_replace() -> None:
    before = _membership()
    after = _membership(
        corpus_id=before.corpus_id,
        document_id=before.document_id,
    )

    change = BaselineMembershipChange(
        document_id=before.document_id,
        kind=BaselineMembershipChangeKind.REPLACE,
        before=before,
        after=after,
    )

    assert change.before == before
    assert change.after == after


def test_membership_change_rejects_add_with_existing_parent() -> None:
    membership = _membership()

    with pytest.raises(BaselineMembershipPlanningError) as error:
        BaselineMembershipChange(
            document_id=membership.document_id,
            kind=BaselineMembershipChangeKind.ADD,
            before=membership,
            after=membership,
        )

    assert error.value.category is BaselineFailureCategory.CORRUPT_STATE


def test_membership_change_rejects_cross_corpus_replacement() -> None:
    before = _membership()
    after = _membership(document_id=before.document_id)

    with pytest.raises(BaselineMembershipPlanningError) as error:
        BaselineMembershipChange(
            document_id=before.document_id,
            kind=BaselineMembershipChangeKind.REPLACE,
            before=before,
            after=after,
        )

    assert error.value.category is BaselineFailureCategory.CORRUPT_STATE


def test_membership_planning_error_rejects_unrelated_category() -> None:
    with pytest.raises(ValueError):
        BaselineMembershipPlanningError(BaselineFailureCategory.PARSE_FAILED)


def test_membership_planning_error_is_payload_free() -> None:
    error = BaselineMembershipPlanningError(
        BaselineFailureCategory.MEMBERSHIP_CONFLICT,
    )

    assert str(error) == "membership_conflict"
    assert "source" not in str(error)
    assert "text" not in str(error)


def _outcome(
    *,
    ordinal: int,
    kind: BaselineDocumentOutcomeKind,
    document_id: DocumentId | None,
    document_version_id: DocumentVersionId | None,
    parse_disposition: BaselineStageDisposition = BaselineStageDisposition.COMPLETED,
    chunk_disposition: BaselineStageDisposition = BaselineStageDisposition.COMPLETED,
    failures: tuple[BaselineFailure, ...] = (),
) -> BaselineDocumentOutcome:
    """Build a document outcome fixture with explicit stage evidence."""
    return BaselineDocumentOutcome(
        ordinal=ordinal,
        kind=kind,
        document_id=document_id,
        document_version_id=document_version_id,
        parse_disposition=parse_disposition,
        chunk_disposition=chunk_disposition,
        failures=failures,
    )


def _document_failure() -> BaselineFailure:
    """Build a recoverable document-scoped failure fixture."""
    return BaselineFailure(
        stage=BaselineStage.CHUNK,
        category=BaselineFailureCategory.CHUNK_FAILED,
        scope=BaselineFailureScope.DOCUMENT,
        severity=BaselineFailureSeverity.RECOVERABLE,
        retryable=False,
    )


def test_planner_adds_successful_new_document() -> None:
    corpus_id = CorpusId.generate()
    document_id = DocumentId.generate()
    version_id = DocumentVersionId.generate()

    plan = plan_baseline_membership(
        corpus_id,
        (),
        (
            _outcome(
                ordinal=0,
                kind=BaselineDocumentOutcomeKind.NEW_DOCUMENT,
                document_id=document_id,
                document_version_id=version_id,
            ),
        ),
    )

    assert plan.final_memberships == (
        _membership(
            corpus_id=corpus_id,
            document_id=document_id,
            document_version_id=version_id,
        ),
    )
    assert plan.changes[0].kind is BaselineMembershipChangeKind.ADD


def test_planner_replaces_successful_changed_document() -> None:
    parent = _membership()
    replacement = DocumentVersionId.generate()

    plan = plan_baseline_membership(
        parent.corpus_id,
        (parent,),
        (
            _outcome(
                ordinal=0,
                kind=BaselineDocumentOutcomeKind.CHANGED_VERSION,
                document_id=parent.document_id,
                document_version_id=replacement,
            ),
        ),
    )

    assert plan.final_memberships[0].document_version_id == replacement
    assert plan.changes[0].kind is BaselineMembershipChangeKind.REPLACE


def test_planner_retains_parent_for_failed_changed_document() -> None:
    parent = _membership()

    plan = plan_baseline_membership(
        parent.corpus_id,
        (parent,),
        (
            _outcome(
                ordinal=0,
                kind=BaselineDocumentOutcomeKind.FAILED,
                document_id=parent.document_id,
                document_version_id=DocumentVersionId.generate(),
                parse_disposition=BaselineStageDisposition.COMPLETED,
                chunk_disposition=BaselineStageDisposition.FAILED,
                failures=(_document_failure(),),
            ),
        ),
    )

    assert plan.final_memberships == (parent,)
    assert plan.changes[0].kind is BaselineMembershipChangeKind.RETAIN


def test_planner_does_not_add_failed_new_document() -> None:
    corpus_id = CorpusId.generate()

    plan = plan_baseline_membership(
        corpus_id,
        (),
        (
            _outcome(
                ordinal=0,
                kind=BaselineDocumentOutcomeKind.FAILED,
                document_id=DocumentId.generate(),
                document_version_id=DocumentVersionId.generate(),
                parse_disposition=BaselineStageDisposition.COMPLETED,
                chunk_disposition=BaselineStageDisposition.FAILED,
                failures=(_document_failure(),),
            ),
        ),
    )

    assert plan.final_memberships == ()
    assert plan.changes == ()


def test_planner_rejects_incomplete_success_evidence() -> None:
    with pytest.raises(BaselineMembershipPlanningError) as error:
        plan_baseline_membership(
            CorpusId.generate(),
            (),
            (
                _outcome(
                    ordinal=0,
                    kind=BaselineDocumentOutcomeKind.NEW_DOCUMENT,
                    document_id=DocumentId.generate(),
                    document_version_id=DocumentVersionId.generate(),
                    chunk_disposition=BaselineStageDisposition.NOT_ATTEMPTED,
                ),
            ),
        )

    assert error.value.category is BaselineFailureCategory.CORRUPT_STATE


def test_planner_rejects_conflicting_duplicate_evidence() -> None:
    corpus_id = CorpusId.generate()
    document_id = DocumentId.generate()

    with pytest.raises(BaselineMembershipPlanningError) as error:
        plan_baseline_membership(
            corpus_id,
            (),
            (
                _outcome(
                    ordinal=0,
                    kind=BaselineDocumentOutcomeKind.NEW_DOCUMENT,
                    document_id=document_id,
                    document_version_id=DocumentVersionId.generate(),
                ),
                _outcome(
                    ordinal=1,
                    kind=BaselineDocumentOutcomeKind.NEW_DOCUMENT,
                    document_id=document_id,
                    document_version_id=DocumentVersionId.generate(),
                ),
            ),
        )

    assert error.value.category is BaselineFailureCategory.MEMBERSHIP_CONFLICT


def test_planner_is_independent_of_completion_order() -> None:
    corpus_id = CorpusId.generate()
    first_document = DocumentId.generate()
    second_document = DocumentId.generate()
    first_version = DocumentVersionId.generate()
    second_version = DocumentVersionId.generate()
    outcomes = (
        _outcome(
            ordinal=4,
            kind=BaselineDocumentOutcomeKind.NEW_DOCUMENT,
            document_id=first_document,
            document_version_id=first_version,
        ),
        _outcome(
            ordinal=1,
            kind=BaselineDocumentOutcomeKind.NEW_DOCUMENT,
            document_id=second_document,
            document_version_id=second_version,
        ),
    )

    first_plan = plan_baseline_membership(corpus_id, (), outcomes)
    second_plan = plan_baseline_membership(corpus_id, (), outcomes[::-1])

    assert second_plan == first_plan


class _FakeMembershipTransaction:
    """Minimal transaction double for atomic membership application tests."""

    def __init__(
        self,
        memberships: tuple[CorpusDocumentVersionMembership, ...],
        *,
        post_memberships: tuple[CorpusDocumentVersionMembership, ...] | None = None,
        fail_on_replace: bool = False,
    ) -> None:
        self.memberships = memberships
        self.post_memberships = post_memberships
        self.fail_on_replace = fail_on_replace
        self.write_count = 0
        self.read_count = 0

    def list_corpus_update_document_memberships(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
    ) -> tuple[CorpusDocumentVersionMembership, ...]:
        del corpus_id, corpus_version
        self.read_count += 1
        if self.read_count == 2 and self.post_memberships is not None:
            return self.post_memberships
        return self.memberships

    def add_corpus_update_document_membership(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
        membership: CorpusDocumentVersionMembership,
    ) -> None:
        del corpus_id, corpus_version
        self.write_count += 1
        self.memberships = tuple(
            sorted(
                (*self.memberships, membership),
                key=lambda value: value.document_id.as_str(),
            )
        )

    def replace_corpus_update_document_membership(
        self,
        corpus_id: CorpusId,
        corpus_version: CorpusVersionId,
        expected_membership: CorpusDocumentVersionMembership,
        replacement_membership: CorpusDocumentVersionMembership,
    ) -> None:
        del corpus_id, corpus_version
        self.write_count += 1
        if self.fail_on_replace:
            raise StorageConflictError()
        if expected_membership not in self.memberships:
            raise StorageConflictError()
        self.memberships = tuple(
            sorted(
                (
                    replacement_membership if value == expected_membership else value
                    for value in self.memberships
                ),
                key=lambda value: value.document_id.as_str(),
            )
        )


def test_application_applies_add_and_returns_canonical_membership() -> None:
    corpus_id = CorpusId.generate()
    document_id = DocumentId.generate()
    version_id = DocumentVersionId.generate()
    plan = plan_baseline_membership(
        corpus_id,
        (),
        (
            _outcome(
                ordinal=0,
                kind=BaselineDocumentOutcomeKind.NEW_DOCUMENT,
                document_id=document_id,
                document_version_id=version_id,
            ),
        ),
    )
    transaction = _FakeMembershipTransaction(())

    persisted = apply_baseline_membership_plan(
        cast(MetadataTransaction, transaction),
        CorpusVersionId.generate(),
        plan,
    )

    assert persisted == plan.final_memberships
    assert transaction.write_count == 1


def test_application_rejects_stale_proposed_membership_before_writing() -> None:
    parent = _membership()
    replacement = _membership(
        corpus_id=parent.corpus_id,
        document_id=parent.document_id,
    )
    plan = plan_baseline_membership(
        parent.corpus_id,
        (parent,),
        (
            _outcome(
                ordinal=0,
                kind=BaselineDocumentOutcomeKind.CHANGED_VERSION,
                document_id=parent.document_id,
                document_version_id=replacement.document_version_id,
            ),
        ),
    )
    transaction = _FakeMembershipTransaction((replacement,))

    with pytest.raises(BaselineMembershipApplicationError) as error:
        apply_baseline_membership_plan(
            cast(MetadataTransaction, transaction),
            CorpusVersionId.generate(),
            plan,
        )

    assert error.value.category is BaselineFailureCategory.MEMBERSHIP_CONFLICT
    assert transaction.write_count == 0


def test_application_maps_storage_conflict_without_database_details() -> None:
    parent = _membership()
    replacement = _membership(
        corpus_id=parent.corpus_id,
        document_id=parent.document_id,
    )
    plan = plan_baseline_membership(
        parent.corpus_id,
        (parent,),
        (
            _outcome(
                ordinal=0,
                kind=BaselineDocumentOutcomeKind.CHANGED_VERSION,
                document_id=parent.document_id,
                document_version_id=replacement.document_version_id,
            ),
        ),
    )
    transaction = _FakeMembershipTransaction((parent,), fail_on_replace=True)

    with pytest.raises(BaselineMembershipApplicationError) as error:
        apply_baseline_membership_plan(
            cast(MetadataTransaction, transaction),
            CorpusVersionId.generate(),
            plan,
        )

    assert error.value.category is BaselineFailureCategory.MEMBERSHIP_CONFLICT
    assert "sqlite" not in str(error.value).lower()


def test_application_rejects_post_write_membership_mismatch_as_corruption() -> None:
    corpus_id = CorpusId.generate()
    document_id = DocumentId.generate()
    version_id = DocumentVersionId.generate()
    unexpected = _membership(corpus_id=corpus_id)
    plan = plan_baseline_membership(
        corpus_id,
        (),
        (
            _outcome(
                ordinal=0,
                kind=BaselineDocumentOutcomeKind.NEW_DOCUMENT,
                document_id=document_id,
                document_version_id=version_id,
            ),
        ),
    )
    transaction = _FakeMembershipTransaction((), post_memberships=(unexpected,))

    with pytest.raises(BaselineMembershipApplicationError) as error:
        apply_baseline_membership_plan(
            cast(MetadataTransaction, transaction),
            CorpusVersionId.generate(),
            plan,
        )

    assert error.value.category is BaselineFailureCategory.CORRUPT_STATE
