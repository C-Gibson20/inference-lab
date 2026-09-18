"""Plan deterministic corpus-membership changes for baseline ingest."""

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import NoReturn

from inference_lab.ingest.baseline_contracts import (
    BaselineDocumentOutcome,
    BaselineDocumentOutcomeKind,
    BaselineFailureCategory,
    BaselineStageDisposition,
)
from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
)
from inference_lab.storage import (
    CorruptStorageError,
    MetadataTransaction,
    RepositoryClosedError,
    RepositoryUnavailableError,
    StorageConcurrencyError,
    StorageConflictError,
    StorageTransactionError,
)


class BaselineMembershipPlanningError(ValueError):
    """Represent a payload-free membership planning failure."""

    def __init__(self, category: BaselineFailureCategory) -> None:
        """Create a bounded planning error."""
        if category not in {
            BaselineFailureCategory.MEMBERSHIP_CONFLICT,
            BaselineFailureCategory.CORRUPT_STATE,
        }:
            raise ValueError(
                "membership planning errors require a membership or state category."
            )
        super().__init__(category.value)
        self.category = category


class BaselineMembershipApplicationError(RuntimeError):
    """Represent a bounded failure while applying a membership plan."""

    def __init__(self, category: BaselineFailureCategory) -> None:
        """Create a payload-free application error."""
        if category not in {
            BaselineFailureCategory.MEMBERSHIP_CONFLICT,
            BaselineFailureCategory.STORAGE_UNAVAILABLE,
            BaselineFailureCategory.CORRUPT_STATE,
        }:
            raise ValueError(
                "membership application errors require a storage-safe category."
            )
        super().__init__(category.value)
        self.category = category


class BaselineMembershipChangeKind(StrEnum):
    """Supported changes to one logical document's corpus membership."""

    RETAIN = "retain"
    ADD = "add"
    REPLACE = "replace"


@dataclass(frozen=True, slots=True)
class BaselineMembershipChange:
    """Describe one deterministic add, retain, or replacement."""

    document_id: DocumentId
    kind: BaselineMembershipChangeKind
    before: CorpusDocumentVersionMembership | None
    after: CorpusDocumentVersionMembership

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate identity ownership and operation-specific invariants."""
        if not isinstance(self.document_id, DocumentId):
            raise TypeError("document_id must be a DocumentId.")
        if not isinstance(self.kind, BaselineMembershipChangeKind):
            raise TypeError("kind must be a BaselineMembershipChangeKind.")
        if self.before is not None and not isinstance(
            self.before,
            CorpusDocumentVersionMembership,
        ):
            raise TypeError("before must be a membership or None.")
        if not isinstance(self.after, CorpusDocumentVersionMembership):
            raise TypeError("after must be a membership.")

        if self.after.document_id != self.document_id:
            raise BaselineMembershipPlanningError(
                BaselineFailureCategory.CORRUPT_STATE,
            )

        if self.before is not None:
            if self.before.document_id != self.document_id:
                raise BaselineMembershipPlanningError(
                    BaselineFailureCategory.CORRUPT_STATE,
                )
            if self.before.corpus_id != self.after.corpus_id:
                raise BaselineMembershipPlanningError(
                    BaselineFailureCategory.CORRUPT_STATE,
                )

        if self.kind is BaselineMembershipChangeKind.ADD:
            if self.before is not None:
                raise BaselineMembershipPlanningError(
                    BaselineFailureCategory.CORRUPT_STATE,
                )
        elif self.kind is BaselineMembershipChangeKind.RETAIN:
            if self.before is None or self.before != self.after:
                raise BaselineMembershipPlanningError(
                    BaselineFailureCategory.CORRUPT_STATE,
                )
        elif self.kind is BaselineMembershipChangeKind.REPLACE:
            if self.before is None or self.before == self.after:
                raise BaselineMembershipPlanningError(
                    BaselineFailureCategory.CORRUPT_STATE,
                )


@dataclass(frozen=True, slots=True)
class BaselineMembershipPlan:
    """Describe canonical parent-to-proposed membership state."""

    corpus_id: CorpusId
    parent_memberships: tuple[CorpusDocumentVersionMembership, ...]
    changes: tuple[BaselineMembershipChange, ...]
    final_memberships: tuple[CorpusDocumentVersionMembership, ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate ownership, uniqueness, and canonical ordering."""
        if not isinstance(self.corpus_id, CorpusId):
            raise TypeError("corpus_id must be a CorpusId.")
        for field_name, values in (
            ("parent_memberships", self.parent_memberships),
            ("final_memberships", self.final_memberships),
        ):
            if not isinstance(values, tuple):
                raise TypeError(f"{field_name} must be a tuple.")
            if any(
                not isinstance(value, CorpusDocumentVersionMembership)
                for value in values
            ):
                raise TypeError(f"{field_name} must contain membership values.")
            if any(value.corpus_id != self.corpus_id for value in values):
                raise BaselineMembershipPlanningError(
                    BaselineFailureCategory.CORRUPT_STATE,
                )
            keys = tuple(value.document_id.as_str() for value in values)
            if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
                raise BaselineMembershipPlanningError(
                    BaselineFailureCategory.CORRUPT_STATE,
                )

        if not isinstance(self.changes, tuple):
            raise TypeError("changes must be a tuple.")
        if any(
            not isinstance(change, BaselineMembershipChange) for change in self.changes
        ):
            raise TypeError("changes must contain BaselineMembershipChange values.")
        change_keys = tuple(change.document_id.as_str() for change in self.changes)
        if change_keys != tuple(sorted(change_keys)):
            raise BaselineMembershipPlanningError(
                BaselineFailureCategory.CORRUPT_STATE,
            )
        if len(change_keys) != len(set(change_keys)):
            raise BaselineMembershipPlanningError(
                BaselineFailureCategory.CORRUPT_STATE,
            )
        if any(change.after.corpus_id != self.corpus_id for change in self.changes):
            raise BaselineMembershipPlanningError(
                BaselineFailureCategory.CORRUPT_STATE,
            )

        expected_by_document = {
            membership.document_id: membership for membership in self.parent_memberships
        }
        for change in self.changes:
            current = expected_by_document.get(change.document_id)
            if change.kind is BaselineMembershipChangeKind.ADD:
                if current is not None:
                    raise BaselineMembershipPlanningError(
                        BaselineFailureCategory.CORRUPT_STATE,
                    )
            elif current != change.before:
                raise BaselineMembershipPlanningError(
                    BaselineFailureCategory.CORRUPT_STATE,
                )
            expected_by_document[change.document_id] = change.after

        expected_final = tuple(
            sorted(
                expected_by_document.values(),
                key=lambda membership: membership.document_id.as_str(),
            )
        )
        if expected_final != self.final_memberships:
            raise BaselineMembershipPlanningError(
                BaselineFailureCategory.CORRUPT_STATE,
            )

    def key(self) -> tuple[str, str]:
        """Return the stable plan identity."""
        return self.corpus_id.as_str(), self.final_membership_digest()

    def final_membership_digest(self) -> str:
        """Return the stable digest of the proposed membership identities."""
        payload = "|".join(
            f"{membership.document_id.as_str()}:{membership.document_version_id.as_str()}"
            for membership in self.final_memberships
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()


def _canonical_parent_memberships(
    corpus_id: CorpusId,
    memberships: tuple[CorpusDocumentVersionMembership, ...],
) -> tuple[CorpusDocumentVersionMembership, ...]:
    """Validate and canonically order the parent membership."""
    if not isinstance(memberships, tuple):
        raise TypeError("parent_memberships must be a tuple.")
    if any(
        not isinstance(membership, CorpusDocumentVersionMembership)
        for membership in memberships
    ):
        raise TypeError("parent_memberships must contain membership values.")
    if any(membership.corpus_id != corpus_id for membership in memberships):
        raise BaselineMembershipPlanningError(
            BaselineFailureCategory.CORRUPT_STATE,
        )
    ordered = tuple(
        sorted(
            memberships,
            key=lambda membership: membership.document_id.as_str(),
        )
    )
    document_ids = tuple(membership.document_id for membership in ordered)
    if len(document_ids) != len(set(document_ids)):
        raise BaselineMembershipPlanningError(
            BaselineFailureCategory.CORRUPT_STATE,
        )
    return ordered


def _successful_outcome(outcome: object) -> bool:
    """Return whether an outcome has complete parse and chunk stages."""
    return (
        isinstance(outcome, BaselineDocumentOutcome)
        and outcome.parse_disposition
        in {
            BaselineStageDisposition.REUSED,
            BaselineStageDisposition.COMPLETED,
        }
        and outcome.chunk_disposition
        in {
            BaselineStageDisposition.REUSED,
            BaselineStageDisposition.COMPLETED,
        }
    )


def _outcome_membership(
    corpus_id: CorpusId,
    document_id: DocumentId,
    document_version_id: DocumentVersionId | None,
) -> CorpusDocumentVersionMembership:
    """Build one membership or fail closed for an incomplete outcome."""
    if not isinstance(document_version_id, DocumentVersionId):
        raise BaselineMembershipPlanningError(
            BaselineFailureCategory.CORRUPT_STATE,
        )
    return CorpusDocumentVersionMembership(
        corpus_id=corpus_id,
        document_id=document_id,
        document_version_id=document_version_id,
    )


_MUTATING_OUTCOME_KINDS = frozenset(
    {
        BaselineDocumentOutcomeKind.UNCHANGED_REUSED,
        BaselineDocumentOutcomeKind.NEW_DOCUMENT,
        BaselineDocumentOutcomeKind.CHANGED_VERSION,
        BaselineDocumentOutcomeKind.BYTE_IDENTICAL_DUPLICATE,
    }
)


def _raise_membership_conflict() -> NoReturn:
    """Raise the bounded error used for incompatible membership evidence."""
    raise BaselineMembershipPlanningError(
        BaselineFailureCategory.MEMBERSHIP_CONFLICT,
    )


def _validate_outcomes(
    outcomes: tuple[BaselineDocumentOutcome, ...],
) -> tuple[BaselineDocumentOutcome, ...]:
    """Validate and canonically order document outcomes."""
    if not isinstance(outcomes, tuple):
        raise TypeError("outcomes must be a tuple.")
    if any(not isinstance(outcome, BaselineDocumentOutcome) for outcome in outcomes):
        raise TypeError("outcomes must contain BaselineDocumentOutcome values.")
    ordered = tuple(sorted(outcomes, key=BaselineDocumentOutcome.key))
    ordinals = tuple(outcome.ordinal for outcome in ordered)
    if len(ordinals) != len(set(ordinals)):
        raise BaselineMembershipPlanningError(
            BaselineFailureCategory.CORRUPT_STATE,
        )
    return ordered


def _outcome_signature(
    outcome: BaselineDocumentOutcome,
) -> tuple[BaselineDocumentOutcomeKind, DocumentVersionId | None]:
    """Return the membership-relevant identity of one document outcome."""
    return outcome.kind, outcome.document_version_id


def plan_baseline_membership(
    corpus_id: CorpusId,
    parent_memberships: tuple[CorpusDocumentVersionMembership, ...],
    outcomes: tuple[BaselineDocumentOutcome, ...],
) -> BaselineMembershipPlan:
    """Plan a deterministic proposed membership from successful outcomes.

    The parent is the starting point. Failed, ambiguous, conflicting, and
    unsupported outcomes leave it unchanged; successful actionable outcomes
    add or replace exactly one document version. Replaying the same successful
    evidence is idempotent, while incompatible evidence fails closed.
    """
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")

    parent = _canonical_parent_memberships(corpus_id, parent_memberships)
    ordered_outcomes = _validate_outcomes(outcomes)
    parent_by_document = {membership.document_id: membership for membership in parent}
    final_by_document = dict(parent_by_document)
    change_by_document: dict[DocumentId, BaselineMembershipChange] = {
        membership.document_id: BaselineMembershipChange(
            document_id=membership.document_id,
            kind=BaselineMembershipChangeKind.RETAIN,
            before=membership,
            after=membership,
        )
        for membership in parent
    }
    actionable: dict[DocumentId, BaselineDocumentOutcome] = {}

    for outcome in ordered_outcomes:
        if outcome.kind not in _MUTATING_OUTCOME_KINDS:
            continue
        if outcome.document_id is None or not _successful_outcome(outcome):
            raise BaselineMembershipPlanningError(
                BaselineFailureCategory.CORRUPT_STATE,
            )

        prior = actionable.get(outcome.document_id)
        if prior is not None:
            if _outcome_signature(prior) != _outcome_signature(outcome):
                _raise_membership_conflict()
            continue
        actionable[outcome.document_id] = outcome

    for document_id in sorted(actionable, key=DocumentId.as_str):
        outcome = actionable[document_id]
        membership = _outcome_membership(
            corpus_id,
            document_id,
            outcome.document_version_id,
        )
        previous = parent_by_document.get(document_id)

        if outcome.kind in {
            BaselineDocumentOutcomeKind.NEW_DOCUMENT,
            BaselineDocumentOutcomeKind.BYTE_IDENTICAL_DUPLICATE,
        }:
            if previous is None:
                change = BaselineMembershipChange(
                    document_id=document_id,
                    kind=BaselineMembershipChangeKind.ADD,
                    before=None,
                    after=membership,
                )
            elif previous == membership:
                change = BaselineMembershipChange(
                    document_id=document_id,
                    kind=BaselineMembershipChangeKind.RETAIN,
                    before=previous,
                    after=previous,
                )
            else:
                _raise_membership_conflict()
        elif outcome.kind is BaselineDocumentOutcomeKind.CHANGED_VERSION:
            if previous is None:
                _raise_membership_conflict()
            if previous == membership:
                change = BaselineMembershipChange(
                    document_id=document_id,
                    kind=BaselineMembershipChangeKind.RETAIN,
                    before=previous,
                    after=previous,
                )
            else:
                change = BaselineMembershipChange(
                    document_id=document_id,
                    kind=BaselineMembershipChangeKind.REPLACE,
                    before=previous,
                    after=membership,
                )
        else:
            if previous != membership:
                _raise_membership_conflict()
            change = BaselineMembershipChange(
                document_id=document_id,
                kind=BaselineMembershipChangeKind.RETAIN,
                before=previous,
                after=membership,
            )

        final_by_document[document_id] = change.after
        change_by_document[document_id] = change

    final_memberships = tuple(
        sorted(
            final_by_document.values(),
            key=lambda membership: membership.document_id.as_str(),
        )
    )
    changes = tuple(
        sorted(
            change_by_document.values(),
            key=lambda change: change.document_id.as_str(),
        )
    )
    return BaselineMembershipPlan(
        corpus_id=corpus_id,
        parent_memberships=parent,
        changes=changes,
        final_memberships=final_memberships,
    )


def apply_baseline_membership_plan(
    transaction: MetadataTransaction,
    proposed_corpus_version: CorpusVersionId,
    plan: BaselineMembershipPlan,
) -> tuple[CorpusDocumentVersionMembership, ...]:
    """Apply a plan atomically and return the canonical staged membership.

    The caller owns the surrounding transaction context. The preflight read
    prevents a stale proposal from receiving partial writes, and the post-write
    read proves that the persisted proposal exactly matches the pure plan.
    """
    if not isinstance(proposed_corpus_version, CorpusVersionId):
        raise TypeError("proposed_corpus_version must be a CorpusVersionId.")
    if not isinstance(plan, BaselineMembershipPlan):
        raise TypeError("plan must be a BaselineMembershipPlan.")

    try:
        current = transaction.list_corpus_update_document_memberships(
            plan.corpus_id,
            proposed_corpus_version,
        )
        if current != plan.parent_memberships:
            raise BaselineMembershipApplicationError(
                BaselineFailureCategory.MEMBERSHIP_CONFLICT,
            )

        for change in plan.changes:
            if change.kind is BaselineMembershipChangeKind.ADD:
                transaction.add_corpus_update_document_membership(
                    plan.corpus_id,
                    proposed_corpus_version,
                    change.after,
                )
            elif change.kind is BaselineMembershipChangeKind.REPLACE:
                if change.before is None:
                    raise BaselineMembershipApplicationError(
                        BaselineFailureCategory.CORRUPT_STATE,
                    )
                transaction.replace_corpus_update_document_membership(
                    plan.corpus_id,
                    proposed_corpus_version,
                    change.before,
                    change.after,
                )

        persisted = transaction.list_corpus_update_document_memberships(
            plan.corpus_id,
            proposed_corpus_version,
        )
        if persisted != plan.final_memberships:
            raise BaselineMembershipApplicationError(
                BaselineFailureCategory.CORRUPT_STATE,
            )
        return persisted
    except BaselineMembershipApplicationError:
        raise
    except StorageConflictError as error:
        raise BaselineMembershipApplicationError(
            BaselineFailureCategory.MEMBERSHIP_CONFLICT,
        ) from error
    except CorruptStorageError as error:
        raise BaselineMembershipApplicationError(
            BaselineFailureCategory.CORRUPT_STATE,
        ) from error
    except (
        RepositoryClosedError,
        RepositoryUnavailableError,
        StorageConcurrencyError,
        StorageTransactionError,
    ) as error:
        raise BaselineMembershipApplicationError(
            BaselineFailureCategory.STORAGE_UNAVAILABLE,
        ) from error
