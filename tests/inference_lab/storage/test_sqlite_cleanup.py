"""Tests for SQLite best-effort cleanup."""

import sqlite3
from collections.abc import Callable

import pytest

from inference_lab.storage._sqlite_cleanup import (
    close_preserving_primary,
    rollback_preserving_primary,
)

type _CleanupOperation = Callable[[sqlite3.Connection], bool]


class _CleanupFailingConnection(sqlite3.Connection):
    def close(self) -> None:
        raise sqlite3.OperationalError("private-close-payload")

    def rollback(self) -> None:
        raise sqlite3.OperationalError("private-rollback-payload")


@pytest.mark.parametrize(
    ("operation", "begin_transaction"),
    [
        (close_preserving_primary, False),
        (rollback_preserving_primary, True),
    ],
    ids=("close", "rollback"),
)
def test_cleanup_failure_does_not_replace_primary_exception(
    operation: _CleanupOperation,
    begin_transaction: bool,
) -> None:
    connection = sqlite3.connect(":memory:", factory=_CleanupFailingConnection)
    if begin_transaction:
        connection.execute("BEGIN")
    primary = RuntimeError("private-primary-payload")

    try:
        with pytest.raises(RuntimeError) as raised:
            try:
                raise primary
            except RuntimeError:
                assert operation(connection) is False
                raise

        assert raised.value is primary
    finally:
        if connection.in_transaction:
            sqlite3.Connection.rollback(connection)
        sqlite3.Connection.close(connection)
