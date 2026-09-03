"""SQLite cleanup that cannot replace an existing primary failure."""

import sqlite3


def close_preserving_primary(connection: sqlite3.Connection | None) -> bool:
    """Close a partial connection and report whether cleanup succeeded."""
    if connection is None:
        return True
    try:
        connection.close()
    except sqlite3.Error:
        return False
    return True


def rollback_preserving_primary(connection: sqlite3.Connection) -> bool:
    """Roll back active work and report whether cleanup succeeded."""
    if not connection.in_transaction:
        return True
    try:
        connection.rollback()
    except sqlite3.Error:
        return False
    return True
