"""Public SQLite transaction input boundaries and rollback guarantees."""

from pathlib import Path
from typing import cast

import pytest

from inference_lab.provenance import CorpusId, CorpusMetadata
from tests.inference_lab.storage.fixtures import open_repository
from tests.inference_lab.storage.input_failure_fixtures import (
    _CORPUS,
    _CORPUS_ID,
    _invalid,
)


def test_bad_repository_read_rolls_back_prior_transaction_writes(
    tmp_path: Path,
) -> None:
    repository = open_repository(tmp_path / "metadata.sqlite3")
    try:
        with pytest.raises(TypeError, match="corpus_id"):
            with repository.transaction() as transaction:
                transaction.add_corpus(_CORPUS)
                transaction.get_corpus(_invalid(CorpusId))
        assert repository.get_corpus(_CORPUS_ID) is None
    finally:
        repository.close()


def test_bad_repository_write_rolls_back_prior_transaction_writes(
    tmp_path: Path,
) -> None:
    repository = open_repository(tmp_path / "metadata.sqlite3")
    try:
        with pytest.raises(TypeError, match="corpus"):
            with repository.transaction() as transaction:
                transaction.add_corpus(_CORPUS)
                transaction.add_corpus(cast(CorpusMetadata, object()))
        assert repository.get_corpus(_CORPUS_ID) is None
    finally:
        repository.close()
