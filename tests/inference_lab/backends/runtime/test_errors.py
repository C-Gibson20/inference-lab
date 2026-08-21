import pytest

from inference_lab.backends.runtime.errors import (
    BackendClosedError,
    BackendError,
    BackendNotReadyError,
    BackendResourceError,
    BackendRequestError,
)


@pytest.mark.parametrize(
    ("error_type", "retryable"),
    [
        (BackendNotReadyError, True),
        (BackendClosedError, False),
        (BackendRequestError, False),
        (BackendResourceError, True),
    ],
)
def test_backend_error_retry_defaults(
    error_type: type[BackendError], retryable: bool
) -> None:
    error = error_type()

    assert error.retryable is retryable
    assert str(error)


def test_retry_hint_can_be_overridden_by_caller() -> None:
    error = BackendRequestError(retryable=True)

    assert error.retryable is True
