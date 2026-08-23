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


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"message": 1}, TypeError, "message must be a string"),
        ({"message": ""}, ValueError, "message must not be empty"),
        ({"retryable": 1}, TypeError, "retryable must be a boolean"),
    ],
)
def test_backend_error_rejects_invalid_constructor_values(
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        BackendError(**kwargs)  # type: ignore[arg-type]
