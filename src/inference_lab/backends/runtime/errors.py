"""Typed exceptions raised by model runtime backends."""

from typing import ClassVar


class BackendError(RuntimeError):
    """Base class for failures raised by a model runtime backend."""

    default_message: ClassVar[str] = "An unspecified backend error occurred."
    default_retryable: ClassVar[bool] = False

    def __init__(
        self, message: str | None = None, *, retryable: bool | None = None
    ) -> None:
        """Create an error with an optional caller-visible retry hint."""
        if message is not None:
            if not isinstance(message, str):
                raise TypeError("message must be a string or None")
            if not message:
                raise ValueError("message must not be empty")
        if retryable is not None and not isinstance(retryable, bool):
            raise TypeError("retryable must be a boolean or None")

        super().__init__(self.default_message if message is None else message)
        self.retryable = self.default_retryable if retryable is None else retryable


class BackendNotReadyError(BackendError):
    """Raised when inference is requested before loading completes."""

    default_message: ClassVar[str] = "The backend is not ready for requests."
    default_retryable: ClassVar[bool] = True


class BackendClosedError(BackendError):
    """Raised when inference is requested after backend shutdown."""

    default_message: ClassVar[str] = (
        "The backend has been closed and cannot accept requests."
    )


class BackendRequestError(BackendError):
    """Raised for invalid requests or unsupported options."""

    default_message: ClassVar[str] = "The backend request was invalid or unsupported."


class BackendExecutionError(BackendError):
    """Raised when execution fails for an otherwise valid request."""

    default_message: ClassVar[str] = "The backend failed while executing the request."


class BackendProtocolError(BackendExecutionError):
    """Raised when a backend violates the runtime stream contract."""

    default_message: ClassVar[str] = "The backend returned an invalid stream."


class BackendResourceError(BackendError):
    """Raised for capacity, memory, or unavailable-device failures."""

    default_message: ClassVar[str] = "The backend failed due to insufficient resources."
    default_retryable: ClassVar[bool] = True


class BackendCancelledError(BackendError):
    """Raised when cancellation must cross a backend boundary."""

    default_message: ClassVar[str] = "The backend operation was cancelled."
