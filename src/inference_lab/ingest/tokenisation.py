"""Versioned local tokenisation contracts for ingest chunking."""

import hashlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, cast, runtime_checkable

from inference_lab._validation import is_local_path_shape, is_strict_integer
from inference_lab.ingest.chunking import TokeniserIdentity


class ChunkTokeniserError(ValueError):
    """Report a bounded, payload-free tokeniser contract failure."""

    def __init__(self, category: str) -> None:
        """Create a tokeniser failure without retaining paths or payloads."""
        super().__init__(category)
        self.category = category


@dataclass(frozen=True, slots=True)
class TokenOffset:
    """Locate one token in Python Unicode code-point coordinates."""

    start: int
    end: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the token offsets are well-typed and non-empty."""
        if not is_strict_integer(self.start) or not is_strict_integer(self.end):
            raise TypeError("token offsets must be integers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("token offsets must be non-empty half-open ranges")


@dataclass(frozen=True, slots=True)
class TokenisedText:
    """Validated token IDs and offsets for one exact input string."""

    token_ids: tuple[int, ...]
    offsets: tuple[TokenOffset, ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not isinstance(self.token_ids, tuple) or not isinstance(self.offsets, tuple):
            raise TypeError("tokenised output collections must be tuples")
        if len(self.token_ids) != len(self.offsets):
            raise ValueError("token IDs and offsets must have equal length")
        if any(not is_strict_integer(token_id) for token_id in self.token_ids):
            raise TypeError("token IDs must be integers")


@runtime_checkable
class ChunkTokeniser(Protocol):
    """Synchronous tokeniser required by the chunking engine."""

    @property
    def identity(self) -> TokeniserIdentity:
        """Return the complete stable tokeniser identity."""
        ...

    def tokenise(self, text: str) -> TokenisedText:
        """Tokenise exact text without special tokens and expose offsets."""
        ...

    def count_tokens(self, text: str) -> int:
        """Return the standalone token count for exact text."""
        ...


def validate_tokenised_text(text: str, tokenised: TokenisedText) -> TokenisedText:
    """Validate offsets against one exact input string."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(tokenised, TokenisedText):
        raise ChunkTokeniserError("invalid_output")
    previous_end = 0
    for offset in tokenised.offsets:
        if offset.end > len(text) or offset.start < previous_end:
            raise ChunkTokeniserError("invalid_offsets")
        previous_end = offset.end
    return tokenised


@dataclass(frozen=True, slots=True)
class FakeChunkTokeniser:
    """Deterministic whitespace tokeniser reserved for contract tests."""

    _identity: TokeniserIdentity = TokeniserIdentity(
        implementation="fake-whitespace",
        contract_version="1.0.0",
        artefact_identity="test-fixture",
        artefact_revision="v1",
        options=(("add-special-tokens", "false"),),
    )

    @property
    def identity(self) -> TokeniserIdentity:
        """Return the explicitly non-production fake identity."""
        return self._identity

    def tokenise(self, text: str) -> TokenisedText:
        """Tokenise non-whitespace runs with stable ordinal token IDs."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        offsets: list[TokenOffset] = []
        in_token = False
        start = 0
        for index, character in enumerate(text):
            if character.isspace():
                if in_token:
                    offsets.append(TokenOffset(start, index))
                    in_token = False
                continue
            if not in_token:
                start = index
                in_token = True
        if in_token:
            offsets.append(TokenOffset(start, len(text)))
        tokenised = TokenisedText(
            token_ids=tuple(range(1, len(offsets) + 1)),
            offsets=tuple(offsets),
        )
        return validate_tokenised_text(text, tokenised)

    def count_tokens(self, text: str) -> int:
        """Return the exact standalone fake-token count."""
        return len(self.tokenise(text).token_ids)


def _file_digest(path: Path) -> str:
    """Hash one regular artefact without exposing its path."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artefact_manifest(root: Path) -> tuple[tuple[str, str], ...]:
    """Return the sorted, path-free inventory used for adapter identity."""
    if not root.is_dir() or root.is_symlink():
        raise ChunkTokeniserError("unavailable")
    entries: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*"), key=lambda candidate: candidate.as_posix()):
        if path.is_symlink():
            raise ChunkTokeniserError("unsafe_artefact")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ChunkTokeniserError("unsafe_artefact")
        relative = path.relative_to(root).as_posix()
        entries.append((relative, _file_digest(path)))
    if not entries:
        raise ChunkTokeniserError("unavailable")
    return tuple(entries)


def _manifest_revision(manifest: tuple[tuple[str, str], ...]) -> str:
    """Derive a stable non-path revision from the complete artefact inventory."""
    payload = "".join(f"{name}\0{digest}\0" for name, digest in manifest)
    return "manifest-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LocalFastChunkTokeniser:
    """Adapt one explicit local Transformers fast tokeniser for ingest."""

    _CONTRACT_VERSION: Final[str] = "1.0.0"

    def __init__(
        self,
        tokeniser: object,
        identity: TokeniserIdentity,
        manifest: tuple[tuple[str, str], ...],
        artefact_dir: Path,
    ) -> None:
        self._tokeniser = tokeniser
        self._identity = identity
        self._manifest = manifest
        self._artefact_dir = artefact_dir

    @classmethod
    def from_directory(cls, artefact_dir: str | Path) -> "LocalFastChunkTokeniser":
        """Load a fast tokeniser from local artefacts with no network fallback."""
        if not isinstance(artefact_dir, (str, Path)):
            raise TypeError("artefact_dir must be a local path")
        if not is_local_path_shape(artefact_dir):
            raise ChunkTokeniserError("unsafe_artefact")
        root = Path(artefact_dir)
        try:
            before = _artefact_manifest(root)
            from transformers import AutoTokenizer, __version__

            tokeniser = AutoTokenizer.from_pretrained(
                root,
                local_files_only=True,
                use_fast=True,
                trust_remote_code=False,
            )
            after = _artefact_manifest(root)
            if before != after:
                raise ChunkTokeniserError("artefact_changed_during_load")
            if not getattr(tokeniser, "is_fast", False):
                raise ChunkTokeniserError("slow_tokeniser")
            revision = _manifest_revision(after)
            identity = TokeniserIdentity(
                implementation="transformers-fast",
                contract_version=cls._CONTRACT_VERSION,
                artefact_identity="local-manifest",
                artefact_revision=revision,
                options=(
                    ("add-special-tokens", "false"),
                    ("library-version", str(__version__)),
                    ("local-files-only", "true"),
                    ("trust-remote-code", "false"),
                    ("use-fast", "true"),
                ),
            )
        except ChunkTokeniserError:
            raise
        except Exception:
            raise ChunkTokeniserError("unavailable") from None
        loaded = cls(tokeniser, identity, after, root)
        validate_chunk_tokeniser(loaded)
        return loaded

    @property
    def identity(self) -> TokeniserIdentity:
        """Return the path-free local artefact identity."""
        return self._identity

    def tokenise(self, text: str) -> TokenisedText:
        """Tokenise exact text without special tokens and validate offsets."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        try:
            tokeniser = cast(Callable[..., Mapping[str, object]], self._tokeniser)
            encoded = tokeniser(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_offsets_mapping=True,
            )
            token_ids = tuple(cast(Iterable[int], encoded["input_ids"]))
            raw_offsets = tuple(
                cast(Iterable[tuple[int, int]], encoded["offset_mapping"])
            )
            tokenised = TokenisedText(
                token_ids=token_ids,
                offsets=tuple(TokenOffset(start, end) for start, end in raw_offsets),
            )
        except ChunkTokeniserError:
            raise
        except Exception:
            raise ChunkTokeniserError("invalid_output") from None
        return validate_tokenised_text(text, tokenised)

    def count_tokens(self, text: str) -> int:
        """Return the standalone token count for exact text."""
        return len(self.tokenise(text).token_ids)


class _ValidatedChunkTokeniser:
    """Sanitise one tokeniser implementation behind the ingest boundary."""

    def __init__(
        self,
        tokeniser: ChunkTokeniser,
        identity: TokeniserIdentity,
    ) -> None:
        self._tokeniser = tokeniser
        self._identity = identity

    @property
    def identity(self) -> TokeniserIdentity:
        """Return the identity captured at boundary validation."""
        return self._identity

    def tokenise(self, text: str) -> TokenisedText:
        """Return validated output while bounding backend exceptions."""
        try:
            tokenised = self._tokeniser.tokenise(text)
        except ChunkTokeniserError:
            raise
        except Exception:
            raise ChunkTokeniserError("unavailable") from None
        try:
            return validate_tokenised_text(text, tokenised)
        except ChunkTokeniserError:
            raise
        except Exception:
            raise ChunkTokeniserError("invalid_output") from None

    def count_tokens(self, text: str) -> int:
        """Return an exact count consistent with validated tokenisation."""
        try:
            count = self._tokeniser.count_tokens(text)
        except ChunkTokeniserError:
            raise
        except Exception:
            raise ChunkTokeniserError("unavailable") from None
        if not is_strict_integer(count) or count < 0:
            raise ChunkTokeniserError("invalid_count")
        if count != len(self.tokenise(text).token_ids):
            raise ChunkTokeniserError("inconsistent_count")
        return count


def validate_chunk_tokeniser(tokeniser: ChunkTokeniser) -> ChunkTokeniser:
    """Validate the public tokeniser boundary and deterministic identity shape."""
    if isinstance(tokeniser, _ValidatedChunkTokeniser):
        return tokeniser
    try:
        implements_protocol = isinstance(tokeniser, ChunkTokeniser)
    except Exception:
        raise ChunkTokeniserError("unavailable") from None
    if not implements_protocol:
        raise ChunkTokeniserError("invalid_identity")
    try:
        identity = tokeniser.identity
    except Exception:
        raise ChunkTokeniserError("unavailable") from None
    if not isinstance(identity, TokeniserIdentity):
        raise ChunkTokeniserError("invalid_identity")
    validated = _ValidatedChunkTokeniser(tokeniser, identity)
    for probe in ("", "one café", "astral 😀"):
        first = validated.tokenise(probe)
        second = validated.tokenise(probe)
        count = validated.count_tokens(probe)
        if first != second or count != len(first.token_ids):
            raise ChunkTokeniserError("non_deterministic")
    return validated
