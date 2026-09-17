"""Versioned parser-capability selection for baseline ingest."""

import re
from dataclasses import dataclass
from enum import StrEnum

from inference_lab.ingest.discovery import SourceClassification, SourceSupport
from inference_lab.ingest.parser import Parser
from inference_lab.ingest.text_native import TextNativeParser
from inference_lab.ingest.versioning import CURRENT_PARSER_VERSION, ParserVersion

_CAPABILITY_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")


class ParserSelectionErrorCategory(StrEnum):
    """Bounded reasons why a parser cannot be selected."""

    UNSUPPORTED_SOURCE = "unsupported_source"
    MISSING_CAPABILITY = "missing_parser_capability"
    PARSER_UNAVAILABLE = "parser_unavailable"
    VERSION_MISMATCH = "parser_version_mismatch"


class ParserSelectionError(ValueError):
    """Represent a privacy-safe parser-selection failure."""

    def __init__(self, category: ParserSelectionErrorCategory) -> None:
        """Create an error containing only its bounded category."""
        if not isinstance(category, ParserSelectionErrorCategory):
            raise TypeError("category must be a ParserSelectionErrorCategory.")
        super().__init__(category.value)
        self.category = category


@dataclass(frozen=True, slots=True)
class ParserRegistration:
    """Bind one parser capability to one exact parser contract version."""

    capability: str
    parser_version: ParserVersion
    parser: Parser

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate capability identity and the parser protocol boundary."""
        if not isinstance(self.capability, str):
            raise TypeError("capability must be a string.")
        if _CAPABILITY_PATTERN.fullmatch(self.capability) is None:
            raise ValueError("capability must be a lowercase hyphenated slug.")
        if not isinstance(self.parser_version, ParserVersion):
            raise TypeError("parser_version must be a ParserVersion.")
        if not callable(getattr(self.parser, "parse", None)):
            raise TypeError("parser must implement parse().")

    def key(self) -> tuple[str, str]:
        """Return the canonical registration identity."""
        return self.capability, self.parser_version.value


@dataclass(frozen=True, slots=True)
class ParserRegistry:
    """Hold the immutable parser registrations available to one run."""

    registrations: tuple[ParserRegistration, ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate deterministic registration ordering and uniqueness."""
        if not isinstance(self.registrations, tuple):
            raise TypeError("registrations must be a tuple.")
        if any(
            not isinstance(registration, ParserRegistration)
            for registration in self.registrations
        ):
            raise TypeError("registrations must contain ParserRegistration values.")
        keys = tuple(registration.key() for registration in self.registrations)
        if keys != tuple(sorted(keys)):
            raise ValueError("registrations must be in canonical order.")
        if len(keys) != len(set(keys)):
            raise ValueError("registrations must be unique.")

    def select(
        self,
        classification: SourceClassification,
        *,
        parser_version: ParserVersion,
    ) -> Parser:
        """Select the exact parser required by a source classification."""
        if not isinstance(classification, SourceClassification):
            raise TypeError("classification must be a SourceClassification.")
        if not isinstance(parser_version, ParserVersion):
            raise TypeError("parser_version must be a ParserVersion.")
        if classification.support is not SourceSupport.SUPPORTED:
            raise ParserSelectionError(ParserSelectionErrorCategory.UNSUPPORTED_SOURCE)
        capability = classification.parser_capability
        if capability is None:
            raise ParserSelectionError(ParserSelectionErrorCategory.MISSING_CAPABILITY)

        capability_registrations = tuple(
            registration
            for registration in self.registrations
            if registration.capability == capability
        )
        if not capability_registrations:
            raise ParserSelectionError(ParserSelectionErrorCategory.PARSER_UNAVAILABLE)
        for registration in capability_registrations:
            if registration.parser_version == parser_version:
                return registration.parser
        raise ParserSelectionError(ParserSelectionErrorCategory.VERSION_MISMATCH)


CURRENT_PARSER_REGISTRY = ParserRegistry(
    registrations=(
        ParserRegistration(
            capability="text-native",
            parser_version=CURRENT_PARSER_VERSION,
            parser=TextNativeParser(),
        ),
    )
)


def select_parser(
    classification: SourceClassification,
    *,
    parser_version: ParserVersion,
    registry: ParserRegistry = CURRENT_PARSER_REGISTRY,
) -> Parser:
    """Select one exact local parser from a versioned capability registry."""
    if not isinstance(registry, ParserRegistry):
        raise TypeError("registry must be a ParserRegistry.")
    return registry.select(classification, parser_version=parser_version)
