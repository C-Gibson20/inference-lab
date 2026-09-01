"""Tests for producing-component and operation metadata."""

from dataclasses import FrozenInstanceError

import pytest

from inference_lab.provenance.errors import InvalidLineageMetadataError
from inference_lab.provenance.operations import (
    ComponentRole,
    ConfigurationIdentity,
    DigestAlgorithm,
    OperationKind,
    ProducingComponent,
    ProducingOperation,
)

_CONFIGURATION_DIGEST = "a" * 64


def _configuration_identity() -> ConfigurationIdentity:
    return ConfigurationIdentity(
        algorithm=DigestAlgorithm.SHA256,
        digest=_CONFIGURATION_DIGEST,
    )


def _component(
    role: str,
    *,
    implementation: str,
    contract_version: str = "1.0.0",
    artefact_identity: str | None = None,
    artefact_revision: str | None = None,
) -> ProducingComponent:
    return ProducingComponent(
        role=ComponentRole(role),
        implementation=implementation,
        contract_version=contract_version,
        artefact_identity=artefact_identity,
        artefact_revision=artefact_revision,
    )


def test_producing_operation_identifies_versioned_components() -> None:
    model = _component(
        "generator-model",
        implementation="local-generator-adapter",
        contract_version="2.1.0",
        artefact_identity="qwen/qwen3",
        artefact_revision="4f3c2a1",
    )
    prompt = _component(
        "summary-prompt",
        implementation="summary-prompt",
        contract_version="1.0.0",
        artefact_identity="document-summary",
        artefact_revision="v3",
    )

    operation = ProducingOperation(
        kind=OperationKind("summarisation"),
        implementation="document-summariser",
        contract_version="1.2.0-rc.1+local",
        configuration_identity=_configuration_identity(),
        components=(model, prompt),
    )

    assert operation.kind.as_str() == "summarisation"
    assert operation.configuration_identity.as_str() == (
        f"sha256:{_CONFIGURATION_DIGEST}"
    )
    assert operation.components == (model, prompt)


def test_operation_and_component_metadata_are_immutable() -> None:
    component = _component("parser", implementation="text-native")
    operation = ProducingOperation(
        kind=OperationKind("parsing"),
        implementation="text-parser",
        contract_version="1.0.0",
        configuration_identity=_configuration_identity(),
        components=(component,),
    )

    with pytest.raises(FrozenInstanceError):
        component.contract_version = "2.0.0"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        operation.components = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("value_type", "value"),
    [
        (OperationKind, "Summarisation"),
        (OperationKind, "summary_generation"),
        (ComponentRole, "generator model"),
        (ComponentRole, ""),
    ],
)
def test_project_owned_kinds_and_roles_require_lowercase_slugs(
    value_type: type[OperationKind] | type[ComponentRole],
    value: str,
) -> None:
    with pytest.raises(InvalidLineageMetadataError, match="lowercase slug"):
        value_type(value)


@pytest.mark.parametrize("value", [None, True, 1, 1.0])
def test_project_owned_kinds_and_roles_reject_wrong_types(value: object) -> None:
    with pytest.raises(TypeError, match="must be a string"):
        OperationKind(value)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a string"):
        ComponentRole(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "version",
    ["", "1", "1.0", "01.0.0", "1.0.0-01", "v1.0.0"],
)
def test_contract_versions_require_canonical_semantic_versions(version: str) -> None:
    with pytest.raises(InvalidLineageMetadataError, match="semantic version"):
        _component(
            "parser",
            implementation="text-native",
            contract_version=version,
        )


def test_external_artefact_revision_is_separate_from_contract_version() -> None:
    component = _component(
        "generator-model",
        implementation="local-generator-adapter",
        contract_version="1.0.0",
        artefact_identity="organisation/model-name",
        artefact_revision="refs/pr/42",
    )

    assert component.contract_version == "1.0.0"
    assert component.artefact_identity == "organisation/model-name"
    assert component.artefact_revision == "refs/pr/42"


@pytest.mark.parametrize(
    ("identity", "revision"),
    [
        ("model-name", None),
        (None, "revision"),
        ("/private/model", "revision"),
        ("model-name", "revision with spaces"),
    ],
)
def test_external_artefact_metadata_rejects_incomplete_or_noncanonical_values(
    identity: str | None,
    revision: str | None,
) -> None:
    with pytest.raises(InvalidLineageMetadataError, match="artefact"):
        _component(
            "generator-model",
            implementation="local-generator-adapter",
            artefact_identity=identity,
            artefact_revision=revision,
        )


@pytest.mark.parametrize(
    ("identity", "revision", "message"),
    [
        (1, None, "artefact_identity"),
        (None, 1, "artefact_revision"),
        ("model-name", 1, "artefact_revision"),
    ],
)
def test_external_artefact_metadata_rejects_wrong_runtime_types(
    identity: object,
    revision: object,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        _component(
            "generator-model",
            implementation="local-generator-adapter",
            artefact_identity=identity,  # type: ignore[arg-type]
            artefact_revision=revision,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("digest", ["", "A" * 64, "a" * 63, "g" * 64])
def test_configuration_identity_requires_canonical_sha256(digest: str) -> None:
    with pytest.raises(InvalidLineageMetadataError, match="configuration digest"):
        ConfigurationIdentity(
            algorithm=DigestAlgorithm.SHA256,
            digest=digest,
        )


def test_configuration_identity_rejects_raw_configuration() -> None:
    with pytest.raises(TypeError, match="configuration digest must be a string"):
        ConfigurationIdentity(
            algorithm=DigestAlgorithm.SHA256,
            digest={"model_path": "/private/model"},  # type: ignore[arg-type]
        )


def test_producing_operation_requires_canonical_unique_components() -> None:
    parser = _component("parser", implementation="text-native")
    tokeniser = _component("tokeniser", implementation="qwen-tokeniser")

    with pytest.raises(InvalidLineageMetadataError, match="canonical order"):
        ProducingOperation(
            kind=OperationKind("chunking"),
            implementation="structure-aware-chunker",
            contract_version="1.0.0",
            configuration_identity=_configuration_identity(),
            components=(tokeniser, parser),
        )

    with pytest.raises(InvalidLineageMetadataError, match="duplicates"):
        ProducingOperation(
            kind=OperationKind("chunking"),
            implementation="structure-aware-chunker",
            contract_version="1.0.0",
            configuration_identity=_configuration_identity(),
            components=(parser, parser),
        )


def test_operation_metadata_rejects_mutable_or_untyped_components() -> None:
    parser = _component("parser", implementation="text-native")

    with pytest.raises(TypeError, match="components must be a tuple"):
        ProducingOperation(
            kind=OperationKind("parsing"),
            implementation="text-parser",
            contract_version="1.0.0",
            configuration_identity=_configuration_identity(),
            components=[parser],  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="ProducingComponent"):
        ProducingOperation(
            kind=OperationKind("parsing"),
            implementation="text-parser",
            contract_version="1.0.0",
            configuration_identity=_configuration_identity(),
            components=("parser",),  # type: ignore[arg-type]
        )


def test_component_metadata_rejects_malformed_fields() -> None:
    with pytest.raises(TypeError, match="component role"):
        ProducingComponent(
            role="parser",  # type: ignore[arg-type]
            implementation="text-native",
            contract_version="1.0.0",
        )
    with pytest.raises(InvalidLineageMetadataError, match="implementation"):
        _component("parser", implementation="Text Native")
    with pytest.raises(TypeError, match="contract_version"):
        ProducingComponent(
            role=ComponentRole("parser"),
            implementation="text-native",
            contract_version=1,  # type: ignore[arg-type]
        )


def test_operation_metadata_rejects_malformed_fields() -> None:
    valid_values = {
        "kind": OperationKind("parsing"),
        "implementation": "text-parser",
        "contract_version": "1.0.0",
        "configuration_identity": _configuration_identity(),
    }

    with pytest.raises(TypeError, match="operation kind"):
        ProducingOperation(**(valid_values | {"kind": "parsing"}))  # type: ignore[arg-type]
    with pytest.raises(InvalidLineageMetadataError, match="implementation"):
        ProducingOperation(**(valid_values | {"implementation": "Text Parser"}))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="contract_version"):
        ProducingOperation(**(valid_values | {"contract_version": 1}))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="configuration_identity"):
        ProducingOperation(**(valid_values | {"configuration_identity": {}}))  # type: ignore[arg-type]


def test_invalid_metadata_errors_do_not_echo_rejected_values() -> None:
    private_marker = "private model path"

    with pytest.raises(InvalidLineageMetadataError) as captured:
        _component(
            "generator-model",
            implementation="local-generator-adapter",
            artefact_identity=private_marker,
            artefact_revision="revision",
        )

    assert private_marker not in str(captured.value)
