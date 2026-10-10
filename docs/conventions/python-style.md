# Python style conventions

These conventions supplement the enforced Ruff formatting and lint rules. They
apply to production code under `src/`, scripts, and tests unless an exception is
stated explicitly.

## Imports

Production modules use absolute imports rooted at `inference_lab`, including
imports from sibling modules and package `__init__.py` files. Absolute imports
make the dependency source unambiguous, remain searchable across the repository,
and behave consistently whether code is reached through a command, test, or
installed package.

Within one domain package, import a symbol from the module that defines it. Do
not route internal dependencies through the package's `__init__.py`, because
doing so can hide dependency direction and create initialisation cycles. Across
domain boundaries, prefer the owning package's public API when it exports the
required symbol; otherwise import the defining module explicitly.

Test modules follow the same policy. Relative imports are banned repository
wide by the enforced `flake8-tidy-imports` rule
(`ban-relative-imports = "all"`), so fixture and helper modules in the same
test package are imported absolutely through their test package path. These
helpers remain local test implementation details rather than an application
API; only their import spelling is shared with production code.

Imports are grouped as standard library, third-party packages, then
`inference_lab` or test-package imports, with one blank line between groups.
Ruff owns ordering within those groups. Wildcard imports are prohibited.

Use `TYPE_CHECKING` imports or postponed annotation evaluation only when they
resolve a genuine runtime cycle or optional typing dependency. Do not add them
pre-emptively.

## Shared validation

Side-effect-free predicates for formats used across domain boundaries belong
in `inference_lab/_validation.py`. They classify values without raising a
domain exception, resolving paths, or touching external state. Each owning
boundary remains responsible for runtime type errors, domain-specific value
errors, and any filesystem or lifecycle action that follows validation.

Validation logic used by only one domain belongs in that domain's
`_validation.py` or responsibility-named implementation module. Do not put
validation predicates in a generic `utils.py` module.

Pure domain transformations belong in a responsibility-named module such as
`normalisation.py`. Public annotated aliases and their constants belong in a
domain `contracts.py` or `types.py` module. A generic `utils.py` must not mix
these responsibilities or act as a miscellaneous helper home.

## Constructor validation

Immutable dataclass constructors enforce their invariants in `__post_init__`
so an invalid instance cannot escape construction. Multi-step validation uses
a concise `__post_init__` that delegates to `_validate()` and, where useful,
responsibility-named helpers. A trivial one- or two-field check may remain
inline when delegation would obscure rather than clarify the invariant.

Semantic and schema version validators at plain Python boundaries raise
`TypeError` for a wrong runtime type and `ValueError`, or the owning domain's
`ValueError` subclass, when a string violates the contract. Side-effect-free
shared predicates remain exception-neutral. Framework-owned model boundaries
may expose their normal validation wrapper instead.

## Shared JSON mechanics

`inference_lab._json` owns the recursive JSON types, compact canonical encoder,
strict UTF-8 object decoder, duplicate-key and non-finite-value rejection, and
primitive structural checks shared across domains. Domain serialisation modules
own their schema field sets, compatibility checks, and translation into
domain-specific, payload-private errors.

Human-readable formats, such as indented evaluation manifests, retain a local
encoder that makes that presentation policy explicit. JSON used as input to a
semantic identity or fingerprint also remains with the identity owner because
changing its byte policy would change persisted identities.

## Shared clock

Production code obtains the current timezone-aware UTC time from
`inference_lab._clock.utc_now`. Callers that require deterministic timestamps
accept a clock callable and use `utc_now` only as its default. Domain modules
must not define equivalent wall-clock factories.

## Language

Project-owned prose, Python identifiers, error messages, configuration names,
and serialised field or enum values use British English spelling. This includes
forms such as `normalise`, `serialise`, `behaviour`, and `utilisation`.

Names owned by Python, a third-party library, or a published external standard
retain their exact spelling at the integration boundary. Examples include
`unicodedata.normalize`, Pydantic's `field_serializer`, and Structlog's `colors`
keyword. Keep those spellings local to the boundary rather than propagating
them into project-owned names.

A spelling correction to a persisted or public project contract is an
incompatible change and requires the owning schema or contract version to
advance. Do not add aliases solely to preserve a superseded first implementation.

## Descriptive names and comments

Module, class, function, constant, and test names, docstrings, and comments
describe behaviour and contracts. They do not reference build-plan items. Do
not use phase numbers or issue identifiers such as `RET-06` or `Phase 6` in
`src/` or `tests/`.

A reader working in the codebase must be able to understand what a module owns
without opening the build plan. A docstring reading "Validated request
boundary for RET-06 evidence reranking" requires an external document to
decode; "Rerank fused candidates into a bounded evidence set" does not. The
same applies to a constant: prefer `RERANK_POLICY_VERSION` over
`RET06_POLICY_VERSION`.

Build-plan vocabulary belongs in `docs/plans/`. A convention document may cite
the issue that owns a contract, because a convention's purpose is to connect a
product decision to the boundary that implements it. Acceptance traceability
belongs in the test-suite plan's coverage-trace table, which maps each build
item to the acceptance evidence that protects it, rather than in test names or
docstrings.

## Package exports

A package's `__init__.py` exports its complete domain vocabulary, not a
minimal entry-point set. A name belongs in `__all__` when it is part of that
package's contract: a value or type a consumer constructs, annotates, catches,
implements, compares, or calls. A name reachable only through a private helper
does not, and neither does an alias of an already-exported name.

Not every exported name has an in-repository importer: later build items are
expected consumers of contract identities, protocols, and versioned
constants. Absence of a current importer is therefore not evidence that a name
should be removed.

Across domain boundaries, a consumer that needs a symbol the owning package
does not export imports the defining module explicitly. Removing a name from
`__all__` changes the required import spelling; it does not make the symbol
unreachable.

## Package docstrings

Every production `__init__.py` has a concise docstring that identifies the
package's owned responsibility. This includes deliberately retained
architectural packages whose implementation is scheduled for later work. A
docstring must describe a real boundary; generic text such as "package" or
"utilities" does not justify an otherwise unnecessary package marker.

Do not make the `src/`, `scripts/`, or `benchmarks/` layout directories Python
packages. Remove empty package markers for obsolete, duplicate, generated-output,
or undocumented boundaries instead of adding filler docstrings. Architecture
placeholders are retained only when the package boundary is documented in
`AGENT.md` or the product specification.

Test `__init__.py` files are import-isolation markers rather than public
packages and are exempt from the package-docstring rule. They may still carry a
docstring when it usefully describes shared fixtures or test scope.

Every non-package test module has a concise module docstring identifying its
test scope. Individual test functions rely on descriptive names and do not need
docstrings that merely restate those names. Use a focused comment when a test
needs to preserve non-obvious rationale or explain why an arrangement matters.
