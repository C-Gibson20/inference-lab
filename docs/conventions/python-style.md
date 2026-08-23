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

Test modules may use explicit relative imports for fixture and helper modules in
the same test package. These helpers are local test implementation details, not
an application API. Imports of application code remain absolute.

Imports are grouped as standard library, third-party packages, then
`inference_lab` or test-package imports, with one blank line between groups.
Ruff owns ordering within those groups. Wildcard imports are prohibited.

Use `TYPE_CHECKING` imports or postponed annotation evaluation only when they
resolve a genuine runtime cycle or optional typing dependency. Do not add them
pre-emptively.

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
