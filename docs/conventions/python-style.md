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
doing so can hide dependency direction and create initialization cycles. Across
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
