# Local source discovery

> **Status:** The discovery, classification, stable hashing, identity resolution,
> and atomic metadata persistence APIs are implemented. They are library
> building blocks until ING-05 supplies pipeline orchestration and SERV-01
> exposes the first operator command.

Source discovery converts an explicit local file or directory scope into
deterministically ordered outcomes for one selected `CorpusId`. It does not
parse content, copy original files into managed storage, create chunks or
indexes, or publish a corpus version.

## Inputs and ordering

Every request names one or more explicit local regular files or directories.
URI-shaped, empty, NUL-containing, and unsupported UNC-shaped inputs are
rejected before filesystem access. Source-specific external identifiers and
family evidence may accompany an explicit file input only; directory discovery
does not infer metadata from filenames or ambient sidecars.

Roots and snapshotted directory entries are sorted by the versioned local path
policy. Overlapping roots that reach the same resolved entry are deduplicated
before request-local ordinals are assigned. Separate live paths remain separate
source observations even when they are hard links or contain identical bytes.

Ordering is deterministic for the same filesystem snapshot and policy version.
Do not assume that case folding, Unicode path representation, inode values, or
directory ordering are portable between filesystems.

## Traversal policy

The baseline policy is `local-source-discovery` version `1.0.0`:

| Area | Behaviour |
| --- | --- |
| Symbolic links and junctions | Never followed, including when explicitly selected. |
| Special files | FIFOs, sockets, devices, and other non-regular entries are skipped with explicit outcomes. |
| Hidden entries | Included; `.gitignore` and similar files do not change discovery. |
| Mount boundaries | An explicitly selected mount may be a root, but recursive traversal does not enter a nested device. |
| Directory reads | Each directory is snapshotted and sorted; a failed entry or subtree remains visible as a bounded failure. |
| Recursion | Traversal is iterative and does not depend on Python recursion depth. |

The integrating workflow must exclude active metadata, index, managed-source,
backup, and restore-staging paths. A selected root that overlaps an exclusion
fails rather than ingesting deployment-owned state. These locations come from
the deployment and are passed explicitly to traversal; discovery does not guess
them.

## Supported formats

The baseline classifier is `text-native-baseline` version `1.0.0`:

| Suffix | Media type | Outcome |
| --- | --- | --- |
| `.txt` | `text/plain` | Supported as UTF-8 by the `text-native` parser capability. |
| `.pdf` | `application/pdf` | Recognised but unsupported; it is counted without receiving identity. |
| Any other or missing suffix | `application/octet-stream` | Unknown; it is counted without receiving identity. |

Classification is deliberately content-free. A misleading `.txt` suffix can
pass discovery and hashing but still fail later parser validation. Adding a
supported format requires a parser capability and a classifier-policy version
change.

## Stable hashing and size limits

Supported regular files are streamed through SHA-256 without retaining their
contents in memory. Discovery compares pre-open, opened-handle, post-read
handle, and post-read path metadata. It discards the digest if it cannot prove
that the same stable file was read throughout.

`ingest.max_source_bytes` limits both the size observed before reading and the
actual bytes consumed during streaming. The default is 52,428,800 bytes
(50 MiB). Empty text files are valid and receive the standard SHA-256 digest of
empty bytes.

## Identity and move behaviour

Paths are private continuity evidence, never document identity. Successful
discovery uses opaque generated source-observation, document, and version IDs.
Candidate reads and writes are restricted to the selected corpus.
Explicit family candidates remain advisory and do not create family membership.

| Outcome | Meaning |
| --- | --- |
| `new_document` | No existing identity matched; a new document, version, observation, membership, and locator are committed. |
| `new_version` | A continuous source has changed, or authoritative evidence selects an existing document; the new version is committed. |
| `unchanged` | The existing observation and version are reused without new parse work or durable identity mutation. |
| `duplicate` | A separate live source has identical bytes and receives independently citable identity plus a duplicate relationship. |
| `ambiguous` / `conflict` | Evidence is insufficient or inconsistent; no identity or locator changes are published. |

A rename or move reuses an observation only when the old locator is absent and
one compatible replacement can be proven from a complete relevant scan.
Copy-delete moves may use exact hash and compatible explicit identifiers.
Incomplete traversal, multiple candidates, a still-live old path, hard links,
and inode reuse intentionally limit automatic move inference.

Each source is resolved and persisted in its own transaction. A bounded
conflict retry rebuilds candidate evidence in a fresh transaction. Exhausted
contention becomes a safe per-source failure; unavailable repository state and
corrupt candidate metadata remain request-fatal.

## Results and logs

Results retain deterministic source order and report counts for files seen,
supported, unsupported, hashed, unchanged, changed, new, duplicate, ambiguous,
conflicting, and failed sources. An empty directory is a successful zero-source
result. `hashed` includes sources whose bytes were hashed successfully even when
continuity, resolution, or persistence later fails; failures during hashing are
excluded. A partial result has at least one usable source and one failure; a
failed result has failures but no usable source.

Structured lifecycle events expose policy identity, bounded categories,
counts, durations, media type, byte count, retry state, outcome kind/reason, and
opaque IDs. They never include source bytes or decoded text, hashes, filenames,
paths, external identifier values, corpus display names, raw exceptions, or
directory listings. `corpus_version` remains null because discovery updates the
catalogue without publishing a corpus version.

Private paths may appear in in-process result objects so the immediate caller
can identify a failed source. Do not serialise those records into logs,
profiling output, or portable reports.

## Original-source hand-off

A `VerifiedSourceSnapshot` proves which bytes discovery hashed; it is not a
durable source copy. STOR-02 or ING-05 must reopen the source and revalidate its
file identity, size, and SHA-256 while copying it into managed storage. Any
mismatch requests rediscovery instead of associating new bytes with the earlier
identity outcome.

Absolute locator history is deployment-local metadata and is not portable
source backup. Keep the original files available under the deployment's
source-data policy until managed-source persistence and its backup procedure
have completed.
