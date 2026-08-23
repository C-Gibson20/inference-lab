# Ingest parser contract and boundaries

The current ingest parser is a deterministic text-native parsing baseline. It establishes the contract consumed by later ingest stages; it is not the
complete document ingestion pipeline.

## Parser contract

### Input

The parser accepts a `ParserInput` containing:

- a stable `DocumentId`;
- a stable `DocumentVersionId`;
- source `bytes`;
- media type `text/plain`;
- a named text encoding supported by the Python codec registry.

Input validation occurs before decoding. Invalid identifiers, non-byte
content, unsupported media types, missing encodings, and unknown encodings are rejected with a named `ParserInputError` category. The parser does not accept paths or read files; callers own source acquisition and preservation.

The source-size limit comes from `Settings.ingest.max_source_bytes`. It is
measured in raw input bytes and checked before decoding. It must not be
replaced with a hardcoded parser limit.

### Output

`TextNativeParser.parse()` returns a validated `ParseOutcome`:

- `succeeded` contains one complete `ParsedDocument` and no failures;
- `failed` contains no document and at least one fatal `ParserFailure`;
- `partial` is part of the shared outcome contract but is not emitted by the
  current text-native baseline.

`ParsedDocument` includes the input identifiers, parser version, canonical
text, ordered structural blocks, paragraph aliases, and character count. It
uses explicit `None` for unavailable pages, sections, language, page count,
and token count.

`NormalisedText.source_text` is the strictly decoded source coordinate space. `NormalisedText.text` is NFC-normalized text with CRLF and CR converted to LF. Whitespace is otherwise preserved. Empty input is a successful empty document; an unavailable structure is `None`, while a present empty collection is `()`.

Paragraph boundaries are derived from `source_text` before normalization. LF,
CRLF, and CR are line endings. One or more lines containing only spaces or
tabs (including no characters) separate paragraphs; separator characters are
excluded from both adjacent source spans. Leading blank lines and trailing
blank-line separators do not produce empty blocks. A single line ending
within a paragraph, including its final line ending, remains paragraph
content. Indentation and other whitespace on content lines are preserved.

### Source spans and provenance

Every emitted block has a `SourceSpan` linked to the input `DocumentVersionId`. Offsets are half-open Unicode code-point offsets into
`source_text`: `[start_offset, end_offset)`. They are not byte offsets and are not offsets into normalized text.

Blocks are emitted in source order. Source-span identifiers are deterministic UUIDs derived from the document version, block ordinal, block kind, and source range. Repeating a parse with the same input, settings, and parser version must produce byte-identical canonical serialization.

### Failure handling and logging

The baseline reports named failures for invalid input, oversized source,
decode failure, and invalid output. Malformed source must never result in a
partial document. Failure records contain category, severity, and pipeline
stage, but not source text, paths, raw bytes, or other payload data.

Parser lifecycle and failure events use privacy-safe structured logging through `structlog`. Log fields may include stable document identity, parser version, failure category, source byte count, output character count, and emitted record count where applicable.
