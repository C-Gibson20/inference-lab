# RUN-04 development corpus sample audit

**Status:** Preliminary local audit with all 28 cases independently reviewed
in a second pass. The reviewer confirmed the source-page references and
recommended wording changes to x05 and x27; those changes are applied in the
dataset. Candidate scores on this set remain development evidence, not a final
model-selection result.

## Scope and handling

The local sample used for RUN-04 contains four English research articles and
one English patent application, all focused on SARM1. Its five source PDFs total
30,168,453 bytes and 524 physical PDF pages. The source files and extracted
document text remain local and are not included in this repository. The
manifest identifies each source by its complete PDF SHA-256 digest.

The audit used `pypdf` 6.19.0 to count physical pages and extract text. A page
was counted as text-bearing when its extracted text was non-empty after
whitespace trimming. This measures extractability, not reading order,
scientific accuracy, or visual quality. Manual inspection found occasional
layout-related spacing and symbol artefacts, particularly around figures,
tables, and chemical notation. The three patent pages with no extracted text
are excluded from the labelled passages until their content is inspected by a
separate visual or OCR review.

## Source inventory

| Document ID prefix | Type | Physical pages | Text-bearing pages | Extracted characters | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `49f353a6158a` | Research article | 7 | 7 | 39,418 | Short experimental report on low-dose BEI effects. |
| `6b2690cdafa6` | Patent application | 471 | 468 | 947,466 | Pages 363–365 have no extracted text; claim pages are text-bearing. |
| `1f483b17372d` | Research article | 12 | 12 | 70,800 | Cysteine-targeted SARM1 inhibitors. |
| `f0c7491ddedff` | Research article | 16 | 16 | 78,439 | Imidazopyridine SARM1 modulators. |
| `602e28e48a50` | Research article | 18 | 18 | 116,362 | SARM1 base-exchange inhibitor pharmacology. |

The totals are 521 text-bearing pages and 1,252,485 extracted characters. The
patent contributes most of the pages, while all documents cover one narrow
target area and the research papers overlap in subject matter. This is a small,
English-only sample; it does not establish the language or document-type
distribution of the complete corpus.

## Development annotations

[`development-cases-expanded.jsonl`](development-cases-expanded.jsonl) adds 28
short, paraphrased questions with one directly supporting physical PDF page
per case. The labels span article results, study limitations, and patent
claims. Page numbers are one-based physical PDF pages, including front matter;
they are not printed page numbers. Each case uses the source PDF digest as its
`document_id`. The companion manifest pins the document inventory and the
exact JSONL digest.

One reviewer drafted and checked these questions and page labels against the
local PDFs. A second reviewer then checked all 28 labels against the cited
physical pages and nearby pages, verified the source PDF digests, and visually
inspected the figure-heavy page used by x01. All cited pages were confirmed to
exist and contain extracted text. The second reviewer recommended clarifying
x05 to ask about the measure used to determine clinical risk, and correcting
x27 to identify the SARM1-expressing HEK293 assay; both query changes are
applied. The existing
[`development-cases.jsonl`](development-cases.jsonl) remains unchanged so the
exploratory candidate-pilot figures can still be traced to their original
16-case set.

The second pass found the labels supported for development use. It did not
broaden the corpus: all five PDFs concern SARM1, and the queries include several
related low-dose BEI activation questions. The reviewer identified x05 as the
case most likely to benefit from subject-matter expert confirmation if the
intended distinction between SARM1 activity and a clinical biomarker remains
unclear.

## Evaluation limits

- The passage unit is a whole PDF page, not the production chunking strategy.
- The 28 cases are manually authored and are not sampled from a user-query
  distribution.
- The set tests English SARM1 literature and patent language only.
- Single-page relevance labels record one directly supporting page; the second
  pass found additional related evidence for some cases, and other pages may
  also support an answer.
- The expanded candidate comparison is recorded separately. It uses this
  small development sample and is not representative-corpus or held-out
  evaluation evidence.
- Held-out EVAL-04 questions were not used or included.
