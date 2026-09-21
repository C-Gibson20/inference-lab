"""Run one local corpus-scoped search in a fresh process."""

import argparse
import json
import sys
from collections.abc import Sequence
from typing import cast

from inference_lab.logging.logging_config import configure_logging
from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.serving import (
    LocalServingFacade,
    ServingSearchError,
    ServingSearchRequest,
    ServingSearchResult,
)
from inference_lab.settings import get_settings
from inference_lab.storage import open_metadata_repository


def _parser() -> argparse.ArgumentParser:
    """Build the bounded operator-facing search example parser."""
    parser = argparse.ArgumentParser(
        description="Search one local, explicitly selected corpus."
    )
    parser.add_argument("--corpus-id", required=True, help="Opaque corpus UUID.")
    parser.add_argument("--query", required=True, help="Query text.")
    parser.add_argument("--limit", type=int, default=10, help="Positive result limit.")
    parser.add_argument(
        "--corpus-version",
        help="Optional owned historical corpus-version UUID.",
    )
    return parser


def _parse_identifier(
    value: str,
    identifier_type: type[CorpusId] | type[CorpusVersionId],
) -> CorpusId | CorpusVersionId:
    """Parse one opaque identifier without echoing invalid input."""
    try:
        return identifier_type.from_str(value)
    except (TypeError, ValueError):
        raise ValueError("invalid_identifier") from None


def _serialise_result(result: ServingSearchResult) -> dict[str, object]:
    """Return a JSON-safe result projection with source coordinates."""
    serving_result = result
    return {
        "corpus_id": serving_result.corpus_id.as_str(),
        "corpus_version": serving_result.corpus_version.as_str(),
        "hits": [
            {
                "record_id": hit.record_id.as_str(),
                "document_id": hit.document_id.as_str(),
                "document_version_id": hit.document_version_id.as_str(),
                "rank": hit.rank,
                "score": hit.score,
                "score_label": hit.score_label,
                "text": hit.text,
                "source_locations": [
                    {
                        "source_span_id": location.source_span_id.as_str(),
                        "parsed_record_id": location.parsed_record_id.as_str(),
                        "start_offset": location.start_offset,
                        "end_offset": location.end_offset,
                        "span_start_offset": location.span_start_offset,
                        "span_end_offset": location.span_end_offset,
                        "content_origin": location.content_origin.value,
                        "block_ordinal": location.block_ordinal,
                        "block_kind": location.block_kind.value,
                        "page_number": location.page_number,
                        "bounding_box": (
                            None
                            if location.bounding_box is None
                            else {
                                "left": location.bounding_box.left,
                                "top": location.bounding_box.top,
                                "right": location.bounding_box.right,
                                "bottom": location.bounding_box.bottom,
                            }
                        ),
                        "page_layout": (
                            None
                            if location.page_layout is None
                            else {
                                "page_number": location.page_layout.page_number,
                                "coordinate_unit": location.page_layout.coordinate_unit.value,
                                "coordinate_origin": location.page_layout.coordinate_origin.value,
                                "page_width": location.page_layout.page_width,
                                "page_height": location.page_layout.page_height,
                                "rotation": int(location.page_layout.rotation),
                                "content_origin": location.page_layout.content_origin.value,
                                "producer": {
                                    "implementation": location.page_layout.producer.implementation,
                                    "contract_version": location.page_layout.producer.contract_version,
                                    "artefact_identity": location.page_layout.producer.artefact_identity,
                                    "artefact_revision": location.page_layout.producer.artefact_revision,
                                },
                            }
                        ),
                    }
                    for location in hit.source_locations
                ],
            }
            for hit in serving_result.hits
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run one search and print its bounded JSON result or error category."""
    args = _parser().parse_args(argv)
    try:
        corpus_id = cast(CorpusId, _parse_identifier(args.corpus_id, CorpusId))
        corpus_version = (
            None
            if args.corpus_version is None
            else cast(
                CorpusVersionId,
                _parse_identifier(args.corpus_version, CorpusVersionId),
            )
        )
        settings = get_settings()
        configure_logging(settings.logging)
        repository = open_metadata_repository(
            settings.storage.metadata_db_path,
            source_dir=settings.storage.source_dir,
            apply_migrations=True,
        )
        try:
            result = LocalServingFacade(repository, settings).search(
                ServingSearchRequest(
                    corpus_id=corpus_id,
                    corpus_version=corpus_version,
                    text=args.query,
                    limit=args.limit,
                )
            )
        finally:
            repository.close()
        print(json.dumps(_serialise_result(result), sort_keys=True))
        return 0
    except ServingSearchError as error:
        print(json.dumps({"error": error.category.value}), file=sys.stderr)
        return 2
    except (TypeError, ValueError):
        print(json.dumps({"error": "invalid_request"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
