"""Local, text-layer PDF extraction used by the ingest parser."""

from dataclasses import dataclass
from io import BytesIO
from typing import cast

from pypdf import PdfReader

from inference_lab.ingest.sources import PageLayoutDescriptor, PageRotation


class PdfExtractionUnsupportedError(ValueError):
    """Identify a supported PDF limitation without retaining source details."""


@dataclass(frozen=True, slots=True)
class PdfPageText:
    """Pair deterministically extracted page text with its native geometry."""

    text: str
    layout: PageLayoutDescriptor


def extract_pdf_pages(content: bytes) -> tuple[PdfPageText, ...]:
    """Extract text-layer content and page geometry without OCR or file access."""
    with BytesIO(content) as stream:
        reader = PdfReader(stream, strict=True)
        try:
            if reader.is_encrypted:
                raise PdfExtractionUnsupportedError

            pages: list[PdfPageText] = []
            for page_number, page in enumerate(reader.pages, start=1):
                media_box = page.mediabox
                lower_left = media_box.lower_left
                upper_right = media_box.upper_right
                width = float(upper_right[0]) - float(lower_left[0])
                height = float(upper_right[1]) - float(lower_left[1])
                rotation = page.rotation or 0
                if rotation not in (0, 90, 180, 270):
                    raise PdfExtractionUnsupportedError
                text = (
                    page.extract_text(extraction_mode="layout") or ""
                    if "/Contents" in page
                    else ""
                )
                pages.append(
                    PdfPageText(
                        text=text,
                        layout=PageLayoutDescriptor(
                            page_number=page_number,
                            coordinate_unit="pdf_points",
                            coordinate_origin="bottom_left",
                            page_width=width,
                            page_height=height,
                            rotation=cast(PageRotation, rotation),
                        ),
                    )
                )
            return tuple(pages)
        finally:
            reader.close()
