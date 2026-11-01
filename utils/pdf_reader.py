"""PDF reading helpers for SmartNotes AI.

This file has one job: take an uploaded PDF and return clean text.
"""

import io
import re

from pypdf import PdfReader

# If a PDF has fewer characters than this, it is probably a scanned
# document (made of images), so there is no real text to read.
MIN_TEXT_CHARS = 50


class PDFReadError(Exception):
    """Raised when a PDF cannot be read. The message is safe to show to students."""


def _clean_text(text):
    """Remove messy spaces and extra blank lines."""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_pdf(uploaded_file):
    """Read an uploaded PDF and return a dictionary with its text and details.

    Returns:
        {"text": str, "pages": int, "words": int, "characters": int}

    Raises:
        PDFReadError: if the file is invalid, locked, or has no readable text.
    """
    try:
        data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
        reader = PdfReader(io.BytesIO(data))
    except Exception as error:
        raise PDFReadError(
            "This file could not be opened. It may be damaged or not a real PDF. "
            "Please try another file."
        ) from error

    # Some PDFs are "locked" but open without a password. Try that first.
    if reader.is_encrypted:
        try:
            if not reader.decrypt(""):
                raise PDFReadError(
                    "This PDF is password-protected. Please remove the password and upload it again."
                )
        except PDFReadError:
            raise
        except Exception as error:
            raise PDFReadError(
                "This PDF is protected and cannot be read. Please upload an unprotected copy."
            ) from error

    page_texts = []
    try:
        total_pages = len(reader.pages)
    except Exception as error:
        raise PDFReadError("The pages of this PDF could not be read. Please try another file.") from error

    for page in reader.pages:
        try:
            page_texts.append(page.extract_text() or "")
        except Exception:
            # Skip one bad page instead of failing the whole document.
            page_texts.append("")

    text = _clean_text("\n\n".join(page_texts))

    if len(text) < MIN_TEXT_CHARS:
        raise PDFReadError(
            "No readable text was found. This looks like a scanned PDF (pictures of pages). "
            "Please upload a PDF with selectable text, or run it through an OCR tool first."
        )

    return {
        "text": text,
        "pages": total_pages,
        "words": len(text.split()),
        "characters": len(text),
    }
