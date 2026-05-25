"""Source type auto-detection — infer the right ingestor from a path or URL.

Used to route incoming sources to the correct ingestor without requiring
the caller to specify the format manually.

Priority of detection:
  1. Content-Type header (if provided from a HEAD request)
  2. File extension
  3. Fallback to 'text'

Recognized types correspond to keys in ``ingestion.factory._REGISTRY``:
  pdf, docx, epub, markdown, txt, html, text
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse


# Extension → source_type mapping (keep lowercase)
_EXTENSION_MAP: dict[str, str] = {
    # PDF
    ".pdf": "pdf",
    # Word
    ".docx": "docx",
    ".doc": "docx",           # legacy — attempts same ingestor
    # EPUB (books)
    ".epub": "epub",
    # Markdown
    ".md": "markdown",
    ".markdown": "markdown",
    ".mdown": "markdown",
    # Plain text
    ".txt": "txt",
    ".text": "txt",
    ".log": "txt",
    # HTML
    ".html": "html",
    ".htm": "html",
    ".xhtml": "html",
}


# Content-Type substring → source_type mapping (checked in order)
_CONTENT_TYPE_PATTERNS: list[tuple[str, str]] = [
    ("application/pdf", "pdf"),
    ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx"),
    ("application/msword", "docx"),
    ("application/epub+zip", "epub"),
    ("text/markdown", "markdown"),
    ("text/x-markdown", "markdown"),
    ("text/html", "html"),
    ("application/xhtml+xml", "html"),
    ("text/plain", "txt"),
]


def detect_source_type(
    path_or_url: str,
    content_type: str | None = None,
) -> str:
    """Infer the source type from a path/URL and optional Content-Type header.

    Parameters
    ----------
    path_or_url:
        Local file path or URL.
    content_type:
        HTTP ``Content-Type`` header value (from a HEAD request). When
        provided, takes precedence over the extension — useful because some
        servers serve PDFs from extensionless URLs.

    Returns
    -------
    str
        One of: ``'pdf'``, ``'docx'``, ``'epub'``, ``'markdown'``,
        ``'txt'``, ``'html'``, or ``'text'`` (fallback).
    """
    # 1. Content-Type takes precedence
    if content_type:
        ct = content_type.lower().strip()
        for pattern, src_type in _CONTENT_TYPE_PATTERNS:
            if pattern in ct:
                return src_type

    # 2. Extension — strip query params from URLs first
    path_part = urlparse(path_or_url).path if "://" in path_or_url else path_or_url
    suffix = Path(path_part).suffix.lower()
    if suffix in _EXTENSION_MAP:
        return _EXTENSION_MAP[suffix]

    # 3. Fallback: unknown extension, no content-type → assume plain text
    #    (html would have content-type; extensionless files are usually plain)
    return "text"


def is_url(source: str) -> bool:
    """Return True if *source* looks like an HTTP/HTTPS URL."""
    parsed = urlparse(source)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def supported_extensions() -> list[str]:
    """Return all recognized file extensions (for UI file filters)."""
    return sorted(_EXTENSION_MAP.keys())


def supported_source_types() -> set[str]:
    """Return all known source_type values."""
    return set(_EXTENSION_MAP.values())
