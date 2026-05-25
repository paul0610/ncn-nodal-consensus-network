"""TXTIngestor — ingest plain text files (.txt, .log, .text).

Handles encoding detection automatically via chardet, falling back to UTF-8.
Chunks the file at paragraph boundaries (double newlines) for downstream
consensus processing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger

from core.config_loader import Config
from core.exceptions import NCNError
from core.models import RawContent
from ingestion.base import IngestorPort


# Safety limits
_MAX_FILE_SIZE_MB = 50
_CHUNK_TARGET_CHARS = 2000  # chunks aim at ~500 tokens


class TXTIngestor(IngestorPort):
    """Read a plain-text file into paragraph-level chunks."""

    def __init__(self, config: Config) -> None:
        self.config = config

    async def ingest(self, source: str) -> list[RawContent]:
        """*source* is a filesystem path to a text file."""
        path = Path(source)
        if not path.exists():
            raise NCNError(f"Text file not found: {source}")

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > _MAX_FILE_SIZE_MB:
            raise NCNError(
                f"Text file too large ({size_mb:.1f} MB > {_MAX_FILE_SIZE_MB} MB)"
            )

        return await asyncio.to_thread(self._read_and_chunk, str(path))

    @staticmethod
    def _read_and_chunk(txt_path: str) -> list[RawContent]:
        raw_bytes = Path(txt_path).read_bytes()
        text = _decode_bytes(raw_bytes)

        chunks = _chunk_text(text, _CHUNK_TARGET_CHARS)
        name = Path(txt_path).name

        contents = [
            RawContent(
                text=chunk,
                source=f"txt:{name}:chunk{i + 1}",
                source_type="txt",
            )
            for i, chunk in enumerate(chunks)
            if chunk.strip()
        ]
        logger.info(f"TXT ingested: {txt_path} → {len(contents)} chunks")
        return contents


def _decode_bytes(raw: bytes) -> str:
    """Decode bytes to string, auto-detecting encoding with chardet fallback.

    Also normalizes line endings to Unix (\\n) so downstream paragraph
    splitting on ``\\n\\n`` works consistently across Windows/Mac/Unix files.
    """
    # Fast path: UTF-8
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        # Try chardet if available, otherwise latin-1
        try:
            import chardet
            detected = chardet.detect(raw)
            encoding = detected.get("encoding") or "latin-1"
            text = raw.decode(encoding, errors="replace")
        except ImportError:
            text = raw.decode("latin-1", errors="replace")

    # Normalize CRLF / CR to LF for consistent paragraph splitting
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _chunk_text(text: str, target_chars: int) -> list[str]:
    """Split text into paragraph-aligned chunks of roughly *target_chars*."""
    if len(text) <= target_chars:
        return [text.strip()]

    chunks: list[str] = []
    paragraphs = text.split("\n\n")
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 > target_chars and current:
            chunks.append(current.strip())
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para

    if current.strip():
        chunks.append(current.strip())

    return chunks
