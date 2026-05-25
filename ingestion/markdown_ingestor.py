"""MarkdownIngestor — ingest Markdown files (.md, .markdown, .mdown).

Chunks at section boundaries (H1/H2/H3 headers) which gives semantically
meaningful units for downstream consensus. Falls back to paragraph-level
splitting for long sections or files without headers.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from loguru import logger

from core.config_loader import Config
from core.exceptions import NCNError
from core.models import RawContent
from ingestion.base import IngestorPort


_MAX_FILE_SIZE_MB = 50
_CHUNK_TARGET_CHARS = 2000

# Match ATX-style headers: ^# Title, ^## Title, ^### Title
_HEADER_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


class MarkdownIngestor(IngestorPort):
    """Read a Markdown file, chunking at headers for semantic coherence."""

    def __init__(self, config: Config) -> None:
        self.config = config

    async def ingest(self, source: str) -> list[RawContent]:
        path = Path(source)
        if not path.exists():
            raise NCNError(f"Markdown file not found: {source}")

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > _MAX_FILE_SIZE_MB:
            raise NCNError(
                f"Markdown file too large ({size_mb:.1f} MB > {_MAX_FILE_SIZE_MB} MB)"
            )

        return await asyncio.to_thread(self._read_and_chunk, str(path))

    @staticmethod
    def _read_and_chunk(md_path: str) -> list[RawContent]:
        raw = Path(md_path).read_text(encoding="utf-8", errors="replace")
        # Normalize line endings for cross-platform consistency
        text = raw.replace("\r\n", "\n").replace("\r", "\n")
        name = Path(md_path).name

        # Strip basic markdown formatting but keep the actual content.
        # We deliberately keep header text and bullet content intact — just
        # remove the raw marks so the consensus sees clean prose.
        chunks = _split_by_headers(text, _CHUNK_TARGET_CHARS)

        contents = [
            RawContent(
                text=_clean_markdown(chunk).strip(),
                source=f"md:{name}:chunk{i + 1}",
                source_type="markdown",
            )
            for i, chunk in enumerate(chunks)
            if chunk.strip()
        ]
        logger.info(f"Markdown ingested: {md_path} → {len(contents)} chunks")
        return contents


def _split_by_headers(text: str, target_chars: int) -> list[str]:
    """Chunk by top-level headers; fallback to paragraph splits for long sections."""
    # Find all header positions
    header_positions = [m.start() for m in _HEADER_RE.finditer(text)]

    if not header_positions:
        # No headers → fall back to paragraph chunking
        return _paragraph_chunks(text, target_chars)

    # Include the text before the first header (preamble) if meaningful
    sections: list[str] = []
    if header_positions[0] > 0:
        preamble = text[: header_positions[0]].strip()
        if preamble:
            sections.append(preamble)

    # Split at each header
    for i, start in enumerate(header_positions):
        end = header_positions[i + 1] if i + 1 < len(header_positions) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)

    # Merge small adjacent sections / split oversized ones
    merged: list[str] = []
    buffer = ""
    for section in sections:
        if len(section) > target_chars:
            # Flush buffer, then split the big section by paragraphs
            if buffer:
                merged.append(buffer.strip())
                buffer = ""
            merged.extend(_paragraph_chunks(section, target_chars))
        elif len(buffer) + len(section) + 2 > target_chars and buffer:
            merged.append(buffer.strip())
            buffer = section
        else:
            buffer = f"{buffer}\n\n{section}" if buffer else section

    if buffer.strip():
        merged.append(buffer.strip())

    return merged


def _paragraph_chunks(text: str, target_chars: int) -> list[str]:
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


def _clean_markdown(text: str) -> str:
    """Strip minimal markdown syntax while preserving content."""
    # Remove code fences markers but keep the inner code
    text = re.sub(r"^```.*$", "", text, flags=re.MULTILINE)
    # Remove emphasis markers (*, **, _, __) without touching bullets
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"([*_])(\S(?:.*?\S)?)\1", r"\2", text)
    # Remove inline code backticks but keep content
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Convert links [text](url) → "text (url)"
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    return text
