"""Query Router — classifies user input intent before the main pipeline.

Routes incoming queries to the correct handler:
  - GREETING   → friendly reply, no ingestion
  - TOO_SHORT  → ask user for more context
  - URL_LEARN  → scrape the URL directly, then answer
  - NORMAL     → standard retrieval + consensus pipeline

This prevents accidental ingestion of greetings ("Holi"),
routes explicit URLs to the page scraper, and rejects noise.
"""

from __future__ import annotations

import re
from enum import Enum

from loguru import logger


# ---------------------------------------------------------------------------
# Intent enum
# ---------------------------------------------------------------------------

class QueryIntent(str, Enum):
    GREETING = "greeting"
    TOO_SHORT = "too_short"
    URL_LEARN = "url_learn"
    NORMAL = "normal"


# ---------------------------------------------------------------------------
# URL detection
# ---------------------------------------------------------------------------

# Common TLDs to recognize bare domains (e.g. "trustperu.ai", "google.com")
_COMMON_TLDS = (
    "com", "org", "net", "io", "ai", "dev", "app", "co", "edu", "gov",
    "info", "xyz", "me", "pe", "us", "uk", "de", "es", "fr", "br",
    "tech", "cloud", "site", "online", "page", "blog", "pro", "cc",
)
_TLD_RE = "|".join(_COMMON_TLDS)

# Match:  https://...  |  http://...  |  www.something  |  bare domain.tld
URL_PATTERN = re.compile(
    r"(?:https?://)\S+"                                      # full URL
    r"|(?:www\.)\S+"                                         # www prefix
    rf"|\b[a-zA-Z0-9][-a-zA-Z0-9]*\.(?:{_TLD_RE})"          # bare domain
    rf"(?:\.[a-zA-Z]{{2,3}})?"                               # optional cc-TLD
    r"(?:/\S*)?",                                            # optional path
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Greeting detection
# ---------------------------------------------------------------------------

GREETINGS: set[str] = {
    # Spanish
    "hola", "holi", "holis", "holaa", "holiii", "ola",
    "buenas", "buenos dias", "buenas tardes", "buenas noches",
    "que tal", "como estas", "como andas", "que onda",
    # English
    "hello", "hi", "hey", "howdy", "good morning",
    "good afternoon", "good evening", "sup", "yo",
}


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

MIN_QUERY_CHARS = 5
MIN_QUERY_WORDS_FOR_SHORT = 1     # single word below 8 chars → too short
MAX_SHORT_WORD_LEN = 8


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def route_query(query: str) -> tuple[QueryIntent, str | None]:
    """Classify query intent and optionally extract a URL.

    Returns
    -------
    (intent, extracted_url_or_None)
    """
    q = query.strip()

    # 1. URL detection (highest priority — even greetings could contain a URL)
    url_match = URL_PATTERN.search(q)
    if url_match:
        url = url_match.group()
        # Normalize: ensure scheme
        if not url.lower().startswith("http"):
            url = "https://" + url
        logger.info(f"QueryRouter: URL detected → {url}")
        return QueryIntent.URL_LEARN, url

    # 2. Greeting detection (strip punctuation for comparison)
    clean = q.lower().rstrip("?¿!¡.,;:)( ")
    if clean in GREETINGS:
        logger.info(f"QueryRouter: greeting detected → '{clean}'")
        return QueryIntent.GREETING, None

    # 3. Too-short / noise detection
    if len(clean) < MIN_QUERY_CHARS:
        logger.info(f"QueryRouter: too short ({len(clean)} chars)")
        return QueryIntent.TOO_SHORT, None

    words = clean.split()
    if len(words) <= MIN_QUERY_WORDS_FOR_SHORT and len(clean) < MAX_SHORT_WORD_LEN:
        logger.info(f"QueryRouter: single short word '{clean}'")
        return QueryIntent.TOO_SHORT, None

    return QueryIntent.NORMAL, None
