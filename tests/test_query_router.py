"""Tests for core/query_router.py — intent classification."""

import pytest

from core.query_router import QueryIntent, route_query


# ---------------------------------------------------------------------------
# Greetings
# ---------------------------------------------------------------------------

class TestGreetings:
    """Greetings should be caught before reaching the pipeline."""

    @pytest.mark.parametrize("text", [
        "Hola",
        "holi",
        "HOLIS",
        "buenas",
        "Hello",
        "hey",
        "que tal",
        "Hola!",
        "holi??",
        "Buenos dias",
    ])
    def test_greetings_detected(self, text):
        intent, url = route_query(text)
        assert intent == QueryIntent.GREETING
        assert url is None

    @pytest.mark.parametrize("text", [
        "hola que es Python",
        "buenas, dime sobre DeepSeek",
    ])
    def test_greeting_with_question_is_normal(self, text):
        """A greeting followed by a real question should be NORMAL."""
        intent, _ = route_query(text)
        assert intent == QueryIntent.NORMAL


# ---------------------------------------------------------------------------
# Too-short queries
# ---------------------------------------------------------------------------

class TestTooShort:

    @pytest.mark.parametrize("text", [
        "??",
        "a",
        "test",
        "abc",
        "x",
    ])
    def test_short_noise_detected(self, text):
        intent, _ = route_query(text)
        assert intent == QueryIntent.TOO_SHORT

    def test_single_long_word_is_normal(self):
        """A single word >= 8 chars should pass (e.g. 'Python3.12')."""
        intent, _ = route_query("ornostocaustico")
        assert intent == QueryIntent.NORMAL


# ---------------------------------------------------------------------------
# URL detection
# ---------------------------------------------------------------------------

class TestURLDetection:

    @pytest.mark.parametrize("text,expected_url_start", [
        ("que opinas de trustperu.ai?", "https://trustperu.ai"),
        ("visita https://google.com/search", "https://google.com/search"),
        ("mira www.ejemplo.com/page", "https://www.ejemplo.com/page"),
        ("revisa anthropic.com", "https://anthropic.com"),
        ("que es openai.io", "https://openai.io"),
    ])
    def test_urls_detected(self, text, expected_url_start):
        intent, url = route_query(text)
        assert intent == QueryIntent.URL_LEARN
        assert url is not None
        assert url.startswith(expected_url_start)

    @pytest.mark.parametrize("text", [
        "que es Python",
        "hablame de machine learning",
        "que es un modelo.pth",
    ])
    def test_non_urls_not_detected(self, text):
        intent, _ = route_query(text)
        assert intent != QueryIntent.URL_LEARN


# ---------------------------------------------------------------------------
# Normal queries
# ---------------------------------------------------------------------------

class TestNormalQueries:

    @pytest.mark.parametrize("text", [
        "que es DeepSeek V3",
        "dime que es un ornostocaustico",
        "cual es la capital de Francia",
        "como funciona el consenso en NCN",
        "hola que es Python",
    ])
    def test_normal_queries(self, text):
        intent, _ = route_query(text)
        assert intent == QueryIntent.NORMAL


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_string(self):
        intent, _ = route_query("")
        assert intent == QueryIntent.TOO_SHORT

    def test_whitespace_only(self):
        intent, _ = route_query("   ")
        assert intent == QueryIntent.TOO_SHORT

    def test_url_with_greeting_prefix(self):
        """URL takes priority over greeting detection."""
        intent, url = route_query("hola mira trustperu.ai")
        assert intent == QueryIntent.URL_LEARN
        assert url is not None

    def test_https_url(self):
        intent, url = route_query("https://example.com/path?q=1")
        assert intent == QueryIntent.URL_LEARN
        assert url == "https://example.com/path?q=1"
