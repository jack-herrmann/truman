"""Tests for the fetcher abstraction (HttpxFetcher, BrightDataFetcher)."""
from __future__ import annotations

import pytest

from intuition.corpus.fetcher import HttpxFetcher, BrightDataFetcher, PageFetcher


@pytest.mark.asyncio
async def test_httpx_fetcher_fetches_real_url():
    """HttpxFetcher can fetch a simple public URL."""
    fetcher = HttpxFetcher(follow_redirects=True, timeout=10)
    # Use a stable, small public resource
    text = await fetcher.fetch("https://www.gutenberg.org/robots.txt")
    assert isinstance(text, str)
    assert "User-agent" in text or "user-agent" in text.lower()


@pytest.mark.asyncio
async def test_httpx_fetcher_raises_on_404():
    """HttpxFetcher raises on HTTP error."""
    fetcher = HttpxFetcher(follow_redirects=True, timeout=5)
    with pytest.raises(Exception):  # httpx.HTTPStatusError
        await fetcher.fetch("https://www.gutenberg.org/nonexistent-page-404-not-found")


# --- BrightDataFetcher._extract_text (static, no SDK needed) ---

def test_bright_data_extract_text_plain_string():
    """_extract_text returns data directly when it's a plain string."""
    class R:
        data = "hello world"
    assert BrightDataFetcher._extract_text(R(), "http://x") == "hello world"


def test_bright_data_extract_text_dict_content_key():
    """_extract_text pulls 'content' from a dict result."""
    class R:
        data = {"content": "page body"}
    assert BrightDataFetcher._extract_text(R(), "http://x") == "page body"


def test_bright_data_extract_text_dict_html_key():
    """_extract_text pulls 'html' from a dict result."""
    class R:
        data = {"html": "<p>hi</p>"}
    assert BrightDataFetcher._extract_text(R(), "http://x") == "<p>hi</p>"


def test_bright_data_extract_text_list_result():
    """_extract_text handles list results (batch API returns)."""
    class R:
        data = ["page content here"]
    assert BrightDataFetcher._extract_text(R(), "http://x") == "page content here"


def test_bright_data_extract_text_no_usable_text_raises():
    """_extract_text raises when dict has no string content."""
    class R:
        data = {"status": 200, "count": 0}
    with pytest.raises(ValueError, match="no usable text"):
        BrightDataFetcher._extract_text(R(), "http://x")


def test_bright_data_constructor_accepts_token():
    """BrightDataFetcher can be created with an explicit token."""
    fetcher = BrightDataFetcher(token="test-token-123")
    assert fetcher._token == "test-token-123"


def test_bright_data_constructor_no_token():
    """BrightDataFetcher works without token (reads BRIGHTDATA_API_TOKEN env var)."""
    fetcher = BrightDataFetcher()
    assert fetcher._token is None


# --- GutenbergCorpus with mock fetcher ---

@pytest.mark.asyncio
async def test_gutenberg_uses_fetcher(temp_dir):
    """GutenbergCorpus uses the provided PageFetcher."""
    from intuition.corpus.gutenberg import GutenbergCorpus, NOVELS

    class MockFetcher(PageFetcher):
        async def fetch(self, url: str, *, timeout: float = 60) -> str:
            return "*** START OF THIS PROJECT GUTENBERG EBOOK ***\n\nChapter 1.\n\nThe end.\n*** END OF THIS PROJECT GUTENBERG EBOOK ***"

    fetcher = MockFetcher()
    corpus = GutenbergCorpus(data_dir=str(temp_dir), fetcher=fetcher)
    key = next(iter(NOVELS))
    path = await corpus.download_novel(key)
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "Chapter 1" in text
    assert "The end" in text
    assert "GUTENBERG" not in text  # cleaned
