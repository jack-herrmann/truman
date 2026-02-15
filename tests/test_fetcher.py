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


def test_bright_data_extract_text_static():
    """_extract_text normalizes result.data to str."""
    class R:
        data = "hello world"
    assert BrightDataFetcher._extract_text(R(), "http://x") == "hello world"

    class R2:
        data = {"content": "page body"}
    assert BrightDataFetcher._extract_text(R2(), "http://x") == "page body"

    class R3:
        data = {"html": "<p>hi</p>"}
    assert BrightDataFetcher._extract_text(R3(), "http://x") == "<p>hi</p>"


def test_bright_data_extract_text_no_usable_text_raises():
    """_extract_text raises when dict has no string content."""
    class R:
        data = {"status": 200, "count": 0}
    with pytest.raises(ValueError, match="no usable text"):
        BrightDataFetcher._extract_text(R(), "http://x")


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
