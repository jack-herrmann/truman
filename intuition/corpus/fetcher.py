"""URL fetcher abstraction — pluggable transport for scraping/downloading.

Supports:
  - HttpxFetcher: direct HTTP via httpx (default, no proxy).
  - BrightDataFetcher: Bright Data SDK for sites that need proxies/anti-bot.

Usage:
    fetcher = HttpxFetcher()                      # default, free, no proxy
    fetcher = BrightDataFetcher()                  # uses BRIGHTDATA_API_TOKEN env var
    fetcher = BrightDataFetcher(token="sk-...")     # explicit token
    text = await fetcher.fetch("https://example.com")
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class PageFetcher(ABC):
    """Async fetcher: given a URL, returns response body as text."""

    @abstractmethod
    async def fetch(self, url: str, *, timeout: float = 60) -> str:
        """Fetch URL and return response body as string. Raises on failure."""
        ...


class HttpxFetcher(PageFetcher):
    """Fetch pages with httpx (no proxy). Default for public domains like Gutenberg."""

    def __init__(self, *, follow_redirects: bool = True, timeout: float = 60) -> None:
        self._follow_redirects = follow_redirects
        self._default_timeout = timeout

    async def fetch(self, url: str, *, timeout: float | None = None) -> str:
        import httpx
        t = timeout if timeout is not None else self._default_timeout
        async with httpx.AsyncClient(
            follow_redirects=self._follow_redirects, timeout=t
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text


class BrightDataFetcher(PageFetcher):
    """Fetch pages via the Bright Data Python SDK (generic web scraper).

    Requires:
        pip install brightdata-sdk

    Auth (pick one):
        - Set env var BRIGHTDATA_API_TOKEN (SDK reads it automatically)
        - Pass token= to constructor

    SDK docs: https://docs.brightdata.com/api-reference/SDK
    """

    def __init__(self, *, token: str | None = None) -> None:
        self._token = token

    async def fetch(self, url: str, *, timeout: float | None = None) -> str:
        try:
            from brightdata import BrightDataClient
        except ImportError as e:
            raise ImportError(
                "Bright Data fetcher requires the SDK: pip install brightdata-sdk\n"
                "Then set BRIGHTDATA_API_TOKEN env var or pass token= to BrightDataFetcher."
            ) from e

        kwargs: dict[str, Any] = {}
        if self._token:
            kwargs["token"] = self._token

        # The SDK async API: client.scrape.generic.url_async([urls])
        # Returns a list of result objects with .success, .data, .cost
        async with BrightDataClient(**kwargs) as client:
            results = await client.scrape.generic.url_async([url])

        result = results[0] if isinstance(results, list) else results

        if not getattr(result, "success", True):
            raise RuntimeError(
                f"Bright Data scrape failed for {url}"
            )

        logger.debug(
            "Bright Data scraped %s (cost=$%.4f, %dms)",
            url,
            getattr(result, "cost", 0.0),
            getattr(result, "elapsed_ms", lambda: 0)(),
        )

        return self._extract_text(result, url)

    @staticmethod
    def _extract_text(result: Any, url: str) -> str:
        """Extract text content from a Bright Data scrape result.

        result.data is typically a string (page HTML/text) for generic scraping.
        We handle dict/list formats defensively in case the API returns structured data.
        """
        data = getattr(result, "data", result)

        # Most common: data is the page content as a string
        if isinstance(data, str):
            return data

        # If data is a list (batch result), take the first item
        if isinstance(data, list) and data:
            item = data[0]
            if isinstance(item, str):
                return item
            if isinstance(item, dict):
                return _extract_from_dict(item, url)
            return str(item)

        # If data is a dict with known content keys
        if isinstance(data, dict):
            return _extract_from_dict(data, url)

        return str(data)


def _extract_from_dict(data: dict, url: str) -> str:
    """Pull text from a dict result, trying common content keys."""
    for key in ("content", "html", "body", "text"):
        v = data.get(key)
        if isinstance(v, str) and v:
            return v
    # Fallback: any large string value
    for v in data.values():
        if isinstance(v, str) and len(v) > 100:
            return v
    raise ValueError(
        f"Bright Data result for {url} had no usable text (keys: {list(data.keys())})"
    )
