"""URL fetcher abstraction — pluggable transport for scraping/downloading.

Supports:
  - HttpxFetcher: direct HTTP via httpx (default, no proxy).
  - BrightDataFetcher: Bright Data SDK for sites that need proxies/anti-bot.
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
    """Fetch pages via Bright Data (proxies, anti-bot). Use for harder-to-scrape sites.

    Requires BRIGHTDATA_API_TOKEN. Use async with BrightDataClient per request.
    """

    def __init__(
        self,
        *,
        timeout: float = 120,
        poll_interval: int = 5,
        poll_timeout: int = 180,
        use_async_mode: bool = True,
    ) -> None:
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._poll_timeout = poll_timeout
        self._use_async_mode = use_async_mode

    async def fetch(self, url: str, *, timeout: float | None = None) -> str:
        try:
            from brightdata import BrightDataClient
        except ImportError as e:
            raise ImportError(
                "Bright Data fetcher requires: pip install brightdata-sdk"
            ) from e
        # SDK requires async with per use; no client reuse outside context
        async with BrightDataClient() as client:
            if self._use_async_mode:
                result = await client.scrape_url(
                    url,
                    mode="async",
                    poll_interval=self._poll_interval,
                    poll_timeout=self._poll_timeout,
                )
            else:
                result = await client.scrape_url(url)
        return self._extract_text(result, url)

    @staticmethod
    def _extract_text(result: Any, url: str) -> str:
        """Normalize result.data to str (may be HTML string or dict)."""
        data = getattr(result, "data", result)
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            for key in ("content", "html", "body", "text"):
                v = data.get(key)
                if isinstance(v, str):
                    return v
            for v in data.values():
                if isinstance(v, str) and len(v) > 100:
                    return v
            raise ValueError(
                f"Bright Data result for {url} had no usable text (keys: {list(data.keys())})"
            )
        return str(data)
