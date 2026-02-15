#!/usr/bin/env python3
"""Download novels from Project Gutenberg.

Uses httpx by default. To use Bright Data (proxies/anti-bot), set
USE_BRIGHTDATA=1 or pass --brightdata (requires BRIGHTDATA_API_TOKEN).
"""
import argparse
import asyncio
import logging
import os
from pathlib import Path

import yaml
from rich.console import Console
from rich.logging import RichHandler

from intuition.corpus.gutenberg import GutenbergCorpus
from intuition.corpus.fetcher import BrightDataFetcher, HttpxFetcher, PageFetcher

logging.basicConfig(level=logging.INFO, handlers=[RichHandler(console=Console(stderr=True))])


def _load_config() -> dict:
    for p in [Path("config.yaml"), Path(__file__).resolve().parent.parent / "config.yaml"]:
        if p.exists():
            return yaml.safe_load(p.read_text()) or {}
    return {}


def _get_fetcher(use_brightdata: bool) -> PageFetcher:
    if use_brightdata:
        return BrightDataFetcher(poll_timeout=180)
    return HttpxFetcher(follow_redirects=True, timeout=60)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Download Gutenberg novels.")
    parser.add_argument(
        "--data-dir",
        default="data/novels",
        help="Directory to save novels (default: data/novels)",
    )
    parser.add_argument(
        "--brightdata",
        action="store_true",
        help="Use Bright Data for fetching (requires BRIGHTDATA_API_TOKEN)",
    )
    args = parser.parse_args()
    config = _load_config()
    use_bd = (
        args.brightdata
        or os.environ.get("USE_BRIGHTDATA", "").lower() in ("1", "true", "yes")
        or str(config.get("corpus", {}).get("use_brightdata", "")).lower()
        in ("1", "true", "yes")
    )
    fetcher = _get_fetcher(use_bd)
    corpus = GutenbergCorpus(data_dir=args.data_dir, fetcher=fetcher)
    results = await corpus.download_all()
    print(f"Downloaded {len(results)} novels")


if __name__ == "__main__":
    asyncio.run(main())
