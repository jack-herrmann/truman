#!/usr/bin/env python3
"""Download novels from Project Gutenberg."""
import asyncio, logging
from rich.console import Console
from rich.logging import RichHandler
from intuition.corpus.gutenberg import GutenbergCorpus
logging.basicConfig(level=logging.INFO, handlers=[RichHandler(console=Console(stderr=True))])

async def main():
    corpus = GutenbergCorpus("data/novels")
    results = await corpus.download_all()
    print(f"Downloaded {len(results)} novels")

if __name__ == "__main__":
    asyncio.run(main())
