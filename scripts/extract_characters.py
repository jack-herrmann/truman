#!/usr/bin/env python3
"""Extract character profiles from downloaded novels."""
import asyncio, logging
from rich.console import Console
from rich.logging import RichHandler
from intuition.corpus.dataset import CharacterDataset
from intuition.corpus.extractor import CharacterExtractor
from intuition.corpus.gutenberg import GutenbergCorpus
from intuition.llm.client import LLMClient
logging.basicConfig(level=logging.INFO, handlers=[RichHandler(console=Console(stderr=True))])

async def main():
    corpus = GutenbergCorpus("data/novels")
    dataset = CharacterDataset("data/characters")
    llm = LLMClient()
    extractor = CharacterExtractor(llm, min_evidence=5)
    for key in corpus.list_available():
        meta = corpus.get_metadata(key)
        text = corpus.read(key)
        profiles = await extractor.extract_from_novel(text, meta["title"], meta["author"])
        for p in profiles:
            dataset.add(p)
        print(f"{meta['title']}: {len(profiles)} characters")

if __name__ == "__main__":
    asyncio.run(main())
