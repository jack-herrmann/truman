"""Novel corpus pipeline."""
from intuition.corpus.gutenberg import GutenbergCorpus
from intuition.corpus.extractor import CharacterExtractor, CharacterProfile
from intuition.corpus.dataset import CharacterDataset
from intuition.corpus.fetcher import PageFetcher, HttpxFetcher, BrightDataFetcher

__all__ = [
    "GutenbergCorpus",
    "CharacterExtractor",
    "CharacterProfile",
    "CharacterDataset",
    "PageFetcher",
    "HttpxFetcher",
    "BrightDataFetcher",
]
