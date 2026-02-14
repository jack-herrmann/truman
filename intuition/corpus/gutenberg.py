"""GutenbergCorpus — download and clean novels from Project Gutenberg."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

NOVELS: dict[str, dict] = {
    "crime_and_punishment": {"id": 2554, "title": "Crime and Punishment", "author": "Fyodor Dostoevsky"},
    "pride_and_prejudice": {"id": 1342, "title": "Pride and Prejudice", "author": "Jane Austen"},
    "jane_eyre": {"id": 1260, "title": "Jane Eyre", "author": "Charlotte Bronte"},
    "great_expectations": {"id": 1400, "title": "Great Expectations", "author": "Charles Dickens"},
    "anna_karenina": {"id": 1399, "title": "Anna Karenina", "author": "Leo Tolstoy"},
    "wuthering_heights": {"id": 768, "title": "Wuthering Heights", "author": "Emily Bronte"},
    "moby_dick": {"id": 2701, "title": "Moby Dick", "author": "Herman Melville"},
    "les_miserables": {"id": 135, "title": "Les Miserables", "author": "Victor Hugo"},
    "the_brothers_karamazov": {"id": 28054, "title": "The Brothers Karamazov", "author": "Fyodor Dostoevsky"},
    "sense_and_sensibility": {"id": 161, "title": "Sense and Sensibility", "author": "Jane Austen"},
    "emma": {"id": 158, "title": "Emma", "author": "Jane Austen"},
    "david_copperfield": {"id": 766, "title": "David Copperfield", "author": "Charles Dickens"},
    "the_scarlet_letter": {"id": 25344, "title": "The Scarlet Letter", "author": "Nathaniel Hawthorne"},
    "frankenstein": {"id": 84, "title": "Frankenstein", "author": "Mary Shelley"},
    "dracula": {"id": 345, "title": "Dracula", "author": "Bram Stoker"},
    "war_and_peace": {"id": 2600, "title": "War and Peace", "author": "Leo Tolstoy"},
    "the_picture_of_dorian_gray": {"id": 174, "title": "The Picture of Dorian Gray", "author": "Oscar Wilde"},
    "middlemarch": {"id": 145, "title": "Middlemarch", "author": "George Eliot"},
    "the_count_of_monte_cristo": {"id": 1184, "title": "The Count of Monte Cristo", "author": "Alexandre Dumas"},
    "notes_from_underground": {"id": 600, "title": "Notes from Underground", "author": "Fyodor Dostoevsky"},
    "heart_of_darkness": {"id": 219, "title": "Heart of Darkness", "author": "Joseph Conrad"},
    "a_tale_of_two_cities": {"id": 98, "title": "A Tale of Two Cities", "author": "Charles Dickens"},
    "the_adventures_of_huckleberry_finn": {"id": 76, "title": "The Adventures of Huckleberry Finn", "author": "Mark Twain"},
    "don_quixote_part1": {"id": 996, "title": "Don Quixote", "author": "Miguel de Cervantes"},
    "madame_bovary": {"id": 2413, "title": "Madame Bovary", "author": "Gustave Flaubert"},
}

_MIRRORS = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
]


class GutenbergCorpus:
    def __init__(self, data_dir: str = "data/novels") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    async def download_novel(self, key: str) -> Path:
        if key not in NOVELS:
            raise KeyError(f"Unknown novel key: {key}")
        meta = NOVELS[key]
        dest = self.data_dir / f"{key}.txt"
        if dest.exists():
            return dest
        novel_id = meta["id"]
        async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
            for pattern in _MIRRORS:
                url = pattern.format(id=novel_id)
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        text = self._clean(resp.text)
                        dest.write_text(text, encoding="utf-8")
                        return dest
                except httpx.HTTPError:
                    continue
        raise RuntimeError(f"Failed to download {key}")

    async def download_all(self) -> dict[str, Path]:
        results = {}
        for key in NOVELS:
            try:
                results[key] = await self.download_novel(key)
            except Exception as exc:
                logger.warning("Skipping %s: %s", key, exc)
        return results

    def list_available(self) -> list[str]:
        return [p.stem for p in sorted(self.data_dir.glob("*.txt")) if p.stem in NOVELS]

    def read(self, key: str) -> str:
        path = self.data_dir / f"{key}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Novel not downloaded: {key}")
        return path.read_text(encoding="utf-8")

    def get_metadata(self, key: str) -> dict:
        if key not in NOVELS:
            raise KeyError(f"Unknown novel key: {key}")
        return NOVELS[key]

    @staticmethod
    def chunk_text(text: str, max_tokens: int = 3000) -> list[str]:
        max_chars = max_tokens * 4
        paragraphs = text.split("\n\n")
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if current_len + len(para) > max_chars and current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            current.append(para)
            current_len += len(para)
        if current:
            chunks.append("\n\n".join(current))
        return chunks

    @staticmethod
    def _clean(raw: str) -> str:
        start_markers = ["*** START OF THE PROJECT GUTENBERG", "*** START OF THIS PROJECT GUTENBERG"]
        end_markers = ["*** END OF THE PROJECT GUTENBERG", "*** END OF THIS PROJECT GUTENBERG",
                       "End of the Project Gutenberg", "End of Project Gutenberg"]
        text = raw
        for marker in start_markers:
            idx = text.find(marker)
            if idx != -1:
                newline = text.find("\n", idx)
                text = text[newline + 1:] if newline != -1 else text[idx:]
                break
        for marker in end_markers:
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx]
                break
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
