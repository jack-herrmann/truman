"""Tests for CharacterDataset and embeddings validation."""
from __future__ import annotations

import numpy as np
import pytest

from intuition.corpus.dataset import CharacterDataset
from intuition.corpus.extractor import CharacterProfile


def test_dataset_loads_profiles(character_data_dir):
    ds = CharacterDataset(str(character_data_dir))
    assert len(ds) == 2
    names = [p.name for p in ds]
    assert "Alice" in names
    assert "Bob" in names


def test_dataset_find_by_name(character_data_dir):
    ds = CharacterDataset(str(character_data_dir))
    alice = ds.find_by_name("alice")
    assert alice is not None
    assert alice.name == "Alice"
    assert ds.find_by_name("Nobody") is None


def test_dataset_load_embeddings(character_data_dir):
    ds = CharacterDataset(str(character_data_dir))
    emb = ds.load_embeddings()
    assert emb.shape[0] == 2
    assert emb.shape[1] == 512


def test_dataset_save_embeddings(temp_dir):
    ds = CharacterDataset(str(temp_dir))
    # No profiles yet; add one so we can compute_embeddings in another test
    profile = CharacterProfile(
        name="Solo",
        novel="Book",
        author="Author",
        personality_analysis="Solo is independent.",
    )
    profile_path = temp_dir / "book_solo.json"
    profile_path.write_text(profile.model_dump_json(indent=2))
    ds._profiles = None
    assert len(ds) == 1
    emb = np.random.randn(1, 64).astype(np.float32)
    path = ds.save_embeddings(emb, path=str(temp_dir / "emb.npy"))
    assert path.exists()
    loaded = np.load(str(path))
    assert loaded.shape == (1, 64)


def test_dataset_load_embeddings_missing_raises(temp_dir):
    ds = CharacterDataset(str(temp_dir))
    with pytest.raises(FileNotFoundError):
        ds.load_embeddings(path=str(temp_dir / "nonexistent.npy"))
