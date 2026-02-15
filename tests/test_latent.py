"""Tests for intuition.latent (VAE, PersonalitySpace, decoder helpers)."""
from __future__ import annotations

import numpy as np
import pytest

from intuition.latent.vae import PersonalityVAE
from intuition.latent.space import PersonalitySpace
from intuition.corpus.dataset import CharacterDataset


def test_vae_forward_shape(small_vae):
    import torch
    x = torch.randn(4, 512)
    recon, mu, log_var = small_vae(x)
    assert recon.shape == (4, 512)
    assert mu.shape == (4, 8)
    assert log_var.shape == (4, 8)


def test_vae_save_load_preserves_hidden_dims(temp_dir):
    vae = PersonalityVAE(input_dim=64, latent_dim=4, hidden_dims=[32, 16])
    path = temp_dir / "vae.pt"
    vae.save(str(path))
    loaded = PersonalityVAE.load(str(path))
    assert loaded.input_dim == 64
    assert loaded.latent_dim == 4
    assert getattr(loaded, "_hidden_dims", None) == [32, 16]
    # Forward should work with same shapes
    import torch
    z = torch.randn(2, 4)
    out = loaded.decode(z)
    assert out.shape == (2, 64)


def test_vae_encode_decode_numpy(small_vae):
    emb = np.random.randn(3, 512).astype(np.float32) * 0.1
    mu, sigma = small_vae.encode_to_numpy(emb)
    assert mu.shape == (3, 8)
    assert sigma.shape == (3, 8)
    recon = small_vae.decode_from_numpy(mu)
    assert recon.shape == (3, 512)


def test_personality_space_requires_matching_embeddings(
    small_vae, character_data_dir
):
    dataset = CharacterDataset(str(character_data_dir))
    embeddings = np.load(str(character_data_dir / "embeddings.npy"))
    assert embeddings.shape[0] == 2
    assert len(dataset) == 2
    space = PersonalitySpace(small_vae, dataset, embeddings)
    assert space.latent_dim == 8
    assert space.num_characters == 2


def test_personality_space_rejects_mismatch(small_vae, character_data_dir):
    dataset = CharacterDataset(str(character_data_dir))
    # 3 rows but only 2 profiles
    embeddings = np.random.randn(3, 512).astype(np.float32) * 0.01
    with pytest.raises(ValueError) as exc_info:
        PersonalitySpace(small_vae, dataset, embeddings)
    assert "must match" in str(exc_info.value)
    assert "generate_seed_data" in str(exc_info.value)


def test_personality_space_sample(small_vae, character_data_dir):
    dataset = CharacterDataset(str(character_data_dir))
    embeddings = np.load(str(character_data_dir / "embeddings.npy"))
    space = PersonalitySpace(small_vae, dataset, embeddings)
    z, sigma = space.sample(temperature=1.0, rng=np.random.default_rng(0))
    assert z.shape == (8,)
    assert sigma.shape == (8,)


def test_personality_space_sample_near(small_vae, character_data_dir):
    dataset = CharacterDataset(str(character_data_dir))
    embeddings = np.load(str(character_data_dir / "embeddings.npy"))
    space = PersonalitySpace(small_vae, dataset, embeddings)
    z, sigma = space.sample_near("alice", radius=0.5, rng=np.random.default_rng(0))
    assert z.shape == (8,)
    with pytest.raises(KeyError):
        space.sample_near("Nonexistent", radius=0.5)


def test_personality_space_interpolate(small_vae, character_data_dir):
    dataset = CharacterDataset(str(character_data_dir))
    embeddings = np.load(str(character_data_dir / "embeddings.npy"))
    space = PersonalitySpace(small_vae, dataset, embeddings)
    z1, _ = space.sample(rng=np.random.default_rng(1))
    z2, _ = space.sample(rng=np.random.default_rng(2))
    mid = space.interpolate(z1, z2, 0.5)
    assert mid.shape == z1.shape
    # t=0 should be close to z1, t=1 to z2
    i0 = space.interpolate(z1, z2, 0.0)
    i1 = space.interpolate(z1, z2, 1.0)
    np.testing.assert_allclose(i0, z1, atol=1e-5)
    np.testing.assert_allclose(i1, z2, atol=1e-5)


def test_personality_space_nearest_characters(small_vae, character_data_dir):
    dataset = CharacterDataset(str(character_data_dir))
    embeddings = np.load(str(character_data_dir / "embeddings.npy"))
    space = PersonalitySpace(small_vae, dataset, embeddings)
    z, _ = space.sample(rng=np.random.default_rng(99))
    neighbors = space.nearest_characters(z, k=2)
    assert len(neighbors) == 2
    for profile, dist in neighbors:
        assert profile.name in ("Alice", "Bob")
        assert dist >= 0


def test_decoder_distance_to_weights():
    from intuition.latent.decoder import PersonalityDecoder
    weights = PersonalityDecoder._distance_to_weights([1.0, 2.0, 4.0])
    assert len(weights) == 3
    assert abs(sum(weights) - 1.0) < 1e-6
    # Closer distance -> larger weight
    assert weights[2] < weights[1] < weights[0]
