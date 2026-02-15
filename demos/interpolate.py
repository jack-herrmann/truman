#!/usr/bin/env python3
"""Interpolate between two personalities in latent space.

Load two saved PersonalityKernel JSON files, interpolate their z-vectors at
several t values (0, 0.25, 0.5, 0.75, 1.0), and optionally decode the
midpoint (t=0.5) to a new kernel via the LLM. Prints summaries so you can
see how personality shifts along the continuum.

Usage:
    # Show interpolation between two saved kernels (no API for endpoints)
    python3 demos/interpolate.py data/checkpoints/final_top1.json data/checkpoints/final_top2.json

    # Decode the midpoint with LLM (requires API key and VAE + matching embeddings)
    python3 demos/interpolate.py --decode-mid kernel_a.json kernel_b.json
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np

# Project root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intuition.api import load_personality, _load_config, _get_llm
from intuition.latent.space import PersonalitySpace
from intuition.latent.vae import PersonalityVAE
from intuition.corpus.dataset import CharacterDataset
from intuition.latent.decoder import PersonalityDecoder


def _header(text: str) -> str:
    return f"\n{'─' * 60}\n  {text}\n{'─' * 60}"


def interpolate_z(z_a: list[float], z_b: list[float], t: float) -> np.ndarray:
    """Linear interpolation between two z vectors. For spherical use PersonalitySpace.interpolate."""
    a = np.array(z_a, dtype=np.float32)
    b = np.array(z_b, dtype=np.float32)
    return (1 - t) * a + t * b


async def main() -> None:
    parser = argparse.ArgumentParser(description="Interpolate between two personalities.")
    parser.add_argument("kernel_a", type=str, help="Path to first PersonalityKernel JSON")
    parser.add_argument("kernel_b", type=str, help="Path to second PersonalityKernel JSON")
    parser.add_argument(
        "--decode-mid",
        action="store_true",
        help="Decode midpoint (t=0.5) with LLM (needs API key and VAE)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of interpolation steps (default 5: t=0, 0.25, 0.5, 0.75, 1.0)",
    )
    args = parser.parse_args()

    path_a = Path(args.kernel_a)
    path_b = Path(args.kernel_b)
    if not path_a.exists() or not path_b.exists():
        print("Error: both kernel files must exist.")
        sys.exit(1)

    k_a = load_personality(str(path_a))
    k_b = load_personality(str(path_b))

    print(_header("Endpoint personalities"))
    print(f"  A: {k_a.name}")
    print(f"     {k_a.behavioral_summary[:200]}…")
    print()
    print(f"  B: {k_b.name}")
    print(f"     {k_b.behavioral_summary[:200]}…")

    print(_header("Interpolated z (summary)"))
    steps = max(2, args.steps)
    for i in range(steps):
        t = i / (steps - 1) if steps > 1 else 1.0
        z = interpolate_z(k_a.z, k_b.z, t)
        label = "A" if t == 0 else ("B" if t == 1.0 else f"t={t:.2f}")
        dist_a = float(np.linalg.norm(z - np.array(k_a.z, dtype=np.float32)))
        dist_b = float(np.linalg.norm(z - np.array(k_b.z, dtype=np.float32)))
        print(f"  {label:6}  |z-A|={dist_a:.3f}  |z-B|={dist_b:.3f}")

    if args.decode_mid:
        print(_header("Decoding midpoint (t=0.5) with LLM"))
        config = _load_config()
        vae_path = Path(config.get("paths", {}).get("checkpoints", "data/checkpoints")) / "vae.pt"
        if not vae_path.exists():
            print("  VAE checkpoint not found. Run: python3 scripts/generate_seed_data.py")
            return
        try:
            vae = PersonalityVAE.load(str(vae_path))
            dataset = CharacterDataset()
            embeddings = dataset.load_embeddings()
            if embeddings.shape[0] != len(dataset):
                print("  Embeddings and dataset size mismatch. Run: python3 scripts/generate_seed_data.py")
                return
            space = PersonalitySpace(vae, dataset, embeddings)
            decoder = PersonalityDecoder(space, _get_llm())
            z_mid = interpolate_z(k_a.z, k_b.z, 0.5)
            sigma_mid = interpolate_z(k_a.sigma, k_b.sigma, 0.5)
            k_mid = await decoder.decode(z_mid, sigma_mid)
            print(f"  Midpoint personality: {k_mid.name}")
            print(f"  {k_mid.behavioral_summary[:350]}…")
        except ValueError as e:
            print(f"  {e}")
        except Exception as e:
            print(f"  Decode failed: {e}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
