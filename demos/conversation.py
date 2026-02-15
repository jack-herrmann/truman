#!/usr/bin/env python3
"""Interactive conversation with a single personality.

Load a saved personality from JSON or create one from the latent space (requires
VAE checkpoint and API key). Then chat in the terminal; the agent responds as
that personality.

Usage:
    # Use a saved personality (no API after load)
    python3 demos/conversation.py --kernel data/checkpoints/gen1_best.json

    # Create a new personality (needs API key; uses VAE if available)
    python3 demos/conversation.py

    # Create personality "like" a character (needs API + VAE + matching embeddings)
    python3 demos/conversation.py --like "Elizabeth Bennet"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Project root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intuition.api import create_agent, create_personality, load_personality, _get_llm, _load_config


def _header(text: str) -> str:
    return f"\n{'─' * 60}\n  {text}\n{'─' * 60}"


async def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    parser = argparse.ArgumentParser(description="Chat with a single personality.")
    parser.add_argument(
        "--kernel",
        type=str,
        default=None,
        help="Path to saved PersonalityKernel JSON (if not set, create one)",
    )
    parser.add_argument(
        "--like",
        type=str,
        default=None,
        help="Create personality near this character name (e.g. 'Elizabeth Bennet')",
    )
    args = parser.parse_args()

    if args.kernel:
        path = Path(args.kernel)
        if not path.exists():
            print(f"Error: kernel file not found: {path}")
            sys.exit(1)
        kernel = load_personality(str(path))
        print(_header(f"Loaded: {kernel.name}"))
    else:
        print("Creating personality...")
        llm = _get_llm()
        if args.like:
            kernel = await create_personality(like=args.like, variation=0.5, llm=llm)
        else:
            kernel = await create_personality(temperature=0.8, llm=llm)
        print(_header(f"Created: {kernel.name}"))

    print(kernel.behavioral_summary[:400] + ("…" if len(kernel.behavioral_summary) > 400 else ""))
    print()

    llm = _get_llm()
    agent = await create_agent(kernel, llm=llm)

    print("Say something to the personality (or 'quit' / 'exit' to stop).\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        response = await agent.respond(user_input)
        print(f"\n{kernel.name}: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())
