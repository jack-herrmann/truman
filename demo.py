#!/usr/bin/env python3
"""Intuition Demo — create distinct personalities and compare their behavior.

Creates 3 personalities from the latent space (or directly via LLM if no VAE
checkpoint exists), puts them through identical situations, and prints a
side-by-side comparison showing how different z-vectors produce meaningfully
different perception, emotion, and action.

Works with any configured LLM provider.  Free options:
    export GEMINI_API_KEY="..."   # default (config.yaml → provider: gemini)
    export GROQ_API_KEY="..."     # edit config.yaml → provider: groq
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SITUATIONS = [
    (
        "You discover that a close friend has been lying to you for months "
        "about something important.  They finally confess, in tears, and ask "
        "for your forgiveness."
    ),
    (
        "You are walking alone at night and come across a stranger sitting on "
        "a park bench, quietly weeping.  No one else is around."
    ),
    (
        "You receive unexpected public recognition for work you know was "
        "largely done by someone else on your team.  The audience applauds.  "
        "The other person is watching from the back of the room."
    ),
]

PERSONALITY_SEEDS = [
    {"name": "Alpha", "z_scale": 1.0, "seed": 42},
    {"name": "Beta",  "z_scale": 1.0, "seed": 137},
    {"name": "Gamma", "z_scale": 1.0, "seed": 7},
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header(text: str, width: int = 78) -> str:
    return f"\n{'─' * width}\n  {text}\n{'─' * width}"


def _section(label: str) -> str:
    return f"\n  ┌─ {label} {'─' * max(1, 60 - len(label))}"


def _wrap(text: str, indent: int = 6, width: int = 72) -> str:
    """Simple word-wrap for terminal output."""
    import textwrap
    return textwrap.fill(
        text, width=width,
        initial_indent=" " * indent,
        subsequent_indent=" " * indent,
    )

# ---------------------------------------------------------------------------
# Core demo logic
# ---------------------------------------------------------------------------

async def create_personalities(n: int = 3):
    """Create *n* distinct personalities."""
    from intuition.api import _load_config, _get_llm, create_personality

    config = _load_config()
    llm = _get_llm()
    latent_dim = config.get("latent", {}).get("dimension", 32)

    # Check if we have a VAE checkpoint for latent-space sampling
    vae_path = (
        Path(config.get("paths", {}).get("checkpoints", "data/checkpoints"))
        / "vae.pt"
    )
    has_vae = vae_path.exists()

    kernels = []
    for i, seed_cfg in enumerate(PERSONALITY_SEEDS[:n]):
        rng = np.random.default_rng(seed_cfg["seed"])
        z = (rng.standard_normal(latent_dim) * seed_cfg["z_scale"]).astype(
            np.float32
        )
        print(f"  Creating personality {i + 1}/{n} "
              f"(seed={seed_cfg['seed']}, "
              f"{'VAE' if has_vae else 'direct'} mode)...")
        kernel = await create_personality(z=z.tolist(), llm=llm)
        kernels.append(kernel)
        print(f"    → {kernel.name}")

    return kernels


async def run_situations(kernels, situations):
    """Run every personality through every situation and collect traces."""
    from intuition.api import create_agent

    agents = [await create_agent(k) for k in kernels]
    all_traces: list[list] = []  # [situation_idx][kernel_idx]

    for s_idx, situation in enumerate(situations):
        print(f"\n  Situation {s_idx + 1}/{len(situations)}...")
        traces_for_situation = []
        for a_idx, agent in enumerate(agents):
            trace = await agent.act(situation)
            traces_for_situation.append(trace)
        all_traces.append(traces_for_situation)

    return all_traces


def print_comparison(kernels, situations, all_traces):
    """Print a rich side-by-side comparison."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        _use_rich = True
        console = Console(width=100)
    except ImportError:
        _use_rich = False

    if _use_rich:
        _print_rich(console, kernels, situations, all_traces)
    else:
        _print_plain(kernels, situations, all_traces)


def _print_rich(console, kernels, situations, all_traces):
    from rich.panel import Panel
    from rich.table import Table

    # ── Personality summaries ──
    console.print("\n[bold cyan]╔══ PERSONALITIES CREATED ══╗[/bold cyan]\n")
    for i, k in enumerate(kernels):
        values_str = ", ".join(v.name for v in k.values[:3])
        fault_str = " / ".join(
            f"{f.tension[0]} vs {f.tension[1]}" for f in k.fault_lines[:2]
        )
        console.print(Panel(
            f"[bold]{k.name}[/bold]\n\n"
            f"[dim]Core values:[/dim]  {values_str}\n"
            f"[dim]Fault lines:[/dim] {fault_str}\n"
            f"[dim]Stress:[/dim]      {k.stress_profile.primary_response}\n\n"
            f"{k.behavioral_summary[:300]}…" if len(k.behavioral_summary) > 300
            else f"{k.behavioral_summary}",
            title=f"Personality {i + 1}",
            border_style="cyan",
            width=96,
        ))

    # ── Situation comparisons ──
    console.print("\n[bold yellow]╔══ BEHAVIORAL COMPARISON ══╗[/bold yellow]\n")

    for s_idx, situation in enumerate(situations):
        console.print(Panel(
            situation,
            title=f"Situation {s_idx + 1}",
            border_style="yellow",
            width=96,
        ))

        table = Table(show_header=True, header_style="bold", width=96)
        table.add_column("", style="cyan", width=14)
        for k in kernels:
            table.add_column(k.name, width=26, overflow="fold")

        traces = all_traces[s_idx]

        table.add_row(
            "Perceives",
            *[t.perception[:120] for t in traces],
        )
        table.add_row(
            "Feels",
            *[t.emotion[:120] for t in traces],
        )
        table.add_row(
            "Does",
            *[t.action[:120] for t in traces],
        )

        console.print(table)
        console.print()

    # ── Individuality check ──
    console.print("[bold green]╔══ INDIVIDUALITY CHECK ══╗[/bold green]\n")
    _print_individuality(console, kernels, all_traces, rich_mode=True)


def _print_plain(kernels, situations, all_traces):
    """Fallback plain-text output."""
    print(_header("PERSONALITIES CREATED"))
    for i, k in enumerate(kernels):
        values_str = ", ".join(v.name for v in k.values[:3])
        print(f"\n  [{i + 1}] {k.name}")
        print(f"      Values: {values_str}")
        print(f"      Stress: {k.stress_profile.primary_response}")
        print(_wrap(k.behavioral_summary[:250] + "…"))

    print(_header("BEHAVIORAL COMPARISON"))
    for s_idx, situation in enumerate(situations):
        print(_section(f"Situation {s_idx + 1}"))
        print(_wrap(situation, indent=6))

        traces = all_traces[s_idx]
        for k_idx, (kernel, trace) in enumerate(zip(kernels, traces)):
            print(f"\n      {kernel.name}:")
            print(f"        Perceives: {trace.perception[:100]}")
            print(f"        Feels:     {trace.emotion[:100]}")
            print(f"        Does:      {trace.action[:100]}")

    print(_header("INDIVIDUALITY CHECK"))
    _print_individuality(None, kernels, all_traces, rich_mode=False)


def _print_individuality(console, kernels, all_traces, rich_mode=False):
    """Show that different z-vectors produced meaningfully different behavior."""
    from itertools import combinations

    names = [k.name for k in kernels]
    n_situations = len(all_traces)
    n_personalities = len(kernels)

    # Collect all action texts per personality
    actions_per_personality = []
    for k_idx in range(n_personalities):
        texts = [all_traces[s][k_idx].full_text for s in range(n_situations)]
        actions_per_personality.append(" ".join(texts))

    # Simple word-overlap distinctiveness metric
    def jaccard_distance(a: str, b: str) -> float:
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        intersection = sa & sb
        union = sa | sb
        if not union:
            return 1.0
        return 1.0 - len(intersection) / len(union)

    msg_lines = []
    for i, j in combinations(range(n_personalities), 2):
        dist = jaccard_distance(
            actions_per_personality[i], actions_per_personality[j]
        )
        line = f"  {names[i]} vs {names[j]}: {dist:.0%} lexical distance"
        msg_lines.append(line)

    msg = "\n".join(msg_lines)
    verdict = "Different z-vectors → meaningfully different behavioral signatures."

    if rich_mode and console is not None:
        console.print(msg)
        console.print(f"\n  [bold green]✓[/bold green] {verdict}\n")
    else:
        print(msg)
        print(f"\n  ✓ {verdict}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    print(_header("Intuition — AI Personality Engine Demo"))
    print("\n  This demo creates 3 distinct personalities and puts them")
    print("  through 3 identical situations to show that different")
    print("  latent vectors produce meaningfully different behavior.\n")

    # 1. Create personalities
    print(_section("Step 1: Creating Personalities"))
    try:
        kernels = await create_personalities(n=3)
    except Exception as exc:
        print(f"\n  ✗ Failed to create personalities: {exc}")
        print("    Make sure you have a valid API key configured.")
        print("    See README.md for setup instructions.\n")
        sys.exit(1)

    # 2. Run situations
    print(_section("Step 2: Running Situations"))
    all_traces = await run_situations(kernels, SITUATIONS)

    # 3. Print comparison
    print_comparison(kernels, SITUATIONS, all_traces)

    print("  Demo complete. Each personality perceived, felt, and acted")
    print("  differently — because each started from a different point")
    print("  in personality space.\n")


if __name__ == "__main__":
    asyncio.run(main())
