#!/usr/bin/env python3
"""Draw yourself — each personality draws a self-portrait (no LLM at draw time).

The drawing is driven entirely by the PersonalityKernel: we map kernel fields
→ drawing spec (position, size, posture, mood, environment) → renderer draws.
So the personalities we train are the ones "doing" the drawing.

================================================================================
STEPS TO RUN (so you actually get a drawing)
================================================================================

1. Install dependencies (once):
   pip install -r requirements.txt

2. Choose one of two ways to run:

   A) Use saved personality kernel(s) — NO API KEY needed to draw:
      python draw_yourself.py --kernel path/to/kernel.json --save my_drawing.png
      (Kernel = a PersonalityKernel JSON saved from training or create_personality.)

   B) Create 4 new personalities then draw — API KEY required:
      Set an API key for your chosen LLM provider, then run:
      export GEMINI_API_KEY="your_key"   # or OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
      python draw_yourself.py --save my_drawing.png
      (Uses config.yaml llm.provider; default is gemini.)

3. To get a PNG file without opening a window (e.g. on a server):
   python draw_yourself.py --no-window --save my_drawing.png
   If you omit --save when using --no-window, the file is saved as
   draw_yourself_output.png so you always get a drawing.

4. If you see "pygame not installed":
   pip install pygame

5. If personality creation fails with an API error:
   Set the correct env var for your provider (see config.yaml comments)
   or put the key in config.yaml (do not commit keys to git).
================================================================================
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Drawing spec (filled from PersonalityKernel; renderer draws from it)
# ---------------------------------------------------------------------------
# Framed as "draw yourself": figure = self, rain = environmental pressure,
# umbrella = protection, mood = how you see your world.


class DrawYourselfSpec(BaseModel):
    """Self-portrait drawing spec derived from a PersonalityKernel."""

    person_x: float = Field(ge=0.0, le=1.0, description="Horizontal position (0=left, 1=right)")
    person_y: float = Field(ge=0.0, le=1.0, description="Vertical position (0=top, 1=bottom)")
    person_size: float = Field(ge=0.05, le=0.6, description="Size of figure (0.05=small, 0.4=large)")
    has_umbrella: bool = Field(description="Whether the figure has protection/shelter")
    umbrella_angle: float = Field(ge=-90.0, le=90.0, default=0.0, description="Tilt in degrees")
    rain_intensity: float = Field(ge=0.0, le=1.0, description="Environmental pressure (0=none, 1=heavy)")
    posture: str = Field(description="hunched | erect | crouched | fleeing")
    mood: str = Field(description="dark | somber | neutral | hopeful | calm | stormy | lonely")


# ---------------------------------------------------------------------------
# Kernel → spec (our personalities do the drawing)
# ---------------------------------------------------------------------------


def kernel_to_drawing_spec(kernel) -> DrawYourselfSpec:
    """Map our trained PersonalityKernel to a self-portrait spec. No LLM."""
    from intuition.core.kernel import PersonalityKernel
    import math

    k = kernel  # type: PersonalityKernel
    z = k.z if k.z else [0.0] * 2
    z0 = z[0] if len(z) > 0 else 0.0
    z1 = z[1] if len(z) > 1 else 0.0
    person_x = 0.5 + 0.3 * math.tanh(z0)
    person_y = 0.5 + 0.3 * math.tanh(z1)

    dominance = getattr(k.social_style, "dominance", 0.5)
    person_size = 0.12 + 0.25 * max(0.0, min(1.0, dominance))

    response = (k.stress_profile.primary_response or "freeze").lower()
    if "fight" in response:
        posture = "erect"
    elif "flight" in response or "flee" in response:
        posture = "fleeing"
    elif "fawn" in response:
        posture = "crouched"
    else:
        posture = "hunched"

    valence = getattr(k.emotional_baseline, "default_valence", 0.0)
    if valence < -0.3:
        mood = "dark"
    elif valence < 0.0:
        mood = "somber"
    elif valence > 0.3:
        mood = "hopeful"
    elif valence > 0.1:
        mood = "calm"
    else:
        mood = "neutral"

    reactivity = getattr(k.emotional_baseline, "reactivity", 0.5)
    rain_intensity = 0.3 + 0.6 * max(0.0, min(1.0, reactivity))

    threshold = getattr(k.stress_profile, "threshold", 0.5)
    conflict = (getattr(k.social_style, "conflict_approach", "") or "").lower()
    has_umbrella = threshold > 0.5 or "avoid" in conflict or "deflect" in conflict
    umbrella_angle = 0.0
    if has_umbrella and len(z) > 2:
        umbrella_angle = 30.0 * max(-1.0, min(1.0, z[2]))

    return DrawYourselfSpec(
        person_x=person_x,
        person_y=person_y,
        person_size=person_size,
        has_umbrella=has_umbrella,
        umbrella_angle=umbrella_angle,
        rain_intensity=rain_intensity,
        posture=posture,
        mood=mood,
    )


# ---------------------------------------------------------------------------
# Renderer: spec → pixels (pygame)
# ---------------------------------------------------------------------------


def _mood_to_colors(mood: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    m = mood.lower()
    if "dark" in m or "stormy" in m or "somber" in m:
        return (40, 45, 60), (80, 80, 90)
    if "lonely" in m:
        return (70, 75, 85), (100, 100, 110)
    if "hopeful" in m or "calm" in m:
        return (100, 110, 130), (60, 70, 90)
    return (90, 95, 110), (70, 75, 85)


def draw_yourself(surface, rect, spec: DrawYourselfSpec, title: str = "") -> None:
    """Draw the self-portrait from spec into the given rect."""
    import math
    import random
    import pygame

    x, y, w, h = rect
    cx = x + w * spec.person_x
    cy = y + h * spec.person_y
    scale = min(w, h) * spec.person_size * 3.0

    sky_color, figure_color = _mood_to_colors(spec.mood)
    rain_color = (180, 190, 210)

    pygame.draw.rect(surface, sky_color, rect)

    rng = random.Random(hash(spec.mood) % 2**32)
    n_drops = int(80 + spec.rain_intensity * 200)
    for _ in range(n_drops):
        rx = x + rng.randint(0, w)
        ry = y + rng.randint(0, h)
        length = 4 + int(spec.rain_intensity * 8)
        pygame.draw.line(surface, rain_color, (rx, ry), (rx + 2, ry + length), 1)

    head_r = max(3, int(scale * 0.35))
    body_len = int(scale * 0.5)
    arm_len = int(scale * 0.35)
    leg_len = int(scale * 0.45)

    post = spec.posture.lower()
    if "hunched" in post:
        body_angle = 0.4
        arm_angle_l, arm_angle_r = -0.3, -0.3
    elif "crouched" in post:
        body_len = int(scale * 0.3)
        leg_len = int(scale * 0.25)
        body_angle = 0.1
        arm_angle_l, arm_angle_r = 0.2, 0.2
    elif "fleeing" in post or "running" in post:
        body_angle = -0.2
        arm_angle_l, arm_angle_r = -0.8, 0.8
    else:
        body_angle = 0.0
        arm_angle_l, arm_angle_r = -0.5, 0.5

    pygame.draw.circle(surface, figure_color, (int(cx), int(cy - scale * 0.5)), head_r)
    bx = cx + math.sin(body_angle) * body_len
    by = cy - scale * 0.5 + head_r + math.cos(body_angle) * body_len
    pygame.draw.line(surface, figure_color, (int(cx), int(cy - scale * 0.5 + head_r)), (int(bx), int(by)), 2)
    ax, ay = bx, by - body_len * 0.3
    pygame.draw.line(surface, figure_color, (int(ax), int(ay)), (int(ax + math.cos(arm_angle_l) * arm_len), int(ay + math.sin(arm_angle_l) * arm_len)), 2)
    pygame.draw.line(surface, figure_color, (int(ax), int(ay)), (int(ax + math.cos(arm_angle_r) * arm_len), int(ay + math.sin(arm_angle_r) * arm_len)), 2)
    pygame.draw.line(surface, figure_color, (int(bx), int(by)), (int(bx - 8), int(by + leg_len)), 2)
    pygame.draw.line(surface, figure_color, (int(bx), int(by)), (int(bx + 8), int(by + leg_len)), 2)

    if spec.has_umbrella:
        rad = math.radians(spec.umbrella_angle)
        ux = int(cx + math.cos(rad) * scale * 0.6)
        uy = int(cy - scale * 0.9)
        u_radius = int(scale * 0.5)
        umbrella_color = (120, 120, 130)
        pygame.draw.circle(surface, umbrella_color, (ux, uy), u_radius, 2)
        pygame.draw.line(surface, figure_color, (int(cx), int(cy - scale * 0.5)), (ux, uy + u_radius), 2)

    if title:
        try:
            font = pygame.font.Font(None, 24)
            text = font.render(title, True, (220, 220, 220))
            surface.blit(text, (x + 4, y + 4))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Demo: create or load kernels → spec each → draw → save and/or show window
# ---------------------------------------------------------------------------

async def run_demo(
    kernels=None,
    use_window: bool = True,
    save_path: str | None = None,
) -> None:
    """Create or load personalities, derive self-portrait spec from each, draw, then save and/or show window."""
    from intuition.api import _get_llm, _load_config, create_personality, load_personality
    import numpy as np

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    # --- Get or create kernels ---
    if kernels is None:
        config = _load_config()
        try:
            llm = _get_llm()
        except Exception as e:
            print("Could not create LLM client. Set an API key for your provider.")
            print("  Examples: export GEMINI_API_KEY='...'  or  export OPENAI_API_KEY='...'")
            print("  See config.yaml and README for provider options.")
            print(f"  Error: {e}")
            sys.exit(1)
        latent_dim = config.get("latent", {}).get("dimension", 32)
        seeds = [42, 137, 7, 99]
        kernels = []
        for i, seed in enumerate(seeds):
            rng = np.random.default_rng(seed)
            z = rng.standard_normal(latent_dim).astype(np.float32)
            try:
                k = await create_personality(z=z.tolist(), llm=llm)
                kernels.append(k)
                print(f"  Personality {i + 1}: {k.name}")
            except Exception as e:
                print(f"  Failed to create personality {i + 1}: {e}")
                print("  Check your API key and config (config.yaml, llm.provider).")
                sys.exit(1)

    if not kernels:
        print("No personalities to draw. Use --kernel path/to/kernel.json or run without --kernel to create 4.")
        sys.exit(1)

    # --- Kernel → spec (no LLM) ---
    specs = []
    titles = []
    for k in kernels:
        spec = kernel_to_drawing_spec(k)
        specs.append(spec)
        titles.append(k.name)
        print(f"  {k.name} draws self: posture={spec.posture}, mood={spec.mood}, umbrella={spec.has_umbrella}")

    # --- Pygame: must have it to draw ---
    try:
        import pygame
        pygame.init()
    except ImportError:
        print("pygame is required to produce a drawing. Install with: pip install pygame")
        sys.exit(1)

    cell_w, cell_h = 320, 280
    n = len(specs)
    cols = 2 if n >= 2 else 1
    rows = (n + cols - 1) // cols
    width, height = cols * cell_w, rows * cell_h

    surf = pygame.Surface((width, height))
    surf.fill((30, 30, 35))
    for i, (spec, title) in enumerate(zip(specs, titles)):
        row, col = i // cols, i % cols
        rect = (col * cell_w, row * cell_h, cell_w, cell_h)
        draw_yourself(surf, rect, spec, title=title)

    if save_path:
        try:
            pygame.image.save(surf, save_path)
            print(f"  Saved: {save_path}")
        except Exception as e:
            print(f"  Could not save to {save_path}: {e}")
            sys.exit(1)

    if use_window:
        screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        print("\n  Close the window to exit (or press Escape).")
        running = True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    running = False
            pygame.time.wait(50)
        pygame.quit()
    else:
        pygame.quit()

    if save_path:
        print("Done. You have a drawing at:", save_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw yourself — personalities draw a self-portrait from their kernel (no LLM at draw time).",
    )
    parser.add_argument("--kernel", action="append", dest="kernels", metavar="PATH", help="Path to saved PersonalityKernel JSON (repeat for multiple). If not set, create 4 (requires API key).")
    parser.add_argument("--no-window", action="store_true", help="Do not open a window. Saves to --save or draw_yourself_output.png.")
    parser.add_argument("--save", metavar="PATH", default=None, help="Save the drawing to this PNG file.")
    args = parser.parse_args()

    # If no window, always save somewhere so the user gets a drawing
    save_path = args.save
    if args.no_window and save_path is None:
        save_path = "draw_yourself_output.png"
        print(f"  (No --save given; saving to {save_path})")

    async def _main():
        kernels = None
        if args.kernels:
            from intuition.api import load_personality
            kernels = []
            for p in args.kernels:
                path = Path(p)
                if not path.exists():
                    print(f"Kernel file not found: {path}")
                    sys.exit(1)
                try:
                    k = load_personality(str(path))
                    kernels.append(k)
                except Exception as e:
                    print(f"Failed to load kernel from {path}: {e}")
                    sys.exit(1)
        await run_demo(kernels=kernels, use_window=not args.no_window, save_path=save_path)

    print("Draw yourself — each personality draws a self-portrait from their kernel.\n")
    asyncio.run(_main())


if __name__ == "__main__":
    main()
