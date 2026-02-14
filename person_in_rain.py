#!/usr/bin/env python3
"""
Person in the Rain — AI Personality Demo
Four agents, four personalities, one prompt.
"""

import pygame
import math
import time
import sys
import random

# ─── Setup ───────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 1400, 800
CANVAS_W, CANVAS_H = 520, 310

# Equal spacing calculation
TOP_AREA = 90
H_MARGIN = (WIDTH - 2 * CANVAS_W) // 3
V_MARGIN = (HEIGHT - TOP_AREA - 2 * CANVAS_H) // 3

CANVAS_POSITIONS = [
    (H_MARGIN, TOP_AREA + V_MARGIN),
    (H_MARGIN * 2 + CANVAS_W, TOP_AREA + V_MARGIN),
    (H_MARGIN, TOP_AREA + V_MARGIN * 2 + CANVAS_H),
    (H_MARGIN * 2 + CANVAS_W, TOP_AREA + V_MARGIN * 2 + CANVAS_H),
]

BG_COLOR = (18, 18, 24)
CANVAS_BG = (245, 245, 242)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Person in the Rain")
font_prompt = pygame.font.SysFont("Georgia", 34, bold=True)

canvases = [pygame.Surface((CANVAS_W, CANVAS_H)) for _ in range(4)]
for c in canvases:
    c.fill(CANVAS_BG)


# ─── Drawing primitives ──────────────────────────────────────────────────────
def cmd_line(surface, x1, y1, x2, y2, color, thickness=2):
    pygame.draw.line(surface, color, (x1, y1), (x2, y2), thickness)

def cmd_circle(surface, x, y, r, color, thickness=0):
    pygame.draw.circle(surface, color, (x, y), r, thickness)

def cmd_ellipse(surface, x, y, w, h, color, thickness=0):
    pygame.draw.ellipse(surface, color, (x - w//2, y - h//2, w, h), thickness)

def cmd_rect(surface, x, y, w, h, color, thickness=0):
    pygame.draw.rect(surface, color, (x, y, w, h), thickness)

def cmd_arc(surface, x, y, w, h, start, end, color, thickness=2):
    pygame.draw.arc(surface, color, (x - w//2, y - h//2, w, h), start, end, thickness)

def cmd_polygon(surface, points, color, thickness=0):
    pygame.draw.polygon(surface, color, points, thickness)

def cmd_lines(surface, points, color, thickness=2, closed=False):
    pygame.draw.lines(surface, color, closed, points, thickness)


# ─── Personality: The Stoic ──────────────────────────────────────────────────
def build_stoic():
    cmds = []
    person_color = (60, 65, 70)
    rain_color = (140, 160, 180)
    ground_color = (120, 125, 115)
    cx, cy_head = 250, 120

    # Person first — slow, deliberate, unhurried
    cmds.append((700, cmd_circle, (cx, cy_head, 22, person_color, 3)))
    cmds.append((280, cmd_line, (cx, cy_head + 22, cx, cy_head + 35, person_color, 3)))
    cmds.append((220, cmd_line, (cx, cy_head + 35, cx, cy_head + 100, person_color, 3)))
    cmds.append((200, cmd_line, (cx, cy_head + 45, cx - 30, cy_head + 85, person_color, 3)))
    cmds.append((160, cmd_line, (cx - 30, cy_head + 85, cx - 28, cy_head + 100, person_color, 2)))
    cmds.append((200, cmd_line, (cx, cy_head + 45, cx + 30, cy_head + 85, person_color, 3)))
    cmds.append((160, cmd_line, (cx + 30, cy_head + 85, cx + 28, cy_head + 100, person_color, 2)))
    cmds.append((180, cmd_line, (cx, cy_head + 100, cx - 18, cy_head + 160, person_color, 3)))
    cmds.append((130, cmd_line, (cx - 18, cy_head + 160, cx - 25, cy_head + 165, person_color, 2)))
    cmds.append((180, cmd_line, (cx, cy_head + 100, cx + 22, cy_head + 155, person_color, 3)))
    cmds.append((130, cmd_line, (cx + 22, cy_head + 155, cx + 30, cy_head + 158, person_color, 2)))
    # Stoic face
    cmds.append((300, cmd_circle, (cx - 7, cy_head - 4, 2, person_color, 0)))
    cmds.append((180, cmd_circle, (cx + 7, cy_head - 4, 2, person_color, 0)))
    cmds.append((220, cmd_line, (cx - 6, cy_head + 8, cx + 6, cy_head + 8, person_color, 2)))

    # Ground
    cmds.append((400, cmd_line, (30, 270, 490, 270, ground_color, 2)))

    # Rain — moderate, steady
    random.seed(42)
    for _ in range(60):
        rx = random.randint(20, 500)
        ry = random.randint(5, 245)
        length = random.randint(12, 22)
        cmds.append((60, cmd_line, (rx, ry, rx - 2, ry + length, rain_color, 1)))

    return cmds


# ─── Personality: The Anxious ─────────────────────────────────────────────────
def build_anxious():
    cmds = []
    cloud_color = (55, 55, 65)
    rain_color = (70, 85, 120)
    heavy_rain = (50, 65, 100)
    person_color = (80, 75, 75)
    umbrella_color = (90, 80, 85)

    random.seed(99)

    # Clouds FIRST — rapid
    cmds.append((120, cmd_ellipse, (110, 40, 170, 60, cloud_color, 0)))
    cmds.append((25, cmd_ellipse, (230, 34, 190, 52, cloud_color, 0)))
    cmds.append((20, cmd_ellipse, (370, 38, 180, 55, cloud_color, 0)))
    cmds.append((15, cmd_ellipse, (460, 44, 130, 48, cloud_color, 0)))
    cmds.append((12, cmd_ellipse, (170, 50, 140, 45, (45, 45, 55), 0)))
    cmds.append((10, cmd_ellipse, (330, 44, 150, 44, (45, 45, 55), 0)))

    # Heavy rain — frantic, 6ms between strokes
    for _ in range(150):
        rx = random.randint(10, 510)
        ry = random.randint(55, 280)
        length = random.randint(14, 32)
        offset = random.randint(-4, 4)
        thickness = random.choice([1, 1, 1, 2])
        c = heavy_rain if random.random() > 0.4 else rain_color
        cmds.append((5, cmd_line, (rx, ry, rx + offset, ry + length, c, thickness)))

    # Tiny person — appears late, small
    px, py = 350, 220
    cmds.append((60, cmd_circle, (px, py, 10, person_color, 2)))
    cmds.append((20, cmd_line, (px, py + 10, px - 3, py + 40, person_color, 2)))
    cmds.append((14, cmd_line, (px - 3, py + 18, px - 14, py + 30, person_color, 2)))
    cmds.append((14, cmd_line, (px - 3, py + 18, px + 10, py + 32, person_color, 2)))
    cmds.append((14, cmd_line, (px - 3, py + 40, px - 8, py + 58, person_color, 2)))
    cmds.append((14, cmd_line, (px - 3, py + 40, px + 4, py + 58, person_color, 2)))

    # Tiny umbrella
    cmds.append((35, cmd_arc, (px + 5, py - 16, 30, 20, 0, math.pi, umbrella_color, 2)))
    cmds.append((15, cmd_line, (px + 5, py - 6, px + 5, py + 15, umbrella_color, 1)))

    # Jagged ground
    ground_points = [(10, 282)]
    gx = 10
    while gx < 510:
        gx += random.randint(12, 35)
        gy = 282 + random.randint(-5, 5)
        ground_points.append((min(gx, 510), gy))
    cmds.append((25, cmd_lines, (ground_points, (70, 70, 65), 1)))

    # Splashes
    for _ in range(25):
        sx = random.randint(30, 490)
        sy = random.randint(272, 285)
        cmds.append((4, cmd_line, (sx, sy, sx + random.randint(-5, 5), sy - random.randint(3, 8), rain_color, 1)))

    return cmds


# ─── Personality: The Romantic ────────────────────────────────────────────────
def build_romantic():
    cmds = []
    person_color = (100, 70, 80)
    rain_color = (100, 130, 175)
    warm_accent = (200, 160, 100)
    flower_color = (190, 110, 120)
    flower2 = (180, 140, 180)
    puddle_color = (160, 175, 200)
    ground_color = (130, 120, 95)

    random.seed(77)

    # Soft sky — slow, atmospheric
    for i in range(0, CANVAS_W, 14):
        alpha_c = (200 + random.randint(-15, 15), 210 + random.randint(-15, 15), 225 + random.randint(-10, 10))
        cmds.append((22, cmd_line, (i, 0, i + random.randint(-3, 3), 65 + random.randint(-10, 10), alpha_c, 3)))

    cx, cy_head = 260, 115

    # Person — very slow, contemplative
    cmds.append((450, cmd_circle, (cx, cy_head, 20, person_color, 2)))
    cmds.append((160, cmd_arc, (cx, cy_head - 5, 42, 30, 0.3, math.pi - 0.3, person_color, 2)))
    cmds.append((220, cmd_arc, (cx - 6, cy_head - 2, 6, 4, 0, math.pi, person_color, 2)))
    cmds.append((180, cmd_arc, (cx + 6, cy_head - 2, 6, 4, 0, math.pi, person_color, 2)))
    cmds.append((200, cmd_arc, (cx, cy_head + 8, 10, 6, math.pi, 2 * math.pi, person_color, 2)))

    cmds.append((130, cmd_line, (cx, cy_head + 20, cx, cy_head + 32, person_color, 2)))
    body_points = [(cx, cy_head + 32), (cx - 2, cy_head + 55), (cx, cy_head + 90)]
    cmds.append((160, cmd_lines, (body_points, person_color, 2)))

    # Arms open
    arm_l = [(cx, cy_head + 40), (cx - 25, cy_head + 30), (cx - 45, cy_head + 20)]
    cmds.append((150, cmd_lines, (arm_l, person_color, 2)))
    cmds.append((90, cmd_line, (cx - 45, cy_head + 20, cx - 50, cy_head + 15, person_color, 1)))
    cmds.append((70, cmd_line, (cx - 45, cy_head + 20, cx - 52, cy_head + 20, person_color, 1)))
    cmds.append((70, cmd_line, (cx - 45, cy_head + 20, cx - 49, cy_head + 25, person_color, 1)))

    arm_r = [(cx, cy_head + 40), (cx + 25, cy_head + 32), (cx + 42, cy_head + 25)]
    cmds.append((150, cmd_lines, (arm_r, person_color, 2)))
    cmds.append((70, cmd_line, (cx + 42, cy_head + 25, cx + 47, cy_head + 20, person_color, 1)))
    cmds.append((70, cmd_line, (cx + 42, cy_head + 25, cx + 48, cy_head + 26, person_color, 1)))

    # Dress
    dress_points = [(cx - 18, cy_head + 65), (cx, cy_head + 90), (cx + 18, cy_head + 65)]
    cmds.append((120, cmd_lines, (dress_points, person_color, 2)))
    cmds.append((100, cmd_line, (cx - 6, cy_head + 90, cx - 12, cy_head + 140, person_color, 2)))
    cmds.append((100, cmd_line, (cx + 6, cy_head + 90, cx + 12, cy_head + 140, person_color, 2)))

    # Scarf
    cmds.append((160, cmd_arc, (cx, cy_head + 30, 16, 10, math.pi, 2 * math.pi, warm_accent, 2)))
    cmds.append((90, cmd_line, (cx + 8, cy_head + 30, cx + 15, cy_head + 50, warm_accent, 2)))

    # Ground with puddles
    cmds.append((250, cmd_line, (20, 260, 500, 260, ground_color, 2)))
    cmds.append((160, cmd_ellipse, (180, 268, 65, 11, puddle_color, 1)))
    cmds.append((130, cmd_ellipse, (350, 272, 50, 9, puddle_color, 1)))
    cmds.append((110, cmd_ellipse, (260, 276, 38, 7, puddle_color, 1)))

    # Flowers
    for fx, fy in [(75, 252), (140, 255), (400, 250), (465, 253)]:
        cmds.append((130, cmd_line, (fx, fy, fx, fy + 12, (80, 130, 70), 1)))
        fc = flower_color if random.random() > 0.5 else flower2
        cmds.append((90, cmd_circle, (fx, fy, 5, fc, 0)))
        cmds.append((50, cmd_circle, (fx, fy, 5, fc, 1)))

    # Gentle rain — slow
    for _ in range(45):
        rx = random.randint(20, 500)
        ry = random.randint(30, 240)
        length = random.randint(10, 18)
        c = rain_color if random.random() > 0.2 else (160, 150, 180)
        cmds.append((80, cmd_line, (rx, ry, rx - 1, ry + length, c, 1)))

    # Rainbow
    rainbow_colors = [(200, 120, 120), (200, 170, 100), (200, 200, 120),
                      (120, 180, 130), (120, 150, 200), (150, 120, 190)]
    for i, rc in enumerate(rainbow_colors):
        cmds.append((110, cmd_arc, (450, 25, 110 + i*10, 55 + i*5, math.pi, 2*math.pi, rc, 2)))

    return cmds


# ─── Personality: The Pragmatist ──────────────────────────────────────────────
def build_pragmatist():
    cmds = []
    person_color = (55, 60, 68)
    coat_color = (70, 75, 90)
    umbrella_top = (50, 55, 75)
    boot_color = (60, 55, 50)
    rain_color = (130, 150, 170)
    ground_color = (110, 110, 100)

    random.seed(123)
    cx, cy_head = 240, 108

    # Umbrella FIRST
    cmds.append((450, cmd_arc, (cx, cy_head - 45, 100, 50, 0, math.pi, umbrella_top, 3)))
    cmds.append((90, cmd_line, (cx - 50, cy_head - 45, cx + 50, cy_head - 45, umbrella_top, 3)))
    cmds.append((70, cmd_line, (cx, cy_head - 45, cx, cy_head + 10, person_color, 2)))
    cmds.append((55, cmd_arc, (cx + 6, cy_head + 10, 12, 10, math.pi, 2*math.pi, person_color, 2)))

    # Person
    cmds.append((160, cmd_circle, (cx, cy_head, 18, person_color, 2)))
    cmds.append((80, cmd_circle, (cx - 6, cy_head - 3, 2, person_color, 0)))
    cmds.append((55, cmd_circle, (cx + 6, cy_head - 3, 2, person_color, 0)))
    cmds.append((65, cmd_arc, (cx, cy_head + 6, 8, 4, math.pi, 2*math.pi, person_color, 1)))

    # Coat
    cmds.append((100, cmd_line, (cx, cy_head + 18, cx, cy_head + 90, person_color, 2)))
    coat_pts = [(cx - 22, cy_head + 25), (cx - 25, cy_head + 85),
                (cx + 25, cy_head + 85), (cx + 22, cy_head + 25)]
    cmds.append((90, cmd_lines, (coat_pts, coat_color, 2, True)))
    for by in [cy_head + 40, cy_head + 55, cy_head + 70]:
        cmds.append((40, cmd_circle, (cx, by, 2, person_color, 0)))

    # Arms
    cmds.append((65, cmd_line, (cx + 22, cy_head + 35, cx + 35, cy_head + 60, person_color, 2)))
    cmds.append((55, cmd_line, (cx - 22, cy_head + 35, cx - 8, cy_head + 10, person_color, 2)))

    # Legs + boots
    cmds.append((60, cmd_line, (cx - 8, cy_head + 85, cx - 14, cy_head + 140, person_color, 2)))
    cmds.append((60, cmd_line, (cx + 8, cy_head + 85, cx + 14, cy_head + 140, person_color, 2)))
    cmds.append((50, cmd_rect, (cx - 22, cy_head + 130, 16, 14, boot_color, 0)))
    cmds.append((50, cmd_rect, (cx + 6, cy_head + 130, 16, 14, boot_color, 0)))

    # Ground
    cmds.append((110, cmd_line, (30, 255, 490, 255, ground_color, 2)))
    cmds.append((65, cmd_ellipse, (340, 260, 45, 8, (150, 160, 170), 1)))

    # Rain — not under umbrella
    for _ in range(55):
        rx = random.randint(15, 510)
        ry = random.randint(10, 245)
        length = random.randint(12, 20)
        if cx - 50 < rx < cx + 50 and cy_head - 50 < ry < cy_head + 90:
            continue
        cmds.append((30, cmd_line, (rx, ry, rx - 1, ry + length, rain_color, 1)))

    # Briefcase
    bx, by = cx + 38, cy_head + 62
    cmds.append((95, cmd_rect, (bx, by, 22, 16, person_color, 2)))
    cmds.append((45, cmd_line, (bx + 6, by, bx + 6, by - 5, person_color, 1)))
    cmds.append((35, cmd_line, (bx + 16, by, bx + 16, by - 5, person_color, 1)))
    cmds.append((30, cmd_line, (bx + 6, by - 5, bx + 16, by - 5, person_color, 1)))

    return cmds


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    clock = pygame.time.Clock()
    sequences = [build_stoic(), build_anxious(), build_romantic(), build_pragmatist()]

    states = []
    for seq in sequences:
        states.append({'seq': seq, 'idx': 0, 'accum': 0.0, 'done': False})

    start_time = time.time()
    intro_duration = 2.5
    running = True

    while running:
        dt = clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                if event.key == pygame.K_r:
                    for c in canvases:
                        c.fill(CANVAS_BG)
                    sequences = [build_stoic(), build_anxious(), build_romantic(), build_pragmatist()]
                    states = []
                    for seq in sequences:
                        states.append({'seq': seq, 'idx': 0, 'accum': 0.0, 'done': False})
                    start_time = time.time()

        elapsed = time.time() - start_time

        # Update drawings
        if elapsed > intro_duration:
            for i, st in enumerate(states):
                if st['done']:
                    continue
                st['accum'] += dt
                while st['idx'] < len(st['seq']) and not st['done']:
                    delay, func, args = st['seq'][st['idx']]
                    if st['accum'] >= delay:
                        st['accum'] -= delay
                        func(canvases[i], *args)
                        st['idx'] += 1
                    else:
                        break
                if st['idx'] >= len(st['seq']):
                    st['done'] = True

        # ── Render ──
        screen.fill(BG_COLOR)

        if elapsed < intro_duration:
            t = min(1.0, elapsed / intro_duration)
            alpha = int(255 * t)
            title_surf = font_prompt.render("draw a person in the rain", True, (alpha, alpha, alpha))
            screen.blit(title_surf, (WIDTH//2 - title_surf.get_width()//2, HEIGHT//2 - 20))
        else:
            # Prompt at top
            prompt_surf = font_prompt.render("draw a person in the rain", True, (210, 215, 220))
            screen.blit(prompt_surf, (WIDTH//2 - prompt_surf.get_width()//2, 30))

            # Canvases
            for i, (cx, cy) in enumerate(CANVAS_POSITIONS):
                pygame.draw.rect(screen, (12, 12, 16), (cx + 3, cy + 3, CANVAS_W, CANVAS_H))
                screen.blit(canvases[i], (cx, cy))
                pygame.draw.rect(screen, (40, 42, 50), (cx - 1, cy - 1, CANVAS_W + 2, CANVAS_H + 2), 1)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
