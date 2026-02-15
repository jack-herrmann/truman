#!/usr/bin/env python3
"""Launcher for the "draw yourself" demo (personalities draw a self-portrait).

This file is kept for backward compatibility. The actual demo lives in
draw_yourself.py. Run either:

  python draw_yourself.py --save out.png
  python person_in_rain.py --save out.png
"""
from draw_yourself import main

if __name__ == "__main__":
    main()
