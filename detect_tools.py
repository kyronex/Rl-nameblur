# detect_tools.py
import cv2
import numpy as np
import time
import logging
from config import cfg

log = logging.getLogger("detect_tools")

# ── write_rects ──
def write_rects(image, rects, color , thickness=2):
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    """
    screen = small.copy()
    write_rects(screen, validated_b, get_color("magenta"),2)
    write_rects(screen, validated_b1, get_color("vert"),1)
    cv2.imshow("screen", screen)
    cv2.waitKey(0)
    """

    #lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    #hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# ── write_circles ──
def write_circles(image, circles, color, thickness=2):
    for (cx, cy, r) in circles:
        cv2.circle(image, (cx, cy), r, color, thickness)

# ── get_color ──
def get_color(name: str, default: tuple = (0, 0, 0)) -> tuple:
    value = cfg.get(f"debug.colors_ttl.{name}")
    return tuple(value) if value is not None else default
