# detect.py — v5 : round() pour cohérence SCALE variable

import cv2
import numpy as np
import time
from config import cfg
from detect_stats import _stats, get_stats, reset_stats
from detect_tools import  process_channel , process_channel_v3

# ── PARAMÈTRES ──
SCALE = cfg.get("detect.scale", 2.0)

# ── FILTRES COULEUR ──
ORANGE_LOW  = np.array(cfg.get("detect.hsv.orange.lower", [8,  140, 170]))
ORANGE_HIGH = np.array(cfg.get("detect.hsv.orange.upper", [22, 255, 255]))
BLUE_LOW    = np.array(cfg.get("detect.hsv.blue.lower",   [100, 130, 150]))
BLUE_HIGH   = np.array(cfg.get("detect.hsv.blue.upper",   [125, 255, 255]))
WHITE_CORE_LOW   = np.array(cfg.get("detect.hsv.white_core.lower",  [0,   0,  200]))
WHITE_CORE_HIGH  = np.array(cfg.get("detect.hsv.white_core.upper",  [230, 30, 255]))
WHITE_EXT_LOW   = np.array(cfg.get("detect.hsv.white_ext.lower",  [0,   0,  200]))
WHITE_EXT_HIGH  = np.array(cfg.get("detect.hsv.white_ext.upper",  [230, 30, 255]))

# ── MORPHOLOGIE ──
CLOSE_H_WIDTH  = max(round(cfg.get("detect.morpho.close_h_width",  25) / SCALE), 3)
CLOSE_H2_WIDTH = max(round(cfg.get("detect.morpho.close_h2_width", 40) / SCALE), 3)
CLOSE_V_HEIGHT = max(round(cfg.get("detect.morpho.close_v_height",  4) / SCALE), 1)

KERNEL_CLOSE_H      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_H_WIDTH, 1))
KERNEL_CLOSE_H2     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_H2_WIDTH, 1))
KERNEL_CLOSE_V      = cv2.getStructuringElement(cv2.MORPH_RECT,    (1, CLOSE_V_HEIGHT))
KERNEL_WHITE_DILATE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

OPEN_NOISE_W      = max(round(cfg.get("detect.morpho.open_noise_w", 24) / SCALE), 1)
OPEN_NOISE_H      = max(round(cfg.get("detect.morpho.open_noise_h",  4) / SCALE), 1)
KERNEL_OPEN_NOISE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_NOISE_W, OPEN_NOISE_H))

ERODE_KERNEL_W   = max(round(cfg.get("detect.erode.kernel_w", 6) / SCALE), 1)
ERODE_KERNEL_H   = max(round(cfg.get("detect.erode.kernel_h", 3) / SCALE), 1)
ERODE_ITERATIONS = cfg.get("detect.erode.iterations", 2)          # entier pur, pas de division
KERNEL_ERODE     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERODE_KERNEL_W, ERODE_KERNEL_H))

# ── FILTRES FORME ──
MIN_AREA   = max(round(cfg.get("detect.filters.min_area",   800) / (SCALE * SCALE)), 1)
MIN_WIDTH  = max(round(cfg.get("detect.filters.min_width",   50) / SCALE), 1)
MAX_WIDTH  = max(round(cfg.get("detect.filters.max_width",  800) / SCALE), 1)
MIN_HEIGHT = max(round(cfg.get("detect.filters.min_height",  10) / SCALE), 1)
MAX_HEIGHT = max(round(cfg.get("detect.filters.max_height", 100) / SCALE), 1)
MIN_RATIO  = cfg.get("detect.filters.min_ratio", 1.5)             # ratio pur, pas de division
MAX_RATIO  = cfg.get("detect.filters.max_ratio", 15.0)            # idem
MIN_FILL   = cfg.get("detect.filters.min_fill",  0.35)            # idem

# ── SPLIT ──
SPLIT_MIN_WIDTH  = max(round(cfg.get("detect.split.min_width",  380) / SCALE), 1)
SPLIT_MIN_HEIGHT = max(round(cfg.get("detect.split.min_height",  60) / SCALE), 1)

# ── Contexte injecté dans les fonctions tools ──
_params = {
    "MIN_AREA":         MIN_AREA,
    "MIN_WIDTH":        MIN_WIDTH,
    "MAX_WIDTH":        MAX_WIDTH,
    "MIN_HEIGHT":       MIN_HEIGHT,
    "MAX_HEIGHT":       MAX_HEIGHT,
    "MIN_RATIO":        MIN_RATIO,
    "MAX_RATIO":        MAX_RATIO,
    "MIN_FILL":         MIN_FILL,
    "ERODE_ITERATIONS": ERODE_ITERATIONS,
    "SPLIT_MIN_WIDTH":  SPLIT_MIN_WIDTH,
    "SPLIT_MIN_HEIGHT": SPLIT_MIN_HEIGHT,
}

_kernels = {
    "close_h":      KERNEL_CLOSE_H,
    "close_h2":     KERNEL_CLOSE_H2,
    "close_v":      KERNEL_CLOSE_V,
    "open_noise":   KERNEL_OPEN_NOISE,
    "white_dilate": KERNEL_WHITE_DILATE,
    "erode":        KERNEL_ERODE,
}

# ═══════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════

def detect_plates_v2(frame):
    """Détecte les cartouches de noms dans une frame."""
    t_start = time.perf_counter()
    plates = []

    # ── 1. Resize ──
    t0 = time.perf_counter()
    h_orig, w_orig = frame.shape[:2]
    small = cv2.resize(frame,(int(w_orig / SCALE), int(h_orig / SCALE)),interpolation=cv2.INTER_LINEAR)
    _stats["resize_ms"] += (time.perf_counter() - t0) * 1000

    # ── 2. HSV ──
    t0 = time.perf_counter()
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    _stats["hsv_ms"] += (time.perf_counter() - t0) * 1000

    # ── 3. Masques couleur ──
    t0 = time.perf_counter()
    mask_orange = cv2.inRange(hsv, ORANGE_LOW, ORANGE_HIGH)
    mask_blue   = cv2.inRange(hsv, BLUE_LOW,   BLUE_HIGH)
    mask_white  = cv2.inRange(hsv, WHITE_CORE_LOW,  WHITE_CORE_HIGH)
    _stats["masks_ms"] += (time.perf_counter() - t0) * 1000

    # ── 4. Dilatation blanc ──
    t0 = time.perf_counter()
    white_dilated = cv2.dilate(mask_white, _kernels["white_dilate"], iterations=1)
    _stats["white_ms"] += (time.perf_counter() - t0) * 1000

    # ── 5. Traitement ORANGE ──
    t0 = time.perf_counter()
    orange_plates = process_channel(mask_orange, white_dilated, _kernels, _params, _stats)
    _stats["orange_ms"] += (time.perf_counter() - t0) * 1000

    # ── 6. Traitement BLEU ──
    t0 = time.perf_counter()
    blue_plates = process_channel(mask_blue, white_dilated, _kernels, _params, _stats)
    _stats["blue_ms"] += (time.perf_counter() - t0) * 1000

    # ── 7. Remap vers résolution originale ──
    for (px, py, pw, ph) in orange_plates + blue_plates:
        plates.append((
            int(px * SCALE),
            int(py * SCALE),
            int(pw * SCALE),
            int(ph * SCALE),
        ))

    _stats["plates_found"] += len(plates)
    _stats["total_ms"]     += (time.perf_counter() - t_start) * 1000
    _stats["total_calls"]  += 1

    return plates

