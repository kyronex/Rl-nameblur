# detect.py — v7
import cv2
import numpy as np
import time
import logging
from config import cfg
from detect_stats import _stats, get_stats, reset_stats
from detect_tools import process_channel

log = logging.getLogger("detect")

_log_level = cfg.get("debug.log_level", "WARNING")
log.setLevel(getattr(logging, _log_level))

# ── Handler console ──
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(name)s | %(message)s"))
    log.addHandler(_handler)

# ── PARAMÈTRES ──
SCALE = cfg.get("detect.scale", 2.0)

# ── FILTRES COULEUR ──
ORANGE_LOW  = np.array(cfg.get("detect.hsv.orange.lower", [8,  140, 170]))
ORANGE_HIGH = np.array(cfg.get("detect.hsv.orange.upper", [22, 255, 255]))

BLUE_LOW    = np.array(cfg.get("detect.hsv.blue.lower",   [100, 130, 150]))
BLUE_HIGH   = np.array(cfg.get("detect.hsv.blue.upper",   [125, 255, 255]))

WHITE_CORE_LOW  = np.array(cfg.get("detect.hsv.white_core.lower", [0,   0,  200]))
WHITE_CORE_HIGH = np.array(cfg.get("detect.hsv.white_core.upper", [230, 30, 255]))
WHITE_EXT_LOW   = np.array(cfg.get("detect.hsv.white_ext.lower",  [0,   0,  200]))
WHITE_EXT_HIGH  = np.array(cfg.get("detect.hsv.white_ext.upper",  [230, 30, 255]))

# ── MORPHO : white dilate ──
WHITE_DILATE_W    = max(round(cfg.get("detect.morpho.white_dilate.width",  28) / SCALE), 5)
WHITE_DILATE_H    = max(round(cfg.get("detect.morpho.white_dilate.height",  9) / SCALE), 3)
WHITE_DILATE_ITER = cfg.get("detect.morpho.white_dilate.iterations", 1)
KERNEL_WHITE_DILATE = cv2.getStructuringElement(cv2.MORPH_RECT, (WHITE_DILATE_W, WHITE_DILATE_H))

# ── MORPHO : close + open ──

PRE_OPEN_SIZE = max(round(cfg.get("detect.morpho.pre_open.size",4) / SCALE), 1)
KERNEL_PRE_OPEN  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (PRE_OPEN_SIZE, PRE_OPEN_SIZE))

CLOSE_H_WIDTH   = max(round(cfg.get("detect.morpho.close.width", 25) / SCALE), 3)
CLOSE_H_HEIGHT   = max(round(cfg.get("detect.morpho.close.height", 2) / SCALE), 1)
KERNEL_CLOSE_H  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_H_WIDTH, CLOSE_H_HEIGHT))

OPEN_NOISE_W    = max(round(cfg.get("detect.morpho.open_noise.width",  24) / SCALE), 1)
OPEN_NOISE_H    = max(round(cfg.get("detect.morpho.open_noise.height",  4) / SCALE), 1)
KERNEL_OPEN_NOISE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_NOISE_W, OPEN_NOISE_H))

# ── GÉOMÉTRIE (seuils scalés) ──
_params = {
    "min_area":   max(round(cfg.get("detect.geometry.min_area",   800)   / (SCALE**2)), 1),
    "max_area":   max(round(cfg.get("detect.geometry.max_area",   20000) / (SCALE**2)), 1),
    "min_width":  max(round(cfg.get("detect.geometry.min_width",  80)    / SCALE), 1),
    "min_height": max(round(cfg.get("detect.geometry.min_height", 16)    / SCALE), 1),
    "min_ratio":  cfg.get("detect.geometry.min_ratio",  3.5),
    "max_ratio":  cfg.get("detect.geometry.max_ratio",  18.0),
    "min_fill":   cfg.get("detect.geometry.min_fill",   0.15),
    "max_fill":   cfg.get("detect.geometry.max_fill",   0.85),
}

_kernels = {
    "close_h":      KERNEL_CLOSE_H,
    "open_noise":   KERNEL_OPEN_NOISE,
    "pre_open_noise":   KERNEL_PRE_OPEN,
}

# ── DEBUG ──
LOG_LEVEL = cfg.get("debug.log_level", "INFO")

# ═══════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════
log.debug(f"KERNEL_CLOSE_H  : {KERNEL_CLOSE_H.shape}  (width={CLOSE_H_WIDTH})")
log.debug(f"KERNEL_OPEN     : {KERNEL_OPEN_NOISE.shape}  (w={OPEN_NOISE_W}, h={OPEN_NOISE_H})")
log.debug(f"KERNEL_WHITE_D  : {KERNEL_WHITE_DILATE.shape}  (w={WHITE_DILATE_W}, h={WHITE_DILATE_H})")
log.debug(f"SCALE={SCALE}  params={_params}")
def detect_plates(frame):
    """
    Détecte les cartouches via fusion orange|blue AND white_dilated.
    """

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
    _stats["mask_orange_ms"] += (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    mask_blue = cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)
    _stats["mask_blue_ms"] += (time.perf_counter() - t0) * 1000

    # ── 4. Fusion orange + blue ──
    t0 = time.perf_counter()
    mask_combined = cv2.bitwise_or(mask_orange, mask_blue)
    _stats["combine_ms"] += (time.perf_counter() - t0) * 1000

    # ── 5. Blanc : core dilaté + ext qualifié ──
    t0 = time.perf_counter()
    mask_white_core = cv2.inRange(hsv, WHITE_CORE_LOW, WHITE_CORE_HIGH)
    mask_white_ext  = cv2.inRange(hsv, WHITE_EXT_LOW,  WHITE_EXT_HIGH)

    white_core_dilated = cv2.dilate(mask_white_core,KERNEL_WHITE_DILATE,iterations=WHITE_DILATE_ITER)

    mask_white_ext_qualified = cv2.bitwise_and(mask_white_ext, white_core_dilated)
    _stats["mask_white_ms"] += (time.perf_counter() - t0) * 1000

    # ── 6. AND couleur + blanc dilaté, puis OR bords qualifiés ──

    t0 = time.perf_counter()                          # ← FIX : t0 reset ici
    mask_and_color = cv2.bitwise_and(mask_combined, white_core_dilated)
    mask_and = cv2.bitwise_or(mask_and_color, mask_white_ext_qualified)
    _stats["combine_wd_ms"] += (time.perf_counter() - t0) * 1000

    # ── DEBUG pixels par canal ──
    log.debug(f"pixels orange          : {cv2.countNonZero(mask_orange)}")
    log.debug(f"pixels blue            : {cv2.countNonZero(mask_blue)}")
    log.debug(f"pixels white_core      : {cv2.countNonZero(mask_white_core)}")
    log.debug(f"pixels white_dilated   : {cv2.countNonZero(white_core_dilated)}")
    log.debug(f"pixels AND orange+wd   : {cv2.countNonZero(cv2.bitwise_and(mask_orange, white_core_dilated))}")
    log.debug(f"pixels AND blue+wd     : {cv2.countNonZero(cv2.bitwise_and(mask_blue, white_core_dilated))}")
    log.debug(f"pixels mask_and final  : {cv2.countNonZero(mask_and)}")

    """
    cv2.imshow("mask_white_core", mask_white_core)
    cv2.imshow("mask_white_ext", mask_white_ext)
    cv2.imshow("white_core_dilated", white_core_dilated)
    cv2.imshow("mask_white_ext_qualified", mask_white_ext_qualified)
    cv2.imshow("mask_combined", mask_combined)
    cv2.imshow("mask_and_color", mask_and_color)
    """

    # ── 7. Passe morpho + contours ──
    t0 = time.perf_counter()
    raw_plates = process_channel(mask_and, _kernels, _params, _stats)
    _stats["channel_ms"] += (time.perf_counter() - t0) * 1000

    # ── 8. Remap vers résolution originale ──
    for (px, py, pw, ph) in raw_plates:
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
