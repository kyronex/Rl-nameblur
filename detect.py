# detect.py — v7
import cv2
import numpy as np
import time
import logging
from config import cfg
from detect_stats import _stats , get_stats, reset_stats
from detect_tools import process_channel

log = logging.getLogger("detect")

# ── Cache kernels ──
_cache = {}

def _build_params():
    """Lit cfg à chaque appel — hot-reload natif."""
    scale = cfg.get("detect.scale", 2.0)

    # ── Couleurs ──
    orange_low  = tuple(cfg.get("detect.hsv.orange.lower", [8,  140, 170]))
    orange_high = tuple(cfg.get("detect.hsv.orange.upper", [22, 255, 255]))
    blue_low    = tuple(cfg.get("detect.hsv.blue.lower",   [100, 130, 150]))
    blue_high   = tuple(cfg.get("detect.hsv.blue.upper",   [125, 255, 255]))
    wc_low      = tuple(cfg.get("detect.hsv.white_core.lower", [0,   0,  200]))
    wc_high     = tuple(cfg.get("detect.hsv.white_core.upper", [230, 30, 255]))
    we_low      = tuple(cfg.get("detect.hsv.white_ext.lower",  [0,   0,  200]))
    we_high     = tuple(cfg.get("detect.hsv.white_ext.upper",  [230, 30, 255]))

    # ── Morpho ──
    wd_w    = max(round(cfg.get("detect.morpho.white_dilate.width",  28) / scale), 5)
    wd_h    = max(round(cfg.get("detect.morpho.white_dilate.height",  9) / scale), 3)
    wd_iter = cfg.get("detect.morpho.white_dilate.iterations", 1)

    pre_sz  = max(round(cfg.get("detect.morpho.pre_open.size", 4) / scale), 1)

    cl_w    = max(round(cfg.get("detect.morpho.close.width",  25) / scale), 3)
    cl_h    = max(round(cfg.get("detect.morpho.close.height",  2) / scale), 1)

    on_w    = max(round(cfg.get("detect.morpho.open_noise.width",  24) / scale), 1)
    on_h    = max(round(cfg.get("detect.morpho.open_noise.height",  4) / scale), 1)

    # ── Clé de cache pour les kernels ──
    kernel_key = (scale, wd_w, wd_h, pre_sz, cl_w, cl_h, on_w, on_h)

    if kernel_key != _cache.get("kernel_key"):
        log.info(f"Kernels recalculés — scale={scale}, key={kernel_key}")
        _cache["kernel_key"] = kernel_key
        _cache["kernels"] = {
            "white_dilate": cv2.getStructuringElement(cv2.MORPH_RECT,    (wd_w,   wd_h)),
            "close_h":      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cl_w,   cl_h)),
            "open_noise":   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (on_w,   on_h)),
            "pre_open_noise": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pre_sz, pre_sz)),
        }

    # ── Géométrie ──
    params = {
        "min_area":   max(round(cfg.get("detect.geometry.min_area",   800)   / scale**2), 1),
        "max_area":   max(round(cfg.get("detect.geometry.max_area",   20000) / scale**2), 1),
        "min_width":  max(round(cfg.get("detect.geometry.min_width",  80)    / scale), 1),
        "min_height": max(round(cfg.get("detect.geometry.min_height", 16)    / scale), 1),
        "min_ratio":  cfg.get("detect.geometry.min_ratio",  3.5),
        "max_ratio":  cfg.get("detect.geometry.max_ratio",  18.0),
        "min_fill":   cfg.get("detect.geometry.min_fill",   0.15),
        "max_fill":   cfg.get("detect.geometry.max_fill",   0.85),
    }

    # ── Arrays numpy couleur ──
    colors = {
        "orange_low":  np.array(orange_low),
        "orange_high": np.array(orange_high),
        "blue_low":    np.array(blue_low),
        "blue_high":   np.array(blue_high),
        "wc_low":      np.array(wc_low),
        "wc_high":     np.array(wc_high),
        "we_low":      np.array(we_low),
        "we_high":     np.array(we_high),
    }

    return scale, colors, _cache["kernels"], params, wd_iter

# ═══════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════

def detect_plates(frame):
    """
    Détecte les cartouches via fusion orange|blue AND white_dilated.
    """
    t_start = time.perf_counter()
    plates = []

    scale, colors, kernels, params, wd_iter = _build_params()
    # ── 1. Resize ──
    t0 = time.perf_counter()
    h_orig, w_orig = frame.shape[:2]
    small = cv2.resize(frame,(int(w_orig / scale), int(h_orig / scale)),interpolation=cv2.INTER_LINEAR)
    _stats["resize_ms"] += (time.perf_counter() - t0) * 1000

    # ── 2. HSV ──
    t0 = time.perf_counter()
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    _stats["hsv_ms"] += (time.perf_counter() - t0) * 1000

    # ── 3. Masques couleur ──
    t0 = time.perf_counter()
    mask_orange = cv2.inRange(hsv,colors["orange_low"], colors["orange_high"])
    _stats["mask_orange_ms"] += (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    mask_blue = cv2.inRange(hsv, colors["blue_low"], colors["blue_high"])
    _stats["mask_blue_ms"] += (time.perf_counter() - t0) * 1000

    # ── 4. Fusion orange + blue ──
    t0 = time.perf_counter()
    mask_combined = cv2.bitwise_or(mask_orange, mask_blue)
    _stats["combine_ms"] += (time.perf_counter() - t0) * 1000

    # ── 5. Blanc : core dilaté + ext qualifié ──
    t0 = time.perf_counter()
    mask_white_core = cv2.inRange(hsv, colors["wc_low"], colors["wc_high"])
    mask_white_ext  = cv2.inRange(hsv, colors["we_low"], colors["we_high"])

    white_core_dilated = cv2.dilate(mask_white_core,kernels["white_dilate"],iterations=wd_iter)

    mask_white_ext_qualified = cv2.bitwise_and(mask_white_ext, white_core_dilated)
    _stats["mask_white_ms"] += (time.perf_counter() - t0) * 1000

    # ── 6. AND couleur + blanc dilaté, puis OR bords qualifiés ──

    t0 = time.perf_counter()                          # ← FIX : t0 reset ici
    mask_and_color = cv2.bitwise_and(mask_combined, white_core_dilated)
    mask_and = cv2.bitwise_or(mask_and_color, mask_white_ext_qualified)
    _stats["combine_wd_ms"] += (time.perf_counter() - t0) * 1000

    # ── DEBUG pixels ──
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"pixels orange        : {cv2.countNonZero(mask_orange)}")
        log.debug(f"pixels blue          : {cv2.countNonZero(mask_blue)}")
        log.debug(f"pixels white_core    : {cv2.countNonZero(mask_white_core)}")
        log.debug(f"pixels white_dilated : {cv2.countNonZero(white_core_dilated)}")
        log.debug(f"pixels mask_and final: {cv2.countNonZero(mask_and)}")

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
    raw_plates = process_channel(mask_and, kernels, params, _stats)
    _stats["channel_ms"] += (time.perf_counter() - t0) * 1000

    # ── 8. Remap vers résolution originale ──
    for (px, py, pw, ph) in raw_plates:
        plates.append((
            int(px * scale),
            int(py * scale),
            int(pw * scale),
            int(ph * scale),
        ))

    _stats["plates_found"] += len(plates)
    _stats["total_ms"]     += (time.perf_counter() - t_start) * 1000
    _stats["total_calls"]  += 1

    return plates
