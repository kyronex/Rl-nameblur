# detect.py — v8 dual detect
import cv2
import numpy as np
import time
import logging
from config import cfg
#from detect_stats import _stats, get_stats, reset_stats
from detect_stats import flush_local, make_local, get_stats, reset_stats
from detect_tools import process_channel

log = logging.getLogger("detect")

# ── Cache kernels par scale ──
_cache_by_scale = {}


def _build_params(scale):                              # ← FIX: scale en paramètre
    """Lit cfg à chaque appel — hot-reload natif."""

    # ── Couleurs ──
    orange_low  = tuple(cfg.get("detect.hsv.orange.lower", [8, 140, 170]))
    orange_high = tuple(cfg.get("detect.hsv.orange.upper", [22, 255, 255]))
    blue_low    = tuple(cfg.get("detect.hsv.blue.lower",   [100, 130, 150]))
    blue_high   = tuple(cfg.get("detect.hsv.blue.upper",   [125, 255, 255]))
    wc_low      = tuple(cfg.get("detect.hsv.white_core.lower", [0, 0, 200]))
    wc_high     = tuple(cfg.get("detect.hsv.white_core.upper", [230, 30, 255]))
    we_low      = tuple(cfg.get("detect.hsv.white_ext.lower",  [0, 0, 200]))
    we_high     = tuple(cfg.get("detect.hsv.white_ext.upper",  [230, 30, 255]))

    # ── Morpho ──
    wd_w    = max(round(cfg.get("detect.morpho.white_dilate.width", 28) / scale), 5)
    wd_h    = max(round(cfg.get("detect.morpho.white_dilate.height", 9) / scale), 3)
    wd_iter = cfg.get("detect.morpho.white_dilate.iterations", 1)

    pre_sz  = max(round(cfg.get("detect.morpho.pre_open.size", 4) / scale), 1)

    cl_w    = max(round(cfg.get("detect.morpho.close.width", 25) / scale), 3)
    cl_h    = max(round(cfg.get("detect.morpho.close.height", 2) / scale), 1)

    on_w    = max(round(cfg.get("detect.morpho.open_noise.width", 24) / scale), 1)
    on_h    = max(round(cfg.get("detect.morpho.open_noise.height", 4) / scale), 1)

    # ── Cache kernels ──
    kernel_key = (scale, wd_w, wd_h, pre_sz, cl_w, cl_h, on_w, on_h)

    if scale not in _cache_by_scale or _cache_by_scale[scale].get("key") != kernel_key:
        log.info(f"Kernels recalculés — scale={scale}, key={kernel_key}")
        _cache_by_scale[scale] = {
            "key": kernel_key,
            "kernels": {
                "white_dilate":   cv2.getStructuringElement(cv2.MORPH_RECT,    (wd_w, wd_h)),
                "close_h":        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cl_w, cl_h)),
                "open_noise":     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (on_w, on_h)),
                "pre_open_noise": cv2.getStructuringElement(cv2.MORPH_RECT,    (pre_sz, pre_sz)),
            },
        }

    kernels = _cache_by_scale[scale]["kernels"]

    # ── Géométrie ──
    params = {
        "min_area":   max(round(cfg.get("detect.geometry.min_area",   800)   / scale**2), 1),
        "max_area":   max(round(cfg.get("detect.geometry.max_area",   20000) / scale**2), 1),
        "min_width":  max(round(cfg.get("detect.geometry.min_width",  80)    / scale), 1),
        "min_height": max(round(cfg.get("detect.geometry.min_height", 16)    / scale), 1),
        "min_ratio":  cfg.get("detect.geometry.min_ratio",  1.5),
        "max_ratio":  cfg.get("detect.geometry.max_ratio",  18.0),
        "min_fill":   cfg.get("detect.geometry.min_fill",   0.15),
        "max_fill":   cfg.get("detect.geometry.max_fill",   0.95),
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

    return colors, kernels, params, wd_iter


def _run_pipeline(frame, scale):
    """
    Pipeline de détection complet sur une frame (full ou ROI crop).
    Retourne une liste de rects (x, y, w, h) en coordonnées de la frame d'entrée.
    """
    local = make_local()
    t_start = time.perf_counter()
    colors, kernels, params, wd_iter = _build_params(scale)

    # ── 1. Resize ──
    t0 = time.perf_counter()
    h_orig, w_orig = frame.shape[:2]
    small = cv2.resize(frame,(int(w_orig / scale), int(h_orig / scale)),interpolation=cv2.INTER_LINEAR)
    local["resize_ms"] += (time.perf_counter() - t0) * 1000

    # ── 2. HSV ──
    t0 = time.perf_counter()
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    local["hsv_ms"] += (time.perf_counter() - t0) * 1000

    # ── 3. Masques couleur ──
    t0 = time.perf_counter()
    mask_orange = cv2.inRange(hsv, colors["orange_low"], colors["orange_high"])
    local["mask_orange_ms"] += (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    mask_blue = cv2.inRange(hsv, colors["blue_low"], colors["blue_high"])
    local["mask_blue_ms"] += (time.perf_counter() - t0) * 1000

    # ── 4. Combine orange | blue ──
    t0 = time.perf_counter()
    mask_combined = cv2.bitwise_or(mask_orange, mask_blue)
    local["combine_ms"] += (time.perf_counter() - t0) * 1000

    # ── 5. Masques blanc ──
    t0 = time.perf_counter()
    mask_white = cv2.inRange(hsv, colors["wc_low"], colors["wc_high"])
    white_dilated = cv2.dilate(mask_white, kernels["white_dilate"], iterations=wd_iter)
    local["mask_white_ms"] += (time.perf_counter() - t0) * 1000

    # ── 6. AND couleur + blanc dilaté ──
    t0 = time.perf_counter()
    mask_and = cv2.bitwise_and(mask_combined, white_dilated)
    local["combine_wd_ms"] += (time.perf_counter() - t0) * 1000

    # ── 7. Morpho + contours ──
    raw_plates = process_channel(mask_and, kernels, params, local)

    # ── 8. Remap vers résolution d'entrée ──
    plates = [
        (int(px * scale), int(py * scale), int(pw * scale), int(ph * scale))
        for (px, py, pw, ph) in raw_plates
    ]

    local["total_ms"]    += (time.perf_counter() - t_start) * 1000
    flush_local(local)

    return plates


# ═══════════════════════════════════════════════════════
#  API PUBLIQUE
# ═══════════════════════════════════════════════════════

def detect_plates(frame):
    """Slow detect — full frame."""
    scale = cfg.get("detect.slow.scale", 2.0)
    return _run_pipeline(frame, scale)


def detect_roi(roi):                                   # ← FIX: prend juste le crop
    """
    Fast detect — ROI crop
    """
    if roi.size == 0:
        return []

    scale = cfg.get("detect.fast.scale", 3.0)
    return _run_pipeline(roi, scale)