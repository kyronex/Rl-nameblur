# detect.py — v8 dual detect
import cv2
import numpy as np
import time
import logging
from config import cfg
#from detect_stats import _stats, get_stats, reset_stats
from detect_stats import flush_local, make_local, get_stats, reset_stats
from detect_tools import process_channel, compute_white_mask, compute_sobel_interiors, refine_and_merge

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
    letter_connect_w     = max(round(cfg.get("detect.morpho.white_dilate.width", 15) / scale), 5)
    letter_connect_h     = max(round(cfg.get("detect.morpho.white_dilate.height", 10) / scale), 3)
    letter_connect_iter  = cfg.get("detect.morpho.white_dilate.iterations", 1)

    gap_fill_w     = max(round(cfg.get("detect.morpho.close.width", 25) / scale), 3)
    gap_fill_h     = max(round(cfg.get("detect.morpho.close.height", 2) / scale), 1)

    # Tailles hardcodées (à mettre en config si besoin de tuner)
    noise_filter_w, noise_filter_h        = 20, 4
    line_split_w, line_split_h            = 5, 6
    sobel_spread_w, sobel_spread_h        = 3, 1
    sobel_erode_w, sobel_erode_h          = 5, 2
    fragment_rejoin_w, fragment_rejoin_h  = 4, 1
    final_split_w, final_split_h          = 1, 2
    roi_connected_w, roi_connected_h      = 3, 1

    kernel_key = (
        scale,
        letter_connect_w, letter_connect_h,
        gap_fill_w, gap_fill_h,
        noise_filter_w, noise_filter_h,
        line_split_w, line_split_h,
        sobel_spread_w, sobel_spread_h,
        sobel_erode_w, sobel_erode_h,
        fragment_rejoin_w, fragment_rejoin_h,
        final_split_w, final_split_h,
        roi_connected_w, roi_connected_h,
    )

    # ── Création uniquement si config changée ──
    cached = _cache_by_scale.get(scale)
    if cached is None or cached["key"] != kernel_key:
        log.info(f"Kernels recalculés — scale={scale}")
        _cache_by_scale[scale] = {
            "key": kernel_key,
            "kernels": {
                # Étape 2b : connecter les lettres d'un pseudo
                "letter_connect":  cv2.getStructuringElement(cv2.MORPH_RECT, (letter_connect_w, letter_connect_h)),
                # Étape 3a : supprimer blobs non allongés horizontalement
                "noise_filter":    cv2.getStructuringElement(cv2.MORPH_RECT, (noise_filter_w, noise_filter_h)),
                # Étape 3b : séparer les cartouches empilées verticalement
                "line_split":      cv2.getStructuringElement(cv2.MORPH_RECT, (line_split_w, line_split_h)),
                # Étape 4d : étaler le gradient Sobel horizontalement
                "sobel_spread":    cv2.getStructuringElement(cv2.MORPH_RECT, (sobel_spread_w, sobel_spread_h)),
                # Étape 5a : isoler l'intérieur dense du Sobel
                "sobel_erode":     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sobel_erode_w, sobel_erode_h)),
                # Étape 5c : reconnecter les fragments après soustraction
                "fragment_rejoin": cv2.getStructuringElement(cv2.MORPH_RECT, (fragment_rejoin_w, fragment_rejoin_h)),
                # Étape 6b : re-séparer les blocs collés après fusion
                "final_split":     cv2.getStructuringElement(cv2.MORPH_RECT, (final_split_w, final_split_h)),
                # Étape 7a : combler les trous dans les cartouches
                "gap_fill":        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_fill_w, gap_fill_h)),
                # Étape 7b : connecter les pixel des lettre dans les cartouches
                "roi_connected":   cv2.getStructuringElement(cv2.MORPH_RECT, (roi_connected_w, roi_connected_h)),
            },
        }

    kernels = _cache_by_scale[scale]["kernels"]
    params = {
         # ── Morpho ──
        "max_gap_x":  max(int(cfg.get("detect.morpho.merge.max_gap_x", 30) / scale), 10),
        "max_gap_y":  max(int(cfg.get("detect.morpho.merge.max_gap_y", 10) / scale), 2),
        # ── Adjust ──
        "max_expand_x_ratio": cfg.get("detect.adjust.max_expand_x_ratio", 0.5),
        "max_expand_y_ratio": cfg.get("detect.adjust.max_expand_y_ratio", 0.3),
        "padding": cfg.get("detect.adjust.padding", 3),
        "adjust_min_blob_area":    cfg.get("detect.adjust.min_blob_area", 10) ,
        "expand_search_px":    cfg.get("detect.adjust.expand_search_px", 10) ,
        # ── Split ──
        "min_valley_width": cfg.get("detect.split.min_valley_width", 0.5),
        "max_valley_density": cfg.get("detect.split.max_valley_density", 0.3),
        "min_fragment_width": cfg.get("detect.split.min_fragment_width", 3),
        # ── Refine ──
        "refine_min_blob_area":    cfg.get("detect.refine.min_blob_area", 10) ,
        "min_text_fill":      cfg.get("detect.refine.min_text_fill", 0.3),
        "var_norm":       cfg.get("detect.refine.thresh.var_norm",  1000.0),
        "coherent_delta":       cfg.get("detect.refine.thresh.coherent_delta",  33),
        "min_bg_score":       cfg.get("detect.refine.thresh.min_bg_score",  10),
        # ── Géométrie ──
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

    return colors, kernels, params, letter_connect_iter

def _run_pipeline(frame, scale):
    """
    Pipeline White-First :
    1. Chercher le texte blanc (signal fort)
    2. Former des blobs (dilatation légère + morpho)
    3. Valider chaque blob par présence de couleur orange/bleu autour
    Retourne une liste de rects (x, y, w, h) en coordonnées frame d'entrée.
    """
    local = make_local()
    t_start = time.perf_counter()
    colors, kernels, params, letter_connect_iter = _build_params(scale)

    # ══════════════════════════════════════════════════
    #  ÉTAPE 1 — Préparation de l'image
    # ══════════════════════════════════════════════════

    # ── 1. Resize ──
    t0 = time.perf_counter()
    h_orig, w_orig = frame.shape[:2]
    small = cv2.resize(frame,(int(w_orig / scale), int(h_orig / scale)),interpolation=cv2.INTER_LINEAR)
    h_small, w_small = small.shape[:2]
    local["resize_ms"] += (time.perf_counter() - t0) * 1000

    # ── Grayscale ──
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    # ── 2. Masque blanc → nettoyage ──
    t0 = time.perf_counter()
    mask_white, white_clean = compute_white_mask(gray, kernels, letter_connect_iter)
    local["compute_white_mask_ms"] += (time.perf_counter() - t0) * 1000

    # ── 3. Sobel → intérieurs ──
    t0 = time.perf_counter()
    interior_v1, interior_v2 = compute_sobel_interiors(gray, white_clean, kernels)
    local["compute_sobel_interiors_ms"] += (time.perf_counter() - t0) * 1000

    # ── 4. Raffinage → fusion → close ──
    t0 = time.perf_counter()
    closed = refine_and_merge(white_clean, interior_v1, interior_v2, kernels)
    local["refine_and_merge_ms"] += (time.perf_counter() - t0) * 1000

    # ── HSV ──
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2Lab)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    # ── 5. Contours + filtre géométrique (ratio, area, fill) ──
    log.debug("START")
    candidates = process_channel(closed,small, mask_white, h_small, params, kernels, local)
    log.debug("END")

    if not candidates:
        local["total_ms"] += (time.perf_counter() - t_start) * 1000
        flush_local(local)
        return []

    local["filter_uniform_ms"] = (time.perf_counter() - t0) * 1000
    local["plates_found"] = len(candidates)

    # ── 8. Remap vers résolution d'entrée ──
    plates = [
        (int(px * scale), int(py * scale), int(pw * scale), int(ph * scale))
        for (px, py, pw, ph) in candidates
    ]
    local["total_ms"] += (time.perf_counter() - t_start) * 1000
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
