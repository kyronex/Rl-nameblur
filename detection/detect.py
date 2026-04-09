# detect.py — v8 dual detect
import cv2
import numpy as np
import logging
from config import cfg
from detection.tools import write_circles , write_rects , get_color
from detection.mask import compute_white_mask, compute_sobel_interior_unified, refine_and_merge ,saturation_variance_mask
from detection.boxes import process_channel
from core.box import Box

log = logging.getLogger("detect")

# ── Cache kernels par scale ──
_cache_by_scale = {}

def _build_params(scale):
    """Lit cfg à chaque appel — hot-reload natif."""

    # ── Couleurs ──
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
        "min_transition":      cfg.get("detect.refine.min_transition", 0.04),
        "min_proj_score":      cfg.get("detect.refine.min_proj_score", 0.10),
        # ── Thresh ──
        "var_norm":       cfg.get("detect.refine.thresh.var_norm",  1000.0),
        "coherent_delta":       cfg.get("detect.refine.thresh.coherent_delta",  33),
        "min_bg_score":       cfg.get("detect.refine.thresh.min_bg_score",  10),
        "hue_bin_size":       cfg.get("detect.refine.thresh.hue_bin_size",  10),
        "max_hue_bins":       cfg.get("detect.refine.thresh.max_hue_bins",  3),
        "min_hue_score":       cfg.get("detect.refine.thresh.min_hue_score",  0.1),
        # ── Géométrie ──
        "min_area":   max(round(cfg.get("detect.geometry.min_area",   800)   / scale**2), 1),
        "max_area":   max(round(cfg.get("detect.geometry.max_area",   20000) / scale**2), 1),
        "min_width":  max(round(cfg.get("detect.geometry.min_width",  80)    / scale), 1),
        "min_height": max(round(cfg.get("detect.geometry.min_height", 16)    / scale), 1),
        "min_ratio":  cfg.get("detect.geometry.min_ratio",  1.5),
        "max_ratio":  cfg.get("detect.geometry.max_ratio",  18.0),
        "min_fill":   cfg.get("detect.geometry.min_fill",   0.15),
        "max_fill":   cfg.get("detect.geometry.max_fill",   0.95),
        # ── Horizontal Bands ──
        "bands_min_fill":   cfg.get("detect.horizontal_bands.bands_min_fill",   0.15),
        "bands_gap_fill":   cfg.get("detect.horizontal_bands.bands_gap_fill",   0.08),
        "bands_max_bands":   cfg.get("detect.horizontal_bands.bands_max_bands",   2),
        # ── Horizontal Alignment ──
        "align_max_y_std_ratio":   cfg.get("detect.horizontal_alignment.align_max_y_std_ratio",  0.20),
        "align_min_cols":   cfg.get("detect.horizontal_alignment.align_min_cols",   3),
        "align_min_col_fill":   cfg.get("detect.horizontal_alignment.align_min_col_fill",  0.15),
        # ── Gradient Bands ──
        "n_zones":   cfg.get("detect.gradient.n_zones",  4),
        "min_drop_ratio":   cfg.get("detect.gradient.min_drop_ratio",  0.20),
        "min_zone_width":   cfg.get("detect.gradient.min_zone_width",   8),
    }

    # ── Arrays numpy couleur ──
    colors = {
        "wc_low":      np.array(wc_low),
        "wc_high":     np.array(wc_high),
        "we_low":      np.array(we_low),
        "we_high":     np.array(we_high),
    }

    return colors, kernels, params, letter_connect_iter

def _run_pipeline(frame, scale):
    colors, kernels, params, letter_connect_iter = _build_params(scale)

    # ══════════════════════════════════════════════════
    #  ÉTAPE 1 — Préparation de l'image
    # ══════════════════════════════════════════════════

    # ── 1. Resize ──
    h_orig, w_orig = frame.shape[:2]
    small = cv2.resize(frame,(int(w_orig / scale), int(h_orig / scale)),interpolation=cv2.INTER_LINEAR)
    h_small, w_small = small.shape[:2]

    # ── Grayscale ──
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    sat_mask = saturation_variance_mask(small, scale)

    # ── 2. Masque blanc → nettoyage ──
    mask_white, white_clean = compute_white_mask(gray, kernels, letter_connect_iter)

    # ── 3. Combinaison : blanc ET dans une zone à forte variance de saturation ──
    combined = cv2.bitwise_and(white_clean, sat_mask)

    # ── 4. Sobel interiors ──
    interior = compute_sobel_interior_unified(gray, combined, kernels)

    # ── 5. Refine and merge ──
    closed = refine_and_merge(combined, interior, kernels)

    # ── 6. Contours + filtre géométrique (ratio, area, fill) ──
    log.debug("START")
    candidates = process_channel(closed,small, mask_white, h_small, params, kernels )
    log.debug("END")

    if not candidates:
        return []

    # ── 7. Remap vers résolution d'entrée ──
    plates = [
        Box(int(b.x * scale), int(b.y * scale), int(b.w * scale), int(b.h * scale), scores=dict(b.scores))
        for b in candidates
    ]

    return plates


# ═══════════════════════════════════════════════════════
#  API PUBLIQUE
# ═══════════════════════════════════════════════════════

def detect_plates(frame):
    """Slow detect — full frame."""
    scale = cfg.get("detect.slow.scale", 2.0)
    return _run_pipeline(frame, scale)

# ═══════════════════════════════════════════════════════
#  FAST TRACKING — léger, pour ROI connues
# ═══════════════════════════════════════════════════════

def ncc_match(roi_gray, template_gray, threshold=0.5):
    """
    Template matching NCC sur un crop ROI.
    Retourne (score, (dx, dy)) ou (0.0, None) si échec.
    """
    if roi_gray is None or template_gray is None:
        return 0.0, None
    if (template_gray.shape[0] > roi_gray.shape[0] or
        template_gray.shape[1] > roi_gray.shape[1]):
        return 0.0, None

    result = cv2.matchTemplate(roi_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < threshold:
        return round(max_val, 3), None

    return round(max_val, 3), max_loc
