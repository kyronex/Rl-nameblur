# detect.py — v8 dual detect
import cv2
import numpy as np
import time
import logging
from config import cfg
#from detect_stats import _stats, get_stats, reset_stats
from detect_stats import flush_local, make_local, get_stats, reset_stats
from detect_tools import write_circles , write_rects , get_color
from detect_tools_mask import compute_white_mask, compute_sobel_interiors, refine_and_merge ,saturation_variance_mask,detect_ball_zones
from detect_tools_boxes import process_channel
from box import Box

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
        "wc_low":      np.array(wc_low),
        "wc_high":     np.array(wc_high),
        "we_low":      np.array(we_low),
        "we_high":     np.array(we_high),
    }

    return colors, kernels, params, letter_connect_iter

def _run_pipeline(frame, scale):
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

    t0 = time.perf_counter()
    sat_mask = saturation_variance_mask(small, scale)
    local["sat_mask_ms"] = (time.perf_counter() - t0) * 1000

    # ── 2. Masque blanc → nettoyage ──
    t0 = time.perf_counter()
    mask_white, white_clean = compute_white_mask(gray, kernels, letter_connect_iter)
    local["compute_white_mask_ms"] += (time.perf_counter() - t0) * 1000

    # ── 3. Combinaison : blanc ET dans une zone à forte variance de saturation ──
    combined = cv2.bitwise_and(white_clean, sat_mask)

    # ── 3. Sobel → intérieurs ──
    t0 = time.perf_counter()
    interior_v1, interior_v2  = compute_sobel_interiors(gray, combined, kernels)
    local["compute_sobel_interiors_ms"] += (time.perf_counter() - t0) * 1000

    # ── 4. Raffinage → fusion → close ──
    t0 = time.perf_counter()
    closed = refine_and_merge(combined, interior_v1, interior_v2, kernels)
    local["refine_and_merge_ms"] += (time.perf_counter() - t0) * 1000

    # ── 5. Contours + filtre géométrique (ratio, area, fill) ──
    log.debug("START")
    candidates = process_channel(closed,small, mask_white, h_small, params, kernels, local )
    log.debug("END")

    if not candidates:
        local["total_ms"] += (time.perf_counter() - t_start) * 1000
        flush_local(local)
        return []

    local["filter_uniform_ms"] = (time.perf_counter() - t0) * 1000
    local["plates_found"] = len(candidates)

    # ── 8. Remap vers résolution d'entrée ──
    plates = [
        Box(int(b.x * scale), int(b.y * scale), int(b.w * scale), int(b.h * scale), scores=dict(b.scores))
        for b in candidates
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

# ── track_roi_fast not used ──
def track_roi_fast(roi):
    if roi is None or roi.size == 0:
        return []

    h_roi, w_roi = roi.shape[:2]
    if h_roi < 5 or w_roi < 10:
        return []

    fast_scale = cfg.get("detect.fast.scale", 2.0)

    if fast_scale > 1.0:
        new_w = max(int(w_roi / fast_scale), 10)
        new_h = max(int(h_roi / fast_scale), 5)
        roi_scaled = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        roi_scaled = roi
        fast_scale = 1.0

    h_s, w_s = roi_scaled.shape[:2]

    # ── 1. Grayscale ──
    if len(roi_scaled.shape) == 3:
        gray = cv2.cvtColor(roi_scaled, cv2.COLOR_RGB2GRAY)
    else:
        gray = roi_scaled

    # ── 2. Seuil blanc ──
    #    On utilise directement le seuil sur la luminance (canal V en HSV ≈ max(R,G,B))
    #    Plus rapide que cvtColor HSV + inRange pour un simple seuil de luminosité
    wc_low_v = cfg.get("detect.hsv.white_core.lower", [0, 0, 200])[2]   # valeur V min
    wc_high_s = cfg.get("detect.hsv.white_core.upper", [230, 35, 255])[1]  # saturation max

    # Seuil luminance : pixels très blancs
    _, white_mask = cv2.threshold(gray, wc_low_v, 255, cv2.THRESH_BINARY)

    # Filtrer les pixels trop saturés (pas blanc) — besoin du HSV pour ça
    hsv = cv2.cvtColor(roi_scaled, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    _, sat_mask = cv2.threshold(sat, wc_high_s, 255, cv2.THRESH_BINARY_INV)

    # Blanc = lumineux ET peu saturé
    white_clean = cv2.bitwise_and(white_mask, sat_mask)

    # ── 3. Dilatation pour connecter les lettres ──
    colors, kernels, params, letter_connect_iter = _build_params(fast_scale)
    dilated = cv2.dilate(white_clean, kernels["letter_connect"], iterations=letter_connect_iter)

    # ── 4. Contours ──
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    # ── 5. Filtrage géométrique minimal ──
    min_w = max(w_s * 0.15, 15)   # au moins 15% de la largeur ROI
    min_h = max(h_s * 0.15, 5)
    min_ratio = 1.2                   # doit être plus large que haut
    min_fill = 0.08                   # au moins 8% de remplissage blanc

    results = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w < min_w or h < min_h:
            continue
        ratio = w / max(h, 1)
        if ratio < min_ratio:
            continue

        # Remplissage blanc dans le rect
        roi_white = white_clean[y:y+h, x:x+w]
        fill = cv2.countNonZero(roi_white) / max(w * h, 1)
        if fill < min_fill:
            continue

        results.append((
            int(x * fast_scale),
            int(y * fast_scale),
            int(w * fast_scale),
            int(h * fast_scale),
        ))

    return results
