# detect_stats.py
"""
Compteurs de performance pour le pipeline de détection.
Tous les timings sont en millisecondes (cumulés entre deux reset).
"""

_stats = {
    # ── appels ──
    "total_calls":      0,

    # ── timings pipeline ──
    "total_ms":         0.0,
    "resize_ms":        0.0,
    "hsv_ms":           0.0,

    # ── timings masques ──
    "mask_orange_ms":   0.0,
    "mask_blue_ms":     0.0,
    "mask_white_ms":    0.0,        # white_core + white_ext + dilate + AND

    # ── timings combine ──
    "combine_ms":       0.0,        # orange | blue
    "combine_wd_ms":    0.0,        # AND avec white dilaté

    # ── timings morpho + contours ──
    "channel_ms":       0.0,        # process_channel total
    "pre_open_ms":         0.0,
    "close_ms":         0.0,
    "open_ms":          0.0,
    "contour_ms":       0.0,        # findContours seul
    "filter_loop_ms":   0.0,        # boucle filtre géométrique

    # ── compteurs détection ──
    "contours_raw":     0,
    "plates_found":     0,

    # ── compteurs rejections ──
    "rej_width":        0,
    "rej_height":       0,
    "rej_ratio":        0,
    "rej_area":         0,
    "rej_fill":         0,
}


def get_stats():
    """Retourne un dict de moyennes par appel."""
    n = max(_stats["total_calls"], 1)
    return {
        "total_calls":          _stats["total_calls"],

        # ── timings pipeline ──
        "total_avg_ms":         round(_stats["total_ms"]         / n, 2),
        "resize_avg_ms":        round(_stats["resize_ms"]        / n, 2),
        "hsv_avg_ms":           round(_stats["hsv_ms"]           / n, 2),

        # ── timings masques ──
        "mask_orange_avg_ms":   round(_stats["mask_orange_ms"]   / n, 2),
        "mask_blue_avg_ms":     round(_stats["mask_blue_ms"]     / n, 2),
        "mask_white_avg_ms":    round(_stats["mask_white_ms"]    / n, 2),

        # ── timings combine ──
        "combine_avg_ms":       round(_stats["combine_ms"]       / n, 2),
        "combine_wd_avg_ms":    round(_stats["combine_wd_ms"]    / n, 2),

        # ── timings morpho + contours ──
        "channel_avg_ms":       round(_stats["channel_ms"]       / n, 2),
        "pre_open_avg_ms":         round(_stats["pre_open_ms"]         / n, 2),
        "close_avg_ms":         round(_stats["close_ms"]         / n, 2),
        "open_avg_ms":          round(_stats["open_ms"]          / n, 2),
        "contour_avg_ms":       round(_stats["contour_ms"]       / n, 2),
        "filter_loop_avg_ms":   round(_stats["filter_loop_ms"]   / n, 2),

        # ── compteurs détection ──
        "contours_raw_avg":     round(_stats["contours_raw"]     / n, 1),
        "plates_found_avg":     round(_stats["plates_found"]     / n, 1),

        # ── compteurs rejections ──
        "rej_width_avg":        round(_stats["rej_width"]        / n, 1),
        "rej_height_avg":       round(_stats["rej_height"]       / n, 1),
        "rej_ratio_avg":        round(_stats["rej_ratio"]        / n, 1),
        "rej_area_avg":         round(_stats["rej_area"]         / n, 1),
        "rej_fill_avg":         round(_stats["rej_fill"]         / n, 1),
    }


def reset_stats():
    """Remet tous les compteurs à zéro."""
    for k in _stats:
        _stats[k] = 0 if isinstance(_stats[k], int) else 0.0
