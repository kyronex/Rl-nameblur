# blur.py
import cv2
import numpy as np
from config import cfg

# ─────────────────────────────────────────
# PIXELISATION
# ─────────────────────────────────────────

def _pixelate_full(frame, pixel_size):
    """
    Pixelate toute l'image en 2 resize seulement:
    - downscale (bilinear)
    - upscale (nearest) dans tout l'image
    """
    h, w = frame.shape[:2]
    if h < 2 or w < 2:
        return frame

    # évite division par 0 / tailles absurdes
    pixel_size = max(2, int(pixel_size))

    small_w = max(1, w // pixel_size)
    small_h = max(1, h // pixel_size)

    small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def _pixelate_roi_copy(dst_frame, x1, y1, x2, y2, pixelated_full):
    """Copie uniquement les zones concernées depuis une image pixelatée globale."""
    dst_frame[y1:y2, x1:x2] = pixelated_full[y1:y2, x1:x2]

# ─────────────────────────────────────────
# SNAP / MERGE RECTANGLES
# ─────────────────────────────────────────

def _snap_to_grid(x, y, w, h, grid=8):
    """Aligne le rectangle sur une grille fixe pour éviter le scintillement."""
    sx = (x // grid) * grid
    sy = (y // grid) * grid

    # Étendre w/h pour compenser le snap
    sw = w + (x - sx)
    sh = h + (y - sy)

    # Arrondir w/h au multiple supérieur
    sw = ((sw + grid - 1) // grid) * grid
    sh = ((sh + grid - 1) // grid) * grid
    return sx, sy, sw, sh

def _rect_expand(rect, margin):
    x, y, w, h = rect
    return (x - margin, y - margin, w + 2 * margin, h + 2 * margin)

def _rects_merge_adjacent(rects, gap=20):
    """
    Fusion simple de rectangles proches / chevauchants.
    Objectif: réduire le nombre de ROI traitées.
    """
    if not rects:
        return []

    # copie mutable
    rects = [tuple(map(int, r)) for r in rects]

    changed = True
    while changed:
        changed = False
        new_rects = []
        used = [False] * len(rects)

        for i in range(len(rects)):
            if used[i]:
                continue
            xi, yi, wi, hi = rects[i]
            x2i = xi + wi
            y2i = yi + hi

            # construire un groupe qu'on fusionne
            cur_x1, cur_y1, cur_x2, cur_y2 = xi, yi, x2i, y2i
            used[i] = True

            for j in range(i + 1, len(rects)):
                if used[j]:
                    continue
                xj, yj, wj, hj = rects[j]
                x2j = xj + wj
                y2j = yj + hj

                # Test "proche" : rectangles qui se touchent/chevauchent à gap près
                # Si gap=0 => fusion seulement si chevauchement.
                intersects_or_close = not (
                    cur_x2 < xj - gap or x2j < cur_x1 - gap or
                    cur_y2 < yj - gap or y2j < cur_y1 - gap
                )

                if intersects_or_close:
                    # union bounding box
                    cur_x1 = min(cur_x1, xj)
                    cur_y1 = min(cur_y1, yj)
                    cur_x2 = max(cur_x2, x2j)
                    cur_y2 = max(cur_y2, y2j)
                    used[j] = True
                    changed = True

            new_rects.append((cur_x1, cur_y1, cur_x2 - cur_x1, cur_y2 - cur_y1))

        rects = new_rects

    return rects

# ─────────────────────────────────────────
# FONCTION PRINCIPALE
# ─────────────────────────────────────────

def apply_blur(frame, plates):
    """
    plates: liste de (x, y, w, h)
    """
    if not plates:
        return

    blur_mode     = cfg.get("blur.mode",       "pixelate")
    pixel_size    = cfg.get("blur.pixel_size",  11)
    blur_strength = cfg.get("blur.strength",    31)
    margin        = cfg.get("blur.margin",      0)
    grid          = cfg.get("blur.snap_grid",   8)

    # NEW: param fusion/skip (facultatif, sinon defaults)
    merge_gap     = cfg.get("blur.merge_gap",  20)      # combien d'espace entre zones pour fusionner
    min_roi_area  = cfg.get("blur.min_roi_area", 80)     # ignore les micro-ROIs
    enable_merge  = cfg.get("blur.merge_rects", True)

    h_frame, w_frame = frame.shape[:2]

    # 1) préparation rectangles + marge + snap + clamp
    rects = []
    for (x, y, w, h) in plates:
        if margin:
            x -= margin
            y -= margin
            w += margin * 2
            h += margin * 2

        if grid > 1:
            x, y, w, h = _snap_to_grid(x, y, w, h, grid)

        # Clamp
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(w_frame, int(x + w))
        y2 = min(h_frame, int(y + h))

        if x2 <= x1 or y2 <= y1:
            continue
        if (x2 - x1) * (y2 - y1) < min_roi_area:
            continue

        rects.append((x1, y1, x2 - x1, y2 - y1))

    if not rects:
        return

    # 2) merge pour réduire le nombre de ROI
    if enable_merge and len(rects) > 1:
        rects = _rects_merge_adjacent(rects, gap=merge_gap)

    if not rects:
        return

    # 3) application blur
    if blur_mode == "pixelate":
        # NEW: pixelate global (2 resize) puis copy seulement sur les zones
        pixelated_full = _pixelate_full(frame, pixel_size)

        for (x1, y1, w, h) in rects:
            x2 = x1 + w
            y2 = y1 + h
            _pixelate_roi_copy(frame, x1, y1, x2, y2, pixelated_full)

    elif blur_mode == "box":
        ksize = (blur_strength | 1)  # impair
        # still ROI-based (box blur n'a pas un "full" aussi simple sans coût)
        for (x1, y1, w, h) in rects:
            roi = frame[y1:y1+h, x1:x1+w]
            if roi.size:
                cv2.blur(roi, (ksize, ksize), dst=roi)

    elif blur_mode == "fill":
        fill_color = cfg.get("blur.fill_color", (80, 80, 80))
        for (x1, y1, w, h) in rects:
            frame[y1:y1+h, x1:x1+w] = fill_color

    else:
        # gaussian
        ksize = (blur_strength | 1)
        for (x1, y1, w, h) in rects:
            roi = frame[y1:y1+h, x1:x1+w]
            if roi.size:
                cv2.GaussianBlur(roi, (ksize, ksize), 0, dst=roi)
