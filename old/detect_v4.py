# detect.py — v4 : split par érosion + composantes connexes

import cv2
import numpy as np
import time
from config import cfg
from detect_stats import _stats, get_stats, reset_stats
from detect_tools import is_cartouche

# ── PARAMÈTRES ──
SCALE = cfg.get("detect.scale", 2.0)

ORANGE_LOW  = np.array(cfg.get("detect.hsv.orange.lower", [8,  140, 170]))
ORANGE_HIGH = np.array(cfg.get("detect.hsv.orange.upper", [22, 255, 255]))
BLUE_LOW    = np.array(cfg.get("detect.hsv.blue.lower",   [100, 130, 150]))
BLUE_HIGH   = np.array(cfg.get("detect.hsv.blue.upper",   [125, 255, 255]))

WHITE_LOW   = np.array(cfg.get("detect.hsv.white.lower",  [0,   0,  200]))
WHITE_HIGH  = np.array(cfg.get("detect.hsv.white.upper",  [180, 60, 255]))

# ── MORPHOLOGIE ──
CLOSE_H_WIDTH   = max(int(cfg.get("detect.morpho.close_h_width",  25) / SCALE), 3)
CLOSE_H2_WIDTH  = max(int(cfg.get("detect.morpho.close_h2_width", 40) / SCALE), 3)
CLOSE_V_HEIGHT  = max(int(cfg.get("detect.morpho.close_v_height",  4) / SCALE), 1)

KERNEL_CLOSE_H  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_H_WIDTH, 1))
KERNEL_CLOSE_H2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_H2_WIDTH, 1))
KERNEL_CLOSE_V  = cv2.getStructuringElement(cv2.MORPH_RECT,    (1, CLOSE_V_HEIGHT))
KERNEL_WHITE_DILATE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

# ── ÉROSION SPLIT ──
ERODE_KERNEL_W = max(int(cfg.get("detect.erode.kernel_w", 6) / SCALE), 1)
ERODE_KERNEL_H = max(int(cfg.get("detect.erode.kernel_h", 3) / SCALE), 1)
ERODE_ITERATIONS = cfg.get("detect.erode.iterations", 2)
KERNEL_ERODE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERODE_KERNEL_W, ERODE_KERNEL_H))

# ── FILTRES FORME ──
MIN_AREA   = int(cfg.get("detect.filters.min_area",   800) / (SCALE * SCALE))
MIN_WIDTH  = int(cfg.get("detect.filters.min_width",   50) / SCALE)
MAX_WIDTH  = int(cfg.get("detect.filters.max_width",  800) / SCALE)
MIN_HEIGHT = int(cfg.get("detect.filters.min_height",  10) / SCALE)
MAX_HEIGHT = int(cfg.get("detect.filters.max_height", 100) / SCALE)
MIN_RATIO  = cfg.get("detect.filters.min_ratio", 1.5)
MAX_RATIO  = cfg.get("detect.filters.max_ratio", 15.0)
MIN_FILL   = cfg.get("detect.filters.min_fill",  0.35)

# ── SPLIT ──
SPLIT_MIN_WIDTH  = int(cfg.get("detect.split.min_width",  380) / SCALE)
SPLIT_MIN_HEIGHT = int(cfg.get("detect.split.min_height",  60) / SCALE)
SPLIT_GAP_RATIO  = cfg.get("detect.split.gap_ratio", 0.3)

MERGE_MAX_GAP    = int(cfg.get("detect.merge.max_gap_px",     15))
MERGE_ROW_THRESH = int(cfg.get("detect.merge.same_row_thresh", 20))


# ── Dict params injecté dans detect_tools ──
_params = {
    "SCALE":           SCALE,
    "MIN_AREA":        MIN_AREA,
    "MIN_WIDTH":       MIN_WIDTH,
    "MAX_WIDTH":       MAX_WIDTH,
    "MIN_HEIGHT":      MIN_HEIGHT,
    "MAX_HEIGHT":      MAX_HEIGHT,
    "MIN_RATIO":       MIN_RATIO,
    "MAX_RATIO":       MAX_RATIO,
    "MIN_FILL":        MIN_FILL,
    "SPLIT_MIN_WIDTH":  SPLIT_MIN_WIDTH,
    "SPLIT_MIN_HEIGHT": SPLIT_MIN_HEIGHT,
    "SPLIT_GAP_RATIO":  SPLIT_GAP_RATIO,
    "MERGE_MAX_GAP":    MERGE_MAX_GAP,
    "MERGE_ROW_THRESH": MERGE_ROW_THRESH,
}

# ═══════════════════════════════════════════════════════
#  SPLIT PAR ÉROSION + COMPOSANTES CONNEXES
# ═══════════════════════════════════════════════════════

def _needs_split(w, h):
    """Le blob est-il trop large ou trop haut pour être une seule cartouche ?"""
    return w > SPLIT_MIN_WIDTH or h > SPLIT_MIN_HEIGHT


def split_by_erosion(bbox, mask):
    """
    Découpe un blob suspect en sous-cartouches.

    1. Extrait le ROI du blob dans le masque
    2. Érode progressivement pour séparer les blobs collés
    3. Composantes connexes sur le résultat
    4. Filtre chaque composante avec les critères géométriques
    5. Retourne les bbox valides en coordonnées globales

    Retourne : liste de (x, y, w, h) en coordonnées du masque complet
    """
    x, y, w, h = bbox
    roi = mask[y:y+h, x:x+w]

    # ── Érosion ──
    eroded = cv2.erode(roi, KERNEL_ERODE, iterations=ERODE_ITERATIONS)

    # ── Composantes connexes ──
    num_labels, labels = cv2.connectedComponents(eroded)

    parts = []
    for label_id in range(1, num_labels):  # 0 = fond
        component = (labels == label_id).astype(np.uint8) * 255

        # Bbox de la composante
        coords = cv2.findNonZero(component)
        if coords is None:
            continue
        cx, cy, cw, ch = cv2.boundingRect(coords)

        # Filtre : trop petit = bruit d'érosion
        if cw < MIN_WIDTH or ch < MIN_HEIGHT:
            print(f"    ✂️ érosion: composante ignorée {cw}×{ch} (trop petit)")
            continue

        if cw * ch < MIN_AREA:
            print(f"    ✂️ érosion: composante ignorée area={cw*ch}")
            continue

        ratio = cw / max(ch, 1)
        if ratio < MIN_RATIO or ratio > MAX_RATIO:
            print(f"    ✂️ érosion: ratio {ratio:.1f} hors [{MIN_RATIO}, {MAX_RATIO}]")
            continue

        # ── Compenser l'érosion : dilater la bbox ──
        # L'érosion a grignoté les bords, on rend la marge
        margin_x = ERODE_KERNEL_W * ERODE_ITERATIONS
        margin_y = ERODE_KERNEL_H * ERODE_ITERATIONS

        fx = max(cx - margin_x, 0)
        fy = max(cy - margin_y, 0)
        fw = min(cx + cw + margin_x, w) - fx
        fh = min(cy + ch + margin_y, h) - fy

        # Remap en coordonnées globales
        parts.append((x + fx, y + fy, fw, fh))
        print(f"    ✂️ érosion: composante {label_id} → "
                   f"({x+fx}, {y+fy}, {fw}×{fh})")

    _stats["splits_h"] += max(len(parts) - 1, 0)

    # Si l'érosion n'a rien donné, on retourne le blob original
    if not parts:
        print(f"    ✂️ érosion: aucune composante → on garde le blob original")
        parts = [bbox]

    return parts

# ─────────────────────────────────────────────
# Traitement d'un canal couleur (orange OU bleu)
# ─────────────────────────────────────────────
def _process_channel(color_mask, white_dilated, label=""):
    """
    Morphologie + contours + filtrage pour UN canal couleur.
    Retourne une liste de bbox (x, y, w, h) en coordonnées scalées.
    """
    # Morpho passe 1
    closed = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, KERNEL_CLOSE_H)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, KERNEL_CLOSE_V)

    # Filtre blanc
    trimmed = cv2.bitwise_and(closed, white_dilated)

    # Morpho passe 2
    trimmed = cv2.morphologyEx(trimmed, cv2.MORPH_CLOSE, KERNEL_CLOSE_H2)

    # Contours
    contours, _ = cv2.findContours(trimmed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    _stats["contours_raw"] += len(contours)

    results = []
    for contour in contours:
        valid, bbox = is_cartouche(contour, _params, _stats)
        if not valid:
            continue

        x, y, w, h = bbox
        _stats["candidates"] += 1

        print(f"  📐 [{label}] candidat: x={x} y={y} w={w} h={h} "
                  f"(réel: {int(w*SCALE)}×{int(h*SCALE)})")

        if _needs_split(w, h):
            print(f"    → blob suspect ({w}×{h}), érosion + CC")
            parts = split_by_erosion(bbox, trimmed)
        else:
            parts = [bbox]

        results.extend(parts)

    return results


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
    small = cv2.resize(frame,
                       (int(w_orig / SCALE), int(h_orig / SCALE)),
                       interpolation=cv2.INTER_LINEAR)
    _stats["resize_ms"] += (time.perf_counter() - t0) * 1000

    # ── 2. HSV ──
    t0 = time.perf_counter()
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    _stats["hsv_ms"] += (time.perf_counter() - t0) * 1000

    # ── 3. Masques couleur ──
    t0 = time.perf_counter()
    mask_orange = cv2.inRange(hsv, ORANGE_LOW, ORANGE_HIGH)
    mask_blue   = cv2.inRange(hsv, BLUE_LOW,   BLUE_HIGH)
    mask_white  = cv2.inRange(hsv, WHITE_LOW,  WHITE_HIGH)
    _stats["masks_ms"] += (time.perf_counter() - t0) * 1000

    # ── 4. Dilatation blanc (commun aux deux canaux) ──
    t0 = time.perf_counter()
    white_dilated = cv2.dilate(mask_white, KERNEL_WHITE_DILATE, iterations=1)
    _stats["white_ms"] += (time.perf_counter() - t0) * 1000

    # ── 5. Traitement ORANGE ──
    t0 = time.perf_counter()
    orange_plates = _process_channel(mask_orange, white_dilated, label="O")
    _stats["orange_ms"] += (time.perf_counter() - t0) * 1000

    # ── 6. Traitement BLEU ──
    t0 = time.perf_counter()
    blue_plates = _process_channel(mask_blue, white_dilated, label="B")
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
    _stats["total_ms"] += (time.perf_counter() - t_start) * 1000
    _stats["total_calls"] += 1

    return plates
