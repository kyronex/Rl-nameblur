# detect.py — v2 : pipeline HSV dual-pass (orange/bleu) + texte blanc

import cv2
import numpy as np
import time

# PARAMÈTRES
SCALE = 2.0

# HSV — fond des cartouches
ORANGE_LOW  = np.array([8, 140, 170])
ORANGE_HIGH = np.array([22, 255, 255])
BLUE_LOW    = np.array([100, 130, 150])
BLUE_HIGH   = np.array([125, 255, 255])

# HSV — texte blanc/lumineux
WHITE_LOW   = np.array([0, 0, 200])
WHITE_HIGH  = np.array([180, 60, 255])

# Morpho — fermeture des masques couleur
KERNEL_CLOSE_H = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (max(int(15 / SCALE), 3), 1)
)
KERNEL_CLOSE_V = cv2.getStructuringElement(cv2.MORPH_RECT,(1, max(int(4 / SCALE), 1)))
KERNEL_WHITE_DILATE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

# Filtre forme (coordonnées ÷SCALE)
MIN_AREA   = int(800 / (SCALE * SCALE))
MIN_WIDTH  = int(50 / SCALE)
MAX_WIDTH  = int(800 / SCALE)
MIN_HEIGHT = int(10 / SCALE)
MAX_HEIGHT = int(100 / SCALE)
MIN_RATIO  = 2.0
MAX_RATIO  = 15.0
MIN_FILL   = 0.35

# BENCHMARK
_stats = {
    "resize_ms":    0.0,
    "hsv_ms":       0.0,
    "masks_ms":     0.0,
    "morpho_ms":    0.0,
    "white_ms":     0.0,
    "contour_ms":   0.0,
    "shape_ms":     0.0,
    "total_ms":     0.0,
    "total_calls":  0,
    "candidates":   0,
    "plates_found": 0,
}

def get_stats():
    n = max(_stats["total_calls"], 1)
    return {
        "resize_avg_ms":   round(_stats["resize_ms"] / n, 2),
        "hsv_avg_ms":      round(_stats["hsv_ms"] / n, 2),
        "masks_avg_ms":    round(_stats["masks_ms"] / n, 2),
        "morpho_avg_ms":   round(_stats["morpho_ms"] / n, 2),
        "white_avg_ms":    round(_stats["white_ms"] / n, 2),
        "contour_avg_ms":  round(_stats["contour_ms"] / n, 2),
        "shape_avg_ms":    round(_stats["shape_ms"] / n, 2),
        "total_avg_ms":    round(_stats["total_ms"] / n, 2),
        "total_calls":     _stats["total_calls"],
        "candidates_avg":  round(_stats["candidates"] / n, 1),
        "plates_found":    _stats["plates_found"],
    }

def reset_stats():
    for k in _stats:
        _stats[k] = 0

# MORPHO SUR MASQUE COULEUR
def process_single_mask(mask_raw):
    closed = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, KERNEL_CLOSE_H)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, KERNEL_CLOSE_V)
    return closed

# FILTRE FORME : CARTOUCHE
def is_cartouche(contour):
    x, y, w, h = cv2.boundingRect(contour)

    if w < MIN_WIDTH or w > MAX_WIDTH:
        return False, None
    if h < MIN_HEIGHT or h > MAX_HEIGHT:
        return False, None

    ratio = w / h
    if ratio < MIN_RATIO or ratio > MAX_RATIO:
        return False, None

    area = cv2.contourArea(contour)
    if area < MIN_AREA:
        return False, None

    rect_area = w * h
    if rect_area == 0:
        return False, None

    fill = area / rect_area
    if fill < MIN_FILL:
        return False, None

    return True, (x, y, w, h)

# FONCTION PRINCIPALE — V2
def detect_plates_v2(frame):
    """
    Pipeline V2 : Resize → HSV → Masques orange/bleu → Morpho →
                  Fusion → Filtre blanc → Contours → Forme → Remap
    Attend une frame RGB.
    """
    _stats["total_calls"] += 1
    plates = []
    t_start = time.perf_counter()

    h_orig, w_orig = frame.shape[:2]

    # ── 1. Resize ──
    t0 = time.perf_counter()
    small = cv2.resize(frame,(int(w_orig / SCALE), int(h_orig / SCALE)),interpolation=cv2.INTER_LINEAR)
    _stats["resize_ms"] += (time.perf_counter() - t0) * 1000

    # ── 2. HSV ──
    t0 = time.perf_counter()
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    _stats["hsv_ms"] += (time.perf_counter() - t0) * 1000

    # ── 3. Masques couleur ──
    t0 = time.perf_counter()
    mask_orange = cv2.inRange(hsv, ORANGE_LOW, ORANGE_HIGH)
    mask_blue   = cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)
    mask_white  = cv2.inRange(hsv, WHITE_LOW, WHITE_HIGH)
    _stats["masks_ms"] += (time.perf_counter() - t0) * 1000

    # ── 4. Morpho sur chaque couleur ──
    t0 = time.perf_counter()
    closed_orange = process_single_mask(mask_orange)
    closed_blue   = process_single_mask(mask_blue)
    blob_mask     = cv2.bitwise_or(closed_orange, closed_blue)
    _stats["morpho_ms"] += (time.perf_counter() - t0) * 1000

    # ── 5. Filtre blanc (trim blobs autour du texte) ──
    t0 = time.perf_counter()
    white_dilated = cv2.dilate(mask_white, KERNEL_WHITE_DILATE, iterations=1)
    blob_trimmed  = cv2.bitwise_and(blob_mask, white_dilated)
    _stats["white_ms"] += (time.perf_counter() - t0) * 1000

    # ── 6. Contours ──
    t0 = time.perf_counter()
    contours, _ = cv2.findContours(
        blob_trimmed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    _stats["contour_ms"] += (time.perf_counter() - t0) * 1000

    # ── 7. Filtrage forme ──
    t0 = time.perf_counter()
    candidates = 0

    for contour in contours:
        valid, bbox = is_cartouche(contour)
        if not valid:
            continue

        x, y, w, h = bbox
        candidates += 1

        # Remap vers résolution originale
        plates.append((
            int(x * SCALE),
            int(y * SCALE),
            int(w * SCALE),
            int(h * SCALE),
        ))

    _stats["shape_ms"]    += (time.perf_counter() - t0) * 1000
    _stats["candidates"]  += candidates
    _stats["plates_found"] += len(plates)
    _stats["total_ms"]    += (time.perf_counter() - t_start) * 1000

    return plates

# TEST INDÉPENDANT
if __name__ == "__main__":
    import dxcam

    # ── output_color="RGB" : cohérent avec COLOR_RGB2HSV dans detect_plates_v2 ──
    camera = dxcam.create(output_color="RGB")
    time.sleep(0.5)
    frame = camera.grab()
    if frame is None:
        time.sleep(1.0)
        frame = camera.grab()

    if frame is None:
        print("❌ Pas de frame capturée")
        exit(1)

    print(f"✅ Frame : {frame.shape[1]}x{frame.shape[0]}")

    # ── Bench V2 — 100 appels ──
    reset_stats()
    for _ in range(100):
        plates = detect_plates_v2(frame)

    stats = get_stats()
    print("=" * 55)
    print(f"  BENCHMARK detect.py — V2 HSV dual-pass — SCALE ÷{SCALE}")
    print("=" * 55)
    for k, v in stats.items():
        print(f"  {k:22s} : {v}")
    print("=" * 55)

    print(f"\n🔍 {len(plates)} cartouche(s) détecté(s)")
    for (x, y, w, h) in plates:
        print(f"   📍 x={x} y={y} w={w} h={h} ratio={w/h:.1f}")
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Affichage : imshow attend BGR → conversion uniquement ici pour le debug
    cv2.imshow("V2 — Detections", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
