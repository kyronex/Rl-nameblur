# detect.py
import cv2
import numpy as np
import time

SCALE = 2.0   # 2.0 = safe, 2.5 = plus rapide, teste les deux

ORANGE_LOW  = np.array([12, 190, 220])
ORANGE_HIGH = np.array([17, 255, 255])

BLUE_LOW  = np.array([105, 180, 200])
BLUE_HIGH = np.array([115, 255, 255])

# ─────────────────────────────────────────
# PARAMÈTRES FORME — CARTOUCHE ÉGYPTIEN
# ─── adaptés auto à SCALE ───────────────

MIN_AREA   = int(500 / (SCALE * SCALE))     # ← adapté

MIN_HEIGHT = int(10 / SCALE)
MAX_HEIGHT = int(100 / SCALE)
MIN_WIDTH  = int(40 / SCALE)
MAX_WIDTH  = int(950 / SCALE)

MIN_RATIO = 2.0
MAX_RATIO = 15.0

MIN_FILL = 0.50
MIN_CONVEXITY = 0.70

# ─────────────────────────────────────────
# MORPHO KERNELS — adaptés à SCALE
# ─────────────────────────────────────────

KERNEL_LINK  = cv2.getStructuringElement(
    cv2.MORPH_RECT, (max(int(10 / SCALE), 3), max(int(2 / SCALE), 1))
)
KERNEL_CLEAN = np.ones((max(int(3 / SCALE), 1), max(int(3 / SCALE), 1)), np.uint8)

# ─────────────────────────────────────────
# BENCHMARK
# ─────────────────────────────────────────

_stats = {
    "hsv_ms":       0.0,
    "mask_ms":      0.0,
    "morph_ms":     0.0,
    "contour_ms":   0.0,
    "filter_ms":    0.0,
    "total_ms":     0.0,
    "total_calls":  0,
    "plates_found": 0,
}

def get_stats():
    n = max(_stats["total_calls"], 1)
    return {
        "hsv_avg_ms":      round(_stats["hsv_ms"] / n, 2),
        "mask_avg_ms":     round(_stats["mask_ms"] / n, 2),
        "morph_avg_ms":    round(_stats["morph_ms"] / n, 2),
        "contour_avg_ms":  round(_stats["contour_ms"] / n, 2),
        "filter_avg_ms":   round(_stats["filter_ms"] / n, 2),
        "total_avg_ms":    round(_stats["total_ms"] / n, 2),
        "total_calls":     _stats["total_calls"],
        "plates_found":    _stats["plates_found"],
    }

def reset_stats():
    for k in _stats:
        _stats[k] = 0

# ─────────────────────────────────────────
# SOUS-FILTRE : FORME CARTOUCHE
# ─────────────────────────────────────────

def is_cartouche(contour):

    x, y, w, h = cv2.boundingRect(contour)

    if w < MIN_WIDTH or w > MAX_WIDTH:
        return False, None
    if h < MIN_HEIGHT or h > MAX_HEIGHT:
        return False, None

    ratio = w / h
    if ratio < MIN_RATIO or ratio > MAX_RATIO:
        return False, None

    area_contour = cv2.contourArea(contour)
    area_rect = w * h
    if area_rect == 0:
        return False, None

    fill = area_contour / area_rect
    if fill < MIN_FILL:
        return False, None

    hull = cv2.convexHull(contour)
    area_hull = cv2.contourArea(hull)
    if area_hull == 0:
        return False, None

    convexity = area_contour / area_hull
    if convexity < MIN_CONVEXITY:
        return False, None

    return True, (x, y, w, h)

# ─────────────────────────────────────────
# FONCTION PRINCIPALE
# ─────────────────────────────────────────

def detect_plates(frame):

    _stats["total_calls"] += 1
    plates = []

    # ── RESIZE ──
    t_start = time.perf_counter()
    h_orig, w_orig = frame.shape[:2]
    small = cv2.resize(frame, (int(w_orig / SCALE), int(h_orig / SCALE)),interpolation=cv2.INTER_LINEAR)

    # ── HSV ──
    t0 = time.perf_counter()
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)          # ← small
    t1 = time.perf_counter()
    _stats["hsv_ms"] += (t1 - t0) * 1000

    # ── Masques couleur ──
    mask_orange = cv2.inRange(hsv, ORANGE_LOW, ORANGE_HIGH)
    mask_blue   = cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)
    mask = cv2.bitwise_or(mask_orange, mask_blue)
    t2 = time.perf_counter()
    _stats["mask_ms"] += (t2 - t1) * 1000

    # ── Morphologie ──
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL_LINK)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL_CLEAN)
    t3 = time.perf_counter()
    _stats["morph_ms"] += (t3 - t2) * 1000

    # ── Contours ──
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    t4 = time.perf_counter()
    _stats["contour_ms"] += (t4 - t3) * 1000

    # ── Filtrage ──
    if hierarchy is None:
        _stats["filter_ms"] += 0
        _stats["total_ms"] += (t4 - t_start) * 1000
        return plates

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue

        is_valid, bbox = is_cartouche(contour)
        if not is_valid:
            x, y, w, h = cv2.boundingRect(contour)
            if h == 0:
                continue
            ratio = w / h
            if (MIN_WIDTH * 0.7 <= w <= MAX_WIDTH and
                MIN_HEIGHT <= h <= MAX_HEIGHT and
                ratio >= MIN_RATIO):
                bbox = (x, y, w, h)
            else:
                continue

        # ── Enfants (lettres) ──
        child_count = 0
        child_idx = hierarchy[0][i][2]
        while child_idx != -1:
            child_count += 1
            child_idx = hierarchy[0][child_idx][0]

        if child_count < 1:
            continue

        # ── REMAP coordonnées vers résolution originale ──  ← NOUVEAU
        bx, by, bw, bh = bbox
        plates.append((
            int(bx * SCALE),
            int(by * SCALE),
            int(bw * SCALE),
            int(bh * SCALE),
        ))

    t5 = time.perf_counter()
    _stats["filter_ms"] += (t5 - t4) * 1000
    _stats["total_ms"]  += (t5 - t_start) * 1000
    _stats["plates_found"] += len(plates)

    return plates

# ─────────────────────────────────────────
# TEST INDÉPENDANT
# ─────────────────────────────────────────

if __name__ == "__main__":
    import dxcam

    camera = dxcam.create()
    frame = camera.grab()

    if frame is not None:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Bench sur 100 appels
        reset_stats()
        for _ in range(100):
            plates = detect_plates(frame_bgr)

        stats = get_stats()
        print("=" * 50)
        print(f"  BENCHMARK detect.py — SCALE ÷{SCALE}")
        print("=" * 50)
        for k, v in stats.items():
            print(f"  {k:20s} : {v}")
        print("=" * 50)

        print(f"\n🔍 {len(plates)} cartouche(s) détecté(s)")
        for (x, y, w, h) in plates:
            cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
            print(f"   📍 x={x} y={y} w={w} h={h} ratio={w/h:.1f}")

        cv2.imshow("Detections", frame_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ Pas de frame capturée")
