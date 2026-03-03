# blur.py
import cv2
import time
from config import cfg

# ─────────────────────────────────────────
# BENCHMARK
# ─────────────────────────────────────────

_stats = {
    "blur_ms":       0.0,
    "total_ms":      0.0,
    "total_calls":   0,
    "zones_blurred": 0,
}

def get_stats():
    n = max(_stats["total_calls"], 1)
    return {
        "blur_avg_ms":   round(_stats["blur_ms"] / n, 2),
        "total_avg_ms":  round(_stats["total_ms"] / n, 2),
        "total_calls":   _stats["total_calls"],
        "zones_blurred": _stats["zones_blurred"],
        "blur_mode":     cfg.get("blur.mode", "pixelate"),
    }

def reset_stats():
    for k in _stats:
        if isinstance(_stats[k], (int, float)):
            _stats[k] = 0

# ─────────────────────────────────────────
# PIXELISATION
# ─────────────────────────────────────────

def _pixelate_roi(roi, pixel_size):
    """Resize down → resize up = effet mosaïque."""
    h, w = roi.shape[:2]
    if h < 2 or w < 2:
        return
    small_w = max(1, w // pixel_size)
    small_h = max(1, h // pixel_size)
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST, dst=roi)

def snap_to_grid(x, y, w, h, grid=8):
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

# ─────────────────────────────────────────
# FONCTION PRINCIPALE
# ─────────────────────────────────────────

def apply_blur(frame, plates):
    if not plates:
        return

    _stats["total_calls"] += 1
    t0 = time.perf_counter()

    blur_mode     = cfg.get("blur.mode",       "pixelate")
    pixel_size    = cfg.get("blur.pixel_size",  11)
    blur_strength = cfg.get("blur.strength",    31)
    margin        = cfg.get("blur.margin",      0)
    grid          = cfg.get("blur.snap_grid",   8)

    h_frame, w_frame = frame.shape[:2]

    for (x, y, w, h) in plates:
        # ── Margin ──
        x -= margin
        y -= margin
        w += margin * 2
        h += margin * 2

        # ── Snap grille (anti-scintillement pixelate/box) ──
        if grid > 1 :
            x, y, w, h = snap_to_grid(x, y, w, h, grid)

        # ── Clamp ──
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_frame, x + w)
        y2 = min(h_frame, y + h)

        if x2 <= x1 or y2 <= y1:
            continue

        roi = frame[y1:y2, x1:x2]

        t_blur = time.perf_counter()

        if blur_mode == "pixelate":
            _pixelate_roi(roi, pixel_size)
        elif blur_mode == "box":
            ksize = blur_strength | 1          # force impair
            cv2.blur(roi, (ksize, ksize), dst=roi)
        elif blur_mode == "fill":
            roi[:] = cfg.get("blur.fill_color", (80, 80, 80))
        else:
            ksize = blur_strength | 1          # force impair
            cv2.GaussianBlur(roi, (ksize, ksize), 0, dst=roi)

        _stats["blur_ms"]       += (time.perf_counter() - t_blur) * 1000
        _stats["zones_blurred"] += 1

    _stats["total_ms"] += (time.perf_counter() - t0) * 1000


# ─────────────────────────────────────────
# TEST INDÉPENDANT
# ─────────────────────────────────────────

if __name__ == "__main__":
    import dxcam
    from detect import detect_plates_v2

    camera = dxcam.create(output_color="RGB")
    frame = camera.grab()

    if frame is not None:
        plates = detect_plates_v2(frame)
        print(f"🔍 {len(plates)} plaque(s) détectée(s)")

        # Bench pixelate
        reset_stats()
        for _ in range(100):
            apply_blur(frame.copy(), plates)
        p_stats = get_stats()

        # Bench gaussian (override temporaire via cfg non supporté ici → test direct)
        print("=" * 50)
        print("  BENCHMARK blur.py (100 appels)")
        print("=" * 50)
        print("  PIXELATE:")
        for k, v in p_stats.items():
            print(f"    {k:20s} : {v}")
        print("=" * 50)

        apply_blur(frame, plates)
        cv2.imshow("Blur", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ Pas de frame capturée")
