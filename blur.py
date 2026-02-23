# blur.py
import cv2
import time
import numpy as np

# ─────────────────────────────────────────
# PARAMÈTRES
# ─────────────────────────────────────────

BLUR_MODE = "pixelate"  # "gaussian" ou "pixelate"
BLUR_STRENGTH = 31      # kernel gaussian (si mode gaussian)
PIXEL_SIZE = 11         # taille des blocs (si mode pixelate)
MARGIN = -6

# ─────────────────────────────────────────
# BENCHMARK
# ─────────────────────────────────────────

_stats = {
    "blur_ms":      0.0,
    "total_ms":     0.0,
    "total_calls":  0,
    "zones_blurred": 0,
}

def get_stats():
    n = max(_stats["total_calls"], 1)
    return {
        "blur_avg_ms":    round(_stats["blur_ms"] / n, 2),
        "total_avg_ms":   round(_stats["total_ms"] / n, 2),
        "total_calls":    _stats["total_calls"],
        "zones_blurred":  _stats["zones_blurred"],
        "blur_mode":      BLUR_MODE,
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

# ─────────────────────────────────────────
# FONCTION PRINCIPALE
# ─────────────────────────────────────────

def apply_blur(frame, plates):
    _stats["total_calls"] += 1
    t0 = time.perf_counter()

    h_frame, w_frame = frame.shape[:2]

    for (x, y, w, h) in plates:
        x1 = max(0, x - MARGIN)
        y1 = max(0, y - MARGIN)
        x2 = min(w_frame, x + w + MARGIN)
        y2 = min(h_frame, y + h + MARGIN)

        roi = frame[y1:y2, x1:x2]

        t_blur = time.perf_counter()

        if BLUR_MODE == "pixelate":
            _pixelate_roi(roi, PIXEL_SIZE)
        else:
            cv2.GaussianBlur(roi, (BLUR_STRENGTH, BLUR_STRENGTH), 0, dst=roi)

        _stats["blur_ms"] += (time.perf_counter() - t_blur) * 1000
        _stats["zones_blurred"] += 1
    _stats["total_ms"] += (time.perf_counter() - t0) * 1000

# ─────────────────────────────────────────
# TEST INDÉPENDANT
# ─────────────────────────────────────────

if __name__ == "__main__":
    import dxcam
    import numpy as np
    from detect import detect_plates_v2

    camera = dxcam.create(output_color="RGB")
    frame = camera.grab()

    if frame is not None:
        plates = detect_plates_v2(frame)
        print(f"🔍 {len(plates)} plaque(s) détectée(s)")

        # Bench gaussian
        BLUR_MODE = "gaussian"
        reset_stats()
        for _ in range(100):
            apply_blur(frame.copy(), plates)
        g_stats = get_stats()

        # Bench pixelate
        BLUR_MODE = "pixelate"
        reset_stats()
        for _ in range(100):
            apply_blur(frame.copy(), plates)
        p_stats = get_stats()

        print("=" * 50)
        print("  BENCHMARK blur.py (100 appels)")
        print("=" * 50)
        print("  GAUSSIAN:")
        for k, v in g_stats.items():
            print(f"    {k:20s} : {v}")
        print("  PIXELATE:")
        for k, v in p_stats.items():
            print(f"    {k:20s} : {v}")
        print("=" * 50)

        frame_result = apply_blur(frame, plates)
        # Affichage : imshow attend BGR → conversion uniquement ici pour le debug
        cv2.imshow("Pixelate", cv2.cvtColor(frame_result, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ Pas de frame capturée")
