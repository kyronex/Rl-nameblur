# blur.py
import cv2
import time

# ─────────────────────────────────────────
# PARAMÈTRES DU FLOU
# ─────────────────────────────────────────

BLUR_STRENGTH = 51
MARGIN = -2

# ─────────────────────────────────────────
# BENCHMARK
# ─────────────────────────────────────────

_stats = {
    "blur_ms":      0.0,
    "cvt_ms":       0.0,
    "total_ms":     0.0,
    "total_calls":  0,
    "zones_blurred": 0,
}

def get_stats():
    n = max(_stats["total_calls"], 1)
    return {
        "blur_avg_ms":    round(_stats["blur_ms"] / n, 2),
        "cvt_avg_ms":     round(_stats["cvt_ms"] / n, 2),
        "total_avg_ms":   round(_stats["total_ms"] / n, 2),
        "total_calls":    _stats["total_calls"],
        "zones_blurred":  _stats["zones_blurred"],
    }

def reset_stats():
    for k in _stats:
        _stats[k] = 0

# ─────────────────────────────────────────
# FONCTION PRINCIPALE
# ─────────────────────────────────────────

def apply_blur(frame, plates, rgb_buffer=None):
    """
    Prend une frame (BGR) et une liste de rectangles [(x, y, w, h), ...]
    Floute les zones + convertit BGR → RGB
    Retourne la frame RGB prête pour vcam.send()
    """
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
        cv2.GaussianBlur(roi, (BLUR_STRENGTH, BLUR_STRENGTH), 0, dst=roi)
        _stats["blur_ms"] += (time.perf_counter() - t_blur) * 1000

        _stats["zones_blurred"] += 1

    # ── Conversion BGR → RGB (fusionnée ici) ──
    t_cvt = time.perf_counter()
    if rgb_buffer is not None:
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=rgb_buffer)
        result = rgb_buffer
    else:
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    _stats["cvt_ms"] += (time.perf_counter() - t_cvt) * 1000

    _stats["total_ms"] += (time.perf_counter() - t0) * 1000

    return result

# ─────────────────────────────────────────
# TEST INDÉPENDANT
# ─────────────────────────────────────────

if __name__ == "__main__":
    import dxcam
    import numpy as np
    from detect import detect_plates

    camera = dxcam.create()
    frame = camera.grab()

    if frame is not None:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        plates = detect_plates(frame_bgr)
        print(f"🔍 {len(plates)} plaque(s) détectée(s)")

        # Bench sur 100 appels
        reset_stats()
        for _ in range(100):
            test_frame = frame_bgr.copy()
            apply_blur(test_frame, plates)

        stats = get_stats()
        print("=" * 50)
        print("  BENCHMARK blur.py (100 appels)")
        print("=" * 50)
        for k, v in stats.items():
            print(f"  {k:20s} : {v}")
        print("=" * 50)

        # Affiche le résultat final
        frame_blurred = apply_blur(frame_bgr, plates)
        cv2.imshow("Avant/Apres flou", frame_blurred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ Pas de frame capturée")
