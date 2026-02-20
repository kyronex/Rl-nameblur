# capture.py
import dxcam
import numpy as np
import time

# ─────────────────────────────────────────
# PARAMÈTRES
# ─────────────────────────────────────────
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

CAPTURE_FPS = 60      # DXCam : juste au-dessus du FPS réel (~38)
VCAM_FPS = 120        # pyvirtualcam : haut pour ne jamais bloquer send()

# ─────────────────────────────────────────
# INIT
# ─────────────────────────────────────────

camera = None

# ─────────────────────────────────────────
# BENCHMARK
# ─────────────────────────────────────────
_stats = {
    "grab_ms": 0.0,
    "none_count": 0,
    "total_calls": 0,
}

def get_stats():
    """Retourne les stats moyennes"""
    n = max(_stats["total_calls"], 1)
    return {
        "grab_avg_ms": round(_stats["grab_ms"] / n, 2),
        "none_count":  _stats["none_count"],
        "total_calls": _stats["total_calls"],
    }

def reset_stats():
    for k in _stats:
        _stats[k] = 0

# ─────────────────────────────────────────
# INITIALISATION
# ─────────────────────────────────────────

def start():
    """Démarre la capture d'écran"""
    global camera
    camera = dxcam.create(output_color="BGR")
    camera.start(target_fps=CAPTURE_FPS)
    print(f"📸 Capture lancée → {SCREEN_WIDTH}x{SCREEN_HEIGHT} @ {CAPTURE_FPS}fps (capture)")
    print(f"🎥 Vcam déclarée @ {VCAM_FPS}fps (pas de blocage send)")

def capture_screen():
    """Récupère la dernière frame"""
    _stats["total_calls"] += 1

    t0 = time.perf_counter()
    frame = camera.get_latest_frame()
    _stats["grab_ms"] += (time.perf_counter() - t0) * 1000

    if frame is None:
        _stats["none_count"] += 1
        return None

    return np.array(frame)

def stop():
    """Arrête la capture"""
    global camera
    if camera:
        camera.stop()
        print("✅ DXCam arrêté")
