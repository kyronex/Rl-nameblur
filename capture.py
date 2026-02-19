# capture.py
import dxcam
import cv2

# ─────────────────────────────────────────
# PARAMÈTRES
# ─────────────────────────────────────────
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
TARGET_FPS = 60

# ─────────────────────────────────────────
# INITIALISATION
# ─────────────────────────────────────────
camera = dxcam.create()

def start():
    """Démarre la capture d'écran"""
    camera.start(target_fps=TARGET_FPS)
    print("✅ DXCam prêt")

def capture_screen():
    """Récupère la dernière frame"""
    frame = camera.get_latest_frame()
    if frame is None:
        return None

    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def stop():
    """Arrête la capture"""
    camera.stop()
    print("✅ DXCam arrêté")
