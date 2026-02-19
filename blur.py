# blur.py
import cv2

# ─────────────────────────────────────────
# PARAMÈTRES DU FLOU
# ─────────────────────────────────────────

# Taille du kernel de flou (doit être impair)
# Plus c'est grand → plus c'est flou
# 51 = illisible, 99 = complètement opaque
BLUR_STRENGTH = 35

# Marge autour de la zone détectée (en pixels)
# Pour être sûr de couvrir toute la plaque
MARGIN = 5

# ─────────────────────────────────────────
# FONCTION PRINCIPALE
# ─────────────────────────────────────────

def apply_blur(frame, plates):
    """
    Prend une frame (BGR) et une liste de rectangles [(x, y, w, h), ...]
    Retourne la frame avec les zones floutées
    """
    h_frame, w_frame = frame.shape[:2]

    for (x, y, w, h) in plates:
        # ──────────────────────────────
        # Ajouter la marge
        # ──────────────────────────────
        x1 = max(0, x - MARGIN)
        y1 = max(0, y - MARGIN)
        x2 = min(w_frame, x + w + MARGIN)
        y2 = min(h_frame, y + h + MARGIN)

        # ──────────────────────────────
        # Extraire la zone
        # ──────────────────────────────
        roi = frame[y1:y2, x1:x2]

        # ──────────────────────────────
        # Appliquer le flou gaussien
        # ──────────────────────────────
        blurred = cv2.GaussianBlur(roi, (BLUR_STRENGTH, BLUR_STRENGTH), 0)

        # ──────────────────────────────
        # Remettre la zone floutée
        # ──────────────────────────────
        frame[y1:y2, x1:x2] = blurred

    return frame


# ─────────────────────────────────────────
# TEST INDÉPENDANT
# ─────────────────────────────────────────

if __name__ == "__main__":
    """
    Test : capture une frame, détecte, floute, affiche
    """
    import dxcam
    import numpy as np
    from detect import detect_plates

    camera = dxcam.create()
    frame = camera.grab()

    if frame is not None:
        # DXCam → RGB, OpenCV → BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Détecte les plaques
        plates = detect_plates(frame_bgr)
        print(f"🔍 {len(plates)} plaque(s) détectée(s)")

        # Applique le flou
        frame_blurred = apply_blur(frame_bgr, plates)

        # Affiche
        cv2.imshow("Avant/Apres flou", frame_blurred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ Pas de frame capturée")
