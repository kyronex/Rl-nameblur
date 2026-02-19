# detect.py
import cv2
import numpy as np

# ─────────────────────────────────────────
# PARAMÈTRES HSV (échelle OpenCV)
# ─────────────────────────────────────────

# On prend une MARGE de ±10 sur H, ±50 sur S, ±50 sur V

# Orange : hsv(27, 81%, 91%) → OpenCV : (13, 207, 232)
# ORANGE_LOW  = np.array([10,150,180])
# ORANGE_HIGH = np.array([16,255,255])
ORANGE_LOW  = np.array([10, 160, 180])
ORANGE_HIGH = np.array([17, 255, 255])
# Bleu : hsv(221, 83%, 92%) → OpenCV : (110, 212, 235)
# BLUE_LOW  = np.array([106,150,180])
# BLUE_HIGH = np.array([114,255,255])
BLUE_LOW  = np.array([105, 180, 200])
BLUE_HIGH = np.array([115, 255, 255])

# ─────────────────────────────────────────
# PARAMÈTRES FORME (plaque d'immatriculation)
# ─────────────────────────────────────────

MIN_AREA = 1000

# Taille minimum en pixels (ignore les trucs trop petits)
MIN_HEIGHT = 15
MAX_HEIGHT = 85
MAX_WIDTH = 350

# Ratio largeur/hauteur d'une plaque
# Une plaque est plus large que haute (~3:1 à ~5:1)
MIN_RATIO = 2.0
MAX_RATIO = 15.0


# ─────────────────────────────────────────
# FONCTION PRINCIPALE
# ─────────────────────────────────────────

def detect_plates(frame):
    """
    Prend une image (BGR de DXCam/OpenCV)
    Retourne une liste de rectangles [(x, y, w, h), ...]
    où des plaques ont été détectées
    """
    plates = []

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_orange = cv2.inRange(hsv, ORANGE_LOW, ORANGE_HIGH)
    mask_blue = cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)
    mask = cv2.bitwise_or(mask_orange, mask_blue)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    #contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Si aucun contour trouvé → on sort
    if hierarchy is None:
        return plates

    # ══════════════════════════════════════
    # ÉTAPE 3 : FORME — garder que les rectangles
    # ══════════════════════════════════════

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            continue

        ratio = w / h
        if not (MIN_RATIO <= ratio <= MAX_RATIO):
            continue

        # ──────────────────────────────────
        # NOUVEAU : compter les enfants
        # ──────────────────────────────────

        child_count = 0
        child_idx = hierarchy[0][i][2]  # premier enfant

        while child_idx != -1:
            child_count += 1
            child_idx = hierarchy[0][child_idx][0]  # enfant suivant

        # Une plaque a des lettres dedans → au moins 1 enfant
        if child_count < 1:
            continue

        plates.append((x, y, w, h))

    return plates
    """
    for contour in contours:
        area = cv2.contourArea(contour)
        # Trop petit → poubelle
        if area < MIN_AREA:
            continue
        # Trouve le rectangle englobant
        x, y, w, h = cv2.boundingRect(contour)

        # Filtre par ratio
        ratio = w / h
        # Vérifie que ça ressemble à une plaque
        if MIN_RATIO <= ratio <= MAX_RATIO:
            plates.append((x, y, w, h))

    return plates
    """


# ─────────────────────────────────────────
# TEST INDÉPENDANT
# ─────────────────────────────────────────

if __name__ == "__main__":
    """
    Test avec DXCam : capture une frame et montre les détections
    """
    import dxcam

    camera = dxcam.create()
    frame = camera.grab()

    if frame is not None:
        # DXCam donne du RGB, OpenCV veut du BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Détecte
        plates = detect_plates(frame_bgr)
        print(f"🔍 {len(plates)} plaque(s) détectée(s)")

        # Dessine les rectangles pour visualiser
        for (x, y, w, h) in plates:
            cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
            print(f"   📍 x={x} y={y} w={w} h={h} ratio={w/h:.1f}")

        # Affiche le résultat
        cv2.imshow("Detections", frame_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ Pas de frame capturée")
