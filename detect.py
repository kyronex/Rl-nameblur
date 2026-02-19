# detect.py
import cv2
import numpy as np

# ─────────────────────────────────────────
# PARAMÈTRES HSV (échelle OpenCV)
# ─────────────────────────────────────────

ORANGE_LOW  = np.array([12, 190, 220])
ORANGE_HIGH = np.array([17, 255, 255])

BLUE_LOW  = np.array([105, 180, 200])
BLUE_HIGH = np.array([115, 255, 255])

# ─────────────────────────────────────────
# PARAMÈTRES FORME — CARTOUCHE ÉGYPTIEN
# ─────────────────────────────────────────

MIN_AREA   = 500

MIN_HEIGHT = 13
MAX_HEIGHT = 100
MIN_WIDTH  = 40
MAX_WIDTH  = 950

# Ratio largeur/hauteur d'un cartouche (allongé)
MIN_RATIO = 2.0
MAX_RATIO = 15.0

# Remplissage minimum (aire contour / aire rectangle)
MIN_FILL = 0.50

# Convexité minimum (aire contour / aire enveloppe convexe)
MIN_CONVEXITY = 0.70


# ─────────────────────────────────────────
# SOUS-FILTRE : FORME CARTOUCHE
# ─────────────────────────────────────────

def is_cartouche(contour):

    x, y, w, h = cv2.boundingRect(contour)

    # ── Taille ──
    if w < MIN_WIDTH or w > MAX_WIDTH:
        return False, None
    if h < MIN_HEIGHT or h > MAX_HEIGHT:
        return False, None

    # ── Ratio  ──
    ratio = w / h
    if ratio < MIN_RATIO or ratio > MAX_RATIO:
        return False, None

    # ── Remplissage : aire contour vs aire rectangle ──
    area_contour = cv2.contourArea(contour)
    area_rect = w * h
    if area_rect == 0:
        return False, None

    fill = area_contour / area_rect
    if fill < MIN_FILL:
        return False, None

    # ── Convexité : contour vs enveloppe convexe ──
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

    plates = []

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_orange = cv2.inRange(hsv, ORANGE_LOW, ORANGE_HIGH)
    mask_blue   = cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)
    mask = cv2.bitwise_or(mask_orange, mask_blue)

    kernel_link = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_link)

    # Nettoyage

    kernel_clean = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_clean)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        return plates

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        ratio = w / h if h > 0 else 0


        is_valid, bbox = is_cartouche(contour)
        if not is_valid:
            # ── Fallback : si ratio OK + taille OK → accepter quand même ──
            # Ça rattrape les plaques coupées par le bord d'écran
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
            x, y, w, h = cv2.boundingRect(contour)


        # ── Passe 2 : enfants (lettres à l'intérieur) ──

        child_count = 0
        child_idx = hierarchy[0][i][2]

        while child_idx != -1:
            child_count += 1
            child_idx = hierarchy[0][child_idx][0]

        if child_count < 1:
            continue

        plates.append(bbox)

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

        plates = detect_plates(frame_bgr)
        print(f"🔍 {len(plates)} cartouche(s) détecté(s)")

        for (x, y, w, h) in plates:
            cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
            ratio = w / h
            print(f"   📍 x={x} y={y} w={w} h={h} ratio={ratio:.1f}")

        cv2.imshow("Detections", frame_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ Pas de frame capturée")
