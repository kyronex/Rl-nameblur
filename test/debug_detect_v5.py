# debug_detect_v5.py
import cv2
import numpy as np
import dxcam
import time
import os

OUTPUT_DIR = "debug_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
for f in os.listdir(OUTPUT_DIR):
    os.remove(os.path.join(OUTPUT_DIR, f))

# PARAMÈTRES
SCALE = 2.0

# HSV — fond des cartouches
ORANGE_LOW  = np.array([8, 140, 170])
ORANGE_HIGH = np.array([22, 255, 255])
BLUE_LOW    = np.array([100, 130, 150])
BLUE_HIGH   = np.array([125, 255, 255])

# HSV — texte blanc/lumineux (S bas, V haut)
WHITE_LOW   = np.array([0, 0, 200])
WHITE_HIGH  = np.array([180, 60, 255])

# Morpho
KERNEL_CLOSE_H = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(max(int(15 / SCALE), 3), 1))
KERNEL_CLOSE_V = cv2.getStructuringElement(cv2.MORPH_RECT,(1, max(int(4 / SCALE), 1)))
KERNEL_WHITE_DILATE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

# Filtre forme (coordonnées SCALE)
MIN_AREA   = int(800 / (SCALE * SCALE))
MIN_WIDTH  = int(50 / SCALE)
MAX_WIDTH  = int(800 / SCALE)
MIN_HEIGHT = int(10 / SCALE)
MAX_HEIGHT = int(100 / SCALE)
MIN_RATIO  = 2.0
MAX_RATIO  = 15.0
MIN_FILL   = 0.35

# Filtre densité texte blanc
MIN_TRANSITIONS = 6
MIN_WHITE_RATIO = 0.05   # au moins 5% de blanc dans la zone

# TRAITEMENT D'UN MASQUE COULEUR
def process_single_mask(mask_raw):
    # Morpho
    closed = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, KERNEL_CLOSE_H)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, KERNEL_CLOSE_V)
    return  closed

def is_cartouche(contour):
    """Vérifie si un contour a la forme d'un cartouche de nom."""
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

# DIAGNOSTIC COMPLET

def run_diagnostic(frame, capture_id=1):
    prefix = os.path.join(OUTPUT_DIR, f"cap{capture_id:02d}")

    # 01 : resize
    h_orig, w_orig = frame.shape[:2]
    small = cv2.resize(frame, (int(w_orig / SCALE), int(h_orig / SCALE)),interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(f"{prefix}_01_resized.png", small)

    # HSV
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    # 02/03 : masques couleur
    mask_orange = cv2.inRange(hsv, ORANGE_LOW, ORANGE_HIGH)
    mask_blue   = cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)

    # 04b : masque blanc (texte)
    mask_white = cv2.inRange(hsv, WHITE_LOW, WHITE_HIGH)

    # Passe ORANGE
    closed_orange = process_single_mask(mask_orange)

    # Passe BLEU
    closed_blue = process_single_mask(mask_blue)

    blob_mask = cv2.bitwise_or(closed_orange, closed_blue)

    # 2. Dilater mask_white de 5px
    white_dilated = cv2.dilate(mask_white, KERNEL_WHITE_DILATE, iterations=1)

    # 3. Réduire les blobs pour épouser le blanc à 5px
    blob_trimmed = cv2.bitwise_and(blob_mask, white_dilated)

    contours, _ = cv2.findContours(blob_trimmed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_detections = []
    for contour in contours:
      valid, bbox = is_cartouche(contour)
      if not valid:
        continue
      x, y, w, h = bbox
      all_detections.append((
            int(x * SCALE),
            int(y * SCALE),
            int(w * SCALE),
            int(h * SCALE),
        ))

    return all_detections

def print_files(capture_id):
    prefix = f"cap{capture_id:02d}"
    files = sorted(f for f in os.listdir(OUTPUT_DIR) if f.startswith(prefix))
    print(f"\n  📁 {len(files)} fichiers dans ./{OUTPUT_DIR}/")
    for f in files:
        print(f"    {f}")

if __name__ == "__main__":
    print("📸 Initialisation DXCam...")
    camera = dxcam.create(output_color="BGR")
    time.sleep(0.5)

    frame = camera.grab()
    if frame is None:
        time.sleep(1.0)
        frame = camera.grab()

    if frame is None:
        print("❌ Impossible de capturer")
        exit(1)

    print(f"✅ Frame : {frame.shape[1]}x{frame.shape[0]}")
    print(f"📁 Sortie : ./{OUTPUT_DIR}/\n")

    capture_id = 1
    detections = run_diagnostic(frame, capture_id)
    print_files(capture_id)

    if detections:
        print(f"\n  ✅  {len(detections)} DÉTECTION(S)")
    else:
        print(f"\n  ❌  AUCUNE DÉTECTION")

    print(f"\n✅ Terminé")
