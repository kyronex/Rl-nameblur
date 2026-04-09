# test_debug.py
import dxcam
import cv2
import numpy as np
import time
import os
from datetime import datetime

from detect import (
    ORANGE_LOW, ORANGE_HIGH,
    BLUE_LOW, BLUE_HIGH,
    detect_plates
)

# ── Créer le dossier test/ ──
os.makedirs("test", exist_ok=True)

# ── Horodatage unique ──
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Capture une frame
cam = dxcam.create()
cam.start(target_fps=60)
time.sleep(1)

frame = cam.get_latest_frame()
cam.stop()

if frame is None:
    print("❌ Pas de frame capturée")
    exit()

frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# ── 1. Capture brute ──
cv2.imwrite(f"test/debug_1_capture_{timestamp}.png", frame_bgr)
print(f"✅ test/debug_1_capture_{timestamp}.png")

# ── 2. Masques HSV ──
hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

mask_orange = cv2.inRange(hsv, ORANGE_LOW, ORANGE_HIGH)
mask_blue   = cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)
mask_total  = cv2.bitwise_or(mask_orange, mask_blue)

cv2.imwrite(f"test/debug_2_masque_orange_{timestamp}.png", mask_orange)
cv2.imwrite(f"test/debug_2_masque_bleu_{timestamp}.png", mask_blue)
cv2.imwrite(f"test/debug_2_masque_total_{timestamp}.png", mask_total)
print(f"✅ test/debug_2_masque_*_{timestamp}.png")

# ── 3. Détections ──
plates = detect_plates(frame_bgr)
print(f"\n📊 {len(plates)} plaque(s) détectée(s)")

for i, (x, y, w, h) in enumerate(plates):
    cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
    print(f"   Plaque {i+1} : x={x} y={y} w={w} h={h}")

cv2.imwrite(f"test/debug_3_detections_{timestamp}.png", frame_bgr)
print(f"✅ test/debug_3_detections_{timestamp}.png")

print(f"\n📁 Fichiers dans test/ avec horodatage : {timestamp}")