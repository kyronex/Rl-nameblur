# hsv_picker.py — Cliquer sur les plaques pour lire le HSV

import cv2
import numpy as np
import dxcam

camera = dxcam.create()
frame = camera.grab()

if frame is None:
    print("Pas de frame")
    exit()

frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Moyenne 5x5 autour du pixel
        zone = hsv[max(0,y-2):y+3, max(0,x-2):x+3]
        h_mean = int(np.mean(zone[:,:,0]))
        s_mean = int(np.mean(zone[:,:,1]))
        v_mean = int(np.mean(zone[:,:,2]))
        print(f"Pixel ({x},{y}) → H={h_mean}  S={s_mean}  V={v_mean}")

cv2.namedWindow("HSV Picker")
cv2.setMouseCallback("HSV Picker", on_click)
cv2.imshow("HSV Picker", frame_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
