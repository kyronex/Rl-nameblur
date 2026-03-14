# detect_tools_mask.py
import cv2
import numpy as np
import time
import logging
from config import cfg
from detect_stats import flush_local
from detect_tools import write_circles , write_rects , get_color

log = logging.getLogger("detect_tools_mask")

# ── detect_ball_zones ── non fonctionnel
def detect_ball_zones(frame, min_r=5, max_r=20):
    """Retourne une liste de (cx, cy, radius) des zones sphériques."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0]
    Cr = ycrcb[:, :, 1]

    blurred = cv2.GaussianBlur(Y, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT,dp=1.5, minDist=80, param1=120, param2=35,minRadius=min_r, maxRadius=max_r)

    zones = []
    if circles is not None:
        for (cx, cy, r) in circles[0]:
            cx, cy, r = int(cx), int(cy), int(r)
            # Valider avec Cr (balle orange → Cr élevé)
            mask = np.zeros(Cr.shape, dtype=np.uint8)
            cv2.circle(mask, (cx, cy), r, 255, -1)
            mean_cr = cv2.mean(Cr, mask=mask)[0]
            if mean_cr > 140:  # balle orange/jaune
                zones.append((cx, cy, r))

    write_circles(frame, zones, get_color("magenta") , 2)
    cv2.imshow("frame", frame)
    cv2.imshow("hsv", hsv)
    cv2.waitKey(0)
    return zones

def saturation_variance_mask(frame, kernel_size=(31, 11), threshold1=3300 , threshold2=8800):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    kw, kh = kernel_size
    mean_s = cv2.blur(s, (kw, kh))
    mean_s2 = cv2.blur(s * s, (kw, kh))
    variance = mean_s2 - mean_s * mean_s
    # Seuillage
    mask = (variance > threshold1).astype(np.uint8) * 255

    frame_f = frame.astype(np.float32)
    kw2, kh2 = 31, 11
    var_total = np.zeros(frame_f.shape[:2], dtype=np.float32)
    for c in range(3):
        ch = frame_f[:, :, c]
        mean_c = cv2.blur(ch, (kw2, kh2))
        var_c = cv2.blur(ch * ch, (kw2, kh2)) - mean_c * mean_c
        var_total += var_c

    uniform_mask = (var_total > threshold2).astype(np.uint8) * 255

    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    uniform_mask_d = cv2.dilate(uniform_mask, kernel_v, iterations=1)
    mask_d = cv2.dilate(mask, kernel_v, iterations=1)
    filtered = cv2.bitwise_and(mask_d, uniform_mask_d)

    return filtered

def compute_white_mask(gray, kernels, letter_connect_iter):
    _, mask_white = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)

    white_dilated = cv2.dilate(mask_white, kernels["letter_connect"], iterations=letter_connect_iter)
    white_horiz = cv2.morphologyEx(white_dilated, cv2.MORPH_OPEN, kernels["noise_filter"])
    white_clean = cv2.erode(white_horiz, kernels["line_split"], iterations=1)
    return mask_white, white_clean

def compute_sobel_interiors(gray, white_clean, kernels):
    sobel_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    sobel_abs = cv2.convertScaleAbs(sobel_x)
    sobel_abs[white_clean == 0] = 0

    sobel_bin = cv2.threshold(sobel_abs, 0, 255, cv2.THRESH_BINARY)[1]
    sobel_dilated = cv2.dilate(sobel_abs, kernels["sobel_spread"], iterations=1)

    interior_v1 = cv2.erode(sobel_dilated, kernels["sobel_erode"], iterations=2)
    interior_v2 = cv2.erode(sobel_bin, kernels["sobel_erode"], iterations=2)
    return interior_v1, interior_v2

def refine_and_merge(white_clean, interior_v1, interior_v2, kernels):
    white_refined_v1 = cv2.subtract(white_clean, interior_v1)
    white_refined_v2 = cv2.subtract(white_clean, interior_v2)

    white_reconnected_v1 = cv2.dilate(white_refined_v1, kernels["fragment_rejoin"], iterations=3)
    white_reconnected_v2 = cv2.dilate(white_refined_v2, kernels["fragment_rejoin"], iterations=3)

    cv2.bitwise_or(white_reconnected_v1, white_reconnected_v2, dst=white_reconnected_v1)
    white_final = cv2.erode(white_reconnected_v1, kernels["final_split"], iterations=1)
    closed = cv2.morphologyEx(white_final, cv2.MORPH_CLOSE, kernels["gap_fill"])
    return closed

