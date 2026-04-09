# detect_tools_mask.py
import cv2
import numpy as np
import time
import logging
from config import cfg
from detect_stats import flush_local
from detect_tools import write_circles , write_rects , get_color

log = logging.getLogger("detect_tools_mask")

# ── detect_ball_zones not used ──
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

def saturation_variance_mask(frame, scale, kernel_size=(24, 11), threshold1=10000, threshold2=28500):
    h_orig, w_orig = frame.shape[:2]

    # Toujours travailler à ~0.125 de l'original pour la perf
    target_effective = 0.125
    internal_factor = target_effective * scale

    if internal_factor < 0.85:
        small = cv2.resize(frame, None, fx=internal_factor, fy=internal_factor,interpolation=cv2.INTER_NEAREST)
    else:
        small = frame
        internal_factor = 1.0

    # Kernels calibrés pour résolution 0.25x (= l'ancien resize 0.5x interne)
    kw, kh = kernel_size
    kw_s = max(kw // 2 | 1, 1)   # 11
    kh_s = max(kh // 2 | 1, 1)   # 5

    channels = cv2.split(small)
    ch0 = channels[0].astype(np.float32)
    ch1 = channels[1].astype(np.float32)
    ch2 = channels[2].astype(np.float32)

    mean0 = cv2.blur(ch0, (kw_s, kh_s))
    mean1 = cv2.blur(ch1, (kw_s, kh_s))
    mean2 = cv2.blur(ch2, (kw_s, kh_s))

    var0 = cv2.sqrBoxFilter(ch0, -1, (kw_s, kh_s), normalize=True) - mean0 * mean0
    var1 = cv2.sqrBoxFilter(ch1, -1, (kw_s, kh_s), normalize=True) - mean1 * mean1
    var2 = cv2.sqrBoxFilter(ch2, -1, (kw_s, kh_s), normalize=True) - mean2 * mean2

    var_total = var0 + var1 + var2

    hsv_s = cv2.max(cv2.max(ch0, ch1), ch2) - cv2.min(cv2.min(ch0, ch1), ch2)
    mean_s = cv2.blur(hsv_s, (kw_s, kh_s))
    var_s = cv2.sqrBoxFilter(hsv_s, -1, (kw_s, kh_s), normalize=True) - mean_s * mean_s

    combined = ((var_s > threshold1 / 4) & (var_total > threshold2 / 4)).astype(np.uint8) * 255

    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    filtered = cv2.dilate(combined, kernel_v, iterations=1)

    if internal_factor < 0.85:
        filtered = cv2.resize(filtered, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    return filtered

# ── saturation_variance_mask_old not used ──
def saturation_variance_mask_old(frame, kernel_size=(24, 11), threshold1=12000, threshold2=30000):

    h_orig, w_orig = frame.shape[:2]
    small = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

    kw, kh = kernel_size
    # Adapter le kernel à la résolution réduite
    kw_s = max(kw // 2 | 1, 1)  # garder impair
    kh_s = max(kh // 2 | 1, 1)

    channels = cv2.split(small)
    ch0 = channels[0].astype(np.float32)
    ch1 = channels[1].astype(np.float32)
    ch2 = channels[2].astype(np.float32)

    mean0 = cv2.blur(ch0, (kw_s, kh_s))
    mean1 = cv2.blur(ch1, (kw_s, kh_s))
    mean2 = cv2.blur(ch2, (kw_s, kh_s))

    var0 = cv2.sqrBoxFilter(ch0, -1, (kw_s, kh_s), normalize=True) - mean0 * mean0
    var1 = cv2.sqrBoxFilter(ch1, -1, (kw_s, kh_s), normalize=True) - mean1 * mean1
    var2 = cv2.sqrBoxFilter(ch2, -1, (kw_s, kh_s), normalize=True) - mean2 * mean2

    var_total = var0 + var1 + var2

    hsv_s = cv2.max(cv2.max(ch0, ch1), ch2) - cv2.min(cv2.min(ch0, ch1), ch2)
    mean_s = cv2.blur(hsv_s, (kw_s, kh_s))
    var_s = cv2.sqrBoxFilter(hsv_s, -1, (kw_s, kh_s), normalize=True) - mean_s * mean_s

    # Seuils divisés par 4 car variance scale avec résolution²
    combined = ((var_s > threshold1 / 4) & (var_total > threshold2 / 4)).astype(np.uint8) * 255

    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))  # kernel réduit aussi
    filtered = cv2.dilate(combined, kernel_v, iterations=1)

    # Remonter à la résolution originale
    filtered = cv2.resize(filtered, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

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
    sobel_abs = cv2.bitwise_and(sobel_abs, white_clean)
    # Branche v1 : dilate → erode
    sobel_dilated = cv2.dilate(sobel_abs, kernels["sobel_spread"], iterations=1)
    interior_v1 = cv2.erode(sobel_dilated, kernels["sobel_erode"], iterations=2)
    # Branche v2 : threshold → erode
    sobel_bin = cv2.threshold(sobel_abs, 0, 255, cv2.THRESH_BINARY)[1]
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
