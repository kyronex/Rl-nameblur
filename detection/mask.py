# mask.py
import cv2
import numpy as np
import logging
from config import cfg
from detection.tools import write_circles , write_rects , get_color

log = logging.getLogger("mask")
_kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))

def saturation_variance_mask(frame, scale, kernel_size=(24, 11), threshold1=10000, threshold2=28500):
    h_orig, w_orig = frame.shape[:2]
    internal_factor = 0.125 * scale

    if internal_factor < 0.85:
        small = cv2.resize(frame, None, fx=internal_factor, fy=internal_factor, interpolation=cv2.INTER_NEAREST)
    else:
        small = frame
        internal_factor = 1.0

    kw, kh = kernel_size
    ks = (max(kw // 2 | 1, 1), max(kh // 2 | 1, 1))

    # Evite split + 3 astype : merge dans un seul array float32
    small_f = small.astype(np.float32)
    ch0, ch1, ch2 = small_f[:,:,0], small_f[:,:,1], small_f[:,:,2]

    # Contiguous pour blur/sqrBoxFilter
    ch0, ch1, ch2 = np.ascontiguousarray(ch0), np.ascontiguousarray(ch1), np.ascontiguousarray(ch2)

    m0 = cv2.blur(ch0, ks)
    m1 = cv2.blur(ch1, ks)
    m2 = cv2.blur(ch2, ks)

    # Variance totale — dst= pour réutiliser buffers
    var_total = cv2.sqrBoxFilter(ch0, cv2.CV_32F, ks, normalize=True)
    cv2.subtract(var_total, m0 * m0, dst=var_total)
    tmp = cv2.sqrBoxFilter(ch1, cv2.CV_32F, ks, normalize=True)
    cv2.subtract(tmp, m1 * m1, dst=tmp)
    cv2.add(var_total, tmp, dst=var_total)
    cv2.sqrBoxFilter(ch2, cv2.CV_32F, ks, normalize=True, dst=tmp)
    cv2.subtract(tmp, m2 * m2, dst=tmp)
    cv2.add(var_total, tmp, dst=var_total)

    # Saturation = max - min (réutilise tmp)
    cv2.max(ch0, ch1, dst=tmp)
    cv2.max(tmp, ch2, dst=tmp)
    hsv_s = tmp  # alias
    buf = cv2.min(ch0, ch1)
    cv2.min(buf, ch2, dst=buf)
    cv2.subtract(hsv_s, buf, dst=hsv_s)

    mean_s = cv2.blur(hsv_s, ks)
    cv2.sqrBoxFilter(hsv_s, cv2.CV_32F, ks, normalize=True, dst=buf)
    cv2.subtract(buf, mean_s * mean_s, dst=buf)  # buf = var_s

    # Threshold combiné — évite array bool intermédiaire
    t1 = threshold1 / 4
    t2 = threshold2 / 4
    mask1 = cv2.compare(buf, t1, cv2.CMP_GT)
    mask2 = cv2.compare(var_total, t2, cv2.CMP_GT)
    cv2.bitwise_and(mask1, mask2, dst=mask1)

    cv2.dilate(mask1, _kernel_v, iterations=1, dst=mask1)

    if internal_factor < 0.85:
        mask1 = cv2.resize(mask1, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    return mask1

def compute_white_mask(gray, kernels, letter_connect_iter):
    _, mask_white = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)

    white_dilated = cv2.dilate(mask_white, kernels["letter_connect"], iterations=letter_connect_iter)
    white_horiz = cv2.morphologyEx(white_dilated, cv2.MORPH_OPEN, kernels["noise_filter"])
    white_clean = cv2.erode(white_horiz, kernels["line_split"], iterations=1)
    return mask_white, white_clean

def compute_sobel_interior_unified(gray, combined, kernels):
    """Une seule branche qui capture ce que v1 OR v2 capturait."""
    sobel_abs = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    cv2.bitwise_and(sobel_abs, combined, dst=sobel_abs)
    cv2.threshold(sobel_abs, 40, 255, cv2.THRESH_BINARY, dst=sobel_abs)  # in-place
    cv2.erode(sobel_abs, kernels["sobel_erode"], iterations=1, dst=sobel_abs)
    return sobel_abs


def refine_and_merge(combined, interior, kernels):
    """Une seule branche subtract → dilate → erode → close."""
    refined = cv2.subtract(combined, interior)
    cv2.dilate(refined, kernels["fragment_rejoin"], iterations=2, dst=refined)
    cv2.erode(refined, kernels["final_split"], iterations=1, dst=refined)
    cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernels["gap_fill"], dst=refined)
    return refined
