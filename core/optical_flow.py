# core/optical_flow.py
import cv2
import numpy as np

# ── Paramètres Lucas-Kanade ──
_LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

_ROI_PAD = 40

_GRID_N = 7  # nb de points par axe (A-1, calage)
_FB_MAX_ERR = 0.25  # seuil d'erreur aller-retour en px (A-2, calage)
_SCALE_CLAMP = 0.10  # variation d'échelle max par frame (S-b, calage)

def _rect_to_points(rect):
    x, y, w, h = rect
    xs = np.linspace(x, x + w, _GRID_N)
    ys = np.linspace(y, y + h, _GRID_N)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32).reshape(-1, 1, 2)
    return pts

def of_track(prev_gray, curr_gray, rect):
    """
    Tente de suivre rect via Lucas-Kanade entre prev_gray et curr_gray.
    Travaille sur un crop local pour la performance.
    """
    x, y, w, h = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
    img_h, img_w = prev_gray.shape[:2]

    cx0 = max(x - _ROI_PAD, 0)
    cy0 = max(y - _ROI_PAD, 0)
    cx1 = min(x + w + _ROI_PAD, img_w)
    cy1 = min(y + h + _ROI_PAD, img_h)

    prev_crop = prev_gray[cy0:cy1, cx0:cx1]
    curr_crop = curr_gray[cy0:cy1, cx0:cx1]

    if prev_crop.size == 0 or curr_crop.size == 0:
        return rect, False , 0.0 , 0.0, None

    pts = _rect_to_points((x - cx0, y - cy0, w, h))
    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_crop, curr_crop, pts, None, **_LK_PARAMS)
    back_pts, status_b, _ = cv2.calcOpticalFlowPyrLK(curr_crop, prev_crop, new_pts, None, **_LK_PARAMS)

    fb_err = np.linalg.norm(pts - back_pts, axis=2).flatten()
    good = (status.flatten() == 1) & (status_b.flatten() == 1) & (fb_err < _FB_MAX_ERR)
    if good.sum() < 2:
        return rect, False , 0.0 , 0.0, None
    delta = np.median(new_pts[good] - pts[good], axis=0).flatten()
    dx, dy = float(delta[0]), float(delta[1])

    # ── S-b : facteur d'échelle robuste par ratio de dispersions ──
    p0 = pts[good].reshape(-1, 2)
    p1 = new_pts[good].reshape(-1, 2)
    spread0 = np.median(np.abs(p0 - np.median(p0, axis=0)))
    spread1 = np.median(np.abs(p1 - np.median(p1, axis=0)))
    if spread0 > 1e-3:
        scale = spread1 / spread0
        scale = max(1.0 - _SCALE_CLAMP, min(scale, 1.0 + _SCALE_CLAMP))
    else:
        scale = 1.0

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    cx = x + w / 2.0 + dx
    cy = y + h / 2.0 + dy
    new_rect = (int(round(cx - new_w / 2.0)), int(round(cy - new_h / 2.0)), new_w, new_h)

    of_stats = {
        "good_sum": int(good.sum()),
        "good_pct": float(good.sum()) / float(pts.shape[0]),
        "fb_err_med": float(np.median(fb_err[good])),
        "scale_raw": float(scale),
    }
    return new_rect, True , dx, dy, of_stats
