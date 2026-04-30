# core/optical_flow.py
import cv2
import numpy as np
from bench import bench

# ── Paramètres Lucas-Kanade ──
_LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

_ROI_PAD = 40

def _rect_to_points(rect):
    x, y, w, h = rect
    pts = np.array([
        [x,     y    ],
        [x + w, y    ],
        [x,     y + h],
        [x + w, y + h],
    ], dtype=np.float32).reshape(-1, 1, 2)
    return pts

def of_track(prev_gray, curr_gray, rect):
    """
    Tente de suivre rect via Lucas-Kanade entre prev_gray et curr_gray.
    Travaille sur un crop local pour la performance.

    Sondes bench :
      - of_lk_call : durée du seul cv2.calcOpticalFlowPyrLK
                     (isole le coût LK pur du crop/median).
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
        return rect, False

    pts = _rect_to_points((x - cx0, y - cy0, w, h))

    with bench.timer("of_lk_call"):
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_crop, curr_crop, pts, None, **_LK_PARAMS
        )

    good = status.flatten() == 1

    if good.sum() < 2:
        return rect, False

    delta = np.median(new_pts[good] - pts[good], axis=0).flatten()
    dx, dy = float(delta[0]), float(delta[1])

    new_rect = (int(round(x + dx)), int(round(y + dy)), w, h)
    return new_rect, True
