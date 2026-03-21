# optical_flow.py
import cv2
import numpy as np

# ── Paramètres Lucas-Kanade ──
_LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

_ROI_PAD = 40


def rect_to_points(rect):
    """
    Convertit (x, y, w, h) en 4 coins float32 pour Lucas-Kanade.
    Retourne shape (4, 1, 2).
    """
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

    Paramètres
    ----------
    prev_gray : np.ndarray  — frame précédente en niveaux de gris
    curr_gray : np.ndarray  — frame courante  en niveaux de gris
    rect      : tuple(x, y, w, h)

    Retourne
    --------
    (new_rect, True)  — OF réussi, new_rect est la nouvelle position
    (rect,     False) — OF échoué, rect inchangé (Option B)
    """
    x, y, w, h = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
    img_h, img_w = prev_gray.shape[:2]

    # ── Crop local avec marge ──
    cx0 = max(x - _ROI_PAD, 0)
    cy0 = max(y - _ROI_PAD, 0)
    cx1 = min(x + w + _ROI_PAD, img_w)
    cy1 = min(y + h + _ROI_PAD, img_h)

    prev_crop = prev_gray[cy0:cy1, cx0:cx1]
    curr_crop = curr_gray[cy0:cy1, cx0:cx1]

    if prev_crop.size == 0 or curr_crop.size == 0:
        return rect, False

    # ── Points en coordonnées locales ──
    pts = rect_to_points((x - cx0, y - cy0, w, h))

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
