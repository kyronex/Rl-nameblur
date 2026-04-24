# tracker/associator.py
from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment
from core.mask import Mask
from tracker.hasher import best_hash_similarity
from tracker.models import Detection, TrackerConfig

class Associator:
    def __init__(self, config: Optional[TrackerConfig] = None):
        self.cfg = config or TrackerConfig()

    # ── poids adaptatifs ──────────────────────────────────
    def _get_weights(self, mask: Mask) -> Tuple[float, float]:
        speed = (mask.vx ** 2 + mask.vy ** 2) ** 0.5
        if speed <= self.cfg.speed_slow:
            return self.cfg.weights_static
        elif speed <= self.cfg.speed_medium:
            return self.cfg.weights_medium
        else:
            return self.cfg.weights_fast

    # ── score composite ───────────────────────────────────
    def compute_score(self, det: Detection, mask: Mask) -> float:
        w_iou, w_hash = self._get_weights(mask)
        iou = compute_iou(det.rect, mask.rect)
        if not mask.hash_history or det.phash is None:
            return iou
        hsim = compute_hash_similarity(det.phash, mask)
        return w_iou * iou + w_hash * hsim

    # ── matrice de coûts ──────────────────────────────────
    def build_cost_matrix(self, detections: List[Detection],
                          masks: List[Mask]) -> np.ndarray:
        n_det = len(detections)
        n_mask = len(masks)
        cost = np.ones((n_det, n_mask), dtype=np.float64)
        for i, det in enumerate(detections):
            for j, mask in enumerate(masks):
                cost[i, j] = 1.0 - self.compute_score(det, mask)
        return cost

    # ── assignation hongroise ─────────────────────────────
    def associate(self, detections: List[Detection],
               masks: List[Mask]) -> Tuple[
        List[Tuple[int, int]], List[int], List[int]
    ]:
        n_det = len(detections)
        n_mask = len(masks)

        if n_det == 0 and n_mask == 0:
            return [], [], []
        if n_det == 0:
            return [], [], list(range(n_mask))
        if n_mask == 0:
            return [], list(range(n_det)), []

        cost = self.build_cost_matrix(detections, masks)
        det_indices, mask_indices = linear_sum_assignment(cost)

        matches = []
        unmatched_dets = set(range(n_det))
        unmatched_masks = set(range(n_mask))

        for di, mi in zip(det_indices, mask_indices):
            score = 1.0 - cost[di, mi]
            if score >= self.cfg.score_threshold:
                matches.append((di, mi))
                unmatched_dets.discard(di)
                unmatched_masks.discard(mi)

        return matches, sorted(unmatched_dets), sorted(unmatched_masks)


def compute_iou(rect1: tuple, rect2: tuple) -> float:
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0

def compute_hash_similarity(det_hash: Optional[int], mask: Mask) -> float:
    if det_hash is None:
        return 0.0
    if len(mask.hash_history) == 0:
        return 0.0
    return best_hash_similarity(det_hash, mask.hash_history)
