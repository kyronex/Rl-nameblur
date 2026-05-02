# tracker/associator.py
from __future__ import annotations
import logging

from typing import List, Tuple, Optional
import numpy as np
import math
from scipy.optimize import linear_sum_assignment
from core.mask import Mask
from tracker.hasher import best_hash_similarity
from tracker.models import Detection, TrackerConfig, MatchScore
from tracker.motion import compute_predicted_rect

log = logging.getLogger("associator")

# Coût prohibitif mais fini — Hungarian gère mieux qu'un +inf
_GATED_COST = 1e6

class Associator:
    def __init__(self, config: Optional[TrackerConfig] = None):
        self.cfg = config or TrackerConfig()

    # ── poids & seuil indexés sur la source ──────────────
    def _get_weights(self, det: Detection) -> Tuple[float, float]:
        if det.source == "fast":
            return self.cfg.weights_source_fast
        return self.cfg.weights_source_slow

    def _get_min_score(self, det: Detection) -> float:
        if det.source == "fast":
            return self.cfg.match_score_min_fast
        return self.cfg.match_score_min_slow

    def _get_source_confidence(self, det: Detection) -> float:
        if det.source == "fast":
            return self.cfg.source_confidence_fast
        return self.cfg.source_confidence_slow

    def _continuity_factor(self, mask: Mask, ts: float) -> float:
        """exp(-dt/tau) ∈ (0, 1]. dt = âge depuis dernière détection."""
        dt = max(0.0, ts - mask.last_detected_ts)
        return math.exp(-dt / self.cfg.continuity_tau_s)

    # ── gating géométrique mask↔det ──────────────────────
    def _geo_gate_passes(self, det_rect: tuple, predicted: tuple, mask: Mask) -> bool:
        # Distance centre-à-centre
        dx = (det_rect[0] + det_rect[2] * 0.5) - (predicted[0] + predicted[2] * 0.5)
        dy = (det_rect[1] + det_rect[3] * 0.5) - (predicted[1] + predicted[3] * 0.5)
        dist = (dx * dx + dy * dy) ** 0.5
        speed = (mask.vx * mask.vx + mask.vy * mask.vy) ** 0.5
        radius = self.cfg.geo_gate_base_radius_px + self.cfg.geo_gate_velocity_k * speed * self.cfg.geo_gate_dt_ref
        passes = dist <= radius
        if not passes:
            log.info(f"  GATE FAIL mask{mask.uid} dist={dist:.0f} radius={radius:.0f} "
                    f"speed={speed:.1f} pred={predicted} det={det_rect}")
        return passes

    # ── score composite (retourne MatchScore traçable) ───
    def _compute_score(self, det: Detection, mask: Mask, predicted: tuple, ts: float) -> MatchScore:
        iou = compute_iou(det.rect, predicted)
        cont = self._continuity_factor(mask, ts)
        bmax = self.cfg.continuity_bonus_max

        if not mask.hash_history or det.phash is None:
            return MatchScore.iou_only(iou, continuity=cont, bonus_max=bmax)

        hsim = compute_hash_similarity(det.phash, mask, top_k=self.cfg.hash_top_k)
        w_iou, w_hash = self._get_weights(det)
        return MatchScore.composite(iou, hsim, w_iou, w_hash,continuity=cont, bonus_max=bmax)


    # ── matrice de coûts ──────────────────────────────────
    def build_cost_matrix(self, detections, masks, ts):
        n_det, n_mask = len(detections), len(masks)
        cost = np.full((n_det, n_mask), _GATED_COST, dtype=np.float64)
        scores: List[List[MatchScore]] = [[MatchScore.gated_score()] * n_mask for _ in range(n_det)]
        predicted_rects = [compute_predicted_rect(m, ts, self.cfg) for m in masks]

        for i, det in enumerate(detections):
            min_score = self._get_min_score(det)
            for j, mask in enumerate(masks):
                if not self._geo_gate_passes(det.rect, predicted_rects[j], mask):
                    log.debug(f"GATED det{i} vs mask{j} | det={det.rect} pred={predicted_rects[j]}")
                    continue  # scores[i][j] reste GATED_SCORE, cost reste _GATED_COST
                ms = self._compute_score(det, mask, predicted_rects[j], ts)
                log.info(f"SCORE det{i} vs mask{j} = {ms.total:.3f} (min={self._get_min_score(det):.3f})")
                if ms.total < min_score:
                    continue
                scores[i][j] = ms
                cost[i, j] = 1.0 - ms.total
        return cost, scores


    # ── assignation hongroise ─────────────────────────────
    def associate(self,detections: List[Detection],masks: List[Mask],ts) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        n_det = len(detections)
        n_mask = len(masks)

        if n_det == 0 and n_mask == 0:
            return [], [], []
        if n_det == 0:
            return [], [], list(range(n_mask))
        if n_mask == 0:
            return [], list(range(n_det)), []

        cost, scores = self.build_cost_matrix(detections, masks, ts)
        det_indices, mask_indices = linear_sum_assignment(cost)

        matches: List[Tuple[int, int]] = []
        unmatched_dets = set(range(n_det))
        unmatched_masks = set(range(n_mask))

        for di, mi in zip(det_indices, mask_indices):
            ms = scores[di][mi]
            if ms.gated:
                continue
            min_score = self._get_min_score(detections[di])
            if ms.total >= min_score:
                detections[di].scores["match"] = ms
                detections[di].scores["source_confidence"] = self._get_source_confidence(detections[di])
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

def compute_hash_similarity(det_hash: Optional[int], mask: Mask, top_k: Optional[int] = None) -> float:
    if det_hash is None:
        return 0.0
    if len(mask.hash_history) == 0:
        return 0.0
    return best_hash_similarity(det_hash, mask.hash_history, top_k=top_k)
