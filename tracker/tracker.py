# tracker/tracker.py
"""
Orchestrateur principal du tracking.
"""

import time
import logging
import numpy as np
from tracker.models import TrackerConfig, Detection
from tracker.registry import MaskRegistry
from tracker.associator import Associator
from tracker.motion import apply_detection, predict_position
from tracker.hasher import compute_phash
from core.mask import MaskState

log = logging.getLogger("tracker")


class Tracker:

    def __init__(self, config: TrackerConfig = None):
        self.cfg = config or TrackerConfig()
        self.registry = MaskRegistry(
            max_masks=self.cfg.max_masks,
            ttl_default=self.cfg.ttl_max,
        )
        self.associator = Associator(self.cfg)

    # ───────────────────────────────────────────────
    #  API PUBLIQUE
    # ───────────────────────────────────────────────

    def update(self, frame: np.ndarray, detections: list, ts: float = None) -> list:
        """
        Args:
            frame:      np.ndarray (h, w, 3)
            detections: list[dict] avec 'rect', 'source', 'confidence'
            ts:         float (default: time.time())
        Returns:
            list[Mask] — CONFIRMED uniquement
        """
        if ts is None:
            ts = time.time()

        # ── 1. Construire Detection objects avec phash ──
        det_objects = []
        for d in detections:
            x, y, w, h = d["rect"]
            crop = frame[y:y+h, x:x+w]
            phash = compute_phash(crop)
            det_objects.append(Detection(
                rect=d["rect"],
                phash=phash,
                source=d.get("source", "slow"),
            ))

        # ── 2. Masks actifs ──
        active = self.registry.masks

        # ── 3. Association ──
        matches, unmatched_dets, unmatched_masks = self.associator.associate(
            det_objects, active
        )

        # ── 4. Mettre à jour les matchés ──
        matched_uids = set()
        for det_idx, mask_idx in matches:
            mask = active[mask_idx]
            det = det_objects[det_idx]
            apply_detection(mask, det.rect, ts, self.cfg)
            if det.phash is not None:
                mask.hash_history.append(det.phash)
                if len(mask.hash_history) > 5:
                    mask.hash_history.pop(0)
            self.registry.mark_matched(mask.uid)
            matched_uids.add(mask.uid)
            log.debug(f"uid={mask.uid} matched det={det_idx}")

        # ── 5. Prédiction inertielle pour masks non-matchés ──
        for mask_idx in unmatched_masks:
            mask = active[mask_idx]
            predict_position(mask, ts, self.cfg)

        # ── 6. Créer nouveaux masks pour détections non-matchées ──
        for det_idx in unmatched_dets:
            det = det_objects[det_idx]
            mask = self.registry.create(det.rect, ts, source=det.source)
            if det.phash is not None:
                mask.hash_history.append(det.phash)
            log.debug(f"uid={mask.uid} created at {det.rect}")

        # ── 7. Tick TTL + expirer les LOST ──
        expired = self.registry.tick_and_expire(matched_uids)
        for m in expired:
            log.debug(f"uid={m.uid} expired")

        # ── 8. Retourner CONFIRMED ──
        return [m for m in self.registry.masks
                if m.state == MaskState.CONFIRMED]

    # ───────────────────────────────────────────────
    #  ACCESSEURS DEBUG
    # ───────────────────────────────────────────────

    def all_masks(self):
        return self.registry.masks

    def stats(self):
        all_m = self.registry.masks
        return {
            "total": len(all_m),
            "pending": sum(1 for m in all_m if m.state == MaskState.PENDING),
            "confirmed": sum(1 for m in all_m if m.state == MaskState.CONFIRMED),
            "lost": sum(1 for m in all_m if m.state == MaskState.LOST),
        }
