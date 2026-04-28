# tracker/tracker.py
"""
Orchestrateur principal du tracking.
"""

import time
import logging
import numpy as np
from config import cfg as _global_cfg
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
        self.registry = MaskRegistry(self.cfg)
        self.associator = Associator(self.cfg)
        self._cfg_version = _global_cfg.version

    # ───────────────────────────────────────────────
    #  API PUBLIQUE
    # ───────────────────────────────────────────────

    def apply_detections(self, frame: np.ndarray, detections: list, ts: float = None, source: str = "slow") -> set:
        """
        Applique une vague de détections (slow OU fast).
        Returns: set[uid] des masks mis à jour.
        """
        if ts is None:
            ts = time.perf_counter()
        H, W = frame.shape[:2]

        # ── 1. Detection objects + phash ──
        skip_phash = (source == "fast")

        det_objects = []
        for d in detections:
            x, y, w, h = (int(v) for v in d.rect)

            # 3. Clamp aux bornes du frame (sécurité anti-crash)
            x = max(0, min(x, W - 1))    # x dans [0, W-1]
            y = max(0, min(y, H - 1))    # y dans [0, H-1]
            w = max(1, min(w, W - x))    # w >= 1, et x+w <= W
            h = max(1, min(h, H - y))

            clamped_rect = (x, y, w, h)
            if skip_phash:
                phash = None
            else:
                crop = frame[y:y+h, x:x+w]
                phash = compute_phash(crop)

            det_objects.append(Detection(
                rect=clamped_rect,
                phash=phash,
                source=d.source or source,
                confidence=d.confidence,
                template=d.template,
                scores=d.scores,
            ))

        active = self.registry.masks

        # ── 2. Association ──
        matches, unmatched_dets, unmatched_masks = self.associator.associate(det_objects, active, ts)

        # ── 3. Matchés : maj motion + phash + champs métier ──
        matched_uids = set()
        for det_idx, mask_idx in matches:
            mask = active[mask_idx]
            det = det_objects[det_idx]
            apply_detection(mask, det.rect, ts, det.source, self.cfg)
            if det.phash is not None:
                mask.hash_history.append(det.phash)
            mask.confidence = det.confidence
            if det.template is not None:
                mask.template = det.template
            if det.scores:
                mask.scores = det.scores
            self.registry.mark_matched(mask.uid)
            matched_uids.add(mask.uid)

        # ── 4. Nouveaux masks (slow uniquement — fast ne crée pas) ──
        if source == "slow":
            for det_idx in unmatched_dets:
                det = det_objects[det_idx]
                mask = self.registry.create(det.rect, ts, source=det.source)
                if det.phash is not None:
                    mask.hash_history.append(det.phash)
                mask.confidence = det.confidence
                mask.template = det.template
                mask.scores = det.scores
                matched_uids.add(mask.uid)

        return matched_uids

    def tick(self, ts: float = None, updated_uids: set = None) -> list:
        """
        À appeler chaque frame : predict inertiel + TTL + purge.
        Returns: list[Mask] CONFIRMED.
        """
        if ts is None:
            ts = time.perf_counter()
        if updated_uids is None:
            updated_uids = set()

        # predict pour les non-matchés cette frame
        for mask in self.registry.masks:
            if mask.uid not in updated_uids:
                predict_position(mask, ts, self.cfg.screen_w, self.cfg.screen_h, self.cfg)

        self.registry.tick_and_expire(updated_uids)

        return [m for m in self.registry.masks if m.state == MaskState.CONFIRMED]

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

    # ───────────────────────────────────────────────
    #  HOT-RELOAD
    # ───────────────────────────────────────────────

    def reload_config(self, new_config: TrackerConfig) -> None:
        """
        Propage un nouveau snapshot de config à tous les sous-composants.
        À appeler entre deux frames (jamais pendant apply_detections/tick)
        depuis le thread main, quand cfg.version a changé.
        """
        self.cfg = new_config
        self.registry.cfg = new_config
        self.associator.cfg = new_config
        log.info("[Tracker] Config rechargée")

    def maybe_reload(self) -> bool:
        """
        Vérifie la version du singleton `cfg` global et recharge si changée.
        À appeler en début de chaque frame — coût steady-state : 1 comparaison int.
        Returns: True si reload effectué, False sinon.
        """
        current = _global_cfg.version
        if current == self._cfg_version:
            return False
        old_version = self._cfg_version
        try:
            self.reload_config(TrackerConfig())
        except Exception as e:
            log.error(f"[Tracker] Échec reload v{old_version}→v{current}: {e}")
            return False
        self._cfg_version = current
        log.info(f"[Tracker] Hot-reload v{old_version}→v{current}")
        return True