# tracker/registry.py
from __future__ import annotations
import logging
from typing import Dict, List, Optional
from core.mask import Mask, MaskState
from tracker.models import TrackerConfig

log = logging.getLogger("registry")

class MaskRegistry:
    def __init__(self, config: TrackerConfig):
        self._masks: Dict[int, Mask] = {}
        self._next_uid: int = 0
        self.cfg = config

    # ── accès ─────────────────────────────────────────────
    @property
    def masks(self) -> List[Mask]:
        return list(self._masks.values())

    def get(self, uid: int) -> Optional[Mask]:
        return self._masks.get(uid)

    def __len__(self) -> int:
        return len(self._masks)

    def __contains__(self, uid: int) -> bool:
        return uid in self._masks

    # ── CRUD ──────────────────────────────────────────────
    def _add(self, mask: Mask) -> Mask:
        if mask.uid in self._masks:
            raise ValueError(f"uid {mask.uid} déjà présent")
        if len(self._masks) >= self.cfg.max_masks:
            self._evict_one()
        self._masks[mask.uid] = mask
        return mask

    def create(self, rect: tuple, ts: float, source: str = "slow",confidence: float = 0.0, **kwargs) -> Mask:
        uid = self._next_uid
        self._next_uid += 1
        mask = Mask(
            uid=uid,
            rect=rect,
            last_detected_rect=rect,
            last_detected_ts=ts,
            last_slow_ts=ts if source == "slow" else 0.0,
            last_source=source,
            ttl=self.cfg.ttl_default,
            confidence=confidence,
            confirm_after=self.cfg.confirm_after,
            lost_after=self.cfg.lost_after,
            hash_history_max=self.cfg.hash_history_max,
            state=MaskState.PENDING,
            frames_matched=1,   # ← la détection qui crée le mask COMPTE comme 1er match
            frames_missing=0,
            **kwargs,
        )
        return self._add(mask)

    def remove(self, uid: int) -> Optional[Mask]:
        return self._masks.pop(uid, None)

    # ── mise à jour post-match ────────────────────────────
    def mark_matched(self, uid: int) -> None:
        mask = self._masks.get(uid)
        if mask:
            mask.transition("matched")
            mask.ttl = self.cfg.ttl_default

    # ── expiration ────────────────────────────────────────
    def tick_and_expire(self, updated_uids: set = None) -> List[Mask]:
        """
        Invariant cycle de vie :
        - mask matched   : ttl reset + transition("matched") via mark_matched()
        - mask non matched : ttl-- + transition("missing")
        - Suppression ssi (ttl <= 0 AND state == LOST)

        Grâce à ttl_default >= lost_after (validé au chargement config),
        un mask atteint toujours LOST avant d'être supprimé.
        """
        if updated_uids is None:
            updated_uids = set()
        expired = []
        for mask in list(self._masks.values()):
            if mask.uid in updated_uids:
                continue
            mask.ttl -= 1
            mask.transition("missing")
            if mask.ttl <= 0 and mask.state == MaskState.LOST:
                expired.append(mask)
                del self._masks[mask.uid]
        return expired

    # ── interne ───────────────────────────────────────────
    def _evict_one(self) -> None:
        if not self._masks:
            return
        worst = min(
            self._masks.values(),
            key=lambda m: (
                0 if m.state == MaskState.LOST else
                1 if m.state == MaskState.PENDING else 2,
                m.ttl,
            ),
        )
        log.warning(
            "registry: capacity full (%d/%d), evicting uid=%d state=%s ttl=%d",
            len(self._masks), self.cfg.max_masks,
            worst.uid, worst.state.name, worst.ttl,
        )
        del self._masks[worst.uid]
