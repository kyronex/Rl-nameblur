# tracker/registry.py
from __future__ import annotations
import logging
from typing import Dict, List, Optional
from core.mask import Mask, MaskState
from tracker.models import TrackerConfig

log = logging.getLogger("tracker.registry")

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

    def create(self, rect: tuple, ts: float, source: str = "slow", confidence: float = 0.0, **kwargs) -> Mask:
        uid = self._next_uid
        self._next_uid += 1
        mask = Mask(
            uid=uid,
            rect=rect,
            last_detected_rect=rect,
            last_detected_ts=ts,
            last_slow_ts=ts if source == "slow" else 0.0,
            last_source=source,
            confidence=confidence,
            confirm_after=self.cfg.confirm_after,
            lost_after_s=self.cfg.lost_after_s,
            hash_history_max=self.cfg.hash_history_max,
            state=MaskState.PENDING,
            frames_matched=1,
            last_seen_ts=ts,
            created_ts=ts,
            lost_since_ts=None,
            **kwargs,
        )
        added = self._add(mask)
        return added

    def remove(self, uid: int) -> Optional[Mask]:
        return self._masks.pop(uid, None)

    # ── mise à jour post-match ────────────────────────────
    def mark_matched(self, uid: int,ts: float, source: str = "unknown") -> None:
        mask = self._masks.get(uid)
        if mask is None:
            return
        mask.transition("matched", ts)
        if source == "slow":
            mask.last_slow_ts = ts

    # ── expiration ────────────────────────────────────────
    def tick_and_expire(self,ts: float, updated_uids: set = None) -> List[Mask]:
        """
         Cycle de vie temporel (B-04) :
          - mask non-matché depuis > lost_after_s         → état LOST
          - mask LOST depuis > expire_after_lost_s        → purge
        """
        if updated_uids is None:
            updated_uids = set()
        expired: List[Mask] = []

        lost_after_s = self.cfg.lost_after_s
        expire_after_lost_s = self.cfg.expire_after_lost_s

        for mask in list(self._masks.values()):
            is_matched_this_tick = mask.uid in updated_uids

            if not is_matched_this_tick:
                # Transition vers LOST si hors-vue depuis trop longtemps
                if mask.state in (MaskState.PENDING, MaskState.CONFIRMED):
                    if (ts - mask.last_seen_ts) >= lost_after_s:
                        mask.transition("missing", ts)  # passe en LOST, set lost_since_ts=ts

            # Purge des LOST trop vieux (qu'ils aient été ré-évalués ce tick ou non)
            if mask.state == MaskState.LOST and mask.lost_since_ts is not None:
                if (ts - mask.lost_since_ts) >= expire_after_lost_s:
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
                m.last_seen_ts,
            ),
        )
        log.warning(
            "registry: capacity full (%d/%d), evicting uid=%d state=%s last_seen=%.3f",
            len(self._masks), self.cfg.max_masks,
            worst.uid, worst.state.name, worst.last_seen_ts,
        )
        del self._masks[worst.uid]
