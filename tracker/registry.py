# tracker/registry.py
from __future__ import annotations
from typing import Dict, List, Optional
from core.mask import Mask, MaskState


class MaskRegistry:
    def __init__(self, max_masks: int = 20, ttl_default: int = 5):
        self._masks: Dict[int, Mask] = {}
        self._next_uid: int = 0
        self.max_masks = max_masks
        self.ttl_default = ttl_default

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
    def add(self, mask: Mask) -> Mask:
        if mask.uid in self._masks:
            raise ValueError(f"uid {mask.uid} déjà présent")
        if len(self._masks) >= self.max_masks:
            self._evict_one()
        self._masks[mask.uid] = mask
        return mask

    def create(self, rect: tuple, ts: float, source: str = "slow",
               confidence: float = 0.0, **kwargs) -> Mask:
        uid = self._next_uid
        self._next_uid += 1
        mask = Mask(
            uid=uid,
            rect=rect,
            last_detected_rect=rect,
            last_detected_ts=ts,
            last_source=source,
            ttl=self.ttl_default,
            confidence=confidence,
            **kwargs,
        )
        return self.add(mask)

    def remove(self, uid: int) -> Optional[Mask]:
        return self._masks.pop(uid, None)

    # ── expiration ────────────────────────────────────────
    def tick_and_expire(self) -> List[Mask]:
        expired = []
        for mask in list(self._masks.values()):
            mask.ttl -= 1
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
        del self._masks[worst.uid]
