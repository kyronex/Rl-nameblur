# core/tracker/registry.py

from __future__ import annotations
import time
from core.mask import Mask , MaskState


class Registry:
    """Registre centralisé des masks actifs."""

    def __init__(self, ttl: int = 30, fast_miss_limit: int = 4, lost_timeout: int = 15):
        self._masks: dict[int, Mask] = {}
        self._next_uid: int = 0
        self._ttl = ttl
        self._fast_miss_limit = fast_miss_limit
        self._lost_timeout = lost_timeout

    # ── Création ─────────────────────────────────────────────
    def create_mask(self, rect: tuple, confidence: float = 0.0,
                    template=None, source: str = "new") -> Mask:
        uid = self._next_uid
        self._next_uid += 1
        now = time.time()
        mask = Mask(
            uid=uid,
            rect=rect,
            last_detected_rect=rect,
            last_detected_ts=now,
            last_source=source,
            ttl=self._ttl,
            confidence=confidence,
            template=template,
        )
        self._masks[uid] = mask
        return mask

    # ── Accès ────────────────────────────────────────────────
    def get_mask(self, uid: int) -> Mask | None:
        return self._masks.get(uid)

    def active_masks(self) -> list[Mask]:
        """PENDING + CONFIRMED (pas LOST)."""
        return [m for m in self._masks.values() if m.state != MaskState.LOST]

    def all_masks(self) -> list[Mask]:
        return list(self._masks.values())

    def confirmed_masks(self) -> list[Mask]:
        return [m for m in self._masks.values() if m.state == MaskState.CONFIRMED]

    # ── Mise à jour d'un mask matché ────────────────────────
    def update_mask(self, uid: int, rect: tuple, confidence: float = 0.0,
                    phash: int | None = None, source: str = "") -> None:
        mask = self._masks.get(uid)
        if mask is None:
            return
        mask.rect = rect
        mask.last_detected_rect = rect
        mask.last_detected_ts = time.time()
        mask.confidence = confidence
        mask.ttl = self._ttl
        mask.fast_miss_count = 0
        if source:
            mask.last_source = source
        if phash is not None:
            mask.add_hash(phash)
        mask.transition("matched")

    # ── Marquer des masks non-matchés ───────────────────────
    def mark_missed(self, uids: list[int]) -> None:
        for uid in uids:
            mask = self._masks.get(uid)
            if mask is None:
                continue
            mask.transition("missing")

    # ── Tick fin de frame : expiration ──────────────────────
    def tick(self) -> list[int]:
        """Décrémente TTL, supprime les expirés. Retourne les uids supprimés."""
        to_remove: list[int] = []

        for uid, mask in self._masks.items():
            mask.ttl -= 1

            if mask.state == MaskState.PENDING:
                # PENDING qui rate trop de fast → supprimé
                if mask.fast_miss_count >= self._fast_miss_limit:
                    to_remove.append(uid)
                elif mask.ttl <= 0:
                    to_remove.append(uid)

            elif mask.state == MaskState.CONFIRMED:
                if mask.ttl <= 0:
                    # passe en LOST avec sursis
                    mask.state = MaskState.LOST
                    mask.ttl = self._lost_timeout

            elif mask.state == MaskState.LOST:
                if mask.ttl <= 0:
                    to_remove.append(uid)

        for uid in to_remove:
            del self._masks[uid]

        return to_remove

    # ── Utils ────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._masks)

    def __repr__(self) -> str:
        return (f"Registry(total={len(self._masks)}, "
                f"confirmed={len(self.confirmed_masks())}, "
                f"lost={len([m for m in self._masks.values() if m.state == MaskState.LOST])})")
