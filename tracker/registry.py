# tracker/registry.py
from __future__ import annotations
import logging
import time  # QUICK-FIX B-01: instrumentation diagnostic zombies
from typing import Dict, List, Optional
from core.mask import Mask, MaskState
from tracker.models import TrackerConfig

log = logging.getLogger("tracker.registry")

class MaskRegistry:
    def __init__(self, config: TrackerConfig):
        self._masks: Dict[int, Mask] = {}
        self._next_uid: int = 0
        self.cfg = config
        # QUICK-FIX B-01: stats par mask pour diagnostic zombies (retirer en B-03)
        # Clé = uid ; valeur = dict {created_ts, match_count_slow, match_count_fast,
        #                            last_match_ts, last_match_source}
        self._b01_stats: Dict[int, dict] = {}
        self._b01_tick_counter: int = 0
        self._b01_zombie_threshold_s: float = 1.0  # log si age > 1s sans purge

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
        # QUICK-FIX B-01
        self._b01_stats[mask.uid] = {
            "created_ts": time.perf_counter(),
            "match_count_slow": 0,
            "match_count_fast": 0,
            "match_count_unknown": 0,
            "last_match_ts": time.perf_counter(),
            "last_match_source": "create",
        }
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
        # QUICK-FIX B-01
        log.debug(
            "[B01] CREATE uid=%d source=%s rect=%s last_seen_ts=%.3f state=%s",
            uid, source, rect, mask.last_seen_ts, mask.state.name,
        )
        return added

    def remove(self, uid: int) -> Optional[Mask]:
        # QUICK-FIX B-01: nettoyer stats
        self._b01_stats.pop(uid, None)
        return self._masks.pop(uid, None)

    # ── mise à jour post-match ────────────────────────────
    def mark_matched(self, uid: int,ts: float, source: str = "unknown") -> None:
        """
        QUICK-FIX B-01: paramètre `source` ajouté pour diagnostic.
        Valeur attendue: "slow" | "fast". "unknown" = site d'appel non instrumenté.
        """
        mask = self._masks.get(uid)
        if mask is None:
            return

        prev_state = mask.state.name
        gap = ts - mask.last_seen_ts
        mask.transition("matched", ts)

        if source == "slow":
            mask.last_slow_ts = ts
        # QUICK-FIX B-01: mise à jour stats
        stats = self._b01_stats.get(uid)
        if stats is not None:
            key = f"match_count_{source}" if source in ("slow", "fast") else "match_count_unknown"
            stats[key] = stats.get(key, 0) + 1
            stats["last_match_ts"] = time.perf_counter()
            stats["last_match_source"] = source
        # Log DEBUG par match (verbeux, activable au besoin)
        log.debug(
            "[B01] MATCH uid=%d source=%s gap_since_last_seen=%.3fs state:%s->%s",
            uid, source, gap, prev_state, mask.state.name,
        )

    # ── expiration ────────────────────────────────────────
    def tick_and_expire(self,ts: float, updated_uids: set = None) -> List[Mask]:
        """
         Cycle de vie temporel (B-04) :
          - mask non-matché depuis > lost_after_s         → état LOST
          - mask LOST depuis > expire_after_lost_s        → purge

        `ts` doit être un perf_counter() cohérent avec celui passé à mark_matched().
        """
        if updated_uids is None:
            updated_uids = set()
        expired: List[Mask] = []

        # QUICK-FIX B-01
        self._b01_tick_counter += 1
        now_wall  = time.perf_counter()
        non_matched_count = 0

        lost_after_s = self.cfg.lost_after_s
        expire_after_lost_s = self.cfg.expire_after_lost_s

        for mask in list(self._masks.values()):
            is_matched_this_tick = mask.uid in updated_uids

            if not is_matched_this_tick:
                non_matched_count += 1
                # Transition vers LOST si hors-vue depuis trop longtemps
                if mask.state in (MaskState.PENDING, MaskState.CONFIRMED):
                    if (ts - mask.last_seen_ts) >= lost_after_s:
                        mask.transition("missing", ts)  # passe en LOST, set lost_since_ts=ts

            # Purge des LOST trop vieux (qu'ils aient été ré-évalués ce tick ou non)
            if mask.state == MaskState.LOST and mask.lost_since_ts is not None:
                if (ts - mask.lost_since_ts) >= expire_after_lost_s:
                    expired.append(mask)
                    stats = self._b01_stats.get(mask.uid, {})
                    lifetime = now_wall - stats.get("created_ts", now_wall)
                    last_match_age = now_wall - stats.get("last_match_ts", now_wall)
                    log.debug(
                        "[B01] EXPIRE uid=%d lifetime=%.2fs lost_for=%.2fs "
                        "matches=%d (slow=%d fast=%d unknown=%d) last_match=%s %.2fs_ago",
                        mask.uid, lifetime, ts - mask.lost_since_ts,
                        stats.get("match_count_slow", 0)
                        + stats.get("match_count_fast", 0)
                        + stats.get("match_count_unknown", 0),
                        stats.get("match_count_slow", 0),
                        stats.get("match_count_fast", 0),
                        stats.get("match_count_unknown", 0),
                        stats.get("last_match_source", "?"),
                        last_match_age,
                    )
                    self._b01_stats.pop(mask.uid, None)
                    del self._masks[mask.uid]

        # QUICK-FIX B-01: zombies-suspects (échantillonné)
        if self._b01_tick_counter % 30 == 0:
            for uid, stats in list(self._b01_stats.items()):
                mask = self._masks.get(uid)
                if mask is None:
                    continue
                age = now_wall - stats["created_ts"]
                if age >= self._b01_zombie_threshold_s:
                    last_match_age = now_wall - stats["last_match_ts"]
                    age_since_seen = ts - mask.last_seen_ts
                    log.debug(
                        "[B01] ZOMBIE-SUSPECT uid=%d age=%.2fs state=%s "
                        "since_last_seen=%.2fs matches=%d (slow=%d fast=%d unknown=%d) "
                        "last_match=%s %.2fs_ago",
                        uid, age, mask.state.name, age_since_seen,
                        stats["match_count_slow"]
                        + stats["match_count_fast"]
                        + stats["match_count_unknown"],
                        stats["match_count_slow"],
                        stats["match_count_fast"],
                        stats["match_count_unknown"],
                        stats["last_match_source"],
                        last_match_age,
                    )

        log.debug(
            "[B01] TICK#%d total_masks=%d non_matched=%d expired=%d",
            self._b01_tick_counter, len(self._masks),
            non_matched_count, len(expired),
        )
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
        stats = self._b01_stats.pop(worst.uid, {})
        log.info(
            "[B01] EVICT uid=%d (capacity full) lifetime=%.2fs matches=%d state=%s",
            worst.uid,
            time.perf_counter() - stats.get("created_ts", time.perf_counter()),
            stats.get("match_count_slow", 0)
            + stats.get("match_count_fast", 0)
            + stats.get("match_count_unknown", 0),
            worst.state.name,
        )
        del self._masks[worst.uid]
