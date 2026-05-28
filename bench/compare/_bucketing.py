# bench/compare/_bucketing.py
from __future__ import annotations
import logging
from dataclasses import dataclass
from bench.compare._config import (
    BUCKET_COLD_TARGET_S,
    BUCKET_HOT_DURATION_S,
    BUCKET_MAX_COLD_DRIFT_S,
    BUCKET_BOUNDARY_GUARD_S,
    BUCKET_MIN_GAP_S,
    BUCKET_EPSILON_S,
)

log = logging.getLogger(__name__)

@dataclass
class BucketSpec:
    mono_start: float
    mono_end:   float

    @property
    def duration_s(self) -> float:
        return self.mono_end - self.mono_start

@dataclass
class ColdSpec(BucketSpec):
    cold_end_target_s:   float
    cold_end_real_s:     float
    cold_drift_s:        float
    cold_drift_warning:  bool
    cold_truncated:      bool   # True si cas B ou C

@dataclass
class HotSpec(BucketSpec):
    index:            int
    is_pivot_snapped: bool

@dataclass
class TailSpec(BucketSpec):
    is_partial: bool = True

@dataclass
class BucketsResult:
    cold:          ColdSpec
    hot:           list[HotSpec]
    tail:          TailSpec | None
    fast_enabled:  bool
    t_min:         float
    t_max:         float

def _next_event_after(timeline: list[dict], t: float) -> float:
    """Premier mono > t dans la timeline. +inf si aucun."""
    for e in timeline:
        if e["mono"] > t:
            return e["mono"]
    return float("inf")

def _compute_cold_end(t_min: float,t_max: float,timeline_agg: list[dict],timeline_fast: list[dict],fast_enabled: bool) -> tuple[float, float, bool, bool]:
    """
    Calcule la fin réelle du cold.
    Cas A : next_agg et next_fast (si fast) existent → cold_end normal.
    Cas B : next_agg existe, next_fast = +inf → cold_end = next_agg + ε, warning.
    Cas C : next_agg = +inf → cold_end = t_max, cold_truncated = True.
    Retourne (cold_end_real, cold_drift_s, cold_drift_warning, cold_truncated).
    """
    t_target = t_min + BUCKET_COLD_TARGET_S
    next_agg = _next_event_after(timeline_agg, t_target)
    if next_agg == float("inf"):
        # Cas C — session trop courte
        return t_max, float("inf"), False, True
    if fast_enabled:
        next_fast = _next_event_after(timeline_fast, t_target)
        if next_fast == float("inf"):
            # Cas B — fast vide après cible
            cold_end = next_agg + BUCKET_EPSILON_S
            drift = cold_end - t_target
            warning = drift > BUCKET_MAX_COLD_DRIFT_S
            log.warning("cold_end: fast timeline vide après cible — cas B")
            return cold_end, drift, warning, False
        cold_end = max(next_agg, next_fast) + BUCKET_EPSILON_S
    else:
        cold_end = next_agg + BUCKET_EPSILON_S
    drift = cold_end - t_target
    warning = drift > BUCKET_MAX_COLD_DRIFT_S
    if warning:
        log.warning("cold_drift=%.3fs dépasse max_cold_drift_s=%.3fs", drift, BUCKET_MAX_COLD_DRIFT_S)
    return cold_end, drift, warning, False

def _find_pivot(T_theorique: float,timeline_agg: list[dict],timeline_fast: list[dict]) -> tuple[float, bool]:
    """
    D2 — Analytique : trouve dans [T-guard, T+guard] un instant le plus proche
    de T_theorique dans un intervalle vide ≥ 2×min_gap_s entre événements agg+fast.
    Retourne (frontière, is_snapped).
    """
    t_lo = T_theorique - BUCKET_BOUNDARY_GUARD_S
    t_hi = T_theorique + BUCKET_BOUNDARY_GUARD_S
    # Fusion des monos agg + fast dans la fenêtre, triés
    events = sorted(
        {e["mono"] for e in timeline_agg + timeline_fast
         if t_lo - BUCKET_MIN_GAP_S <= e["mono"] <= t_hi + BUCKET_MIN_GAP_S}
    )
    # Intervalles entre événements consécutifs
    # On ajoute des sentinelles pour couvrir les bords de la fenêtre
    sentinels = [t_lo - BUCKET_MIN_GAP_S] + events + [t_hi + BUCKET_MIN_GAP_S]
    best: float | None = None
    best_dist = float("inf")
    for i in range(len(sentinels) - 1):
        gap_start = sentinels[i]
        gap_end   = sentinels[i + 1]
        gap_width = gap_end - gap_start
        if gap_width < 2 * BUCKET_MIN_GAP_S:
            continue  # Intervalle trop petit
        # Instant candidat dans cet intervalle le plus proche de T
        candidate = max(gap_start + BUCKET_MIN_GAP_S,min(gap_end - BUCKET_MIN_GAP_S, T_theorique))
        # Doit rester dans [t_lo, t_hi]
        if not (t_lo <= candidate <= t_hi):
            continue
        dist = abs(candidate - T_theorique)
        if dist < best_dist:
            best_dist = dist
            best = candidate
    if best is not None:
        return best, True
    return T_theorique, False

def _generate_hot_buckets(cold_end: float,t_max: float,timeline_agg: list[dict],timeline_fast: list[dict]) -> tuple[list[HotSpec], TailSpec | None]:
    """
    Génère les hot_i et le tail depuis cold_end jusqu'à t_max.
    """
    hot: list[HotSpec] = []
    t_cursor = cold_end
    i = 0
    while True:
        T_theorique = t_cursor + BUCKET_HOT_DURATION_S
        if T_theorique > t_max:
            break  # résidu → tail
        frontier, is_snapped = _find_pivot(T_theorique, timeline_agg, timeline_fast)
        # Borne : frontier ne peut pas dépasser t_max
        frontier = min(frontier, t_max)
        hot.append(HotSpec(
            mono_start=t_cursor,
            mono_end=frontier,
            index=i,
            is_pivot_snapped=is_snapped,
        ))
        t_cursor = frontier
        i += 1
    tail: TailSpec | None = None
    if t_max - t_cursor > 0:
        tail = TailSpec(mono_start=t_cursor, mono_end=t_max)
    return hot, tail


def compute_buckets(timeline_agg:   list[dict],timeline_fast:  list[dict],timeline_frame: list[dict]) -> BucketsResult | None:
    """
    Calcule le bucketing adaptatif cold / hot_i / tail.
    Retourne None si les timelines sont insuffisantes (< 2 événements agg).
    """
    if len(timeline_agg) < 2:
        log.info("compute_buckets: timeline_agg < 2 événements — bucketing ignoré")
        return None
    fast_enabled = len(timeline_fast) > 0
    all_monos = (
        [e["mono"] for e in timeline_agg]
        + [e["mono"] for e in timeline_fast]
        + [e["mono"] for e in timeline_frame]
    )
    t_min = min(all_monos)
    t_max = max(all_monos)
    # ── Cold ──
    cold_end, drift, drift_warning, cold_truncated = _compute_cold_end(t_min, t_max, timeline_agg, timeline_fast, fast_enabled)
    cold_end = min(cold_end, t_max)
    cold = ColdSpec(
        mono_start=t_min,
        mono_end=cold_end,
        cold_end_target_s=BUCKET_COLD_TARGET_S,
        cold_end_real_s=cold_end - t_min,
        cold_drift_s=min(drift, t_max - t_min),
        cold_drift_warning=drift_warning,
        cold_truncated=cold_truncated,
    )
    # ── Hot + tail ──
    if cold_truncated:
        log.info("compute_buckets: cold_truncated — pas de hot ni tail")
        return BucketsResult(
            cold=cold, hot=[], tail=None,
            fast_enabled=fast_enabled, t_min=t_min, t_max=t_max,
        )
    hot, tail = _generate_hot_buckets(cold_end, t_max, timeline_agg, timeline_fast)
    log.info(
        "compute_buckets: cold=%.3fs, hot=%d, tail=%s, fast=%s",
        cold.duration_s, len(hot),
        f"{tail.duration_s:.3f}s" if tail else "None",
        fast_enabled,
    )
    return BucketsResult(
        cold=cold, hot=hot, tail=tail,
        fast_enabled=fast_enabled, t_min=t_min, t_max=t_max,
    )
