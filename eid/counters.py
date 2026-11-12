# eid/counters.py
"""
Compteurs globaux du pipeline EID.

Contrat (Lot PP-D8 — clés typées) :
    - Le registre fait foi : `CounterKey` (compteurs incrémentaux) et
      `GaugeKey` (jauges, valeur instantanée). Aucune clé n'appartient
      aux deux ensembles.
    - `bump()` n'accepte que `CounterKey`, `_set_gauge()` n'accepte que
      `GaugeKey` ; tout autre type (y compris un littéral str) lève
      TypeError. Filet runtime pour les appels non typés.
    - Les valeurs des membres sont explicites et identiques aux anciennes
      clés str : rapport et JSON inchangés.
    - Les vues de lecture (`stats`, `as_dict`, `gauges`, `anomalies`)
      renvoient des clés `str` simples.
    - Ne pas additionner compteurs et jauges dans les rapports.
    - Le préfixe `anomaly_` réserve les clés de signalement ;
      `anomalies()` filtre sur ce préfixe.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Iterator, Tuple

ANOMALY_PREFIX = "anomaly_"


class CounterKey(StrEnum):
    SLOW_OBS = "slow_obs"
    FAST_OBS = "fast_obs"
    CLUSTERS_SLOW_CREATED = "clusters_slow_created"
    CLUSTERS_FAST_CREATED = "clusters_fast_created"
    CLUSTERS_SLOW_FINALIZED = "clusters_slow_finalized"
    EVICTED_FAST = "evicted_fast"
    HUNGARIAN_BATCHES = "hungarian_batches"
    BATCHES_READ = "batches_read"
    BATCH_OBS_CUMUL = "batch_obs_cumul"
    BATCH_FRAME_OUT_OF_ORDER = "batch_frame_out_of_order"
    GATE_TEMPORAL_REJECT_SLOW = "gate_temporal_reject_slow"
    COST_REJECT_SLOW = "cost_reject_slow"
    COLS_GATE_ONLY = "cols_gate_only"
    COLS_COST_REJECTED = "cols_cost_rejected"
    FAST_MATCHED = "fast_matched"
    GATE_TEMPORAL_REJECT_FAST = "gate_temporal_reject_fast"
    COST_REJECT_FAST = "cost_reject_fast"
    TRACK_ADOPTED = "track_adopted"
    EID_ACTIVE_CLUSTERS_OVER_THRESHOLD = "eid_active_clusters_over_threshold"
    ANOMALY_IDENTITY_TRACK_MISMATCH = "anomaly_identity_track_mismatch"
    ANOMALY_FAST_AMBIGUOUS = "anomaly_fast_ambiguous"
    SRC_OBS_READ_TOTAL = "src_obs_read_total"
    SRC_OBS_BUILT = "src_obs_built"
    SRC_OBS_REJECT_SOURCE = "src_obs_reject_source"
    SRC_OBS_REJECT_DEGENERATE_BOX = "src_obs_reject_degenerate_box"
    SRC_OBS_REJECT_MISSING_GEOMETRY = "src.obs.reject.missing_geometry"
    SRC_OBS_REJECT_MALFORMED = "src_obs_reject_malformed"
    SRC_OBS_SLOW_TS_FALLBACK = "src_obs_slow_ts_fallback"
    SIGHTING_EMITTED = "sighting.emitted"
    SIGHTING_EMITTED_NO_ANCHOR = "sighting.emitted.no_anchor"
    SIGHTING_CALLBACK_ERROR = "sighting.callback.error"
    SIG_PAIR_REJECT_NO_COMMON_CHANNEL = "sig.pair.reject.no_common_channel"
    ACTIVE_SLOW_CLUSTERS_WARN_EXCEEDED = "active_slow_clusters_warn_exceeded"
    ACTIVE_FAST_CLUSTERS_WARN_EXCEEDED = "active_fast_clusters_warn_exceeded"
    SRC_OBS_SIG_EMPTY = "src_obs_sig_empty"
    SRC_OBS_DEDUP_SKIPPED = "src_obs_dedup_skipped"
    SRC_OBS_MISSING_FRAME_ID = "src_obs_missing_frame_id"
    SRC_DEDUP_KEPT = "src_dedup_kept"
    SRC_DEDUP_PURGES = "src_dedup_purges"
    # --- Lot C3 Phase 1 : compteurs C3.8 (incrément par l'appelant) ---
    SIG_GATE_H_TXT = "sig_gate_h_txt"
    SIG_GATE_L_RATIO = "sig_gate_l_ratio"
    SIG_GATE_OCCUPANCY = "sig_gate_occupancy"
    SIG_GATE_NO_DESCRIPTOR = "sig_gate_no_descriptor"
    SIG_GATE_HS_DIVERGENCE = "sig_gate_hs_divergence"
    SIG_HS_PLAN_RAW = "sig_hs_plan_raw"
    SIG_H_TXT_PENALIZED = "sig_h_txt_penalized"
    REF_PROMOTIONS = "ref_promotions"
    REF_PROMO_EARLY = "ref_promo_early"
    REF_PROMO_WINDOW = "ref_promo_window"
    PROMO_BLOCKED_IDENTITY = "promo_blocked_identity"
    PROMO_BLOCKED_QUALITY = "promo_blocked_quality"
    REF_LOCKED_NO_PROMO = "ref_locked_no_promo"
    REF_INITIAL_SIGNATURE_EMPTY = "ref_initial_signature_empty"


class GaugeKey(StrEnum):
    EID_ACTIVE_CLUSTERS_SLOW = "eid_active_clusters_slow"
    EID_ACTIVE_CLUSTERS_FAST = "eid_active_clusters_fast"


# Garde au chargement : aucune valeur partagée entre les deux registres.
_overlap = {k.value for k in CounterKey} & {k.value for k in GaugeKey}
if _overlap:
    raise RuntimeError(f"[Counters] Clés à la fois compteur et jauge : {sorted(_overlap)}")
del _overlap


class Counters:
    _warn_slow_active: bool
    _warn_fast_active: bool

    __slots__ = ("_d", "_g", "_warn_slow_active", "_warn_fast_active")

    def __init__(self) -> None:
        self._d: dict[CounterKey, int] = {k: 0 for k in CounterKey}
        # Jauges initialisées à 0 : stats() expose les mêmes clés
        # qu'avant PP-D8, même sans échantillonnage (flux vide).
        self._g: dict[GaugeKey, int] = {k: 0 for k in GaugeKey}
        self._warn_slow_active = False
        self._warn_fast_active = False

    # — Écriture —
    def bump(self, key: CounterKey, n: int = 1) -> None:
        if not isinstance(key, CounterKey):
            raise TypeError(
                f"[Counters] bump() attend un CounterKey, reçu "
                f"{type(key).__name__} : {key!r}."
            )
        if not isinstance(n, int):
            raise TypeError(f"[Counters] bump() : n doit être un int, reçu {type(n).__name__}.")
        self._d[key] += n

    def _set_gauge(self, key: GaugeKey, value: int) -> None:
        if not isinstance(key, GaugeKey):
            raise TypeError(
                f"[Counters] _set_gauge() attend un GaugeKey, reçu "
                f"{type(key).__name__} : {key!r}."
            )
        self._g[key] = int(value)

    def sample_active_clusters(self, n_slow: int, n_fast: int, warn_threshold: int) -> None:
        self._set_gauge(GaugeKey.EID_ACTIVE_CLUSTERS_SLOW, n_slow)
        self._set_gauge(GaugeKey.EID_ACTIVE_CLUSTERS_FAST, n_fast)
        slow_active = n_slow > warn_threshold
        fast_active = n_fast > warn_threshold
        if slow_active and not self._warn_slow_active:
            self.bump(CounterKey.ACTIVE_SLOW_CLUSTERS_WARN_EXCEEDED)
        if fast_active and not self._warn_fast_active:
            self.bump(CounterKey.ACTIVE_FAST_CLUSTERS_WARN_EXCEEDED)
        self._warn_slow_active = slow_active
        self._warn_fast_active = fast_active
        if (n_slow + n_fast) > warn_threshold:
            self.bump(CounterKey.EID_ACTIVE_CLUSTERS_OVER_THRESHOLD)

    # — Lecture —
    def get(self, key: CounterKey | GaugeKey) -> int:
        if isinstance(key, GaugeKey):
            return self._g[key]
        if isinstance(key, CounterKey):
            return self._d[key]
        raise TypeError(
            f"[Counters] get() attend un CounterKey ou GaugeKey, reçu "
            f"{type(key).__name__} : {key!r}."
        )

    def as_dict(self) -> dict[str, int]:
        """Compteurs seuls."""
        return {k.value: v for k, v in self._d.items()}

    def gauges(self) -> dict[str, int]:
        """Jauges seules."""
        return {k.value: v for k, v in self._g.items()}

    def stats(self) -> dict[str, int]:
        """Vue fusionnée compteurs + jauges (sans collision). Ne pas sommer."""
        return {**self.as_dict(), **self.gauges()}

    def items(self) -> Iterator[Tuple[str, int]]:
        return iter(self.stats().items())

    def anomalies(self) -> dict[str, int]:
        return {k.value: v for k, v in self._d.items() if k.value.startswith(ANOMALY_PREFIX)}