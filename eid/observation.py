# eid/observation.py
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional
import logging
from core.signature import deserialize_signature
from eid.counters import Counters, CounterKey

log = logging.getLogger("eid.observation")

DEDUP_GEOM_NDIGITS = 1

@dataclass(frozen=True, slots=True)
class Observation:
    uid: int
    frame_id: int
    ts: float
    rect: Tuple[float, float, float, float]  # (x, y, w, h)
    source: str                      # "slow" ou "fast"
    signature: Dict[str, Any]        # Dict exhaustif généré par signature.py
    # --- Ajouts depuis l'ancien ObservationBank ---
    confidence: Optional[float] = None  # non consommé (D20)
    vx: Optional[float] = None  # non consommé (D20)
    vy: Optional[float] = None  # non consommé (D20)
    dedup_key: Optional[Tuple] = None
    seq: Optional[int] = field(default=None, compare=False)  # Indispensable pour la calibration

    @property
    def cx(self) -> float:
        return self.rect[0] + self.rect[2] / 2.0

    @property
    def cy(self) -> float:
        return self.rect[1] + self.rect[3] / 2.0

    @property
    def rw(self) -> float:
        return self.rect[2]

    @property
    def rh(self) -> float:
        return self.rect[3]

    @property
    def has_shape(self) -> bool:
        """Indique si la signature contient une forme globale valide."""
        return bool(self.signature and self.signature.get("sig_shape") is not None)

    @property
    def has_map(self) -> bool:
        """Indique si la signature contient une carte glissante (plaques partielles)."""
        return bool(self.signature and self.signature.get("sig_map") is not None)

    @staticmethod
    def from_entry(entry: dict, counters: Counters) -> Optional["Observation"]:
        """Construit une observation depuis une entry bank.

        `counters` est obligatoire et reçoit les compteurs directement.
        Le retour est `Optional[Observation]` uniquement. Les motifs
        (rejet fatal + warnings non-fatals) sont incrémentés EN INTERNE
        via `counters.bump(CounterKey.SRC_*)`, au site exact de détection :
          - warnings : `counters.bump(CounterKey.SRC_OBS_SIG_EMPTY)`,
                       `counters.bump(CounterKey.SRC_OBS_SLOW_TS_FALLBACK)`.

        Rejet (comportement A2-1 strict) :
          w <= 0 ou h <= 0  =>  `counters.bump(CounterKey.SRC_OBS_REJECT_DEGENERATE_BOX)`
                                puis `return None`.

        `counters` n'est JAMAIS stocké comme attribut de l'instance
        `Observation` (frozen=True, slots=True).
        """
        counters.bump(CounterKey.SRC_OBS_READ_TOTAL)

        try:
            entry_source = entry.get("last_source")
            if entry_source not in ("slow", "fast"):
                counters.bump(CounterKey.SRC_OBS_REJECT_SOURCE)
                return None

            if entry.get("frame_id") is None:
                counters.bump(CounterKey.SRC_OBS_MISSING_FRAME_ID)
                return None

            if entry.get("rx") is None or entry.get("ry") is None:
                counters.bump(CounterKey.SRC_OBS_REJECT_MISSING_GEOMETRY)
                return None

            # Seul test qui a du sens : au moins un descripteur NON-None.
            rw = float(entry.get("rw", 0.0))
            rh = float(entry.get("rh", 0.0))
            if rw <= 0.0 or rh <= 0.0:
                counters.bump(CounterKey.SRC_OBS_REJECT_DEGENERATE_BOX)
                return None

            sig = deserialize_signature(entry)
            if sig.get("sig_hs") is None and sig.get("sig_shape") is None:
                counters.bump(CounterKey.SRC_OBS_SIG_EMPTY)
                sig = None

            if entry_source == "slow":
                raw_slow_ts = entry.get("last_slow_ts", 0.0)
                if raw_slow_ts == 0.0:
                    entry_ts = entry.get("last_detected_ts", 0.0)
                    counters.bump(CounterKey.SRC_OBS_SLOW_TS_FALLBACK)
                else:
                    entry_ts = raw_slow_ts
                ldr = (entry.get("ldr_x"), entry.get("ldr_y"),
                       entry.get("ldr_w"), entry.get("ldr_h"))
                if entry_ts is not None and all(v is not None for v in ldr):
                    dedup_key = ("slow", round(float(entry_ts), 4)) + tuple(
                        round(float(v), DEDUP_GEOM_NDIGITS) for v in ldr
                    )
                else:
                    dedup_key = None
            else:
                entry_ts = entry.get("last_detected_ts", 0.0)
                lfr = (entry.get("lfr_x"), entry.get("lfr_y"),
                       entry.get("lfr_w"), entry.get("lfr_h"))
                if all(v is not None for v in lfr):
                    dedup_key = ("fast",) + tuple(
                        round(float(v), DEDUP_GEOM_NDIGITS) for v in lfr
                    )
                else:
                    dedup_key = None

            counters.bump(CounterKey.SRC_OBS_BUILT)
            return Observation(
                uid        = int(entry["uid"]),
                frame_id   = int(entry["frame_id"]),
                ts         = float(entry_ts),
                rect       = (float(entry["rx"]),
                              float(entry["ry"]), rw, rh),
                source     = entry_source,
                signature  = sig or {},
                confidence = float(entry["confidence"]) if entry.get("confidence") is not None else None,
                vx         = float(entry["vx"]) if entry.get("vx") is not None else None,
                vy         = float(entry["vy"]) if entry.get("vy") is not None else None,
                dedup_key  = dedup_key,
            )

        except (KeyError, ValueError, TypeError, IndexError) as exc:
            log.warning("[Observation] entry malformée : %s", exc)
            counters.bump(CounterKey.SRC_OBS_REJECT_MALFORMED)
            return None