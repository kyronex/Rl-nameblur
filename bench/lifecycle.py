# bench/lifecycle.py
from __future__ import annotations

from enum import Enum
from typing import Optional, TypedDict


class LifecycleEvent(str, Enum):
    CREATED   = "CREATED"
    DETECTED  = "DETECTED"
    CONFIRMED = "CONFIRMED"
    LOST      = "LOST"
    REVIVE    = "REVIVE"
    EXPIRED   = "EXPIRED"
    EVICTED   = "EVICTED"


class LifecycleRecord(TypedDict, total=False):
    """
    Contrat d'un événement lifecycle dans le JSONL.
    Meta-fields (schema_version, ts, mono, session_id, mode="events")
    sont injectés par BenchJsonlWriter._enqueue() — NON dans ce module.
    """
    # ── Abstraction bench ──────────────────────────────────────
    event:              str

    # ── Identité mask ──────────────────────────────────────────
    mask_id:            int

    # ── État réel du mask au moment de l'événement ────────────
    # ∈ MaskState (PENDING / CONFIRMED / LOST) — jamais EXPIRED / CREATED / REVIVE
    state:              str

    # ── Géométrie (tuples mask.rect[0..3]) ─────────────────────
    rx:                 float
    ry:                 float
    rw:                 float
    rh:                 float

    # ── Qualité de détection ────────────────────────────────────
    confidence:         float

    # ── Timestamps ──────────────────────────────────────────────
    created_ts:         float   # mask.created_ts  (epoch s)
    event_ts:           float   # timestamp de l'événement (epoch s)

    # ── Compteurs de匹配 cumulés ───────────────────────────────
    total_matches_cumul: int
    frames_matched:     int     # frames_matched au moment de l'événement

    # ── Provenance ──────────────────────────────────────────────
    # source alimenté par mask.last_source  (valeurs typiques : "new",
    # "slow", "fast" — voir Mask.last_source dans mask.py)
    source:             str

    # ── Contexte LOST ───────────────────────────────────────────
    # Positionné pour CONFIRMED/LOST/REVIVE/EXPIRED ; None pour CREATED
    lost_since_ts:       Optional[float]

    # ── Raison de l'événement (optionnel) ─────────────────────
    reason:             Optional[str]

    # ── Flag revive ────────────────────────────────────────────
    revived:            Optional[bool]  # True si LOST→CONFIRMED (re-détection), None sinon
    frame_id:          int

    scores:       dict
    hash_history: list

if __name__ == "__main__":
    members = [e.name for e in LifecycleEvent]
    print(members)
    assert len(members) == 6, f"attendu 6, obtenu {len(members)}"

class DetectionRecord(TypedDict, total=False):
    frame_id:   int
    rx: float
    ry: float
    rw: float
    rh: float
    phash:      Optional[int]
    source:     str              # "slow"
    confidence: float
    scores:     dict