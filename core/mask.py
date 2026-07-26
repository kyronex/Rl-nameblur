# core/mask.py
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List , Deque
import numpy as np
from core.box import Box
from bench.bench     import bench
from bench.lifecycle import LifecycleEvent


class MaskState(Enum):
    PENDING   = auto()   # vient d'apparaître, pas encore confirmé
    CONFIRMED = auto()   # vu assez de fois → on blur
    LOST      = auto()   # plus détecté, en sursis

@dataclass(frozen=True, slots=True)
class MaskKinematics:
    """Sous-objet cinematique FAST d'un Mask — resoud le conflit slow/fast (B-05d Axe 2-b).
    - slow (vx, vy, vw, vh, last_slow_ts) RESTE sur Mask.
    - fast ecrit UNIQUEMENT dans ce sous-objet.
    - last_fast_ts est mis a jour quand NCC confirme ET ncc_v_gate >= 0.55 (LOT 4a-1).
    Le slow ne write JAMAIS dans fast_kin.
    """
    vx_f:         float = 0.0
    vy_f:         float = 0.0
    last_fast_ts: float = 0.0

@dataclass(slots=True)
class FastMaskView:
    """
    Évolution du contrat fast :
        Tout nouveau champ requis par le fast tracker doit être ajouté explicitement ici. Cela force la revue du contrat slow→fast à chaque évolution (résout B21).
    """
    uid:              int
    rect:             tuple
    template:         Optional[np.ndarray]
    last_detected_ts: float
    vx:               float
    vy:               float
    vw:               float
    vh:               float
    state:            MaskState
    confidence:       float
    frame_id:           int            = -1
    fast_kin:  Optional[MaskKinematics] = None
    current_rect: Optional[tuple] = None
    template_capture_ts: float         = 0.0

    # -- proprietes compat lecture: .vx_f/.vy_f/.last_fast_ts delegues a fast_kin --
    @property
    def vx_f(self) -> float:
        return self.fast_kin.vx_f if self.fast_kin is not None else 0.0
    @property
    def vy_f(self) -> float:
        return self.fast_kin.vy_f if self.fast_kin is not None else 0.0
    @property
    def last_fast_ts(self) -> float:
        return self.fast_kin.last_fast_ts if self.fast_kin is not None else 0.0

def _serialize_scores(scores: dict) -> dict:
    """Sérialise le dict scores (hétérogène) pour export/log."""
    out = {}
    for k, v in scores.items():
        if isinstance(v, (int, float)):
            out[k] = round(float(v), 4)
        elif hasattr(v, "to_dict"):
            out[k] = v.to_dict()
        else:
            out[k] = repr(v)
    return out

@dataclass
class Mask:
    # --- Identité & géométrie ---
    uid:                int
    rect:               tuple   # (x, y, w, h)
    last_detected_rect: tuple
    last_detected_ts:   float
    frame_id:           int            = -1
    last_slow_ts:       float          = 0.0
    last_source:        str            = "new"
    last_fast_rect:     Optional[tuple] = None

    # --- Cinématique ---
    vx:                 float          = 0.0
    vy:                 float          = 0.0
    vw:                 float          = 0.0
    vh:                 float          = 0.0
    # --- Cinématique fast ---
    confidence:         float          = 0.0
    template:           Optional[np.ndarray] = None
    template_capture_ts: float         = 0.0
    fast_miss_count:    int            = 0
    fast_kin: MaskKinematics = field(default_factory=lambda: MaskKinematics())
    scores:             dict           = field(default_factory=dict)

    # --- Cycle de vie : état + compteurs ---
    state:              MaskState      = MaskState.PENDING
    frames_matched:     int            = 0
    total_matches_cumul: int           = 0

    # --- Cycle de vie : timestamps
    last_seen_ts:       float          = 0.0
    lost_since_ts:      Optional[float]= None
    created_ts:         float          = 0.0

    # --- Cycle de vie : timestamps capture (latences: mask_revive_latency_ms, mask_confirm_latency_ms) ---
    last_seen_frame_ts:       float          = 0.0
    lost_since_frame_ts:      Optional[float]= None

    confirm_after:      int            = field(default=1, repr=False)
    lost_after_s:       float          = field(default=1.0, repr=False)
    expire_after_lost_s: float         = field(default=10.0, repr=False)
    hash_history_max:   int            = field(default=5, repr=False)

    hash_history:       Deque[int]     = field(init=False)

    def __post_init__(self):
        self.hash_history = deque(maxlen=self.hash_history_max)
        if self.last_seen_ts == 0.0:
            self.last_seen_ts = self.last_detected_ts
        if self.created_ts == 0.0:
            self.created_ts = self.last_detected_ts

    def transition(self, event: str, ts: float, detected_frame_ts: float ,reason: str = "unknown") -> MaskState:
        """Fait progresser l'état du mask en fonction d'un événement.
            `last_seen_ts` est rafraîchi exclusivement sur `event="matched"` toute sonde mesurant l'âge du dernier match doit lire `last_seen_ts`.
            Séparation stricte des deux bases de temps :
            - Timestamps capture (`*_frame_ts`) → latences (capture − capture)
            - Timestamps perf_counter (`*_ts`)  → TTL (perf_counter − perf_counter)
            Règle : ne JAMAIS mêler les deux bases dans un même calcul de latence. NO abs(), NO clamp() pour masquer un signe négatif — corriger la cause.
        """
        if event == "matched":
            bench.count("mask_transition_matched_total")
            prev_lost_since_frame_ts = self.lost_since_frame_ts
            self.frames_matched += 1
            self.total_matches_cumul += 1
            self.last_seen_ts = ts
            self.last_seen_frame_ts = detected_frame_ts
            self.lost_since_ts = None

            if self.state == MaskState.PENDING and self.frames_matched >= self.confirm_after:
                self.state = MaskState.CONFIRMED
                bench.count("mask_promote_total")
                bench.probe("mask_confirm_latency_ms", (detected_frame_ts - self.created_ts) * 1000.0)
                bench.emit_lifecycle(LifecycleEvent.CONFIRMED, self, reason=None)
            elif self.state == MaskState.LOST:
                if prev_lost_since_frame_ts is not None:
                    bench.probe("mask_revive_latency_ms", (detected_frame_ts - prev_lost_since_frame_ts) * 1000.0)
                self.state = MaskState.CONFIRMED
                self.last_seen_frame_ts = detected_frame_ts
                self.frames_matched = 1
                bench.count("mask_revive_total")
                bench.emit_lifecycle(LifecycleEvent.REVIVE, self, reason=None)
        elif event == "missing":
            bench.count("mask_transition_missing_total")
            if self.state in (MaskState.PENDING, MaskState.CONFIRMED):
                self.state = MaskState.LOST
                self.lost_since_ts = ts
                self.lost_since_frame_ts = detected_frame_ts
                self.frames_matched = 0
                bench.count("mask_to_lost_total")
                bench.probe("mask_lost_latency_ms", (detected_frame_ts - self.created_ts) * 1000.0)
                bench.emit_lifecycle(LifecycleEvent.LOST, self, reason=None)
        return self.state

    def to_fast_view(self) -> FastMaskView:
        """
        Émet un snapshot immuable pour le thread fast tracker.
        """
        return FastMaskView(
            uid=self.uid,
            frame_id=self.frame_id,
            rect=self.rect,
            template=self.template,
            template_capture_ts=self.template_capture_ts,
            last_detected_ts=self.last_detected_ts,
            vx=self.vx,
            vy=self.vy,
            vw=self.vw,
            vh=self.vh,
            state=self.state,
            confidence=self.confidence,
            fast_kin=self.fast_kin,
            current_rect=self.rect
        )

    def to_dict(self) -> dict:
        return {
            "uid":               self.uid,
            "rx":                self.rect[0],
            "ry":                self.rect[1],
            "rw":                self.rect[2],
            "rh":                self.rect[3],
            "ldr_x":             self.last_detected_rect[0],
            "ldr_y":             self.last_detected_rect[1],
            "ldr_w":             self.last_detected_rect[2],
            "ldr_h":             self.last_detected_rect[3],
            "last_detected_ts":  round(self.last_detected_ts, 4),
            "last_source":       self.last_source,
            "vx":                round(self.vx, 2),
            "vy":                round(self.vy, 2),
            "vw":                round(self.vw, 2),
            "vh":                round(self.vh, 2),
            "confidence":        round(self.confidence, 4),
            "fast_miss_count":   self.fast_miss_count,
            "scores":            _serialize_scores(self.scores),
            "state":             self.state.name,
            "frames_matched":    self.frames_matched,
            "total_matches_cumul":  self.total_matches_cumul,
            "last_seen_ts":      round(self.last_seen_ts, 4),
            "created_ts":        round(self.created_ts, 4),
            "lost_since_ts":     round(self.lost_since_ts, 4) if self.lost_since_ts is not None else None,
        }
