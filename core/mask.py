# core/mask.py — Dataclass Mask (étape 2)
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List
import numpy as np
from core.box import Box

class MaskState(Enum):
    PENDING   = auto()   # vient d'apparaître, pas encore confirmé
    CONFIRMED = auto()   # vu assez de fois → on blur
    LOST      = auto()   # plus détecté, en sursis (TTL)

@dataclass
class Mask:
    uid:                int
    rect:               tuple                          # (x, y, w, h)
    last_detected_rect: tuple                          # (x, y, w, h)
    last_detected_ts:   float
    last_source:        str            = "new"
    ttl:                int            = 0
    vx:                 float          = 0.0
    vy:                 float          = 0.0
    vw:                 float          = 0.0
    vh:                 float          = 0.0
    confidence:         float          = 0.0
    template:           Optional[np.ndarray] = None
    fast_miss_count:    int            = 0
    box:                Optional[Box]  = None
    scores:             List[float]    = field(default_factory=list)

    state:              MaskState      = MaskState.PENDING
    hash_history:       List[int]      = field(default_factory=list)
    frames_matched:     int            = 0
    frames_missing:     int            = 0

    confirm_after:      int            = field(default=3, repr=False)
    lost_after:         int            = field(default=5, repr=False)

    def transition(self, event: str) -> MaskState:
        if event == "matched":
            self.frames_matched += 1
            self.frames_missing = 0
            if self.state == MaskState.PENDING and self.frames_matched >= self.confirm_after:
                self.state = MaskState.CONFIRMED
            elif self.state == MaskState.LOST:
                self.state = MaskState.CONFIRMED
        elif event == "missing":
            self.frames_missing += 1
            if self.state in (MaskState.PENDING, MaskState.CONFIRMED):
                if self.frames_missing >= self.lost_after:
                    self.state = MaskState.LOST
        return self.state

    def add_hash(self, h: int, max_history: int = 5) -> None:
        self.hash_history.append(h)
        if len(self.hash_history) > max_history:
            self.hash_history = self.hash_history[-max_history:]

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
            "ttl":               self.ttl,
            "vx":                round(self.vx, 2),
            "vy":                round(self.vy, 2),
            "vw":                round(self.vw, 2),
            "vh":                round(self.vh, 2),
            "confidence":        round(self.confidence, 4),
            "fast_miss_count":   self.fast_miss_count,
            "state":             self.state.name,
            "frames_matched":    self.frames_matched,
            "frames_missing":    self.frames_missing,
        }
