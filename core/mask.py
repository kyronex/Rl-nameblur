# mask.py — Dataclass Mask (étape 0)
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
from core.box import Box


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
    confidence:         float          = 0.0
    template:           Optional[np.ndarray] = None
    fast_miss_count:    int            = 0
    box:                Optional[Box]  = None
    scores:             List[float]    = field(default_factory=list)

    # ── helpers ───────────────────────────────────────────
    def to_dict(self) -> dict:
        """Sérialisation plate pour csv_bench."""
        return {
            "uid":                self.uid,
            "rx":                 self.rect[0],
            "ry":                 self.rect[1],
            "rw":                 self.rect[2],
            "rh":                 self.rect[3],
            "ldr_x":             self.last_detected_rect[0],
            "ldr_y":             self.last_detected_rect[1],
            "ldr_w":             self.last_detected_rect[2],
            "ldr_h":             self.last_detected_rect[3],
            "last_detected_ts":  round(self.last_detected_ts, 4),
            "last_source":       self.last_source,
            "ttl":               self.ttl,
            "vx":                round(self.vx, 2),
            "vy":                round(self.vy, 2),
            "confidence":        round(self.confidence, 4),
            "fast_miss_count":   self.fast_miss_count,
        }
