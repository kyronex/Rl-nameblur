# core/box.py
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class Box:
    SCORE_KEY = "score"

    x: int
    y: int
    w: int
    h: int
    scores: dict = field(default_factory=dict)
    template: np.ndarray = field(default=None, repr=False)

    @property
    def rect(self):
        return (self.x, self.y, self.w, self.h)

    @property
    def confidence(self):
        return float(self.scores.get(self.SCORE_KEY, 0.0))

    def copy_with(self, x=None, y=None, w=None, h=None):
        return Box(
            x=x if x is not None else self.x,
            y=y if y is not None else self.y,
            w=w if w is not None else self.w,
            h=h if h is not None else self.h,
            scores=dict(self.scores),
            template=self.template,
        )

    @staticmethod
    def from_rect(rect):
        return Box(x=rect[0], y=rect[1], w=rect[2], h=rect[3])

    @staticmethod
    def merge_scores(*boxes) -> dict:
        merged: dict = {}
        for box in boxes:
            for k, v in box.scores.items():
                if not isinstance(v, (int, float)):
                    continue
                merged[k] = max(merged.get(k, 0.0), float(v))
        return merged
