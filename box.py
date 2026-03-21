from dataclasses import dataclass, field
import numpy as np

@dataclass
class Box:
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
      s = self.scores
      td = s.get("transition_density", 0.0)
      rf = s.get("row_fill", 0.0)
      vp = s.get("vproj", 0.0)
      dn = s.get("density", 0.0)
      cc = s.get("cc", 0.0)
      hr = s.get("hreg", 0.0)
      bg = s.get("bg_score", 0.0)
      return (0.20 * td
            + 0.10 * rf
            + 0.10 * vp
            + 0.10 * dn
            + 0.15 * cc
            + 0.15 * hr
            + 0.20 * bg)

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
    def merge_scores(*boxes):
        merged = {}
        for box in boxes:
            for k, v in box.scores.items():
                merged[k] = max(merged.get(k, 0.0), v)
        return merged
