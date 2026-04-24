# tracker/models.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from config import cfg

@dataclass
class TrackerConfig:
    # Frame / écran
    screen_w: int = field(default_factory=lambda: cfg.get("screen.width"))
    screen_h: int = field(default_factory=lambda: cfg.get("screen.height"))

    # registry / lifecycle
    max_masks: int = field(default_factory=lambda: cfg.get("masks.max_masks"))
    ttl_default: int = field(default_factory=lambda: cfg.get("masks.ttl_default"))
    confirm_after: int = field(default_factory=lambda: cfg.get("masks.confirm_after"))
    lost_after: int = field(default_factory=lambda: cfg.get("masks.lost_after"))

    # associator
    speed_slow: float = field(default_factory=lambda: cfg.get("masks.associator.speed_slow"))
    speed_medium: float = field(default_factory=lambda: cfg.get("masks.associator.speed_medium"))
    weights_static: tuple = field(default_factory=lambda: cfg.get("masks.associator.weights_static"))
    weights_medium: tuple = field(default_factory=lambda: cfg.get("masks.associator.weights_medium"))
    weights_fast: tuple = field(default_factory=lambda: cfg.get("masks.associator.weights_fast"))
    score_threshold: float = field(default_factory=lambda: cfg.get("masks.associator.score_threshold"))

    # motion — apply_detection
    smooth_alpha: float = field(default_factory=lambda: cfg.get("masks.motion.smooth_alpha"))
    dead_zone: float = field(default_factory=lambda: cfg.get("masks.motion.dead_zone"))
    velocity_dead_zone: float = field(default_factory=lambda: cfg.get("masks.motion.velocity_dead_zone"))
    dt_slow_max: float = field(default_factory=lambda: cfg.get("masks.motion.dt_slow_max"))
    teleport_thresh: float = field(default_factory=lambda: cfg.get("masks.motion.teleport_thresh"))
    vx_max: float = field(default_factory=lambda: cfg.get("masks.motion.vx_max"))
    vy_max: float = field(default_factory=lambda: cfg.get("masks.motion.vy_max"))
    vw_max: float = field(default_factory=lambda: cfg.get("masks.motion.vw_max"))
    vh_max: float = field(default_factory=lambda: cfg.get("masks.motion.vh_max"))

    # motion — predict_position
    dt_cap:       float = field(default_factory=lambda: cfg.get("masks.motion.dt_cap"))
    damping_rate: float = field(default_factory=lambda: cfg.get("masks.motion.damping_rate"))
    min_mask_size:        float = field(default_factory=lambda: cfg.get("masks.motion.min_mask_size"))

    # tracker — Hash history
    hash_history_max: int = field(default_factory=lambda: cfg.get("masks.hash_history_max"))

    def __post_init__(self):
        # YAML parse [a, b] en list → on force en tuple (immutable, conforme à l'annotation)
        self.weights_static = tuple(self.weights_static)
        self.weights_medium = tuple(self.weights_medium)
        self.weights_fast = tuple(self.weights_fast)

        if len(self.weights_static) != 2 or len(self.weights_medium) != 2 or len(self.weights_fast) != 2:
            raise ValueError("TrackerConfig: weights_* doivent contenir exactement 2 valeurs (w_iou, w_hash)")

        if self.ttl_default < self.lost_after:
            raise ValueError(
                f"TrackerConfig: ttl_default ({self.ttl_default}) < lost_after ({self.lost_after}) — "
                "un mask peut expirer par TTL avant d'être marqué LOST. "
                "Augmente ttl_default ou diminue lost_after dans config.yaml."
            )
        if self.confirm_after >= self.lost_after:
            raise ValueError(
                f"TrackerConfig: confirm_after ({self.confirm_after}) >= lost_after ({self.lost_after}) — "
                "un mask ne pourra jamais être confirmé avant d'être perdu."
            )

@dataclass
class Detection:
    rect: tuple
    phash: Optional[int] = None
    source: str = "slow"
    confidence: float = 1.0
    template: Optional[object] = None
    scores: dict = field(default_factory=dict)