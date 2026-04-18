# tracker/models.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TrackerConfig:
    # Frame / écran
    screen_w: int = 1920
    screen_h: int = 1080

    max_masks: int = 20
    speed_slow: float = 10.0
    speed_medium: float = 50.0
    weights_static: tuple = (0.6, 0.4)
    weights_medium: tuple = (0.5, 0.5)
    weights_fast: tuple = (0.8, 0.2)
    score_threshold: float = 0.3

    # motion — apply_detection
    motion_smooth_alpha: float = 1.0
    motion_dead_zone: float = 3.0
    motion_velocity_dead_zone: float = 10.0
    motion_dt_slow_max: float = 0.5         # secondes
    motion_teleport_thresh: float = 300.0   # pixels
    motion_vx_max: float = 4000.0           # px/s
    motion_vy_max: float = 2000.0           # px/s
    motion_vw_max: float = 1000.0           # px/s — vitesse max changement largeur
    motion_vh_max: float = 500.0            # px/s — vitesse max changement hauteur

    # motion — predict_position
    predict_dt_cap: float = 0.10            # secondes
    predict_damping_rate: float = 2.0       # damping = 1 - dt * rate
    min_mask_size: float = 10.0             # px — taille min prédite w/h

    # registry / lifecycle
    ttl_default: int = 500
    confirm_hits: int = 1
    lost_after: int = 500

    # Hash history
    hash_history_max: int = 5

@dataclass
class Detection:
    rect: tuple
    phash: Optional[int] = None
    source: str = "slow"
    confidence: float = 1.0
    template: Optional[object] = None
    scores: dict = field(default_factory=dict)