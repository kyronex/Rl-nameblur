# tracker/models.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrackerConfig:
    iou_threshold: float = 0.3
    hash_weight: float = 0.4
    iou_weight: float = 0.6
    max_masks: int = 20
    ttl_default: int = 5
    confirm_after: int = 3
    lost_after: int = 5
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
    motion_dt_slow_max: float = 0.5        # secondes
    motion_teleport_thresh: float = 300.0   # pixels
    motion_vx_max: float = 4000.0           # px/s
    motion_vy_max: float = 2000.0           # px/s

    # motion — predict_position
    predict_dt_cap: float = 0.10            # secondes
    predict_damping_rate: float = 2.0       # damping = 1 - dt * rate


@dataclass
class Detection:
    rect: tuple
    phash: Optional[int] = None
    source: str = "slow"
