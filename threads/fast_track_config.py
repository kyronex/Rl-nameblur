# threads/fast_track_config.py
from __future__ import annotations
from dataclasses import dataclass, field
from config import cfg


@dataclass
class FastTrackConfig:
    """
    Snapshot immuable de la configuration du FastTrackThread.
    Construit via `cfg.get(...)` au moment de l'instanciation.
    Reload effectué entre deux ticks worker via `reload_config()`.
    """
    # Frame / écran
    screen_w: int = field(default_factory=lambda: cfg.get("screen.width"))
    screen_h: int = field(default_factory=lambda: cfg.get("screen.height"))
    # NCC / ROI
    roi_margin:    int   = field(default_factory=lambda: cfg.get("detect.fast.roi_margin"))
    ncc_threshold: float = field(default_factory=lambda: cfg.get("detect.fast.ncc_threshold"))
    # Stale tracking
    max_stale_frames: int = field(default_factory=lambda: cfg.get("detect.fast.max_stale_frames"))
    # Worker loop
    event_timeout_s: float = field(default_factory=lambda: cfg.get("detect.fast.event_timeout_s"))
    # Adaptive margin (pour _adaptive_margin)
    am_base:   float = field(default_factory=lambda: cfg.get("masks.adaptive_margin.base"))
    am_factor: float = field(default_factory=lambda: cfg.get("masks.adaptive_margin.factor"))
    am_min:    int   = field(default_factory=lambda: cfg.get("masks.adaptive_margin.min"))
    am_max:    int   = field(default_factory=lambda: cfg.get("masks.adaptive_margin.max"))

    def __post_init__(self):
        # Bornes adaptive_margin cohérentes
        if not (self.am_min <= self.am_base <= self.am_max):
            raise ValueError(
                f"FastTrackConfig: adaptive_margin invalide "
                f"(min={self.am_min}, base={self.am_base}, max={self.am_max}) "
                f"— attendu min ≤ base ≤ max"
            )
        if self.am_factor < 0:
            raise ValueError(f"FastTrackConfig: am_factor doit être ≥ 0 (reçu {self.am_factor})")
        # NCC threshold dans [0, 1]
        if not (0.0 <= self.ncc_threshold <= 1.0):
            raise ValueError(
                f"FastTrackConfig: ncc_threshold doit être ∈ [0, 1] "
                f"(reçu {self.ncc_threshold})"
            )
        # roi_margin ≥ 0
        if self.roi_margin < 0:
            raise ValueError(f"FastTrackConfig: roi_margin doit être ≥ 0 (reçu {self.roi_margin})")
        # max_stale_frames ≥ 0
        if self.max_stale_frames < 0:
            raise ValueError(
                f"FastTrackConfig: max_stale_frames doit être ≥ 0 "
                f"(reçu {self.max_stale_frames})"
            )
        # event_timeout_s > 0
        if self.event_timeout_s <= 0:
            raise ValueError(
                f"FastTrackConfig: event_timeout_s doit être > 0 "
                f"(reçu {self.event_timeout_s})"
            )
        # Écran cohérent
        if self.screen_w <= 0 or self.screen_h <= 0:
            raise ValueError(
                f"FastTrackConfig: screen_w/h invalides "
                f"({self.screen_w}×{self.screen_h})"
            )
