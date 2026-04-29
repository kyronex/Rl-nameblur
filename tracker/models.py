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
    weights_source_slow: tuple = field(default_factory=lambda: cfg.get("masks.associator.weights_source_slow"))
    weights_source_fast: tuple = field(default_factory=lambda: cfg.get("masks.associator.weights_source_fast"))
    match_score_min_slow: float = field(default_factory=lambda: cfg.get("masks.associator.match_score_min_slow"))
    match_score_min_fast: float = field(default_factory=lambda: cfg.get("masks.associator.match_score_min_fast"))
    source_confidence_slow: float = field(default_factory=lambda: cfg.get("masks.associator.source_confidence_slow"))
    source_confidence_fast: float = field(default_factory=lambda: cfg.get("masks.associator.source_confidence_fast"))
    geo_gate_base_radius_px: float = field(default_factory=lambda: cfg.get("masks.associator.geo_gate_base_radius_px"))
    geo_gate_velocity_k: float = field(default_factory=lambda: cfg.get("masks.associator.geo_gate_velocity_k"))
    geo_gate_dt_ref: float = field(default_factory=lambda: cfg.get("masks.associator.geo_gate_dt_ref"))

    # motion — apply_detection
    smooth_alpha: float = field(default_factory=lambda: cfg.get("masks.motion.smooth_alpha"))
    dead_zone_min_px: float = field(default_factory=lambda: cfg.get("masks.motion.dead_zone_min_px"))
    dead_zone_rel: float = field(default_factory=lambda: cfg.get("masks.motion.dead_zone_rel"))
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
    hash_top_k: int = field(default_factory=lambda: cfg.get("masks.hash_top_k"))

    def __post_init__(self):
        # YAML parse [a, b] en list → on force en tuple (immutable, conforme à l'annotation)
        self.weights_source_slow = tuple(self.weights_source_slow)
        self.weights_source_fast = tuple(self.weights_source_fast)

        if len(self.weights_source_slow) != 2 or len(self.weights_source_fast) != 2:
            raise ValueError("TrackerConfig: weights_source_* doivent contenir exactement 2 valeurs (w_iou, w_hash)")

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
        EPS = 1e-6
        for name, w in (("weights_source_slow", self.weights_source_slow),
                        ("weights_source_fast", self.weights_source_fast)):
            s = sum(w)
            if abs(s - 1.0) > EPS:
                raise ValueError(
                    f"TrackerConfig: {name}={w} doit sommer à 1.0 (actuel: {s:.4f}). "
                    "Les poids (w_iou, w_hash) sont une pondération convexe : "
                    "ajuste config.yaml pour que w_iou + w_hash == 1.0."
                )

@dataclass(frozen=True, slots=True)
class MatchScore:
    """
    Résultat d'évaluation d'une paire (mask, detection) par l'associator.

    Contrat :
        - iou, hash, total ∈ [0.0, 1.0]
        - total = w_iou * iou + w_hash * hash (poids selon detection.source)
        - gated=True ⇒ paire rejetée par gating géométrique (total non significatif)
        - reason : étiquette courte de décision ("ok", "iou_only", "geo_gate", ...)

    Construction : utiliser les factories `gated_score()`, `iou_only()`, `composite()`
    plutôt que le constructeur direct.
    """
    iou:    float
    hash:   float
    total:  float
    gated:  bool
    reason: str

    @classmethod
    def gated_score(cls) -> "MatchScore":
        """Paire rejetée par le gate géométrique. Retourne un singleton partagé."""
        return _GATED_SCORE

    @classmethod
    def iou_only(cls, iou: float) -> "MatchScore":
        """Score basé uniquement sur l'IoU (pas d'historique hash exploitable)."""
        return cls(iou=iou, hash=0.0, total=iou, gated=False, reason="iou_only")

    @classmethod
    def composite(cls, iou: float, hsim: float, w_iou: float, w_hash: float) -> "MatchScore":
        """Score pondéré IoU + hash perceptuel."""
        total = w_iou * iou + w_hash * hsim
        return cls(iou=iou, hash=hsim, total=total, gated=False, reason="ok")

# Singleton pour le cas gated (immuable grâce à frozen=True, safe à partager)
_GATED_SCORE = MatchScore(iou=0.0, hash=0.0, total=0.0, gated=True, reason="geo_gate")

@dataclass
class Detection:
    rect: tuple
    phash: Optional[int] = None
    source: str = "slow"
    confidence: float = 1.0
    template: Optional[object] = None
    scores: dict = field(default_factory=dict)