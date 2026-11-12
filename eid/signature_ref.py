# eid/signature_ref.py
from typing import Optional, Any
from dataclasses import dataclass
from enum import Enum, StrEnum
from eid.observation import Observation
from core.signature import sig_get, MAP_REF_H

# Constantes de troncature (alignées avec signature.py)
MAP_CLIP_LEFT = 1
MAP_CLIP_RIGHT = 2

def dim_distance(obs_w: float, obs_h: float, ref_w: float, ref_h: float) -> float:
    """d_dim (étape 4) — ratio de taille saturé, symétrique, borné [0,1].
    Utilise le ratio max(w/w', w'/w) - 1, saturé à 1.0 — une boîte deux fois
    plus grande ou deux fois plus petite atteint déjà le coût maximal.
    """
    if ref_w <= 0.0 or ref_h <= 0.0 or obs_w <= 0.0 or obs_h <= 0.0:
        return 1.0
    rw_ratio = max(obs_w / ref_w, ref_w / obs_w) - 1.0
    rh_ratio = max(obs_h / ref_h, ref_h / obs_h) - 1.0
    return min(1.0, max(rw_ratio, rh_ratio))

@dataclass(frozen=True, slots=True)
class QualityScore:
    total: float
    f_clip: float | None = None
    f_h: float | None = None
    f_fill: float | None = None
    f_hs: float | None = None

class AnchorState(Enum):
    PROBATION = "probation"
    LOCKED = "locked"


class PromoOutcome(StrEnum):
    early = "early"
    window = "window"
    blocked_identity = "blocked_identity"
    blocked_quality = "blocked_quality"
    locked_no_promo = "locked_no_promo"


@dataclass(frozen=True, slots=True)
class PromotionResult:
    outcome: PromoOutcome | None = None
    frame: int | None = None
    blocked_cost: float | None = None


def _calculate_quality_score(obs: Observation) -> QualityScore:
    """Calcule le score historique de qualité; une signature vide vaut -1.0."""
    if not obs.signature:
        return QualityScore(-1.0)
    _, _, w, h = obs.rect
    area = float(w * h)
    if area <= 0.0:
        return QualityScore(-1.0)
    sig_map_clip = sig_get(obs.signature, "sig_map_clip", None)
    sig_map_l = float(sig_get(obs.signature, "sig_map_l", 1.0))
    base_score = area * sig_map_l
    if sig_map_clip == 0:
        base_score += 10_000_000.0
    return QualityScore(base_score)

def _dispersion_for_family_e(costs: list[float], family: str, cfg) -> Optional[float]:
    """
    Dispersion par famille (max - min) avec seuil d'échantillons minimal.
    Lit le seuil n_min_<family> sur cfg selon family :
        "hs"    -> cfg.n_min_hs
        "shape" -> cfg.n_min_shape
        "map"   -> cfg.n_min_map
    """
    if family == "hs":
        n_min = cfg.n_min_hs
    elif family == "shape":
        n_min = cfg.n_min_shape
    elif family == "map":
        n_min = cfg.n_min_map
    else:
        raise ValueError(f"unknown family: {family!r}")
    if len(costs) < n_min:
        return None
    return max(costs) - min(costs)

class AnchorManager:
    """Référence de cluster: probation puis verrouillage, sans ancre vide."""
    __slots__ = ("best_obs", "best_score", "best_quality", "update_count", "state",
                 "n_obs", "first_ts", "last_ts", "last_result", "promo_outcome", "promo_frame")

    def __init__(self):
        self.best_obs: Optional[Observation] = None
        self.best_score = -1.0
        self.best_quality: Optional[QualityScore] = None
        self.update_count = 0
        self.state = AnchorState.PROBATION
        self.n_obs = 0
        self.first_ts: Optional[float] = None
        self.last_ts: Optional[float] = None
        self.last_result = PromotionResult()
        self.promo_outcome: PromoOutcome | None = None
        self.promo_frame: int | None = None

    def try_update(self, obs: Observation, config=None) -> PromotionResult:
        if self.first_ts is None:
            self.first_ts = obs.ts
        self.last_ts = obs.ts
        self.n_obs += 1
        if self.state is AnchorState.LOCKED:
            self.last_result = PromotionResult()
            return self.last_result
        if not obs.signature:
            span = 0.0 if self.first_ts is None or self.last_ts is None else max(0.0, self.last_ts - self.first_ts)
            if config is not None and (self.n_obs >= config.promo_window_n_obs or span >= config.promo_window_span_s) and self.best_obs is None:
                self.state = AnchorState.LOCKED
                self.last_result = PromotionResult(PromoOutcome.locked_no_promo, obs.frame_id)
            else:
                self.last_result = PromotionResult(PromoOutcome.blocked_identity, obs.frame_id)
            return self.last_result
        candidate = _calculate_quality_score(obs)
        if candidate.total < 0.0:
            self.last_result = PromotionResult(PromoOutcome.blocked_quality, obs.frame_id)
            return self.last_result
        if self.best_obs is None or candidate.total > self.best_score:
            self.best_score = candidate.total
            self.best_quality = candidate
            self.best_obs = obs
            self.update_count += 1
        sig = self.best_obs.signature
        clip = sig_get(sig, "sig_map_clip", None)
        h_txt = sig_get(sig, "sig_map_h_txt", None)
        if clip == 0 and h_txt is not None and h_txt >= MAP_REF_H:
            self.state = AnchorState.LOCKED
            self.promo_outcome = PromoOutcome.early
            self.promo_frame = self.best_obs.frame_id
            self.last_result = PromotionResult(PromoOutcome.early, self.best_obs.frame_id)
            return self.last_result
        span = 0.0 if self.first_ts is None or self.last_ts is None else max(0.0, self.last_ts - self.first_ts)
        if config is not None and (self.n_obs >= config.promo_window_n_obs or span >= config.promo_window_span_s):
            self.state = AnchorState.LOCKED
            self.last_result = PromotionResult(PromoOutcome.window, self.best_obs.frame_id)
            return self.last_result
        self.last_result = PromotionResult()
        return self.last_result

    def freeze_promotion(self) -> None:
        """Fige window/locked_no_promo une seule fois à la finalisation."""
        if self.promo_outcome is not None:
            return
        self.promo_outcome = PromoOutcome.window if self.best_obs is not None else PromoOutcome.locked_no_promo
        self.promo_frame = self.best_obs.frame_id if self.best_obs is not None else None

    @property
    def anchor(self) -> Optional[Observation]:
        return self.best_obs

    @property
    def anchor_locked(self) -> bool:
        return self.state is AnchorState.LOCKED

