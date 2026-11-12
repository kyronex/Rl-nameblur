# eid/eid_distance.py
"""Distance signature C3 (Lot C3a).

SigDistance (contrat C3.0) et sig_distance() conforme a INV-D1..D4.
Gates sequentiels, mutuellement exclusifs par construction (P1) :
  1. disponibilite (4.1/4.3) -> "no_descriptor"
  2. occupancy               -> "occupancy"
  3. hs                      -> "hs_divergence"
  4. map : l_ratio           -> "l_ratio"
  5. map : h_txt             -> "h_txt"
Calcul paresseux (P2) : chaque composante est calculee juste avant
le gate qui la consomme ; le glissement map n'est paye que par les
paires ayant franchi tous les gates.
Regle 4.4 : descripteur absent => None => ignore.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from eid.config import StreamConfig

# --- Constantes internes (alignees sur core/signature.py) ---
_HS_PLAN_RAW: int = 0
_HS_PLAN_ERODED: int = 1
_SHAPE_BANDS: int = 16          # signature.SHAPE_BANDS
_SHAPE_SENTINEL: float = -1.0   # signature.SHAPE_EMPTY_SENTINEL (c_k / sy_k)

_DIAG_FIELDS: Tuple[str, ...] = (
    "d_hs", "d_shape", "d_map", "occ_min",
    "hs_plan", "map_offset", "l_ratio", "h_txt_min", "n_bands_valid",
)

@dataclass(frozen=True, slots=True)
class SigDistance:
    """Resultat d'une comparaison signature <-> signature.

    INVARIANT : cost est None SI ET SEULEMENT SI gated is True.
    """
    cost:        Optional[float]   # [0,1] ou None si gate
    gated:       bool
    gate_reason: Optional[str]     # "no_descriptor" | "occupancy" | "hs_divergence" | "l_ratio" | "h_txt"
                                   # (+ "no_anchor" | "geo_shortcircuit" emis par builder.py)
    confidence:  float             # [0,1] - fiabilite de la MESURE, jamais fusionnee au cost

    # Composantes elementaires - None si indisponible OU non calculee (paresseux)
    d_hs:        Optional[float]
    d_shape:     Optional[float]
    d_map:       Optional[float]
    occ_min:     Optional[float]   # min(occ_ref, occ_obs), None si shape absent

    # Diagnostic
    hs_plan:     Optional[str]     # "raw" | "eroded" | None (hs absent)
    map_offset:  Optional[int]     # decalage retenu par le glissement
    l_ratio:     Optional[float]
    h_txt_min:   Optional[int]
    n_bands_valid: Optional[int]   # bandes pleines des deux cotes (sur 16)

    @classmethod
    def gated_only(cls, reason: str, **diag: Any) -> "SigDistance":
        """Resultat rejete : cost=None, confidence=0.0, diagnostics optionnels."""
        unknown = set(diag) - set(_DIAG_FIELDS)
        if unknown:
            raise TypeError(f"Champs de diagnostic inconnus : {sorted(unknown)}")
        fields: Dict[str, Any] = {k: None for k in _DIAG_FIELDS}
        fields.update(diag)
        return cls(cost=None, gated=True, gate_reason=reason, confidence=0.0, **fields)

# ====================================================================
#                          Helpers internes
# ====================================================================

def _shape_array(sig: Dict[str, Any]) -> Optional[np.ndarray]:
    s = sig.get("sig_shape")
    if s is None:
        return None
    a = np.asarray(s, dtype=np.float64).ravel()
    assert a.size == 3 * _SHAPE_BANDS, f"sig_shape de taille {a.size}, 48 attendu"
    return a

def _map_array(sig: Dict[str, Any]) -> Optional[np.ndarray]:
    m = sig.get("sig_map")
    if m is None or sig.get("sig_map_l") is None:
        return None
    a = np.asarray(m, dtype=np.float64)
    assert a.ndim == 2, f"sig_map de dimension {a.ndim}, 2 attendu"
    return a

def _h_txt_penalty(h_txt_min: int, gate: int, full: int) -> float:
    """Rampe lineaire [gate, full] -> [0.5, 1.0] (C3.2)."""
    if h_txt_min >= full:
        return 1.0
    if h_txt_min <= gate:
        return 0.5
    return 0.5 + 0.5 * (h_txt_min - gate) / (full - gate)

def _eroded_usable(sig: Dict[str, Any], px_min: int) -> bool:
    mode = sig.get("sig_hs_mode")
    px = sig.get("sig_hs_px")
    return (mode is not None and len(mode) > 1 and mode[1] == "eroded"
            and px is not None and len(px) > 1 and px[1] >= px_min)

def _select_hs_plan(sig1: Dict[str, Any], sig2: Dict[str, Any], px_min: int) -> int:
    """Plan erode ssi utilisable des DEUX cotes, sinon brut (arbitrage hs_px_min A)."""
    if _eroded_usable(sig1, px_min) and _eroded_usable(sig2, px_min):
        return _HS_PLAN_ERODED
    return _HS_PLAN_RAW

def _compute_hs_distance(hs1: Any, hs2: Any, plan: int) -> float:
    """L1 normalisee sur le plan choisi (variation totale, bornee [0,1])."""
    a = np.asarray(hs1, dtype=np.float64)
    b = np.asarray(hs2, dtype=np.float64)
    assert a.ndim == 3 and a.shape == b.shape, f"sig_hs incompatibles : {a.shape} / {b.shape}"
    a = a[plan].ravel()
    b = b[plan].ravel()
    diff = float(np.abs(a - b).sum())
    norm = float(np.abs(a).sum() + np.abs(b).sum())
    if norm <= 0.0:
        return 0.0
    return diff / norm

def _occupancy(shape: np.ndarray) -> float:
    """Fraction de bandes pleines (c != -1) sur 16 (arbitrage #2 A2)."""
    c = shape[_SHAPE_BANDS:2 * _SHAPE_BANDS]
    return float(np.count_nonzero(c != _SHAPE_SENTINEL)) / _SHAPE_BANDS

def _compute_shape_distance(s1: np.ndarray, s2: np.ndarray,
                            cfg: StreamConfig) -> Tuple[float, int]:
    """d sur 16 bandes ; c et sy sur le masque commun (1.1) ; repli 1.2a."""
    K = _SHAPE_BANDS
    d1, c1, y1 = s1[:K], s1[K:2 * K], s1[2 * K:]
    d2, c2, y2 = s2[:K], s2[K:2 * K], s2[2 * K:]
    dist_d = float(np.mean(np.abs(d1 - d2)))       # d = 0 sur bande vide : comparable
    full = (c1 != _SHAPE_SENTINEL) & (c2 != _SHAPE_SENTINEL)
    n_full = int(np.count_nonzero(full))
    if n_full > 0:
        dist_c = float(np.mean(np.abs(c1[full] - c2[full])))
        dist_y = float(np.mean(np.abs(y1[full] - y2[full])))
    else:
        dist_c, dist_y = 1.0, 1.0                  # repli 1.2a
    d_shape = cfg.shape_w_d * dist_d + cfg.shape_w_c * dist_c + cfg.shape_w_y * dist_y
    return float(d_shape), n_full

def _compute_map_sliding_distance(m1: np.ndarray, m2: np.ndarray) -> Tuple[float, int]:
    """Glissement horizontal (C3.1 d_map). Hauteur fixe MAP_N=16 (P4 B)."""
    assert m1.shape[0] == m2.shape[0], \
        f"sig_map de hauteurs differentes : {m1.shape[0]} / {m2.shape[0]}"
    if m1.shape[1] > m2.shape[1]:
        m1, m2 = m2, m1
    w1, w2 = m1.shape[1], m2.shape[1]
    best_dist = 1.0
    best_offset = 0
    for offset in range(w2 - w1 + 1):
        dist = float(np.mean(np.abs(m1 - m2[:, offset:offset + w1])))
        if dist < best_dist:
            best_dist = dist
            best_offset = offset
    return float(best_dist), int(best_offset)

# ====================================================================
#                       API publique C3
# ====================================================================

def sig_distance(sig_ref: Dict[str, Any], sig_obs: Dict[str, Any],
                 cfg: StreamConfig) -> SigDistance:
    """Distance de signature C3 - retourne un SigDistance type (C3.0).

    INV-D1 : aucun effet de bord, aucun etat global, aucune ecriture
             de compteur depuis cette fonction.
    INV-D2 : symetrie (a,b) <-> (b,a) a rtol=1e-6.
    INV-D3 : cost is None <=> gated is True.
    INV-D4 : cost et confidence ne fusionnent jamais.
    """
    # 1. Disponibilite (4.1 / 4.3)
    shape_ref, shape_obs = _shape_array(sig_ref), _shape_array(sig_obs)
    map_ref, map_obs = _map_array(sig_ref), _map_array(sig_obs)
    has_shape = shape_ref is not None and shape_obs is not None
    has_map = map_ref is not None and map_obs is not None
    n_families = int(has_shape) + int(has_map)
    if n_families == 0 or n_families < cfg.min_cost_families:
        return SigDistance.gated_only("no_descriptor")

    # 2. Occupancy (par cote, min)
    occ_min: Optional[float] = None
    if has_shape:
        occ_min = min(_occupancy(shape_ref), _occupancy(shape_obs))
        if occ_min < cfg.occupancy_min:
            return SigDistance.gated_only("occupancy", occ_min=occ_min)

    # 3. hs : gate seul, jamais dans le cout ; absent => ignore (4.4)
    d_hs: Optional[float] = None
    plan_label: Optional[str] = None
    hs_ref, hs_obs = sig_ref.get("sig_hs"), sig_obs.get("sig_hs")
    if hs_ref is not None and hs_obs is not None:
        plan_idx = _select_hs_plan(sig_ref, sig_obs, cfg.hs_px_min)
        plan_label = "eroded" if plan_idx == _HS_PLAN_ERODED else "raw"
        d_hs = _compute_hs_distance(hs_ref, hs_obs, plan_idx)
        if d_hs > cfg.hs_gate_max:
            return SigDistance.gated_only("hs_divergence", occ_min=occ_min,
                                          d_hs=d_hs, hs_plan=plan_label)

    # 4-5. Gates map (l_ratio puis h_txt) ; absent => ignore (4.4)
    l_ratio: Optional[float] = None
    h_txt_min: Optional[int] = None
    conf_h_txt = 1.0
    if has_map:
        l1, l2 = int(sig_ref["sig_map_l"]), int(sig_obs["sig_map_l"])
        l_ratio = float(min(l1, l2) / max(l1, l2))  # sig_map_l >= 1 (clamp signature.py)
        if l_ratio < cfg.map_l_ratio_min:
            return SigDistance.gated_only("l_ratio", occ_min=occ_min, d_hs=d_hs,
                                          hs_plan=plan_label, l_ratio=l_ratio)
        h_txt_min = int(min(sig_ref["sig_map_h_txt"], sig_obs["sig_map_h_txt"]))
        if h_txt_min < cfg.map_h_txt_gate:
            return SigDistance.gated_only("h_txt", occ_min=occ_min, d_hs=d_hs,
                                          hs_plan=plan_label, l_ratio=l_ratio,
                                          h_txt_min=h_txt_min)
        conf_h_txt = _h_txt_penalty(h_txt_min, cfg.map_h_txt_gate, cfg.map_h_txt_full)

    # 6. Composantes du cout (calculees seulement ici - P2)
    d_shape: Optional[float] = None
    n_bands_valid: Optional[int] = None
    if has_shape:
        d_shape, n_bands_valid = _compute_shape_distance(shape_ref, shape_obs, cfg)
    d_map: Optional[float] = None
    map_offset: Optional[int] = None
    if has_map:
        d_map, map_offset = _compute_map_sliding_distance(map_ref, map_obs)

    # 7. Fusion ponderee renormalisee (C3.3) - n_families >= 1 garanti par l'etape 1
    num, den = 0.0, 0.0
    if d_shape is not None:
        num += cfg.w_shape * d_shape; den += cfg.w_shape
    if d_map is not None:
        num += cfg.w_map * d_map; den += cfg.w_map
    cost = num / den

    # 8. Confidence (C3.4) - multiplication, jamais fusion au cost
    conf_hs = 1.0 if plan_label == "eroded" else cfg.hs_raw_penalty   # brut OU absent (P3)
    conf_bands = (n_bands_valid / float(_SHAPE_BANDS)) if n_bands_valid else 0.0
    confidence = float(conf_h_txt * conf_hs * conf_bands)

    return SigDistance(
        cost=float(cost), gated=False, gate_reason=None, confidence=confidence,
        d_hs=d_hs, d_shape=d_shape, d_map=d_map, occ_min=occ_min,
        hs_plan=plan_label, map_offset=map_offset, l_ratio=l_ratio,
        h_txt_min=h_txt_min, n_bands_valid=n_bands_valid,
    )