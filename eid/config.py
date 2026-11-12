# eid/config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class StreamConfig:
    # -- Identite (clusters_slow) --
    temporal_radius_slow_s: float = 1.0   # secondes (fenetre de rapprochement)
    temporal_radius_evict_s: float = 2.0  # secondes (delai avant finalisation/purge du cluster)
    k_pos_norm:            float = 3.0    # normalisation positionnelle : d_pos = dist / (k * diagonale)
    w_geo:                float = 0.3     # poids geometrie historique, non lu en C3
    w_pos:                float = 0.7     # part de d_pos au sein de w_geo ; w_pos + w_dim == 1 (valide)
    w_dim:                float = 0.3     # part de d_dim au sein de w_geo ; w_pos + w_dim == 1 (valide)
    c_max:                float = 0.35    # seuil du cout composite [0,1] (a calibrer, etape 9)
    # -- Continuite (clusters_fast) --
    temporal_radius_fast_s: float = 0.35  # secondes
    # -- Detection d'anomalies geometriques --
    max_travel_px_per_s:     float = 25.0 * 120.0
    max_scale_ratio:         float = 1.6   # ratio w ou h avant flag
    fast_radius_floor_px:  float = 24.0   # rayon plancher quand gap_s >= 0
    fast_ambiguity_ratio:  float = 0.75
    # -- Distance signature (Smart Signature) - Lot C3a --
    # Supprimes : sig_weights, map_ambiguity_min, map_ambiguity_max (C3.5),
    # map_row_tolerance (C3-g REPORTE), min_sig_coverage (couvert par conf_bands).
    # Gating C3.7 :
    hs_px_min:            int   = 64    # plan erode ssi mode[1]=="eroded" ET px[1]>=hs_px_min, des DEUX cotes
    hs_raw_penalty:       float = 0.85  # multiplicateur confiance : plan brut OU sig_hs absent
    hs_gate_max:          float = 0.55  # au-dela : gate "hs_divergence"
    shape_w_d:            float = 0.45  # poids inter-familles (somme = 1.0)
    shape_w_c:            float = 0.35
    shape_w_y:            float = 0.20
    occupancy_min:        float = 0.25  # PROVISOIRE : gate "occupancy" si min(occ_ref, occ_obs) < seuil ; 0.0 = desactive
    map_l_ratio_min:      float = 0.60  # sous ce ratio : gate "l_ratio"
    map_h_txt_gate:       int   = 6     # gate dur
    map_h_txt_full:       int   = 12    # confiance pleine au-dela
    min_cost_families:    int   = 2     # 4.3 : familles (shape, map) requises des deux cotes, sinon "no_descriptor"
    # Poids de fusion C3.3 (renormalises sur composantes presentes) :
    w_shape:              float = 0.40
    w_map:                float = 0.60
    # -- Lots temporels (etape 6) --
    epsilon_close_s:      float = 0.15   # fenetre d'agregation d'un lot (a calibrer, etape 9)
    batch_max_span_factor: int = 4
    # -- Dedup en lecture (batching.py) --
    dedup_ttl_s :       float = 2.0    # D3 - duree de retention d'une dedup_key.
    # -- Dispersion inter-familles (Lot E) --
    cluster_max_dispersion: float = 0.30  # A CALIBRER - seuil max de dispersion tolere avant flag cluster
    n_min_hs:    int = 5   # A CALIBRER - nb minimum d'echantillons famille histogramme (hs) pour dispersion fiable
    n_min_shape: int = 5   # A CALIBRER - nb minimum d'echantillons famille forme (shape) pour dispersion fiable
    n_min_map:   int = 5   # A CALIBRER - nb minimum d'echantillons famille mapping (map) pour dispersion fiable
    # -- Promotion de l'ancre (L12) --
    promo_window_n_obs: int = 8
    promo_window_span_s: float = 0.5
    promo_max_drift: Optional[float] = None
    # -- Perf / observabilite (Lot E) --
    active_clusters_warn_threshold: int = 200  # A CALIBRER - seuil de clusters actifs declenchant un warning perf

    def __post_init__(self):
        assert abs(self.w_pos + self.w_dim - 1.0) < 1e-6, "w_pos + w_dim doit valoir 1"
        assert 0.0 < self.c_max <= 1.0, "c_max doit etre dans (0, 1]"
        # NOTE: assertion c_max_fast retirees - parametre orphelin depuis A2.
        assert self.temporal_radius_evict_s > self.temporal_radius_slow_s, "L'eviction doit avoir lieu apres le rayon de rapprochement"
        assert self.temporal_radius_slow_s > self.temporal_radius_fast_s > self.epsilon_close_s, \
            (f"Ordre requis : temporal_radius_slow_s({self.temporal_radius_slow_s}) > "
             f"temporal_radius_fast_s({self.temporal_radius_fast_s}) > "
             f"epsilon_close_s({self.epsilon_close_s})")
        # -- Gating cinematique (A2) --
        assert self.max_travel_px_per_s > 0.0, "max_travel_px_per_s doit etre > 0"
        assert self.fast_radius_floor_px >= 0.0, "fast_radius_floor_px doit etre >= 0"
        assert 0.0 < self.fast_ambiguity_ratio <= 1.0, "fast_ambiguity_ratio doit etre dans (0, 1]"
        # -- C3.7 : gating signature --
        assert self.hs_px_min >= 0, "hs_px_min doit etre >= 0"
        assert 0.0 < self.hs_raw_penalty <= 1.0, "hs_raw_penalty doit etre dans (0, 1]"
        assert 0.0 < self.hs_gate_max <= 1.0, "hs_gate_max doit etre dans (0, 1]"
        s = self.shape_w_d + self.shape_w_c + self.shape_w_y
        assert abs(s - 1.0) < 1e-6, "shape_w_d + shape_w_c + shape_w_y doit valoir 1"
        assert 0.0 <= self.occupancy_min <= 1.0, "occupancy_min doit etre dans [0, 1]"
        assert 0.0 < self.map_l_ratio_min <= 1.0, "map_l_ratio_min doit etre dans (0, 1]"
        assert 0 <= self.map_h_txt_gate < self.map_h_txt_full, "map_h_txt_gate < map_h_txt_full requis"
        assert 1 <= self.min_cost_families <= 2, "min_cost_families doit etre dans [1, 2]"
        # -- C3.3 : poids de fusion --
        ws = self.w_shape + self.w_map
        assert abs(ws - 1.0) < 1e-6, "w_shape + w_map doit valoir 1"
        # -- Dispersion (Lot E) --
        assert self.cluster_max_dispersion > 0.0, "cluster_max_dispersion doit etre > 0"
        assert min(self.n_min_map, self.n_min_shape, self.n_min_hs) >= 1 \
            and self.n_min_map >= max(self.n_min_shape, self.n_min_hs), \
            "INV-E6 : n_min_map >= max(n_min_shape, n_min_hs) et tous >= 1 requis"
        assert self.active_clusters_warn_threshold > 0, "active_clusters_warn_threshold doit etre > 0"