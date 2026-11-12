# eid/cluster.py
from dataclasses import dataclass, field
from typing import Optional, List, Deque, Dict, Any
from eid.sighting import Sighting
from collections import deque
from enum import Enum, auto

from eid.observation import Observation # (ou Observation selon ton import)
from eid.signature_ref import AnchorManager, PromotionResult # 🟢 IMPORT DE L'ÉTAPE 4
from core.signature import serialize_signature

class ClusterState(Enum):
    ACTIVE = auto()

@dataclass
class ClusterSlow:
    cluster_id:       int
    eid:              int
    observations:     Deque[Observation] = field(default_factory=lambda: deque(maxlen=50))
    n_obs_total:      int = 0
    created_ts:       float = 0.0
    created_at_frame: int = 0
    last_seen_frame:  int = 0
    last_seen_ts:     float = 0.0
    state: ClusterState = ClusterState.ACTIVE
    prime_sighting: Optional[Sighting] = None
    # 🟢 REMPLACE LES VIEUX ATTRIBUTS ref_signature, ref_history, etc.
    anchor: AnchorManager = field(default_factory=AnchorManager)

    @property
    def last_obs(self) -> Observation:
        return self.observations[-1]

    def add_observation(self, obs: Observation, config=None) -> PromotionResult:
        if self.n_obs_total == 0:
            self.created_ts = obs.ts
            self.created_at_frame = obs.frame_id
        self.observations.append(obs)
        self.n_obs_total += 1
        self.last_seen_ts = obs.ts
        self.last_seen_frame = obs.frame_id
        # 🟢 MISE À JOUR DE LA BEST ANCHOR O(1)
        return self.anchor.try_update(obs, config)

    def finalize_promotion(self) -> None:
        """Fige le résultat de probation avant compilation de la sortie."""
        self.anchor.freeze_promotion()
        self.anchor.state = self.anchor.state.LOCKED

    def compile_prime_sighting(self) -> Sighting:
        """
        Snapshot à la demande (E.3, INV-E7).

        N'altère JAMAIS `self.state` et ne finalise pas le cluster.
        La finalisation est signalée par `StreamEIDBuilder` lors de l'éviction
        (ou du flush) : le cluster est retiré, ce snapshot est compilé, puis
        transmis au callback s'il est configuré. Appelable N fois sans changer
        le résultat.

        Forme finale du dict (D.1, 7 sous-blocs) :
          - "eid"       : identité numérique de cluster (commun)
          - "identity"  : C1 — ancre, verrouillage, dérive             (PLACEHOLDER)
          - "dispersion": C2 — homogénéité du cluster                 (PLACEHOLDER)
          - "signal"    : C3 — confidence agrégée de paire            (migré ici)
          - "geometry"  : C4 — gate géométrique, compteurs bruts      (PLACEHOLDER)
          - "tracking"  : A3-b/B.3 — mask_uid, switches               (migré ici)
          - "lifecycle" : commun — span, cardinalité, frames          (déjà connu)

        Arbitrages L11 : `anchor_signature` est dans `identity` (L11,
        ARB-L11-01), la qualité est exportée dans `identity` en L12, `stats` est supprimé, et
        `lifecycle` est construit par `Sighting.to_json()`.
        """
        if self.prime_sighting is not None:
            return self.prime_sighting

        # 🟢 RÉCUPÉRATION SÉCURISÉE DE L'ANCRE
        best_obs = self.anchor.anchor

        # --- "eid" : identifiant numérique (D.1, commun) -------------------
        # Le plan D.1 spécifie "eid": int au niveau racine — pas de sous-dict.
        eid_value = self.eid

        # --- "identity" (C1) — PLACEHOLDER ---------------------------------
        identity_block: Dict[str, Any] = {
            # PLACEHOLDER — lot C1 : contenu réel défini en C1.2/C1.3/C1.6.
            "anchor_signature": serialize_signature(best_obs.signature) if best_obs else None,  # L11 — ARB-L11-01
            "anchor_locked": self.anchor.anchor_locked,
            "anchor_quality": (self.anchor.best_quality.total if self.anchor.best_quality else None),
            "anchor_quality_components": {
                "f_clip": (self.anchor.best_quality.f_clip if self.anchor.best_quality else None),
                "f_h": (self.anchor.best_quality.f_h if self.anchor.best_quality else None),
                "f_fill": (self.anchor.best_quality.f_fill if self.anchor.best_quality else None),
                "f_hs": (self.anchor.best_quality.f_hs if self.anchor.best_quality else None),
            },
            "n_post_promo_outliers": None,      # reporté L12-P3
            "promo_outcome": self.anchor.promo_outcome.value if self.anchor.promo_outcome else None,
            "promo_frame": self.anchor.promo_frame,
            "n_promo_blocked_identity": None,
            "promo_blocked_max_cost": None,
        }

        # --- "dispersion" (C2) — PLACEHOLDER -------------------------------
        dispersion_block: Dict[str, Any] = {
            # PLACEHOLDER — lot C2 : contenu réel défini en C2.3/C2.4/C2.6.
            "at_admission":              None,  # PLACEHOLDER — lot C2 (C2.4)
            "at_admission_excl_outliers": None, # PLACEHOLDER — lot C2 (C2.4)
            "final":                     None,  # PLACEHOLDER — lot C2 (C2.4)
            "final_excl_outliers":       None,  # PLACEHOLDER — lot C2 (C2.4)
            "n_hs":    None,                        # PLACEHOLDER — lot C2 (C2.4)
            "n_shape": None,                        # PLACEHOLDER — lot C2 (C2.4)
            "n_map":   None,                        # PLACEHOLDER — lot C2 (C2.4)
        }

        # --- "signal" (C3) — migration des champs existants ----------------
        signal_block: Dict[str, Any] = {
            # PLACEHOLDER — lot C3 : confidence_mean / confidence_min / n_gate_*
            "confidence_mean":    None,  # PLACEHOLDER — lot C3 (C2.7)
            "confidence_min":     None,  # PLACEHOLDER — lot C3 (C2.7)
            "n_gate_hs":          None,    # PLACEHOLDER — lot C3 (C3.8)
            "n_gate_occupancy":   None,    # PLACEHOLDER — lot C3 (C3.8)
            "n_gate_h_txt":       None,    # PLACEHOLDER — lot C3 (C3.8)
            "n_gate_l_ratio":     None,    # PLACEHOLDER — lot C3 (C3.8)
        }

        # --- "geometry" (C4) — PLACEHOLDER ---------------------------------
        geometry_block: Dict[str, Any] = {
            # PLACEHOLDER — lot C4 : contenu réel défini en C4.1/C4.2/A3-c.
            "n_geo_pass":  None,    # PLACEHOLDER — lot C4
            "n_geo_fail":  None,    # PLACEHOLDER — lot C4
            "n_geo_na":    None,    # PLACEHOLDER — lot C4
            "contaminated": None,  # PLACEHOLDER — lot C4 (C1-d/A3-c)
        }

        # --- "tracking" (A3-b/B.3) — migration partielle --------------------
        tracking_block: Dict[str, Any] = {
            # PLACEHOLDER — lots A3-b / B.3 (cf. D.6).
            "mask_uids_seen":     None,  # PLACEHOLDER — lot A3-b (D5, maxlen aligné)
            "mask_uid_switches":  None,     # PLACEHOLDER — lot B.3 (n_uid_switch)
            "uid_agree":          None,     # PLACEHOLDER — lot B.3
            "uid_disagree":       None,     # PLACEHOLDER — lot B.3
        }

        self.prime_sighting = Sighting(
            eid=eid_value,
            first_frame=self.created_at_frame,
            last_frame=self.last_seen_frame,
            created_ts=self.created_ts,
            last_seen_ts=self.last_seen_ts,
            n_obs=self.n_obs_total,
            identity=identity_block,
            dispersion=dispersion_block,
            signal=signal_block,
            geometry=geometry_block,
            tracking=tracking_block,
        )
        # INV-E7 : PAS de mutation de self.state ici. La finalisation et
        # l'émission sont prises en charge par le builder.
        return self.prime_sighting

@dataclass
class ClusterFast:
    """Registre de CONTINUITÉ — observations slow + fast, piste géométrique."""
    cluster_id:       int
    eid:              Optional[int] = None
    # Stockage borné
    observations:     Deque[Observation] = field(default_factory=lambda: deque(maxlen=100))
    created_ts:       float = 0.0
    created_at_frame: int = 0
    last_seen_frame:  int = 0
    last_seen_ts:     float = 0.0
    adopted_at_frame: Optional[int] = None
    n_slow:           int = 0
    n_fast:           int = 0
    # État du cycle de vie
    state: ClusterState = ClusterState.ACTIVE
    @property
    def last_obs(self) -> Observation:
        return self.observations[-1]

    def add_observation(self, obs: Observation) -> None:
        """Point d'entrée UNIQUE pour alimenter la piste.

        Symétrique de `ClusterSlow.add_observation`. Encapsuler ici rend
        structurellement impossible d'écraser le `deque(maxlen=100)` par une
        list depuis l'appelant (fuite mémoire en live) et garantit que
        `n_slow` / `n_fast` restent cohérents avec `observations`.

        `created_at_frame` est positionné sur la PREMIÈRE observation : le test
        porte sur `observations` évalué AVANT l'append. Ne pas déplacer ce bloc
        après l'append (le deque plafonne à 100, pas les compteurs n_slow/n_fast
        — ils divergeraient comme critère de "première obs").
        """
        if not self.observations:
            self.created_ts = obs.ts
            self.created_at_frame = obs.frame_id
        self.observations.append(obs)
        self.last_seen_ts = obs.ts
        self.last_seen_frame = obs.frame_id
        if obs.source == "slow":
            self.n_slow += 1
        else:
            self.n_fast += 1
