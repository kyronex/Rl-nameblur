# eid/cost.py
"""
Solveur pur pour la génération des matrices de coûts (Hungarian Assignment).
Agnostique du cycle de vie : aucune mutation d'état n'est effectuée ici.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Callable, Any, Tuple, Dict, Optional

from eid.observation import Observation
from eid.config import StreamConfig

if TYPE_CHECKING:
    from eid.eid_distance import SigDistance

# ─── CONSTANTES DE REJET ───────────────────────────────────────────────
# Coûts "infinis" pour le solveur hongrois.
# Distinguer la cause permet au post-traitement de mesurer la performance du gating.
BIG_COST_COST = 10.0   # Évalué par le moteur de coût mais rejeté (> c_max)
BIG_COST_GATE = 20.0   # Rejeté par le filtre temporel (jamais évalué)

# ─── STRUCTURES DE DONNÉES ─────────────────────────────────────────────

@dataclass(frozen=True)
class CalibRecord:
    """Enregistrement immutable d'une évaluation pour l'export de calibration."""
    obs_uid: int
    cluster_id: int
    gap_s: float
    dist: "SigDistance"
    cost: float = 0.0
    cost_sig: float = 0.0
    accepted: bool = True
    seq: int = 0
    frame_id: int = 0
    ts: float = 0.0
    eid: Optional[int] = None

@dataclass
class CostMatrix:
    """
    Résultat de l'évaluation complète d'un lot d'observations contre les clusters éligibles.
    Prêt à être consommé par scipy.optimize.linear_sum_assignment.
    """
    matrix: np.ndarray                                    # Matrice des coûts (n_obs x n_clusters)
    rows: List[Observation]                               # Mapping des lignes (index -> observation)
    col_ids: List[int]                                    # Mapping des colonnes (index -> cluster_id)
    payload: List[List[Optional[Tuple["SigDistance", float]]]]         # Métadonnées de calcul (sd, cost_sig) pour chaque cellule
    # Rapport d'exécution (remonte les informations à l'orchestrateur)
    n_gate_rejects: int = 0
    n_cost_rejects: int = 0
    n_cols_gate_only: int = 0        # C2 — colonnes rejetées UNIQUEMENT par le gate temporel
    n_cols_cost_rejected: int = 0    # C2 — colonnes évaluées mais rejetées par toutes les obs
    calib_pairs: List[CalibRecord] = field(default_factory=list)

# ─── LOGIQUE MÉTIER PURE ───────────────────────────────────────────────

def _get_cluster_ts(cluster: Any) -> float:
    """
    Extrait le timestamp de référence du cluster.
    S'adapte à la structure (privilégie l'ancre si disponible).
    """
    return getattr(cluster, "last_seen_ts", 0.0)

def _get_eligible_slow_clusters(config: StreamConfig,batch: List[Observation],clusters_slow: Dict[int, Any]) -> List[int]:
    """
    Filtre préliminaire (O(N*M)) : retourne les IDs des clusters
    qui sont à portée temporelle d'au moins UNE observation du lot.
    """
    eligible_ids = []
    for cid, cluster in clusters_slow.items():
        last_ts = _get_cluster_ts(cluster)
        # Un cluster est éligible s'il croise temporellement au moins une obs du batch
        for obs in batch:
            gap_s = obs.ts - last_ts
            if 0.0 <= gap_s <= config.temporal_radius_slow_s:
                eligible_ids.append(cid)
                break
    return sorted(eligible_ids)

def build_slow_cost_matrix(config: StreamConfig,batch: List[Observation],clusters_slow: Dict[int, Any],cost_fn: Callable[[Observation, Any], Optional[Tuple["SigDistance", float]]]) -> Optional[CostMatrix]:
    """
    Construit la matrice dense d'assignation pour l'identité (slow).
    Garantit l'absence d'effets de bord : l'orchestrateur se chargera de
    muter ses compteurs à partir du résultat retourné.
    """
    col_ids = _get_eligible_slow_clusters(config, batch, clusters_slow)
    n_rows, n_cols = len(batch), len(col_ids)
    if n_rows == 0 or n_cols == 0:
        return None
    # Allocation de la matrice (remplie de rejets temporels par défaut)
    matrix = np.full((n_rows, n_cols), BIG_COST_GATE, dtype=float)
    payload: List[List[Optional[Tuple[Any, float]]]] = [[None] * n_cols for _ in range(n_rows)]
    n_gate_rejects = 0
    n_cost_rejects = 0
    calib_pairs = []
    for i, obs in enumerate(batch):
        for j, cid in enumerate(col_ids):
            cluster = clusters_slow[cid]
            gap_s = obs.ts - _get_cluster_ts(cluster)
            # 1. Gating temporel strict
            if gap_s < 0.0 or gap_s > config.temporal_radius_slow_s:
                n_gate_rejects += 1
                continue
            # 2. Déballage (sd: SigDistance, cost_sig: float)
            pair = cost_fn(obs, cluster)
            if pair is None:
                n_cost_rejects += 1
                matrix[i, j] = BIG_COST_COST
                continue
            sd, cost_sig = pair
            accepted = (not sd.gated) and cost_sig <= config.c_max
            # 3. Construction du CalibRecord (toujours populée)
            calib_pairs.append(CalibRecord(
                obs_uid=obs.uid,
                cluster_id=cid,
                gap_s=gap_s,
                dist=sd,
                cost=cost_sig,
                cost_sig=cost_sig,
                accepted=accepted,
                seq=obs.seq,
                frame_id=obs.frame_id,
                ts=obs.ts,
                eid=getattr(cluster, "eid", None),
            ))
            # 4. Métadonnées de cellule (sd, cost_sig)
            payload[i][j] = (sd, cost_sig)
            # 5. Gating du coût composite
            if not accepted:
                n_cost_rejects += 1
                matrix[i, j] = BIG_COST_COST
                continue
            # 6. Validation de la paire
            matrix[i, j] = cost_sig
    # C2 — diagnostic par colonne, sémantique désambiguïsée.
    #   gate_only  : colonne jamais évaluée (toutes les obs hors fenêtre)
    #   cost_rejected : colonne évaluée mais rejetée par toutes les obs
    #                   → symptôme d'un c_max trop strict ou d'une signature faible
    col_gate_only = (matrix >= BIG_COST_GATE).all(axis=0)
    col_no_match = (matrix >= BIG_COST_COST).all(axis=0)
    n_cols_gate_only = int(col_gate_only.sum())
    n_cols_cost_rejected = int((col_no_match & ~col_gate_only).sum())

    return CostMatrix(
        matrix=matrix,
        rows=list(batch),
        col_ids=col_ids,
        payload=payload,
        n_gate_rejects=n_gate_rejects,
        n_cost_rejects=n_cost_rejects,
        n_cols_gate_only=n_cols_gate_only,
        n_cols_cost_rejected=n_cols_cost_rejected,
        calib_pairs=calib_pairs
    )