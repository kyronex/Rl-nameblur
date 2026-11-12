# eid/builder.py
"""
Orchestrateur EID (Streaming).
Pilote le cycle de vie des clusters actifs, interroge le solveur pur (cost.py)
et émet les sightings finalisés vers le consommateur aval. La finalisation
est signalée par l'éviction (ou `flush`) : le builder retire le cluster,
compile son snapshot et l'envoie au callback, sans modifier `cl.state`.
"""

from typing import Optional, Dict, List, Tuple, Callable, Any
import logging
from eid.sighting import Sighting
from scipy.optimize import linear_sum_assignment

# Imports des étapes précédentes (supposés à jour)
from eid.observation import Observation
from eid.config import StreamConfig
from eid.cost import build_slow_cost_matrix, BIG_COST_COST
from eid.eid_distance import sig_distance, SigDistance
from eid.signature_ref import dim_distance, PromotionResult, PromoOutcome

# Import conceptuel de l'étape 5
from eid.cluster import ClusterSlow, ClusterFast
from eid.counters import Counters, CounterKey, GaugeKey

logger = logging.getLogger(__name__)

EID_START = 1000

class StreamEIDBuilder:
    def __init__(self, config: StreamConfig,on_cluster_finalized: Optional[Callable[[Sighting], None]] = None,calib_writer: Optional[Any] = None,counters: Optional[Counters] = None):
        self.config = config
        self._clock_ts: float = float("-inf")
        self._emitted_sightings: List[Sighting] = []
        self.on_sighting = on_cluster_finalized
        self.on_cluster_finalized = on_cluster_finalized
        self.calib_writer = calib_writer
        self.clusters_slow: Dict[int, ClusterSlow] = {}
        self.clusters_fast: Dict[int, ClusterFast] = {}
        self._next_cluster_slow_id = 0
        self._next_cluster_fast_id = 0
        self._next_eid = EID_START
        # --- Compteurs globaux nettoyés ---
        self.counters: Counters = counters if counters is not None else Counters()

    # ─── CYCLE DE VIE & ÉVICTION ──────────────────────────────────────────

    def _finalize_slow(self, cid: int) -> None:
        """Finalise un cluster d'identité : retrait, empreinte, compteur, émission.
        Point unique partagé par l'éviction et flush() (lot 4)."""
        cl = self.clusters_slow.pop(cid)
        # Figer la probation avant de compiler la sortie finale.
        cl.finalize_promotion()
        # L13 : aucun résultat intermédiaire ou motif de blocage ne sort.
        assert cl.anchor.promo_outcome in {
            PromoOutcome.early, PromoOutcome.window, PromoOutcome.locked_no_promo
        }, "promo_outcome invalide à la finalisation"
        promotion_counters = {
            PromoOutcome.early: (CounterKey.REF_PROMOTIONS, CounterKey.REF_PROMO_EARLY),
            PromoOutcome.window: (CounterKey.REF_PROMOTIONS, CounterKey.REF_PROMO_WINDOW),
            PromoOutcome.locked_no_promo: (CounterKey.REF_LOCKED_NO_PROMO,),
        }
        for key in promotion_counters[cl.anchor.promo_outcome]:
            self.counters.bump(key)
        sighting = cl.compile_prime_sighting()
        self.counters.bump(CounterKey.CLUSTERS_SLOW_FINALIZED)
        if cl.anchor.anchor is None:
            self.counters.bump(CounterKey.SIGHTING_EMITTED_NO_ANCHOR)
        self._emitted_sightings.append(sighting)
        self.counters.bump(CounterKey.SIGHTING_EMITTED)
        if self.on_sighting is not None:
            try:
                self.on_sighting(sighting)
            except Exception:
                logger.exception("Sighting callback failed for EID %s", sighting.eid)
                self.counters.bump(CounterKey.SIGHTING_CALLBACK_ERROR)

    def _finalize_fast(self, cid: int) -> None:
        """Supprime une piste géométrique. Point unique partagé par l'éviction et
        flush() (lot 4 : réutilise EVICTED_FAST, pas de nouvelle clé)."""
        del self.clusters_fast[cid]
        self.counters.bump(CounterKey.EVICTED_FAST)

    def _evict_stale_slow(self, now_ts: float) -> None:
        """
        Retire les clusters d'identité sortis du rayon d'éviction.
        Compile le snapshot final (Sighting) et l'émet au consommateur ;
        le cluster n'est pas marqué par une mutation de `cl.state`.
        """
        stale_cids = []
        for cid, cl in self.clusters_slow.items():
            if now_ts - cl.last_seen_ts > self.config.temporal_radius_evict_s:
                stale_cids.append(cid)

        for cid in stale_cids:
            self._finalize_slow(cid)

    def _evict_stale_fast(self, now_ts: float) -> None:
        """Purge la mémoire des pistes géométriques mortes."""
        stale_cids = [cid for cid, cl in self.clusters_fast.items()
                      if now_ts - cl.last_seen_ts > self.config.temporal_radius_fast_s]
        for cid in stale_cids:
            self._finalize_fast(cid)

    def flush(self) -> None:
        """Fin de flux : finalise tous les clusters slow et fast, sans timestamp factice.

        Ordre slow puis fast (identique à process_batch). Idempotent : un second
        appel ne finalise rien et laisse les jauges à 0. Non appelé en cas
        d'exception (décision lot 4 : flush() dans le try, pas dans un finally).
        """
        for cid in list(self.clusters_slow):
            self._finalize_slow(cid)
        for cid in list(self.clusters_fast):
            self._finalize_fast(cid)
        self.counters.sample_active_clusters(len(self.clusters_slow), len(self.clusters_fast), self.config.active_clusters_warn_threshold)

    # ─── TRAITEMENT PRINCIPAL ─────────────────────────────────────────────

    def process_batch(self, batch: List[Observation]) -> List[dict]:
        """Orchestre un lot temporel. Résout l'identité (slow) puis la continuité (fast)."""
        if not batch:
            return []
        self._clock_ts = max(self._clock_ts, batch[-1].ts)
        self._evict_stale_slow(self._clock_ts)
        self._evict_stale_fast(self._clock_ts)
        slow_obs = [o for o in batch if o.source == "slow"]
        resolved_slow = self._solve_slow_identity(slow_obs) if slow_obs else {}
        results = []
        for obs in batch:
            if obs.source == "slow":
                self.counters.bump(CounterKey.SLOW_OBS)
                results.append(self._process_slow_obs(obs, resolved_slow.get(obs.uid)))
            else:
                self.counters.bump(CounterKey.FAST_OBS)
                results.append(self._process_fast_obs(obs))
        self.counters.sample_active_clusters(len(self.clusters_slow), len(self.clusters_fast), self.config.active_clusters_warn_threshold)
        return results

    # ─── RÉSOLUTION DE L'IDENTITÉ (HUNGARIAN) ─────────────────────────────

    def _eval_slow_pair(self, obs: Observation, cluster: ClusterSlow) -> Optional[Tuple[SigDistance, float]]:
        """Fonction de coût pure passée au solveur (Étape 6)."""
        cfg = self.config
        anchor = cluster.anchor.anchor  # Best Anchor (Étape 4)
        if not anchor:
            return SigDistance.gated_only("no_anchor"), BIG_COST_COST
        # 1. Géométrie par rapport à l'ancre
        dx = obs.cx - anchor.cx
        dy = obs.cy - anchor.cy
        dist = (dx * dx + dy * dy) ** 0.5
        diag = (anchor.rw ** 2 + anchor.rh ** 2) ** 0.5
        denom = max(cfg.k_pos_norm * diag, 1.0)
        d_pos = min(1.0, dist / denom)
        d_dim = dim_distance(obs.rw, obs.rh, anchor.rect[2], anchor.rect[3])
        # 2. Smart Signature (Étape 3)
        sig_d = sig_distance(anchor.signature, obs.signature, cfg)
        self._count_sig(sig_d)
        if sig_d.gated or sig_d.cost is None:
            return None
        cost_sig = min(1.0, max(0.0, sig_d.cost))
        return sig_d, cost_sig

    def _count_sig(self, sd: SigDistance) -> None:
        """Compte les signaux par paire évaluée ; INV-D1 est respecté car l'orchestrateur compte."""
        cfg = self.config
        if sd.gate_reason in ("insufficient_families", "no_common_channel"):
            self.counters.bump(CounterKey.SIG_PAIR_REJECT_NO_COMMON_CHANNEL)
        elif sd.gate_reason == "h_txt":
            self.counters.bump(CounterKey.SIG_GATE_H_TXT)
        elif sd.gate_reason == "l_ratio":
            self.counters.bump(CounterKey.SIG_GATE_L_RATIO)
        elif sd.gate_reason == "hs_divergence":
            self.counters.bump(CounterKey.SIG_GATE_HS_DIVERGENCE)
        elif sd.gate_reason == "occupancy":
            self.counters.bump(CounterKey.SIG_GATE_OCCUPANCY)
        elif sd.gate_reason == "no_descriptor":
            self.counters.bump(CounterKey.SIG_GATE_NO_DESCRIPTOR)
        if sd.cost is None and not sd.gated:
            self.counters.bump(CounterKey.SIG_PAIR_REJECT_NO_COMMON_CHANNEL)
        if sd.hs_plan == "raw":
            self.counters.bump(CounterKey.SIG_HS_PLAN_RAW)
        if not sd.gated and sd.h_txt_min is not None and sd.h_txt_min < cfg.map_h_txt_full:
            self.counters.bump(CounterKey.SIG_H_TXT_PENALIZED)

    def _solve_slow_identity(self, slow_obs: List[Observation]) -> Dict[int, Tuple[int, SigDistance]]:
        """Appelle le solveur pur et traduit la CostMatrix en assignations."""
        cm = build_slow_cost_matrix(self.config, slow_obs, self.clusters_slow,cost_fn=self._eval_slow_pair)
        if not cm:
            return {}
        # Mise à jour des stats depuis le solveur pur
        self.counters.bump(CounterKey.HUNGARIAN_BATCHES)
        self.counters.bump(CounterKey.GATE_TEMPORAL_REJECT_SLOW, cm.n_gate_rejects)
        self.counters.bump(CounterKey.COST_REJECT_SLOW, cm.n_cost_rejects)
        self.counters.bump(CounterKey.COLS_GATE_ONLY, cm.n_cols_gate_only)
        self.counters.bump(CounterKey.COLS_COST_REJECTED, cm.n_cols_cost_rejected)
        if self.calib_writer:
            for rec in cm.calib_pairs:
                self.calib_writer.add_record(rec)
        # Assignation hongroise
        rows, cols = linear_sum_assignment(cm.matrix)
        assignments = {}
        for r, c in zip(rows, cols):
            if cm.matrix[r, c] >= BIG_COST_COST:
                continue
            cell = cm.payload[r][c]
            if cell is None:
                continue
            sd, cost_sig = cell
            assignments[cm.rows[r].uid] = (cm.col_ids[c], sd)
        return assignments

    # ─── APPLICATION AUX CLUSTERS ─────────────────────────────────────────

    def _process_slow_obs(self, obs: Observation, match: Optional[Tuple[int, SigDistance]]) -> dict:
        """Applique l'observation slow à son cluster identité (ou en crée un)."""
        if match is None:
            cid_slow, eid_identity = self._create_slow(obs)
        else:
            cid_slow, sd = match
            self._update_slow(cid_slow, obs)
            eid_identity = self.clusters_slow[cid_slow].eid
        # Rattachement à la piste fast (continuité)
        cid_fast = self._process_fast_track(obs, eid_identity)
        return {
            "uid": obs.uid,
            "source": "slow",
            "eid": eid_identity,
            "cluster_slow_id": cid_slow,
            "cluster_fast_id": cid_fast
        }

    def _process_fast_obs(self, obs: Observation) -> dict:
        """Les observations fast n'alimentent QUE la piste géométrique."""
        cid_fast = self._process_fast_track(obs, None)
        track = self.clusters_fast[cid_fast]
        return {
            "uid": obs.uid,
            "source": "fast",
            "eid": track.eid,
            "cluster_slow_id": None,
            "cluster_fast_id": cid_fast
        }

    def _process_fast_track(self, obs: Observation, identity_eid: Optional[int]) -> int:
        """
        Trouve la piste géométrique la plus proche ou en crée une.

        A2 (option i) — gating cinématique sur les centres cx/cy (pas vx/vy) :
            radius_px = max_travel_px_per_s * gap_s + fast_radius_floor_px
        Le gate temporel (gap_s > temporal_radius_fast_s) est CONSERVÉ : il exprime
        "trop vieux pour être la même piste", le rayon exprime "trop loin pour être
        le même objet". Les deux sont complémentaires, pas redondants.
        """
        cfg = self.config
        best_cid, best_d = None, float("inf")
        second_d = float("inf")

        for cid, cl in self.clusters_fast.items():
            gap_s = obs.ts - cl.last_seen_ts
            if gap_s < 0 or gap_s > cfg.temporal_radius_fast_s:
                self.counters.bump(CounterKey.GATE_TEMPORAL_REJECT_FAST)
                continue

            last = cl.last_obs
            d_px = ((obs.cx - last.cx) ** 2 + (obs.cy - last.cy) ** 2) ** 0.5
            radius_px = cfg.max_travel_px_per_s * gap_s + cfg.fast_radius_floor_px
            if d_px > radius_px:
                self.counters.bump(CounterKey.COST_REJECT_FAST)
                continue

            # Distance normalisée par son propre rayon → comparable entre gaps
            d_norm = d_px / radius_px
            if d_norm < best_d:
                second_d, best_d = best_d, d_norm
                best_cid = cid
            elif d_norm < second_d:
                second_d = d_norm

        # Ambiguïté : le second candidat est presque aussi bon que le meilleur
        if best_cid is not None and second_d < float("inf"):
            if best_d > 0.0 and (best_d / second_d) > cfg.fast_ambiguity_ratio:
                self.counters.bump(CounterKey.ANOMALY_FAST_AMBIGUOUS)

        # A4 — insertion assurée par les factories ; symétrie préservée
        if best_cid is None:
            best_cid = self._create_fast(obs)
        else:
            self.clusters_fast[best_cid].add_observation(obs)
            self.counters.bump(CounterKey.FAST_MATCHED)
        track = self.clusters_fast[best_cid]

        # Adoption EID
        if identity_eid is not None:
            if track.eid is None:
                track.eid = identity_eid
                self.counters.bump(CounterKey.TRACK_ADOPTED)
            elif track.eid != identity_eid:
                self.counters.bump(CounterKey.ANOMALY_IDENTITY_TRACK_MISMATCH)
        return best_cid

    # ─── FACTORIES / UPDATES ─────────────────────────────────────────────

    def _create_slow(self, obs: Observation) -> Tuple[int, int]:
        cid = self._next_cluster_slow_id
        self._next_cluster_slow_id += 1
        eid = self._next_eid
        self._next_eid += 1
        # Création et initialisation avec l'ancre
        cl = ClusterSlow(cluster_id=cid, eid=eid)
        if not obs.signature:
            self.counters.bump(CounterKey.REF_INITIAL_SIGNATURE_EMPTY)
        promo = cl.add_observation(obs, self.config)
        self._count_promotion(promo)
        self.clusters_slow[cid] = cl
        self.counters.bump(CounterKey.CLUSTERS_SLOW_CREATED)
        return cid, eid

    def _update_slow(self, cid: int, obs: Observation) -> None:
        cl = self.clusters_slow[cid]
        promo = cl.add_observation(obs, self.config)
        self._count_promotion(promo)


    def _count_promotion(self, result: PromotionResult) -> None:
        """Compte uniquement les tentatives bloquées; les promotions sont finales."""
        blocked = {
            PromoOutcome.blocked_identity: CounterKey.PROMO_BLOCKED_IDENTITY,
            PromoOutcome.blocked_quality: CounterKey.PROMO_BLOCKED_QUALITY,
        }
        key = blocked.get(result.outcome)
        if key is not None:
            self.counters.bump(key)

    def _create_fast(self, obs: Observation) -> int:
        cid = self._next_cluster_fast_id
        self._next_cluster_fast_id += 1
        cl = ClusterFast(cluster_id=cid)
        cl.add_observation(obs)
        self.clusters_fast[cid] = cl
        self.counters.bump(CounterKey.CLUSTERS_FAST_CREATED)
        return cid