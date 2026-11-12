# eid/stream_eid_builder.py
"""
Point d'entrée du pipeline EID en mode streaming.
Consomme des observations (JSONL), orchestre le cycle de vie via StreamEIDBuilder,
et produit les empreintes finales EID (Prime Sightings) ainsi que les assignations.

Usage:
    python -m eid.stream_eid_builder --input logs/json/bank_20260806_230228.jsonl
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from eid.config import StreamConfig
from eid.builder import StreamEIDBuilder
from eid.batching import JsonlSource, DedupSource, BatchReader
from eid.calibration import CalibrationWriter
from eid.counters import CounterKey, GaugeKey, ANOMALY_PREFIX, Counters
from eid.sighting import Sighting

log = logging.getLogger("stream_eid_builder")

def _find_ndarray(obj, path="root"):
    """Diagnostic : parcourt récursivement et retourne les chemins des ndarray."""
    import numpy as np
    hits = []
    if isinstance(obj, np.ndarray):
        hits.append((path, f"ndarray shape={obj.shape} dtype={obj.dtype}"))
    elif isinstance(obj, (np.floating, np.integer, np.bool_)):
        hits.append((path, f"numpy scalar {type(obj).__name__}"))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            hits += _find_ndarray(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            hits += _find_ndarray(v, f"{path}[{i}]")
    return hits


def _safe_dumps(obj, label: str) -> str:
    """json.dumps avec diagnostic sur échec."""
    try:
        return json.dumps(obj, ensure_ascii=False)
    except TypeError as exc:
        hits = _find_ndarray(obj, label)
        log.error("=" * 72)
        log.error("❌ ÉCHEC SÉRIALISATION [%s] : %s", label, exc)
        log.error("Clés racine : %s", list(obj.keys()) if isinstance(obj, dict) else type(obj))
        if hits:
            log.error("Champs numpy détectés (%d) :", len(hits))
            for p, desc in hits:
                log.error("   → %-50s %s", p, desc)
        else:
            log.error("Aucun numpy trouvé — type exotique ailleurs.")
        log.error("=" * 72)
        raise

def main() -> int:
    ap = argparse.ArgumentParser(description="Moteur EID Streaming (Identité & Continuité)")
    ap.add_argument("--input", required=True, type=Path, help="Fichier bank JSONL en entrée (observations brutes)")
    ap.add_argument("--output", type=Path, default=None, help="Assignations frame-par-frame en sortie")
    ap.add_argument("--verbose", action="store_true", help="Active le mode debug")
    ap.add_argument("--dump-pairs", action="store_true", help="Dump toutes les paires évaluées (Hungarian) pour la calibration")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s"
    )

    if not args.input.exists():
        log.error("Fichier introuvable : %s", args.input)
        return 1

    out_assign = args.output or args.input.with_name(args.input.stem + "_eid_assignments.jsonl")
    out_sightings = args.input.with_name(args.input.stem + "_eid_sightings.jsonl")

    config = StreamConfig()
    log.info("Config initialisée : %s", config)

    # Callback d'éviction : déclenché par le Builder quand un cluster est finalisé
    n_sightings = 0
    def on_finalized(sighting: Sighting):
        nonlocal n_sightings
        line = _safe_dumps(sighting.to_json(), "sighting")
        with open(out_sightings, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        n_sightings += 1

    # Purge du fichier des empreintes s'il existait
    if out_sightings.exists():
        out_sightings.unlink()

    counters = Counters()
    builder = StreamEIDBuilder(config, on_cluster_finalized=on_finalized, calib_writer=None, counters=counters)

    calib = None
    if args.dump_pairs:
        calib_path = Path("logs/json") / f"calibration_{args.input.stem}.jsonl"
        calib = CalibrationWriter(str(calib_path))
        calib.write_manifest(asdict(config), str(args.input))
        builder.calib_writer = calib
        log.info("Calibration activée → %s", calib_path)

    n_written = 0
    n_batches = 0

    # ─── CORRECTION ICI : Initialisation propre du reader ──────────────────
    # === NOUVEAU cablage Lot A-1/A-2 : SourceStats -> JsonlSource -> DedupSource -> BatchReader ===
    source = DedupSource(JsonlSource(args.input, builder.counters, config), builder.counters, config)
    reader = BatchReader(source, config, builder.counters)

    log.info("🚀 Démarrage du traitement...")

    try:
        with open(out_assign, "w", encoding="utf-8") as out:
            # Iteration explicite du batch reader
            while True:
                batch = reader.next_batch()
                if not batch:
                    break
                n_batches += 1
                log.debug("batch #%d — %d obs — ts=[%.3f .. %.3f]",n_batches, len(batch), batch[0].ts, batch[-1].ts)
                # D1 - instrumentation deplacee : BatchReader n a plus builder=.
                assignments = builder.process_batch(batch)
                for assign in assignments:
                    out.write(_safe_dumps(assign, "assign") + "\n")
                    n_written += 1

        # Lot 4 : fin de flux. On finalise tous les clusters slow et fast, sans timestamp factice.
        # Appelé dans le try, pas dans un finally : pas de flush en cas d'exception (décision lot 4).
        builder.flush()

    except Exception as e:
        log.error("Erreur critique lors du traitement : %s", e, exc_info=args.verbose)
        return 1
    finally:
        if calib:
            calib.close()

    # ─── RAPPORT D'EXÉCUTION ──────────────────────────────────────────────────
    # PP-D8 : lectures via CounterKey (StrEnum, clés str inchangées dans stats()).
    # Vue fusionnée compteurs + jauges : ne jamais sommer ses valeurs.
    K = CounterKey
    stats = builder.counters.stats()
    anomalies = builder.counters.anomalies()

    log.info("=" * 72)
    log.info("✅ DUAL CLUSTERING EID BUILDER — RÉSUMÉ")
    log.info("=" * 72)
    log.info("Observations reçues      : %d (Slow) / %d (Fast)", stats[K.SLOW_OBS], stats[K.FAST_OBS])
    log.info("Clusters créés           : %d (Identité) / %d (Pistes)", stats[K.CLUSTERS_SLOW_CREATED], stats[K.CLUSTERS_FAST_CREATED])
    log.info("Clusters finalisés (EID) : %d (Empreintes exportées)", n_sightings)
    log.info("Sightings sans ancre / callback en erreur : %d / %d",
             stats[K.SIGHTING_EMITTED_NO_ANCHOR], stats[K.SIGHTING_CALLBACK_ERROR])

    # Ratio de fragmentation — cible du plan : < 0.15
    if stats[K.SLOW_OBS] > 0:
        ratio = stats[K.CLUSTERS_SLOW_CREATED] / stats[K.SLOW_OBS]
        lvl = log.info if ratio < 0.15 else log.warning
        lvl("  └─ Ratio clusters/obs_slow : %.3f (cible < 0.15)", ratio)

    n_batches = stats[K.BATCHES_READ]
    avg_batch = (stats[K.BATCH_OBS_CUMUL] / n_batches) if n_batches else 0.0
    log.info("Lots temporels (calibration epsilon_close_s) :")
    log.info("  ├─ Lots lus            : %d", n_batches)
    log.info("  ├─ Taille moyenne      : %.2f obs", avg_batch)
    log.info("  ├─ Taille max          : %d obs", reader.batch_size_max)
    n_ooo = stats[K.BATCH_FRAME_OUT_OF_ORDER]
    if n_ooo > 0:
        log.warning("  └─ Frames hors ordre    : %d", n_ooo)
    else:
        log.info("  └─ Frames hors ordre    : %d", n_ooo)

    log.info("Identité (Hungarian) :")
    log.info("  ├─ Lots résolus        : %d (lots contenant du slow)", stats[K.HUNGARIAN_BATCHES])
    log.info("  ├─ Rejets temporels    : %d (paires)", stats[K.GATE_TEMPORAL_REJECT_SLOW])
    log.info("  ├─ Rejets par coût     : %d (paires)", stats[K.COST_REJECT_SLOW])
    log.info("  ├─ Colonnes gate-only  : %d (cluster jamais évalué du lot)", stats[K.COLS_GATE_ONLY])
    log.info("  └─ Colonnes rejetées   : %d (évalué, aucun match) ← symptôme c_max/signature", stats[K.COLS_COST_REJECTED])

    log.info("Continuité (gating cinématique) :")
    log.info("  ├─ Obs rattachées      : %d", stats[K.FAST_MATCHED])
    log.info("  ├─ Pistes créées       : %d", stats[K.CLUSTERS_FAST_CREATED])
    log.info("  ├─ Pistes évincées     : %d", stats[K.EVICTED_FAST])
    log.info("  ├─ EID adoptés         : %d", stats[K.TRACK_ADOPTED])
    log.info("  ├─ Rejets temporels    : %d (paires)", stats[K.GATE_TEMPORAL_REJECT_FAST])
    log.info("  └─ Rejets hors rayon   : %d (paires)", stats[K.COST_REJECT_FAST])
    log.info("Qualité de l'entrée :")
    src_lines = [
        (K.SRC_OBS_READ_TOTAL, "Entrées lues"),
        (K.SRC_OBS_BUILT, "Observations construites"),
        (K.SRC_OBS_REJECT_SOURCE, "Rejets source"),
        (K.SRC_OBS_REJECT_DEGENERATE_BOX, "Rejets box dégénérée"),
        (K.SRC_OBS_REJECT_MISSING_GEOMETRY, "Géométrie manquante"),
        (K.SRC_OBS_REJECT_MALFORMED, "Rejets malformées"),
        (K.SRC_OBS_SLOW_TS_FALLBACK, "Fallback slow_ts"),
        (K.SRC_OBS_SIG_EMPTY, "Signatures vides"),
        (K.SRC_OBS_DEDUP_SKIPPED, "Déduplication ignorée"),
        (K.SRC_OBS_MISSING_FRAME_ID, "frame_id absent"),
        (K.SRC_DEDUP_KEPT, "Déduplication retenue"),
        (K.SRC_DEDUP_PURGES, "Purges déduplication"),
    ]
    for i, (key, label) in enumerate(src_lines):
        branch = "└─" if i == len(src_lines) - 1 else "├─"
        log.info("  %s %-24s : %d", branch, label, stats[key])
    n_read = stats[K.SRC_OBS_READ_TOTAL]
    n_rej = (stats[K.SRC_OBS_REJECT_SOURCE] + stats[K.SRC_OBS_REJECT_DEGENERATE_BOX]
             + stats[K.SRC_OBS_REJECT_MISSING_GEOMETRY] + stats[K.SRC_OBS_REJECT_MALFORMED]
             + stats[K.SRC_OBS_MISSING_FRAME_ID])
    n_kept = stats[K.SRC_OBS_BUILT]
    _kept_derived = n_read - n_rej
    if _kept_derived != n_kept:
        log.warning(
            "  ! Écart src_obs_built : lu=%d, dérivé(read-rejets)=%d",
            n_kept, _kept_derived,
        )

    log.info("Signature (par paire évaluée) :")
    sig_lines = [
        (K.SIG_GATE_H_TXT, "Gate h_txt"),
        (K.SIG_GATE_L_RATIO, "Gate l_ratio"),
        (K.SIG_GATE_OCCUPANCY, "Gate occupancy"),
        (K.SIG_GATE_NO_DESCRIPTOR, "Gate no_descriptor"),
        (K.SIG_GATE_HS_DIVERGENCE, "Gate hs_divergence"),
        (K.SIG_HS_PLAN_RAW, "Plan HS raw"),
        (K.SIG_H_TXT_PENALIZED, "h_txt pénalisé"),
    ]
    for i, (key, label) in enumerate(sig_lines):
        branch = "└─" if i == len(sig_lines) - 1 else "├─"
        log.info("  %s %-24s : %d", branch, label, stats[key])

    log.info("Jauges clusters actifs (valeurs non additionnables (compteurs vs jauges)) :")
    log.info("  ├─ Slow dernière valeur : %d", stats[GaugeKey.EID_ACTIVE_CLUSTERS_SLOW])
    log.info("  └─ Fast dernière valeur : %d", stats[GaugeKey.EID_ACTIVE_CLUSTERS_FAST])
    log.info("Anomalies détectées :")
    for k, v in sorted(anomalies.items()):
        label = k[len(ANOMALY_PREFIX):]
        if v > 0:
            log.warning("  ├─ %-23s : %d", label, v)
        else:
            log.info("  ├─ %-23s : 0", label)

    if calib:
        log.info("Paires calibrées dumpées : %d", calib.n_written)

    log.info("=" * 72)
    log.info("Sorties :")
    log.info(" 📄 Assignations : %s (%d lignes)", out_assign.name, n_written)
    log.info(" 🎯 Empreintes   : %s (%d EIDs générés)", out_sightings.name, n_sightings)
    log.info("=" * 72)

    return 0

if __name__ == "__main__":
    sys.exit(main())