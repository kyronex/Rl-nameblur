# bench/compare/_config.py
"""
Constantes globales pour bench/compare.

Centralise tous les paramètres numériques, chemins et seuils utilisés
par les modules d'analyse comparative. Aucune magic number ailleurs.
"""
from __future__ import annotations

import sys
from pathlib import Path
import yaml

LOG_FORMAT = "%(levelname)s — %(message)s"

# ---------------------------------------------------------------------------
# Chemins projet
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DIR_JSON = PROJECT_ROOT / "logs" / "json"
DIR_RESULTS = PROJECT_ROOT / "logs" / "results"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Lecture statique config.yaml (sans watcher, sans handle persistant)
# ---------------------------------------------------------------------------

_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with _CONFIG_PATH.open(encoding="utf-8") as _fh:
    _raw_cfg = yaml.safe_load(_fh)

def _get(path: str, default):
    """Lecture one-shot d'une clé YAML pointée (ex: 'debug.bench.agg.interval_s')."""
    node = _raw_cfg
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node

# ---------------------------------------------------------------------------
# Schéma du rapport
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Calculs statistiques
# ---------------------------------------------------------------------------

ROUND_DIGITS = 3
PERCENTILE_MIN_SAMPLES = 20
PERCENTILES = [90, 95, 99]

# ---------------------------------------------------------------------------
# Périodes attendues (canaux JSONL)
# ---------------------------------------------------------------------------

EXPECTED_PERIOD_S = {
    "agg":   _get("debug.bench.agg.interval_s",  1.0),
    "frame": None,
    "fast":  _get("debug.bench.fast.interval_s", 1.0),
}

# ---------------------------------------------------------------------------
# Seuils détection des gaps temporels
# ---------------------------------------------------------------------------

GAPS_STAT_FACTOR = 3   # gap statistique : interval > median × 3
GAPS_FIXED_FACTOR = 2  # gap fixe       : interval > expected_period × 2

# ---------------------------------------------------------------------------
# Helpers — arrondi
# ---------------------------------------------------------------------------

def _r(val: float | None) -> float | None:
    """Arrondi à ROUND_DIGITS décimales, None passthrough."""
    if val is None:
        return None
    return round(val, ROUND_DIGITS)

# ---------------------------------------------------------------------------
# Bucketing adaptatif cold/hot (S4)
# ---------------------------------------------------------------------------

BUCKET_COLD_TARGET_S    = _get("debug.bench.compare.buckets.cold_target_s",    5.0)
BUCKET_HOT_DURATION_S   = _get("debug.bench.compare.buckets.hot_duration_s",  10.0)
BUCKET_MAX_COLD_DRIFT_S = _get("debug.bench.compare.buckets.max_cold_drift_s", 3.0)
BUCKET_BOUNDARY_GUARD_S = _get("debug.bench.compare.buckets.boundary_guard_s", 0.5)
BUCKET_MIN_GAP_S        = _get("debug.bench.compare.buckets.min_gap_s",        0.1)
BUCKET_EPSILON_S        = _get("debug.bench.compare.buckets.epsilon_s",        0.001)

assert BUCKET_COLD_TARGET_S > 0,     "cold_target_s doit être > 0"
assert BUCKET_HOT_DURATION_S > 0,    "hot_duration_s doit être > 0"
assert BUCKET_MAX_COLD_DRIFT_S >= 0, "max_cold_drift_s doit être >= 0"
assert BUCKET_BOUNDARY_GUARD_S >= 0, "boundary_guard_s doit être >= 0"
assert 0 <= BUCKET_MIN_GAP_S < BUCKET_BOUNDARY_GUARD_S, \
    "min_gap_s doit être dans [0, boundary_guard_s["
assert BUCKET_EPSILON_S >= 0,        "epsilon_s doit être >= 0"

# ---------------------------------------------------------------------------
# Stats forme de distribution (S5a)
# ---------------------------------------------------------------------------

SKEWNESS_MIN_SAMPLES = _get("debug.bench.compare.shape.skewness_min_samples", 50)
KURTOSIS_MIN_SAMPLES = _get("debug.bench.compare.shape.kurtosis_min_samples", 100)

assert SKEWNESS_MIN_SAMPLES > 0, "SKEWNESS_MIN_SAMPLES doit être > 0"
assert KURTOSIS_MIN_SAMPLES > 0, "KURTOSIS_MIN_SAMPLES doit être > 0"

# ---------------------------------------------------------------------------
# Anomalies S5b — Spikes (MAD) + Drift (OLS)
# ---------------------------------------------------------------------------
SPIKE_MIN_SAMPLES = _get("debug.bench.compare.anomalies.spike_min_samples", 20)
SPIKE_MAD_FACTOR  = _get("debug.bench.compare.anomalies.spike_mad_factor",  3.5)
DRIFT_MIN_SAMPLES = _get("debug.bench.compare.anomalies.drift_min_samples", 30)

assert isinstance(SPIKE_MIN_SAMPLES, int) and SPIKE_MIN_SAMPLES >= 2, \
    "SPIKE_MIN_SAMPLES doit être un int >= 2"
assert isinstance(SPIKE_MAD_FACTOR, (int, float)) and SPIKE_MAD_FACTOR > 0, \
    "SPIKE_MAD_FACTOR doit être > 0"
assert isinstance(DRIFT_MIN_SAMPLES, int) and DRIFT_MIN_SAMPLES >= 2, \
    "DRIFT_MIN_SAMPLES doit être un int >= 2"
# ---------------------------------------------------------------------------
# Frame budget (S6a) — Décomposition temporelle de la boucle frame
# ---------------------------------------------------------------------------

FRAME_BUDGET_REFERENCE = "main_loop_ms"

FRAME_BUDGET_GROUPS = {
    "slow_poll":    "main_slow_poll_ms",
    "distribute":   "main_distribute_ms",
    "copy":         "main_copy_ms",
    "match":        "main_match_ms",
    "fast_poll":    "main_fast_poll_ms",
    "predict":      "main_predict_ms",
    "prepare":      "main_prepare_ms",
    "blur":         "main_blur_ms",
    "send":         "main_send_ms",
    "stats":        "main_stats_ms",
}

FRAME_BUDGET_CONDITIONAL = frozenset({"match", "fast_poll"})

# Cohérence interne : tout groupe conditionnel doit exister dans la whitelist
assert FRAME_BUDGET_CONDITIONAL.issubset(FRAME_BUDGET_GROUPS.keys()), \
    "FRAME_BUDGET_CONDITIONAL contient un groupe absent de FRAME_BUDGET_GROUPS"

# Seuils tunables via YAML
FRAME_BUDGET_ENABLED              = _get("debug.bench.compare.frame_budget.enabled",              True)
FRAME_BUDGET_MIN_PRESENCE_RATE    = _get("debug.bench.compare.frame_budget.min_presence_rate",    0.5)
FRAME_BUDGET_UNACCOUNTED_WARN_PCT = _get("debug.bench.compare.frame_budget.unaccounted_warn_pct", 15.0)

assert isinstance(FRAME_BUDGET_ENABLED, bool), \
    "FRAME_BUDGET_ENABLED doit être un booléen"
assert isinstance(FRAME_BUDGET_MIN_PRESENCE_RATE, (int, float)) \
    and 0.0 <= FRAME_BUDGET_MIN_PRESENCE_RATE <= 1.0, \
    "FRAME_BUDGET_MIN_PRESENCE_RATE doit être dans [0.0, 1.0]"
assert isinstance(FRAME_BUDGET_UNACCOUNTED_WARN_PCT, (int, float)) \
    and FRAME_BUDGET_UNACCOUNTED_WARN_PCT >= 0.0, \
    "FRAME_BUDGET_UNACCOUNTED_WARN_PCT doit être >= 0.0"

# ---------------------------------------------------------------------------
# Correlations Spearman par bucket (S6c)
# ---------------------------------------------------------------------------

CORRELATIONS_ENABLED         = _get("debug.bench.compare.correlations.enabled",              True)
CORRELATIONS_MIN_ABS_RHO     = _get("debug.bench.compare.correlations.min_abs_rho",          0.5)
CORRELATIONS_MAX_PAIRS       = _get("debug.bench.compare.correlations.max_pairs_per_bucket", 50)
CORRELATIONS_PROBE_AGG       = _get("debug.bench.compare.correlations.probe_aggregation",    "sum")
CORRELATIONS_BLACKLIST_GLOB  = _get("debug.bench.compare.correlations.blacklist_patterns",   ["bench_*"])
CORRELATIONS_BLACKLIST_EXACT = _get("debug.bench.compare.correlations.blacklist_exact",      [])

# Seuils d'étiquetage strength (lus depuis YAML)
CORRELATIONS_STRENGTH_MODERATE    = _get("debug.bench.compare.correlations.strength.moderate",    0.5)
CORRELATIONS_STRENGTH_STRONG      = _get("debug.bench.compare.correlations.strength.strong",      0.7)
CORRELATIONS_STRENGTH_VERY_STRONG = _get("debug.bench.compare.correlations.strength.very_strong", 0.9)

# Seuil minimal d'échantillons : réutilisation de PERCENTILE_MIN_SAMPLES
# (couplage assumé — cf. bench-compare.md §"Corrélations Spearman")

assert isinstance(CORRELATIONS_ENABLED, bool), \
    "CORRELATIONS_ENABLED doit être un booléen"
assert isinstance(CORRELATIONS_MIN_ABS_RHO, (int, float)) \
    and 0.0 <= CORRELATIONS_MIN_ABS_RHO <= 1.0, \
    "CORRELATIONS_MIN_ABS_RHO doit être dans [0.0, 1.0]"
assert isinstance(CORRELATIONS_MAX_PAIRS, int) and CORRELATIONS_MAX_PAIRS >= 1, \
    "CORRELATIONS_MAX_PAIRS doit être un int >= 1"
assert CORRELATIONS_PROBE_AGG in {"sum", "mean", "max"}, \
    "CORRELATIONS_PROBE_AGG doit être 'sum', 'mean' ou 'max'"
assert isinstance(CORRELATIONS_BLACKLIST_GLOB, list) \
    and all(isinstance(p, str) for p in CORRELATIONS_BLACKLIST_GLOB), \
    "CORRELATIONS_BLACKLIST_GLOB doit être une liste de str"
assert isinstance(CORRELATIONS_BLACKLIST_EXACT, list) \
    and all(isinstance(p, str) for p in CORRELATIONS_BLACKLIST_EXACT), \
    "CORRELATIONS_BLACKLIST_EXACT doit être une liste de str"

assert isinstance(CORRELATIONS_STRENGTH_MODERATE, (int, float)), \
    "CORRELATIONS_STRENGTH_MODERATE doit être numérique"
assert isinstance(CORRELATIONS_STRENGTH_STRONG, (int, float)), \
    "CORRELATIONS_STRENGTH_STRONG doit être numérique"
assert isinstance(CORRELATIONS_STRENGTH_VERY_STRONG, (int, float)), \
    "CORRELATIONS_STRENGTH_VERY_STRONG doit être numérique"
assert 0.0 < CORRELATIONS_STRENGTH_MODERATE \
    < CORRELATIONS_STRENGTH_STRONG \
    < CORRELATIONS_STRENGTH_VERY_STRONG <= 1.0, \
    "Seuils strength doivent être strictement croissants dans ]0, 1]"
assert CORRELATIONS_MIN_ABS_RHO <= CORRELATIONS_STRENGTH_MODERATE, \
    "CORRELATIONS_MIN_ABS_RHO doit être <= CORRELATIONS_STRENGTH_MODERATE " \
    "(sinon des paires reportées seraient non étiquetables)"
