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

def _get(path: str, default):
    """Lecture one-shot d'une clé YAML pointée (ex: 'debug.bench.agg.interval_s')."""
    node = _raw_cfg
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node

with _CONFIG_PATH.open(encoding="utf-8") as _fh:
    _raw_cfg = yaml.safe_load(_fh)

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
