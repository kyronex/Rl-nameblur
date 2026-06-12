# bench/compare/_anomalies.py
"""
Détection d'anomalies sur séries temporelles (S5b).

Opèrent sur des list[tuple[value, mono]] triées par mono croissant
(tri à la charge de l'appelant). Ne connaissent JAMAIS le schéma JSONL.
La collecte des paires est métier (cf. _aggregate.collect_frame_exact_pairs).
"""

from statistics import median, stdev
import numpy as np
from bench.compare._config import (
    SPIKE_MIN_SAMPLES,
    SPIKE_MAD_FACTOR,
    DRIFT_MIN_SAMPLES,
)

def empty_anomalies() -> dict:
    """Bloc anomalies tout-null (utilisé pour fast_probes et cas dégénérés)."""
    return {
        "spike_count":         None,
        "spike_max_value":     None,
        "spike_max_deviation": None,
        "drift_slope":         None,
        "drift_intercept":     None,
        "drift_r2":            None,
    }

def _compute_spikes(values: list[float]) -> tuple[int | None, float | None, float | None, set[int]]:
    """
    Détection de spikes via MAD (Median Absolute Deviation).
    Retourne (spike_count, spike_max_value, spike_max_deviation, spike_indices).
    Gardes :
      - len(values) < SPIKE_MIN_SAMPLES → (None, None, None, set())
      - MAD == 0 (distribution dégénérée) → (None, None, None, set())
    Critère : |value - median| > SPIKE_MAD_FACTOR * MAD
    Déviation reportée = |value - median| / MAD (sans unité).
    `spike_indices` : ensemble des indices i tels que values[i] est un spike.
    Vide (set()) si garde déclenchée OU spike_count == 0. Utilisé en aval
    par compute_anomalies pour le préfiltrage E9 du drift.
    """
    if len(values) < SPIKE_MIN_SAMPLES:
        return None, None, None, set()
    med = median(values)
    abs_devs = [abs(v - med) for v in values]
    mad = median(abs_devs)
    if mad == 0:
        return None, None, None, set()
    threshold = SPIKE_MAD_FACTOR * mad
    spike_count = 0
    max_abs_dev = 0.0
    max_value = None
    spike_indices: set[int] = set()
    for i, (v, abs_dev) in enumerate(zip(values, abs_devs)):
        if abs_dev > threshold:
            spike_count += 1
            spike_indices.add(i)
            if abs_dev > max_abs_dev:
                max_abs_dev = abs_dev
                max_value = v
    if spike_count == 0:
        return 0, None, None, set()
    return spike_count, float(max_value), float(max_abs_dev / mad), spike_indices

def _compute_drift(pairs_sorted: list[tuple[float, float]]) -> tuple[float | None, float | None, float | None]:
    """
    Régression linéaire OLS sur (value, mono) via numpy.polyfit(deg=1).
    Variable explicative x = mono (secondes), variable expliquée y = value.
    Retourne (slope, intercept, r_squared).
    Pré-requis : pairs au format (value, mono), triées par mono croissant
                 (responsabilité appelant).
    Gardes :
      - len(pairs_sorted) < DRIFT_MIN_SAMPLES → (None, None, None)
      - variance des values nulle (stdev == 0) → (None, None, None)
      - variance des mono nulle (stdev == 0) → (None, None, None)
      - SS_tot == 0 (garde défensive redondante) → (None, None, None)
    R² calculé manuellement : 1 - SS_res / SS_tot.
    """
    if len(pairs_sorted) < DRIFT_MIN_SAMPLES:
        return None, None, None
    values = [p[0] for p in pairs_sorted]
    monos  = [p[1] for p in pairs_sorted]
    if stdev(values) == 0 or stdev(monos) == 0:
        return None, None, None
    x = np.asarray(monos, dtype=float)
    y = np.asarray(values, dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    if ss_tot == 0:
        return None, None, None
    r_squared = 1.0 - (ss_res / ss_tot)
    return float(slope), float(intercept), float(r_squared)

def compute_anomalies(pairs_sorted: list[tuple[float, float]]) -> dict:
    """
    Calcule les 6 métriques d'anomalies S5b sur une série temporelle.
    Args:
        pairs_sorted: liste de (value, mono) triée par mono croissant.Tri à la charge de l'appelant (cf. Q-Patch-2).
    Returns:
        dict avec clés : spike_count, spike_max_value, spike_max_deviation,drift_slope, drift_intercept, drift_r2.
        Valeurs None si sous seuils ou cas dégénérés (variance/MAD nul).
    Contrat E9 : le drift est calculé sur la série débarrassée des points identifiés comme spike (préfiltrage). Si aucun spike n'est identifié
    (garde déclenchée ou spike_count == 0), la série brute est utilisée.
    Le recheck DRIFT_MIN_SAMPLES s'effectue dans _compute_drift.
    """
    values = [p[0] for p in pairs_sorted]
    spike_count, spike_max_value, spike_max_deviation, spike_indices = _compute_spikes(values)
    # E9 — préfiltrage drift : retrait des points spike avant OLS
    if spike_indices:
        filtered_pairs = [
            pair for i, pair in enumerate(pairs_sorted)
            if i not in spike_indices
        ]
    else:
        filtered_pairs = pairs_sorted  # aucun spike → série brute
    drift_slope, drift_intercept, drift_r2 = _compute_drift(filtered_pairs)
    return {
        "spike_count":         spike_count,
        "spike_max_value":     spike_max_value,
        "spike_max_deviation": spike_max_deviation,
        "drift_slope":         drift_slope,
        "drift_intercept":     drift_intercept,
        "drift_r2":            drift_r2,
    }
