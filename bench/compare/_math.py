# bench/compare/_math.py
"""
Primitives mathématiques pures (S5a/S4bis).

Opèrent exclusivement sur des list[float] / valeurs scalaires.
Ne connaissent JAMAIS le schéma JSONL (aucun row["probes"]).
"""

from statistics import quantiles, stdev
from scipy.stats import skew, kurtosis  # type: ignore[import-untyped]
from bench.compare._config import (
    SKEWNESS_MIN_SAMPLES,
    KURTOSIS_MIN_SAMPLES,
    PERCENTILE_MIN_SAMPLES,
    _r,
)

def delta_pct(target: float | None, ref: float | None) -> float | None:
    """((target - ref) / ref) * 100. None si l'un ou l'autre est None ou ref == 0."""
    if target is None or ref is None or ref == 0:
        return None
    return _r((target - ref) / ref * 100)

def delta_abs(target: float | None, reference: float | None) -> float | None:
    """
    Delta absolu (D8) : target - reference.
    Utilisé pour skewness / kurtosis_excess où le delta_pct n'a pas de sens
    (valeurs centrées autour de 0, signe porteur d'information).
    Retourne None si l'un des deux opérandes est None.
    """
    if target is None or reference is None:
        return None
    return target - reference

def percentile_value(data: list[float], pct: int) -> float | None:
    """
    Calcule un percentile via statistics.quantiles (method='inclusive').
    Retourne None si len(data) < PERCENTILE_MIN_SAMPLES.
    """
    if len(data) < PERCENTILE_MIN_SAMPLES:
        return None
    qs = quantiles(data, n=100, method="inclusive")
    return qs[pct - 1]

def quartile_values(data: list[float]) -> tuple[float | None, float | None, float | None]:
    """
    Calcule (Q1, Q3, IQR) via statistics.quantiles (method='inclusive', n=4).
    Retourne (None, None, None) si len(data) < PERCENTILE_MIN_SAMPLES.
    Cohérent avec percentile_value (même seuil, même méthode quantile).
    IQR = Q3 - Q1.
    """
    if len(data) < PERCENTILE_MIN_SAMPLES:
        return None, None, None
    qs = quantiles(data, n=4, method="inclusive")
    q1, q3 = qs[0], qs[2]
    return q1, q3, q3 - q1

def safe_skewness(samples: list[float]) -> float | None:
    """
    Skewness Fisher-Pearson (scipy.stats.skew, bias=False).
    Retourne None si :
      - len(samples) < SKEWNESS_MIN_SAMPLES (D2)
      - variance nulle (stdev == 0) → wrapper défensif (D13)
    """
    if len(samples) < SKEWNESS_MIN_SAMPLES:
        return None
    if stdev(samples) == 0:
        return None
    return float(skew(samples, bias=False))

def safe_kurtosis_excess(samples: list[float]) -> float | None:
    """
    Kurtosis excess (scipy.stats.kurtosis, fisher=True, bias=False).
    Référence loi normale = 0.
    Retourne None si :
      - len(samples) < KURTOSIS_MIN_SAMPLES (D3)
      - variance nulle (stdev == 0) → wrapper défensif (D13)
    """
    if len(samples) < KURTOSIS_MIN_SAMPLES:
        return None
    if stdev(samples) == 0:
        return None
    return float(kurtosis(samples, fisher=True, bias=False))
