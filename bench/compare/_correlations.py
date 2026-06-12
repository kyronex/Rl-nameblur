# bench/compare/_correlations.py
"""
Calcul des corrélations Spearman par bucket (S6c).

Spec : voir bench-compare.md §"Corrélations Spearman".
Config : voir _config.py constantes CORRELATIONS_*.
"""

from __future__ import annotations

import fnmatch
import logging
import math
from typing import Iterable

from scipy.stats import spearmanr  # type: ignore[import-untyped]

from ._config import (
    CORRELATIONS_BLACKLIST_EXACT,
    CORRELATIONS_BLACKLIST_GLOB,
    CORRELATIONS_ENABLED,
    CORRELATIONS_MAX_PAIRS,
    CORRELATIONS_MIN_ABS_RHO,
    CORRELATIONS_PROBE_AGG,
    CORRELATIONS_STRENGTH_MODERATE,
    CORRELATIONS_STRENGTH_STRONG,
    CORRELATIONS_STRENGTH_VERY_STRONG,
    PERCENTILE_MIN_SAMPLES,  # réutilisé comme seuil min_samples (cf. _config.py)
    _r,
)
from ._stats import collect_frame_metrics_per_row

logger = logging.getLogger(__name__)

def _empty_correlations() -> dict:
    """Bloc neutre — utilisé quand corrélations désactivées ou aucune donnée."""
    return {
        "summary": {
            "n_rows": 0,
            "n_metrics_total": 0,
            "n_metrics_excluded_blacklist": 0,
            "n_metrics_excluded_zero_var": 0,
            "n_pairs_evaluated": 0,
            "n_pairs_below_threshold": 0,
            "n_pairs_low_samples": 0,
            "n_pairs_reported": 0,
            "truncated_by_max_pairs": False,
        },
        "pairs": [],
    }

def _is_blacklisted(name: str) -> bool:
    if name in CORRELATIONS_BLACKLIST_EXACT:
        return True
    for pattern in CORRELATIONS_BLACKLIST_GLOB:
        if fnmatch.fnmatchcase(name, pattern):
            return True
    return False

def _classify_strength(abs_rho: float) -> str:
    """Étiquette strength selon |rho|. Assume abs_rho >= CORRELATIONS_MIN_ABS_RHO."""
    if abs_rho >= CORRELATIONS_STRENGTH_VERY_STRONG:
        return "very_strong"
    if abs_rho >= CORRELATIONS_STRENGTH_STRONG:
        return "strong"
    return "moderate"

def _extract_series(rows: list[dict[str, float]],metric_name: str) -> list[float]:
    """
    Extrait la série brute (avec None pour les frames sans la métrique).
    Le filtrage NaN pairwise est délégué à scipy/numpy en aval.
    """
    return [row.get(metric_name, math.nan) for row in rows]

def _has_zero_variance(series: list[float]) -> bool:
    """
    True si toutes les valeurs non-NaN sont identiques (variance stricte = 0).
    Utilise une comparaison directe — pas de tolérance epsilon (cf. spec δ').
    """
    seen: float | None = None
    for v in series:
        if math.isnan(v):
            continue
        if seen is None:
            seen = v
        elif v != seen:
            return False
    # Si seen is None → série entièrement NaN → traitée comme zero-variance (exclue)
    return True


def compute_correlations(frame_rows: list[dict],bucket_label: str) -> dict:
    if not CORRELATIONS_ENABLED:
        return _empty_correlations()
    if not frame_rows:
        return _empty_correlations()
    # ── 1. Extraction des observations par ligne ──
    rows, metric_types = collect_frame_metrics_per_row(frame_rows,probe_aggregation=CORRELATIONS_PROBE_AGG)
    if not rows:
        return _empty_correlations()
    n_rows = len(rows)
    all_metrics = sorted(metric_types.keys())
    n_metrics_total = len(all_metrics)
    # ── 2. Filtre blacklist ──
    kept_after_blacklist: list[str] = []
    n_excluded_blacklist = 0
    for name in all_metrics:
        if _is_blacklisted(name):
            n_excluded_blacklist += 1
            logger.info(
                "[correlations][%s] sonde exclue (blacklist) : %s",
                bucket_label, name,
            )
        else:
            kept_after_blacklist.append(name)
    # ── 3. Filtre zero-variance ──
    series_by_metric: dict[str, list[float]] = {}
    kept_metrics: list[str] = []
    n_excluded_zero_var = 0
    for name in kept_after_blacklist:
        series = _extract_series(rows, name)
        if _has_zero_variance(series):
            n_excluded_zero_var += 1
            logger.info(
                "[correlations][%s] sonde exclue (variance nulle) : %s",
                bucket_label, name,
            )
            continue
        series_by_metric[name] = series
        kept_metrics.append(name)
    # ── 4. Énumération des paires + calcul Spearman ──
    candidates: list[dict] = []
    n_pairs_evaluated = 0
    n_pairs_below_threshold = 0
    n_pairs_low_samples = 0
    for i in range(len(kept_metrics)):
        for j in range(i + 1, len(kept_metrics)):
            a = kept_metrics[i]
            b = kept_metrics[j]
            n_pairs_evaluated += 1
            sa = series_by_metric[a]
            sb = series_by_metric[b]
            # N effectif = nb de lignes où les deux métriques sont définies (non-NaN)
            n_effective = sum(
                1 for va, vb in zip(sa, sb)
                if not math.isnan(va) and not math.isnan(vb)
            )
            if n_effective < PERCENTILE_MIN_SAMPLES:
                n_pairs_low_samples += 1
                logger.info(
                    "[correlations][%s] paire exclue (n_samples=%d < %d) : (%s, %s)",
                    bucket_label, n_effective, PERCENTILE_MIN_SAMPLES, a, b,
                )
                continue
            # scipy gère le pairwise via nan_policy='omit'
            result = spearmanr(sa, sb, nan_policy="omit")
            rho = float(result.statistic) if hasattr(result, "statistic") else float(result[0])
            # Garde-fou : scipy peut retourner NaN si dégénérescence résiduelle
            if math.isnan(rho):
                n_pairs_low_samples += 1
                logger.info(
                    "[correlations][%s] paire exclue (rho=NaN) : (%s, %s)",
                    bucket_label, a, b,
                )
                continue
            if abs(rho) < CORRELATIONS_MIN_ABS_RHO:
                n_pairs_below_threshold += 1
                continue
            candidates.append({
                "a": a,
                "b": b,
                "rho": rho,
                "n_samples": n_effective,
                "strength": _classify_strength(abs(rho)),
            })
    # ── 5. Tri par |rho| décroissant + cap max_pairs_per_bucket ──
    candidates.sort(key=lambda d: abs(d["rho"]), reverse=True)
    truncated = len(candidates) > CORRELATIONS_MAX_PAIRS
    reported = candidates[:CORRELATIONS_MAX_PAIRS]
    if truncated:
        logger.info(
            "[correlations][%s] cap max_pairs_per_bucket atteint : %d candidates → %d reportées",
            bucket_label, len(candidates), CORRELATIONS_MAX_PAIRS,
        )
    # ── 6. Arrondi rho via _r (cohérence projet) ──
    pairs_out = [
        {
            "a": p["a"],
            "b": p["b"],
            "rho": _r(p["rho"]),
            "n_samples": p["n_samples"],
            "strength": p["strength"],
        }
        for p in reported
    ]
    return {
        "summary": {
            "n_rows": n_rows,
            "n_metrics_total": n_metrics_total,
            "n_metrics_excluded_blacklist": n_excluded_blacklist,
            "n_metrics_excluded_zero_var": n_excluded_zero_var,
            "n_pairs_evaluated": n_pairs_evaluated,
            "n_pairs_below_threshold": n_pairs_below_threshold,
            "n_pairs_low_samples": n_pairs_low_samples,
            "n_pairs_reported": len(pairs_out),
            "truncated_by_max_pairs": truncated,
        },
        "pairs": pairs_out,
    }
