# bench/compare/_stats.py

from statistics import median , quantiles , stdev
from scipy.stats import skew, kurtosis
import numpy as np
from bench.compare._config import (
    SKEWNESS_MIN_SAMPLES,
    KURTOSIS_MIN_SAMPLES,
    SPIKE_MIN_SAMPLES,
    SPIKE_MAD_FACTOR,
    DRIFT_MIN_SAMPLES,
    EXPECTED_PERIOD_S,
    GAPS_STAT_FACTOR,
    GAPS_FIXED_FACTOR,
    PERCENTILE_MIN_SAMPLES,
    _r,
)

def agg_probes(rows: list[dict]) -> dict[str, dict]:
    """
    Agrège les sondes depuis les lignes du canal agg.
    Retourne {probe_name: {avg, min, max, count_agg}}.
    Lit uniquement la section `probes` (structure fixe documentée).
    Note : `count_agg` est la somme des `count` lus dans les lignes du canal
    agg. Il n'est PAS comparable à `samples_approx` (nombre de lignes du
    canal frame). Voir docs/bench-compare.md §5.3.
    """
    accum: dict[str, dict] = {}
    for row in rows:
        probes = row.get("probes")
        if not isinstance(probes, dict):
            continue
        for probe, stats in probes.items():
            if not isinstance(stats, dict):
                continue
            avg = stats.get("avg")
            mn = stats.get("min")
            mx = stats.get("max")
            cnt = stats.get("count")
            if None in (avg, mn, mx, cnt) or cnt == 0:
                continue
            key = probe
            if key not in accum:
                accum[key] = {
                    "sum_weighted": 0.0,
                    "min": mn,
                    "max": mx,
                    "count_agg": 0,
                }
            accum[key]["sum_weighted"] += avg * cnt
            accum[key]["min"] = min(accum[key]["min"], mn)
            accum[key]["max"] = max(accum[key]["max"], mx)
            accum[key]["count_agg"] += cnt
    result: dict[str, dict] = {}
    for key, acc in accum.items():
        result[key] = {
            "avg": acc["sum_weighted"] / acc["count_agg"],
            "min": acc["min"],
            "max": acc["max"],
            "count_agg": acc["count_agg"],
        }
    return result

def agg_rates(rows: list[dict]) -> dict[str, float]:
    """Moyenne arithmétique des valeurs rates sur toutes les lignes agg."""
    accum: dict[str, list[float]] = {}
    for row in rows:
        rates = row.get("rates")
        if not isinstance(rates, dict):
            continue
        for name, val in rates.items():
            if isinstance(val, (int, float)):
                accum.setdefault(name, []).append(float(val))
    return {name: sum(vals) / len(vals) for name, vals in accum.items()}

def agg_gauges(rows: list[dict]) -> dict[str, float]:
    """Moyenne arithmétique des valeurs gauges sur toutes les lignes agg."""
    accum: dict[str, list[float]] = {}
    for row in rows:
        gauges = row.get("gauges")
        if not isinstance(gauges, dict):
            continue
        for name, val in gauges.items():
            if isinstance(val, (int, float)):
                accum.setdefault(name, []).append(float(val))
    return {name: sum(vals) / len(vals) for name, vals in accum.items()}

def session_duration(rows: list[dict]) -> float | None:
    """ts dernière ligne - ts première ligne du canal agg."""
    timestamps = [
        r["ts"] for r in rows if "ts" in r and isinstance(r["ts"], (int, float))
    ]
    if len(timestamps) < 2:
        return None
    return float(max(timestamps) - min(timestamps))

def extract_timeline(rows: list[dict]) -> list[dict]:
    """
    Extrait la timeline (ts wall-clock + mono) depuis les lignes d'un canal.
    Ignore les lignes sans `ts` ou `mono` exploitables.
    Trie par mono croissant pour garantir l'ordre temporel.
    """
    timeline: list[dict] = []
    for row in rows:
        ts = row.get("ts")
        mono = row.get("mono")
        if isinstance(ts, (int, float)) and isinstance(mono, (int, float)):
            timeline.append({"ts": float(ts), "mono": float(mono)})
    timeline.sort(key=lambda e: e["mono"])
    return timeline

def compute_frames(timeline: list[dict]) -> int:
    """Nombre d'événements (lignes JSONL) sur le canal."""
    return len(timeline)

def compute_duration_mono(timeline: list[dict]) -> float | None:
    """
    Durée effective basée sur l'horloge monotone : max(mono) − min(mono).
    Retourne None si moins de 2 événements.
    """
    if len(timeline) < 2:
        return None
    monos = [e["mono"] for e in timeline]
    return float(max(monos) - min(monos))

def compute_temporal_events(timeline: list[dict], canal: str) -> dict:
    """
    Calcule statistiques temporelles sur les intervalles entre événements consécutifs (basés sur `mono`).
    Retourne :
      - median_interval_s : médiane des intervalles
      - gaps_stat         : nombre d'intervalles > median × GAPS_STAT_FACTOR
      - gaps_fixed        : nombre d'intervalles > expected × GAPS_FIXED_FACTOR
                            (None si EXPECTED_PERIOD_S[canal] is None — canal event-driven)
    Retourne dict avec valeurs None si moins de 2 événements.
    """
    if len(timeline) < 2:
        return {
            "median_interval_s": None,
            "gaps_stat": None,
            "gaps_fixed": None,
        }
    monos = [e["mono"] for e in timeline]
    intervals = [monos[i + 1] - monos[i] for i in range(len(monos) - 1)]
    med = median(intervals)
    gaps_stat = sum(1 for itv in intervals if itv > med * GAPS_STAT_FACTOR)
    expected = EXPECTED_PERIOD_S.get(canal)
    if expected is None:
        gaps_fixed = None
    else:
        gaps_fixed = sum(1 for itv in intervals if itv > expected * GAPS_FIXED_FACTOR)
    return {
        "median_interval_s": med,
        "gaps_stat": gaps_stat,
        "gaps_fixed": gaps_fixed,
    }

def delta_pct(target: float | None, ref: float | None) -> float | None:
    """((target - ref) / ref) * 100. None si l'un ou l'autre est None ou ref == 0."""
    if target is None or ref is None or ref == 0:
        return None
    return _r((target - ref) / ref * 100)

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

def collect_frame_samples(frame_rows: list[dict]) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """
    Depuis les lignes du canal frame, collecte :
      - exact_samples  : lignes où count == 1, valeur = avg
      - approx_samples : toutes les lignes, valeur = avg
    Retourne (exact_samples, approx_samples) indexés par probe_name.
    Lit uniquement la section `probes` (structure fixe documentée).
    """
    exact: dict[str, list[float]] = {}
    approx: dict[str, list[float]] = {}
    for row in frame_rows:
        probes = row.get("probes")
        if not isinstance(probes, dict):
            continue
        for probe, stats in probes.items():
            if not isinstance(stats, dict):
                continue
            avg = stats.get("avg")
            cnt = stats.get("count")
            if avg is None or cnt is None:
                continue
            key = probe
            approx.setdefault(key, []).append(float(avg))
            if cnt == 1:
                exact.setdefault(key, []).append(float(avg))
    return exact, approx

def collect_fast_approx_samples(fast_rows: list[dict]) -> dict[str, list[float]]:
    """
    Depuis les lignes du canal fast, collecte approx_samples par probe_name.
    *_exact toujours null pour les sondes fast_* (pas de données individuelles).
    Lit uniquement la section `probes` (structure fixe documentée).
    """
    approx: dict[str, list[float]] = {}
    for row in fast_rows:
        probes = row.get("probes")
        if not isinstance(probes, dict):
            continue
        for probe, stats in probes.items():
            if not isinstance(stats, dict):
                continue
            avg = stats.get("avg")
            if avg is None:
                continue
            key = probe
            approx.setdefault(key, []).append(float(avg))
    return approx

def filter_rows_by_mono(rows: list[dict], t_start: float, t_end: float) -> list[dict]:
    """
    Filtre les lignes dont le champ `mono` appartient à [t_start, t_end).
    Borne droite exclusive. Lignes sans `mono` exploitable exclues.
    """
    result = []
    for row in rows:
        mono = row.get("mono")
        if isinstance(mono, (int, float)) and t_start <= mono < t_end:
            result.append(row)
    return result

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

def collect_frame_exact_pairs(frame_rows: list[dict]) -> dict[str, list[tuple[float, float]]]:
    """
    Depuis les lignes du canal frame, collecte les paires (avg, mono)
    pour les lignes où count == 1 (échantillons exacts S5b).
    Indexées par probe_name. Aucun tri (responsabilité appelant).
    Lit uniquement la section `probes` (structure fixe documentée).
    """
    pairs: dict[str, list[tuple[float, float]]] = {}
    for row in frame_rows:
        mono = row.get("mono")
        if not isinstance(mono, (int, float)):
            continue
        probes = row.get("probes")
        if not isinstance(probes, dict):
            continue
        for probe, stats in probes.items():
            if not isinstance(stats, dict):
                continue
            avg = stats.get("avg")
            cnt = stats.get("count")
            if avg is None or cnt != 1:
                continue
            pairs.setdefault(probe, []).append((float(avg), float(mono)))
    return pairs

def _empty_anomalies() -> dict:
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

