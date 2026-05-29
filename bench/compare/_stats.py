# bench/compare/_stats.py

from statistics import median , quantiles
from bench.compare._config import (
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


