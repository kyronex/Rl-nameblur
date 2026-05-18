# logs/bench_compare.py
"""
Analyse comparative des sessions de benchmark RL-NameBlur.

Sélectionne la session la plus récente disponible (logs/json/ + logs/results/)
comme cible, la compare à :
  - référence absolue : session la plus ancienne
  - référence relative : avant-dernière session (null si N == 2)

Produit : logs/results/<session_id>/<session_id>.json
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from statistics import quantiles

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(levelname)s — %(message)s"
ROUND_DIGITS = 3
PERCENTILE_MIN_SAMPLES = 20
PERCENTILES = [90, 95, 99]

DIR_JSON = Path("logs/json")
DIR_RESULTS = Path("logs/results")

FORMAT_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers — arrondi
# ---------------------------------------------------------------------------


def _r(val: float | None) -> float | None:
    """Arrondi à ROUND_DIGITS décimales, None passthrough."""
    if val is None:
        return None
    return round(val, ROUND_DIGITS)


# ---------------------------------------------------------------------------
# Helpers — JSONL
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    """Lit un fichier JSONL. Ignore les lignes malformées avec warning."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                log.warning("Ligne ignorée — %s:%d — %s", path.name, lineno, exc)
    return rows


# ---------------------------------------------------------------------------
# Découverte des sessions
# ---------------------------------------------------------------------------


def _session_id_from_stem(stem: str) -> str | None:
    """
    Extrait le session_id (YYYYMMDD_HHMMSS) depuis un stem de fichier.
    Exemple : bench_agg_20260519_091540 → 20260519_091540
    """
    parts = stem.split("_", maxsplit=2)
    if len(parts) == 3:
        return parts[2]
    return None


def _ingest_directory(directory: Path, priority: bool, candidates: dict) -> None:
    """
    Parcourt un répertoire et alimente le dict candidates.
    Si priority=True (logs/json/), écrase les entrées existantes avec warning.
    Sessions sans fichier agg ignorées avec warning.
    """
    if not directory.exists():
        return

    jsonl_files = list(directory.glob("bench_*_????????_??????.jsonl"))
    jsonl_files += list(directory.glob("*/bench_*_????????_??????.jsonl"))

    grouped: dict[str, dict[str, Path]] = {}
    for path in jsonl_files:
        sid = _session_id_from_stem(path.stem)
        if sid is None:
            continue
        grouped.setdefault(sid, {})
        if "bench_agg_" in path.stem:
            grouped[sid]["agg"] = path
        elif "bench_fast_" in path.stem:
            grouped[sid]["fast"] = path
        elif "bench_frame_" in path.stem:
            grouped[sid]["frame"] = path

    for sid, files in grouped.items():
        if "agg" not in files:
            log.warning("Session %s ignorée — fichier agg introuvable.", sid)
            continue
        if sid in candidates and not priority:
            continue
        if sid in candidates and priority:
            log.warning(
                "Doublon session_id %s — logs/json/ prioritaire, "
                "logs/results/%s/ sera remplacé.",
                sid,
                sid,
            )
        candidates[sid] = {
            "agg": files["agg"],
            "fast": files.get("fast"),
            "frame": files.get("frame"),
        }


def _find_sessions() -> dict[str, dict[str, Path]]:
    """
    Retourne {session_id: {"agg": Path, "fast": Path|None, "frame": Path|None}}
    pour toutes les sessions disponibles dans logs/json/ et logs/results/.
    logs/json/ est prioritaire en cas de doublon session_id.
    """
    candidates: dict[str, dict[str, Path]] = {}
    _ingest_directory(DIR_RESULTS, priority=False, candidates=candidates)
    _ingest_directory(DIR_JSON, priority=True, candidates=candidates)
    return candidates


# ---------------------------------------------------------------------------
# Chargement d'une session
# ---------------------------------------------------------------------------


def _load_session(files: dict[str, Path]) -> tuple[list, list, list]:
    """
    Retourne (agg_rows, frame_rows, fast_rows).
    Émet un warning si frame ou fast est absent.
    """
    sid = files["agg"].stem.split("bench_agg_")[-1]

    agg_rows = _read_jsonl(files["agg"])

    frame_path = files.get("frame")
    if frame_path:
        frame_rows = _read_jsonl(frame_path)
    else:
        frame_rows = []
        log.warning(
            "Fichier frame absent pour session %s "
            "— tous les percentiles *_exact seront null.",
            sid,
        )

    fast_path = files.get("fast")
    if fast_path:
        fast_rows = _read_jsonl(fast_path)
    else:
        fast_rows = []
        log.warning(
            "Fichier fast absent pour session %s — sondes fast_* seront null.",
            sid,
        )

    return agg_rows, frame_rows, fast_rows


# ---------------------------------------------------------------------------
# Agrégation — canal agg
# ---------------------------------------------------------------------------


def _agg_probes(rows: list[dict]) -> dict[str, dict]:
    """
    Agrège les sondes depuis les lignes du canal agg.
    Retourne {probe_name: {avg, min, max, count}}.
    Ignore les clés non-domaine (ts, rates, gauges).
    """
    accum: dict[str, dict] = {}

    for row in rows:
        for domain, probes in row.items():
            if domain in ("ts", "rates", "gauges") or not isinstance(probes, dict):
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
                key = f"{domain}_{probe}"
                if key not in accum:
                    accum[key] = {
                        "sum_weighted": 0.0,
                        "min": mn,
                        "max": mx,
                        "count": 0,
                    }
                accum[key]["sum_weighted"] += avg * cnt
                accum[key]["min"] = min(accum[key]["min"], mn)
                accum[key]["max"] = max(accum[key]["max"], mx)
                accum[key]["count"] += cnt

    result: dict[str, dict] = {}
    for key, acc in accum.items():
        result[key] = {
            "avg": acc["sum_weighted"] / acc["count"],
            "min": acc["min"],
            "max": acc["max"],
            "count": acc["count"],
        }
    return result


def _agg_rates(rows: list[dict]) -> dict[str, float]:
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


def _agg_gauges(rows: list[dict]) -> dict[str, float]:
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


def _session_duration(rows: list[dict]) -> float | None:
    """ts dernière ligne - ts première ligne du canal agg."""
    timestamps = [
        r["ts"] for r in rows if "ts" in r and isinstance(r["ts"], (int, float))
    ]
    if len(timestamps) < 2:
        return None
    return float(max(timestamps) - min(timestamps))


# ---------------------------------------------------------------------------
# Percentiles — canaux frame et fast
# ---------------------------------------------------------------------------


def _percentile_value(data: list[float], pct: int) -> float | None:
    """
    Calcule un percentile via statistics.quantiles (method='inclusive').
    Retourne None si len(data) < PERCENTILE_MIN_SAMPLES.
    """
    if len(data) < PERCENTILE_MIN_SAMPLES:
        return None
    qs = quantiles(data, n=100, method="inclusive")
    return qs[pct - 1]


def _collect_frame_samples(
    frame_rows: list[dict],
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """
    Depuis les lignes du canal frame, collecte :
      - exact_samples  : lignes où count == 1, valeur = avg
      - approx_samples : toutes les lignes, valeur = avg
    Retourne (exact_samples, approx_samples) indexés par probe_name.
    """
    exact: dict[str, list[float]] = {}
    approx: dict[str, list[float]] = {}

    for row in frame_rows:
        for domain, probes in row.items():
            if domain in ("ts", "rates", "gauges") or not isinstance(probes, dict):
                continue
            for probe, stats in probes.items():
                if not isinstance(stats, dict):
                    continue
                avg = stats.get("avg")
                cnt = stats.get("count")
                if avg is None or cnt is None:
                    continue
                key = f"{domain}_{probe}"
                approx.setdefault(key, []).append(float(avg))
                if cnt == 1:
                    exact.setdefault(key, []).append(float(avg))

    return exact, approx


def _collect_fast_approx_samples(
    fast_rows: list[dict],
) -> dict[str, list[float]]:
    """
    Depuis les lignes du canal fast, collecte approx_samples par probe_name.
    *_exact toujours null pour les sondes fast_* (pas de données individuelles).
    """
    approx: dict[str, list[float]] = {}

    for row in fast_rows:
        for domain, probes in row.items():
            if domain in ("ts", "rates", "gauges") or not isinstance(probes, dict):
                continue
            for probe, stats in probes.items():
                if not isinstance(stats, dict):
                    continue
                avg = stats.get("avg")
                if avg is None:
                    continue
                key = f"{domain}_{probe}"
                approx.setdefault(key, []).append(float(avg))

    return approx


def _build_percentile_block(
    probe_name: str,
    exact_samples: dict[str, list[float]],
    approx_samples: dict[str, list[float]],
) -> dict:
    """
    Construit le bloc percentiles pour une sonde.
    Sondes fast_* : samples_exact = 0 et tous *_exact = null.
    """
    is_fast = probe_name.startswith("fast_")

    exact_data = [] if is_fast else exact_samples.get(probe_name, [])
    approx_data = approx_samples.get(probe_name, [])

    block: dict = {
        "samples_exact": 0 if is_fast else len(exact_data),
        "samples_approx": len(approx_data),
    }

    for pct in PERCENTILES:
        block[f"p{pct}_exact"] = None if is_fast else _percentile_value(exact_data, pct)
        block[f"p{pct}_approx"] = _percentile_value(approx_data, pct)

    return block


# ---------------------------------------------------------------------------
# Construction du bloc session
# ---------------------------------------------------------------------------


def _build_session_block(
    agg_rows: list[dict],
    frame_rows: list[dict],
    fast_rows: list[dict],
) -> dict:
    """
    Construit le bloc session complet : {duration_s, probes, rates, gauges}.
    """
    base_probes = _agg_probes(agg_rows)
    rates = _agg_rates(agg_rows)
    gauges = _agg_gauges(agg_rows)
    duration = _session_duration(agg_rows)

    exact_samples, frame_approx = _collect_frame_samples(frame_rows)
    fast_approx = _collect_fast_approx_samples(fast_rows)

    approx_samples: dict[str, list[float]] = {**frame_approx, **fast_approx}

    probes: dict[str, dict] = {}
    for probe_name, stats in base_probes.items():
        pct_block = _build_percentile_block(probe_name, exact_samples, approx_samples)
        probes[probe_name] = {
            "avg": _r(stats["avg"]),
            "min": _r(stats["min"]),
            "max": _r(stats["max"]),
            "count": stats["count"],
            **{k: _r(v) for k, v in pct_block.items()},
        }

    return {
        "duration_s": _r(duration),
        "probes": probes,
        "rates": {k: _r(v) for k, v in rates.items()},
        "gauges": {k: _r(v) for k, v in gauges.items()},
    }


# ---------------------------------------------------------------------------
# Calcul des deltas
# ---------------------------------------------------------------------------


def _delta_pct(target: float | None, ref: float | None) -> float | None:
    """((target - ref) / ref) * 100. None si l'un ou l'autre est None ou ref == 0."""
    if target is None or ref is None or ref == 0:
        return None
    return _r((target - ref) / ref * 100)


def _build_probe_deltas(target_probes: dict, ref_probes: dict) -> dict:
    """
    Construit les deltas pour toutes les sondes présentes dans target ou ref.
    Couvre avg, min, max et tous les percentiles (exact + approx).
    """
    all_keys = set(target_probes) | set(ref_probes)
    deltas: dict[str, dict] = {}

    for key in sorted(all_keys):
        t = target_probes.get(key, {})
        r = ref_probes.get(key, {})
        entry: dict = {}

        for field in ("avg", "min", "max"):
            entry[f"{field}_delta_pct"] = _delta_pct(t.get(field), r.get(field))

        for pct in PERCENTILES:
            for method in ("exact", "approx"):
                fname = f"p{pct}_{method}"
                entry[f"{fname}_delta_pct"] = _delta_pct(t.get(fname), r.get(fname))

        deltas[key] = entry

    return deltas


def _build_scalar_deltas(target: dict, ref: dict) -> dict:
    """Construit les deltas pour rates ou gauges (valeurs scalaires)."""
    all_keys = set(target) | set(ref)
    return {
        key: {"delta_pct": _delta_pct(target.get(key), ref.get(key))}
        for key in sorted(all_keys)
    }


def _appeared_disappeared(
    target_probes: dict, ref_probes: dict
) -> tuple[list, list]:
    """Retourne (appeared_in_target, disappeared_in_target)."""
    t_keys = set(target_probes)
    r_keys = set(ref_probes)
    return sorted(t_keys - r_keys), sorted(r_keys - t_keys)


def _build_comparison(
    ref_session_id: str,
    ref_block: dict,
    target_block: dict,
) -> dict:
    """Construit un bloc de comparaison complet target vs référence."""
    appeared, disappeared = _appeared_disappeared(
        target_block["probes"], ref_block["probes"]
    )
    return {
        "reference_session": ref_session_id,
        "reference": ref_block,
        "deltas": {
            "probes": _build_probe_deltas(
                target_block["probes"], ref_block["probes"]
            ),
            "rates": _build_scalar_deltas(
                target_block["rates"], ref_block["rates"]
            ),
            "gauges": _build_scalar_deltas(
                target_block["gauges"], ref_block["gauges"]
            ),
        },
        "appeared_in_target": appeared,
        "disappeared_in_target": disappeared,
    }


# ---------------------------------------------------------------------------
# Déplacement des fichiers
# ---------------------------------------------------------------------------


def _move_session_to_results(session_id: str, files: dict[str, Path]) -> None:
    """
    Déplace les fichiers d'une session depuis logs/json/ vers
    logs/results/<session_id>/. Ne déplace que les fichiers dans logs/json/.
    En cas de doublon confirmé, vide le dossier results/ existant avant déplacement.
    """
    dest_dir = DIR_RESULTS / session_id

    if dest_dir.exists():
        for f in dest_dir.iterdir():
            f.unlink()
        log.warning("Dossier logs/results/%s/ vidé (doublon résolu).", session_id)
    else:
        dest_dir.mkdir(parents=True, exist_ok=True)

    for path in files.values():
        if path and path.is_relative_to(DIR_JSON):
            shutil.move(str(path), dest_dir / path.name)
            log.info("Déplacé : %s → logs/results/%s/", path.name, session_id)


# ---------------------------------------------------------------------------
# Écriture du rapport
# ---------------------------------------------------------------------------


def _write_report(report: dict, report_path: Path) -> None:
    """
    Écrit le rapport JSON via fichier temporaire (.tmp) + replace atomique.
    Lève OSError en cas d'échec.
    """
    tmp_path = report_path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tmp_path.replace(report_path)
    log.info("Rapport écrit : %s", report_path)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


def main() -> None:
    sessions = _find_sessions()

    if len(sessions) == 0:
        log.error("Aucune session disponible.")
        sys.exit(1)

    if len(sessions) < 2:
        log.error("Moins de 2 sessions disponibles — comparaison impossible.")
        sys.exit(1)

    sorted_ids = sorted(sessions.keys())
    target_id = sorted_ids[-1]
    absolute_id = sorted_ids[0]
    relative_id = sorted_ids[-2] if len(sorted_ids) >= 3 else None

    log.info("Cible       : %s", target_id)
    log.info("Abs. réf.   : %s", absolute_id)
    log.info("Rel. réf.   : %s", relative_id if relative_id else "N/A (N==2)")

    target_agg, target_frame, target_fast = _load_session(sessions[target_id])
    abs_agg, abs_frame, abs_fast = _load_session(sessions[absolute_id])

    target_block = _build_session_block(target_agg, target_frame, target_fast)
    abs_block = _build_session_block(abs_agg, abs_frame, abs_fast)

    rel_block = None
    if relative_id:
        rel_agg, rel_frame, rel_fast = _load_session(sessions[relative_id])
        rel_block = _build_session_block(rel_agg, rel_frame, rel_fast)

    report = {
        "format_version": FORMAT_VERSION,
        "generated_at": datetime.now().isoformat(),
        "target_session": target_id,
        "target": target_block,
        "comparisons": {
            "absolute": _build_comparison(absolute_id, abs_block, target_block),
            "relative": (
                _build_comparison(relative_id, rel_block, target_block)
                if rel_block is not None
                else None
            ),
        },
    }

    target_in_json = sessions[target_id]["agg"].is_relative_to(DIR_JSON)
    report_dir = DIR_RESULTS / target_id
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{target_id}.json"

    try:
        _write_report(report, report_path)
    except OSError as exc:
        log.error("Échec écriture rapport — aucun fichier déplacé. %s", exc)
        sys.exit(1)

    if target_in_json:
        _move_session_to_results(target_id, sessions[target_id])


if __name__ == "__main__":
    main()
