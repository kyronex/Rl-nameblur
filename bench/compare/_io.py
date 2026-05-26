# bench/compare/_io.py

"""
I/O JSONL, découverte des sessions, déplacement FS, écriture rapport.

Toutes les fonctions de ce module ont des effets de bord filesystem
ou manipulent le format JSONL. Aucun calcul statistique ici.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
import json

from bench.compare._config import DIR_JSON, DIR_RESULTS

log = logging.getLogger(__name__)

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

def find_sessions() -> dict[str, dict[str, Path]]:
    """
    Retourne {session_id: {"agg": Path, "fast": Path|None, "frame": Path|None}}
    pour toutes les sessions disponibles dans logs/json/ et logs/results/.
    logs/json/ est prioritaire en cas de doublon session_id.
    """
    candidates: dict[str, dict[str, Path]] = {}
    _ingest_directory(DIR_RESULTS, priority=False, candidates=candidates)
    _ingest_directory(DIR_JSON, priority=True, candidates=candidates)
    return candidates


def load_session(files: dict[str, Path]) -> tuple[list, list, list]:
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
# Déplacement des fichiers
# ---------------------------------------------------------------------------

def move_session_to_results(session_id: str, files: dict[str, Path]) -> None:
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

def write_report(report: dict, report_path: Path) -> None:
    """
    Écrit le rapport JSON via fichier temporaire (.tmp) + replace atomique.
    Lève OSError en cas d'échec.
    """
    tmp_path = report_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(report_path)
    log.info("Rapport écrit : %s", report_path)