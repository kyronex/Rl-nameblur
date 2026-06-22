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
    Pattern : bench_{mode}_{YYYYMMDD}_{HHMMSS}_{index}.jsonl
    Exemples :
      bench_agg_20260519_091540_0    → 20260519_091540
      bench_frame_20260519_091540_12 → 20260519_091540
    Le suffixe _<index> est retiré pour produire un session_id stable.
    """
    parts = stem.split("_", maxsplit=2)
    if len(parts) == 3:
        return parts[2].rsplit("_", 1)[0]
    return None

def _index_from_stem(stem: str) -> int:
    """
    Extrait l'index entier depuis un stem de fichier.
    Exemple : bench_frame_20260519_091540_3 → 3
    Retourne 0 si absent (comportement backward-compatible).
    """
    parts = stem.rsplit("_", maxsplit=1)
    if len(parts) == 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return 0

def _ingest_directory(directory: Path, priority: bool, candidates: dict) -> None:
    """
    Parcourt un répertoire et alimente le dict candidates.
    Si priority=True (logs/json/), écrase les entrées existantes avec warning.
    Sessions sans fichier agg ignorées avec warning.
    Chaque mode est acumulé dans une liste triée par index entier
    (support natif du format rotatif bench_{mode}_{date}_{time}_{index}.jsonl).
    """
    if not directory.exists():
        return

    # Pattern corrigé : {mode}_{date}_{time}_{index}.jsonl
    jsonl_files = list(directory.glob("bench_*_*_*_*.jsonl"))
    jsonl_files += list(directory.glob("*/bench_*_*_*_*.jsonl"))

    grouped: dict[str, dict[str, list[Path]]] = {}
    for path in jsonl_files:
        sid = _session_id_from_stem(path.stem)
        if sid is None:
            continue
        # Initialiser toutes les listes au premier passage (évite KeyError)
        grouped.setdefault(sid, {"agg": [], "fast": [], "frame": []})
        if "bench_agg_" in path.stem:
            grouped[sid]["agg"].append(path)
        elif "bench_fast_" in path.stem:
            grouped[sid]["fast"].append(path)
        elif "bench_frame_" in path.stem:
            grouped[sid]["frame"].append(path)

    for sid, files in grouped.items():
        if not files["agg"]:
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
        # Tri numérique par index fichier (0,1,2,...,10 — pas lexicographique)
        for mode in ("agg", "fast", "frame"):
            files[mode].sort(key=lambda p: _index_from_stem(p.stem))
        candidates[sid] = {
            "agg":   files["agg"],
            "fast":  files["fast"],
            "frame": files["frame"],
        }

def find_sessions() -> dict[str, dict[str, list[Path]]]:
    """
    Retourne {session_id: {"agg": [Path, ...], "fast": [Path, ...], "frame": [Path, ...]}}
    pour toutes les sessions disponibles dans logs/json/ et logs/results/.
    logs/json/ est prioritaire en cas de doublon session_id.

    Chaque liste est triée par index entier croissant (support du format rotatif).
    Une session sans fast ou sans frame a une liste vide (pas None).
    """
    candidates: dict[str, dict[str, list[Path]]] = {}
    _ingest_directory(DIR_RESULTS, priority=False, candidates=candidates)
    _ingest_directory(DIR_JSON, priority=True, candidates=candidates)
    return candidates

def load_session(files: dict[str, list[Path]]) -> tuple[list, list, list]:
    """
    Charge et concatène TOUTES les lignes JSONL d'une session,
    dans l'ordre des index fichier (0 → 1 → 2 → ...).
    Retourne (agg_rows, frame_rows, fast_rows).
    Émet un warning si frame ou fast est absent (liste vide).
    """
    sid = _session_id_from_stem(files["agg"][0].stem)
    # Concaténation de tous les fichiers agg, triés par index
    agg_rows: list = []
    for path in files["agg"]:
        agg_rows.extend(_read_jsonl(path))
    # frame : concaténation multi-fichiers si présent
    if files["frame"]:
        frame_rows: list = []
        for path in files["frame"]:
            frame_rows.extend(_read_jsonl(path))
    else:
        frame_rows = []
        log.warning("Fichier frame absent pour session %s — tous les percentiles *_exact seront null.",sid,)
    # fast : concaténation multi-fichiers si présent
    if files["fast"]:
        fast_rows: list = []
        for path in files["fast"]:
            fast_rows.extend(_read_jsonl(path))
    else:
        fast_rows = []
        log.warning("Fichier fast absent pour session %s — sondes fast_* seront null.",sid,)
    return agg_rows, frame_rows, fast_rows

# ---------------------------------------------------------------------------
# Déplacement des fichiers
# ---------------------------------------------------------------------------

def move_session_to_results(session_id: str, files: dict[str, list[Path]]) -> None:
    """
    Déplace TOUS les fichiers d'une session (tous les index de chaque mode)
    depuis logs/json/ vers logs/results/<session_id>/.
    Ne déplace que les fichiers contenus dans logs/json/.
    En cas de doublon confirmé, vide le dossier results/ existant avant déplacement.
    """
    dest_dir = DIR_RESULTS / session_id
    if dest_dir.exists():
        for f in dest_dir.iterdir():
            f.unlink()
        log.warning("Dossier logs/results/%s/ vidé (doublon résolu).", session_id)
    else:
        dest_dir.mkdir(parents=True, exist_ok=True)
    for path_list in files.values():
        for path in path_list:
            if path.is_relative_to(DIR_JSON):
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