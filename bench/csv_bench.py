# csv_bench.py — CSV benchmark extraction
import os
import time
import csv
from config import cfg

# ── état interne ──────────────────────────────────────────────
_frame_file = None
_frame_writer = None
_frame_headers_written = False

_agg_file = None
_agg_writer = None
_agg_headers_written = False

_mask_file = None
_mask_writer = None
_mask_headers_written = False

_fast_file = None
_fast_writer = None
_fast_headers_written = False

_frame_enabled = False
_agg_enabled = False
_mask_enabled = False
_fast_enabled = False

def csv_open():
    """Ouvre 0 a 4 fichiers CSV selon la config."""
    global _frame_file, _frame_writer, _frame_headers_written
    global _agg_file, _agg_writer, _agg_headers_written
    global _mask_file, _mask_writer, _mask_headers_written
    global _fast_file, _fast_writer, _fast_headers_written
    global _frame_enabled, _agg_enabled,_mask_enabled,_fast_enabled

    _frame_enabled = cfg.get("debug.csv.per_frame", False)
    _agg_enabled = cfg.get("debug.csv.aggregated", False)
    _mask_enabled = cfg.get("debug.csv.mask", False)
    _fast_enabled  = cfg.get("debug.csv.fast", False)

    if not (_frame_enabled or _agg_enabled or _mask_enabled or _fast_enabled):
        return

    os.makedirs("logs", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    if _frame_enabled:
        path_f = f"logs/bench_frame_{ts}.csv"
        _frame_file = open(path_f, "w", newline="")
        _frame_writer = None
        _frame_headers_written = False
        print(f"📊 CSV per-frame  → {path_f}")

    if _agg_enabled:
        path_a = f"logs/bench_agg_{ts}.csv"
        _agg_file = open(path_a, "w", newline="")
        _agg_writer = None
        _agg_headers_written = False
        print(f"📊 CSV aggregated → {path_a}")

    if _mask_enabled:
        path_m = f"logs/bench_mask_{ts}.csv"
        _mask_file = open(path_m, "w", newline="")
        _mask_writer = None
        _mask_headers_written = False
        print(f"📊 CSV mask      → {path_m}")

    if _fast_enabled:
        path_ft = f"logs/bench_fast_{ts}.csv"
        _fast_file = open(path_ft, "w", newline="")
        _fast_writer = None
        _fast_headers_written = False
        print(f"📊 CSV fast       → {path_ft}")

def csv_write_frame(row: dict):
    """Écrit une ligne par frame (valeurs instantanées via bench.last())."""
    global _frame_writer, _frame_headers_written
    if not _frame_enabled or _frame_file is None:
        return
    # filtrer les None → colonnes vides
    cleaned = {k: (v if v is not None else "") for k, v in row.items()}
    if not _frame_headers_written:
        _frame_writer = csv.DictWriter(_frame_file, fieldnames=list(cleaned.keys()))
        _frame_writer.writeheader()
        _frame_headers_written = True
    _frame_writer.writerow(cleaned)


def csv_write_agg(row: dict):
    """Écrit une ligne agrégée (avg/max/min via bench.flat_row())."""
    global _agg_writer, _agg_headers_written
    if not _agg_enabled or _agg_file is None:
        return
    cleaned = {k: (v if v is not None else "") for k, v in row.items()}
    if not _agg_headers_written:
        _agg_writer = csv.DictWriter(_agg_file, fieldnames=list(cleaned.keys()))
        _agg_writer.writeheader()
        _agg_headers_written = True
    _agg_writer.writerow(cleaned)

def csv_write_mask(row: dict):
    """Écrit une ligne de masque."""
    global _mask_writer, _mask_headers_written
    if not _mask_enabled or _mask_file is None:
        return
    cleaned = {k: (v if v is not None else "") for k, v in row.items()}
    if not _mask_headers_written:
        _mask_writer = csv.DictWriter(_mask_file, fieldnames=list(cleaned.keys()))
        _mask_writer.writeheader()
        _mask_headers_written = True
    _mask_writer.writerow(cleaned)

def csv_write_fast(row: dict):
    """Écrit une ligne agrégée FastTrackThread (sondes fast_*, of_lk_*, etc.)."""
    global _fast_writer, _fast_headers_written
    if not _fast_enabled or _fast_file is None:
        return
    cleaned = {k: (v if v is not None else "") for k, v in row.items()}
    if not _fast_headers_written:
        _fast_writer = csv.DictWriter(
            _fast_file, fieldnames=list(cleaned.keys()), extrasaction="ignore"
        )
        _fast_writer.writeheader()
        _fast_headers_written = True
    _fast_writer.writerow(cleaned)

def csv_flush():
    """Flush les fichiers si ouverts."""
    for f in (_frame_file, _agg_file, _mask_file, _fast_file):
        if f is not None:
            f.flush()


def csv_close():
    """Ferme proprement tous les fichiers."""
    global _frame_file, _frame_writer, _frame_headers_written
    global _agg_file, _agg_writer, _agg_headers_written
    global _mask_file, _mask_writer, _mask_headers_written
    global _fast_file, _fast_writer, _fast_headers_written

    if _frame_file is not None:
        _frame_file.flush()
        _frame_file.close()
        _frame_file = None
        _frame_writer = None
        _frame_headers_written = False

    if _agg_file is not None:
        _agg_file.flush()
        _agg_file.close()
        _agg_file = None
        _agg_writer = None
        _agg_headers_written = False

    if _mask_file is not None:
        _mask_file.flush()
        _mask_file.close()
        _mask_file = None
        _mask_writer = None
        _mask_headers_written = False

    if _fast_file is not None:
        _fast_file.flush();
        _fast_file.close()
        _fast_file = None;
        _fast_writer = None;
        _fast_headers_written = False
