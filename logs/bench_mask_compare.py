# bench_mask_compare.py
"""
bench_mask_compare.py
─────────────────────
Analyse comparative chronologique des CSV de benchmark de masques.

Lit tous les fichiers `bench_mask_*.csv` d'un répertoire, les trie par date,
utilise le plus ancien comme baseline (ou un fichier spécifié), et produit un
rapport d'évolution des métriques clés (santé tracker, FPs fossilisés, hashes).

Usage:
    python bench_mask_compare.py <dir>
    python bench_mask_compare.py <dir> --baseline bench_mask_20260424_180335.csv
    python bench_mask_compare.py <dir> --export report.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import pandas as pd


# ────────────────────────────────────────────────────────────────────────────
# Parsing & chargement
# ────────────────────────────────────────────────────────────────────────────

FNAME_RE = re.compile(r"bench_mask_(\d{8})_(\d{6})\.csv$")


def find_bench_files(root: Path) -> list[Path]:
    """Retourne les fichiers bench_mask_*.csv triés chronologiquement."""
    files = [p for p in root.iterdir() if FNAME_RE.search(p.name)]

    def sort_key(p: Path) -> str:
        m = FNAME_RE.search(p.name)
        return f"{m.group(1)}{m.group(2)}" if m else p.name

    return sorted(files, key=sort_key)


def load_csv(path: Path) -> pd.DataFrame:
    """Charge un CSV et dérive les colonnes hash_history."""
    df = pd.read_csv(path)

    if "hash_history" in df.columns:
        hh_series = df["hash_history"].fillna("").astype(str)
        hh_lists = hh_series.apply(lambda s: [int(h) for h in s.split("|") if h])

        df["hh_len"] = hh_lists.apply(len)
        df["hh_unique"] = hh_lists.apply(lambda lst: len(set(lst)) if lst else 0)
        df["hh_hamming_avg"] = hh_lists.apply(_hamming_avg)
        df["hh_hamming_max"] = hh_lists.apply(_hamming_max)
        df["hh_frozen"] = (df["hh_unique"] <= 1) & (df["hh_len"] >= 3)

    return df


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _hamming_avg(lst: list[int]) -> float:
    if len(lst) < 2:
        return 0.0
    return sum(_hamming(lst[i], lst[i - 1]) for i in range(1, len(lst))) / (len(lst) - 1)


def _hamming_max(lst: list[int]) -> int:
    if len(lst) < 2:
        return 0
    return max(_hamming(lst[i], lst[i - 1]) for i in range(1, len(lst)))


# ────────────────────────────────────────────────────────────────────────────
# Calcul des métriques
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class Metrics:
    file: str
    n_rows: int
    n_frames: int
    n_uids: int
    duration_s: float

    # Sources
    pct_slow: float
    pct_fast: float
    pct_predict: float

    # Santé trackers
    pct_fast_miss_nonzero: float
    confidence_mean: float
    confidence_fast_mean: float
    confidence_slow_mean: float

    # FP fossilisés (heuristique)
    n_fp_suspects: int
    pct_fp_suspects: float
    max_frames_missing: int
    max_hh_constant_streak: int  # + long streak hash figé sur un uid

    # Hash quality (ne concerne que les lignes avec hh_len >= 2)
    hh_hamming_avg_mean: float
    pct_hh_frozen: float  # % lignes avec hash_history 100% identique (len>=3)

    # Vitesses
    pct_teleport: float  # déplacement > 500px en 1 frame sans slow


def compute_metrics(df: pd.DataFrame, file_name: str) -> Metrics:
    n_rows = len(df)
    n_frames = df["frame_id"].nunique() if "frame_id" in df else 0
    n_uids = df["uid"].nunique() if "uid" in df else 0
    duration = (
        df["timestamp"].max() - df["timestamp"].min()
        if "timestamp" in df and n_rows > 0
        else 0.0
    )

    # Sources
    src = df["last_source"].value_counts(normalize=True) if "last_source" in df else {}
    pct_slow = 100 * src.get("slow", 0.0)
    pct_fast = 100 * src.get("fast", 0.0)
    pct_predict = 100 * src.get("predict", 0.0)

    # Santé
    pct_fast_miss = (
        100 * (df["fast_miss_count"] > 0).mean() if "fast_miss_count" in df else 0.0
    )
    conf_mean = df["confidence"].mean() if "confidence" in df else 0.0
    conf_fast = (
        df.loc[df["last_source"] == "fast", "confidence"].mean()
        if "last_source" in df
        else 0.0
    )
    conf_slow = (
        df.loc[df["last_source"] == "slow", "confidence"].mean()
        if "last_source" in df
        else 0.0
    )

    # FPs fossilisés : frames_missing > 50  ET  hash_history quasi figé
    #                  ET  confidence slow faible  ET  vitesse ~0
    fp_mask = pd.Series(False, index=df.index)
    if {"frames_missing", "hh_hamming_avg", "vx", "vy"}.issubset(df.columns):
        fp_mask = (
            (df["frames_missing"] > 50)
            & (df["hh_hamming_avg"] < 2)
            & (df["vx"].abs() < 5)
            & (df["vy"].abs() < 5)
        )

    n_fp = int(fp_mask.sum())
    pct_fp = 100 * n_fp / n_rows if n_rows else 0.0

    max_missing = int(df["frames_missing"].max()) if "frames_missing" in df else 0

    # Plus long streak de hh_frozen par UID
    max_streak = 0
    if "hh_frozen" in df and "uid" in df:
        for _, grp in df.sort_values(["uid", "frame_id"]).groupby("uid"):
            streak = cur = 0
            for v in grp["hh_frozen"]:
                cur = cur + 1 if v else 0
                streak = max(streak, cur)
            max_streak = max(max_streak, streak)

    hh_avg = df.loc[df["hh_len"] >= 2, "hh_hamming_avg"].mean() if "hh_len" in df else 0.0
    pct_frozen = 100 * df["hh_frozen"].mean() if "hh_frozen" in df else 0.0

    # Téléportation : grand saut en 1 frame pour un même uid en fast
    pct_tel = 0.0
    if {"uid", "frame_id", "x", "y", "last_source"}.issubset(df.columns):
        d = df.sort_values(["uid", "frame_id"]).copy()
        d["dx"] = d.groupby("uid")["x"].diff().abs()
        d["dy"] = d.groupby("uid")["y"].diff().abs()
        d["dframe"] = d.groupby("uid")["frame_id"].diff()
        tel = (
            (d["dframe"] == 1)
            & (d["last_source"] == "fast")
            & ((d["dx"] > 500) | (d["dy"] > 500))
        )
        pct_tel = 100 * tel.sum() / len(d) if len(d) else 0.0

    return Metrics(
        file=file_name,
        n_rows=n_rows,
        n_frames=int(n_frames),
        n_uids=int(n_uids),
        duration_s=round(float(duration), 2),
        pct_slow=round(pct_slow, 2),
        pct_fast=round(pct_fast, 2),
        pct_predict=round(pct_predict, 2),
        pct_fast_miss_nonzero=round(pct_fast_miss, 3),
        confidence_mean=round(float(conf_mean), 4),
        confidence_fast_mean=round(float(conf_fast), 4),
        confidence_slow_mean=round(float(conf_slow), 4),
        n_fp_suspects=n_fp,
        pct_fp_suspects=round(pct_fp, 2),
        max_frames_missing=max_missing,
        max_hh_constant_streak=max_streak,
        hh_hamming_avg_mean=round(float(hh_avg), 2),
        pct_hh_frozen=round(float(pct_frozen), 2),
        pct_teleport=round(pct_tel, 3),
    )


# ────────────────────────────────────────────────────────────────────────────
# Comparaison & affichage
# ────────────────────────────────────────────────────────────────────────────

# Direction d'amélioration : +1 = plus c'est grand mieux c'est, -1 = inverse, 0 = neutre
METRIC_DIRECTION = {
    "n_rows": 0,
    "n_frames": 0,
    "n_uids": 0,
    "duration_s": 0,
    "pct_slow": 0,
    "pct_fast": 0,
    "pct_predict": 0,
    "pct_fast_miss_nonzero": -1,
    "confidence_mean": +1,
    "confidence_fast_mean": +1,
    "confidence_slow_mean": +1,
    "n_fp_suspects": -1,
    "pct_fp_suspects": -1,
    "max_frames_missing": -1,
    "max_hh_constant_streak": -1,
    "hh_hamming_avg_mean": 0,
    "pct_hh_frozen": -1,
    "pct_teleport": -1,
}

# Seuil minimum pour considérer une variation comme significative
SIGNIF_THRESHOLDS = {
    "pct_fast_miss_nonzero": 0.5,
    "confidence_mean": 0.02,
    "confidence_fast_mean": 0.02,
    "confidence_slow_mean": 0.02,
    "pct_fp_suspects": 1.0,
    "max_frames_missing": 20,
    "max_hh_constant_streak": 5,
    "pct_hh_frozen": 2.0,
    "pct_teleport": 0.1,
}


def status_symbol(metric: str, baseline_val, current_val) -> str:
    direction = METRIC_DIRECTION.get(metric, 0)
    if direction == 0:
        return "·"
    try:
        delta = current_val - baseline_val
    except TypeError:
        return "·"
    threshold = SIGNIF_THRESHOLDS.get(metric, 0)
    if abs(delta) < threshold:
        return "="
    if (delta > 0 and direction > 0) or (delta < 0 and direction < 0):
        return "✓"  # amélioration
    return "✗"  # régression


def _print_report(baseline: Metrics, others: list[Metrics]) -> None:
    b = asdict(baseline)

    print("\n" + "═" * 100)
    print(f" BASELINE : {baseline.file}")
    print("═" * 100)
    for k, v in b.items():
        if k == "file":
            continue
        print(f"  {k:30s} {v}")

    if not others:
        print("\n  Aucun autre fichier à comparer.\n")
        return

    print("\n" + "═" * 100)
    print(" COMPARAISON CHRONOLOGIQUE vs BASELINE")
    print("═" * 100)

    # Header
    metrics_keys = [k for k in b.keys() if k != "file"]
    header = f"{'metric':30s} │ {'baseline':>12s}"
    for o in others:
        label = o.file.replace("bench_mask_", "").replace(".csv", "")
        header += f" │ {label:>18s}"
    print(header)
    print("─" * len(header))

    for k in metrics_keys:
        base_val = b[k]
        line = f"{k:30s} │ {base_val:>12}"
        for o in others:
            cur = asdict(o)[k]
            sym = status_symbol(k, base_val, cur)
            cell = f"{cur} {sym}"
            line += f" │ {cell:>18s}"
        print(line)

    # Légende
    print("\n  Légende : ✓ amélioration · ✗ régression · = stable · · neutre\n")


def export_csv(all_metrics: list[Metrics], out: Path) -> None:
    pd.DataFrame([asdict(m) for m in all_metrics]).to_csv(out, index=False)
    print(f"→ Export : {out}")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", type=Path, help="Répertoire contenant les bench_mask_*.csv")
    ap.add_argument("--baseline", type=str, default=None,
                    help="Nom du fichier baseline (défaut: plus ancien chronologiquement)")
    ap.add_argument("--export", type=Path, default=None,
                    help="Exporte un CSV récapitulatif")
    args = ap.parse_args()

    if not args.directory.is_dir():
        sys.exit(f"✗ Répertoire introuvable : {args.directory}")

    files = find_bench_files(args.directory)
    if not files:
        sys.exit(f"✗ Aucun fichier bench_mask_*.csv dans {args.directory}")

    print(f"\n→ {len(files)} fichier(s) trouvé(s) dans {args.directory}")
    for f in files:
        print(f"   · {f.name}")

    # Sélection baseline
    if args.baseline:
        baseline_path = args.directory / args.baseline
        if not baseline_path.exists():
            sys.exit(f"✗ Baseline introuvable : {baseline_path}")
    else:
        baseline_path = files[0]

    print(f"\n→ Baseline : {baseline_path.name}")

    # Calcul des métriques
    all_metrics: list[Metrics] = []
    baseline_metrics: Metrics | None = None

    for f in files:
        try:
            df = load_csv(f)
            m = compute_metrics(df, f.name)
            all_metrics.append(m)
            if f == baseline_path:
                baseline_metrics = m
        except Exception as e:
            print(f"  ⚠ Erreur sur {f.name} : {e}")

    if baseline_metrics is None:
        sys.exit("✗ Impossible de calculer les métriques de la baseline")

    # Autres fichiers (hors baseline), ordre chronologique
    others = [m for m in all_metrics if m.file != baseline_metrics.file]
    _print_report(baseline_metrics, others)

    if args.export:
        export_csv(all_metrics, args.export)


if __name__ == "__main__":
    main()
