#!/usr/bin/env python3
"""
bench_pipeline.py — Benchmark du pipeline de détection plaque Rocket League.

Appelle les MÊMES fonctions que detect.py, dans le MÊME ordre.
Aucune recopie de logique — uniquement instrumentation (timer + compteurs).

Usage:
    python bench_pipeline.py frame.png
    python bench_pipeline.py frame.png --n 20
    python bench_pipeline.py frame.png --scale 2.0
"""
import logging
from config import cfg
def setup_logging():
    level_str = cfg.get("debug.log_level", "WARNING")
    level = getattr(logging, level_str.upper(), logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(name)s | %(levelname)s | %(message)s"
    )
setup_logging()

import cv2
import numpy as np
import sys
import time
import argparse

from detect import _build_params
from detect_tools_mask import (saturation_variance_mask,compute_white_mask,refine_and_merge,compute_sobel_interior_unified)
from detect_tools_boxes import (process_channel)
from detect_tools import (write_circles,write_rects,get_color)
log = logging.getLogger("bench_pipeline")
# ═══════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════

def _timed(fn, *args, **kwargs):
    """Exécute fn, retourne (résultat, durée_ms)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - t0) * 1000


def _count(obj):
    """Compte: pixels non-nuls pour un mask, len() pour une liste."""
    if obj is None:
        return 0
    if isinstance(obj, np.ndarray) and obj.ndim == 2:
        return int(np.count_nonzero(obj))
    if isinstance(obj, (list, tuple)):
        return len(obj)
    return 0


# ═══════════════════════════════════════════════════
# Pipeline bench — miroir exact de detect._run_pipeline
# ═══════════════════════════════════════════════════

def bench_once(frame, scale):
    """Exécute le pipeline étape par étape, retourne un dict de timings (ms)."""
    timings = {}
    colors, kernels, params, letter_connect_iter = _build_params(scale)

    # ── 1. Resize ──
    t = time.perf_counter()
    h_orig, w_orig = frame.shape[:2]
    w_small = int(w_orig / scale)
    h_small = int(h_orig / scale)
    small = cv2.resize(frame, (w_small, h_small), interpolation=cv2.INTER_LINEAR)
    timings["1_resize"] = (time.perf_counter() - t) * 1000

    # ── 2. Grayscale ──
    t = time.perf_counter()
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    timings["2_grayscale"] = (time.perf_counter() - t) * 1000

    # ── 3. Saturation variance mask ──
    sat_mask, ms = _timed(saturation_variance_mask, small , scale)
    timings["3_sat_var_mask"] = ms

    # ── 4. White mask ──
    (mask_white, white_clean), ms = _timed(compute_white_mask, gray, kernels, letter_connect_iter)
    timings["4_white_mask"] = ms

    # ── 5. Combine white + sat ──
    t = time.perf_counter()
    combined = cv2.bitwise_and(white_clean, sat_mask)
    timings["5_combine"] = (time.perf_counter() - t) * 1000

    # ── 6. Sobel interiors ──
    interior, ms = _timed(compute_sobel_interior_unified, gray, combined, kernels)
    timings["6_sobel"] = ms

    # ── 7. Refine and merge ──
    closed, ms = _timed(refine_and_merge, combined, interior, kernels)
    timings["7_refine_merge"] = ms

    # ── 8. Process channel (contours + filtres géométriques) ──
    candidates, ms = _timed(process_channel, closed ,small, mask_white, h_small, params, kernels)
    timings["8_process_channel"] = ms

    # ── 9. Remap scale ──
    t = time.perf_counter()
    plates = []
    for box in candidates:
        plates.append({
            "x": int(box.x * scale),
            "y": int(box.y * scale),
            "w": int(box.w * scale),
            "h": int(box.h * scale),
            "score": box.confidence,
        })
    timings["9_remap_scale"] = (time.perf_counter() - t) * 1000

    # ── Total ──
    timings["TOTAL"] = sum(v for k, v in timings.items() if k != "TOTAL")

    return timings, plates, {
        "small": small,
        "gray": gray,
        "sat_mask": sat_mask,
        "mask_white": mask_white,
        "white_clean": white_clean,
        "combined": combined,
        "closed": closed,
        "candidates": candidates,
    }


# ═══════════════════════════════════════════════════
# Affichage
# ═══════════════════════════════════════════════════

def print_report(timings, plates, n_run=None):
    header = "BENCH PIPELINE"
    if n_run is not None:
        header += f" (moyenne sur {n_run} runs)"
    print(f"\n{'═' * 60}")
    print(f"  {header}")
    print(f"{'═' * 60}")
    print(f"  {'Étape':<30} {'ms':>10}")
    print(f"  {'─' * 45}")
    for key, val in timings.items():
        marker = " ◄" if key == "TOTAL" else ""
        print(f"  {key:<30} {val:>9.3f}{marker}")
    print(f"  {'─' * 45}")
    print(f"  Plaques détectées: {len(plates)}")
    for i, p in enumerate(plates):
        print(f"    [{i}] x={p['x']} y={p['y']} "
              f"w={p['w']} h={p['h']} score={p['score']}")
    print(f"{'═' * 60}\n")


def show_debug(artifacts):
    """Affiche les images intermédiaires (optionnel, --show)."""

    def to_bgr(img):
        if img is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    row1 = np.hstack([
        to_bgr(artifacts["gray"]),
        to_bgr(artifacts["sat_mask"]),
        to_bgr(artifacts["mask_white"]),
    ])
    row2 = np.hstack([
        to_bgr(artifacts["white_clean"]),
        to_bgr(artifacts["combined"]),
        to_bgr(artifacts["closed"]),
    ])
    row3 = np.hstack([
        to_bgr(artifacts["interior_v1"]),
        to_bgr(artifacts["interior_v2"]),
        to_bgr(artifacts["small"]),
    ])

    mosaic = np.vstack([row1, row2, row3])
    h, w = mosaic.shape[:2]
    if w > 1920:
        ratio = 1920 / w
        mosaic = cv2.resize(mosaic, (1920, int(h * ratio)))

    cv2.imshow("bench_pipeline debug", mosaic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Bench detect pipeline")
    parser.add_argument("image", help="Chemin vers une frame PNG/JPG")
    parser.add_argument("--n", type=int, default=10, help="Nombre de runs")
    parser.add_argument("--scale", type=float, default=None,
                        help="Scale override (sinon lit detect.slow.scale)")
    parser.add_argument("--fast-scale", type=float, default=None,
                        help="Bench aussi en fast scale")
    parser.add_argument("--show", action="store_true",
                        help="Affiche les artefacts intermédiaires")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"ERREUR: impossible de lire {args.image}")
        sys.exit(1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(f"Frame: {frame.shape[1]}x{frame.shape[0]}")

    scales = []
    if args.scale:
        scales.append(("custom", args.scale))
    else:
        scales.append(("slow", cfg.get("detect.slow.scale", 2.0)))
    if args.fast_scale:
        scales.append(("fast", args.fast_scale))

    for label, scale in scales:
        print(f"\n{'='*60}")
        print(f"  {label.upper()} — scale={scale}  ({int(frame.shape[1]/scale)}x{int(frame.shape[0]/scale)})")
        print(f"  {args.n} runs")
        print(f"{'='*60}")

        # Warmup
        bench_once(frame_rgb, scale)

        # Collect
        all_timings = []
        last_plates = []
        last_artifacts = {}
        for i in range(args.n):
            timings, plates, artifacts = bench_once(frame_rgb, scale)
            all_timings.append(timings)
            last_plates = plates
            last_artifacts = artifacts

        # Aggregate
        keys = list(all_timings[0].keys())
        print(f"\n{'Étape':<28} {'Moy (ms)':>10} {'Min':>10} {'Max':>10} {'%':>7}")
        print("-" * 68)

        total_avg = np.mean([t["TOTAL"] for t in all_timings])

        for k in keys:
            vals = [t[k] for t in all_timings]
            avg = np.mean(vals)
            mn = np.min(vals)
            mx = np.max(vals)
            if k in ("raw_box_count", "plates_found"):
                print(f"{k:<28} {avg:>10.1f} {mn:>10.0f} {mx:>10.0f}")
            elif k == "TOTAL":
                print("-" * 68)
                print(f"{'TOTAL':<28} {avg:>10.2f} {mn:>10.2f} {mx:>10.2f} {'100%':>7}")
            else:
                pct = (avg / total_avg * 100) if total_avg > 0 else 0
                print(f"{k:<28} {avg:>10.2f} {mn:>10.2f} {mx:>10.2f} {pct:>6.1f}%")

        print(f"\n  → Throughput: {1000/total_avg:.1f} FPS pipeline seul")
        print(f"  → Plaques détectées: {len(last_plates)}")
        for i, p in enumerate(last_plates):
            print(f"    [{i}] x={p['x']} y={p['y']} w={p['w']} h={p['h']} score={p['score']}")

        if args.show:
            show_debug(last_artifacts)


if __name__ == "__main__":
    main()
