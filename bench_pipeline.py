# bench_pipeline.py
"""
Benchmark complet du pipeline detect.
Usage :
    python bench_pipeline.py frame.png          # une image
    python bench_pipeline.py frame.png --n 20   # moyenne sur 20 runs
"""

import cv2
import numpy as np
import sys
import time
import argparse
import logging

logging.basicConfig(level=logging.WARNING)

from config import cfg
from detect import _build_params
from detect_tools_mask import (
    saturation_variance_mask,
    compute_white_mask,
    compute_sobel_interiors,
    refine_and_merge
)
from detect_tools_boxes import (
    extract_raw_boxes,
    adjust_resolve,
    split_wide_boxes,
    validate_text,
    validate_background,
    merge_nearby_horizontal,
    expand_plates,
    filter_geometry,
    process_channel,
)
from detect_tools import (
    write_circles ,
    write_rects ,
    get_color
)



def bench_once(frame, scale):
    """Exécute le pipeline étape par étape, retourne un dict de timings (ms)."""
    timings = {}

    # ── Resize ──
    t = time.perf_counter()
    h_orig, w_orig = frame.shape[:2]
    w_small = int(w_orig / scale)
    h_small = int(h_orig / scale)
    small = cv2.resize(frame, (w_small, h_small), interpolation=cv2.INTER_AREA)
    timings["resize"] = (time.perf_counter() - t) * 1000

    # ── Grayscale ──
    t = time.perf_counter()
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    timings["grayscale"] = (time.perf_counter() - t) * 1000

    # ── Build params + kernels ──
    t = time.perf_counter()
    colors, kernels, params, letter_connect_iter = _build_params(scale)
    letter_connect_iter = cfg.get("detect.morpho.white_dilate.iterations", 1)
    timings["build_params"] = (time.perf_counter() - t) * 1000

    # ── Saturation variance mask ──
    t = time.perf_counter()
    sat_mask = saturation_variance_mask(small,scale)
    timings["sat_variance_mask"] = (time.perf_counter() - t) * 1000

    # ── White mask ──
    t = time.perf_counter()
    mask_white, white_clean = compute_white_mask(gray, kernels, letter_connect_iter)
    timings["white_mask"] = (time.perf_counter() - t) * 1000

    # ── Combine sat + white ──
    t = time.perf_counter()
    combined = cv2.bitwise_and(white_clean, sat_mask)
    timings["combine"] = (time.perf_counter() - t) * 1000

    # ── Sobel interiors ──
    t = time.perf_counter()
    interior_v1, interior_v2 = compute_sobel_interiors(gray, combined, kernels)
    timings["sobel_interiors"] = (time.perf_counter() - t) * 1000

    # ── Refine and merge ──
    t = time.perf_counter()
    closed = refine_and_merge(combined, interior_v1, interior_v2, kernels)
    timings["refine_merge"] = (time.perf_counter() - t) * 1000

    # ── process_channel breakdown ──
    # extract_raw_boxes
    t = time.perf_counter()
    boxes = extract_raw_boxes(closed, params)
    timings["extract_raw_boxes"] = (time.perf_counter() - t) * 1000
    timings["raw_box_count"] = len(boxes)

    # adjust_resolve #1
    t = time.perf_counter()
    boxes_ar = adjust_resolve(boxes, mask_white, h_small, params)
    timings["adjust_resolve_1"] = (time.perf_counter() - t) * 1000

    # split_wide_boxes
    t = time.perf_counter()
    split = split_wide_boxes(boxes_ar, mask_white, params)
    timings["split_wide"] = (time.perf_counter() - t) * 1000

    # adjust_resolve #2
    t = time.perf_counter()
    split_ar = adjust_resolve(split, mask_white, h_small, params, resolve=False)
    timings["adjust_resolve_2"] = (time.perf_counter() - t) * 1000

    # validate_text
    t = time.perf_counter()
    validated_t = validate_text(split_ar, mask_white, params, kernels)
    timings["validate_text"] = (time.perf_counter() - t) * 1000

    # merge_nearby_horizontal
    t = time.perf_counter()
    merge = merge_nearby_horizontal(validated_t, params["max_gap_x"], params["max_gap_y"])
    timings["merge_horiz"] = (time.perf_counter() - t) * 1000

    # validate_background
    t = time.perf_counter()
    validated_b = validate_background(merge, mask_white, small, params)
    timings["validate_bg"] = (time.perf_counter() - t) * 1000

    # adjust_resolve #3
    t = time.perf_counter()
    validated_b_ar = adjust_resolve(validated_b, mask_white, h_small, params, resolve=False)
    timings["adjust_resolve_3"] = (time.perf_counter() - t) * 1000

    # expand_plates
    t = time.perf_counter()
    expanded = expand_plates(validated_b_ar, small)
    timings["expand"] = (time.perf_counter() - t) * 1000

    # filter_geometry
    t = time.perf_counter()
    plates = filter_geometry(expanded, closed, params)
    """
    screen = small.copy()
    write_rects(screen, validated_t, get_color("magenta"),2)
    write_rects(screen, plates, get_color("vert"),1)
    cv2.imshow("screen", screen)
    cv2.waitKey(0)
    """
    timings["filter_geometry"] = (time.perf_counter() - t) * 1000
    timings["plates_found"] = len(plates)

    # ── Total ──
    timings["TOTAL"] = sum(
        v for k, v in timings.items()
        if k not in ("raw_box_count", "plates_found", "TOTAL")
    )

    return timings


def main():
    parser = argparse.ArgumentParser(description="Bench detect pipeline")
    parser.add_argument("image", help="Chemin vers une frame PNG/JPG")
    parser.add_argument("--n", type=int, default=10, help="Nombre de runs")
    parser.add_argument("--scale", type=float, default=None,
                        help="Scale override (sinon lit detect.slow.scale)")
    parser.add_argument("--fast-scale", type=float, default=None,
                        help="Bench aussi en fast scale")
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
        for i in range(args.n):
            t = bench_once(frame_rgb, scale)
            all_timings.append(t)

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


if __name__ == "__main__":
    main()
