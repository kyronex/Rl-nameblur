# detect_tools.py
import cv2
import time
import logging
log = logging.getLogger("detect_v3")

def _needs_split(w, h, params):
    return w >= params["SPLIT_MIN_WIDTH"] and h >= params["SPLIT_MIN_HEIGHT"]


def split_by_erosion(bbox, mask, kernels, params, stats):
    """Sépare un blob large par érosion + composantes connexes."""
    x, y, w, h = bbox
    roi = mask[y:y+h, x:x+w]

    eroded = cv2.erode(roi, kernels["erode"], iterations=params["ERODE_ITERATIONS"])
    num_labels, labels, comp_stats, _ = cv2.connectedComponentsWithStats(eroded)

    parts = []
    for i in range(1, num_labels):
        cx, cy, cw, ch, carea = comp_stats[i]
        if carea < params["MIN_AREA"]:
            continue
        ratio = cw / max(ch, 1)
        if params["MIN_RATIO"] <= ratio <= params["MAX_RATIO"] and cw >= params["MIN_WIDTH"]:
            parts.append((x + cx, y + cy, cw, ch))

    if len(parts) >= 2:
        stats["splits_h"] += 1
        return parts

    return [bbox]


def process_channel(mask_color, white_dilated, kernels, params, stats):
    """Pipeline v2 — morpho complète + split."""
    closed  = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, kernels["close_h"])
    closed  = cv2.morphologyEx(closed,     cv2.MORPH_CLOSE, kernels["close_v"])
    closed  = cv2.morphologyEx(closed,     cv2.MORPH_OPEN,  kernels["open_noise"])

    trimmed = cv2.bitwise_and(closed, white_dilated)
    trimmed = cv2.morphologyEx(trimmed, cv2.MORPH_CLOSE, kernels["close_h2"])

    contours, _ = cv2.findContours(trimmed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stats["contours_raw"] += len(contours)

    t_loop = time.perf_counter()
    results = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        if area < params["MIN_AREA"]:
            stats["rej_area"] += 1
            continue
        if w < params["MIN_WIDTH"] or w > params["MAX_WIDTH"]:
            stats["rej_width"] += 1
            continue
        if h < params["MIN_HEIGHT"] or h > params["MAX_HEIGHT"]:
            stats["rej_height"] += 1
            continue

        ratio = w / max(h, 1)
        if ratio < params["MIN_RATIO"] or ratio > params["MAX_RATIO"]:
            stats["rej_ratio"] += 1
            continue

        fill = cv2.contourArea(cnt) / max(area, 1)
        if fill < params["MIN_FILL"]:
            stats["rej_fill"] += 1
            continue

        stats["candidates"] += 1
        bbox = (x, y, w, h)

        if _needs_split(w, h, params):
            parts = split_by_erosion(bbox, trimmed, kernels, params, stats)
        else:
            parts = [bbox]

        results.extend(parts)

    stats["contour_loop_ms"] += (time.perf_counter() - t_loop) * 1000
    return results


def process_channel_v3(masked, kernels, params, stats):

    debug = True
    # ── A. Pixels actifs avant traitement ──
    if debug:
        print(f"[v3] AND pixels actifs     : {cv2.countNonZero(masked)}")

    # ── B. Fermeture horizontale ──
    t0 = time.perf_counter()
    closed = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, kernels["close_h"])
    stats["v3_close_ms"] += (time.perf_counter() - t0) * 1000

    if debug:
        print(f"[v3] après CLOSE pixels    : {cv2.countNonZero(closed)}")

    # ── C. Ouverture bruit ──
    t0 = time.perf_counter()
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernels["open_noise"])
    stats["v3_open_ms"] += (time.perf_counter() - t0) * 1000

    if debug:
        print(f"[v3] après OPEN pixels     : {cv2.countNonZero(opened)}")

    # ── D. Contours ──
    t0 = time.perf_counter()
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stats["v3_contour_ms"] += (time.perf_counter() - t0) * 1000
    stats["v3_contours_raw"] += len(contours)

    if debug:
        print(f"[v3] contours bruts        : {len(contours)}")
        for cnt in contours:
            x, y, w, h  = cv2.boundingRect(cnt)
            area_contour = cv2.contourArea(cnt)
            area_bbox    = w * h
            print(f"[v3]   contour : bbox={w}×{h} area_contour={area_contour:.0f} area_bbox={area_bbox} min={params['MIN_AREA_V3']} max={params['MAX_AREA_V3']}")

    # ── E. Filtre ──
    min_width  = params.get("MIN_WIDTH_V3",  40)
    min_height = params.get("MIN_HEIGHT_V3",  8)
    min_ratio  = params.get("MIN_RATIO_V3",  3.5)
    max_ratio  = params.get("MAX_RATIO_V3", 18.0)
    min_area   = params.get("MIN_AREA_V3")
    max_area   = params.get("MAX_AREA_V3")
    min_fill   = params.get("MIN_FILL_V3", 0.15)
    max_fill   = params.get("MAX_FILL_V3", 0.85)


    plates = []
    t_loop = time.perf_counter()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / max(h, 1)
        area  = w * h

        if w < min_width:
            stats["v3_rej_width"] += 1
            continue
        if h < min_height:
            stats["v3_rej_height"] += 1
            continue
        if ratio < min_ratio or ratio > max_ratio:
            stats["v3_rej_ratio"] += 1
            continue
        if area < min_area or area > max_area:
            stats["v3_rej_area_min"] += 1
            continue

        roi_pixels = opened[y:y+h, x:x+w]
        fill = cv2.countNonZero(roi_pixels) / area
        if fill < min_fill or fill > max_fill:
            stats["v3_rej_fill"] += 1
            if debug:
                print(f"[FILTRE_FILL] bbox={w}×{h} fill={fill:.2f} "
                      f"min={min_fill} max={max_fill} → rejeté")
            continue

        if debug and w > 50:
            print(
                f"[FILTRE_REEL] bbox={w}×{h} ratio={ratio:.1f} area={area} "
                f"rej_width={w < min_width} rej_height={h < min_height} "
                f"rej_ratio={ratio < min_ratio or ratio > max_ratio} "
                f"rej_area={area < min_area or area > max_area}"
            )

        plates.append((x, y, w, h))
        stats["v3_plates_found"] += 1

    stats["v3_contour_loop_ms"] += (time.perf_counter() - t_loop) * 1000

    return plates

