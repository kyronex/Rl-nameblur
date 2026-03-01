# detect_tools.py

import cv2
import time
import logging

log = logging.getLogger("detect_tools")

def process_channel(masked, kernels, params, stats):
    # ── A. Pixels actifs avant traitement ──
    t0 = time.perf_counter()
    pre_opened = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernels["pre_open_noise"])
    stats["pre_open_ms"] += (time.perf_counter() - t0) * 1000

    # ── B. Fermeture horizontale ──
    t0 = time.perf_counter()
    closed = cv2.morphologyEx(pre_opened, cv2.MORPH_CLOSE, kernels["close_h"])
    stats["close_ms"] += (time.perf_counter() - t0) * 1000

    # ── C. Ouverture bruit ──
    t0 = time.perf_counter()
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernels["open_noise"])
    stats["open_ms"] += (time.perf_counter() - t0) * 1000

    # ── D. Contours ──
    t0 = time.perf_counter()
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stats["contour_ms"] += (time.perf_counter() - t0) * 1000
    stats["contours_raw"] += len(contours)

    # ── E. Filtre géométrique ──
    t0 = time.perf_counter()
    plates = []

    # ── DEBUG pixels ──
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"AND pixels actifs     : {cv2.countNonZero(masked)}")
        log.debug(f"après PRE OPEN pixels    : {cv2.countNonZero(pre_opened)}")
        log.debug(f"après CLOSE pixels    : {cv2.countNonZero(closed)}")
        log.debug(f"après OPEN pixels     : {cv2.countNonZero(opened)}")
        log.debug(f"contours bruts        : {len(contours)}")

    """ cv2.imshow("masked", masked)
    cv2.imshow("pre_opened", pre_opened)
    cv2.imshow("closed", closed)
    cv2.imshow("opened", opened)
    cv2.waitKey(0) """

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / max(h, 1)
        area = w * h

        if w < params["min_width"]:
            log.debug(f"REJET width   | x={x} y={y} w={w} h={h} ratio={ratio:.2f} area={area} | min_width={params['min_width']}")
            stats["rej_width"] += 1
            continue
        if h < params["min_height"]:
            log.debug(f"REJET height  | x={x} y={y} w={w} h={h} ratio={ratio:.2f} area={area} | min_height={params['min_height']}")
            stats["rej_height"] += 1
            continue

        if ratio < params["min_ratio"] or ratio > params["max_ratio"]:
            log.debug(f"REJET ratio   | x={x} y={y} w={w} h={h} ratio={ratio:.2f} area={area} | range=[{params['min_ratio']:.1f}, {params['max_ratio']:.1f}]")
            stats["rej_ratio"] += 1
            continue

        if area < params["min_area"]:
            log.debug(f"REJET area<   | x={x} y={y} w={w} h={h} ratio={ratio:.2f} area={area} | min_area={params['min_area']}")
            stats["rej_area"] += 1
            continue
        if area > params["max_area"]:
            log.debug(f"REJET area>   | x={x} y={y} w={w} h={h} ratio={ratio:.2f} area={area} | max_area={params['max_area']}")
            stats["rej_area"] += 1
            continue

        fill = cv2.contourArea(cnt) / max(area, 1)
        if fill < params["min_fill"] or fill > params["max_fill"]:
            log.debug(f"REJET fill    | x={x} y={y} w={w} h={h} ratio={ratio:.2f} area={area} fill={fill:.2f} | range=[{params['min_fill']:.2f}, {params['max_fill']:.2f}]")
            stats["rej_fill"] += 1
            continue

        log.debug(f"ACCEPT        | x={x} y={y} w={w} h={h} ratio={ratio:.2f} area={area} fill={fill:.2f}")
        plates.append((x, y, w, h))

    stats["filter_loop_ms"] += (time.perf_counter() - t0) * 1000
    stats["plates_found"] += len(plates)

    log.debug(f"plates retenues       : {len(plates)}")

    return plates
