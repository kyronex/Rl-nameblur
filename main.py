# main.py — v10 (benchmark instrumentation)
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

import os
import time
import csv
import cv2
import numpy as np
import pyvirtualcam

from capture_thread import CaptureThread
from detect_thread import DetectThread
from fast_track_thread import FastTrackThread
from send_thread import SendThread
from detect_stats import get_stats as detect_stats
from detect_stats import reset_stats as detect_diag_reset
from blur import apply_blur
from blur import get_stats as blur_stats, reset_stats as blur_reset
log = logging.getLogger("main")

# ── PARAMÈTRES — lus depuis config.yaml ──
SCREEN_WIDTH  = cfg.get("screen.width")
SCREEN_HEIGHT = cfg.get("screen.height")
CAPTURE_FPS   = cfg.get("screen.capture_fps")
VCAM_FPS      = cfg.get("screen.vcam_fps")
cfg.start_watcher()

_next_mask_id = 0

# ═══════════════════════════════════════════════════════
#  CSV BENCHMARK
# ═══════════════════════════════════════════════════════

_csv_file = None
_csv_writer = None

CSV_HEADERS = [
    "timestamp",
    "frame_id",
    # B1 — latence par étape
    "loop_ms",
    "capture_wait_ms",
    "slow_poll_ms",
    "fast_poll_ms",
    "predict_ms",
    "blur_ms",
    "send_ms",
    # B2 — fraîcheur
    "detect_age_ms",
    "fast_age_ms",
    "mask_age_avg_ms",
    # B3 — compteurs instantanés
    "slow_updated",
    "fast_updated",
    "predicted",
    "mask_count",
    # B4 — stabilité
    "jitter_center_px",
    "jitter_corners_px",
    "masks_created",
    "masks_killed",
]

def csv_open():
    global _csv_file, _csv_writer
    os.makedirs("logs", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = f"logs/bench_{ts}.csv"
    _csv_file = open(path, "w", newline="")
    _csv_writer = csv.DictWriter(_csv_file, fieldnames=CSV_HEADERS)
    _csv_writer.writeheader()
    print(f"📊 CSV benchmark → {path}")

def csv_write(row: dict):
    if _csv_writer is not None:
        _csv_writer.writerow(row)

def csv_close():
    global _csv_file, _csv_writer
    if _csv_file is not None:
        _csv_file.flush()
        _csv_file.close()
        _csv_file = None
        _csv_writer = None

# ═══════════════════════════════════════════════════════
#  UTILITAIRES TRACKING
# ═══════════════════════════════════════════════════════

def ttl_color(ttl):
    if ttl >= 3:
        return tuple(cfg.get("debug.colors.fresh"))
    elif ttl >= 2:
        return tuple(cfg.get("debug.colors.persist"))
    else:
        return tuple(cfg.get("debug.colors.dying"))

def ttl_label(ttl, source="S"):
    return f"TTL={ttl} [{source}]"

def compute_iou(r1, r2):
    """IoU (Intersection over Union) entre deux rectangles (x,y,w,h)."""
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def center_distance(r1, r2):
    """Distance euclidienne entre les centres de deux rectangles (x,y,w,h)."""
    cx1 = r1[0] + r1[2] / 2
    cy1 = r1[1] + r1[3] / 2
    cx2 = r2[0] + r2[2] / 2
    cy2 = r2[1] + r2[3] / 2
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

def rect_corners(r):
    """Retourne les 4 coins (x,y) d'un rectangle (x,y,w,h)."""
    x, y, w, h = r
    return [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]

def corners_distance(r1, r2):
    """Distance moyenne entre les 4 coins de deux rectangles."""
    c1 = rect_corners(r1)
    c2 = rect_corners(r2)
    total = 0.0
    for (ax, ay), (bx, by) in zip(c1, c2):
        total += ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    return total / 4.0

def update_mask(mask, new_rect, detect_ts, source):
    smooth_alpha = cfg.get("masks.smooth_alpha", 1.0)
    dead_zone    = cfg.get("masks.dead_zone", 3)
    vel_dz       = cfg.get("masks.velocity_dead_zone", 5)

    nx, ny, nw, nh = new_rect
    ox, oy, ow, oh = mask['rect']

    if (abs(nx - ox) < dead_zone and abs(ny - oy) < dead_zone and abs(nw - ow) < dead_zone and abs(nh - oh) < dead_zone):
        if source == "slow":
            mask['ttl'] = cfg.get("masks.ttl_max")
            mask['fast_miss_count'] = 0
        mask['last_detected_ts'] = detect_ts
        mask['last_source'] = source
        return

    # ── Vélocité filtrée ──
    lx, ly, _, _ = mask['last_detected_rect']
    dt = detect_ts - mask['last_detected_ts']
    if dt > 0.001:
        mask['vx'] = 0.0 if abs(nx - lx) < vel_dz else (nx - lx) / dt
        mask['vy'] = 0.0 if abs(ny - ly) < vel_dz else (ny - ly) / dt

    # ── Smooth EMA ──
    a = smooth_alpha
    mask['rect'] = (
        ox + a * (nx - ox),
        oy + a * (ny - oy),
        ow + a * (nw - ow),
        oh + a * (nh - oh),
    )
    mask['last_detected_rect'] = (float(nx), float(ny), float(nw), float(nh))
    mask['last_detected_ts'] = detect_ts
    mask['last_source'] = source
    if source == "slow":
        mask['ttl'] = cfg.get("masks.ttl_max")
        mask['fast_miss_count'] = 0
    elif source == "fast":
        mask['fast_miss_count'] = 0

def match_or_add(active_masks, new_rect, detect_ts, source="slow"):
    global _next_mask_id
    best_idx = -1
    nx, ny, nw, nh = new_rect

    ttl_max     = cfg.get("masks.ttl_max")
    iou_thresh  = cfg.get("matching.iou_thresh")
    dist_thresh = cfg.get("matching.dist_thresh")
    max_masks   = cfg.get("masks.max_masks")
    mode        = cfg.get("matching.mode")

    if mode == "distance":
        best_dist = float('inf')
        for i, m in enumerate(active_masks):
            d = center_distance(m['rect'], new_rect)
            if d < best_dist:
                best_dist = d
                best_idx = i
        if best_dist > dist_thresh:
            best_idx = -1
    else:
        best_score = 0.0
        for i, m in enumerate(active_masks):
            score = compute_iou(m['rect'], new_rect)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_score < iou_thresh:
            best_idx = -1

    if best_idx >= 0:
        update_mask(active_masks[best_idx], new_rect, detect_ts, source)
        return active_masks[best_idx]['uid']

    if len(active_masks) >= max_masks:
        return None

    _next_mask_id += 1
    active_masks.append({
        'uid':                 _next_mask_id,
        'rect':                (float(nx), float(ny), float(nw), float(nh)),
        'ttl':                 ttl_max,
        'last_detected_rect':  (float(nx), float(ny), float(nw), float(nh)),
        'last_detected_ts':    detect_ts,
        'last_source':         source,
        'fast_miss_count':     0,
        'vx':                  0.0,
        'vy':                  0.0,
    })
    return _next_mask_id

def pad_rect(x, y, w, h, max_w, max_h):
    return (
        max(x, 0),
        max(y, 0),
        min(w, max_w - max(x, 0)),
        min(h, max_h - max(y, 0)),
    )

def draw_debug(frame, active_masks):
    for m in active_masks:
        x, y, w, h = (int(v) for v in m['rect'])
        ttl = m['ttl']
        color = ttl_color(ttl)
        thickness = 2 if ttl >= 3 else 1

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

        source = m.get('last_source', '?')[0].upper()
        label = ttl_label(ttl, source)
        label_y = max(y - 6, 14)
        cv2.putText(frame, label, (x, label_y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    total = len(active_masks)
    cv2.putText(frame, f"Masks: {total}", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return frame

# ═══════════════════════════════════════════════════════
#  STATS / BENCHMARK
# ═══════════════════════════════════════════════════════

_main_stats = {
    "loop_ms":       0.0,
    "blur_ms":       0.0,
    "total_frames":  0,
    "mask_peak":     0,
    "slow_updates":  0,
    "fast_updates":  0,
    "predict_count": 0,
}

# ── B1 accumulateurs pour print 2s ──
_interval_stats = {
    "capture_wait_ms": 0.0,
    "slow_poll_ms":    0.0,
    "fast_poll_ms":    0.0,
    "predict_ms":      0.0,
    "blur_ms":         0.0,
    "send_ms":         0.0,
    # B2
    "detect_age_ms":   0.0,
    "fast_age_ms":     0.0,
    "mask_age_ms":     0.0,
    # B3
    "slow_updates":    0,
    "fast_updates":    0,
    "predict_frames":  0,
    "frames":          0,
    # B4
    "jitter_center":   0.0,
    "jitter_corners":  0.0,
    "masks_created":   0,
    "masks_killed":    0,
}

def _reset_interval():
    for k in _interval_stats:
        _interval_stats[k] = 0.0 if isinstance(_interval_stats[k], float) else 0

# snapshot des rects du frame précédent pour jitter
_prev_rects = {}   # uid → rect

def print_all_stats():
    matching_mode = cfg.get("matching.mode")
    n = max(_main_stats["total_frames"], 1)
    cs = capturer.get_stats()
    ds = detector.get_stats()
    bs = blur_stats()
    ss = sender.get_stats()
    dd = detect_stats()

    print("\n" + "=" * 55)
    print("        BENCHMARK PIPELINE v10.0 (instrumented)")
    print("=" * 55)
    print(f"\n  📷 CAPTURE (thread @ {CAPTURE_FPS}fps cible)")
    for k, v in cs.items():
        print(f"    {k:22s} : {v}")

    print(f"\n  🔍 SLOW DETECT (thread)")
    for k, v in ds.items():
        print(f"    {k:22s} : {v}")

    if fast_enabled:
        fs = fast_tracker.get_stats()
        print(f"\n  ⚡ FAST TRACK (ROI)")
        for k, v in fs.items():
            print(f"    {k:22s} : {v}")

    print(f"\n  🌀 BLUR + CVT (fusionnés)")
    for k, v in bs.items():
        print(f"    {k:22s} : {v}")

    print(f"\n  📤 SEND (thread, vcam @ {VCAM_FPS}fps déclaré)")
    for k, v in ss.items():
        print(f"    {k:22s} : {v}")

    print(f"\n  🔬 DETECT DIAGNOSTIC")
    for k, v in dd.items():
        print(f"    {k:22s} : {v}")

    print(f"\n  🎬 MAIN LOOP")
    loop_avg = round(_main_stats['loop_ms'] / n, 2)
    blur_avg = round(_main_stats['blur_ms'] / n, 2)
    print(f"    {'loop_avg_ms':22s} : {loop_avg}")
    print(f"    {'blur_avg_ms':22s} : {blur_avg}")
    print(f"    {'total_frames':22s} : {_main_stats['total_frames']}")
    print(f"    {'mask_peak':22s} : {_main_stats['mask_peak']}")
    print(f"    {'slow_updates':22s} : {_main_stats['slow_updates']}")
    print(f"    {'fast_updates':22s} : {_main_stats['fast_updates']}")
    print(f"    {'predict_count':22s} : {_main_stats['predict_count']}")
    print(f"    {'ttl_max':22s} : {cfg.get('masks.ttl_max')}")
    print(f"    {'matching_mode':22s} : {matching_mode}")
    if matching_mode == "distance":
        print(f"    {'dist_thresh':22s} : {cfg.get('matching.dist_thresh')}")
    else:
        print(f"    {'iou_thresh':22s} : {cfg.get('matching.iou_thresh')}")

    old_loop = 32.66
    new_loop = loop_avg
    saved = round(old_loop - new_loop, 2)
    old_fps = round(1000 / old_loop, 1)
    new_fps = round(1000 / max(new_loop, 0.01), 1)

    print(f"\n  📉 GAIN vs v5.1")
    print(f"    {'v5.1 loop_avg':22s} : {old_loop} ms ({old_fps} FPS)")
    print(f"    {'v10 loop_avg':22s} : {new_loop} ms ({new_fps} FPS)")
    print(f"    {'économisé':22s} : {saved} ms/frame")
    print(f"    {'accélération':22s} : x{round(old_loop / max(new_loop, 0.01), 2)}")
    print("=" * 55)

# ═══════════════════════════════════════════════════════
#  LANCEMENT
# ═══════════════════════════════════════════════════════

capturer = CaptureThread(target_fps=CAPTURE_FPS)
capturer.start()

detector = DetectThread()
detector.start()

fast_enabled = cfg.get("detect.fast.enabled", True)
if fast_enabled:
    fast_tracker = FastTrackThread(SCREEN_WIDTH, SCREEN_HEIGHT)
    fast_tracker.start()

with pyvirtualcam.Camera(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, fps=VCAM_FPS) as vcam:
    print(f"✅ Caméra virtuelle prête → {vcam.device}")
    debug_draw = cfg.get("debug.draw")
    predict = cfg.get("predict.active", True)

    if fast_enabled:
        print("⚡ FAST TRACKING ACTIVÉ")
    print("📸 En cours... (Ctrl+C pour arrêter)")

    sender = SendThread(vcam, SCREEN_WIDTH, SCREEN_HEIGHT)
    sender.start()

    fps_timer = time.time()
    frame_count = 0

    csv_open()

    try:
        active_masks = []
        last_detect_version = 0
        last_fast_version = 0
        last_frame_id = 0

        capturer.reset_stats()
        detector.reset_stats()
        if fast_enabled:
            fast_tracker.reset_stats()
        detect_diag_reset()
        blur_reset()
        sender.reset_stats()
        _reset_interval()

        while True:
            t_loop = time.perf_counter()
            now = time.perf_counter()

            # ── snapshot masques avant pour jitter (B4) ──
            rects_before = {m['uid']: m['rect'] for m in active_masks}
            masks_before_count = len(active_masks)

            # ── 1. Capture (NON BLOQUANT) ──
            t0 = time.perf_counter()
            frame, frame_ts = capturer.get_frame()
            t_capture_wait = (time.perf_counter() - t0) * 1000
            if frame is None:
                time.sleep(0.001)
                continue

            # ── 2. Ne donner au detect QUE si frame nouvelle ──
            frame_id = capturer.get_frame_id()
            if frame_id > last_frame_id:
                last_frame_id = frame_id
                detector.give_frame(frame, frame_ts)
                if fast_enabled and active_masks:
                    fast_tracker.give_frame_and_masks(frame, active_masks, frame_ts)

            updated_uids = set()
            row_slow_updated = 0
            row_fast_updated = 0
            row_predicted = 0
            row_detect_age = 0.0
            row_fast_age = 0.0

            # ── 3. Slow : nouvelle détection disponible ? ──
            t0 = time.perf_counter()
            current_version = detector.get_detect_count()
            slow_updated = False
            if current_version > last_detect_version:
                slow_updated = True
                row_slow_updated = 1
                for m in active_masks:
                    m['ttl'] -= 1
                active_masks = [m for m in active_masks if m['ttl'] > 0]

                last_detect_version = current_version
                _main_stats["slow_updates"] += 1
                new_plates, detect_ts = detector.get_zones()
                row_detect_age = (now - detect_ts) * 1000 if detect_ts else 0.0
                for p in new_plates:
                    x, y, w, h = p
                    padded = pad_rect(x, y, w, h, SCREEN_WIDTH, SCREEN_HEIGHT)
                    uid = match_or_add(active_masks, padded, detect_ts, source="slow")
                    if uid is not None:
                        updated_uids.add(uid)
            t_slow_poll = (time.perf_counter() - t0) * 1000

            # ── 3b. Fast : résultats ROI disponibles ? ──
            t0 = time.perf_counter()
            if fast_enabled and not slow_updated:
                fast_version, fast_results, fast_ts = fast_tracker.get_results()
                if fast_version > last_fast_version:
                    last_fast_version = fast_version
                    _main_stats["fast_updates"] += 1
                    row_fast_updated = 1
                    row_fast_age = (now - fast_ts) * 1000 if fast_ts else 0.0

                    found_uids = set()

                    for mask_uid, new_rect in fast_results:
                        if new_rect is not None:
                            found_uids.add(mask_uid)
                            for m in active_masks:
                                if m['uid'] == mask_uid:
                                    padded = pad_rect(*new_rect, SCREEN_WIDTH, SCREEN_HEIGHT)
                                    update_mask(m, padded, fast_ts, source="fast")
                                    updated_uids.add(mask_uid)
                                    break

                    # ── Masks non trouvés par le fast ──
                    for m in active_masks:
                        if m['uid'] not in found_uids and m['last_source'] != "new":
                            m['fast_miss_count'] += 1

                 # ── Évaluation kill (à chaque cycle, pas seulement sur nouveau résultat) ──
                fast_miss_thresh = cfg.get("masks.fast_miss_threshold", 5)
                fast_miss_timeout = cfg.get("masks.fast_miss_timeout_ms", 300) / 1000.0
                for m in active_masks:
                    time_since = now - m['last_detected_ts']
                    if m['fast_miss_count'] >= fast_miss_thresh:
                        m['ttl'] -= 1
                        m['fast_miss_count'] = 0
                    elif time_since >= fast_miss_timeout:
                        m['ttl'] -= 1
                        m['fast_miss_count'] = 0
                        m['last_detected_ts'] = now
                active_masks = [m for m in active_masks if m['ttl'] > 0]
            t_fast_poll = (time.perf_counter() - t0) * 1000

            # ── 4. Predict les masques NON mis à jour (en secondes) ──
            t0 = time.perf_counter()
            if predict:
                for m in active_masks:
                    if m['uid'] not in updated_uids:
                        row_predicted = 1
                        dt = now - m['last_detected_ts']
                        dt_capped = min(dt, 0.10)
                        damping = max(0.0, 1.0 - dt * 2.0)
                        lx, ly, lw, lh = m['last_detected_rect']
                        w, h = m['rect'][2], m['rect'][3]
                        x = lx + m['vx'] * dt_capped * damping
                        y = ly + m['vy'] * dt_capped * damping
                        x = max(0.0, min(x, SCREEN_WIDTH - w))
                        y = max(0.0, min(y, SCREEN_HEIGHT - h))
                        m['rect'] = (x, y, w, h)
                        _main_stats["predict_count"] += 1
            t_predict = (time.perf_counter() - t0) * 1000

            # ── 5. Cap max masques ──
            if len(active_masks) > cfg.get("masks.max_masks"):
                active_masks.sort(key=lambda m: m['ttl'], reverse=True)
                active_masks = active_masks[:cfg.get("masks.max_masks")]

            # ── 6. Blur ou debug ──
            blur_zones = [
                (int(m['rect'][0]), int(m['rect'][1]),
                 int(m['rect'][2]), int(m['rect'][3]))
                for m in active_masks
            ]

            # ── 7. Envoi vers OBS (zéro copie) ──
            buf = sender.borrow()
            np.copyto(buf, frame)
            t_blur_start = time.perf_counter()

            if debug_draw:
                apply_blur(buf, blur_zones)
                draw_debug(buf, active_masks)
            else:
                apply_blur(buf, blur_zones)

            t_blur_end = time.perf_counter()
            t_send_start = time.perf_counter()
            sender.publish()
            t_send_end = time.perf_counter()

            t_blur_ms = (t_blur_end - t_blur_start) * 1000
            t_send_ms = (t_send_end - t_send_start) * 1000

            # ── 8. Jitter (B4) ──
            jitter_center_sum = 0.0
            jitter_corners_sum = 0.0
            jitter_n = 0
            masks_after_uids = {m['uid'] for m in active_masks}
            masks_created = 0
            masks_killed = 0

            for m in active_masks:
                uid = m['uid']
                if uid in rects_before:
                    jitter_center_sum += center_distance(rects_before[uid], m['rect'])
                    jitter_corners_sum += corners_distance(rects_before[uid], m['rect'])
                    jitter_n += 1
                else:
                    masks_created += 1

            for uid in rects_before:
                if uid not in masks_after_uids:
                    masks_killed += 1

            jitter_center_avg = (jitter_center_sum / jitter_n) if jitter_n > 0 else 0.0
            jitter_corners_avg = (jitter_corners_sum / jitter_n) if jitter_n > 0 else 0.0

            # ── B2 : mask_age_avg ──
            mask_age_avg = 0.0
            if active_masks:
                mask_age_avg = sum((now - m['last_detected_ts']) * 1000 for m in active_masks) / len(active_masks)

            # ── 8b. Stats loop ──
            loop_ms = (time.perf_counter() - t_loop) * 1000
            _main_stats["loop_ms"] += loop_ms
            _main_stats["blur_ms"] += t_blur_ms
            _main_stats["total_frames"] += 1
            _main_stats["mask_peak"] = max(
                _main_stats["mask_peak"], len(active_masks)
            )

            # ── accumulateurs intervalle 2s ──
            _interval_stats["capture_wait_ms"] += t_capture_wait
            _interval_stats["slow_poll_ms"]    += t_slow_poll
            _interval_stats["fast_poll_ms"]    += t_fast_poll
            _interval_stats["predict_ms"]      += t_predict
            _interval_stats["blur_ms"]         += t_blur_ms
            _interval_stats["send_ms"]         += t_send_ms
            _interval_stats["detect_age_ms"]   += row_detect_age
            _interval_stats["fast_age_ms"]     += row_fast_age
            _interval_stats["mask_age_ms"]     += mask_age_avg
            _interval_stats["slow_updates"]    += row_slow_updated
            _interval_stats["fast_updates"]    += row_fast_updated
            _interval_stats["predict_frames"]  += row_predicted
            _interval_stats["frames"]          += 1
            _interval_stats["jitter_center"]   += jitter_center_avg
            _interval_stats["jitter_corners"]  += jitter_corners_avg
            _interval_stats["masks_created"]   += masks_created
            _interval_stats["masks_killed"]    += masks_killed

            # ── CSV row ──
            csv_write({
                "timestamp":        round(now, 6),
                "frame_id":         frame_id,
                "loop_ms":          round(loop_ms, 3),
                "capture_wait_ms":  round(t_capture_wait, 3),
                "slow_poll_ms":     round(t_slow_poll, 3),
                "fast_poll_ms":     round(t_fast_poll, 3),
                "predict_ms":       round(t_predict, 3),
                "blur_ms":          round(t_blur_ms, 3),
                "send_ms":          round(t_send_ms, 3),
                "detect_age_ms":    round(row_detect_age, 2),
                "fast_age_ms":      round(row_fast_age, 2),
                "mask_age_avg_ms":  round(mask_age_avg, 2),
                "slow_updated":     row_slow_updated,
                "fast_updated":     row_fast_updated,
                "predicted":        row_predicted,
                "mask_count":       len(active_masks),
                "jitter_center_px": round(jitter_center_avg, 2),
                "jitter_corners_px":round(jitter_corners_avg, 2),
                "masks_created":    masks_created,
                "masks_killed":     masks_killed,
            })

            # ── 9. FPS counter + print enrichi ──
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 2.0:
                fps = frame_count / elapsed
                mode = "DEBUG" if debug_draw else "PROD"
                fast_tag = "+FAST" if fast_enabled else ""
                n = max(_interval_stats["frames"], 1)
                n_slow = max(_interval_stats["slow_updates"], 1)
                n_fast = max(_interval_stats["fast_updates"], 1)

                cap_avg   = round(_interval_stats["capture_wait_ms"] / n, 2)
                slow_avg  = round(_interval_stats["slow_poll_ms"] / n, 2)
                fast_avg  = round(_interval_stats["fast_poll_ms"] / n, 2)
                pred_avg  = round(_interval_stats["predict_ms"] / n, 2)
                blur_avg  = round(_interval_stats["blur_ms"] / n, 2)
                send_avg  = round(_interval_stats["send_ms"] / n, 2)
                loop_avg  = round((cap_avg + slow_avg + fast_avg + pred_avg + blur_avg + send_avg), 2)

                det_age   = round(_interval_stats["detect_age_ms"] / n_slow, 1)
                fst_age   = round(_interval_stats["fast_age_ms"] / n_fast, 1) if fast_enabled else 0.0
                msk_age   = round(_interval_stats["mask_age_ms"] / n, 1)

                s_up = int(_interval_stats["slow_updates"])
                f_up = int(_interval_stats["fast_updates"])
                p_fr = int(_interval_stats["predict_frames"])
                p_pct = round(100.0 * p_fr / n, 1)

                j_c = round(_interval_stats["jitter_center"] / n, 2)
                j_4 = round(_interval_stats["jitter_corners"] / n, 2)
                m_cr = int(_interval_stats["masks_created"])
                m_ki = int(_interval_stats["masks_killed"])

                print(f"⚡ {fps:.1f} FPS | {len(active_masks)} masque(s) | {mode} {fast_tag}")
                print(f"  ├─ loop: {loop_avg}ms (cap:{cap_avg} slow:{slow_avg} fast:{fast_avg} pred:{pred_avg} blur:{blur_avg} send:{send_avg})")
                print(f"  ├─ age: slow={det_age}ms fast={fst_age}ms mask_avg={msk_age}ms")
                print(f"  ├─ updates: slow={s_up} fast={f_up} predict={p_fr}/{int(n)} ({p_pct}%)")
                print(f"  └─ masks: jitter_c={j_c}px jitter_4={j_4}px created={m_cr} killed={m_ki}")

                frame_count = 0
                fps_timer = time.time()
                _reset_interval()

                # flush CSV toutes les 2s
                if _csv_file is not None:
                    _csv_file.flush()

    except KeyboardInterrupt:
        print("\n🛑 Arrêt propre")
        print_all_stats()

    finally:
        csv_close()
        sender.stop()
        if fast_enabled:
            fast_tracker.stop()
        detector.stop()
        capturer.stop()
