# main.py —v12 (main allégé)
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

import time
import numpy as np
import pyvirtualcam

from capture_thread   import CaptureThread
from detect_thread    import DetectThread
from fast_track_thread import FastTrackThread
from send_thread      import SendThread
from blur             import apply_blur
from bench            import bench
from csv_bench        import csv_open, csv_write_frame, csv_write_agg, csv_flush, csv_close
from mask_manager     import match_and_update,update_mask,predict_masks,compute_jitter,compute_mask_age,pad_rect,draw_debug,kill_fast_miss, increment_fast_miss

log = logging.getLogger("main")

# ── PARAMÈTRES ──
SCREEN_WIDTH  = cfg.get("screen.width")
SCREEN_HEIGHT = cfg.get("screen.height")
CAPTURE_FPS   = cfg.get("screen.capture_fps")
VCAM_FPS      = cfg.get("screen.vcam_fps")
cfg.start_watcher()


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
    predict    = cfg.get("predict.active", True)

    if fast_enabled:
        print("⚡ FAST TRACKING ACTIVÉ")
    print("📸 En cours... (Ctrl+C pour arrêter)")

    sender = SendThread(vcam, SCREEN_WIDTH, SCREEN_HEIGHT)
    sender.start()

    fps_timer   = time.perf_counter()
    frame_count = 0
    csv_open()

    csv_agg_interval = cfg.get("debug.csv_agg_interval", 2.0)
    last_agg_time    = time.perf_counter()

    try:
        active_masks       = []
        last_detect_version = 0
        last_fast_version   = 0
        last_frame_id       = 0
        last_csv_frame      = -1

        while True:
            t_loop_start = time.perf_counter()
            now = time.perf_counter()

            # ── snapshot masques avant pour jitter ──
            rects_before = {m['uid']: m['rect'] for m in active_masks}

            # ── 1. Capture (NON BLOQUANT) ──
            with bench.timer("capture_wait"):
                frame, frame_ts = capturer.get_frame()
            if frame is None:
                time.sleep(0.001)
                continue

            # ── 2. Nouvelle frame → distribuer aux threads ──
            frame_id = capturer.get_frame_id()
            if frame_id > last_frame_id:
                last_frame_id = frame_id
                detector.give_frame(frame, frame_ts)
                if fast_enabled and active_masks:
                    fast_tracker.give_frame_and_masks(frame, active_masks, frame_ts)

            updated_uids    = set()
            row_slow_updated = 0
            row_fast_updated = 0
            row_predicted    = 0
            row_detect_age   = 0.0
            row_fast_age     = 0.0

            # ── 3. Slow detect ──
            with bench.timer("slow_poll"):
                new_plates, detect_ts, current_version, detected_frame_ts = detector.get_result()
                slow_updated = False

                if current_version > last_detect_version:
                    slow_updated         = True
                    last_detect_version  = current_version
                    row_slow_updated     = 1

                    _read_ts = time.perf_counter()
                    row_detect_age = (max(0.0, (_read_ts - detected_frame_ts) * 1000)if detected_frame_ts else 0.0)

                    padded = [
                        pad_rect(*box.rect, SCREEN_WIDTH, SCREEN_HEIGHT)
                        for box in new_plates
                    ]


            with bench.timer("match"):
                if slow_updated and new_plates:
                    uids = match_and_update(active_masks, padded, detect_ts, source="slow" , now=now)

                    for i, box in enumerate(new_plates):
                        if uids[i] is not None:
                            for m in active_masks:
                                if m['uid'] == uids[i]:
                                    m['confidence'] = box.confidence
                                    m['template']   = box.template
                                    break
                            updated_uids.add(uids[i])

                    increment_fast_miss(active_masks, updated_uids)
                    active_masks = kill_fast_miss(active_masks, now)

            # ── 3b. Fast track ──
            with bench.timer("fast_poll"):
                if fast_enabled and not slow_updated:
                    fast_version, fast_results, fast_ts = fast_tracker.get_results()
                    if fast_version > last_fast_version:
                        last_fast_version = fast_version
                        row_fast_updated = 1
                        row_fast_age = (now - fast_ts) * 1000 if fast_ts else 0.0

                        found_uids = set()
                        for mask_uid, new_rect, score in fast_results:
                            if new_rect is not None:
                                found_uids.add(mask_uid)
                                for m in active_masks:
                                    if m['uid'] == mask_uid:
                                        padded = pad_rect(*new_rect, SCREEN_WIDTH, SCREEN_HEIGHT)
                                        update_mask(m, padded, fast_ts, source="fast")
                                        updated_uids.add(mask_uid)
                                        break

                        increment_fast_miss(active_masks, updated_uids)
                        active_masks = kill_fast_miss(active_masks, now)

            # ── 4. Prédiction ──
            with bench.timer("predict"):
                if predict:
                    n_predicted = predict_masks(active_masks, updated_uids, now,SCREEN_WIDTH, SCREEN_HEIGHT)
                    row_predicted = 1 if n_predicted > 0 else 0

            # ── 5. Cap max masques ──
            max_masks = cfg.get("masks.max_masks")
            if len(active_masks) > max_masks:
                active_masks.sort(key=lambda m: m['ttl'], reverse=True)
                active_masks = active_masks[:max_masks]

            # ── 6. Blur / debug draw ──
            blur_zones = [
                (int(m['rect'][0]), int(m['rect'][1]),
                 int(m['rect'][2]), int(m['rect'][3]))
                for m in active_masks
            ]

            buf = sender.borrow()
            np.copyto(buf, frame)

            with bench.timer("blur"):
                apply_blur(buf, blur_zones)
                if debug_draw:
                    draw_debug(buf, active_masks)

            # ── 7. Envoi ──
            with bench.timer("send"):
                sender.publish()

            # ── 8. Jitter + ages ──
            jitter_center_avg, jitter_corners_avg, masks_created, masks_killed = compute_jitter(active_masks, rects_before)
            mask_age_avg = compute_mask_age(active_masks, now)

            bench.count("frames")
            bench.count("masks_total", len(active_masks))

            loop_ms = round((time.perf_counter() - t_loop_start) * 1000, 3)
            # ── CSV ──
            if frame_id != last_csv_frame:
                last_csv_frame = frame_id

                def _safe_last(name):
                    v = bench.last(name)
                    return round(v, 3) if v is not None else None

                csv_write_frame({
                    "timestamp":         round(now, 6),
                    "frame_id":          frame_id,
                    "capture_wait_ms":   _safe_last("capture_wait"),
                    "slow_poll_ms":      _safe_last("slow_poll"),
                    "match_ms":          _safe_last("match"),
                    "fast_poll_ms":      _safe_last("fast_poll"),
                    "predict_ms":        _safe_last("predict"),
                    "blur_ms":           _safe_last("blur"),
                    "send_ms":           _safe_last("send"),
                    "detect_age_ms":     round(row_detect_age, 2),
                    "fast_age_ms":       round(row_fast_age, 2),
                    "mask_age_avg_ms":   round(mask_age_avg, 2),
                    "slow_updated":      row_slow_updated,
                    "fast_updated":      row_fast_updated,
                    "predicted":         row_predicted,
                    "mask_count":        len(active_masks),
                    "jitter_center_px":  round(jitter_center_avg, 2),
                    "jitter_corners_px": round(jitter_corners_avg, 2),
                    "masks_created":     masks_created,
                    "masks_killed":      masks_killed,
                    "loop_ms":           loop_ms,
                })

            # ── 9. FPS print toutes les 2s ──
            frame_count += 1
            elapsed = time.perf_counter() - fps_timer
            if elapsed >= 2.0:
                fps      = frame_count / elapsed
                mode     = "DEBUG" if debug_draw else "PROD"
                fast_tag = "+FAST" if fast_enabled else ""
                print(f"⚡ {fps:.1f} FPS | {len(active_masks)} masque(s) | {mode} {fast_tag}")
                bench.print_summary()

                if now - last_agg_time >= csv_agg_interval:
                    csv_write_agg(bench.flat_row())
                    last_agg_time = now

                bench.reset()
                frame_count = 0
                fps_timer   = time.perf_counter()
                csv_flush()

    except KeyboardInterrupt:
        print("\n🛑 Arrêt propre")

    finally:
        csv_close()
        sender.stop()
        if fast_enabled:
            fast_tracker.stop()
        detector.stop()
        capturer.stop()
