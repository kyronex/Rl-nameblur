# main.py — v13 (migration Mask dataclass)
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

from threads                 import CaptureThread, DetectThread, FastTrackThread, SendThread
from bench.bench             import bench
from bench.csv_bench         import csv_open, csv_write_frame, csv_write_agg,csv_write_mask, csv_flush, csv_close
from core.mask_manager       import draw_debug, pad_rect, compute_mask_age
from core.blur               import apply_blur
from tracker.tracker import Tracker
from tracker.models import TrackerConfig , Detection
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

tracker = Tracker(TrackerConfig())

fast_enabled = cfg.get("detect.fast.enabled", True)
fast_tracker = None
if fast_enabled:
    fast_tracker = FastTrackThread()
    fast_tracker.start()

with pyvirtualcam.Camera(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, fps=VCAM_FPS) as vcam:
    print(f"✅ Caméra virtuelle prête → {vcam.device}")
    debug_draw = cfg.get("debug.overlay.enabled", False)

    if fast_enabled:
        print("⚡ FAST TRACKING ACTIVÉ")
    print("📸 En cours... (Ctrl+C pour arrêter)")

    sender = SendThread(vcam, SCREEN_WIDTH, SCREEN_HEIGHT)
    sender.start()

    fps_timer   = time.perf_counter()
    frame_count = 0
    csv_open()

    csv_agg_interval = cfg.get("debug.csv.agg_interval", 2.0)
    last_agg_time    = time.perf_counter()

    # ── DEBUG state (delta minimal) ──
    _dbg_prev_all = 0
    _dbg_prev_confirmed = 0

    try:
        active_masks       = []
        last_detect_version = 0
        last_fast_version   = 0
        last_frame_id       = 0
        last_csv_frame      = -1

        while True:
            tracker.maybe_reload()
            if fast_tracker is not None:
                fast_tracker.maybe_reload()
            now = time.perf_counter()

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
                    views = [m.to_fast_view() for m in active_masks]
                    fast_tracker.give_frame_and_views(frame, views, frame_ts)

            updated_uids = set()
            row_slow_updated = row_fast_updated = row_predicted = 0
            row_detect_age = row_fast_age = 0.0

            # ── 3. Slow detect ──
            with bench.timer("slow_poll"):
                new_plates, detect_ts, current_version, detected_frame_ts = detector.get_result()
                slow_updated = current_version > last_detect_version
                if slow_updated:
                    last_detect_version = current_version
                    row_slow_updated = 1
                    _read_ts = time.perf_counter()
                    row_detect_age = (max(0.0, (_read_ts - detected_frame_ts) * 1000)
                                      if detected_frame_ts else 0.0)

                    print(f"[SLOW] v={current_version} plates={len(new_plates) if new_plates else 0}")

            with bench.timer("match"):
                if slow_updated and new_plates:
                    dets = [Detection(
                        rect=pad_rect(*box.rect, SCREEN_WIDTH, SCREEN_HEIGHT),
                        source="slow",
                        confidence=box.confidence,
                        template=box.template,
                        scores=box.scores,
                    ) for box in new_plates]
                    updated_uids |= tracker.apply_detections(frame, dets, detected_frame_ts, "slow")

                    print(f"  └─ apply slow: dets={len(dets)} updated={len(updated_uids)} "f"all={len(tracker.all_masks())}")

            # ── 3b. Fast track ──
            with bench.timer("fast_poll"):
                if fast_enabled and not slow_updated:
                    fast_version, fast_results, fast_ts = fast_tracker.get_results()
                    if fast_version > last_fast_version:
                        last_fast_version = fast_version
                        row_fast_updated = 1
                        row_fast_age = (now - fast_ts) * 1000 if fast_ts else 0.0
                        dets = [Detection(
                            rect=pad_rect(*new_rect, SCREEN_WIDTH, SCREEN_HEIGHT),
                            source="fast",
                        ) for _uid, new_rect, _score in fast_results if new_rect is not None]
                        if dets:
                            updated_uids |= tracker.apply_detections(frame, dets, fast_ts, "fast")

                            print(f"[FAST] v={fast_version} dets={len(dets)} " f"updated={len(updated_uids)} all={len(tracker.all_masks())}")

            # ── 4. Tick (predict + TTL + purge) ──
            with bench.timer("predict"):
                confirmed = tracker.tick(now, updated_uids)
                row_predicted = 1 if (len(tracker.all_masks()) - len(updated_uids)) > 0 else 0

            _all_n = len(tracker.all_masks())
            _conf_n = len(confirmed)
            if _all_n != _dbg_prev_all or _conf_n != _dbg_prev_confirmed:
                print(f"[TICK] all {_dbg_prev_all}→{_all_n} | confirmed {_dbg_prev_confirmed}→{_conf_n}")
                _dbg_prev_all = _all_n
                _dbg_prev_confirmed = _conf_n

            active_masks = confirmed  # pour fast_tracker.give_frame_and_masks

            # ── 6. Blur / debug draw ──
            blur_zones = [
                (int(m.rect[0]), int(m.rect[1]),
                 int(m.rect[2]), int(m.rect[3]))
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

            # ── 8.  ages ──
            mask_age_avg = compute_mask_age(active_masks, now)

            bench.count("frames")
            bench.count("masks_total", len(active_masks))

            loop_ms = round((time.perf_counter() - now) * 1000, 3)
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
                    "loop_ms":           loop_ms,
                })

                for m in active_masks:
                    s = m.scores if isinstance(m.scores, dict) else {}
                    csv_write_mask({
                        "timestamp":       round(now, 6),
                        "frame_id":        frame_id,
                        "uid":             m.uid,
                        "x":               int(m.rect[0]),
                        "y":               int(m.rect[1]),
                        "w":               int(m.rect[2]),
                        "h":               int(m.rect[3]),
                        "last_x":          int(m.last_detected_rect[0]),
                        "last_y":          int(m.last_detected_rect[1]),
                        "last_w":          int(m.last_detected_rect[2]),
                        "last_h":          int(m.last_detected_rect[3]),
                        "last_ts":         m.last_detected_ts,
                        "last_slow_ts":    m.last_slow_ts,
                        "last_source":     m.last_source,
                        "state":           m.state,
                        "frames_matched":  m.frames_matched,
                        "frames_missing":  m.frames_missing,
                        "confidence":      round(m.confidence, 4),
                        "hash_history": "|".join(str(h) for h in m.hash_history),
                        "ttl":             round(m.ttl, 3),
                        "fast_miss_count": m.fast_miss_count,
                        "vx":              round(m.vx, 3),
                        "vy":              round(m.vy, 3),
                        "vw":              round(m.vw, 3),
                        "vh":              round(m.vh, 3),
                        # ── détail confidence ──
                        "score":               round(s.get("score", 0.0), 4),
                        "score_brut":          round(s.get("score_brut", 0.0), 4),
                        "fp_penalty":          round(s.get("fp_penalty", 0.0), 4),
                        "transition_density":  round(s.get("transition_density", 0.0), 4),
                        "s_td":                round(s.get("s_td", 0.0), 4),
                        "density_raw":         round(s.get("density_raw", 0.0), 4),
                        "s_dens":              round(s.get("s_dens", 0.0), 4),
                        "cc_raw":              round(s.get("cc_raw", 0.0), 4),
                        "s_cc":                round(s.get("s_cc", 0.0), 4),
                        "s_hreg":              round(s.get("s_hreg", 0.0), 4),
                        "row_fill":            round(s.get("row_fill", 0.0), 4),
                        "tiers_active":        round(s.get("tiers_active", 0.0), 4),
                        "vproj":               round(s.get("vproj", 0.0), 4),
                        "s_vp":                round(s.get("s_vp", 0.0), 4),
                        "s_pf":                round(s.get("s_pf", 0.0), 4),
                        "bg_score":            round(s.get("bg_score", 0.0), 4),
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
