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
from bench.csv_bench         import csv_open, csv_write_frame, csv_write_agg,csv_write_mask,csv_write_fast, csv_flush, csv_close
from core.mask_manager       import draw_debug, pad_rect
from core.blur               import apply_blur
from core.mask import MaskState
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
    log.info(f"✅ Caméra virtuelle prête → {vcam.device}")
    debug_draw = cfg.get("debug.overlay.enabled", False)

    if fast_enabled:
        log.info("⚡ FAST TRACKING ACTIVÉ")
    log.info("📸 En cours... (Ctrl+C pour arrêter)")

    sender = SendThread(vcam, SCREEN_WIDTH, SCREEN_HEIGHT)
    sender.start()


    try:
        confirmed_masks       = []
        last_detect_version   = 0
        last_fast_version     = 0
        last_frame_id         = 0
        frame_count           = 0
        fps_timer             = time.perf_counter()

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
                if fast_enabled :
                    snapshot = [m for m in tracker.all_masks() if m.state == MaskState.CONFIRMED]
                    if snapshot:
                        views = [m.to_fast_view() for m in snapshot]
                        fast_tracker.give_frame_and_views(frame, views, frame_ts)

            updated_uids = set()

            # ── 3. Slow detect ──
            with bench.timer("slow_poll"):
                new_plates, detect_ts, current_version, detected_frame_ts = detector.get_result()
            slow_updated = current_version > last_detect_version
            if slow_updated:
                last_detect_version = current_version

            if slow_updated and new_plates:
                with bench.timer("match"):
                    dets = [Detection(
                        rect=pad_rect(*box.rect, SCREEN_WIDTH, SCREEN_HEIGHT),
                        source="slow",
                        confidence=box.confidence,
                        template=box.template,
                        scores=box.scores,
                    ) for box in new_plates]
                    matched, created = tracker.apply_detections(frame, dets, detected_frame_ts, "slow")
                    updated_uids |= matched | created

            # ── 3b. Fast track ──
            if fast_enabled and not slow_updated:
                with bench.timer("fast_poll"):
                    fast_version, fast_results, fast_ts = fast_tracker.get_results()
                if fast_version > last_fast_version:
                    last_fast_version = fast_version
                    uid_to_rect = {
                        uid: pad_rect(*new_rect, SCREEN_WIDTH, SCREEN_HEIGHT)
                        for uid, new_rect, _score in fast_results
                        if new_rect is not None
                    }
                    if uid_to_rect:
                        matched = tracker.apply_fast_direct(frame, uid_to_rect, fast_ts)
                        updated_uids |= matched

            # ── 4. Tick (predict + TTL + purge) ──
            with bench.timer("predict"):
                confirmed_masks = tracker.tick(now, updated_uids)

            # ── 5. Blur / debug draw ──
            blur_zones = [
                (int(m.rect[0]), int(m.rect[1]), int(m.rect[2]), int(m.rect[3]))
                for m in confirmed_masks
            ]

            buf = sender.borrow()
            np.copyto(buf, frame)

            with bench.timer("blur"):
                apply_blur(buf, blur_zones)
                if debug_draw:
                    draw_debug(buf, confirmed_masks)

            # ── 6. Envoi ──
            with bench.timer("send"):
                sender.publish()

            # ── 7. Stats ──
            bench.count("frames")
            bench.count("masks_total", len(confirmed_masks))

            # ── 9. FPS print toutes les 2s ──
            frame_count += 1
            elapsed = time.perf_counter() - fps_timer
            if elapsed >= 2.0:
                fps      = frame_count / elapsed
                mode     = "DEBUG" if debug_draw else "PROD"
                fast_tag = "+FAST" if fast_enabled else ""
                log.info(f"⚡ {fps:.1f} FPS | {len(confirmed_masks)} masque(s) | {mode} {fast_tag}")
                bench.print_summary()

    except KeyboardInterrupt:
        log.info("\n🛑 Arrêt propre")

    finally:
        csv_close()
        sender.stop()
        if fast_enabled:
            fast_tracker.stop()
        detector.stop()
        capturer.stop()