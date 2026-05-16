# main.py — v13 (migration Mask dataclass)
import logging
from config import cfg
from bench import bench

def setup_logging():
    level_str = cfg.get("debug.log_level", "WARNING")
    level = getattr(logging, level_str.upper(), logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(name)s | %(levelname)s | %(message)s"
    )
setup_logging()

import sys
import time
from datetime import datetime
import numpy as np
import pyvirtualcam
from capture.config   import CaptureConfig
from capture.selector import SourceSelector
from capture.base     import CaptureSourceNotFound
from threads                 import CaptureThread, DetectThread, FastTrackThread, SendThread

from core.mask_manager       import draw_debug, pad_rect
from core.blur               import apply_blur
from core.mask               import MaskState
from tracker.tracker         import Tracker
from tracker.models          import TrackerConfig , Detection
log = logging.getLogger("main")

# ── PARAMÈTRES ──
SCREEN_WIDTH  = cfg.get("screen.width")
SCREEN_HEIGHT = cfg.get("screen.height")
CAPTURE_FPS   = cfg.get("screen.capture_fps")
VCAM_FPS      = cfg.get("screen.vcam_fps")
cfg.start_watcher()

# ═══════════════════════════════════════════════════════
#  RÉSOLUTION SOURCE CAPTURE                            ← AJOUT Phase 6
# ═══════════════════════════════════════════════════════
try:
    capture_cfg = CaptureConfig()
    _source      = SourceSelector.resolve(capture_cfg)
except CaptureSourceNotFound:
    log.error(
        "\n[ERREUR] Aucune source de capture disponible.\n"
        "\n"
        "Solutions :\n"
        "  1. Lancez Rocket League en mode Borderless Window\n"
        "  2. OU lancez OBS avec une source \"Game Capture\"\n"
        "     et activez la Virtual Camera\n"
        "\n"
        "Puis relancez le script."
    )
    sys.exit(1)

# ═══════════════════════════════════════════════════════
#  LANCEMENT
# ═══════════════════════════════════════════════════════
capturer = CaptureThread(source=_source, target_fps=CAPTURE_FPS)
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
        fps_timer             = time.perf_counter()

        while True:
            tracker.maybe_reload()
            if fast_tracker is not None:
                fast_tracker.maybe_reload()
            now = time.perf_counter()

            # ── 1. Capture (NON BLOQUANT) ──
            with bench.timer("main_capture_wait_ms"):
                frame, frame_ts = capturer.get_frame()
            if frame is None:
                time.sleep(0.001)
                continue

            # ── 2. Nouvelle frame → distribuer aux threads ──
            frame_id = capturer.get_frame_id()
            bench.gauge("main_frame_id", frame_id)
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
            with bench.timer("main_slow_poll_ms"):
                new_plates, detect_ts, current_version, detected_frame_ts = detector.get_result()
            slow_updated = current_version > last_detect_version
            if slow_updated:
                last_detect_version = current_version

            if slow_updated and new_plates:
                with bench.timer("main_match_ms"):
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
                with bench.timer("main_fast_poll_ms"):
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
            with bench.timer("main_predict_ms"):
                confirmed_masks = tracker.tick(now, updated_uids)

            # ── 5. Blur / debug draw ──
            blur_zones = [
                (int(m.rect[0]), int(m.rect[1]), int(m.rect[2]), int(m.rect[3]))
                for m in confirmed_masks
            ]

            buf = sender.borrow()
            np.copyto(buf, frame)

            with bench.timer("main_blur_ms"):
                apply_blur(buf, blur_zones)
                if debug_draw:
                    draw_debug(buf, confirmed_masks)

            # ── 6. Envoi ──
            with bench.timer("main_send_ms"):
                sender.publish()

            # ── 7. Stats ──
            bench.count("main_frames_total")
            bench.gauge("main_masks_total", len(confirmed_masks))
            bench.push_frame()
            # ── 8. FPS print toutes les 2s ──
            elapsed = time.perf_counter() - fps_timer
            if elapsed >= 2.0:
                fps  = bench.rate("main_frames_total", window_s=elapsed)
                mode     = "DEBUG" if debug_draw else "PROD"
                fast_tag = "+FAST" if fast_enabled else ""

                # Stats tracker — lues depuis bench (gauges posées dans tracker.tick)
                n_confirmed = bench.read_gauge("registry_confirmed") or 0
                n_pending   = bench.read_gauge("registry_pending")   or 0
                n_lost      = bench.read_gauge("registry_lost")      or 0

                # Stats motion — bench.last = valeur instantanée la plus récente
                # (approximation court terme, remplacé par summary_window à l'étape 2)
                stale_last = bench.last("staleness_slow_ms")
                if stale_last is not None:
                    motion_tag = f"staleness_slow last={stale_last:.1f}ms"
                else:
                    motion_tag = "staleness_slow n/a"

                ts_str = datetime.now().strftime("%H:%M:%S")

                log.info(
                    f"[{ts_str}] ⚡ {fps:.1f} FPS | "
                    f"masks C={n_confirmed} P={n_pending} L={n_lost} | "
                    f"{motion_tag} | "
                    f"{mode} {fast_tag}"
                )

                fps_timer = time.perf_counter()

    except KeyboardInterrupt:
        log.info("\n🛑 Arrêt propre")

    finally:
        sender.stop()
        if fast_enabled:
            fast_tracker.stop()
        detector.stop()
        capturer.stop()
        bench.shutdown()