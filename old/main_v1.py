# main_v1.py
import time

import cv2
import numpy as np
import pyvirtualcam

from capture import capture_screen, start, stop, SCREEN_WIDTH, SCREEN_HEIGHT, TARGET_FPS
from capture import get_stats as capture_stats, reset_stats as capture_reset
from detect import detect_plates
from detect import get_stats as detect_stats, reset_stats as detect_reset
from blur import apply_blur
from blur import get_stats as blur_stats, reset_stats as blur_reset

# ─────────────────────────────────────────
# LANCEMENT
# ─────────────────────────────────────────
start()

FRAME_TIME = 1.0 / TARGET_FPS
fps_timer = time.time()
frame_count = 0

rgb_buffer = np.empty((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)

# Stats main loop
_main_stats = {
    "send_ms":    0.0,
    "cvt_ms":     0.0,
    "loop_ms":    0.0,
    "total_frames": 0,
}

def print_all_stats():
    n = max(_main_stats["total_frames"], 1)
    cs = capture_stats()
    ds = detect_stats()
    bs = blur_stats()

    print("\n" + "=" * 55)
    print("        BENCHMARK PIPELINE COMPLET")
    print("=" * 55)

    print(f"\n  📷 CAPTURE")
    for k, v in cs.items():
        print(f"    {k:22s} : {v}")

    print(f"\n  🔍 DETECT")
    for k, v in ds.items():
        print(f"    {k:22s} : {v}")

    print(f"\n  🌀 BLUR")
    for k, v in bs.items():
        print(f"    {k:22s} : {v}")

    print(f"\n  🎬 MAIN LOOP")
    print(f"    {'cvt_avg_ms':22s} : {round(_main_stats['cvt_ms'] / n, 2)}")
    print(f"    {'send_avg_ms':22s} : {round(_main_stats['send_ms'] / n, 2)}")
    print(f"    {'loop_avg_ms':22s} : {round(_main_stats['loop_ms'] / n, 2)}")
    print(f"    {'total_frames':22s} : {_main_stats['total_frames']}")

    # ── Gain vs ancienne version ──
    old_loop = 39.86
    new_loop = round(_main_stats['loop_ms'] / n, 2)
    saved = round(old_loop - new_loop, 2)
    print(f"\n  📉 GAIN")
    print(f"    {'ancien loop_avg':22s} : {old_loop} ms")
    print(f"    {'nouveau loop_avg':22s} : {new_loop} ms")
    print(f"    {'économisé':22s} : {saved} ms")

    print("=" * 55)

with pyvirtualcam.Camera(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, fps=TARGET_FPS) as vcam:
    print(f"✅ Caméra virtuelle prête → {vcam.device}")
    print("📸 En cours... (Ctrl+C pour arrêter)")

    try:
        frame_id = 0
        plates = []
        skip = 10

        # Reset tous les compteurs
        capture_reset()
        detect_reset()
        blur_reset()

        while True:
            t_loop = time.perf_counter()

            # ── 1. Capture ──
            frame = capture_screen()
            if frame is None:
                continue

            # ── 2. Détection ──
            if frame_id % skip == 0:
                plates = detect_plates(frame)
                if len(plates) >= 5:
                    skip = 2
                elif len(plates) >= 2:
                    skip = 3
                else:
                    skip = 5

            frame_id += 1

            # ── 3. Flou ──
            frame = apply_blur(frame, plates)

            # ── 4. Envoi vers OBS ──
            t_cvt = time.perf_counter()
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=rgb_buffer)
            _main_stats["cvt_ms"] += (time.perf_counter() - t_cvt) * 1000

            t_send = time.perf_counter()
            vcam.send(rgb_buffer)
            _main_stats["send_ms"] += (time.perf_counter() - t_send) * 1000

            _main_stats["loop_ms"] += (time.perf_counter() - t_loop) * 1000
            _main_stats["total_frames"] += 1

            # ── 5. FPS counter ──
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 2.0:
                fps = frame_count / elapsed
                print(f"⚡ {fps:.1f} FPS | {len(plates)} plaque(s) | skip={skip}")
                frame_count = 0
                fps_timer = time.time()

    except KeyboardInterrupt:
        print("\n🛑 Arrêt propre")
        print_all_stats()

    finally:
        stop()
