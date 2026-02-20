# main.py
import time

import numpy as np
import pyvirtualcam

from capture import capture_screen, start, stop, SCREEN_WIDTH, SCREEN_HEIGHT
from capture import CAPTURE_FPS, VCAM_FPS
from capture import get_stats as capture_stats, reset_stats as capture_reset
from detect_thread import DetectThread
from blur import apply_blur
from blur import get_stats as blur_stats, reset_stats as blur_reset

# ─────────────────────────────────────────
# LANCEMENT
# ─────────────────────────────────────────
start()
detector = DetectThread()
detector.start()

fps_timer = time.time()
frame_count = 0

rgb_buffer = np.empty((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)

# Stats main loop
_main_stats = {
    "send_ms":      0.0,
    "loop_ms":      0.0,
    "total_frames": 0,
}

def print_all_stats():
    n = max(_main_stats["total_frames"], 1)
    cs = capture_stats()
    ds = detector.get_stats()
    bs = blur_stats()

    print("\n" + "=" * 55)
    print("        BENCHMARK PIPELINE v3 (FUSED BLUR+CVT)")
    print("=" * 55)

    print(f"\n  📷 CAPTURE (DXCam @ {CAPTURE_FPS}fps)")
    for k, v in cs.items():
        print(f"    {k:22s} : {v}")

    print(f"\n  🔍 DETECT (thread)")
    for k, v in ds.items():
        print(f"    {k:22s} : {v}")

    print(f"\n  🌀 BLUR + CVT (fusionnés)")
    for k, v in bs.items():
        print(f"    {k:22s} : {v}")

    print(f"\n  🎬 MAIN LOOP (vcam @ {VCAM_FPS}fps déclaré)")
    print(f"    {'send_avg_ms':22s} : {round(_main_stats['send_ms'] / n, 2)}")
    print(f"    {'loop_avg_ms':22s} : {round(_main_stats['loop_ms'] / n, 2)}")
    print(f"    {'total_frames':22s} : {_main_stats['total_frames']}")

    # ── Gain vs v1 ──
    old_loop = 35.81
    new_loop = round(_main_stats['loop_ms'] / n, 2)
    saved = round(old_loop - new_loop, 2)
    old_fps = round(1000 / old_loop, 1)
    new_fps = round(1000 / max(new_loop, 0.01), 1)

    print(f"\n  📉 GAIN vs v1")
    print(f"    {'v1 loop_avg':22s} : {old_loop} ms ({old_fps} FPS)")
    print(f"    {'v3 loop_avg':22s} : {new_loop} ms ({new_fps} FPS)")
    print(f"    {'économisé':22s} : {saved} ms/frame")
    print(f"    {'accélération':22s} : x{round(old_loop / max(new_loop, 0.01), 2)}")

    print("=" * 55)

with pyvirtualcam.Camera(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, fps=VCAM_FPS) as vcam:
    print(f"✅ Caméra virtuelle prête → {vcam.device}")
    print("📸 En cours... (Ctrl+C pour arrêter)")

    try:
        plates = []
        frame_id = 0
        SKIP = 2

        capture_reset()
        detector.reset_stats()
        blur_reset()

        while True:
            t_loop = time.perf_counter()

            # ── 1. Capture ──
            frame = capture_screen()
            if frame is None:
                continue

            # ── 2. Donner frame au thread (non bloquant) ──
            detector.give_frame(frame)

            # ── 3. Récupérer dernières zones connues ──
            plates = detector.get_zones()

            # ── 4. Blur + conversion RGB (fusionnés) ──
            if frame_id % SKIP == 0:
                frame_rgb = apply_blur(frame, plates, rgb_buffer=rgb_buffer)

            # ── 5. Envoi vers OBS ──
            t_send = time.perf_counter()
            vcam.send(frame_rgb)
            _main_stats["send_ms"] += (time.perf_counter() - t_send) * 1000

            _main_stats["loop_ms"] += (time.perf_counter() - t_loop) * 1000
            _main_stats["total_frames"] += 1
            frame_id += 1

            # ── 6. FPS counter ──
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 2.0:
                fps = frame_count / elapsed
                skipped = (SKIP - 1) / SKIP * 100
                print(f"⚡ {fps:.1f} FPS | {len(plates)} plaque(s) | skip {skipped:.0f}%")
                frame_count = 0
                fps_timer = time.time()

    except KeyboardInterrupt:
        print("\n🛑 Arrêt propre")
        print_all_stats()

    finally:
        detector.stop()
        stop()
