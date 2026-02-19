# main.py
import time
import cv2
import pyvirtualcam
from capture import capture_screen, start, stop, SCREEN_WIDTH, SCREEN_HEIGHT, TARGET_FPS
from detect import detect_plates
from blur import apply_blur

# ─────────────────────────────────────────
# LANCEMENT
# ─────────────────────────────────────────
start()

FRAME_TIME = 1.0 / TARGET_FPS
fps_timer = time.time()
frame_count = 0

with pyvirtualcam.Camera(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, fps=TARGET_FPS) as vcam:
    print(f"✅ Caméra virtuelle prête → {vcam.device}")
    print("📸 En cours... (Ctrl+C pour arrêter)")

    try:
        frame_id = 0
        plates = []
        skip = 10
        while True:
            loop_start = time.time()

            # ── 1. Capture ──
            frame = capture_screen()
            if frame is None:
                continue

            # ── 2. Détection ──
            if frame_id % skip == 0:
                plates = detect_plates(frame)
                if len(plates) >= 5:
                    skip = 1
                elif len(plates) >= 2:
                    skip = 3
                else:
                    skip = 10

            frame_id += 1

            # ── 3. Flou ──
            frame = apply_blur(frame, plates)


            # ── 4. Envoi vers OBS ──
            vcam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # ── 5. FPS counter ──
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 2.0:
                fps = frame_count / elapsed
                print(f"⚡ {fps:.1f} FPS | {len(plates)} plaque(s)")
                frame_count = 0
                fps_timer = time.time()

            # ── 6. Limiteur FPS ──
            sleep_time = FRAME_TIME - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n🛑 Arrêt propre")

    finally:
        stop()
