# bench_capture.py
import dxcam
import time

cam = dxcam.create(output_color="BGR")
cam.start(target_fps=144)

count = 0
t0 = time.time()

while time.time() - t0 < 5.0:
    frame = cam.get_latest_frame()
    if frame is not None:
        count += 1

cam.stop()
elapsed = time.time() - t0
print(f"Frames capturées : {count}")
print(f"Durée : {elapsed:.1f}s")
print(f"FPS réel DXCam : {count / elapsed:.1f}")
