# detect_thread.py — Slow detect (full frame)
import threading
import time
from detect import detect_plates


class DetectThread:
    """Slow detect — full frame, scale lent."""

    def __init__(self):
        self._latest_frame = None
        self._latest_zones = []
        self._frame_lock = threading.Lock()
        self._zones_lock = threading.Lock()
        self._running = False
        self._thread = None

        self._total_detect_ms = 0.0
        self._detect_count = 0
        self._detect_lock = threading.Lock()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print("[DetectThread/Slow] Démarré")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        print("[DetectThread/Slow] Arrêté")

    def give_frame(self, frame):
        copy = frame.copy()
        with self._frame_lock:
            self._latest_frame = copy

    def get_zones(self):
        with self._zones_lock:
            return self._latest_zones.copy()

    def get_detect_count(self):
        with self._detect_lock:
            return self._detect_count

    def get_stats(self):
        with self._detect_lock:
            n = max(self._detect_count, 1)
            return {
                "slow_detect_avg_ms": round(self._total_detect_ms / n, 2),
                "slow_detect_count":  self._detect_count,
                "slow_detect_total_ms": round(self._total_detect_ms, 2),
            }

    def reset_stats(self):
        with self._detect_lock:
            self._total_detect_ms = 0.0
            self._detect_count = 0

    def _worker(self):
        while self._running:
            with self._frame_lock:
                frame = self._latest_frame

            if frame is None:
                time.sleep(0.001)
                continue

            t0 = time.perf_counter()
            zones = detect_plates(frame)
            dt = (time.perf_counter() - t0) * 1000

            with self._zones_lock:
                self._latest_zones = zones

            with self._detect_lock:
                self._detect_count += 1
                self._total_detect_ms += dt
