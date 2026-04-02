# detect_thread.py — v2
import threading
import time
from detect import detect_plates

class DetectThread:
    """Slow detect — full frame, scale lent."""

    def __init__(self):
        self._latest_frame = None
        self._latest_frame_ts = 0.0
        self._latest_zones = []
        self._latest_zones_ts = 0.0
        self._latest_frame_ts_detected = 0.0
        self._frame_lock = threading.Lock()
        self._zones_lock = threading.Lock()
        self._running = False
        self._thread = None
        self._detect_count = 0

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

    def give_frame(self, frame, ts):
        with self._frame_lock:
            self._latest_frame = frame.copy()
            self._latest_frame_ts = ts

    def get_result(self):
        with self._zones_lock:
            return self._latest_zones.copy(), self._latest_zones_ts, self._detect_count, self._latest_frame_ts_detected

    def _worker(self):
        last_processed_ts = None
        while self._running:
            with self._frame_lock:
                frame = self._latest_frame
                frame_ts = self._latest_frame_ts

            if frame is None:
                time.sleep(0.001)
                continue

            if last_processed_ts == frame_ts:
                time.sleep(0.001)
                continue

            last_processed_ts = frame_ts
            zones = detect_plates(frame)

            with self._zones_lock:
                self._latest_zones = zones
                self._latest_zones_ts = time.perf_counter()
                self._latest_frame_ts_detected = frame_ts
                self._detect_count += 1
