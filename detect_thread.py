# detect_thread.py
import threading
import time
from detect import detect_plates

class DetectThread:
    def __init__(self):
        self._latest_frame = None
        self._latest_zones = []
        self._frame_lock = threading.Lock()
        self._zones_lock = threading.Lock()
        self._running = False
        self._thread = None

        # Benchmark
        self._total_detect_ms = 0.0
        self._detect_count = 0
        self._detect_lock = threading.Lock()

    # ──────────── Contrôle ────────────

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print("[DetectThread] Démarré")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        print("[DetectThread] Arrêté")

    # ──────────── Interface ────────────

    def give_frame(self, frame):
        """Donne une frame (non bloquant, écrase la précédente)."""
        copy = frame.copy()
        with self._frame_lock:
            self._latest_frame = copy

    def get_zones(self):
        """Récupère les dernières zones détectées."""
        with self._zones_lock:
            return self._latest_zones.copy()

    def get_detect_count(self):
        """Nombre de détections terminées (sert de version)."""
        with self._detect_lock:
            return self._detect_count

    # ──────────── Benchmark ────────────

    def get_stats(self):
        with self._detect_lock:
            n = max(self._detect_count, 1)
            return {
                "detect_avg_ms":    round(self._total_detect_ms / n, 2),
                "detect_count":     self._detect_count,
                "detect_total_ms":  round(self._total_detect_ms, 2),
            }

    def reset_stats(self):
        with self._detect_lock:
            self._total_detect_ms = 0.0
            self._detect_count = 0

    # ──────────── Worker ────────────

    def _worker(self):
        while self._running:
            with self._frame_lock:
                frame = self._latest_frame
                self._latest_frame = None

            if frame is None:
                time.sleep(0.001)
                continue

            t0 = time.perf_counter()
            zones = detect_plates(frame)
            dt = (time.perf_counter() - t0) * 1000

            with self._zones_lock:
                self._latest_zones = zones

            with self._detect_lock:
                self._total_detect_ms += dt
                self._detect_count += 1
