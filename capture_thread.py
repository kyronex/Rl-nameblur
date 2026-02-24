# capture_thread.py
import threading
import time
import dxcam

class CaptureThread:
    """Capture en continu, la main loop prend la dernière frame sans bloquer."""

    def __init__(self, target_fps=120):
        self._target_fps = target_fps
        self._camera = None
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._thread = None
        self._frame_id = 0

        # Stats
        self._grab_count = 0
        self._grab_total_ms = 0.0
        self._none_count = 0
        self._stats_lock = threading.Lock()

    def start(self):
        self._camera = dxcam.create(output_color="RGB")
        self._camera.start(target_fps=self._target_fps)
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print(f"[CaptureThread] Démarré @ {self._target_fps} fps cible")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._camera:
            self._camera.stop()
        print("[CaptureThread] Arrêté")

    def get_frame(self):
        """Récupère la dernière frame (non bloquant)."""
        with self._frame_lock:
            return self._latest_frame

    def get_frame_id(self):
        """Retourne l'ID de la dernière frame capturée."""
        with self._frame_lock:
            return self._frame_id

    def get_stats(self):
        with self._stats_lock:
            n = max(self._grab_count, 1)
            return {
                "grab_avg_ms": round(self._grab_total_ms / n, 2),
                "grab_count":  self._grab_count,
                "none_count":  self._none_count,
            }

    def reset_stats(self):
        with self._stats_lock:
            self._grab_count = 0
            self._grab_total_ms = 0.0
            self._none_count = 0

    def _worker(self):
        while self._running:
            t0 = time.perf_counter()
            frame = self._camera.get_latest_frame()
            dt = (time.perf_counter() - t0) * 1000

            if frame is None:
                with self._stats_lock:
                    self._none_count += 1
                continue

            frame = frame.copy()

            with self._stats_lock:
                self._grab_count += 1
                self._grab_total_ms += dt

            with self._frame_lock:
                self._latest_frame = frame
                self._frame_id += 1