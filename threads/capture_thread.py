# threads/capture_thread.py
import threading
import time
import dxcam
import logging

log = logging.getLogger("capture_thread")

class CaptureThread:
    """Capture en continu, la main loop prend la dernière frame sans bloquer."""

    def __init__(self, target_fps=120):
        self._target_fps = target_fps
        self._camera = None
        self._latest_frame = None
        self._latest_ts = 0.0
        self._frame_lock = threading.Lock()
        self._running = False
        self._thread = None
        self._frame_id = 0

    def start(self):
        self._camera = dxcam.create(output_color="RGB")
        self._camera.start(target_fps=self._target_fps)
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        log.info(f"[CaptureThread] Démarré @ {self._target_fps} fps cible")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._camera:
            self._camera.stop()
        log.info("[CaptureThread] Arrêté")

    def get_frame(self):
        """Récupère la dernière frame (non bloquant)."""
        with self._frame_lock:
            return self._latest_frame, self._latest_ts

    def get_frame_id(self):
        """Retourne l'ID de la dernière frame capturée."""
        with self._frame_lock:
            return self._frame_id

    def _worker(self):
        while self._running:
            frame = self._camera.get_latest_frame()
            if frame is None:
                continue
            frame = frame.copy()
            ts = time.perf_counter()
            with self._frame_lock:
                self._latest_frame = frame
                self._latest_ts = ts
                self._frame_id += 1