# send_thread.py
import threading
import time
import numpy as np


class SendThread:
    """Envoie les frames à la vcam dans un thread séparé."""

    def __init__(self, vcam):
        self._vcam = vcam
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._thread = None

        # Benchmark
        self._total_send_ms = 0.0
        self._send_count = 0
        self._send_lock = threading.Lock()

    # ──────────── Contrôle ────────────

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print("[SendThread] Démarré")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print("[SendThread] Arrêté")

    # ──────────── Interface ────────────

    def give_frame(self, frame_rgb: np.ndarray):
        """Donne une frame à envoyer (non bloquant, écrase la précédente)."""
        with self._frame_lock:
            self._latest_frame = frame_rgb

    # ──────────── Benchmark ────────────

    def get_stats(self):
        with self._send_lock:
            n = max(self._send_count, 1)
            return {
                "send_avg_ms":   round(self._total_send_ms / n, 2),
                "send_count":    self._send_count,
                "send_total_ms": round(self._total_send_ms, 2),
            }

    def reset_stats(self):
        with self._send_lock:
            self._total_send_ms = 0.0
            self._send_count = 0

    # ──────────── Worker ────────────

    def _worker(self):
        while self._running:
            # Prendre la frame
            with self._frame_lock:
                frame = self._latest_frame
                self._latest_frame = None

            if frame is None:
                time.sleep(0.001)
                continue

            # Send + bench
            t0 = time.perf_counter()
            self._vcam.send(frame)
            dt = (time.perf_counter() - t0) * 1000

            # Mettre à jour bench
            with self._send_lock:
                self._total_send_ms += dt
                self._send_count += 1
