# send_thread.py
import threading
import time
import numpy as np

class SendThread:
    """Envoie les frames à la vcam dans un thread séparé — double buffer zéro copie."""

    def __init__(self, vcam, width, height):
        self._vcam = vcam
        self._running = False
        self._thread = None

        # ── Double buffer (swap, pas copie) ──
        shape = (height, width, 3)
        self._buf_a = np.zeros(shape, dtype=np.uint8)
        self._buf_b = np.zeros(shape, dtype=np.uint8)
        self._write_buf = self._buf_a   # main écrit ici
        self._send_buf  = self._buf_b   # worker envoie celui-ci
        self._has_frame = False
        self._lock = threading.Lock()

        # Benchmark
        self._total_send_ms = 0.0
        self._send_count = 0
        self._total_borrow_ms = 0.0
        self._borrow_count = 0
        self._stats_lock = threading.Lock()

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

    def borrow(self):
        #Retourne le buffer d'écriture pour que main écrive directement dedans.
        return self._write_buf

    def publish(self):
        """Signale que le write buffer est prêt → swap avec le send buffer."""
        t0 = time.perf_counter()
        with self._lock:
            self._write_buf, self._send_buf = self._send_buf, self._write_buf
            self._has_frame = True
        dt = (time.perf_counter() - t0) * 1000

        with self._stats_lock:
            self._total_borrow_ms += dt
            self._borrow_count += 1

    # ──────────── Benchmark ────────────

    def get_stats(self):
        with self._stats_lock:
            n_send = max(self._send_count, 1)
            n_pub  = max(self._borrow_count, 1)
            return {
                "send_avg_ms":    round(self._total_send_ms / n_send, 2),
                "send_count":     self._send_count,
                "send_total_ms":  round(self._total_send_ms, 2),
                "publish_avg_ms": round(self._total_borrow_ms / n_pub, 2),
                "publish_count":  self._borrow_count,
            }

    def reset_stats(self):
        with self._stats_lock:
            self._total_send_ms = 0.0
            self._send_count = 0
            self._total_borrow_ms = 0.0
            self._borrow_count = 0

    # ──────────── Worker ────────────

    def _worker(self):
        while self._running:
            with self._lock:
                if not self._has_frame:
                    frame_to_send = None
                else:
                    frame_to_send = self._send_buf
                    self._has_frame = False

            if frame_to_send is None:
                time.sleep(0.001)
                continue

            t0 = time.perf_counter()
            self._vcam.send(frame_to_send)
            dt = (time.perf_counter() - t0) * 1000

            with self._stats_lock:
                self._total_send_ms += dt
                self._send_count += 1
