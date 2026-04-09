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
        with self._lock:
            self._write_buf, self._send_buf = self._send_buf, self._write_buf
            self._has_frame = True

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

            self._vcam.send(frame_to_send)