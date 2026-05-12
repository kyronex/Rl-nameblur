# threads/capture_thread.py
from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple

import numpy as np
from capture.base import CaptureSource

log = logging.getLogger("capture_thread")

class CaptureThread:
    """Capture en continu, la main loop prend la dernière frame sans bloquer."""

    def __init__(self, source: CaptureSource, target_fps: int = 120) -> None:
        self._source: CaptureSource = source
        self._target_fps: int = target_fps
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0
        self._frame_lock = threading.Lock()
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._frame_id: int = 0

    def start(self) -> None:
        self._source.start(target_fps=self._target_fps)
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        log.info(
            f"[CaptureThread] Démarré @ {self._target_fps} fps cible "
            f"(source={self._source.name})"
        )

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._source.stop()
        log.info("[CaptureThread] Arrêté")

    def get_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """Récupère la dernière frame (non bloquant)."""
        with self._frame_lock:
            return self._latest_frame, self._latest_ts

    def get_frame_id(self) -> int:
        """Retourne l'ID de la dernière frame capturée."""
        with self._frame_lock:
            return self._frame_id

    def _worker(self) -> None:
        _diag_count = 0
        _diag_t0 = time.perf_counter()
        source_name = self._source.name

        while self._running:
            frame = self._source.grab()
            if frame is None:
                time.sleep(0.001)
                continue

            _diag_count += 1
            _diag_elapsed = time.perf_counter() - _diag_t0
            if _diag_elapsed >= 5.0:
                log.info(
                    f"[CaptureThread] FPS réel {source_name} : "
                    f"{_diag_count / _diag_elapsed:.1f}"
                )
                _diag_count = 0
                _diag_t0 = time.perf_counter()

            frame = frame.copy()
            ts = time.perf_counter()
            with self._frame_lock:
                self._latest_frame = frame
                self._latest_ts = ts
                self._frame_id += 1