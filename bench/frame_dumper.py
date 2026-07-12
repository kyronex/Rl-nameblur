# bench/frame_dumper.py
from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path

import cv2

from config import cfg

from bench.lifecycle import LifecycleRecord   # dataclass, pas dict

log = logging.getLogger(__name__)


class FrameDumperWriter:
    """
    Dump JPEG snapshots of selected frames to disk.

    Thread-safe, non-blocking on the caller.  Three counters track
    the health of the pipeline (ring miss / queue drop / invalid fid).
    """

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------
    def __init__(self, capture_thread, session_id: int):
        self._capture = capture_thread
        self._session_id = session_id

        self._path = Path(cfg.get("debug.bench.frame_dumper.path", "logs/frames")).expanduser().resolve()
        self._q: queue.Queue = queue.Queue(maxsize=cfg.get("debug.bench.frame_dumper.queue_maxsize", 256))
        self._tail: int = cfg.get("debug.bench.frame_dumper.tail_frames", 0)
        self._jpeg_quality: int = cfg.get("debug.bench.frame_dumper.jpeg_quality", 75)

        self._seen: set[int] = set()           # déduplication locale
        self._count_written: int = 0

        # 3 compteurs contractuels
        self._ring_miss_count: int = 0
        self._queue_drop_count: int = 0
        self._skip_invalid_frame_id: int = 0

        self._thread: threading.Thread | None = None
        self._running = False

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self):
        self._path.mkdir(parents=True, exist_ok=True)
        self._running = True
        self._thread = threading.Thread(
            target=self._worker,
            name=f"FrameDumper-{self._session_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        if not self._running:
            return
        self._running = False
        try:
            self._q.put_nowait(None)           # poison-pill
        except queue.Full:
            pass                               # queue sature → le worker terminera de lui-même
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # ingestion : appelé sur le flux events (list[LifecycleRecord])
    # ------------------------------------------------------------------
    def on_events(self, events: list[LifecycleRecord]):
        log.info(f"FrameDumper: {len(events)} events")
        if not self._running:
            return
        for event in events:
            frame_id: int = event.get("frame_id", -1)
            log.info(f"frame_id: {frame_id} ")
            if frame_id == -1:
                self._skip_invalid_frame_id += 1
                continue
            # frame itself + tail range
            ids_to_dump: set[int] = {frame_id} | set(
                frame_id + k for k in range(1, self._tail + 1)
            )
            for fid in ids_to_dump:
                self._select(fid)

    def _select(self, frame_id: int):
        """Tentative unique ddump pour un frame_id donné."""
        if frame_id in self._seen:             # déduplication
            return
        self._seen.add(frame_id)

        frame = self._capture.get_ring_frame(frame_id)
        if frame is None:
            self._ring_miss_count += 1
            return

        try:
            self._q.put_nowait((frame_id, frame.copy()))
        except queue.Full:
            self._queue_drop_count += 1

    # ------------------------------------------------------------------
    # writer thread
    # ------------------------------------------------------------------
    def _worker(self):
        while True:
            item = self._q.get()
            if item is None:                   # sentinel : arrêt propre
                break
            frame_id, frame = item
            self._write_frame(frame_id, frame)

    def _write_frame(self, frame_id: int, frame):
        """Encode une frame en JPEG et l'écrit sur disque."""
        path = self._path / f"frame_{self._session_id}_{frame_id}.jpg"
        ok, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        )
        if not ok:
            log.warning("cv2.imencode a échoué pour frame_id=%d", frame_id)
            return
        try:
            with open(path, "wb") as fh:
                fh.write(buf.tobytes())
            self._count_written += 1
        except Exception:
            log.exception("échec écriture %s", path)

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "ring_miss": self._ring_miss_count,
            "queue_drop": self._queue_drop_count,
            "skip_invalid_frame_id": self._skip_invalid_frame_id,
            "written": self._count_written,
        }
