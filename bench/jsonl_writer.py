# bench/jsonl_writer.py
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bench.bench import BenchRegistry

log = logging.getLogger("bench.jsonl_writer")

# ── Modes de snapshot supportés ──────────────────────────────────────────────
_VALID_MODES = ("agg", "frame", "fast")

class BenchJsonlWriter:
    """Thread daemon qui écrit périodiquement un snapshot bench en JSONL.

    Trois modes :
      - "agg"   → appelle bench.snapshot_all(window_s)   — 1 ligne / interval_s
      - "frame" → appelle bench.snapshot_frame()         — 1 ligne / frame (push externe)
      - "fast"  → appelle bench.snapshot_fast()          — 1 ligne / interval_s

    Architecture queue :
      - Queue bornée (maxsize configurable).
      - Producteur : _enqueue() appelé par _tick() ou push_frame() — put_nowait().
      - Consommateur : _writer_loop() — thread daemon dédié.
      - Saturation : ligne droppée + bench.count("bench_writer_dropped") incrémenté.

    Cycle de vie :
      - start() → ouvre fichier + démarre thread → True si OK, False si OSError.
      - stop()  → vide la queue (drain) + join(timeout) + ferme fichier.
    """

    def __init__(
        self,
        bench_registry: "BenchRegistry",
        *,
        mode: str,
        path: str,
        session_id: str,
        interval_s: float = 1.0,
        queue_maxsize: int = 10000,
        shutdown_timeout_s: float = 2.0,
    ):
        if mode not in _VALID_MODES:
            raise ValueError(f"[BenchJsonlWriter] mode invalide : {mode!r} — attendu : {_VALID_MODES}")

        self._bench = bench_registry
        self._mode = mode
        self._session_id = session_id
        self._interval_s = max(0.1, interval_s)
        self._queue_maxsize = queue_maxsize
        self._shutdown_timeout_s = shutdown_timeout_s

        # Insertion session_id avant l'extension
        base, ext = os.path.splitext(path)
        self._path = f"{base}_{session_id}{ext}" if ext else f"{base}_{session_id}"

        self._fh = None
        self._q: queue.Queue[str | None] = queue.Queue(maxsize=queue_maxsize)

        self._tick_thread: threading.Thread | None = None   # producteur périodique (agg / fast)
        self._writer_thread: threading.Thread | None = None # consommateur queue → fichier
        self._stop_event = threading.Event()

    # ─────────────────────────────────────────────────────────────
    #  Cycle de vie
    # ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Ouvre le fichier et démarre les threads.

        Retourne True si OK, False sur OSError (writer désactivé sans crash).
        """
        try:
            dirname = os.path.dirname(self._path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            self._fh = open(self._path, "a", buffering=1, encoding="utf-8")
        except OSError as e:
            log.warning(
                "[bench.writer.%s] échec ouverture '%s' : %s — writer désactivé",
                self._mode, self._path, e,
            )
            return False

        # Thread consommateur (toujours présent)
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name=f"BenchWriter-{self._mode}",
            daemon=True,
        )
        self._writer_thread.start()

        # Thread producteur périodique (agg + fast uniquement)
        if self._mode in ("agg", "fast"):
            self._tick_thread = threading.Thread(
                target=self._tick_loop,
                name=f"BenchTick-{self._mode}",
                daemon=True,
            )
            self._tick_thread.start()

        log.info(
            "[bench.writer.%s] démarré (path=%s, interval=%.2fs, queue=%d)",
            self._mode, self._path, self._interval_s, self._queue_maxsize,
        )
        return True

    def stop(self):
        """Arrêt propre : drain queue + join threads + fermeture fichier."""
        self._stop_event.set()

        # Arrêt producteur périodique
        if self._tick_thread is not None:
            self._tick_thread.join(timeout=self._shutdown_timeout_s)
            self._tick_thread = None

        # Poison pill → débloquer le consommateur s'il attend sur get()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass

        if self._writer_thread is not None:
            self._writer_thread.join(timeout=self._shutdown_timeout_s)
            if self._writer_thread.is_alive():
                log.warning(
                    "[bench.writer.%s] thread toujours vivant après %.1fs — abandon",
                    self._mode, self._shutdown_timeout_s,
                )
            self._writer_thread = None

        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except OSError:
                pass
            self._fh = None

        log.info("[bench.writer.%s] arrêté", self._mode)

    # ─────────────────────────────────────────────────────────────
    #  API publique — mode "frame"
    # ─────────────────────────────────────────────────────────────

    def push_frame(self):
        """Enqueue un snapshot frame (mode 'frame' uniquement).

        Appelé depuis la boucle principale à chaque frame capturée.
        No-op si mode != 'frame'.
        """
        if self._mode != "frame":
            return
        snap = self._bench.snapshot_frame()
        if snap:
            self._enqueue(snap)

    # ─────────────────────────────────────────────────────────────
    #  Threads internes
    # ─────────────────────────────────────────────────────────────

    def _tick_loop(self):
        """Producteur périodique pour modes agg et fast."""
        while not self._stop_event.wait(self._interval_s):
            try:
                self._tick()
            except Exception:
                log.exception("[bench.writer.%s] erreur tick", self._mode)

    def _tick(self):
        """Construit le snapshot selon le mode et l'enqueue."""
        if self._mode == "agg":
            snap = self._bench.snapshot_all(self._interval_s)
        elif self._mode == "fast":
            snap = self._bench.snapshot_fast()
        else:
            return

        if snap:
            self._enqueue(snap)

    def _enqueue(self, snap: dict):
        """Sérialise + enqueue. Drop + sonde si queue pleine."""
        line = json.dumps(
            {
                "ts": time.time(),
                "mono": time.perf_counter(),
                "session_id": self._session_id,
                "mode": self._mode,
                **snap,
            },
            separators=(",", ":"),
        )
        try:
            self._q.put_nowait(line)
            # Auto-sonde taille queue (best-effort, pas de lock)
            self._bench.probe(
                f"bench_writer_{self._mode}_queue_size",
                float(self._q.qsize()),
            )
        except queue.Full:
            self._bench.count(f"bench_writer_{self._mode}_dropped")
            log.debug(
                "[bench.writer.%s] queue pleine — ligne droppée (dropped total=%d)",
                self._mode,
                self._bench.read_count(f"bench_writer_{self._mode}_dropped"),
            )

    def _writer_loop(self):
        """Consommateur : lit la queue et écrit dans le fichier."""
        while True:
            try:
                line = self._q.get(timeout=0.5)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue

            # Poison pill
            if line is None:
                break

            if self._fh is not None:
                try:
                    self._fh.write(line + "\n")
                except OSError as e:
                    log.error("[bench.writer.%s] erreur écriture : %s", self._mode, e)
