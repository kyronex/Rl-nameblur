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

# ── Sections autorisées par canal (cf. docs/bench-jsonl-schema.md §7) ────────
# Matrice de validation défensive : toute section présente dans le snap mais
# absente de cet ensemble pour un mode donné sera rejetée par _validate_snap().
# Note : §7 décrit les sections AUTORISÉES (pas obligatoires) — l'absence
# d'une section autorisée est légitime (pas de donnée sur la fenêtre).
_ALLOWED_SECTIONS: dict[str, frozenset[str]] = {
    "agg":   frozenset({"probes", "gauges", "rates"}),
    "frame": frozenset({"probes", "gauges", "counts"}),
    "fast":  frozenset({"probes", "gauges", "rates"}),
}

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

        self._logged_violations: set[tuple[str, str]] = set()
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
        if not snap:
            self._bench.count("bench_writer_frame_empty_snap")
            return
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

    def _validate_snap(self, snap) -> bool:
        """Filtre défensif §7 — valide la structure du snap AVANT enrichissement.
        En cas de rejet :
          - Incrémente le compteur bench dédié (auto-exclu via _is_writer_probe).
          - Émet un warning UNIQUE par (mode, motif) — déduplication via
            self._logged_violations.

        Returns:
            True si le snap est valide et peut être écrit.
            False si le snap doit être rejeté (early return dans _enqueue).
        """
        # Règle 1a : snap doit être un dict
        if not isinstance(snap, dict):
            self._bench.count(f"bench_writer_{self._mode}_rejected_invalid_type")
            key = (self._mode, "__non_dict__")
            if key not in self._logged_violations:
                self._logged_violations.add(key)
                log.warning(
                    "[bench.writer.%s] snap rejeté : type %s (attendu : dict) — warning unique",
                    self._mode,
                    type(snap).__name__,
                )
            return False

        # Règle 1b : snap doit être non vide
        if not snap:
            self._bench.count(f"bench_writer_{self._mode}_rejected_empty")
            key = (self._mode, "__empty__")
            if key not in self._logged_violations:
                self._logged_violations.add(key)
                log.warning(
                    "[bench.writer.%s] snap rejeté : dict vide — warning unique",
                    self._mode,
                )
            return False

        allowed = _ALLOWED_SECTIONS[self._mode]

        # Règle 2 : sections autorisées uniquement
        for section in snap.keys():
            if section not in allowed:
                self._bench.count(f"bench_writer_{self._mode}_rejected_forbidden_section")
                key = (self._mode, section)
                if key not in self._logged_violations:
                    self._logged_violations.add(key)
                    log.warning(
                        "[bench.writer.%s] section interdite %r rejetée (autorisées : %s) — "
                        "warning unique par section",
                        self._mode,
                        section,
                        sorted(allowed),
                    )
                return False

        # Règle 3 : chaque section présente doit être un dict
        for section, value in snap.items():
            if not isinstance(value, dict):
                self._bench.count(f"bench_writer_{self._mode}_rejected_invalid_type")
                key = (self._mode, f"__section_type__:{section}")
                if key not in self._logged_violations:
                    self._logged_violations.add(key)
                    log.warning(
                        "[bench.writer.%s] section %r de type %s rejetée (attendu : dict) — "
                        "warning unique par section",
                        self._mode,
                        section,
                        type(value).__name__,
                    )
                return False

        return True

    def _enqueue(self, snap: dict):
        """Sérialise + enqueue. Drop + sonde si queue pleine."""
        if not self._validate_snap(snap):
            return

        line = json.dumps(
            {
                "schema_version": 1,
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
        """Consommateur : lit la queue et écrit dans le fichier.
        Deux causes de sortie de la boucle principale :
        - timeout sur get() + stop_event positionné
        - poison pill reçue (line is None)
        Dans les deux cas, un drain unique vide la queue avant de retourner.
        """
        while True:
            try:
                line = self._q.get(timeout=0.5)
            except queue.Empty:
                if self._stop_event.is_set():
                    break          # → drain
                continue           # pas encore arrêté, on reboucle

            if line is None:
                break              # poison pill → drain

            if self._fh is not None:
                try:
                    self._fh.write(line + "\n")
                except OSError as e:
                    log.error("[bench.writer.%s] erreur écriture : %s", self._mode, e)

        # ── Drain unique ──────────────────────────────────────────────────────────
        # Atteint depuis les deux chemins de sortie (stop_event ou poison pill).
        # Vide les lignes restantes ; s'arrête sur Empty ou nouvelle poison pill.
        while True:
            try:
                line = self._q.get_nowait()
            except queue.Empty:
                break

            if line is None:
                break

            if self._fh is not None:
                try:
                    self._fh.write(line + "\n")
                except OSError as e:
                    log.error("[bench.writer.%s] erreur écriture : %s", self._mode, e)

