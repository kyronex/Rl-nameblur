# bench/bench.py
import time
import threading
from collections import defaultdict, deque
from datetime import datetime
import numpy as np
import logging

from dataclasses import asdict, is_dataclass

from config import cfg
from bench.jsonl_writer import BenchJsonlWriter
from bench.lifecycle import LifecycleRecord, LifecycleEvent

log = logging.getLogger("bench.bench")

"""
Bench — registre centralisé de métriques runtime (probes / counts / gauges).
T-1  Canaux JSONL : 3 writers dédiés (agg / frame / fast), session_id partagé
     calculé au boot (cf. _maybe_start_writers). Détail rapport : voir
     docs/bench-jsonl-schema.md.
T-2  Nommage sondes : <domaine>_<action>[_<qualifieur>], préfixe domaine
     obligatoire. Préfixe 'bench_writer_' RÉSERVÉ aux auto-sondes des writers,
     exclu des snapshots métier via _is_writer_probe (spécifique à ce module).
T-3  Sémantique des snapshot_*() :
       • snapshot_all()   → CUMULATIF depuis start (jamais reset implicite) → agg
       • snapshot_frame() → DIFFÉRENTIEL (vidange buffers à chaque appel)   → frame
       • snapshot_fast()  → fenêtre glissante history_window_s, filtre fast_* → fast
     reset() réservé aux tests / redémarrages explicites.
"""

# ── Préfixes auto-sondes writer (à exclure des snapshots pour éviter la récursion observable) ──
_WRITER_PROBE_PREFIX = "bench_writer_"

# ── Sondes considérées "fast" (canal dédié snapshot_fast) ──
_FAST_PROBE_PREFIXES = ("fast_",)

class BenchRegistry:
    """Registre centralisé + timer + snapshots multi-canaux."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._counters = defaultdict(int)
                    inst._last = {}
                    inst._gauges = {}
                    inst._probe_history = defaultdict(deque)
                    inst._count_history = defaultdict(deque)
                    inst._gauge_history = defaultdict(deque)
                    inst._enabled = False
                    inst._probe_lock = threading.Lock()
                    inst._history_window = float(cfg.get("debug.bench.history_window_s", 60.0))
                    inst._writers = {}
                    inst._session_id = ""
                    # Buffer frame : sondes accumulées entre 2 appels snapshot_frame()
                    inst._frame_probes = defaultdict(list)
                    inst._frame_counts = defaultdict(int)
                    inst._events = {}
                    inst._detections = defaultdict(list)
                    cls._instance = inst
        return cls._instance

    # ─────────────────────────────────────────────────────────────
    #  Activation
    # ─────────────────────────────────────────────────────────────

    def enable(self):
        self._enabled = True
        self._maybe_start_writers()

    def disable(self):
        self._enabled = False

    @property
    def active(self):
        return self._enabled

    # ─────────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _purge_history(hist: deque, cutoff: float):
        """Purge paresseuse : retire tous les points horodatés avant `cutoff`."""
        while hist and hist[0][0] < cutoff:
            hist.popleft()

    # ─────────────────────────────────────────────────────────────
    #  Mesures
    # ─────────────────────────────────────────────────────────────

    def probe(self, name: str, duration_ms: float):
        if not self._enabled:
            return
        now = time.perf_counter()
        cutoff = now - self._history_window
        with self._probe_lock:
            self._last[name] = duration_ms
            hist = self._probe_history[name]
            hist.append((now, duration_ms))
            self._purge_history(hist, cutoff)
            # Buffer frame
            self._frame_probes[name].append(duration_ms)

    def timer(self, name: str) -> "_BenchTimer":
        """Context manager mesurant un bloc et écrivant via probe(name, ms)."""
        return _BenchTimer(name, self)

    def last(self, name: str):
        """Dernière mesure brute. Retourne None si jamais mesuré."""
        return self._last.get(name, None)

    def count(self, name: str, n: int = 1):
        if not self._enabled:
            return
        now = time.perf_counter()
        cutoff = now - self._history_window
        with self._probe_lock:
            self._counters[name] += n
            hist = self._count_history[name]
            hist.append((now, n))
            self._purge_history(hist, cutoff)
            # Buffer frame
            self._frame_counts[name] += n

    def read_count(self, name: str) -> int:
        """Lecture du compteur cumulatif (depuis start). Retourne 0 si jamais incrémenté."""
        with self._probe_lock:
            return self._counters.get(name, 0)

    def gauge(self, name: str, value: float):
        """Valeur instantanée écrasante + historisation."""
        if not self._enabled:
            return
        now = time.perf_counter()
        cutoff = now - self._history_window
        with self._probe_lock:
            self._gauges[name] = value
            hist = self._gauge_history[name]
            hist.append((now, value))
            self._purge_history(hist, cutoff)

    def read_gauge(self, name: str):
        """Lecture de la dernière valeur de gauge. Retourne None si jamais posée."""
        with self._probe_lock:
            return self._gauges.get(name, None)

    def rate(self, name: str, window_s: float) -> float:
        """Débit (count / fenêtre) sur les `window_s` dernières secondes."""
        if not self._enabled:
            return 0.0
        now = time.perf_counter()
        cutoff = now - window_s
        with self._probe_lock:
            hist = self._count_history.get(name)
            if not hist:
                return 0.0
            self._purge_history(hist, cutoff)
            if not hist:
                return 0.0
            total = sum(n for _, n in hist)
            return total / window_s

    # ─────────────────────────────────────────────────────────────
    #  Snapshots pour writers
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _is_writer_probe(name: str) -> bool:
        return name.startswith(_WRITER_PROBE_PREFIX)

    @staticmethod
    def _is_fast_probe(name: str) -> bool:
        return any(name.startswith(p) for p in _FAST_PROBE_PREFIXES)

    @staticmethod
    def _jsonable_val(v):
        """Réduit une valeur à un primitif JSON-safe. Fallback str() → jamais de crash."""
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return str(v)

    def _jsonable_scores(self, scores):
        """Convertit un dict {str: MatchScore|primitive} en dict 100% JSON-safe.
        MatchScore est @dataclass(frozen=True, slots=True) → PAS de __dict__ :
        on passe par dataclasses.asdict() (compatible slots), jamais par vars().
        """
        if not scores:
            return {}
        out = {}
        for k, v in scores.items():
            if is_dataclass(v) and not isinstance(v, type):
                out[str(k)] = {kk: self._jsonable_val(vv) for kk, vv in asdict(v).items()}
            else:
                out[str(k)] = self._jsonable_val(v)
        return out

    def emit_lifecycle(self, event: str, mask, reason: str | None = None):
        from core.mask import MaskState
        """Émet un événement lifecycle pour un mask.
        Tous les champs du record sont peuplés conformément à LifecycleRecord (lifecycle.py).
        session_id est ajouté par le writer._enqueue() — non dupliqué ici.
        """
        # ── Accès thread-safe aux attributs du mask (peuvent être un Mask ou un int) ──
        try:
            mask_id = int(getattr(mask, "uid", mask))
        except Exception:
            mask_id = int(mask) if mask is not None else 0

        event_ts    = time.perf_counter()
        state       = getattr(mask, "state", None)
        state_val   = getattr(state, "name", str(state)) if state is not None else ""
        rect        = getattr(mask, "rect", ()) or ()
        conf        = getattr(mask, "confidence", 0.0)
        created_ts  = getattr(mask, "created_ts", 0.0)
        total_cumul = getattr(mask, "total_matches_cumul", 0)
        frames_m    = getattr(mask, "frames_matched", 0)
        last_src    = getattr(mask, "last_source", None)
        lost_since  = getattr(mask, "lost_since_ts", None)
        frame_id    = getattr(mask, "frame_id", -1)
        hash_hist   = getattr(mask, "hash_history", None)
        scores      = getattr(mask, "scores", None)

        record: LifecycleRecord = {
            "event":               event.value,
            "mask_id":             mask_id,
            "state":               state_val,
            "rx":                  float(rect[0]) if len(rect) > 0 else 0.0,
            "ry":                  float(rect[1]) if len(rect) > 1 else 0.0,
            "rw":                  float(rect[2]) if len(rect) > 2 else 0.0,
            "rh":                  float(rect[3]) if len(rect) > 3 else 0.0,
            "confidence":          float(conf),
            "created_ts":          float(created_ts),
            "event_ts":            float(event_ts),
            "total_matches_cumul": int(total_cumul),
            "frames_matched":      int(frames_m),
            "source":              last_src if last_src is not None else None,
            "lost_since_ts":       float(lost_since) if lost_since is not None else None,
            "reason":              reason if reason is not None else None,
            "revived":             True if (event.value == LifecycleEvent.REVIVE.value and state_val in (MaskState.PENDING.name, MaskState.CONFIRMED.name)) else None,
            "frame_id":            int(frame_id),
            "scores":              self._jsonable_scores(scores) if scores else {},
            "hash_history":        list(hash_hist or []),
        }
        with self._probe_lock:
            self._events[mask_id] = record

    def emit_detection(self, det):
        if not self._enabled:
            return
        frame_id = int(getattr(det, "frame_id", -1))
        record = {
            "frame_id":   frame_id,
            "rx": float(det.rect[0]), "ry": float(det.rect[1]),
            "rw": float(det.rect[2]), "rh": float(det.rect[3]),
            "phash":      det.phash,
            "source":     det.source,
            "confidence": float(det.confidence),
            "scores":     self._jsonable_scores(det.scores),
        }
        with self._probe_lock:
            self._detections[frame_id].append(record)

    def snapshot_all(self, window_s: float) -> dict:
        """Snapshot agrégé canal 'agg' — exclut les sondes fast et writer."""
        if not self._enabled:
            return {}

        now = time.perf_counter()
        cutoff = now - window_s

        probes = {}
        gauges = {}
        rates = {}

        with self._probe_lock:
            for name, hist in self._probe_history.items():
                if self._is_writer_probe(name) or self._is_fast_probe(name):
                    continue
                values = [v for ts, v in hist if ts >= cutoff]
                if not values:
                    continue
                probes[name] = {
                    "avg": sum(values) / len(values),
                    "max": max(values),
                    "min": min(values),
                    "count": len(values),
                }

            for name, value in self._gauges.items():
                if self._is_fast_probe(name):
                    continue
                gauges[name] = value

            for name, hist in self._count_history.items():
                if self._is_writer_probe(name) or self._is_fast_probe(name):
                    continue
                total = sum(n for ts, n in hist if ts >= cutoff)
                if total == 0:
                    continue
                rates[name] = total / window_s

        if not (probes or gauges or rates):
            return {}
        return {"probes": probes, "gauges": gauges, "rates": rates}

    def snapshot_frame(self) -> dict:
        """Snapshot différentiel depuis le dernier appel.

        Vide les buffers `_frame_probes` et `_frame_counts` après lecture.
        Retourne {} si aucune donnée bufferisée.
        """
        if not self._enabled:
            return {}

        with self._probe_lock:
            if not self._frame_probes and not self._frame_counts:
                return {}

            probes = {}
            for name, values in self._frame_probes.items():
                if self._is_writer_probe(name) or self._is_fast_probe(name):
                    continue
                if not values:
                    continue
                probes[name] = {
                    "avg": sum(values) / len(values),
                    "max": max(values),
                    "min": min(values),
                    "count": len(values),
                }

            counts = {}
            for name, n in self._frame_counts.items():
                if self._is_writer_probe(name) or self._is_fast_probe(name):
                    continue
                if n == 0:
                    continue
                counts[name] = n

            # Snapshot gauges instantané (pas de buffering — sémantique état courant)
            gauges = {
                name: value
                for name, value in self._gauges.items()
                if not self._is_fast_probe(name)
            }

            # Vidage buffers
            self._frame_probes.clear()
            self._frame_counts.clear()

        if not (probes or counts or gauges):
            return {}
        return {"probes": probes, "counts": counts, "gauges": gauges}

    def snapshot_fast(self) -> dict:
        """Snapshot des sondes fast_* sur la fenêtre de rétention."""
        if not self._enabled:
            return {}

        now = time.perf_counter()
        cutoff = now - self._history_window

        probes = {}
        gauges = {}
        rates = {}

        with self._probe_lock:
            for name, hist in self._probe_history.items():
                if not self._is_fast_probe(name):
                    continue
                values = [v for ts, v in hist if ts >= cutoff]
                if not values:
                    continue
                probes[name] = {
                    "avg": sum(values) / len(values),
                    "max": max(values),
                    "min": min(values),
                    "count": len(values),
                }

            for name, value in self._gauges.items():
                if self._is_fast_probe(name):
                    gauges[name] = value

            for name, hist in self._count_history.items():
                if not self._is_fast_probe(name):
                    continue
                total = sum(n for ts, n in hist if ts >= cutoff)
                if total == 0:
                    continue
                rates[name] = total / self._history_window

        if not (probes or gauges or rates):
            return {}
        return {"probes": probes, "gauges": gauges, "rates": rates}

    def snapshot_events(self):
        if not self._enabled:
            return {}
        with self._probe_lock:
            drained = list(self._events.values())
            self._events.clear()
        return {"events": {"records": drained}}

    def snapshot_detections(self):
        if not self._enabled:
            return {}
        with self._probe_lock:
            drained = [r for recs in self._detections.values() for r in recs]
            self._detections.clear()
        return {"detections": {"records": drained}}

    # ─────────────────────────────────────────────────────────────
    #  Frame push (proxy vers writer frame)
    # ─────────────────────────────────────────────────────────────

    def push_frame(self):
        """Appelé par main.py à chaque frame capturée. No-op si writer frame inactif."""
        writer = self._writers.get("frame")
        if writer is not None:
            writer.push_frame()

    def push_events(self):
        """Appelé par main.py à chaque frame capturée. No-op si writer frame inactif."""
        writer = self._writers.get("events")
        if writer is not None:
            writer.push_events()

    def push_detections(self):
        writer = self._writers.get("detections")
        if writer is not None:
            writer.push_detections()

    def last_events(self) -> list[dict]:
        """Lecture NON-destructive des LifecycleRecords courants (le buffer n'est pas vidé : snapshot_events() du writer reste intact).
        Retourne [] si le writer events est inactif."""
        writer = self._writers.get("events")
        if writer is None:
            return []
        return list(self._events.values())


    # ─────────────────────────────────────────────────────────────
    #  Résumé console
    # ─────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Résumé global sur la fenêtre de rétention courante (history_window_s)."""
        with self._probe_lock:
            result = {}
            for name, hist in self._probe_history.items():
                if not hist:
                    continue
                values = [v for _, v in hist]
                result[name] = {
                    "avg": sum(values) / len(values),
                    "max": max(values),
                    "min": min(values),
                    "count": len(values),
                    "total": sum(values),
                }
            for name, total in self._counters.items():
                result.setdefault(name, {})["total"] = total
            for name, value in self._gauges.items():
                result.setdefault(name, {})["gauge"] = value
            return result

    def print_summary(self):
        s = self.summary()
        if not s:
            print("[bench] (aucune mesure)")
            return
        print("[bench] résumé :")
        for name in sorted(s.keys()):
            entry = s[name]
            parts = []
            if "avg" in entry:
                parts.append(
                    f"avg={entry['avg']:.2f}ms "
                    f"max={entry['max']:.2f}ms "
                    f"min={entry['min']:.2f}ms "
                    f"n={entry['count']}"
                )
            if "gauge" in entry:
                parts.append(f"gauge={entry['gauge']}")
            if "total" in entry and "avg" not in entry:
                parts.append(f"total={entry['total']}")
            print(f"  {name:<28} {' | '.join(parts)}")

    def reset(self):
        """Reset complet des probes, compteurs, gauges, historiques et buffers frame."""
        with self._probe_lock:
            self._counters.clear()
            self._last.clear()
            self._gauges.clear()
            self._probe_history.clear()
            self._count_history.clear()
            self._gauge_history.clear()
            self._frame_probes.clear()
            self._frame_counts.clear()

    # ─────────────────────────────────────────────────────────────
    #  Writers JSONL — cycle de vie
    # ─────────────────────────────────────────────────────────────

    def _maybe_start_writers(self):
        """Démarre les 3 writers (agg / frame / fast) selon config R6."""
        if self._writers:
            return
        if not cfg.get("debug.bench.writer.enabled", False):
            return

        # session_id partagé entre writers
        session_fmt = cfg.get("debug.bench.writer.session_id_format", "%Y%m%d_%H%M%S")
        self._session_id = datetime.now().strftime(session_fmt)

        queue_maxsize = int(cfg.get("debug.bench.writer.queue_maxsize", 10000))
        shutdown_timeout_s = float(cfg.get("debug.bench.writer.shutdown_timeout_s", 2.0))
        max_chars_cfg = cfg.get("debug.bench.writer.max_chars", None)
        max_chars = int(max_chars_cfg) if max_chars_cfg is not None else None

        for mode in ("agg", "frame", "fast", "events", "detections"):
            if not cfg.get(f"debug.bench.{mode}.enabled", False):
                continue
            path = cfg.get(f"debug.bench.{mode}.path", f"logs/json/bench_{mode}.jsonl")
            interval_s = float(cfg.get(f"debug.bench.{mode}.interval_s", 1.0))

            writer = BenchJsonlWriter(
                self,
                mode=mode,
                path=path,
                session_id=self._session_id,
                interval_s=interval_s,
                queue_maxsize=queue_maxsize,
                shutdown_timeout_s=shutdown_timeout_s,
                max_chars=max_chars
            )
            if writer.start():
                self._writers[mode] = writer

    def shutdown(self):
        """Arrêt propre : stoppe tous les writers actifs."""
        for mode, writer in list(self._writers.items()):
            writer.stop()
        self._writers.clear()


class _BenchTimer:
    """Context manager interne."""
    __slots__ = ("_name", "_registry", "_t0")

    def __init__(self, name, registry):
        self._name = name
        self._registry = registry
        self._t0 = 0.0

    def __enter__(self):
        if self._registry.active:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self._registry.active and self._t0:
            elapsed = (time.perf_counter() - self._t0) * 1000
            self._registry.probe(self._name, elapsed)
        return False


# ── singleton global ──
bench = BenchRegistry()
