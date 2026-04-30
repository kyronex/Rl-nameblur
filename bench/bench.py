# bench.py
import time
import threading
from collections import defaultdict
from config import cfg

# Sondes routées vers le CSV "fast" (FastTrackThread).
# Tout le reste va dans flat_row() (CSV agg principal).
FAST_PREFIXES = ("fast_", "of_lk_")

class BenchRegistry:
    """Registre centralisé + timer + affichage."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._probes = defaultdict(list)
                    inst._counters = defaultdict(int)
                    inst._last = {}
                    inst._enabled = False
                    inst._probe_lock = threading.Lock()
                    cls._instance = inst
        return cls._instance

    # ── activation ──
    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    @property
    def active(self):
        return self._enabled

    # ── mesures ──
    def probe(self, name: str, duration_ms: float):
        if not self._enabled:
            return
        with self._probe_lock:
            self._probes[name].append(duration_ms)
            self._last[name] = duration_ms

    def last(self, name: str):
        """Dernière mesure brute. Retourne None si jamais mesuré."""
        return self._last.get(name, None)

    def count(self, name: str, n: int = 1):
        if not self._enabled:
            return
        with self._probe_lock:
            self._counters[name] += n

    # ── timer context manager ──
    def timer(self, name: str):
        return _BenchTimer(name, self)

    # ── helpers ──
    @staticmethod
    def _is_fast(name: str) -> bool:
        return name.startswith(FAST_PREFIXES)

    # ── collecte ──
    def summary(self, include_fast: bool = True, only_fast: bool = False) -> dict:
        """
        include_fast=True,  only_fast=False  → tout (compat existante)
        include_fast=False                   → exclut sondes fast_*/of_lk_*
        only_fast=True                       → uniquement fast_*/of_lk_*
        """
        with self._probe_lock:
            result = {}
            for name, values in self._probes.items():
                if not values:
                    continue
                if only_fast and not self._is_fast(name):
                    continue
                if not include_fast and self._is_fast(name):
                    continue
                result[name] = {
                    "avg":   round(sum(values) / len(values), 3),
                    "max":   round(max(values), 3),
                    "min":   round(min(values), 3),
                    "count": len(values),
                }
            for name, val in self._counters.items():
                if only_fast and not self._is_fast(name):
                    continue
                if not include_fast and self._is_fast(name):
                    continue
                result.setdefault(name, {})["total"] = val
            return result

    def flat_row(self) -> dict:
        """Dict aplati pour csv_write_agg() — exclut sondes fast_*/of_lk_*."""
        s = self.summary(include_fast=False)
        row = {"timestamp": time.strftime("%H:%M:%S")}
        for name, data in s.items():
            for key, val in data.items():
                row[f"{name}_{key}"] = val
        return row

    def fast_row(self) -> dict:
        """Dict aplati pour csv_write_fast() — uniquement sondes fast_*/of_lk_*."""
        s = self.summary(only_fast=True)
        row = {"timestamp": time.strftime("%H:%M:%S")}
        for name, data in s.items():
            for key, val in data.items():
                row[f"{name}_{key}"] = val
        return row

    # ── affichage ──
    def print_summary(self):
        if not self._enabled:
            return
        s = self.summary()
        if not s:
            return
        lines = ["  ┌─ BENCH ─────────────────────────────"]
        for name, data in sorted(s.items()):
            parts = []
            if "avg" in data:
                parts.append(f"avg={data['avg']}ms")
                parts.append(f"max={data['max']}ms")
                parts.append(f"n={data['count']}")
            if "total" in data:
                parts.append(f"total={data['total']}")
            lines.append(f"  │ {name:.<25s} {' | '.join(parts)}")
        lines.append("  └────────────────────────────────────")
        print("\n".join(lines))

    def reset(self, only_fast: bool = False, only_main: bool = False):
        """
        Sans arg          → reset complet (compat existante)
        only_fast=True    → reset uniquement sondes fast_*/of_lk_*
        only_main=True    → reset uniquement sondes non-fast
        """
        with self._probe_lock:
            if not only_fast and not only_main:
                self._probes.clear()
                self._counters.clear()
                return
            target_is_fast = only_fast  # True si on vise les fast, False sinon
            keys_p = [k for k in list(self._probes.keys())
                      if self._is_fast(k) == target_is_fast]
            keys_c = [k for k in list(self._counters.keys())
                      if self._is_fast(k) == target_is_fast]
            for k in keys_p:
                del self._probes[k]
            for k in keys_c:
                del self._counters[k]


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
if cfg.get("debug.bench.enabled", False):
    bench.enable()
