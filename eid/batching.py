# eid/batching.py
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from abc import ABC, abstractmethod
import logging
from dataclasses import replace
from eid.observation import Observation
from eid.config import StreamConfig
from eid.counters import Counters, CounterKey
log = logging.getLogger("batching")

class ObservationSource(ABC):
    def __init__(self, counters: Counters):
        if not isinstance(counters, Counters):
            raise TypeError("[ObservationSource] `counters` doit être un Counters.")
        self.counters = counters
    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

class JsonlSource(ObservationSource):
    """Lecture JSONL pure : aucune déduplication et aucun seq attribué."""
    def __init__(self, path: Path, counters: Counters, config: StreamConfig):
        super().__init__(counters); self.path = path; self.config = config
    def __iter__(self):
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line: continue
                try: snap = json.loads(line)
                except json.JSONDecodeError as e:
                    log.warning("Ligne ignorée (JSON invalide) : %s", e); continue
                for entry in snap.get("entries", []):
                    obs = Observation.from_entry(entry, self.counters)
                    if obs is not None: yield obs

class DedupSource(ObservationSource):
    """Déduplication amortie puis attribution d'un seq dense."""
    def __init__(self, source: ObservationSource, counters: Counters, config: Optional[StreamConfig] = None):
        super().__init__(counters); self.source = source; self.config = config
    def __iter__(self):
        last_keys: Dict[str, Dict[Tuple, float]] = {"slow": {}, "fast": {}}
        purge_clocks: Dict[str, float] = {}
        seq = 0
        ttl = self.config.dedup_ttl_s if self.config is not None else 2.0
        since_purge = 0; PURGE_EVERY = 256
        for obs in self.source:
            prev_clock = purge_clocks.get(obs.source, float("-inf"))
            clock_src = obs.ts if obs.ts > prev_clock else prev_clock
            purge_clocks[obs.source] = clock_src
            since_purge += 1

            if since_purge >= PURGE_EVERY:
                for src, s_keys in last_keys.items():
                    cutoff_s = purge_clocks.get(src, float("-inf")) - ttl
                    for key in [k for k, v in s_keys.items() if v < cutoff_s]:
                        del s_keys[key]
                since_purge = 0
                self.counters.bump(CounterKey.SRC_DEDUP_PURGES)

            if obs.dedup_key is not None:
                source_keys = last_keys[obs.source]
                cutoff_src = clock_src - ttl
                if obs.dedup_key in source_keys:
                    if source_keys[obs.dedup_key] >= cutoff_src:
                        self.counters.bump(CounterKey.SRC_OBS_DEDUP_SKIPPED)
                        continue
                source_keys[obs.dedup_key] = obs.ts
                self.counters.bump(CounterKey.SRC_DEDUP_KEPT)
            yield replace(obs, seq=seq); seq += 1

class BatchReader:
    """Lecteur par lots, indépendant du builder et du sliding."""
    def __init__(self, source: ObservationSource, config: StreamConfig, counters: Counters):
        self._it = iter(source); self.config = config; self._counters = counters
        self._pending = None; self._exhausted = False; self._batch_size_max = 0; self._last_frame_id = None
    @property
    def batch_size_max(self) -> int: return self._batch_size_max
    def _pull(self):
        obs = next(self._it, None)
        if obs is not None:
            if self._last_frame_id is not None and obs.frame_id < self._last_frame_id:
                self._counters.bump(CounterKey.BATCH_FRAME_OUT_OF_ORDER)
            self._last_frame_id = obs.frame_id
        return obs
    def _record(self, batch):
        n = len(batch); self._counters.bump(CounterKey.BATCHES_READ); self._counters.bump(CounterKey.BATCH_OBS_CUMUL, n)
        self._batch_size_max = max(self._batch_size_max, n)
    def next_batch(self) -> List[Observation]:
        if self._exhausted and self._pending is None: return []
        first = self._pending; self._pending = None
        if first is None:
            first = self._pull()
            if first is None: self._exhausted = True; return []
        batch = [first]; t0 = first.ts
        while True:
            nxt = self._pull()
            if nxt is None: self._exhausted = True; break
            if nxt.ts - t0 <= self.config.epsilon_close_s: batch.append(nxt)
            else: self._pending = nxt; break
        self._record(batch); return batch
