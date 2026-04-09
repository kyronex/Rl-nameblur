# config.py
import yaml
import threading
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config.yaml"

_RESTART_REQUIRED_KEYS = {
    "screen.width",
    "screen.height",
    "screen.capture_fps",
    "screen.vcam_fps",
}

def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _flatten(d: dict, prefix: str = "") -> dict:
    """{'screen': {'width': 1920}} → {'screen.width': 1920}"""
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out

def _diff(old: dict, new: dict) -> set:
    keys = set(old) | set(new)
    return {k for k in keys if old.get(k) != new.get(k)}

class Config:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._data: dict = {}
        self._flat: dict = {}
        self._last_good: dict = {}
        self._last_good_flat: dict = {}
        self._file_mtime: float = 0.0
        self._reload_lock = threading.Lock()
        self._watcher_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Chargement initial — fatal si raté
        try:
            raw = _load_yaml(CONFIG_PATH)
            self._data = raw
            self._flat = _flatten(raw)
            self._last_good = raw
            self._last_good_flat = self._flat.copy()
            self._file_mtime = CONFIG_PATH.stat().st_mtime
            logger.info(f"[Config] Chargé : {CONFIG_PATH}")
        except Exception as e:
            logger.critical(f"[Config] Impossible de charger {CONFIG_PATH} : {e}")
            raise SystemExit(1)

    # ── Accès ──────────────────────────────────────────────────────────────

    def get(self, key: str, default=None):
        """cfg.get('screen.width') → 1920"""
        with self._reload_lock:
            return self._flat.get(key, default)

    def section(self, name: str) -> dict:
        """cfg.section('masks') → {'ttl_max': 4, ...}"""
        with self._reload_lock:
            return dict(self._data.get(name, {}))

    # ── Hot-reload ─────────────────────────────────────────────────────────

    def start_watcher(self, interval: float = 1.0):
        self._stop_event.clear()
        self._watcher_thread = threading.Thread(
            target=self._watch_loop,
            args=(interval,),
            daemon=True,
            name="ConfigWatcher",
        )
        self._watcher_thread.start()
        logger.info("[Config] Watcher démarré")

    def stop_watcher(self):
        self._stop_event.set()

    def _watch_loop(self, interval: float):
        while not self._stop_event.is_set():
            time.sleep(interval)
            try:
                mtime = CONFIG_PATH.stat().st_mtime
                if mtime <= self._file_mtime:
                    continue

                raw = _load_yaml(CONFIG_PATH)
                new_flat = _flatten(raw)
                changed = _diff(self._last_good_flat, new_flat)

                restart_needed = changed & _RESTART_REQUIRED_KEYS
                if restart_needed:
                    logger.warning(
                        f"[Config] Clés nécessitant un redémarrage modifiées "
                        f"(ignorées à chaud) : {restart_needed}"
                    )
                    hot_changed = changed - _RESTART_REQUIRED_KEYS
                else:
                    hot_changed = changed

                if hot_changed:
                    with self._reload_lock:
                        self._data = raw
                        self._flat = new_flat
                        self._last_good = raw
                        self._last_good_flat = new_flat.copy()
                        self._file_mtime = mtime
                    logger.info(f"[Config] Rechargé — clés modifiées : {hot_changed}")

            except Exception as e:
                logger.error(
                    f"[Config] Erreur reload ({e}) — "
                    f"ancienne config conservée"
                )

cfg = Config()