# threads/fast_track_thread.py
import cv2
import threading
import math
import logging
from typing import List, Optional
from config import cfg as _global_cfg
from core.mask import FastMaskView
from detection.detect import ncc_match
from core.optical_flow import of_track
from threads.fast_track_config import FastTrackConfig

log = logging.getLogger("fast_track_thread")

class FastTrackThread:
    """
    Reçoit la frame courante + les FastMaskView actives.
    Tracking OF → NCC confirme → fallback stale si échec.

    Contrat slow→fast (B21) :
        Le thread fast consomme uniquement des FastMaskView (snapshots
        immuables), jamais des Mask. Cela garantit l'absence de race
        condition sur les champs mutables (vx, vy, template,
        last_detected_ts) que le slow peut modifier en parallèle.

    Hot-reload (B8) :
        La config est encapsulée dans `FastTrackConfig` (snapshot).
        `maybe_reload()` peut être appelé depuis le main thread (recommandé)
        ou depuis le worker en début de tick (filet de sécurité).
        L'accès à `self.cfg` dans le worker est protégé par `_cfg_lock`.

    F1 : le worker est piloté par Event — traite uniquement
         quand une nouvelle frame est déposée.
    """

    def __init__(self, config: Optional[FastTrackConfig] = None):
        self.cfg = config or FastTrackConfig()
        self._cfg_lock = threading.Lock()
        self._cfg_version = _global_cfg.version

        # Frame courante (input)
        self._latest_frame = None
        self._latest_frame_ts = 0.0
        self._latest_views: List[FastMaskView] = []
        self._frame_lock = threading.Lock()

        self._new_frame_event = threading.Event()

        # Résultats (output)
        self._results = []
        self._results_ts = 0.0
        self._result_version = 0
        self._results_lock = threading.Lock()

        # État interne worker
        self._prev_gray = None
        self._last_known = {}
        self._last_processed_ts = -1.0

        # Lifecycle
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ──────────── Contrôle ────────────

    def start(self):
        if self._running and self._thread and self._thread.is_alive():
            log.warning("[FastTrackThread] start() ignoré : déjà actif")
            return
        # Reset state au démarrage
        self._prev_gray = None
        self._last_known = {}
        self._last_processed_ts = -1.0
        self._new_frame_event.clear()

        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        log.info("[FastTrackThread] Démarré")

    def stop(self):
        if not self._running:
            return
        self._running = False
        self._new_frame_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                log.warning("[FastTrackThread] worker n'a pas terminé dans le timeout")
        self._thread = None
        log.info("[FastTrackThread] Arrêté")

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ──────────── Interface ────────────

    def give_frame_and_views(self, frame, views: List[FastMaskView], frame_ts: float):
        with self._frame_lock:
            #self._latest_frame = frame
            self._latest_frame = frame.copy()
            self._latest_frame_ts = frame_ts
            self._latest_views = list(views)
        self._new_frame_event.set()

    def get_results(self):
        with self._results_lock:
            return self._result_version, list(self._results), self._results_ts

    # ──────────── NCC sur ROI ────────────
    def _adaptive_margin(self, mask: FastMaskView, now: float, snap: FastTrackConfig) -> int:
        speed = math.sqrt(mask.vx * mask.vx + mask.vy * mask.vy)
        dt = now - mask.last_detected_ts
        if dt < 0:
            dt = 0

        margin = snap.am_base + speed * dt * snap.am_factor
        return int(max(snap.am_min, min(margin, snap.am_max)))

    def _ncc_on_roi(self, gray, rect, template, snap: FastTrackConfig, margin: Optional[int] = None):
        x, y, w, h = rect
        if margin is None:
            margin = snap.roi_margin
        rx = max(x - margin, 0)
        ry = max(y - margin, 0)
        rx2 = min(x + w + margin, snap.screen_w)
        ry2 = min(y + h + margin, snap.screen_h)

        roi = gray[ry:ry2, rx:rx2]
        if roi.size == 0:
            return None, 0.0

        score, loc = ncc_match(roi, template, threshold=snap.ncc_threshold)
        if loc is None:
            return None, score

        dx, dy = loc
        new_rect = (rx + dx, ry + dy, template.shape[1], template.shape[0])
        return new_rect, score

    # ──────────── Worker ────────────

    def _worker(self):
        while self._running:
            with self._cfg_lock:
                snap = self.cfg

            triggered = self._new_frame_event.wait(timeout=snap.event_timeout_s)
            if not self._running:
                break
            self._new_frame_event.clear()
            if not triggered:
                continue

            try:
                # Filet de sécurité hot-reload (au cas où main n'appelle pas)
                self.maybe_reload()
                with self._cfg_lock:
                    snap = self.cfg

                # ── 1. Récupérer frame + views ──
                with self._frame_lock:
                    frame = self._latest_frame
                    frame_ts = self._latest_frame_ts
                    views = self._latest_views

                if frame is None or not views:
                    self._prev_gray = None
                    continue

                if frame_ts <= self._last_processed_ts:
                    continue
                self._last_processed_ts = frame_ts

                # ── 2. Convertir en gris ──
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # ── 3. Pas de prev_gray → init et publier positions actuelles ──
                if self._prev_gray is None:
                    self._prev_gray = curr_gray
                    results = []
                    for v in views:
                        self._last_known[v.uid] = {"rect": v.rect, "stale": 0}
                        results.append((v.uid, v.rect, 1.0))
                    with self._results_lock:
                        self._results = results
                        self._results_ts = frame_ts
                        self._result_version += 1
                    continue

                # ── 4. Tracking par view ──
                results = []
                active_ids = set()
                max_stale = snap.max_stale_frames

                for v in views:
                    active_ids.add(v.uid)
                    if v.uid not in self._last_known:
                        self._last_known[v.uid] = {"rect": v.rect, "stale": 0}

                    last_state = self._last_known[v.uid]

                    # ── 4a. OF ──
                    candidate_rect, of_succeeded = of_track(
                        self._prev_gray, curr_gray, last_state["rect"]
                    )
                    if not of_succeeded:
                        candidate_rect = last_state["rect"]

                    # ── 4b. NCC confirme ──
                    if v.template is not None:
                        margin = self._adaptive_margin(v, frame_ts, snap)
                        ncc_rect, score = self._ncc_on_roi(
                            curr_gray, candidate_rect, v.template, snap, margin=margin
                        )
                    else:
                        ncc_rect, score = None, 0.0

                    if ncc_rect is not None:
                        last_state["rect"] = ncc_rect
                        last_state["stale"] = 0
                        results.append((v.uid, ncc_rect, score))
                    else:
                        last_state["stale"] += 1
                        if last_state["stale"] <= max_stale:
                            results.append((v.uid, last_state["rect"], score))
                        else:
                            results.append((v.uid, None, score))

                # ── 5. Purger masques disparus ──
                for old_id in list(self._last_known.keys()):
                    if old_id not in active_ids:
                        del self._last_known[old_id]

                # ── 6. Mémoriser frame courante ──
                self._prev_gray = curr_gray

                # ── 7. Publier ──
                with self._results_lock:
                    self._results = results
                    self._results_ts = frame_ts
                    self._result_version += 1

            except Exception as e:
                log.exception(f"[FastTrackThread] Erreur dans le worker: {e}")
                # On ne crashe pas le thread : tick suivant.

    # ───────────────────────────────────────────────
    #  HOT-RELOAD
    # ───────────────────────────────────────────────

    def reload_config(self, new_config: FastTrackConfig) -> None:
        """
        Remplace atomiquement le snapshot de config.
        Appelable depuis le main thread entre deux ticks worker.
        """
        with self._cfg_lock:
            self.cfg = new_config
        log.info("[FastTrackThread] Config rechargée")

    def maybe_reload(self) -> bool:
        """
        Vérifie la version du singleton `cfg` global et recharge si changée.
        Appelable depuis main (recommandé) ou worker (filet).
        Coût steady-state : 1 comparaison int.
        Returns: True si reload effectué, False sinon.
        """
        current = _global_cfg.version
        if current == self._cfg_version:
            return False
        old_version = self._cfg_version
        try:
            new_cfg = FastTrackConfig()
            self.reload_config(new_cfg)
        except Exception as e:
            log.error(f"[FastTrackThread] Échec reload v{old_version}→v{current}: {e}")
            return False
        self._cfg_version = current
        log.info(f"[FastTrackThread] Hot-reload v{old_version}→v{current}")
        return True