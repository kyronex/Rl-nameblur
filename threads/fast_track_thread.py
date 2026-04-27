# threads/fast_track_thread.py
import cv2
import threading
import math
from typing import List
from config import cfg
from core.mask import FastMaskView
from detection.detect import ncc_match
from core.optical_flow import of_track

_MAX_STALE_FRAMES = cfg.get("detect.fast.max_stale_frames", 15)

class FastTrackThread:
    """
    Reçoit la frame courante + les masques actifs.
    Tracking OF → NCC confirme → fallback stale si échec.
    F1 : le worker est piloté par Event — traite uniquement
         quand une nouvelle frame est déposée.
    """

    def __init__(self, screen_width, screen_height):
        self._screen_w = screen_width
        self._screen_h = screen_height

        self._latest_frame = None
        self._latest_frame_ts = 0.0
        self._latest_views = []
        self._frame_lock = threading.Lock()

        self._new_frame_event = threading.Event()

        self._results = []
        self._results_ts = 0.0
        self._result_version = 0
        self._results_lock = threading.Lock()

        self._prev_gray = None
        self._last_known = {}
        self._last_processed_ts = -1.0

        self._running = False
        self._thread = None

        self._roi_margin = cfg.get("detect.fast.roi_margin", 30)
        self._ncc_threshold = cfg.get("detect.fast.ncc_threshold", 0.5)

    # ──────────── Contrôle ────────────

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print("[FastTrackThread] Démarré")

    def stop(self):
        self._running = False
        self._new_frame_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        print("[FastTrackThread] Arrêté")

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
    def _adaptive_margin(self, mask, now):
        base   = cfg.get("masks.adaptive_margin.base", 10)
        factor = cfg.get("masks.adaptive_margin.factor", 1.5)
        m_min  = cfg.get("masks.adaptive_margin.min", 10)
        m_max  = cfg.get("masks.adaptive_margin.max", 80)

        speed = math.sqrt(mask.vx * mask.vx + mask.vy * mask.vy)

        dt = now - mask.last_detected_ts
        if dt < 0:
            dt = 0

        margin = base + speed * dt * factor
        return int(max(m_min, min(margin, m_max)))

    def _ncc_on_roi(self, gray, rect, template, margin=None):
        x, y, w, h = rect
        if margin is None:
            margin = self._roi_margin
        rx = max(x - margin, 0)
        ry = max(y - margin, 0)
        rx2 = min(x + w + margin, self._screen_w)
        ry2 = min(y + h + margin, self._screen_h)

        roi = gray[ry:ry2, rx:rx2]
        if roi.size == 0:
            return None, 0.0

        score, loc = ncc_match(roi, template, threshold=self._ncc_threshold)

        if loc is None:
            return None, score

        dx, dy = loc
        new_rect = (rx + dx, ry + dy, template.shape[1], template.shape[0])
        return new_rect, score

    # ──────────── Worker ────────────

    def _worker(self):
        max_stale = _MAX_STALE_FRAMES
        event_timeout = cfg.get("detect.fast.event_timeout_s", 0.5)
        while self._running:
            triggered = self._new_frame_event.wait(timeout=event_timeout)
            if not self._running:
                break
            self._new_frame_event.clear()

            if not triggered:
                continue

            # ── 1. Récupérer frame + view ──
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

            # ── 3. Pas de prev_gray → initialiser et publier positions actuelles ──
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

            for v in views:
                active_ids.add(v.uid)
                # Initialiser last_known si nouveau masque
                if v.uid not in self._last_known:
                    self._last_known[v.uid] = {"rect": v.rect, "stale": 0}

                last_state = self._last_known[v.uid]

                # ── 4a. OF ──
                candidate_rect, of_succeeded = of_track(self._prev_gray, curr_gray, last_state["rect"])

                if not of_succeeded:
                    candidate_rect = last_state["rect"]

                # ── 4b. NCC confirme ──
                if v.template is not None:
                    margin = self._adaptive_margin(v, frame_ts)
                    ncc_rect, score = self._ncc_on_roi(curr_gray, candidate_rect, v.template, margin=margin)
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
