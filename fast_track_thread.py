# fast_track_thread.py
import cv2
import threading
import time
import numpy as np
from config import cfg
from detect import ncc_match
from box import Box
from optical_flow import of_track

_MAX_STALE_FRAMES = cfg.get("detect.fast.max_stale_frames", 15)


class FastTrackThread:
    """
    Reçoit la frame courante + les masques actifs.
    Tracking OF → NCC confirme → Option B si échec.
    """

    def __init__(self, screen_width, screen_height):
        self._screen_w = screen_width
        self._screen_h = screen_height

        self._latest_frame = None
        self._latest_frame_ts = 0.0
        self._latest_masks = []
        self._frame_lock = threading.Lock()

        self._results = []
        self._results_ts = 0.0
        self._result_version = 0
        self._results_lock = threading.Lock()

        self._prev_gray = None
        self._last_known = {}

        self._stats_lock = threading.Lock()
        self._total_ms = 0.0
        self._track_count = 0
        self._roi_count = 0
        self._found_count = 0
        self._of_ok_count = 0
        self._of_fb_count = 0

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
        if self._thread:
            self._thread.join(timeout=1.0)
        print("[FastTrackThread] Arrêté")

    # ──────────── Interface ────────────

    def give_frame_and_masks(self,frame, masks, frame_ts):
        with self._frame_lock:
            self._latest_frame = frame
            self._latest_frame_ts = frame_ts
            self._latest_masks = list(masks)

    def get_results(self):
        with self._results_lock:
            return self._result_version, list(self._results), self._results_ts

    def get_stats(self):
        with self._stats_lock:
            count = self._track_count
            if count == 0:
                return {
                    "avg_ms": 0.0,
                    "count": 0,
                    "roi_total": 0,
                    "found_total": 0,
                    "of_ok_rate": 0.0,
                    "of_fb_rate": 0.0,
                }
            return {
                "avg_ms": round(self._total_ms / count, 2),
                "count": count,
                "roi_total": self._roi_count,
                "found_total": self._found_count,
                "of_ok_rate": round(self._of_ok_count / max(self._roi_count, 1), 3),
                "of_fb_rate": round(self._of_fb_count / max(self._roi_count, 1), 3),
            }

    def reset_stats(self):
        with self._stats_lock:
            self._total_ms    = 0.0
            self._track_count = 0
            self._roi_count   = 0
            self._found_count = 0
            self._of_ok_count = 0
            self._of_fb_count = 0
    # ──────────── NCC sur ROI ────────────

    def _ncc_on_roi(self, gray, rect, template):
        x, y, w, h = rect
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
        interval = 1.0 / cfg.get("detect.fast.fps", 60)
        max_stale = _MAX_STALE_FRAMES

        while self._running:
            time.sleep(interval)

            # ── 1. Récupérer frame + masques ──
            with self._frame_lock:
                frame = self._latest_frame
                frame_ts = self._latest_frame_ts
                masks = self._latest_masks

            if frame is None or not masks:
                self._prev_gray = None
                continue

            t0 = time.perf_counter()

            # ── 2. Convertir en gris ──
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── 3. Pas de prev_gray → initialiser et publier positions actuelles ──
            if self._prev_gray is None:
                self._prev_gray = curr_gray
                results = []
                for m in masks:
                    uid = m["uid"]
                    rect = m["rect"]
                    template = m.get("template")
                    self._last_known[uid] = {"rect": rect, "stale": 0}
                    results.append((uid, rect, 1.0))
                with self._results_lock:
                    self._results = results
                    self._results_ts = frame_ts
                    self._result_version += 1
                continue

            # ── 4. Tracking par masque ──
            results = []
            active_ids = set()
            roi_count = 0
            found_count = 0
            of_ok = 0
            of_fb = 0

            for m in masks:
                uid = m["uid"]
                rect = m["rect"]
                template = m.get("template")
                active_ids.add(uid)

                # Initialiser last_known si nouveau masque
                if uid not in self._last_known:
                    self._last_known[uid] = {"rect": rect, "stale": 0}

                last_state = self._last_known[uid]

                # ── 4a. OF ──
                candidate_rect, of_succeeded = of_track(self._prev_gray, curr_gray, last_state["rect"])

                if of_succeeded:
                    of_ok += 1
                else:
                    of_fb += 1
                    candidate_rect = last_state["rect"]

                # ── 4b. NCC confirme ──
                roi_count += 1

                if template is not None:
                    ncc_rect, score = self._ncc_on_roi(curr_gray, candidate_rect, template)
                else:
                    ncc_rect, score = None, 0.0

                if ncc_rect is not None:
                    last_state["rect"] = ncc_rect
                    last_state["stale"] = 0
                    found_count += 1
                    results.append((uid, ncc_rect, score))
                else:
                    last_state["stale"] += 1
                    if last_state["stale"] <= max_stale:
                        results.append((uid, last_state["rect"], score))
                    else:
                        results.append((uid, None, score))

            # ── 5. Purger masques disparus ──
            for old_id in list(self._last_known.keys()):
                if old_id not in active_ids:
                    del self._last_known[old_id]

            # ── 6. Mémoriser frame courante ──
            self._prev_gray = curr_gray

            dt = (time.perf_counter() - t0) * 1000

            # ── 7. Publier ──
            with self._results_lock:
                self._results = results
                self._results_ts = frame_ts
                self._result_version += 1

            with self._stats_lock:
                self._total_ms += dt
                self._track_count += 1
                self._roi_count += roi_count
                self._found_count += found_count
                self._of_ok_count += of_ok
                self._of_fb_count += of_fb
