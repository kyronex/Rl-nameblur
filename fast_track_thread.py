# fast_track_thread.py
import threading
import time
from config import cfg
from detect import track_roi_fast


class FastTrackThread:
    """
    Reçoit la frame courante + les masques actifs.
    Crop ROI élargie autour de chaque masque, re-détecte, publie les résultats.
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
        self._results_lock = threading.Lock()
        self._result_version = 0

        self._running = False
        self._thread = None

        # Stats
        self._total_ms = 0.0
        self._track_count = 0
        self._roi_count = 0
        self._found_count = 0
        self._stats_lock = threading.Lock()

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

    def give_frame_and_masks(self, frame, masks, ts):
        frame_copy = frame.copy()
        masks_copy = [
            {"id": m["uid"], "rect": m["rect"]}
            for m in masks
        ]
        with self._frame_lock:
            self._latest_frame = frame_copy
            self._latest_frame_ts = ts
            self._latest_masks = masks_copy

    def get_results(self):
        with self._results_lock:
            return self._result_version, self._results.copy(), self._results_ts

    # ──────────── Stats ────────────

    def get_stats(self):
        with self._stats_lock:
            n = max(self._track_count, 1)
            return {
                "fast_avg_ms":      round(self._total_ms / n, 2),
                "track_count":      self._track_count,
                "roi_per_track":    round(self._roi_count / n, 1),
                "found_per_track":  round(self._found_count / n, 1),
            }

    def reset_stats(self):
        with self._stats_lock:
            self._total_ms = 0.0
            self._track_count = 0
            self._roi_count = 0
            self._found_count = 0

    # ──────────── Helpers ────────────

    def _expand_roi(self, rect):
        """Élargit un rect de roi_margin % de chaque côté, clampé à l'écran."""
        margin = cfg.get("detect.fast.roi_margin", 0.3)
        x, y, w, h = rect

        dx = int(w * margin)
        dy = int(h * margin)

        rx = max(int(x) - dx, 0)
        ry = max(int(y) - dy, 0)
        rx2 = min(int(x + w) + dx, self._screen_w)
        ry2 = min(int(y + h) + dy, self._screen_h)

        return rx, ry, rx2 - rx, ry2 - ry

    def _best_plate(self, plates_frame, original_rect):
        """
        Parmi les plates (déjà en coordonnées frame),
        retourne la plus proche du centre de original_rect.
        Retourne None si trop loin.
        """
        if not plates_frame:
            return None

        ox, oy, ow, oh = original_rect
        ocx = ox + ow / 2
        ocy = oy + oh / 2

        best = None
        best_dist = float("inf")

        for (px, py, pw, ph) in plates_frame:
            pcx = px + pw / 2
            pcy = py + ph / 2
            dist = ((pcx - ocx) ** 2 + (pcy - ocy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best = (px, py, pw, ph)

        max_dist = cfg.get("matching.dist_thresh", 60)
        if best_dist > max_dist:
            return None

        return best

    # ──────────── Worker ────────────

    def _worker(self):
        while self._running:
            # ── Récupérer ET consommer frame + masks ──
            with self._frame_lock:
                frame = self._latest_frame
                frame_ts = self._latest_frame_ts
                masks = self._latest_masks
                self._latest_frame = None

            if frame is None or not masks:
                time.sleep(0.001)
                continue

            t0 = time.perf_counter()
            results = []
            roi_count = 0
            found_count = 0

            for mask_info in masks:
                mask_id = mask_info["id"]
                rect = mask_info["rect"]

                # ── Expand ROI ──
                roi_x, roi_y, roi_w, roi_h = self._expand_roi(rect)
                if roi_w < 10 or roi_h < 10:
                    results.append((mask_id, None))
                    continue

                # ── Crop ──
                roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
                if roi.size == 0:
                    results.append((mask_id, None))
                    continue

                roi_count += 1

                # ── Detect dans le crop ──
                plates_local = track_roi_fast(roi)  # retourne coords relatives au ROI

                # ── Reconvertir en coordonnées frame ──
                plates_frame = [
                    (px + roi_x, py + roi_y, pw, ph)
                    for (px, py, pw, ph) in plates_local
                ]

                # ── Trouver le meilleur match ──
                best = self._best_plate(plates_frame, rect)

                if best is not None:
                    found_count += 1
                    results.append((mask_id, best))
                else:
                    results.append((mask_id, None))

            dt = (time.perf_counter() - t0) * 1000

            # ── Publier résultats ──
            with self._results_lock:
                self._results = results
                self._results_ts = frame_ts
                self._result_version += 1

            with self._stats_lock:
                self._total_ms += dt
                self._track_count += 1
                self._roi_count += roi_count
                self._found_count += found_count
