# threads/fast_track_thread.py
import cv2
import time
import threading
import math
import logging
import numpy as np

from typing import List, Optional
from dataclasses import dataclass
from config import cfg as _global_cfg
from core.mask import FastMaskView
from detection.detect import ncc_match
from core.optical_flow import of_track
from threads.fast_track_config import FastTrackConfig
from bench import bench
from bench.lifecycle import LifecycleEvent

log = logging.getLogger("threads.fast_track_thread")

OF_FAIL_STREAK_STALE = 5
OF_EXTRAP_MAX_STREAK = 3
OF_EXTRAP_MAX_V_PX_PER_S = 2000.0

class _FastLifecyclePayload:
    """
    Thin wrapper around a FastMaskView so bench.emit_lifecycle() can read
    all required attributes plus the per-source counters (cumul=0 for fast).

    LifecycleEvent values: CREATED, CONFIRMED, LOST, REVIVE, EXPIRED, EVICTED
    """
    def __init__(self, view: FastMaskView, source: str = "fast"):
        self.uid                = view.uid
        self.state              = view.state
        self.rect               = view.rect
        self.confidence         = view.confidence
        self.frame_id           = view.frame_id
        self.total_matches_cumul = 0   # fast thread doesn't track cumul
        self.frames_matched     = 0    # fast thread doesn't track per-frame
        self.last_source             = source

class FastTrackThread:
    """
    Reçoit la frame courante + les FastMaskView actives.
    Tracking OF → NCC confirme → fallback stale si échec.
    F1 : le worker est piloté par Event — traite uniquement quand une nouvelle frame est déposée.
    """

    def __init__(self, config: Optional[FastTrackConfig] = None, registry=None):
        self.cfg = config or FastTrackConfig()
        self.registry = registry
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
        self._last_known = {} # {uid: {"rect": rect, "stale": int, "ts": float, "vx_f": float, "vy_f": float}}
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
            self._latest_frame = frame
            #self._latest_frame = frame.copy()
            self._latest_frame_ts = frame_ts
            self._latest_views = list(views)
        bench.count("F_frame_deposited_total")
        # A5-1 + R2-B: reset_from_snapshot called BEFORE worker triggers
        self.reset_from_snapshot(views)
        self._new_frame_event.set()

    def reset_from_snapshot(self, views: List[FastMaskView]):
        for v in views:
            self._last_known[v.uid] = {
                "rect": v.rect,
                "stale": 0,
                "of_fail_streak": 0,
                "bypass_streak": 0,
                "ts": self._latest_frame_ts if self._latest_frame_ts else time.perf_counter(),
                "vx_f": v.vx_f if v.fast_kin is not None else 0.0,
                "vy_f": v.vy_f if v.fast_kin is not None else 0.0,
            }
        # Purge uids no longer in the slow snapshot
        snapshot_ids = {v.uid for v in views}
        for uid in list(self._last_known.keys()):
            if uid not in snapshot_ids:
                del self._last_known[uid]

    def get_results(self):
        with self._results_lock:
            return self._result_version, list(self._results), self._results_ts

    # ──────────── NCC sur ROI ────────────
    def _adaptive_margin(self, mask: FastMaskView, now: float, snap: FastTrackConfig) -> int:
        if mask.fast_kin is not None and mask.fast_kin.last_fast_ts > 0.0:
            vx, vy = mask.fast_kin.vx_f, mask.fast_kin.vy_f
        else:
            vx, vy = 0.0, 0.0
        speed = math.sqrt(mask.vx * mask.vx + mask.vy * mask.vy)
        dt = now - mask.last_detected_ts
        if dt < 0:
            dt = 0
        margin = snap.am_base + speed * dt * snap.am_factor
        return int(max(snap.am_min, min(margin, snap.am_max)))

    def _ncc_on_roi(self, gray, rect, template, snap: FastTrackConfig, margin: int):
        x, y, w, h = (int(v) for v in rect)
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
        new_rect = (float(rx) + dx, float(ry) + dy, float(w), float(h))
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
                bench.gauge("F_snap._ncc_v_gate", snap.ncc_v_gate)
                bench.gauge("F_snap._ncc_refresh_gate", snap.ncc_refresh_gate)
                # ── 1. Récupérer frame + views ──
                with self._frame_lock:
                    frame = self._latest_frame
                    frame_ts = self._latest_frame_ts
                    views = self._latest_views
                if frame is None or not views:
                    self._prev_gray = None
                    continue
                if frame_ts <= self._last_processed_ts:
                    bench.count("F_frame_dropped_dedup_total")
                    continue
                self._last_processed_ts = frame_ts
                # Lag = délai entre dépôt de la frame et début effectif du traitement.
                lag_ms = (time.perf_counter() - frame_ts) * 1000.0
                bench.probe("F_wakeup_lag_ms", lag_ms)
                # ── Tick complet ──
                with bench.timer("fast_tick_ms"):
                    # ── 2. Convertir en gris ──
                    with bench.timer("fast_cvt_ms"):
                        curr_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    # ── 3. Pas de prev_gray → init ──
                    if self._prev_gray is None:
                        self._prev_gray = curr_gray
                        results = []
                        for v in views:
                            self._last_known[v.uid] = {"rect": v.rect, "stale": 0, "ts": frame_ts}
                            results.append((v.uid, v.rect, 1.0,0.0,0.0))
                        with self._results_lock:
                            self._results = results
                            self._results_ts = frame_ts
                            self._result_version += 1
                        continue
                    # ── 4. Tracking par view ──
                    results = []
                    active_ids = set()
                    max_stale = snap.max_stale_frames
                    # Stockage temporaire pour séparer phases OF / NCC dans les sondes
                    of_outcomes = []
                    # ── 4a. Phase OF (toutes les views) ──
                    with bench.timer("fast_of_total_ms"):
                        for v in views:
                            bench.count("F_mask_of_processed_total")
                            active_ids.add(v.uid)
                            if v.uid not in self._last_known:
                                self._last_known[v.uid] = {"rect": v.rect, "stale": 0, "ts": frame_ts}
                            last_state = self._last_known[v.uid]
                            candidate_rect, of_succeeded, dx_of, dy_of, of_stats = of_track(self._prev_gray, curr_gray, last_state["rect"])

                            # ── Instrumentation calage _GRID_N / _FB_MAX_ERR / _SCALE_CLAMP ──
                            if of_stats is not None:
                                bench.note("F_of_good_points_pct", of_stats["good_pct"])
                                bench.note("F_of_fb_err_med", of_stats["fb_err_med"])
                                bench.note("F_of_scale_raw", of_stats["scale_raw"])

                            # H6 PATCH 1a + 2: roi_source + gauges streak inconditionnels
                            roi_src = "of_ok" if of_succeeded else "fallback"

                            if of_succeeded:
                                last_state["of_fail_streak"] = 0
                                _dt_of = frame_ts - last_state.get("ts", frame_ts)
                                if _dt_of > 1e-6:
                                    _raw_vx = dx_of / _dt_of
                                    _raw_vy = dy_of / _dt_of
                                    last_state["vx_of"] = max(-OF_EXTRAP_MAX_V_PX_PER_S,min(_raw_vx, OF_EXTRAP_MAX_V_PX_PER_S))
                                    last_state["vy_of"] = max(-OF_EXTRAP_MAX_V_PX_PER_S,min(_raw_vy, OF_EXTRAP_MAX_V_PX_PER_S))
                            else:
                                # H6 Fix 2: per-UID consecutive OF-failure counter ──
                                last_state["of_fail_streak"] = last_state.get("of_fail_streak", 0) + 1
                                if last_state.get("of_fail_streak", 0) <= OF_EXTRAP_MAX_STREAK:
                                    _vx_of = last_state.get("vx_of", 0.0)
                                    _vy_of = last_state.get("vy_of", 0.0)
                                    _dt_of = frame_ts - last_state.get("ts", frame_ts)
                                    if abs(_vx_of) > 1e-9 or abs(_vy_of) > 1e-9:
                                        shift_x = _vx_of * _dt_of
                                        shift_y = _vy_of * _dt_of
                                        last_rect = last_state.get("rect", candidate_rect)
                                        candidate_rect = (last_rect[0] + shift_x,last_rect[1] + shift_y,last_rect[2], last_rect[3])
                                    else:
                                        bench.note("F_extrapolation_px", 0.0)
                                else:
                                    candidate_rect = last_state["rect"]
                            of_outcomes.append((v, last_state, candidate_rect, of_succeeded))
                    # ── 4b. Phase NCC + fallback stale ──
                    with bench.timer("fast_ncc_total_ms"):
                        for v, last_state, candidate_rect, of_succeeded in of_outcomes:
                            if last_state.get("of_fail_streak", 0) >= OF_FAIL_STREAK_STALE:
                                last_state["stale"] += 1
                                if last_state["stale"] > max_stale:
                                    bench.emit_lifecycle(LifecycleEvent.LOST,_FastLifecyclePayload(v, source="fast"),reason="of_streak_expire")
                                else:
                                    bench.count("F_of_streak_skip_total")
                                    bench.note("F_of_streak_skip_uid", v.uid)
                                    bench.note("F_of_fail_streak_at_skip", last_state.get("of_fail_streak", 0))
                                continue  # skip NCC; next UID
                            ncc_bypassed = False
                            bench.count("F_mask_ncc_processed_total")
                            if v.template is not None:
                                if v.template_capture_ts > 0:
                                    _tpl_age_ms = (frame_ts - v.template_capture_ts) * 1000.0
                                else:
                                    _tpl_age_ms = float("inf")
                                _of_healthy = of_succeeded and last_state.get("of_fail_streak", 0) == 0
                                _needs_resync = _tpl_age_ms >= snap.template_refresh_ms
                                if _of_healthy and not _needs_resync:
                                    # Bypass : on fait confiance au rect OF, aucun matchTemplate.
                                    ncc_bypassed = True
                                    bench.count("F_ncc_bypass_total")
                                    bench.note("F_bypass_last_uid", v.uid)
                                    ncc_rect, score = None, 0.0
                                else:
                                    with bench.timer("fast_margin_ms"):
                                        margin = self._adaptive_margin(v, frame_ts, snap)
                                    with bench.timer("fast_ncc_call_ms"):
                                        ncc_rect, score = self._ncc_on_roi(curr_gray, candidate_rect, v.template, snap, margin=margin)
                                    if v.template_capture_ts > 0:
                                        bench.probe("F_template_age_ms", _tpl_age_ms)
                            else:
                                ncc_rect, score = None, 0.0
                            if ncc_rect is not None:
                                # ── Vélocité inter-tick : centre(ncc_rect) − centre(last_state["rect"]) / dt ──
                                prev_ts = last_state.get("ts", frame_ts)
                                dt = frame_ts - prev_ts
                                if dt > 1e-6:
                                    dx = ncc_rect[0] - last_state["rect"][0]
                                    dy = ncc_rect[1] - last_state["rect"][1]
                                    vx_f = max(-snap.max_v_px_per_s, min(dx / dt, snap.max_v_px_per_s))
                                    vy_f = max(-snap.max_v_px_per_s, min(dy / dt, snap.max_v_px_per_s))
                                else:
                                    vx_f = 0.0
                                    vy_f = 0.0
                                ncc_v_gate = snap.ncc_v_gate if hasattr(snap, 'ncc_v_gate') else 0.55
                                ncc_refresh_gate = snap.ncc_refresh_gate if hasattr(snap, 'ncc_refresh_gate') else 0.40
                                #log.info(f"uid={v.uid} score={score} vx={vx_f} vy={vy_f} ncc_v_gate={ncc_v_gate}")
                                if score >= ncc_v_gate:
                                    last_state["vx_f"] = vx_f
                                    last_state["vy_f"] = vy_f
                                if v.template_capture_ts > 0:
                                    template_age_ms = (frame_ts - v.template_capture_ts) * 1000.0
                                    _refresh_nominal = score >= ncc_refresh_gate and template_age_ms >= snap.template_refresh_ms
                                    _refresh_forced  = template_age_ms > 1.7 * snap.template_refresh_ms
                                    if _refresh_nominal or _refresh_forced:
                                        tx, ty, tw, th = (int(c) for c in ncc_rect)
                                        tx0 = max(tx, 0); ty0 = max(ty, 0)
                                        tx1 = min(tx + tw, snap.screen_w)
                                        ty1 = min(ty + th, snap.screen_h)
                                        if tx1 > tx0 and ty1 > ty0:
                                            v.template = curr_gray[ty0:ty1, tx0:tx1].copy()
                                            v.template_capture_ts = frame_ts
                                last_state["rect"] = ncc_rect
                                last_state["stale"] = 0
                                last_state["bypass_streak"] = 0
                                last_state["ts"]   = frame_ts
                                results.append((v.uid, ncc_rect, score,last_state.get("vx_f", 0.0),last_state.get("vy_f", 0.0)))
                                v.current_rect = ncc_rect
                                # ── Lifecycle: NCC a confirmé le mask (source=fast) ──
                                bench.emit_lifecycle(LifecycleEvent.CONFIRMED, _FastLifecyclePayload(v, source="fast"), reason="ncc_fast")
                            elif ncc_bypassed:
                                last_state["bypass_streak"] = last_state.get("bypass_streak", 0) + 1
                                bench.count("F_bypass_consecutive_total")
                                bench.note("F_bypass_streak_at_publish", last_state["bypass_streak"])
                                plop = last_state["bypass_streak"]
                                #log.info(f"last_state[bypass_streak]={plop} snap.bypass_streak_max={snap.bypass_streak_max}")
                                if last_state["bypass_streak"] >= snap.bypass_streak_max:
                                    if self.registry is not None:
                                        self.registry.force_lost(v.uid, frame_ts, reason="bypass_streak_exhausted")
                                    bench.count("F_bypass_forced_lost_total")
                                    continue
                                # PATCH 4 : chemin bypass — propagation vitesse OF
                                last_state["vx_f"] = last_state.get("vx_of", last_state.get("vx_f", 0.0))
                                last_state["vy_f"] = last_state.get("vy_of", last_state.get("vy_f", 0.0))
                                last_state["rect"] = candidate_rect
                                last_state["stale"] = 0
                                last_state["ts"]    = frame_ts
                                results.append((v.uid, candidate_rect, 1.0,last_state.get("vx_f", 0.0), last_state.get("vy_f", 0.0)))
                                v.current_rect = candidate_rect
                                bench.emit_lifecycle(LifecycleEvent.CONFIRMED, _FastLifecyclePayload(v, source="fast"), reason="of_bypass")
                            else:
                                if (v.template is not None and v.template_capture_ts > 0
                                        and of_succeeded and last_state.get("of_fail_streak", 0) == 0):
                                    _forced_age_ms = (frame_ts - v.template_capture_ts) * 1000.0
                                    if _forced_age_ms > 1.7 * snap.template_refresh_ms:
                                        fx, fy, fw, fh = (int(c) for c in candidate_rect)
                                        fx0 = max(fx, 0); fy0 = max(fy, 0)
                                        fx1 = min(fx + fw, snap.screen_w)
                                        fy1 = min(fy + fh, snap.screen_h)
                                        if fx1 > fx0 and fy1 > fy0:
                                            v.template = curr_gray[fy0:fy1, fx0:fx1].copy()
                                            v.template_capture_ts = frame_ts
                                last_state["stale"] += 1
                                if last_state["stale"] > max_stale:
                                    # ── Lifecycle: mask expiré en stale (source=fast) ──
                                    bench.emit_lifecycle(LifecycleEvent.LOST, _FastLifecyclePayload(v, source="fast"), reason="stale_expire")
                                else:
                                    bench.count("F_ncc_stale_skipped_total")
                                    bench.note("F_ncc_stale_skip_uid", v.uid)
                                    bench.note("F_ncc_score_at_skip", score)
                                    if v.template is not None and v.template_capture_ts > 0:
                                        bench.note("F_ncc_template_age_at_skip", (frame_ts - v.template_capture_ts) * 1000.0)
                                    bench.note("F_roi_source_at_skip", int(of_succeeded and last_state.get("of_fail_streak", 0) == 0))
                                    # σ texture du patch candidate_rect (curr_gray) + magnitude mouvement OF
                                    _sx, _sy, _sw, _sh = (int(c) for c in candidate_rect)
                                    _sx0 = max(_sx, 0); _sy0 = max(_sy, 0)
                                    _sx1 = min(_sx + _sw, snap.screen_w)
                                    _sy1 = min(_sy + _sh, snap.screen_h)
                                    if _sx1 > _sx0 and _sy1 > _sy0:
                                        bench.note("F_roi_texture_std_at_skip", float(curr_gray[_sy0:_sy1, _sx0:_sx1].std()))
                                    bench.note("F_roi_motion_px_s_at_skip", math.hypot(last_state.get("vx_of", 0.0), last_state.get("vy_of", 0.0)))
                                    _last_rect = last_state.get("rect", candidate_rect)
                                    _dc_x = (candidate_rect[0] + candidate_rect[2] / 2.0) - (_last_rect[0] + _last_rect[2] / 2.0)
                                    _dc_y = (candidate_rect[1] + candidate_rect[3] / 2.0) - (_last_rect[1] + _last_rect[3] / 2.0)
                                    _drift_px = math.hypot(_dc_x, _dc_y)
                                    bench.note("F_roi_drift_px_at_skip", _drift_px)
                                    if (last_state.get("bypass_streak", 0) == 0 and _drift_px > snap.drift_sync_threshold_px):
                                        _cx, _cy, _cw, _ch = (int(c) for c in candidate_rect)
                                        fx0 = max(_cx, 0); fy0 = max(_cy, 0)
                                        fx1 = min(_cx + _cw, snap.screen_w)
                                        fy1 = min(_cy + _ch, snap.screen_h)
                                        if fx1 > fx0 and fy1 > fy0:
                                            v.template = curr_gray[fy0:fy1, fx0:fx1].copy()
                                            v.template_capture_ts = frame_ts
                                            bench.count("F_ncc_drift_resync_total")
                    # ── 5. Purger masques disparus ──
                    with bench.timer("fast_publish_ms"):
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