# main.py — v9 (timestamp-based prediction)
import logging
from config import cfg

def setup_logging():
    level_str = cfg.get("debug.log_level", "WARNING")
    level = getattr(logging, level_str.upper(), logging.WARNING)

    logging.basicConfig(
        level=level,
        format="%(name)s | %(levelname)s | %(message)s"
    )
setup_logging()

import time
import cv2
import numpy as np
import pyvirtualcam

from capture_thread import CaptureThread
from detect_thread import DetectThread
from fast_track_thread import FastTrackThread          # ← AJOUT
from send_thread import SendThread
from detect_stats import get_stats as detect_stats
from detect_stats import reset_stats as detect_diag_reset
from blur import apply_blur
from blur import get_stats as blur_stats, reset_stats as blur_reset

# ── PARAMÈTRES — lus depuis config.yaml ──
SCREEN_WIDTH  = cfg.get("screen.width")
SCREEN_HEIGHT = cfg.get("screen.height")
CAPTURE_FPS   = cfg.get("screen.capture_fps")
VCAM_FPS      = cfg.get("screen.vcam_fps")
cfg.start_watcher()

_next_mask_id = 0
# ═══════════════════════════════════════════════════════
#  UTILITAIRES TRACKING
# ═══════════════════════════════════════════════════════

def ttl_color(ttl):
    if ttl >= 3:
        return tuple(cfg.get("debug.colors.fresh"))
    elif ttl >= 2:
        return tuple(cfg.get("debug.colors.persist"))
    else:
        return tuple(cfg.get("debug.colors.dying"))

def ttl_label(ttl, source="S"):                        # ← MODIFIÉ
    return f"TTL={ttl} [{source}]"

def compute_iou(r1, r2):
    """IoU (Intersection over Union) entre deux rectangles (x,y,w,h)."""
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def center_distance(r1, r2):
    """Distance euclidienne entre les centres de deux rectangles (x,y,w,h)."""
    cx1 = r1[0] + r1[2] / 2
    cy1 = r1[1] + r1[3] / 2
    cx2 = r2[0] + r2[2] / 2
    cy2 = r2[1] + r2[3] / 2
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

def update_mask(mask, new_rect, detect_ts, source):
    """Met à jour un masque existant : dead zone, smooth, vélocité filtrée, TTL."""
    smooth_alpha = cfg.get("masks.smooth_alpha", 1.0)
    dead_zone    = cfg.get("masks.dead_zone", 3)
    vel_dz       = cfg.get("masks.velocity_dead_zone", 5)

    nx, ny, nw, nh = new_rect
    ox, oy, ow, oh = mask['rect']

    # ── Dead zone : micro-mouvement → juste rafraîchir TTL ──
    if (abs(nx - ox) < dead_zone and abs(ny - oy) < dead_zone and abs(nw - ow) < dead_zone and abs(nh - oh) < dead_zone):
        mask['ttl'] = cfg.get("masks.ttl_max")
        mask['last_detected_ts'] = detect_ts
        mask['last_source'] = source
        return

    # ── Vélocité filtrée ──
    lx, ly, _, _ = mask['last_detected_rect']
    dt = detect_ts - mask['last_detected_ts']
    if dt > 0.001:
        mask['vx'] = 0.0 if abs(nx - lx) < vel_dz else (nx - lx) / dt
        mask['vy'] = 0.0 if abs(ny - ly) < vel_dz else (ny - ly) / dt

    # ── Smooth EMA ──
    a = smooth_alpha
    mask['rect'] = (
        ox + a * (nx - ox),
        oy + a * (ny - oy),
        ow + a * (nw - ow),
        oh + a * (nh - oh),
    )
    mask['last_detected_rect'] = (float(nx), float(ny), float(nw), float(nh))
    mask['last_detected_ts'] = detect_ts
    mask['ttl'] = cfg.get("masks.ttl_max")
    mask['last_source'] = source



def match_or_add(active_masks, new_rect, detect_ts, source="slow"):
    global _next_mask_id
    best_idx = -1
    nx, ny, nw, nh = new_rect

    ttl_max     = cfg.get("masks.ttl_max")
    iou_thresh  = cfg.get("matching.iou_thresh")
    dist_thresh = cfg.get("matching.dist_thresh")
    max_masks   = cfg.get("masks.max_masks")
    mode        = cfg.get("matching.mode")

    if mode == "distance":
        best_dist = float('inf')
        for i, m in enumerate(active_masks):
            d = center_distance(m['rect'], new_rect)
            if d < best_dist:
                best_dist = d
                best_idx = i
        if best_dist > dist_thresh:
            best_idx = -1
    else:
        best_score = 0.0
        for i, m in enumerate(active_masks):
            score = compute_iou(m['rect'], new_rect)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_score < iou_thresh:
            best_idx = -1

    if best_idx >= 0:
        update_mask(active_masks[best_idx], new_rect, detect_ts, source)
        return active_masks[best_idx]['uid']

    if len(active_masks) >= max_masks:
        return None

    _next_mask_id += 1
    active_masks.append({
        'uid':                 _next_mask_id,
        'rect':                (float(nx), float(ny), float(nw), float(nh)),
        'ttl':                 ttl_max,
        'last_detected_rect':  (float(nx), float(ny), float(nw), float(nh)),
        'last_detected_ts': detect_ts,
        'last_source':         source,
        'vx':                  0.0,
        'vy':                  0.0,
    })
    return _next_mask_id

def pad_rect(x, y, w, h, max_w, max_h):
    return (
        max(x, 0),
        max(y, 0),
        min(w, max_w - max(x, 0)),
        min(h, max_h - max(y, 0)),
    )

def draw_debug(frame, active_masks):
    for m in active_masks:
        x, y, w, h = (int(v) for v in m['rect'])
        ttl = m['ttl']
        color = ttl_color(ttl)
        thickness = 2 if ttl >= 3 else 1

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

        source = m.get('last_source', '?')[0].upper()
        label = ttl_label(ttl, source)
        label_y = max(y - 6, 14)
        cv2.putText(frame, label, (x, label_y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    total = len(active_masks)
    cv2.putText(frame, f"Masks: {total}", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


# ═══════════════════════════════════════════════════════
#  STATS / BENCHMARK
# ═══════════════════════════════════════════════════════

_main_stats = {
    "loop_ms":       0.0,
    "total_frames":  0,
    "mask_peak":     0,
    "slow_updates":  0,
    "fast_updates":  0,
    "predict_count": 0,
}


def print_all_stats():
    matching_mode = cfg.get("matching.mode")
    n = max(_main_stats["total_frames"], 1)
    cs = capturer.get_stats()
    ds = detector.get_stats()
    bs = blur_stats()
    ss = sender.get_stats()
    dd = detect_stats()

    print("\n" + "=" * 55)
    print("        BENCHMARK PIPELINE v8.0 (dual detect)")
    print("=" * 55)
    print(f"\n  📷 CAPTURE (thread @ {CAPTURE_FPS}fps cible)")
    for k, v in cs.items():
        print(f"    {k:22s} : {v}")

    print(f"\n  🔍 SLOW DETECT (thread)")
    for k, v in ds.items():
        print(f"    {k:22s} : {v}")

    # ── AJOUT : section fast ──
    if fast_enabled:
        fs = fast_tracker.get_stats()
        print(f"\n  ⚡ FAST TRACK (ROI)")
        for k, v in fs.items():
            print(f"    {k:22s} : {v}")

    print(f"\n  🌀 BLUR + CVT (fusionnés)")
    for k, v in bs.items():
        print(f"    {k:22s} : {v}")

    print(f"\n  📤 SEND (thread, vcam @ {VCAM_FPS}fps déclaré)")
    for k, v in ss.items():
        print(f"    {k:22s} : {v}")

    print(f"\n  🔬 DETECT DIAGNOSTIC")
    for k, v in dd.items():
        print(f"    {k:22s} : {v}")

    print(f"\n  🎬 MAIN LOOP")
    loop_avg = round(_main_stats['loop_ms'] / n, 2)
    print(f"    {'loop_avg_ms':22s} : {loop_avg}")
    print(f"    {'total_frames':22s} : {_main_stats['total_frames']}")
    print(f"    {'mask_peak':22s} : {_main_stats['mask_peak']}")
    print(f"    {'slow_updates':22s} : {_main_stats['slow_updates']}")
    print(f"    {'fast_updates':22s} : {_main_stats['fast_updates']}")
    print(f"    {'predict_count':22s} : {_main_stats['predict_count']}")
    print(f"    {'ttl_max':22s} : {cfg.get('masks.ttl_max')}")
    print(f"    {'matching_mode':22s} : {matching_mode}")
    if matching_mode == "distance":
        print(f"    {'dist_thresh':22s} : {cfg.get('matching.dist_thresh')}")
    else:
        print(f"    {'iou_thresh':22s} : {cfg.get('matching.iou_thresh')}")

    old_loop = 32.66
    new_loop = loop_avg
    saved = round(old_loop - new_loop, 2)
    old_fps = round(1000 / old_loop, 1)
    new_fps = round(1000 / max(new_loop, 0.01), 1)

    print(f"\n  📉 GAIN vs v5.1")
    print(f"    {'v5.1 loop_avg':22s} : {old_loop} ms ({old_fps} FPS)")
    print(f"    {'v8.0 loop_avg':22s} : {new_loop} ms ({new_fps} FPS)")
    print(f"    {'économisé':22s} : {saved} ms/frame")
    print(f"    {'accélération':22s} : x{round(old_loop / max(new_loop, 0.01), 2)}")
    print("=" * 55)


# ═══════════════════════════════════════════════════════
#  LANCEMENT
# ═══════════════════════════════════════════════════════

capturer = CaptureThread(target_fps=CAPTURE_FPS)
capturer.start()

detector = DetectThread()
detector.start()

fast_enabled = cfg.get("detect.fast.enabled", True)
if fast_enabled:
    fast_tracker = FastTrackThread(SCREEN_WIDTH, SCREEN_HEIGHT)
    fast_tracker.start()

with pyvirtualcam.Camera(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, fps=VCAM_FPS) as vcam:
    print(f"✅ Caméra virtuelle prête → {vcam.device}")
    debug_draw = cfg.get("debug.draw")
    predict = cfg.get("predict.active", True)

    if fast_enabled:
        print("⚡ FAST TRACKING ACTIVÉ")
    print("📸 En cours... (Ctrl+C pour arrêter)")

    sender = SendThread(vcam, SCREEN_WIDTH, SCREEN_HEIGHT)
    sender.start()

    fps_timer = time.time()
    frame_count = 0

    try:
        active_masks = []
        last_detect_version = 0
        last_fast_version = 0
        last_frame_id = 0

        capturer.reset_stats()
        detector.reset_stats()
        if fast_enabled:
            fast_tracker.reset_stats()
        detect_diag_reset()
        blur_reset()
        sender.reset_stats()

        while True:
            t_loop = time.perf_counter()
            now = time.perf_counter()

            # ── 1. Capture (NON BLOQUANT) ──
            frame, frame_ts  = capturer.get_frame()
            if frame is None:
                time.sleep(0.001)
                continue

            # ── 2. Ne donner au detect QUE si frame nouvelle ──
            frame_id = capturer.get_frame_id()
            if frame_id > last_frame_id:
                last_frame_id = frame_id
                detector.give_frame(frame,frame_ts)
                # ── AJOUT : alimenter le fast tracker ──
                if fast_enabled and active_masks:
                    fast_tracker.give_frame_and_masks(frame, active_masks,frame_ts)

            updated_uids = set()
            # ── 3. Slow : nouvelle détection disponible ? ──
            current_version = detector.get_detect_count()
            slow_updated = False
            if current_version > last_detect_version:
                slow_updated = True
                for m in active_masks:
                    m['ttl'] -= 1
                active_masks = [m for m in active_masks if m['ttl'] > 0]

                last_detect_version = current_version
                _main_stats["slow_updates"] += 1         # ← AJOUT
                new_plates, detect_ts = detector.get_zones()
                for p in new_plates:
                    x, y, w, h = p
                    padded = pad_rect(x, y, w, h, SCREEN_WIDTH, SCREEN_HEIGHT)
                    uid = match_or_add(active_masks, padded, detect_ts, source="slow")  # ← MODIFIÉ
                    if uid is not None:
                        updated_uids.add(uid)

            # ── 3b. Fast : résultats ROI disponibles ? ──  # ← BLOC AJOUTÉ
            if fast_enabled and not slow_updated:
                fast_version, fast_results, fast_ts = fast_tracker.get_results()
                if fast_version > last_fast_version:
                    last_fast_version = fast_version
                    _main_stats["fast_updates"] += 1
                    found_uids = set()

                    for mask_uid, new_rect in fast_results:
                        if new_rect is not None:
                            found_uids.add(mask_uid)


                    tracked_uids = {r[0] for r in fast_results}
                    for m in active_masks:
                        if m['uid'] in tracked_uids and m['uid'] not in found_uids:
                            m['ttl'] -= 1

                    # Purger les morts
                    active_masks = [m for m in active_masks if m['ttl'] > 0]

                    for mask_uid, new_rect in fast_results:
                        if new_rect is None:
                            continue
                        # Cibler directement le masque par uid
                        for m in active_masks:
                            if m['uid'] == mask_uid:
                                padded = pad_rect(*new_rect, SCREEN_WIDTH, SCREEN_HEIGHT)
                                update_mask(m, padded, fast_ts, source="fast")
                                updated_uids.add(mask_uid)
                                break


            # ── 4. Predict les masques NON mis à jour (en secondes) ──
            if predict:
                for m in active_masks:
                    if m['uid'] not in updated_uids:
                        dt = now - m['last_detected_ts']       # secondes depuis dernière détection
                        # ── CAP : ne pas prédire au-delà de 200ms ──
                        dt_capped = min(dt, 0.10)
                        # ── DAMPING : confiance décroissante ──
                        damping = max(0.0, 1.0 - dt * 2.0)
                        # Prédiction = dernière position détectée + vélocité × temps écoulé
                        lx, ly, lw, lh = m['last_detected_rect']
                        w, h = m['rect'][2], m['rect'][3]
                        x = lx + m['vx'] * dt_capped * damping
                        y = ly + m['vy'] * dt_capped * damping
                        # Clamp aux bords écran
                        x = max(0.0, min(x, SCREEN_WIDTH - w))
                        y = max(0.0, min(y, SCREEN_HEIGHT - h))
                        m['rect'] = (x, y, w, h)
                        _main_stats["predict_count"] += 1

            # ── 5. Cap max masques ──
            if len(active_masks) > cfg.get("masks.max_masks"):
                active_masks.sort(key=lambda m: m['ttl'], reverse=True)
                active_masks = active_masks[:cfg.get("masks.max_masks")]

            # ── 6. Blur ou debug ──
            blur_zones = [
                (int(m['rect'][0]), int(m['rect'][1]),
                 int(m['rect'][2]), int(m['rect'][3]))
                for m in active_masks
            ]

            # ── 7. Envoi vers OBS (zéro copie) ──
            buf = sender.borrow()
            np.copyto(buf, frame)

            if debug_draw:
                apply_blur(buf, blur_zones)
                draw_debug(buf, active_masks)
            else:
                apply_blur(buf, blur_zones)

            sender.publish()

            # ── 8. Stats loop ──
            _main_stats["loop_ms"] += (time.perf_counter() - t_loop) * 1000
            _main_stats["total_frames"] += 1
            _main_stats["mask_peak"] = max(
                _main_stats["mask_peak"], len(active_masks)
            )

            # ── 9. FPS counter ──
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 2.0:
                fps = frame_count / elapsed
                mode = "DEBUG" if debug_draw else "PROD"
                fast_tag = "+FAST" if fast_enabled else ""
                print(f"⚡ {fps:.1f} FPS | {len(active_masks)} masque(s) | {mode} {fast_tag}")
                frame_count = 0
                fps_timer = time.time()

    except KeyboardInterrupt:
        print("\n🛑 Arrêt propre")
        print_all_stats()

    finally:
        sender.stop()
        if fast_enabled:
            fast_tracker.stop()
        detector.stop()
        capturer.stop()
