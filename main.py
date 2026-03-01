# main.py — v7 (refacto nommage, zéro changement logique)
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
from send_thread import SendThread
from detect import get_stats as detect_stats
from blur import apply_blur
from blur import get_stats as blur_stats, reset_stats as blur_reset

# ── PARAMÈTRES — lus depuis config.yaml ──
SCREEN_WIDTH  = cfg.get("screen.width")
SCREEN_HEIGHT = cfg.get("screen.height")
CAPTURE_FPS   = cfg.get("screen.capture_fps")
VCAM_FPS      = cfg.get("screen.vcam_fps")

cfg.start_watcher()


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


def ttl_label(ttl):
    return f"TTL={ttl}"


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


def match_or_add(active_masks, new_rect, frame_id):
    best_idx = -1
    nx, ny, nw, nh = new_rect

    ttl_max = cfg.get("masks.ttl_max")
    smooth_alpha = cfg.get("masks.smooth_alpha")
    iou_thresh   = cfg.get("matching.iou_thresh")
    dist_thresh  = cfg.get("matching.dist_thresh")

    if cfg.get("matching.mode") == "distance":
        best_val = float('inf')
        for i, m in enumerate(active_masks):
            d = center_distance(m['rect'], new_rect)
            if d < best_val:
                best_val = d
                best_idx = i
        matched = best_val <= dist_thresh and best_idx >= 0
    else:
        best_val = 0.0
        for i, m in enumerate(active_masks):
            iou = compute_iou(m['rect'], new_rect)
            if iou > best_val:
                best_val = iou
                best_idx = i
        matched = best_val >= iou_thresh and best_idx >= 0

    if matched:
        m = active_masks[best_idx]
        ox, oy, ow, oh = m['rect']

        # Vélocité basée sur last_detected (pas sur rect prédit)
        lx, ly, _, _ = m['last_detected_rect']
        dt = frame_id - m['last_detected_frame']
        if dt > 0:
            m['vx'] = (nx - lx) / dt
            m['vy'] = (ny - ly) / dt

        # Smooth exponentiel — stockage float
        sx = smooth_alpha * nx + (1 - smooth_alpha) * ox
        sy = smooth_alpha * ny + (1 - smooth_alpha) * oy
        sw = smooth_alpha * nw + (1 - smooth_alpha) * ow
        sh = smooth_alpha * nh + (1 - smooth_alpha) * oh

        m['last_detected_rect']  = (float(nx), float(ny), float(nw), float(nh))
        m['last_detected_frame'] = frame_id
        m['rect'] = (sx, sy, sw, sh)
        m['ttl']  = ttl_max
    else:
        active_masks.append({
            'rect': (float(nx), float(ny), float(nw), float(nh)),
            'ttl':  ttl_max,
            'vx':   0.0,
            'vy':   0.0,
            'last_detected_rect': (float(nx), float(ny), float(nw), float(nh)),
            'last_detected_frame': frame_id,
        })


def predict(active_masks):
    for mask in active_masks:
        if mask['ttl'] < cfg.get("masks.ttl_max") * 0.5:
            continue
        x, y, w, h = mask['rect']
        x += mask['vx']
        y += mask['vy']
        mask['rect'] = (x, y, w, h)


def pad_rect(x, y, w, h, max_w, max_h):
    x2 = max(x - cfg.get("masks.margin"), 0)
    y2 = max(y - cfg.get("masks.margin"), 0)
    w2 = min(w + 2 * cfg.get("masks.margin"), max_w - x2)
    h2 = min(h + 2 * cfg.get("masks.margin"), max_h - y2)
    return (x2, y2, w2, h2)


def draw_debug(frame, active_masks):
    for m in active_masks:
        x, y, w, h = (int(v) for v in m['rect'])
        ttl = m['ttl']
        color = ttl_color(ttl)
        thickness = 2 if ttl >= 3 else 1

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

        label = ttl_label(ttl)
        label_y = max(y - 6, 14)
        cv2.putText(frame, label, (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    total = len(active_masks)
    cv2.putText(frame, f"Masks: {total}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


# ═══════════════════════════════════════════════════════
#  STATS / BENCHMARK
# ═══════════════════════════════════════════════════════

_main_stats = {
    "loop_ms":      0.0,
    "total_frames": 0,
    "mask_peak":    0,
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
    print("        BENCHMARK PIPELINE v7.0 (capture threadée)")
    print("=" * 55)
    print(f"\n  📷 CAPTURE (thread @ {CAPTURE_FPS}fps cible)")
    for k, v in cs.items():
        print(f"    {k:22s} : {v}")

    print(f"\n  🔍 DETECT (thread)")
    for k, v in ds.items():
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
    print(f"    {'ttl_max':22s} : {cfg.get("masks.ttl_max")}")
    print(f"    {'margin_px':22s} : {cfg.get("masks.margin")}")
    print(f"    {'matching_mode':22s} : {matching_mode}")
    if matching_mode == "distance":
        print(f"    {'dist_thresh':22s} : {cfg.get("matching.dist_thresh")}")
    else:
        print(f"    {'iou_thresh':22s} : {cfg.get("matching.iou_thresh")}")

    old_loop = 32.66
    new_loop = loop_avg
    saved = round(old_loop - new_loop, 2)
    old_fps = round(1000 / old_loop, 1)
    new_fps = round(1000 / max(new_loop, 0.01), 1)

    print(f"\n  📉 GAIN vs v5.1")
    print(f"    {'v5.1 loop_avg':22s} : {old_loop} ms ({old_fps} FPS)")
    print(f"    {'v7.0 loop_avg':22s} : {new_loop} ms ({new_fps} FPS)")
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

fps_timer = time.time()
frame_count = 0

with pyvirtualcam.Camera(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, fps=VCAM_FPS) as vcam:
    print(f"✅ Caméra virtuelle prête → {vcam.device}")
    debug_draw = cfg.get("debug.draw")

    if debug_draw:
        print("🎨 MODE DEBUG VISUEL ACTIVÉ")
        print("   🟩 Vert   = détection fraîche (TTL 3+)")
        print("   🟨 Jaune  = masque persisté   (TTL 2)")
        print("   🟥 Rouge  = masque mourant    (TTL 1)")
    print("📸 En cours... (Ctrl+C pour arrêter)")

    sender = SendThread(vcam, SCREEN_WIDTH, SCREEN_HEIGHT)
    sender.start()

    try:
        active_masks = []
        last_detect_version = 0
        last_frame_id = 0

        capturer.reset_stats()
        detector.reset_stats()
        blur_reset()
        sender.reset_stats()

        while True:
            t_loop = time.perf_counter()

            # ── 1. Capture (NON BLOQUANT) ──
            frame = capturer.get_frame()
            if frame is None:
                time.sleep(0.001)
                continue

            # ── 2. Ne donner au detect QUE si frame nouvelle ──
            frame_id = capturer.get_frame_id()
            if frame_id > last_frame_id:
                last_frame_id = frame_id
                detector.give_frame(frame)

            # ── 3. Vérifier si nouvelle détection disponible ──
            current_version = detector.get_detect_count()

            if current_version > last_detect_version:
                for m in active_masks:
                    m['ttl'] -= 1
                active_masks = [m for m in active_masks if m['ttl'] > 0]

                last_detect_version = current_version
                new_plates = detector.get_zones()
                for p in new_plates:
                    x, y, w, h = p
                    padded = pad_rect(x, y, w, h, SCREEN_WIDTH, SCREEN_HEIGHT)
                    match_or_add(active_masks, padded, frame_id)

            """ else:
                predict(active_masks) """

            # ── 4. Cap max masques ──
            if len(active_masks) > cfg.get("masks.max_masks"):
                active_masks.sort(key=lambda m: m['ttl'], reverse=True)
                active_masks = active_masks[:cfg.get("masks.max_masks")]

            # ── 5. Blur ou debug ──
            blur_zones = [
                (int(m['rect'][0]), int(m['rect'][1]),
                 int(m['rect'][2]), int(m['rect'][3]))
                for m in active_masks
            ]

            # ── 6. Envoi vers OBS (zéro copie) ──
            buf = sender.borrow()
            np.copyto(buf, frame)

            if debug_draw:
                apply_blur(buf, blur_zones)
                draw_debug(buf, active_masks)
            else:
                apply_blur(buf, blur_zones)

            sender.publish()

            # ── 7. Stats loop ──
            _main_stats["loop_ms"] += (time.perf_counter() - t_loop) * 1000
            _main_stats["total_frames"] += 1
            _main_stats["mask_peak"] = max(
                _main_stats["mask_peak"], len(active_masks)
            )

            # ── 8. FPS counter ──
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 2.0:
                fps = frame_count / elapsed
                mode = "DEBUG" if debug_draw else "PROD"
                print(f"⚡ {fps:.1f} FPS | {len(active_masks)} masque(s) | {mode}")
                frame_count = 0
                fps_timer = time.time()

    except KeyboardInterrupt:
        print("\n🛑 Arrêt propre")
        print_all_stats()

    finally:
        sender.stop()
        detector.stop()
        capturer.stop()
