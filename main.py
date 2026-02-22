# main.py
import time

import cv2
import numpy as np
import pyvirtualcam

from capture_thread import CaptureThread
from detect_thread import DetectThread
from send_thread import SendThread
from blur import apply_blur
from blur import get_stats as blur_stats, reset_stats as blur_reset

# ─────────────────────────────────────────
# PARAMÈTRES ÉCRAN
# ─────────────────────────────────────────
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
CAPTURE_FPS = 120
VCAM_FPS = 120

# ─────────────────────────────────────────
# CONFIG ROCKET LEAGUE
# ─────────────────────────────────────────
TTL_MAX    = 8
MARGIN     = 6
SKIP       = 1
IOU_THRESH = 0.15
MAX_MASKS  = 20

DEBUG_DRAW = True

# ─────────────────────────────────────────
# COULEURS DEBUG (BGR)
# ─────────────────────────────────────────
COLOR_FRESH   = (0, 255, 0)
COLOR_PERSIST = (0, 255, 255)
COLOR_DYING   = (0, 0, 255)

def ttl_color(ttl):
    if ttl >= 3:
        return COLOR_FRESH
    elif ttl >= 2:
        return COLOR_PERSIST
    else:
        return COLOR_DYING

def ttl_label(ttl):
    return f"TTL={ttl}"

# ─────────────────────────────────────────
# IoU (Intersection over Union)
# ─────────────────────────────────────────
def compute_iou(r1, r2):
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


def match_or_add(active_masks, new_rect, ttl_max, iou_thresh):
    best_iou = 0.0
    best_idx = -1

    for i, m in enumerate(active_masks):
        iou = compute_iou(m['rect'], new_rect)
        if iou > best_iou:
            best_iou = iou
            best_idx = i

    if best_iou >= iou_thresh and best_idx >= 0:
        active_masks[best_idx]['rect'] = new_rect
        active_masks[best_idx]['ttl'] = ttl_max
    else:
        active_masks.append({
            'rect': new_rect,
            'ttl':  ttl_max,
        })


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def pad_rect(x, y, w, h, margin, max_w, max_h):
    x2 = max(x - margin, 0)
    y2 = max(y - margin, 0)
    w2 = min(w + 2 * margin, max_w - x2)
    h2 = min(h + 2 * margin, max_h - y2)
    return (x2, y2, w2, h2)

def draw_debug(frame, active_masks):
    for m in active_masks:
        x, y, w, h = m['rect']
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

# ─────────────────────────────────────────
# LANCEMENT
# ─────────────────────────────────────────
capturer = CaptureThread(target_fps=CAPTURE_FPS)
capturer.start()

detector = DetectThread()
detector.start()

fps_timer = time.time()
frame_count = 0

rgb_buffer = np.empty((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)

_main_stats = {
    "loop_ms":      0.0,
    "total_frames": 0,
    "mask_peak":    0,
}

def print_all_stats():
    n = max(_main_stats["total_frames"], 1)
    cs = capturer.get_stats()
    ds = detector.get_stats()
    bs = blur_stats()
    ss = sender.get_stats()

    print("\n" + "=" * 55)
    print("        BENCHMARK PIPELINE v6.0 (capture threadée)")
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

    print(f"\n  🎬 MAIN LOOP")
    loop_avg = round(_main_stats['loop_ms'] / n, 2)
    print(f"    {'loop_avg_ms':22s} : {loop_avg}")
    print(f"    {'total_frames':22s} : {_main_stats['total_frames']}")
    print(f"    {'mask_peak':22s} : {_main_stats['mask_peak']}")
    print(f"    {'ttl_max':22s} : {TTL_MAX}")
    print(f"    {'margin_px':22s} : {MARGIN}")
    print(f"    {'skip':22s} : {SKIP}")
    print(f"    {'iou_thresh':22s} : {IOU_THRESH}")
    print(f"    {'debug_draw':22s} : {DEBUG_DRAW}")

    old_loop = 32.66
    new_loop = loop_avg
    saved = round(old_loop - new_loop, 2)
    old_fps = round(1000 / old_loop, 1)
    new_fps = round(1000 / max(new_loop, 0.01), 1)

    print(f"\n  📉 GAIN vs v5.1")
    print(f"    {'v5.1 loop_avg':22s} : {old_loop} ms ({old_fps} FPS)")
    print(f"    {'v6.0 loop_avg':22s} : {new_loop} ms ({new_fps} FPS)")
    print(f"    {'économisé':22s} : {saved} ms/frame")
    print(f"    {'accélération':22s} : x{round(old_loop / max(new_loop, 0.01), 2)}")

    print("=" * 55)

with pyvirtualcam.Camera(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, fps=VCAM_FPS) as vcam:
    print(f"✅ Caméra virtuelle prête → {vcam.device}")
    if DEBUG_DRAW:
        print("🎨 MODE DEBUG VISUEL ACTIVÉ")
        print("   🟩 Vert   = détection fraîche (TTL 3-4)")
        print("   🟨 Jaune  = masque persisté   (TTL 2)")
        print("   🟥 Rouge  = masque mourant    (TTL 1)")
    print("📸 En cours... (Ctrl+C pour arrêter)")

    sender = SendThread(vcam)
    sender.start()

    try:
        active_masks = []
        frame_id = 0
        last_detect_version = 0

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

            # ── 2. Donner frame au detect thread ──
            detector.give_frame(frame)

            # ── 3. Vérifier si nouvelle détection disponible ──
            current_version = detector.get_detect_count()
            has_new_detect = current_version > last_detect_version

            if has_new_detect:
                last_detect_version = current_version
                new_plates = detector.get_zones()

                # ── 4. Ajouter avec IoU + TTL + padding ──
                for p in new_plates:
                    x, y, w, h = p
                    padded = pad_rect(x, y, w, h, MARGIN,
                                    SCREEN_WIDTH, SCREEN_HEIGHT)
                    match_or_add(active_masks, padded, TTL_MAX, IOU_THRESH)

                # ── 5. Décrémenter TTL ──
                for m in active_masks:
                    m['ttl'] -= 1

                # ── 6. Purger les morts ──
                active_masks = [m for m in active_masks if m['ttl'] > 0]

                # ── 7. Cap max masques ──
                if len(active_masks) > MAX_MASKS:
                    active_masks.sort(key=lambda m: m['ttl'], reverse=True)
                    active_masks = active_masks[:MAX_MASKS]

            # ── 8. Construire liste rects pour blur ──
            blur_zones = [m['rect'] for m in active_masks]

            # ── 9. Blur + conversion RGB ──
            if frame_id % SKIP == 0:
                if DEBUG_DRAW:
                    debug_frame = draw_debug(frame.copy(), active_masks)
                    np.copyto(rgb_buffer,
                            cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB))
                    frame_rgb = rgb_buffer
                else:
                    frame_rgb = apply_blur(frame, blur_zones,
                                        rgb_buffer=rgb_buffer)

            # ── 10. Envoi vers OBS ──
            sender.give_frame(frame_rgb)

            # ── 11. Stats loop ──
            _main_stats["loop_ms"] += (time.perf_counter() - t_loop) * 1000
            _main_stats["total_frames"] += 1
            _main_stats["mask_peak"] = max(_main_stats["mask_peak"],
                                        len(active_masks))
            frame_id += 1

            # ── 12. FPS counter ──
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 2.0:
                fps = frame_count / elapsed
                skipped = (SKIP - 1) / SKIP * 100
                mode = "DEBUG" if DEBUG_DRAW else "PROD"
                print(f"⚡ {fps:.1f} FPS | {len(active_masks)} masque(s) | "
                    f"skip {skipped:.0f}% | {mode}")
                frame_count = 0
                fps_timer = time.time()

    except KeyboardInterrupt:
        print("\n🛑 Arrêt propre")
        print_all_stats()

    finally:
        sender.stop()
        detector.stop()
        capturer.stop()
