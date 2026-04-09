# debug_detect.py — Diagnostic visuel, capture immédiate
# Lance pendant que Rocket League est visible à l'écran
# Capture automatiquement au lancement

import cv2
import numpy as np
import dxcam
import time
import os

# ─────────────────────────────────────────
# DOSSIER DE SORTIE
# ─────────────────────────────────────────
OUTPUT_DIR = "debug_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# PARAMÈTRES
# ─────────────────────────────────────────
SCALE = 2.0

BLUR_KERNEL = (3, 3)
SOBEL_THRESH = 50

DILATE_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (max(int(15 / SCALE), 3), max(int(2 / SCALE), 1))
)
DILATE_ITERATIONS = 2

MIN_AREA   = int(400 / (SCALE * SCALE))
MIN_WIDTH  = int(50 / SCALE)
MAX_WIDTH  = int(400 / SCALE)
MIN_HEIGHT = int(12 / SCALE)
MAX_HEIGHT = int(50 / SCALE)
MIN_RATIO  = 2.5
MAX_RATIO  = 14.0
MIN_FILL   = 0.45
MIN_CHILDREN = 1

ORANGE_LOW  = np.array([8, 150, 180])
ORANGE_HIGH = np.array([22, 255, 255])
BLUE_LOW    = np.array([100, 140, 160])
BLUE_HIGH   = np.array([125, 255, 255])
HSV_MIN_COVERAGE = 0.20


# ─────────────────────────────────────────
# DIAGNOSTIC COMPLET
# ─────────────────────────────────────────

def run_diagnostic(frame_bgr, capture_id):
    stats = {}
    prefix = os.path.join(OUTPUT_DIR, f"cap{capture_id:02d}")
    h_orig, w_orig = frame_bgr.shape[:2]
    sw, sh = int(w_orig / SCALE), int(h_orig / SCALE)

    # 0. Original
    cv2.imwrite(f"{prefix}_0_original.png", frame_bgr)

    # 1. Resize
    t0 = time.perf_counter()
    small = cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_LINEAR)
    stats["resize_ms"] = (time.perf_counter() - t0) * 1000
    cv2.imwrite(f"{prefix}_1_resized.png", small)

    # 2. Grayscale
    t0 = time.perf_counter()
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    stats["gray_ms"] = (time.perf_counter() - t0) * 1000
    cv2.imwrite(f"{prefix}_2_gray.png", gray)

    # 3. Blur
    t0 = time.perf_counter()
    blurred = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
    stats["blur_ms"] = (time.perf_counter() - t0) * 1000
    cv2.imwrite(f"{prefix}_3_blurred.png", blurred)

    # 4. Sobel horizontal
    t0 = time.perf_counter()
    sobel = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_abs = cv2.convertScaleAbs(sobel)
    stats["sobel_ms"] = (time.perf_counter() - t0) * 1000
    cv2.imwrite(f"{prefix}_4_sobel.png", sobel_abs)

    # 5. Seuil binaire
    t0 = time.perf_counter()
    _, binary = cv2.threshold(sobel_abs, SOBEL_THRESH, 255, cv2.THRESH_BINARY)
    stats["thresh_ms"] = (time.perf_counter() - t0) * 1000
    cv2.imwrite(f"{prefix}_5_binary.png", binary)

    # 6. Dilatation
    t0 = time.perf_counter()
    dilated = cv2.dilate(binary, DILATE_KERNEL, iterations=DILATE_ITERATIONS)
    stats["dilate_ms"] = (time.perf_counter() - t0) * 1000
    cv2.imwrite(f"{prefix}_6_dilated.png", dilated)

    # 7. Contours
    t0 = time.perf_counter()
    contours, hierarchy = cv2.findContours(
        dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    stats["contour_ms"] = (time.perf_counter() - t0) * 1000
    stats["contours_total"] = len(contours)

    vis_all = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis_all, contours, -1, (0, 0, 255), 1)
    cv2.imwrite(f"{prefix}_7_all_contours.png", vis_all)

    # 8. Filtrage forme
    t0 = time.perf_counter()
    shape_passed = []
    reject_reasons = {"area": 0, "width": 0, "height": 0, "ratio": 0, "fill": 0}

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            reject_reasons["area"] += 1
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < MIN_WIDTH or w > MAX_WIDTH:
            reject_reasons["width"] += 1
            continue
        if h < MIN_HEIGHT or h > MAX_HEIGHT:
            reject_reasons["height"] += 1
            continue
        ratio = w / h
        if ratio < MIN_RATIO or ratio > MAX_RATIO:
            reject_reasons["ratio"] += 1
            continue
        fill = area / (w * h)
        if fill < MIN_FILL:
            reject_reasons["fill"] += 1
            continue
        shape_passed.append({
            "idx": i, "x": x, "y": y, "w": w, "h": h,
            "ratio": ratio, "fill": fill, "area": area
        })

    stats["shape_ms"] = (time.perf_counter() - t0) * 1000
    stats["shape_passed"] = len(shape_passed)
    stats["reject_reasons"] = reject_reasons

    vis_shape = small.copy()
    for s in shape_passed:
        cv2.rectangle(vis_shape, (s["x"], s["y"]),
                      (s["x"] + s["w"], s["y"] + s["h"]), (0, 255, 255), 2)
        cv2.putText(vis_shape, f'r={s["ratio"]:.1f} f={s["fill"]:.2f}',
                    (s["x"], s["y"] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    cv2.imwrite(f"{prefix}_8_shape_candidates.png", vis_shape)

    # 9. Filtrage enfants
    t0 = time.perf_counter()
    children_passed = []
    rejected_children = 0

    for s in shape_passed:
        i = s["idx"]
        child_count = 0
        if hierarchy is not None:
            child_idx = hierarchy[0][i][2]
            while child_idx != -1:
                child_count += 1
                child_idx = hierarchy[0][child_idx][0]
        if child_count < MIN_CHILDREN:
            rejected_children += 1
            continue
        s["children"] = child_count
        children_passed.append(s)

    stats["children_ms"] = (time.perf_counter() - t0) * 1000
    stats["children_passed"] = len(children_passed)
    stats["rejected_children"] = rejected_children

    vis_children = small.copy()
    for s in children_passed:
        cv2.rectangle(vis_children, (s["x"], s["y"]),
                      (s["x"] + s["w"], s["y"] + s["h"]), (255, 255, 0), 2)
        cv2.putText(vis_children, f'ch={s["children"]}',
                    (s["x"], s["y"] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
    cv2.imwrite(f"{prefix}_9_children_candidates.png", vis_children)

    # 10. Validation HSV
    t0 = time.perf_counter()
    hsv_small = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    detections = []
    rejected_hsv = []

    for s in children_passed:
        x, y, w, h = s["x"], s["y"], s["w"], s["h"]
        roi_hsv = hsv_small[y:y+h, x:x+w]
        if roi_hsv.size == 0:
            rejected_hsv.append({**s, "coverage": 0.0})
            continue
        mask_o = cv2.inRange(roi_hsv, ORANGE_LOW, ORANGE_HIGH)
        mask_b = cv2.inRange(roi_hsv, BLUE_LOW, BLUE_HIGH)
        mask_color = cv2.bitwise_or(mask_o, mask_b)
        total_px = w * h
        color_px = cv2.countNonZero(mask_color)
        coverage = color_px / total_px if total_px > 0 else 0.0
        s["coverage"] = round(coverage, 3)
        if coverage >= HSV_MIN_COVERAGE:
            detections.append(s)
        else:
            rejected_hsv.append(s)

    stats["hsv_ms"] = (time.perf_counter() - t0) * 1000
    stats["hsv_passed"] = len(detections)
    stats["rejected_hsv"] = rejected_hsv

    # 11. Image finale combinée
    vis_final = small.copy()
    for s in rejected_hsv:
        cv2.rectangle(vis_final, (s["x"], s["y"]),
                      (s["x"] + s["w"], s["y"] + s["h"]), (0, 0, 255), 2)
        cv2.putText(vis_final, f'HSV={s["coverage"]}',
                    (s["x"], s["y"] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    for s in detections:
        cv2.rectangle(vis_final, (s["x"], s["y"]),
                      (s["x"] + s["w"], s["y"] + s["h"]), (0, 255, 0), 3)
        cv2.putText(vis_final, f'OK={s["coverage"]}',
                    (s["x"], s["y"] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.imwrite(f"{prefix}_10_final_result.png", vis_final)

    # 12. HSV debug
    h_channel = hsv_small[:, :, 0]
    h_vis = cv2.applyColorMap(
        cv2.normalize(h_channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_HSV
    )
    cv2.imwrite(f"{prefix}_11_hsv_hue_map.png", h_vis)

    mask_o_full = cv2.inRange(hsv_small, ORANGE_LOW, ORANGE_HIGH)
    mask_b_full = cv2.inRange(hsv_small, BLUE_LOW, BLUE_HIGH)
    mask_combined = cv2.bitwise_or(mask_o_full, mask_b_full)
    cv2.imwrite(f"{prefix}_12_hsv_mask_orange_blue.png", mask_combined)

    masked_color = cv2.bitwise_and(small, small, mask=mask_combined)
    cv2.imwrite(f"{prefix}_13_hsv_masked_color.png", masked_color)

    # 13. Zoom sur chaque candidat rejeté HSV
    for j, s in enumerate(rejected_hsv):
        x, y, w, h = s["x"], s["y"], s["w"], s["h"]
        roi_bgr = small[y:y+h, x:x+w]
        if roi_bgr.size > 0:
            zoom = cv2.resize(roi_bgr, (w * 4, h * 4),
                              interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f"{prefix}_14_zoom_rejected_{j}_bgr.png", zoom)

            roi_hsv = hsv_small[y:y+h, x:x+w]
            m_o = cv2.inRange(roi_hsv, ORANGE_LOW, ORANGE_HIGH)
            m_b = cv2.inRange(roi_hsv, BLUE_LOW, BLUE_HIGH)
            m_c = cv2.bitwise_or(m_o, m_b)
            zoom_mask = cv2.resize(m_c, (w * 4, h * 4),
                                   interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f"{prefix}_14_zoom_rejected_{j}_hsv_mask.png", zoom_mask)

    total_ms = sum(v for k, v in stats.items() if k.endswith("_ms"))
    stats["TOTAL_ms"] = round(total_ms, 2)

    return stats, detections


def print_stats(stats):
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC DÉTECTION — Étape par étape")
    print("=" * 60)

    print(f"\n  ⏱️  TIMING")
    for k in ["resize_ms", "gray_ms", "blur_ms", "sobel_ms",
              "thresh_ms", "dilate_ms", "contour_ms", "shape_ms",
              "children_ms", "hsv_ms"]:
        if k in stats:
            print(f"    {k:22s} : {stats[k]:.2f} ms")
    print(f"    {'TOTAL':22s} : {stats.get('TOTAL_ms', 0):.2f} ms")

    print(f"\n  📊  ENTONNOIR")
    print(f"    {'Contours trouvés':35s} : {stats.get('contours_total', 0)}")

    rr = stats.get("reject_reasons", {})
    print(f"    → Passent forme{' ':16s} : {stats.get('shape_passed', 0)}")
    for reason, count in rr.items():
        print(f"       rejetés {reason:7s} : {count}")

    print(f"    → Passent enfants{' ':14s} : {stats.get('children_passed', 0)}")
    print(f"       rejetés enfants : {stats.get('rejected_children', 0)}")

    print(f"    → Passent HSV (FINAL){' ':10s} : {stats.get('hsv_passed', 0)}")
    rej_hsv = stats.get("rejected_hsv", [])
    print(f"       rejetés HSV     : {len(rej_hsv)}")

    if rej_hsv:
        print(f"\n  🔴  REJETÉS HSV (détail)")
        for s in rej_hsv:
            print(f"    pos=({s['x']},{s['y']}) "
                  f"size={s['w']}x{s['h']} "
                  f"ratio={s['ratio']:.1f} "
                  f"fill={s['fill']:.2f} "
                  f"children={s['children']} "
                  f"coverage={s['coverage']}")

    print("=" * 60)


def print_files(capture_id):
    prefix = f"cap{capture_id:02d}"
    files = sorted(f for f in os.listdir(OUTPUT_DIR) if f.startswith(prefix))
    print(f"\n  📁 Fichiers sauvés dans ./{OUTPUT_DIR}/")
    for f in files:
        print(f"    {f}")
    print()


# ─────────────────────────────────────────
# MAIN — capture immédiate
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("📸 Initialisation DXCam...")
    camera = dxcam.create(output_color="BGR")

    # Petit délai pour laisser DXCam se stabiliser
    time.sleep(0.5)

    frame = camera.grab()
    if frame is None:
        # Retry une fois
        time.sleep(1.0)
        frame = camera.grab()

    if frame is None:
        print("❌ Impossible de capturer une frame")
        exit(1)

    print(f"✅ Frame capturée : {frame.shape[1]}x{frame.shape[0]}")
    print(f"📁 Sortie : ./{OUTPUT_DIR}/\n")

    capture_id = 1
    stats, detections = run_diagnostic(frame, capture_id)
    print_stats(stats)
    print_files(capture_id)

    if detections:
        print(f"  ✅  DÉTECTIONS : {len(detections)}")
        for e in detections:
            print(f"    pos=({e['x']},{e['y']}) "
                  f"size={e['w']}x{e['h']} "
                  f"ratio={e['ratio']:.1f} "
                  f"coverage={e['coverage']} "
                  f"children={e['children']}")
    else:
        print(f"  ❌  AUCUNE DÉTECTION")

    print(f"\n✅ Terminé — regarde les fichiers dans ./{OUTPUT_DIR}/")
