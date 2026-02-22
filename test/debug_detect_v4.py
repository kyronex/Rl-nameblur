# debug_detect_v4.py "Dual-pass + texte blanc interne"
import cv2
import numpy as np
import dxcam
import time
import os

OUTPUT_DIR = "debug_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
for f in os.listdir(OUTPUT_DIR):
    os.remove(os.path.join(OUTPUT_DIR, f))

# PARAMÈTRES

SCALE = 2.0

# HSV — fond des cartouches
ORANGE_LOW  = np.array([8, 140, 170])
ORANGE_HIGH = np.array([22, 255, 255])
BLUE_LOW    = np.array([100, 130, 150])
BLUE_HIGH   = np.array([125, 255, 255])

# HSV — texte blanc/lumineux (S bas, V haut)
WHITE_LOW   = np.array([0, 0, 200])
WHITE_HIGH  = np.array([180, 60, 255])

# Morpho
KERNEL_CLOSE_H = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (max(int(15 / SCALE), 3), 1)
)
KERNEL_CLOSE_V = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (1, max(int(4 / SCALE), 1))
)

# Filtre forme (coordonnées SCALE)
MIN_AREA   = int(800 / (SCALE * SCALE))
MIN_WIDTH  = int(50 / SCALE)
MAX_WIDTH  = int(500 / SCALE)
MIN_HEIGHT = int(10 / SCALE)
MAX_HEIGHT = int(60 / SCALE)
MIN_RATIO  = 2.0
MAX_RATIO  = 15.0
MIN_FILL   = 0.35

# Filtre densité texte blanc
MIN_TRANSITIONS = 6
MIN_WHITE_RATIO = 0.05   # au moins 5% de blanc dans la zone

def analyze_white_text( mask_white, x, y, w, h):
    """
    Dans le rect (x,y,w,h), analyse le masque blanc :
    - white_ratio : proportion de pixels blancs
    - transitions : max transitions 0↔255 sur 3 lignes
    Retourne (white_ratio, transitions, roi_white)
    """
    roi_white = mask_white[y:y+h, x:x+w]

    if roi_white.size == 0:
        return 0.0, 0, roi_white

    # Proportion de blanc
    white_ratio = np.count_nonzero(roi_white) / roi_white.size

    # Transitions sur 3 lignes horizontales
    best_trans = 0
    for dy_ratio in [0.33, 0.50, 0.66]:
        row = int(h * dy_ratio)
        if row < 0 or row >= roi_white.shape[0]:
            continue
        line = roi_white[row, :]
        if len(line) < 2:
            continue
        diffs = np.diff((line > 127).astype(np.int8))
        trans = int(np.count_nonzero(diffs))
        best_trans = max(best_trans, trans)

    return white_ratio, best_trans

# CONNECTED COMPONENTS COLORÉ

def make_cc_colormap(mask_closed):
    h, w = mask_closed.shape[:2]
    num_labels, labels, cc_stats, _ = cv2.connectedComponentsWithStats(
        mask_closed, connectivity=8
    )
    cc_color = np.zeros((h, w, 3), dtype=np.uint8)
    for lbl in range(1, num_labels):
        cc_color[labels == lbl] = np.random.randint(50, 255, 3).tolist()
    return cc_color, num_labels, labels, cc_stats

# TRAITEMENT D'UN MASQUE COULEUR

def process_single_mask(mask_raw, mask_white, color_name, stats):
    """
    Prend un masque brut (orange ou bleu) + masque blanc global.
    Retourne (detections, closed_mask)
    """
    detections = []

    # Morpho
    closed = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, KERNEL_CLOSE_H)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, KERNEL_CLOSE_V)

    # Connected components
    num_labels, labels, cc_stats, _ = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )
    stats[f"components_{color_name}"] = num_labels - 1

    # Parcours des composantes
    for lbl in range(1, num_labels):
        x  = cc_stats[lbl, cv2.CC_STAT_LEFT]
        y  = cc_stats[lbl, cv2.CC_STAT_TOP]
        w  = cc_stats[lbl, cv2.CC_STAT_WIDTH]
        h  = cc_stats[lbl, cv2.CC_STAT_HEIGHT]
        area = cc_stats[lbl, cv2.CC_STAT_AREA]

        # Filtre taille
        if area < MIN_AREA:
            stats["rejected"] += 1
            stats["rejected_reasons"]["area"] = stats["rejected_reasons"].get("area", 0) + 1
            continue

        if w < MIN_WIDTH or w > MAX_WIDTH:
            stats["rejected"] += 1
            stats["rejected_reasons"]["width"] = stats["rejected_reasons"].get("width", 0) + 1
            continue

        if h < MIN_HEIGHT or h > MAX_HEIGHT:
            stats["rejected"] += 1
            stats["rejected_reasons"]["height"] = stats["rejected_reasons"].get("height", 0) + 1
            continue

        ratio = w / h
        if ratio < MIN_RATIO or ratio > MAX_RATIO:
            stats["rejected"] += 1
            stats["rejected_reasons"]["ratio"] = stats["rejected_reasons"].get("ratio", 0) + 1
            continue

        fill = area / (w * h) if (w * h) > 0 else 0
        if fill < MIN_FILL:
            stats["rejected"] += 1
            stats["rejected_reasons"]["fill"] = stats["rejected_reasons"].get("fill", 0) + 1
            continue

        # Filtre texte blanc
        white_ratio, transitions = analyze_white_text(mask_white, x, y, w, h)

        if transitions < MIN_TRANSITIONS:
            stats["rejected"] += 1
            stats["rejected_reasons"]["transitions"] = stats["rejected_reasons"].get("transitions", 0) + 1
            continue

        if white_ratio < MIN_WHITE_RATIO:
            stats["rejected"] += 1
            stats["rejected_reasons"]["white_ratio"] = stats["rejected_reasons"].get("white_ratio", 0) + 1
            continue

        # Accepté
        detections.append({
            "color":       color_name,
            "x":           int(x * SCALE),
            "y":           int(y * SCALE),
            "w":           int(w * SCALE),
            "h":           int(h * SCALE),
            "ratio":       round(ratio, 1),
            "fill":        round(fill, 2),
            "transitions": transitions,
            "white_ratio": round(white_ratio, 3),
        })

    return detections, closed

# DIAGNOSTIC COMPLET

def run_diagnostic(frame, capture_id=1):
    prefix = os.path.join(OUTPUT_DIR, f"cap{capture_id:02d}")

    stats = {
        "rejected": 0,
        "rejected_reasons": {},
        "detected": 0,
        "components_orange": 0,
        "components_blue": 0,
    }

    # 01 : resize
    h_orig, w_orig = frame.shape[:2]
    small = cv2.resize(frame, (int(w_orig / SCALE), int(h_orig / SCALE)),interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(f"{prefix}_01_resized.png", small)

    # HSV
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    # 02/03 : masques couleur
    mask_orange = cv2.inRange(hsv, ORANGE_LOW, ORANGE_HIGH)
    mask_blue   = cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)

    # 04b : masque blanc (texte)
    mask_white = cv2.inRange(hsv, WHITE_LOW, WHITE_HIGH)
    cv2.imwrite(f"{prefix}_04b_mask_white.png", mask_white)

    # Passe ORANGE
    det_orange, closed_orange = process_single_mask(mask_orange, mask_white, "orange", stats)
    cv2.imwrite(f"{prefix}_05_closed_orange.png", closed_orange)

    # Passe BLEU
    det_blue, closed_blue = process_single_mask(mask_blue, mask_white, "blue", stats)
    cv2.imwrite(f"{prefix}_06_closed_blue.png", closed_blue)

    # 07a/07b : CC séparées
    cc_orange, _, _, _ = make_cc_colormap(closed_orange)
    cc_blue, _, _, _   = make_cc_colormap(closed_blue)

    # 07c : CC fusionnées (superposition)
    cc_merged = np.maximum(cc_orange, cc_blue)
    cv2.imwrite(f"{prefix}_07c_cc_merged.png", cc_merged)

    # 1. Convertir cc_merged en masque binaire (tout pixel non-noir = blob)
    blob_mask = cv2.cvtColor(cc_merged, cv2.COLOR_BGR2GRAY)
    blob_mask = (blob_mask > 0).astype(np.uint8) * 255

    # 2. Intersection : blanc UNIQUEMENT dans les blobs
    white_in_blobs = cv2.bitwise_and(mask_white, blob_mask)

# 3. Debug
    cv2.imwrite(f"{prefix}_08a_blob_mask.png", blob_mask)
    cv2.imwrite(f"{prefix}_08b_white_in_blobs.png", white_in_blobs)

    all_detections = det_orange + det_blue
    stats["detected"] = len(all_detections)

    # 12 : résultat full res
    result = frame.copy()
    for d in all_detections:
        color = (245, 39, 238)
        x, y, w, h = d["x"], d["y"], d["w"], d["h"]
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

    cv2.imwrite(f"{prefix}_12_result_fullres.png", result)

    return stats, all_detections

def print_stats(stats):
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC v3.4 — Dual-pass + texte blanc")
    print("=" * 60)
    print(f"  {'SCALE':30s} : {SCALE}")
    print(f"  {'MIN_TRANSITIONS':30s} : {MIN_TRANSITIONS}")
    print(f"  {'MIN_WHITE_RATIO':30s} : {MIN_WHITE_RATIO}")
    print(f"  {'Composantes orange':30s} : {stats.get('components_orange', 0)}")
    print(f"  {'Composantes bleu':30s} : {stats.get('components_blue', 0)}")
    print(f"\n  📊  ENTONNOIR")
    print(f"    {'Détectés (FINAL)':30s} : {stats.get('detected', 0)}")
    print(f"    {'Rejetés':30s} : {stats.get('rejected', 0)}")

    reasons = stats.get("rejected_reasons", {})
    if reasons:
        print(f"\n    Raisons de rejet :")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"       {reason:18s} : {count}")
    print("=" * 60)


def print_files(capture_id):
    prefix = f"cap{capture_id:02d}"
    files = sorted(f for f in os.listdir(OUTPUT_DIR) if f.startswith(prefix))
    print(f"\n  📁 {len(files)} fichiers dans ./{OUTPUT_DIR}/")
    for f in files:
        print(f"    {f}")



if __name__ == "__main__":
    print("📸 Initialisation DXCam...")
    camera = dxcam.create(output_color="BGR")
    time.sleep(0.5)

    frame = camera.grab()
    if frame is None:
        time.sleep(1.0)
        frame = camera.grab()

    if frame is None:
        print("❌ Impossible de capturer")
        exit(1)

    print(f"✅ Frame : {frame.shape[1]}x{frame.shape[0]}")
    print(f"📁 Sortie : ./{OUTPUT_DIR}/\n")

    capture_id = 1
    stats, detections = run_diagnostic(frame, capture_id)
    print_stats(stats)
    print_files(capture_id)

    if detections:
        print(f"\n  ✅  {len(detections)} DÉTECTION(S)")
        for d in detections:
            print(f"    📍 [{d['color']:6s}] pos=({d['x']},{d['y']}) "
                  f"size={d['w']}x{d['h']} "
                  f"ratio={d['ratio']:.1f} "
                  f"fill={d['fill']:.2f} "
                  f"trans={d['transitions']} "
                  f"white={d['white_ratio']:.1%}")
    else:
        print(f"\n  ❌  AUCUNE DÉTECTION")

    print(f"\n✅ Terminé")
