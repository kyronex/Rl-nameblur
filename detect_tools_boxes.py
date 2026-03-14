# detect_tools_boxes.py
import cv2
import numpy as np
import time
import logging
from config import cfg
from detect_stats import flush_local
from detect_tools import write_circles , write_rects , get_color

log = logging.getLogger("detect_tools")

# ── extract_raw_boxes ──
def extract_raw_boxes(masked, params):
    """Contours → bounding boxes brutes (filtre min_area seulement)."""
    contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= params["min_area"]:
            boxes.append((x, y, w, h))
    return boxes

# ── adjust_boxes ──
def adjust_boxes(boxes, mask_white, h_img, params):
    min_blob = params.get("adjust_min_blob_area", 10)
    expand_search = params.get("expand_search_px", 2)
    retract_density = 0.25
    result = []
    for (x, y, w, h) in boxes:
        roi = mask_white[y:y+h, x:x+w]
        if roi.size == 0:
            result.append((x, y, w, h))
            continue
        n_labels, labels, stats_cc, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
        valid = []
        for i in range(1, n_labels):
            area_i = stats_cc[i, cv2.CC_STAT_AREA]
            if area_i >= min_blob:
                valid.append(i)
        if not valid:
            result.append((x, y, w, h))
            continue
        # ── Rétraction verticale seulement ──
        ry1 = min(stats_cc[i, cv2.CC_STAT_TOP] for i in valid)
        ry2 = max(stats_cc[i, cv2.CC_STAT_TOP] + stats_cc[i, cv2.CC_STAT_HEIGHT] for i in valid)
        # Garder X/W originaux, ajuster Y/H
        nx, nw = x, w
        ny = y + ry1
        nh = ry2 - ry1
        # Haut
        while nh > 1:
            row = mask_white[ny:ny+1, nx:nx+nw]
            if row.size == 0 or cv2.countNonZero(row) / max(nw, 1) >= retract_density:
                break
            ny += 1
            nh -= 1
        # ── Rétraction BAS par densité ──
        while nh > 1:
            row = mask_white[ny+nh-1:ny+nh, nx:nx+nw]
            if row.size == 0 or cv2.countNonZero(row) / max(nw, 1) >= retract_density:
                break
            nh -= 1
        # ── Expand HAUT (inchangé, sans seuil) ──
        if ry1 <= 3 and ny > 0:
            for _ in range(expand_search):
                if ny <= 0:
                    break
                row = mask_white[ny-1:ny, nx:nx+nw]
                if row.size == 0 or cv2.countNonZero(row) == 0:
                    break
                ny -= 1
                nh += 1
        # ── Expand BAS (inchangé, sans seuil) ──
        if ry2 >= h - 3 and ny + nh < h_img:
            for _ in range(expand_search):
                if ny + nh >= h_img:
                    break
                row = mask_white[ny+nh:ny+nh+1, nx:nx+nw]
                if row.size == 0 or cv2.countNonZero(row) == 0:
                    break
                nh += 1
        result.append((nx, ny, nw, nh))
    return result

# ── merge_nearby_horizontal ──
def merge_nearby_horizontal(boxes, max_gap_x=30, max_gap_y=10):
    """Fusionne les boxes proches horizontalement et alignées en Y."""
    if not boxes:
        return []
    sorted_boxes = sorted(boxes, key=lambda b: b[0])
    anchor = sorted_boxes[0]
    group = [anchor]
    gx2 = anchor[0] + anchor[2]
    groups = []
    for b in sorted_boxes[1:]:
        ay1, ay2 = anchor[1], anchor[1] + anchor[3]
        by1, by2 = b[1], b[1] + b[3]
        overlap = min(ay2, by2) - max(ay1, by1)
        y_ok = overlap > max_gap_y
        gap_x = b[0] - gx2
        # Différence de hauteur
        h_diff = abs(anchor[3] - b[3])
        # Écart entre centres Y
        cy_anchor = anchor[1] + anchor[3] / 2
        cy_b = b[1] + b[3] / 2
        cy_diff = abs(cy_anchor - cy_b)
        if gap_x < max_gap_x and y_ok and h_diff <= 3 and cy_diff <= 2:
            group.append(b)
            gx2 = max(gx2, b[0] + b[2])
        else:
            groups.append(group)
            anchor = b
            group = [b]
            gx2 = b[0] + b[2]
    groups.append(group)
    merged = []
    for group in groups:
        x1 = min(b[0] for b in group)
        y1 = min(b[1] for b in group)
        x2 = max(b[0] + b[2] for b in group)
        y2 = max(b[1] + b[3] for b in group)
        merged.append((x1, y1, x2 - x1, y2 - y1))
    return merged

# ── split_wide_boxes ──
def split_wide_boxes(boxes, mask_white, params):
    """Fractionne les boîtes trop larges en sous-blobs via projection verticale."""
    min_valley_w = params.get("min_valley_width", 8)
    max_density  = params.get("max_valley_density", 0.10)
    min_frag_w   = params.get("min_fragment_width", 40)
    result = []
    for (bx, by, bw, bh) in boxes:
        roi = mask_white[by:by+bh, bx:bx+bw]
        if roi.size == 0:
            result.append((bx, by, bw, bh))
            continue
        # ── Projection verticale ──
        col_sum = np.sum(roi > 0, axis=0)  # pixels blancs par colonne
        threshold = max_density * bh
        # ── Détecter les vallées ──
        is_empty = col_sum <= threshold
        valleys = []
        start = None
        for i, empty in enumerate(is_empty):
            if empty and start is None:
                start = i
            elif not empty and start is not None:
                if i - start >= min_valley_w:
                    valleys.append((start, i))
                start = None
        if start is not None and len(is_empty) - start >= min_valley_w:
            valleys.append((start, len(is_empty)))
        if not valleys:
            result.append((bx, by, bw, bh))
            continue
        # ── Couper aux vallées ──
        cuts = [0]
        for (vs, ve) in valleys:
            mid = (vs + ve) // 2
            cuts.append(mid)
        cuts.append(bw)
        # ── Construire les fragments ──
        for i in range(len(cuts) - 1):
            fx = cuts[i]
            fw = cuts[i + 1] - fx
            if fw < min_frag_w:
                continue
            # Extraire le ROI blanc du fragment
            frag_roi = mask_white[by:by+bh, bx+fx:bx+fx+fw]
            coords = cv2.findNonZero(frag_roi)
            if coords is None:
                continue
            # Bounding box tight sur les pixels blancs
            rx, ry, rw, rh = cv2.boundingRect(coords)
            result.append((bx + fx + rx, by + ry, rw, rh))
    return result

# ── validate_text ──
def has_text(roi, x1, y1, x2, y2, min_fill=0.08, min_tiers=1):
    crop = roi[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    h = crop.shape[0]
    t1 = h // 3
    t2 = 2 * h // 3
    tiers = [crop[:t1, :], crop[t1:t2, :], crop[t2:, :]]
    active = sum(
        1 for t in tiers
        if t.size > 0 and cv2.countNonZero(t) / t.size >= min_fill
    )
    return active >= min_tiers

def sweep_and_cut(ccs, roi, min_text_fill=0.08):
    validated = []
    group = []           # CC dans le groupe courant
    gx1, gy1, gx2, gy2 = None, None, None, None  # bbox union courante
    for cc in ccs:
        cx1, cy1, cx2, cy2 = cc
        if not group:
            # Premier élément
            group.append(cc)
            gx1, gy1, gx2, gy2 = cx1, cy1, cx2, cy2
            continue
        # ── Union candidate ──
        nx1 = min(gx1, cx1)
        ny1 = min(gy1, cy1)
        nx2 = max(gx2, cx2)
        ny2 = max(gy2, cy2)
        if has_text(roi, nx1, ny1, nx2, ny2, min_text_fill):
            group.append(cc)
            gx1, gy1, gx2, gy2 = nx1, ny1, nx2, ny2
        else:
            # Creux détecté → valider le groupe précédent si assez bon
            if len(group) >= 1:
                if has_text(roi, gx1, gy1, gx2, gy2, min_text_fill):
                    validated.append((gx1, gy1, gx2, gy2))
            # Recommencer avec la CC courante
            group = [cc]
            gx1, gy1, gx2, gy2 = cx1, cy1, cx2, cy2
    # ── Dernier groupe ──
    if len(group) >= 1:
        if has_text(roi, gx1, gy1, gx2, gy2, min_text_fill):
            validated.append((gx1, gy1, gx2, gy2))
    return validated

def validate_text(boxes, mask_white, params, kernels):
    result = []
    min_blob_area = params["refine_min_blob_area"]
    min_text_fill = params["min_text_fill"]
    kernel_rc = kernels["roi_connected"]
    for (bx, by, bw, bh) in boxes:
        roi = mask_white[by:by+bh, bx:bx+bw]
        if roi.size == 0:
            continue
        roi_connected = cv2.dilate(roi, kernel_rc, iterations=1)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi_connected)
        if num_labels <= 1:
            continue
        # stats colonnes : LEFT=0, TOP=1, WIDTH=2, HEIGHT=3, AREA=4
        s = stats[1:]  # exclure le fond (label 0)
        areas = s[:, cv2.CC_STAT_AREA]
        valid = areas >= min_blob_area
        if not np.any(valid):
            continue
        s = s[valid]
        x1 = s[:, cv2.CC_STAT_LEFT]
        y1 = s[:, cv2.CC_STAT_TOP]
        x2 = x1 + s[:, cv2.CC_STAT_WIDTH]
        y2 = y1 + s[:, cv2.CC_STAT_HEIGHT]
        # Trier par X
        order = np.argsort(x1)
        cc_boxes = list(zip(x1[order].tolist(), y1[order].tolist(),
                            x2[order].tolist(), y2[order].tolist()))
        groups = sweep_and_cut(cc_boxes, roi, min_text_fill)
        for (gx1, gy1, gx2, gy2) in groups:
            result.append((bx + gx1, by + gy1, gx2 - gx1, gy2 - gy1))
    return result

# ── validate_background ──
def validate_background(boxes, mask_white, rgb, params):
    var_norm        = params.get("var_norm", 200.0)
    coherent_delta  = params.get("coherent_delta", 33)
    min_score       = params.get("min_bg_score", 0.5)
    result = []
    for (x, y, w, h) in boxes:
        w_pad = 2
        x = max(0, x - w_pad)
        w = w + w_pad * 2
        # ROI directe en gray (évite cvtColor sur RGB)
        rgb_roi = rgb[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(rgb_roi, cv2.COLOR_RGB2GRAY)
        local_mean = cv2.blur(gray_roi, (3, 3))
        # Calcul diff en place, en int16
        diff = cv2.absdiff(gray_roi, local_mean)  # uint8, évite int16
        fond_mask = mask_white[y:y+h, x:x+w] == 0
        fond_diff = diff[fond_mask]
        fond_count = fond_diff.size
        if fond_count == 0:
            continue
        # Mesure 1 : variance locale médiane
        median_val = float(np.median(fond_diff))
        median_var = median_val * median_val
        s_var = max(0.0, 1.0 - median_var / var_norm)
        # Mesure 2 : ratio cohérent
        s_coh = float(np.count_nonzero(fond_diff < coherent_delta)) / fond_count
        score = s_var * s_coh
        if score >= min_score:
            result.append((x, y, w, h))
    return result

# ── filter_geometry ──
def filter_geometry(boxes, masked, params):
    """Filtre ratio/area/fill sur des boxes (déjà mergées)."""
    min_area   = params["min_area"]
    max_area   = params["max_area"]
    min_width  = params["min_width"]
    min_height = params["min_height"]
    min_ratio  = params["min_ratio"]
    max_ratio  = params["max_ratio"]
    min_fill   = params["min_fill"]
    max_fill   = params["max_fill"]

    plates = []
    for (x, y, w, h) in boxes:
        if w < min_width or h < min_height:
            continue
        area = w * h
        if area < min_area or area > max_area:
            continue
        ratio = w / h  # h >= min_height > 0, pas besoin de max(h,1)
        if ratio < min_ratio or ratio > max_ratio:
            continue
        fill = cv2.countNonZero(masked[y:y+h, x:x+w]) / area
        if fill < min_fill or fill > max_fill:
            continue
        plates.append((x, y, w, h))
    return plates

# ── resolve_overlaps ──
def recrop_from_white(mask_white, x, y, w, h, ref_x, ref_y, ref_w, ref_h):
    """
    Recadre une box faible en se basant sur le mask_white
    dans la zone hors intersection avec la référence.

    On masque la zone de la référence, puis on cherche le bounding box
    des pixels blancs restants.
    Retourne (nx, ny, nw, nh) ou None si rien ne reste.
    """
    roi = mask_white[y:y + h, x:x + w].copy()

    # ── Masquer la zone d'intersection (en coords locales) ──
    ix1 = max(ref_x, x) - x
    iy1 = max(ref_y, y) - y
    ix2 = min(ref_x + ref_w, x + w) - x
    iy2 = min(ref_y + ref_h, y + h) - y

    if ix2 > ix1 and iy2 > iy1:
        roi[iy1:iy2, ix1:ix2] = 0

    # ── Bounding box des pixels restants ──
    coords = cv2.findNonZero(roi)
    if coords is None:
        return None

    bx, by, bw, bh = cv2.boundingRect(coords)
    return (x + bx, y + by, bw, bh)

def edge_confidence(mask_white, x, y, w, h, img_h, img_w):
    """
    Score de confiance d'une box basé sur le contraste
    bande intérieure (2px) vs bande extérieure (2px) sur 4 bords.
    Retourne un float entre -1.0 et 1.0.
    """
    BAND = 2
    total = 0.0
    count = 0
    xw = x + w
    yh = y + h
    # ── Bord gauche ──
    in_x2 = min(x + BAND, xw)
    out_x1 = max(x - BAND, 0)
    if in_x2 > x and out_x1 < x:
        inner = mask_white[y:yh, x:in_x2]
        outer = mask_white[y:yh, out_x1:x]
        if inner.size > 0 and outer.size > 0:
            total += cv2.countNonZero(inner) / inner.size - cv2.countNonZero(outer) / outer.size
            count += 1
    # ── Bord droit ──
    in_x1 = max(xw - BAND, x)
    out_x2 = min(xw + BAND, img_w)
    if xw > in_x1 and out_x2 > xw:
        inner = mask_white[y:yh, in_x1:xw]
        outer = mask_white[y:yh, xw:out_x2]
        if inner.size > 0 and outer.size > 0:
            total += cv2.countNonZero(inner) / inner.size - cv2.countNonZero(outer) / outer.size
            count += 1
    # ── Bord haut ──
    in_y2 = min(y + BAND, yh)
    out_y1 = max(y - BAND, 0)
    if in_y2 > y and out_y1 < y:
        inner = mask_white[y:in_y2, x:xw]
        outer = mask_white[out_y1:y, x:xw]
        if inner.size > 0 and outer.size > 0:
            total += cv2.countNonZero(inner) / inner.size - cv2.countNonZero(outer) / outer.size
            count += 1
    # ── Bord bas ──
    in_y1 = max(yh - BAND, y)
    out_y2 = min(yh + BAND, img_h)
    if yh > in_y1 and out_y2 > yh:
        inner = mask_white[in_y1:yh, x:xw]
        outer = mask_white[yh:out_y2, x:xw]
        if inner.size > 0 and outer.size > 0:
            total += cv2.countNonZero(inner) / inner.size - cv2.countNonZero(outer) / outer.size
            count += 1
    if count == 0:
        return 0.0
    return total / count

def resolve_overlaps(boxes, mask_white, params):
    if len(boxes) < 2:
        return list(boxes)
    img_h, img_w = mask_white.shape[:2]
    # ── 1. Calculer le score de chaque box ──
    scored = []
    for (x, y, w, h) in boxes:
        score = edge_confidence(mask_white, x, y, w, h, img_h, img_w)
        scored.append(((x, y, w, h), score))
    # ── 2. Trier par score décroissant ──
    scored.sort(key=lambda s: (s[1], s[0][2] * s[0][3]), reverse=True)
    # ── 3. Résoudre les chevauchements ──
    min_residual_area = int(params["min_area"] * 0.35)
    result = []
    for rect, _ in scored:
        if rect is None:
            continue
        cx, cy, cw, ch = rect
        # Comparer avec toutes les références déjà validées (score supérieur)
        for rx, ry, rw, rh in result:
            # ── Intersection ? ──
            ix1 = max(cx, rx)
            iy1 = max(cy, ry)
            ix2 = min(cx + cw, rx + rw)
            iy2 = min(cy + ch, ry + rh)
            if ix2 <= ix1 or iy2 <= iy1:
                continue  # pas de chevauchement
            # ── Recadrer la box courante (faible) via mask_white ──
            roi = mask_white[cy:cy+ch, cx:cx+cw].copy()
            overlap_y1 = max(ry, cy) - cy
            overlap_y2 = min(ry + rh, cy + ch) - cy
            if overlap_y2 > overlap_y1:
                roi[overlap_y1:overlap_y2, :] = 0

            if roi.size == 0 or cv2.countNonZero(roi) == 0:
                cx, cy, cw, ch = 0, 0, 0, 0
                break
            coords = cv2.findNonZero(roi)
            bx, by, bw, bh = cv2.boundingRect(coords)
            cx, cy, cw, ch = cx + bx, cy + by, bw, bh

        # ── Vérifier taille minimale ──
        if cw * ch >= min_residual_area and cw >= 10 and ch >= 5:
            result.append((cx, cy, cw, ch))
    return result

# ── tight_crop_white ──
def tight_crop_white(boxes, mask_white):
    """Recadre chaque box au bounding-box réel des pixels blancs qu'elle contient."""
    result = []
    for (x, y, w, h) in boxes:
        roi = mask_white[y:y+h, x:x+w]
        if roi.size == 0 or cv2.countNonZero(roi) == 0:
            continue
        coords = cv2.findNonZero(roi)
        rx, ry, rw, rh = cv2.boundingRect(coords)
        nx, ny = x + rx, y + ry
        result.append((nx, ny, rw, rh))
    return result

# ── adjust_resolve ──
def adjust_resolve(boxes, mask_white, h_img, params):
    adjusted = adjust_boxes(boxes, mask_white, h_img, params)
    resolved  = resolve_overlaps(adjusted, mask_white, params)
    cropped  = tight_crop_white(resolved, mask_white)
    return resolved

# ── expand_plates ──
def expand_plates(boxes, img):
    img_h, img_w = img.shape[:2]
    expanded = []
    for (x, y, w, h) in boxes:
        nx = max(x - 2, 0)
        ny = max(y - 1, 0)
        nx2 = min(x + w + 2, img_w)
        ny2 = min(y + h + 1, img_h)
        expanded.append((nx, ny, nx2 - nx, ny2 - ny))
    return expanded


def process_channel(masked,rgb, mask_white, h_img, params, kernels, stats):
    t0 = time.perf_counter()
    plates = []
    screen = rgb.copy()
    Vert = tuple(cfg.get("debug.colors_ttl.vert"))
    Magenta = tuple(cfg.get("debug.colors_ttl.magenta"))
    Void = tuple(cfg.get("debug.colors_ttl.void"))

    boxes = extract_raw_boxes(masked, params)

    log.debug("boxes_ar")
    boxes_ar = adjust_resolve(boxes, mask_white, h_img, params)

    log.debug("split_wide_boxes")
    split = split_wide_boxes(boxes_ar, mask_white, params)

    log.debug("split_ar")
    split_ar = adjust_resolve(split, mask_white, h_img, params)

    log.debug("validate_text")
    validated_t = validate_text(split_ar, mask_white, params, kernels)

    log.debug("validated_t_ar")
    validated_t_ar = adjust_resolve(validated_t, mask_white, h_img, params)

    log.debug("merge_nearby_horizontal")
    merge = merge_nearby_horizontal(validated_t_ar, params["max_gap_x"],params["max_gap_y"])

    log.debug("validate_background")
    validated_b = validate_background(merge,mask_white, rgb, params)
    write_rects(screen, validated_b, Void , 2)

    log.debug("validated_b_ar")
    validated_b_ar = adjust_resolve(validated_b, mask_white, h_img, params)
    write_rects(screen, validated_b_ar, Magenta , 1)

    log.debug("expand_plates")
    expanded = expand_plates(validated_b_ar,rgb)
    write_rects(screen, expanded, Vert , 1)


    log.debug("filter_geometry")
    plates = filter_geometry(expanded, masked, params)

    return plates

# ── has_blob_continuity not used ──
def has_blob_continuity(masked, x, y, w, h, direction):
    # Prendre une zone de 2px : 1px dans la box + 1px dans l'expansion
    if direction == "right":
        check = masked[y:y+h, x+w-1:x+w+1]
    elif direction == "left":
        check = masked[y:y+h, x-1:x+1]
    elif direction == "down":
        check = masked[y+h-1:y+h+1, x:x+w]
    elif direction == "up":
        check = masked[y-1:y+1, x:x+w]
    else:
        return False
    if check.size == 0 or cv2.countNonZero(check) == 0:
        return False
    # connectedComponents sur cette zone de 2px
    n_labels, labels = cv2.connectedComponents(check, connectivity=8)
    for label_id in range(1, n_labels):
        coords = np.argwhere(labels == label_id)
        if direction in ("right", "left"):
            cols_touched = set(coords[:, 1])  # axe X
        else:
            cols_touched = set(coords[:, 0])  # axe Y
        if len(cols_touched) >= 2:
            return True
    return False