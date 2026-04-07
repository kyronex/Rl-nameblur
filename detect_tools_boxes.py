# detect_tools_boxes.py
import cv2
import numpy as np
import time
import logging
from config import cfg
from detect_tools import write_circles , write_rects , get_color
from box import Box

log = logging.getLogger("detect_tools_boxes")

# ── _extract_raw_boxes ──
def _extract_raw_boxes(masked, params):
    """Contours → bounding boxes brutes (filtre min_area seulement)."""
    contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= params["min_area"]:
            boxes.append(Box(x, y, w, h))
    return boxes

# ── _adjust_boxes ──
def _adjust_boxes(boxes, mask_white, h_img, params):
    min_blob = params.get("adjust_min_blob_area", 10)
    expand_search = params.get("expand_search_px", 2)
    retract_density = 0.25
    result = []
    for box in boxes:
        x, y, w, h = box.rect
        roi = mask_white[y:y+h, x:x+w]
        if roi.size == 0:
            result.append(box)
            continue
        n_labels, labels, stats_cc, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
        valid = []
        for i in range(1, n_labels):
            area_i = stats_cc[i, cv2.CC_STAT_AREA]
            if area_i >= min_blob:
                valid.append(i)
        if not valid:
            result.append(box)
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
        result.append(box.copy_with(x=nx, y=ny, w=nw, h=nh))
    return result

# ── _merge_nearby_horizontal ──
def _merge_nearby_horizontal(boxes, max_gap_x=30, max_gap_y=10):
    """Fusionne les boxes proches horizontalement et alignées en Y."""
    if not boxes:
        return []
    sorted_boxes = sorted(boxes, key=lambda b: b.x)
    anchor = sorted_boxes[0]
    group = [anchor]
    gx2 = anchor.x + anchor.w
    groups = []
    for b in sorted_boxes[1:]:
        ay1, ay2 = anchor.y, anchor.y + anchor.h
        by1, by2 = b.y, b.y + b.h
        overlap = min(ay2, by2) - max(ay1, by1)
        y_ok = overlap > max_gap_y
        gap_x = b.x - gx2
        # Différence de hauteur
        h_diff = abs(anchor.h - b.h)
        # Écart entre centres Y
        cy_anchor =  anchor.y + anchor.h / 2
        cy_b = b.y + b.h / 2
        cy_diff = abs(cy_anchor - cy_b)
        if gap_x < max_gap_x and y_ok and h_diff <= 3 and cy_diff <= 2:
            group.append(b)
            gx2 = max(gx2, b.x + b.w)
        else:
            groups.append(group)
            anchor = b
            group = [b]
            gx2 = b.x + b.w
    groups.append(group)
    merged = []
    for group in groups:
        x1 = min(b.x for b in group)
        y1 = min(b.y for b in group)
        x2 = max(b.x + b.w for b in group)
        y2 = max(b.y + b.h for b in group)
        merged_scores = Box.merge_scores(*group)
        merged.append(Box(x1, y1, x2 - x1, y2 - y1, scores=merged_scores))
    return merged

# ── _split_wide_boxes ──
def _split_wide_boxes(boxes, mask_white, params):
    """Fractionne les boîtes trop larges en sous-blobs via projection verticale."""
    min_valley_w = params.get("min_valley_width", 8)
    max_density  = params.get("max_valley_density", 0.10)
    min_frag_w   = params.get("min_fragment_width", 40)
    result = []
    for box in boxes:
        bx, by, bw, bh = box.rect
        roi = mask_white[by:by+bh, bx:bx+bw]
        if roi.size == 0:
            result.append(box)
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
            result.append(box)
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
            result.append(box.copy_with(x=bx + fx + rx, y=by + ry, w=rw, h=rh))
    return result

# ── _validate_text_v2 ──
_EMPTY_PROJ = {"proj_ratio": 0.0, "fill_ratio": 0.0, "compactness": 0.0, "s_pf": 0.0}

def _projection_fill_score(crop):
    if crop.size == 0:
        return _EMPTY_PROJ

    h, w = crop.shape
    total_area = h * w

    # ── Projections (évite crop > 0, crop est déjà binaire 0/255) ──
    col_any = np.any(crop, axis=0)
    row_any = np.any(crop, axis=1)

    active_cols = int(np.count_nonzero(col_any))
    active_rows = int(np.count_nonzero(row_any))

    # ── Aires ──
    proj_area  = active_cols * active_rows
    proj_ratio = proj_area / total_area

    white_count = cv2.countNonZero(crop)
    fill_ratio  = white_count / total_area

    # ── Compacité ──
    compactness = fill_ratio / max(proj_ratio, 0.01)

    # ── Score final (sans round = ~10% plus rapide) ──
    s_proj = max(0.0, 1.0 - abs(proj_ratio - 0.70) * 2.5)   # 1/0.40 = 2.5
    s_comp = max(0.0, 1.0 - abs(compactness - 0.40) * 2.857) # 1/0.35 ≈ 2.857

    s_pf = s_proj * s_comp

    return {
        "proj_ratio":  proj_ratio,
        "fill_ratio":  fill_ratio,
        "compactness": compactness,
        "s_pf":  s_pf,
    }

_EMPTY_SCORES = {
    "transition_density": 0.0, "row_fill": 0.0,"vproj": 0.0, "density": 0.0, "cc": 0.0, "hreg": 0.0,
    "proj_ratio": 0.0, "fill_ratio": 0.0,"compactness": 0.0, "proj_score": 0.0,
}

def _cc_metrics_from_stats(cc_stats, gx1, gy1, gx2, gy2):
    """
    Calcule cc (nombre de CC) et hreg (régularité des hauteurs)
    à partir des stats CC déjà connues, filtrées sur la bbox du groupe.

    Remplace : connectedComponentsWithStats(crop) dans _has_text
    """
    # ── Filtrer les CC dont le centre tombe dans la bbox du groupe ──
    cx = cc_stats[:, 0] + cc_stats[:, 2] * 0.5   # center_x = left + width/2
    cy = cc_stats[:, 1] + cc_stats[:, 3] * 0.5   # center_y = top + height/2

    inside = ((cx >= gx1) & (cx < gx2) &(cy >= gy1) & (cy < gy2))
    sel = cc_stats[inside]
    cc = len(sel)

    if cc < 2:
        heights = sel[:, 3] if cc == 1 else np.array([0])
        mean_h = float(heights[0]) if cc == 1 else 0.0
        return cc, 0.0, mean_h

    heights = sel[:, 3].astype(np.float32)
    mean_h = float(np.mean(heights))
    if mean_h < 1.0:
        return cc, 0.0, mean_h

    var_norm = float(np.var(heights)) / (mean_h * mean_h)
    hreg = max(0.0, 1.0 - var_norm * 5.0)

    return cc, hreg, mean_h


# ══════════════════════════════════════════════════════════════
# PRÉ-CALCUL GROUPE CHEAP
# ══════════════════════════════════════════════════════════════
def _compute_cheap_metrics(crop, cc_stats, x1, y1, x2, y2, min_fill=0.08):
    """
    Calculs légers : numpy basique, 1 seul countNonZero.
    Retourne un dict plat, zéro décision ici.
    """
    h, w = crop.shape

    # density_raw — 1 seul countNonZero pour tout le groupe
    nz          = cv2.countNonZero(crop)
    density_raw = float(nz) / crop.size

    # cc + hreg — 1 seul appel
    cc_raw, hreg, _ = _cc_metrics_from_stats(cc_stats, x1, y1, x2, y2)

    # transitions horizontales
    if w > 1:
        left  = crop[:, :-1]
        right = crop[:, 1:]
        transition_density = float(np.count_nonzero(left != right)) / (h * max(w - 1, 1))
    else:
        transition_density = 0.0

    # row fill
    row_any  = np.any(crop, axis=1)
    row_fill = float(np.count_nonzero(row_any)) / max(h, 1)

    # tiers
    t1    = h // 3
    t2    = 2 * h // 3
    tiers = [crop[:t1, :], crop[t1:t2, :], crop[t2:, :]]
    tiers_active = sum(
        1 for t in tiers
        if t.size > 0 and cv2.countNonZero(t) / t.size >= min_fill
    )

    # scores normalisés cheap
    s_td   = max(0.0, 1.0 - abs(transition_density - 0.45) * 3.33)
    s_dens = max(0.0, 1.0 - abs(density_raw - 0.50) * 2)
    s_cc   = max(0.0, 1.0 - abs(cc_raw - 8) * 0.125)

    return {
        "transition_density": transition_density,
        "s_td":               s_td,
        "density_raw":        density_raw,
        "s_dens":             s_dens,
        "cc_raw":             cc_raw,
        "s_cc":               s_cc,
        "s_hreg":             hreg,
        "row_fill":           row_fill,
        "tiers_active":       tiers_active,
    }


# ══════════════════════════════════════════════════════════════
# PRÉ-CALCUL GROUPE HEAVY
# ══════════════════════════════════════════════════════════════
def _compute_heavy_metrics(crop):
    """
    Calculs lourds : np.var + _projection_fill_score.
    Appelé UNIQUEMENT si _decide_early() passe.
    Retourne un dict plat, zéro décision ici.
    """
    h = crop.shape[0]

    # projection verticale
    col_sum = np.sum(crop, axis=0)
    col_norm = col_sum / max(h * 255.0, 1.0)
    vproj = float(np.var(col_norm))
    s_vp    = max(0.0, 1.0 - abs(vproj - 0.08) / 0.10)

    # projection fill score
    pf      = _projection_fill_score(crop)

    return {
        "vproj":      vproj,
        "s_vp":       s_vp,
        "s_pf":     pf["s_pf"],
    }


# ══════════════════════════════════════════════════════════════
# DÉCISION EARLY  (lecture seule sur cheap)
# ══════════════════════════════════════════════════════════════
def _decide_early(cheap, min_transition=0.20, min_tiers=2):
    # step 0 — binaire absolu
    if cheap["density_raw"] > 0.95:
        return False, "density_raw>0.95"

    # step 1 — transitions
    if cheap["transition_density"] < min_transition:
        return False, "transition<min"

    # step 2 — row fill
    if cheap["row_fill"] < 0.5:
        return False, "row_fill<0.5"

    # step 3 — tiers
    if cheap["tiers_active"] < min_tiers:
        return False, "tiers<min"

     # ★ NOUVEAU step 3.5 — combo mort : pas de transitions + pas de régularité
    if cheap["s_td"] == 0.0 and cheap["s_hreg"] < 0.20:
        return False, "s_td=0+hreg_dead"

    # ★ NOUVEAU step 3.6 — row_fill dégradé + hreg faible
    if cheap["row_fill"] < 0.90 and cheap["s_hreg"] < 0.15:
        return False, "row_fill_partial+hreg_low"

    """
    # step 3.5 — binaire post-tiers
    if cheap["density_raw"] > 0.65 and cheap["cc_raw"] < 4:
        return False, "density>0.65+cc<4"
    """

    return True, "ok"


# ══════════════════════════════════════════════════════════════
# DÉCISION FINALE  (lecture seule sur cheap + heavy)
# ══════════════════════════════════════════════════════════════
def _decide_final(crop, cheap, heavy, min_proj_score=0.10):
    """
    Step 8 — score + pénalité. ZERO calcul.
    Retourne (result: bool, scores: dict, score_final: float)
    """
    h, w = crop.shape
    score_brut = (
        cheap["s_td"]   * 1.0 +
        heavy["s_vp"]   * 0.5 +
        cheap["s_dens"] * 0.5 +
        cheap["s_hreg"] * 1.5 +
        heavy["s_pf"]   * 0.8 +
        cheap["s_cc"]   * 1.0
    ) / 5.3
    # pénalité FP zones grises
    fp_penalty = 0.0

    ar = w / max(h, 1)
    if ar < 2.0 and cheap["transition_density"] < 0.15:
        fp_penalty += 0.10  # blob carré sans transitions

    if cheap["density_raw"] > 0.85:
        fp_penalty += 0.20   # trop dense → blob solide, pas du texte

    if heavy["vproj"] < 0.01:
        fp_penalty += 0.15   # projection verticale morte → aucune structure colonne

    if cheap["transition_density"] < 0.05 or cheap["transition_density"] > 0.75:
        fp_penalty += 0.15    # transition_density hors range texte

    if h < 6:
        fp_penalty += 0.10    # hauteur trop petite

    if heavy["vproj"] < 0.01 and cheap["s_hreg"] < 0.10:
        fp_penalty += 0.15    # combo : vproj mort ET s_hreg mort → aucune structure

    if heavy["vproj"] < 0.01 and cheap["s_hreg"] > 0.90 and heavy["s_pf"] < 0.75:
        fp_penalty += 0.10    # fausse régularité : s_hreg élevé mais vproj mort

    # cc_raw = 1 → blob unique, pas multi-caractères
    if cheap["cc_raw"] <= 1 and cheap["s_hreg"] < 0.50:
        fp_penalty += 0.05

    # s_hreg nul → aucune régularité horizontale
    if cheap["s_hreg"] == 0.0:
        fp_penalty += 0.10

    # vproj faible + hreg élevé → fausse structure
    if heavy["vproj"] < 0.03 and cheap["s_hreg"] > 0.90:
        fp_penalty += 0.15

     # ══════════════════════════════
    # ★ NOUVELLES PÉNALITÉS FP
    # ══════════════════════════════

    # Combo : transitions mortes + hreg mort → blob sans structure texte
    if cheap["s_td"] == 0.0 and cheap["s_hreg"] < 0.30:
        fp_penalty += 0.20

    # row_fill < 1.0 + hreg faible → zone fragmentée non-texte
    if cheap["row_fill"] < 0.90 and cheap["s_hreg"] < 0.20:
        fp_penalty += 0.10

    # cc_raw extrême haut (> 6) + hreg faible → bruit/texture
    if cheap["cc_raw"] > 6 and cheap["s_hreg"] < 0.20:
        fp_penalty += 0.15

    # s_pf très bas + hreg mort → aucune structure colonne/ligne
    if heavy["s_pf"] < 0.18 and cheap["s_hreg"] < 0.15:
        fp_penalty += 0.10

    # aspect ratio carré + transitions faibles (élargi)
    if ar < 2.5 and cheap["transition_density"] < 0.18 and cheap["s_hreg"] < 0.30:
        fp_penalty += 0.10

    if cheap["row_fill"] < 0.95 and cheap["transition_density"] < 0.20 and cheap["s_hreg"] < 0.15:
        fp_penalty += 0.15

    score = max(0.0, score_brut - fp_penalty)

    scores = {
        "score":              score,
        "score_brut":         score_brut,
        "fp_penalty":         fp_penalty,
        "transition_density": cheap["transition_density"],
        "s_td":               cheap["s_td"],
        "density_raw":        cheap["density_raw"],
        "s_dens":             cheap["s_dens"],
        "cc_raw":             cheap["cc_raw"],
        "s_cc":               cheap["s_cc"],
        "s_hreg":             cheap["s_hreg"],
        "row_fill":           cheap["row_fill"],
        "tiers_active":       cheap["tiers_active"],
        "vproj":              heavy["vproj"],
        "s_vp":               heavy["s_vp"],
        "s_pf":               heavy["s_pf"],
    }

    return score > min_proj_score, scores, score, score_brut, fp_penalty


# ══════════════════════════════════════════════════════════════
# POINT D'ENTRÉE PUBLIC
# ══════════════════════════════════════════════════════════════
def _has_text(roi, x1, y1, x2, y2, cc_stats,min_fill=0.08, min_tiers=2,min_transition=0.20, min_proj_score=0.10):
    """
    Orchestrateur pur — aucun calcul direct ici.
    Séquence : crop → cheap → early → heavy → final
    """
    crop = roi[y1:y2, x1:x2]
    if crop.size == 0:
        return False, _EMPTY_SCORES

    print(f"[SCORE_DEBUG] pos=({x1},{y1},{x2},{y2})")
    print(f"min_fill={min_fill}  min_tiers={min_tiers}  min_transition={min_transition}  min_proj_score={min_proj_score}")

    # ── CHEAP ──
    cheap = _compute_cheap_metrics(crop, cc_stats, x1, y1, x2, y2, min_fill)
    print(f"  [CHEAP] density_raw={cheap['density_raw']:.3f}  "
          f"cc_raw={cheap['cc_raw']:.0f}  "
          f"transition={cheap['transition_density']:.3f}  "
          f"row_fill={cheap['row_fill']:.3f}  "
          f"tiers_active={cheap['tiers_active']}   "
          f"s_hreg={cheap['s_hreg']:.3f}  "
          f"s_td={cheap['s_td']:.3f}  s_dens={cheap['s_dens']:.3f}  s_cc={cheap['s_cc']:.3f}  ")

    # ── EARLY DECISION ──
    passed, reason = _decide_early(cheap, min_transition, min_tiers)
    if not passed:
        print(f"  [EARLY] → rejet ({reason})")
        return False, {**_EMPTY_SCORES,"transition_density": cheap["s_td"],"row_fill":cheap["row_fill"],"density":cheap["s_dens"]}

    print(f"  [EARLY] → passé")

    # ── HEAVY ──
    heavy = _compute_heavy_metrics(crop)
    print(f"  [HEAVY] vproj={heavy['vproj']:.3f}  s_vp={heavy['s_vp']:.3f}  "
          f"s_pf={heavy['s_pf']:.3f}")

    # ── FINAL DECISION ──
    result, scores, score, score_brut, fp_penalty = _decide_final(crop,cheap, heavy, min_proj_score)
    print(f"  [FINAL] score_brut={score_brut:.3f}  "
          f"fp_penalty={fp_penalty:.3f}  "
          f"score_final={score:.3f}  "
          f"→ {'OK' if result else 'rejet'}")

    return result, scores

def _sweep_and_cut(cc_stats, cc_x2, cc_y2, roi,min_text_fill, min_transition, min_proj_score):
    validated = []
    n = len(cc_stats)
    if n == 0:
        return validated

    # Accès direct aux colonnes
    all_x1 = cc_stats[:, cv2.CC_STAT_LEFT]
    all_y1 = cc_stats[:, cv2.CC_STAT_TOP]

    gx1 = int(all_x1[0])
    gy1 = int(all_y1[0])
    gx2 = int(cc_x2[0])
    gy2 = int(cc_y2[0])
    g_ok, g_scores = _has_text(roi, gx1, gy1, gx2, gy2, cc_stats, min_text_fill, 2, min_transition, min_proj_score)

    for i in range(1, n):
        cx1 = int(all_x1[i])
        cy1 = int(all_y1[i])
        cx2 = int(cc_x2[i])
        cy2 = int(cc_y2[i])

        nx1 = min(gx1, cx1)
        ny1 = min(gy1, cy1)
        nx2 = max(gx2, cx2)
        ny2 = max(gy2, cy2)

        valid, scores = _has_text(roi, nx1, ny1, nx2, ny2, cc_stats,min_text_fill, 2, min_transition, min_proj_score)

        if valid:
            gx1, gy1, gx2, gy2 = nx1, ny1, nx2, ny2
            g_ok, g_scores = valid, scores
        else:
            if g_ok:
                validated.append((gx1, gy1, gx2, gy2, g_scores))
            gx1, gy1, gx2, gy2 = cx1, cy1, cx2, cy2
            g_ok, g_scores = _has_text(roi, gx1, gy1, gx2, gy2, cc_stats,min_text_fill, 2, min_transition, min_proj_score)

    if g_ok:
        validated.append((gx1, gy1, gx2, gy2, g_scores))

    return validated

def _validate_text(boxes, mask_white, params, kernels):
    result = []
    min_blob_area  = params["refine_min_blob_area"]
    min_text_fill  = params["min_text_fill"]
    min_transition = params["min_transition"]
    min_proj_score = params["min_proj_score"]
    kernel_rc      = kernels["roi_connected"]

    for box in boxes:
        bx, by, bw, bh = box.rect
        roi = mask_white[by:by+bh, bx:bx+bw]
        if roi.size == 0:
            continue

        # ── Dilatation ──
        roi_connected = cv2.dilate(roi, kernel_rc, iterations=1)

        # ── CC ──
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi_connected, connectivity=4)

        if num_labels <= 1:
            continue

        # ── Filtrage + tri ──
        s = stats[1:]
        valid_mask = s[:, cv2.CC_STAT_AREA] >= min_blob_area
        if not np.any(valid_mask):
            continue

        cc_stats = s[valid_mask].astype(np.int32)

        # Tri par x1
        order = np.argsort(cc_stats[:, cv2.CC_STAT_LEFT])
        cc_stats = cc_stats[order]

        # x2, y2 pré-calculés
        cc_x2 = cc_stats[:, cv2.CC_STAT_LEFT] + cc_stats[:, cv2.CC_STAT_WIDTH]
        cc_y2 = cc_stats[:, cv2.CC_STAT_TOP]  + cc_stats[:, cv2.CC_STAT_HEIGHT]

        groups = _sweep_and_cut(cc_stats, cc_x2, cc_y2, roi,min_text_fill, min_transition, min_proj_score)

        for (gx1, gy1, gx2, gy2, scores) in groups:
            new_box = box.copy_with(x=bx + gx1, y=by + gy1, w=gx2 - gx1, h=gy2 - gy1)
            new_box.scores.update(scores)
            result.append(new_box)

    return result

# ── _validate_background ──
def _validate_background(boxes, mask_white, rgb, params):
    var_norm        = params.get("var_norm", 200.0)
    coherent_delta  = params.get("coherent_delta", 33)
    min_score       = params.get("min_bg_score", 0.5)

    hue_bin_size    = params.get("hue_bin_size", 10)
    max_hue_bins    = params.get("max_hue_bins", 3)
    min_hue_score   = params.get("min_hue_score", 0.1)

    #log.debug(f"[_validate_background] nb_boxes={len(boxes)} var_norm={var_norm} "f"coherent_delta={coherent_delta} min_score={min_score} "f"hue_bin_size={hue_bin_size} max_hue_bins={max_hue_bins}"f"min_hue_score={min_hue_score}")

    result = []
    for box in boxes:
        x, y, w, h = box.rect
        w_pad = 2
        x = max(0, x - w_pad)
        w = w + w_pad * 2

        fond_mask = mask_white[y:y+h, x:x+w] == 0
        fond_count = int(np.count_nonzero(fond_mask))
        if fond_count == 0:
            log.debug(f"  [bg] box({x},{y},{w},{h}) → SKIP fond_count=0")
            continue

        # ── Veto teinte (le plus discriminant, calculé en premier) ──
        rgb_roi = rgb[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(rgb_roi, cv2.COLOR_RGB2HSV)
        fond_hue = hsv_roi[:, :, 0][fond_mask]

        if fond_hue.size > 0:
            binned = fond_hue // hue_bin_size
            bin_counts = np.bincount(binned.ravel(), minlength=180 // hue_bin_size)
            min_bin_count = max(1, int(fond_hue.size * 0.05))
            n_bins = int(np.count_nonzero(bin_counts >= min_bin_count))
            n_bins = max(n_bins, 1)
            s_hue = max(0.0, 1.0 - (n_bins - 1) / max(1, max_hue_bins - 1))
        else:
            n_bins = 0
            s_hue = 1.0
        #log.debug(f"  [bg] box({x},{y},{w},{h}) n_bins={n_bins} s_hue={s_hue:.3f}")
        if s_hue < min_hue_score:
            log.debug(
                f"  [bg] box({x},{y},{w},{h}) n_bins={n_bins} "
                f"s_hue={s_hue:.3f} → REJECT (veto hue)"
            )
            continue

        # ROI directe en gray (évite cvtColor sur RGB)
        gray_roi = cv2.cvtColor(rgb_roi, cv2.COLOR_RGB2GRAY)
        local_mean = cv2.blur(gray_roi, (3, 3))
        diff = cv2.absdiff(gray_roi, local_mean)
        fond_diff = diff[fond_mask]

        # Mesure 1 : variance locale médiane
        median_val = float(np.median(fond_diff))
        median_var = median_val * median_val
        s_var = max(0.0, 1.0 - median_var / var_norm)
        # Mesure 2 : ratio cohérent
        s_coh = float(np.count_nonzero(fond_diff < coherent_delta)) / fond_count

        score = 0.3 * s_var + 0.3 * s_coh + 0.4 * s_hue
        accepted = score >= min_score
        #log.debug(f"  [bg] box({x},{y},{w},{h}) "f"fond_count={fond_count} "f"median_val={median_val:.1f} "f"s_var={s_var:.3f} "f"s_coh={s_coh:.3f} "f"n_bins={n_bins} "f"s_hue={s_hue:.3f} "f"score={score:.3f} "f"→ {'KEEP' if accepted else 'REJECT'}")

        new_box = box.copy_with(x=x, y=y, w=w, h=h)
        new_box.scores["bg_score"] = score

        if accepted:
            result.append(new_box)
    return result

# ── _resolve_overlaps ──
def _edge_confidence(mask_white, x, y, w, h, img_h, img_w):
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

def _resolve_overlaps(boxes, mask_white, params):
    if len(boxes) < 2:
        return list(boxes)
    img_h, img_w = mask_white.shape[:2]
    # ── 1. Calculer le score de chaque box ──
    scored = []
    for box in boxes:
        x, y, w, h = box.rect
        score = _edge_confidence(mask_white, x, y, w, h, img_h, img_w)
        scored.append((box, score))
    # ── 2. Trier par score décroissant ──
    scored.sort(key=lambda s: (s[1], s[0].w * s[0].h), reverse=True)
    # ── 3. Résoudre les chevauchements ──
    min_residual_area = int(params["min_area"] * 0.35)
    result = []
    for box, _ in scored:
        if box is None:
            continue
        cx, cy, cw, ch = box.rect
        # Comparer avec toutes les références déjà validées (score supérieur)
        for ref in result:
            rx, ry, rw, rh = ref.rect
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
            result.append(box.copy_with(x=cx, y=cy, w=cw, h=ch))
    return result

# ── _tight_crop_white ──
def _tight_crop_white(boxes, mask_white):
    """Recadre chaque box au bounding-box réel des pixels blancs qu'elle contient."""
    result = []
    for box in boxes:
        x, y, w, h = box.rect
        roi = mask_white[y:y+h, x:x+w]
        if roi.size == 0 or cv2.countNonZero(roi) == 0:
            continue
        coords = cv2.findNonZero(roi)
        rx, ry, rw, rh = cv2.boundingRect(coords)
        result.append(box.copy_with(x=x + rx, y=y + ry, w=rw, h=rh))
    return result

# ── _adjust_resolve ──
def _adjust_resolve(boxes, mask_white, h_img, params, resolve=True):
    adjusted = _adjust_boxes(boxes, mask_white, h_img, params)
    if resolve:
        #cropped  = _tight_crop_white(resolved, mask_white)
        adjusted = _resolve_overlaps(adjusted, mask_white, params)
    return adjusted

# ── _expand_plates ──
def _expand_plates(boxes, img):
    img_h, img_w = img.shape[:2]
    expanded = []
    for box in boxes:
        x, y, w, h = box.rect
        nx = max(x - 2, 0)
        ny = max(y - 1, 0)
        nx2 = min(x + w + 2, img_w)
        ny2 = min(y + h + 1, img_h)
        expanded.append(box.copy_with(x=nx, y=ny, w=nx2 - nx, h=ny2 - ny))
    return expanded

# ── _filter_geometry ──
def _filter_geometry(boxes, masked, params):
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
    for box in boxes:
        x, y, w, h = box.rect
        #log.info(f"_filter_geometry: x={x} y={y} w={w} h={h}")
        if w < min_width or h < min_height:
            #log.info(f"_filter_geometry: w={w} < min_width={min_width} or h={h} < min_height={min_height}")
            continue
        area = w * h
        if area < min_area or area > max_area:
            #log.info(f"_filter_geometry: min_area={min_area}  area={area} max_area={max_area}")
            continue
        ratio = w / h  # h >= min_height > 0, pas besoin de max(h,1)

        if ratio < min_ratio or ratio > max_ratio:
            #log.info(f"_filter_geometry: min_ratio={min_ratio}   ratio={ratio}  max_ratio={max_ratio}")
            continue
        fill = cv2.countNonZero(masked[y:y+h, x:x+w]) / area
        if fill < min_fill or fill > max_fill:
            #log.info(f"_filter_geometry: min_fill={min_fill} fill={fill}  max_fill={max_fill}")
            continue
        plates.append(box)
    return plates

# ── _filter_horizontal_alignment ──
def _filter_horizontal_alignment(boxes, mask_white, params):
    max_y_std_ratio = params.get("align_max_y_std_ratio", 0.20)
    min_cols        = params.get("align_min_cols", 3)
    min_col_fill    = params.get("align_min_col_fill", 0.15)

    kept = []
    for box in boxes:
        x, y, w, h = box.rect
        roi = mask_white[y:y+h, x:x+w]

        min_col_px = max(2, int(h * min_col_fill))

        # ── Vectorisé : plus de boucle Python ──
        # Masque binaire 0/1
        binary = (roi > 0).astype(np.float32)

        # Nombre de pixels blancs par colonne
        col_counts = binary.sum(axis=0)                    # shape (w,)

        # Indice Y de chaque ligne → broadcast sur toutes les colonnes
        y_indices = np.arange(h, dtype=np.float32)[:, None]  # shape (h,1)

        # Somme pondérée des Y par colonne
        y_sum = (binary * y_indices).sum(axis=0)           # shape (w,)

        # Colonnes actives = assez de pixels
        active = col_counts >= min_col_px

        if active.sum() < min_cols:
            kept.append(box)
            continue

        # Centre de masse Y par colonne active
        y_means = y_sum[active] / col_counts[active]

        y_std_ratio = float(np.std(y_means)) / h

        if y_std_ratio > max_y_std_ratio:
            continue

        kept.append(box)

    return kept

# ── _filter_horizontal_bands ──
def _filter_horizontal_bands(boxes, mask_white, params):
    min_fill  = params.get("bands_min_fill", 0.15)
    gap_fill  = params.get("bands_gap_fill", 0.08)
    max_bands = params.get("bands_max_bands", 2)

    #log.debug(f"bands: min_fill={min_fill} gap_fill={gap_fill} max_bands={max_bands}")

    kept = []
    for box in boxes:
        x, y, w, h = box.rect
        roi = mask_white[y:y+h, x:x+w]

        if w == 0:
            log.debug(f"  bands_skip: {box.rect} w=0")
            continue

        row_sums = np.count_nonzero(roi, axis=1).astype(np.float32) / w

        band_count = 0
        in_band = False

        for fill in row_sums:
            if fill >= min_fill:
                if not in_band:
                    band_count += 1
                    in_band = True
            elif fill < gap_fill:
                in_band = False
            # Entre gap_fill et min_fill : hystérésis, on reste dans l'état courant
            # C'est voulu : évite de compter des micro-bandes sur du bruit

        #log.debug(f"  box {box.rect} bands={band_count}")

        if band_count > max_bands:
            log.debug(f"  bands_reject: {box.rect} bands={band_count}")
            continue

        kept.append(box)

    return kept

# ── _filter_perspective_gradient ──
def _filter_perspective_gradient(boxes, mask_white, params):
    n_zones        = params.get("n_zones", 4)
    min_drop_ratio = params.get("min_drop_ratio", 0.50)
    min_zone_width = params.get("min_zone_width", 8)

    #log.debug(f"gradient: n_zones={n_zones} min_drop_ratio={min_drop_ratio} min_zone_w={min_zone_width}")

    kept = []
    for box in boxes:
        x, y, w, h = box.rect

        zone_w = w // n_zones
        #log.debug(f"  box {box.rect} zone_w={zone_w}")

        if zone_w < min_zone_width:
            # Box trop étroite pour découper en zones fiables → on garde
            log.debug(f"  gradient_skip: {box.rect} zone_w={zone_w} < {min_zone_width}")
            kept.append(box)
            continue

        roi = mask_white[y:y+h, x:x+w]

        densities = []
        skip = False
        for i in range(n_zones):
            x0 = i * zone_w
            # FIX: dernière zone étendue jusqu'au bord droit
            x1 = x0 + zone_w if i < n_zones - 1 else w
            zone = roi[:, x0:x1]
            total = zone.size
            if total == 0:
                log.debug(f"  gradient_skip: {box.rect} zone {i} empty")
                kept.append(box)
                skip = True
                break
            densities.append(np.count_nonzero(zone) / total)

        if skip:
            continue

        # FIX: monotonie non-stricte avec au moins une variation réelle
        diffs = [densities[i+1] - densities[i] for i in range(n_zones - 1)]
        all_decreasing = all(d <= 0 for d in diffs) and any(d < 0 for d in diffs)
        all_increasing = all(d >= 0 for d in diffs) and any(d > 0 for d in diffs)

        if all_decreasing or all_increasing:
            max_density = max(densities)
            if max_density > 0:
                drop_ratio = abs(densities[0] - densities[-1]) / max_density
                log.debug(f"  densities={[f'{d:.3f}' for d in densities]} drop={drop_ratio:.3f}")
                if drop_ratio > min_drop_ratio:
                    log.debug(f"  gradient_reject: {box.rect} drop={drop_ratio:.2f}")
                    continue

        kept.append(box)

    return kept

# ── _make_template ──
def _make_template( boxes, frame):
    x, y, w, h = boxes
    crop = frame[y:y+h, x:x+w]
    if len(crop.shape) == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return crop.copy()


def process_channel(masked,rgb, mask_white, h_img, params, kernels):
    plates = []

    log.debug("boxes")
    boxes = _extract_raw_boxes(masked, params)
    log.debug("boxes_ar")
    boxes_ar = _adjust_resolve(boxes, mask_white, h_img, params)
    log.debug("_filter_geometry")
    geometryed = _filter_geometry(boxes_ar, masked, params)

    log.debug("_split_wide_boxes")
    split = _split_wide_boxes(geometryed, mask_white, params)
    log.debug("split_ar")
    split_ar = _adjust_resolve(split, mask_white, h_img, params, resolve=False)

    log.debug("_validate_background")
    validated_b = _validate_background(split_ar,mask_white, rgb, params)

    log.debug("_validate_text")
    validated_t = _validate_text(validated_b, mask_white, params, kernels)
    log.debug("_merge_nearby_horizontal")
    merge = _merge_nearby_horizontal(validated_t, params["max_gap_x"],params["max_gap_y"])

    log.debug("merge_ar")
    merge_ar = _adjust_resolve(merge, mask_white, h_img, params, resolve=False)
    log.debug("_expand_plates")
    expanded = _expand_plates(merge_ar,rgb)

    log.debug("_filter_horizontal_bands")
    banded = _filter_horizontal_bands(expanded, mask_white, params)
    log.debug("_filter_horizontal_alignment")
    aligned = _filter_horizontal_alignment(banded, mask_white, params)
    log.debug("_filter_perspective_gradient")
    plates = _filter_perspective_gradient(aligned, mask_white, params)

    log.debug("_make_template")
    for box in plates:
        box.template = _make_template(box.rect, rgb)
        print("box.x, box.y, box.w, box.h",box.x, box.y, box.w, box.h)
        print("box.confidence",box.confidence)
        print("box.scores",box.scores)


    return plates

def process_channel_test(masked, rgb, mask_white, h_img, params, kernels):
    import time

    def _t(name, fn, *args, **kwargs):
        t = time.perf_counter()
        r = fn(*args, **kwargs)
        ms = (time.perf_counter() - t) * 1000
        log.info(f"  {name:<30s} {ms:6.2f}ms")
        return r

    log.info("── process_channel breakdown ──")

    # ── PHASE 1 : extraction + filtre géométrique immédiat (quasi gratuit) ──
    boxes        = _t("_extract_raw_boxes",       _extract_raw_boxes, masked, params)
    boxes_ar     = _t("_adjust_resolve_1",        _adjust_resolve, boxes, mask_white, h_img, params)
    geometryed   = _t("_filter_geometry",         _filter_geometry, boxes_ar, masked, params)

    # ── PHASE 2 : ajustement sur le set réduit ──
    split        = _t("_split_wide_boxes",        _split_wide_boxes, geometryed, mask_white, params)
    split_ar     = _t("_adjust_resolve_2",        _adjust_resolve, split, mask_white, h_img, params, resolve=False)

    # ── PHASE 3 : veto hue rapide AVANT _validate_text (le plus coûteux) ──
    validated_b  = _t("_validate_background",     _validate_background, split_ar, mask_white, rgb, params)

    # ── PHASE 4 : _validate_text sur set déjà filtré ──
    validated_t  = _t("_validate_text",           _validate_text, validated_b, mask_white, params, kernels)
    merge        = _t("_merge_nearby_horizontal",            _merge_nearby_horizontal, validated_t, params["max_gap_x"], params["max_gap_y"])

    # ── PHASE 5 : ajustement final + filtres fins ──
    merge_ar     = _t("_adjust_resolve_3",        _adjust_resolve, merge, mask_white, h_img, params, resolve=False)
    expanded     = _t("_expand_plates",           _expand_plates, merge_ar, rgb)
    banded       = _t("_filter_horizontal_bands", _filter_horizontal_bands, expanded, mask_white, params)
    aligned      = _t("_filter_horizontal_alignment", _filter_horizontal_alignment, banded, mask_white, params)
    plates       = _t("_filter_perspective_gradient",      _filter_perspective_gradient, aligned, mask_white, params)

    """
    screen = rgb.copy()
    write_rects(screen, plates, get_color("vert"),1)

    cv2.imshow("screen", screen)
    cv2.waitKey(0)
    """
    for box in plates:
        box.template = _make_template(box.rect, rgb)

    return plates

