# mask_manager.py — v1
import logging
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from config import cfg

log = logging.getLogger("mask_manager")

_next_mask_id = 0

# ═══════════════════════════════════════════════════════
#  GÉOMÉTRIE
# ═══════════════════════════════════════════════════════

def compute_iou(r1, r2):
    """IoU entre deux rectangles (x,y,w,h)."""
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    xa = max(x1, x2);  ya = max(y1, y2)
    xb = min(x1+w1, x2+w2); yb = min(y1+h1, y2+h2)
    inter = max(0, xb-xa) * max(0, yb-ya)
    if inter == 0:
        return 0.0
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0.0

def center_distance(r1, r2):
    """Distance euclidienne entre centres (x,y,w,h)."""
    cx1 = r1[0] + r1[2]/2;  cy1 = r1[1] + r1[3]/2
    cx2 = r2[0] + r2[2]/2;  cy2 = r2[1] + r2[3]/2
    return ((cx1-cx2)**2 + (cy1-cy2)**2)**0.5

def rect_corners(r):
    x, y, w, h = r
    return [(x,y), (x+w,y), (x,y+h), (x+w,y+h)]

def corners_distance(r1, r2):
    """Distance moyenne entre les 4 coins."""
    c1 = rect_corners(r1);  c2 = rect_corners(r2)
    return sum(((ax-bx)**2+(ay-by)**2)**0.5
               for (ax,ay),(bx,by) in zip(c1,c2)) / 4.0

def pad_rect(x, y, w, h, screen_w, screen_h):
    """Applique le padding config et clamp aux bornes écran."""
    pad_x = cfg.get("masks.pad_x", 0)
    pad_y = cfg.get("masks.pad_y", 0)
    x2 = max(0, x - pad_x)
    y2 = max(0, y - pad_y)
    w2 = min(screen_w - x2, w + 2*pad_x)
    h2 = min(screen_h - y2, h + 2*pad_y)
    return (x2, y2, w2, h2)

# ═══════════════════════════════════════════════════════
#  TTL / COULEURS DEBUG
# ═══════════════════════════════════════════════════════

def ttl_color(ttl):
    if ttl >= 3:
        return tuple(cfg.get("debug.colors.fresh"))
    elif ttl >= 2:
        return tuple(cfg.get("debug.colors.persist"))
    else:
        return tuple(cfg.get("debug.colors.dying"))

def ttl_label(ttl, source="S"):
    return f"TTL={ttl} [{source}]"

# ═══════════════════════════════════════════════════════
#  MATCHING HONGROIS
# ═══════════════════════════════════════════════════════

def match_hungarian(active_masks, new_rects):
    """
    Matching hongrois basé sur IoU ou distance selon config.
    Retourne (matched_pairs, unmatched_masks, unmatched_detections)
    matched_pairs : list[(mask_idx, det_idx)]
    """
    if not active_masks or not new_rects:
        return [], list(range(len(active_masks))), list(range(len(new_rects)))

    mode       = cfg.get("matching.mode", "iou")
    iou_thresh = cfg.get("matching.iou_thresh", 0.1)
    dist_thresh= cfg.get("matching.dist_thresh", 80)

    n = len(active_masks)
    m = len(new_rects)
    cost = np.zeros((n, m), dtype=np.float32)

    for i, mask in enumerate(active_masks):
        for j, det in enumerate(new_rects):
            if mode == "iou":
                cost[i, j] = 1.0 - compute_iou(mask['rect'], det)
            else:
                cost[i, j] = center_distance(mask['rect'], det)

    row_ind, col_ind = linear_sum_assignment(cost)

    matched, unmatched_masks, unmatched_dets = [], [], []

    matched_rows = set()
    matched_cols = set()

    for r, c in zip(row_ind, col_ind):
        if mode == "iou":
            valid = (1.0 - cost[r, c]) >= iou_thresh
        else:
            valid = cost[r, c] <= dist_thresh
        if valid:
            matched.append((r, c))
            matched_rows.add(r)
            matched_cols.add(c)

    unmatched_masks = [i for i in range(n) if i not in matched_rows]
    unmatched_dets  = [j for j in range(m) if j not in matched_cols]

    return matched, unmatched_masks, unmatched_dets


def match_hungarian_ambiguity(active_masks, new_rects):
    """
    Matching hongrois avec rejet des matchs ambigus (ratio top-2).
    """
    matched, unmatched_masks, unmatched_dets = match_hungarian(active_masks, new_rects)

    if not active_masks or not new_rects:
        return matched, unmatched_masks, unmatched_dets

    mode          = cfg.get("matching.mode", "iou")
    ambiguity_ratio = cfg.get("matching.ambiguity_ratio", 0.0)
    if ambiguity_ratio <= 0.0:
        return matched, unmatched_masks, unmatched_dets

    n = len(active_masks)
    m = len(new_rects)
    cost = np.zeros((n, m), dtype=np.float32)

    for i, mask in enumerate(active_masks):
        for j, det in enumerate(new_rects):
            if mode == "iou":
                cost[i, j] = 1.0 - compute_iou(mask['rect'], det)
            else:
                cost[i, j] = center_distance(mask['rect'], det)

    ambiguous_rows = set()
    if m >= 2:
        for r, c in matched:
            row_costs = np.sort(cost[r])
            best  = row_costs[0]
            second= row_costs[1]
            if best < 1e-9:
                continue
            if second / best < (1.0 + ambiguity_ratio):
                ambiguous_rows.add(r)

    filtered_matched = [(r, c) for r, c in matched if r not in ambiguous_rows]
    extra_unmatched_m = [r for r, c in matched if r in ambiguous_rows]
    extra_unmatched_d = [c for r, c in matched if r in ambiguous_rows]

    return (
        filtered_matched,
        unmatched_masks + extra_unmatched_m,
        unmatched_dets  + extra_unmatched_d,
    )

# ═══════════════════════════════════════════════════════
#  CYCLE DE VIE MASQUES
# ═══════════════════════════════════════════════════════

def _new_mask(rect, detect_ts):
    global _next_mask_id
    uid = _next_mask_id
    _next_mask_id += 1
    return {
        'uid':               uid,
        'rect':              rect,
        'last_detected_rect': rect,
        'last_detected_ts':  detect_ts,
        'last_source':       'new',
        'ttl':               cfg.get("masks.ttl_max"),
        'vx':                0.0,
        'vy':                0.0,
        'confidence':        0.0,
        'template':          None,
        'fast_miss_count':   0,
    }


def update_mask(mask, new_rect, detect_ts, source):
    """Met à jour un masque existant (smooth EMA + vélocité)."""
    smooth_alpha = cfg.get("masks.smooth_alpha", 1.0)
    dead_zone    = cfg.get("masks.dead_zone", 3)
    vel_dz       = cfg.get("masks.velocity_dead_zone", 5)

    nx, ny, nw, nh = new_rect
    ox, oy, ow, oh = mask['rect']

    # dead zone : pas de mouvement significatif
    if (abs(nx-ox) < dead_zone and abs(ny-oy) < dead_zone
            and abs(nw-ow) < dead_zone and abs(nh-oh) < dead_zone):
        if source == "slow":
            mask['ttl'] = cfg.get("masks.ttl_max")
        mask['fast_miss_count'] = 0
        mask['last_detected_ts'] = detect_ts
        mask['last_source'] = source
        return

    # vélocité filtrée
    lx, ly, _, _ = mask['last_detected_rect']
    dt = detect_ts - mask['last_detected_ts']
    if dt > 0.001:
        mask['vx'] = 0.0 if abs(nx-lx) < vel_dz else (nx-lx)/dt
        mask['vy'] = 0.0 if abs(ny-ly) < vel_dz else (ny-ly)/dt

    # EMA
    a = smooth_alpha
    mask['rect'] = (
        ox + a*(nx-ox),
        oy + a*(ny-oy),
        ow + a*(nw-ow),
        oh + a*(nh-oh),
    )

    if source == "slow":
        mask['ttl'] = cfg.get("masks.ttl_max")

    mask['last_detected_rect'] = new_rect
    mask['last_detected_ts']   = detect_ts
    mask['last_source']        = source
    mask['fast_miss_count']    = 0


def match_and_update(active_masks, new_rects, detect_ts, source):
    """
    Matching + update/création.
    Retourne la liste des uid assignés (None si nouveau masque créé).
    """
    use_ambiguity = cfg.get("matching.use_ambiguity", False)
    if use_ambiguity:
        matched, unmatched_masks, unmatched_dets = match_hungarian_ambiguity(active_masks, new_rects)
    else:
        matched, unmatched_masks, unmatched_dets = match_hungarian(active_masks, new_rects)

    assigned_uids = [None] * len(new_rects)

    for mask_idx, det_idx in matched:
        update_mask(active_masks[mask_idx], new_rects[det_idx], detect_ts, source)
        assigned_uids[det_idx] = active_masks[mask_idx]['uid']

    for det_idx in unmatched_dets:
        new_m = _new_mask(new_rects[det_idx], detect_ts)
        active_masks.append(new_m)
        assigned_uids[det_idx] = new_m['uid']
        log.debug(f"Nouveau masque uid={new_m['uid']} rect={new_rects[det_idx]}")

    return assigned_uids

# ═══════════════════════════════════════════════════════
#  PRÉDICTION
# ═══════════════════════════════════════════════════════

def predict_masks(active_masks, updated_uids, now, screen_w, screen_h):
    """
    Prédit la position des masques non mis à jour.
    Retourne le nombre de masques prédits.
    """
    count = 0
    for m in active_masks:
        if m['uid'] in updated_uids:
            continue
        count += 1
        dt          = now - m['last_detected_ts']
        dt_capped   = min(dt, 0.10)
        damping     = max(0.0, 1.0 - dt * 2.0)
        lx, ly, _, _= m['last_detected_rect']
        w, h        = m['rect'][2], m['rect'][3]
        x = lx + m['vx'] * dt_capped * damping
        y = ly + m['vy'] * dt_capped * damping
        x = max(0.0, min(x, screen_w - w))
        y = max(0.0, min(y, screen_h - h))
        m['rect'] = (x, y, w, h)
    return count

# ═══════════════════════════════════════════════════════
#  KILL FAST MISS
# ═══════════════════════════════════════════════════════

def kill_fast_miss(active_masks, now):
    """
    Décrémente TTL selon fast_miss_count ou timeout.
    Retourne la liste filtrée (ttl > 0).
    """
    fast_miss_thresh  = cfg.get("masks.fast_miss_threshold", 5)
    fast_miss_timeout = cfg.get("masks.fast_miss_timeout_ms", 300) / 1000.0

    for m in active_masks:
        time_since = now - m['last_detected_ts']
        if m['fast_miss_count'] >= fast_miss_thresh:
            m['ttl'] -= 1
            m['fast_miss_count'] = 0
        elif time_since >= fast_miss_timeout:
            m['ttl'] -= 1
            m['fast_miss_count'] = 0
            m['last_detected_ts'] = now

    return [m for m in active_masks if m['ttl'] > 0]

# ═══════════════════════════════════════════════════════
#  JITTER + AGE
# ═══════════════════════════════════════════════════════

def compute_jitter(active_masks, rects_before):
    """
    Calcule le jitter centre/coins et compte créations/kills.
    Retourne (center_avg, corners_avg, masks_created, masks_killed).
    """
    after_uids       = {m['uid'] for m in active_masks}
    jitter_c_sum     = 0.0
    jitter_4_sum     = 0.0
    jitter_n         = 0
    masks_created    = 0
    masks_killed     = 0

    for m in active_masks:
        uid = m['uid']
        if uid in rects_before:
            jitter_c_sum += center_distance(rects_before[uid], m['rect'])
            jitter_4_sum += corners_distance(rects_before[uid], m['rect'])
            jitter_n += 1
        else:
            masks_created += 1

    for uid in rects_before:
        if uid not in after_uids:
            masks_killed += 1

    center_avg  = (jitter_c_sum / jitter_n) if jitter_n > 0 else 0.0
    corners_avg = (jitter_4_sum / jitter_n) if jitter_n > 0 else 0.0
    return center_avg, corners_avg, masks_created, masks_killed


def compute_mask_age(active_masks, now):
    """Age moyen des masques en ms. Retourne 0.0 si aucun masque."""
    if not active_masks:
        return 0.0
    return sum((now - m['last_detected_ts']) * 1000 for m in active_masks) / len(active_masks)

# ═══════════════════════════════════════════════════════
#  DEBUG DRAW
# ═══════════════════════════════════════════════════════

def draw_debug(frame, active_masks):
    """Dessine les rectangles et labels TTL sur le frame (in-place)."""
    for m in active_masks:
        x, y, w, h = (int(v) for v in m['rect'])
        color = ttl_color(m['ttl'])
        source = m.get('last_source', '?')[0].upper()
        label  = ttl_label(m['ttl'], source)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame, label,
            (x, max(y-5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            cv2.LINE_AA,
        )
