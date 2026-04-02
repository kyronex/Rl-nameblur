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
#  CYCLE DE VIE MASQUES
# ═══════════════════════════════════════════════════════

def _new_mask(rect, detect_ts, now=None):
    global _next_mask_id
    uid = _next_mask_id
    _next_mask_id += 1
    creation_ts = max(now, detect_ts) if now is not None else detect_ts
    return {
        'uid':               uid,
        'rect':              rect,
        'last_detected_rect': rect,
        'last_detected_ts':  creation_ts,
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
    vel_dz       = cfg.get("masks.velocity_dead_zone", 10)

    dt_slow_max      = cfg.get("masks.dt_slow_max", 500) / 1000.0   # ms → s
    teleport_thresh  = cfg.get("masks.teleport_thresh", 300)         # px
    vx_max           = cfg.get("masks.vx_max", 4000)                 # px/s
    vy_max           = cfg.get("masks.vy_max", 2000)                 # px/s

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

     # ── Fix 3 : vélocité calculée UNIQUEMENT sur base Slow ──
    if source == "slow":
        lx, ly, _, _ = mask['last_detected_rect']
        dt = detect_ts - mask['last_detected_ts']

        if dt > 0.001:
            delta_x = nx - lx
            delta_y = ny - ly

            # ── Amendement B : reset si dt trop grand (cut caméra) ──
            if dt > dt_slow_max:
                log.debug(
                    f"uid={mask['uid']} Amend-B reset vx/vy "
                    f"dt={dt*1000:.0f}ms > dt_slow_max={dt_slow_max*1000:.0f}ms"
                )
                mask['vx'] = 0.0
                mask['vy'] = 0.0

            # ── Amendement C : reset si saut spatial (téléportation) ──
            elif abs(delta_x) > teleport_thresh or abs(delta_y) > teleport_thresh:
                log.debug(
                    f"uid={mask['uid']} Amend-C reset vx/vy "
                    f"dx={delta_x:.0f} dy={delta_y:.0f} > thresh={teleport_thresh}"
                )
                mask['vx'] = 0.0
                mask['vy'] = 0.0

            else:
                raw_vx = 0.0 if abs(delta_x) < vel_dz else delta_x / dt
                raw_vy = 0.0 if abs(delta_y) < vel_dz else delta_y / dt

                # ── Amendement A : borne vx/vy max ──
                mask['vx'] = max(-vx_max, min(raw_vx, vx_max))
                mask['vy'] = max(-vy_max, min(raw_vy, vy_max))

                if raw_vx != mask['vx'] or raw_vy != mask['vy']:
                    log.debug(
                        f"uid={mask['uid']} Amend-A clamp "
                        f"vx {raw_vx:.0f}→{mask['vx']:.0f} "
                        f"vy {raw_vy:.0f}→{mask['vy']:.0f}"
                    )

        # ── Fix 3 : last_detected_rect mis à jour Slow seulement ──
        mask['last_detected_rect'] = new_rect

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

    mask['last_detected_ts']   = detect_ts
    mask['last_source']        = source
    mask['fast_miss_count']    = 0

def increment_fast_miss(active_masks, updated_uids):
    """
    À appeler UNE FOIS par frame dans main, après fast poll.
    Incrémente fast_miss_count uniquement pour les masques
    non mis à jour ce cycle.
    NE décrémente PAS TTL — c'est le rôle exclusif de kill_fast_miss.
    """
    for m in active_masks:
        if m['uid'] not in updated_uids:
            m['fast_miss_count'] += 1

# ═══════════════════════════════════════════════════════
#  MATCHING / HONGROIS
# ═══════════════════════════════════════════════════════
# ── Extracteurs communs ───────────────────────────────────────────────────────

def _extract_centers_masks(active_masks):
    return np.array(
        [
            (mask['rect'][0] + mask['rect'][2] / 2.0,
             mask['rect'][1] + mask['rect'][3] / 2.0)
            for mask in active_masks
        ],
        dtype=np.float32
    )


def _extract_centers_dets(new_rects):
    return np.array(
        [
            (det[0] + det[2] / 2.0,
             det[1] + det[3] / 2.0)
            for det in new_rects
        ],
        dtype=np.float32
    )

def _precompute_masks_np(active_masks):
    rects = np.array(
        [mask['rect'] for mask in active_masks],
        dtype=np.float32
    )                                                      # (n, 4)
    centers = np.stack(
        [
            rects[:, 0] + rects[:, 2] / 2.0,
            rects[:, 1] + rects[:, 3] / 2.0
        ],
        axis=1
    )                                                      # (n, 2)
    return rects, centers
# ── Mode distance ─────────────────────────────────────────────────────────────

def _cost_distance(masks_np, new_rects):
    _, centers_masks  = masks_np                          # (n, 2) — pré-calculé
    centers_dets      = _extract_centers_dets(new_rects)  # (m, 2)
    diff              = centers_masks[:, None, :] - centers_dets[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


# ── Mode IoU ──────────────────────────────────────────────────────────────────

def _cost_iou(active_masks, new_rects):
    rects_masks = np.array([mask['rect'] for mask in active_masks],dtype=np.float32)                          # (n, 4)
    rects_dets = np.array(new_rects,dtype=np.float32)                                                      # (m, 4)

    # Coins masks (n,)
    mx1 = rects_masks[:, 0]
    my1 = rects_masks[:, 1]
    mx2 = rects_masks[:, 0] + rects_masks[:, 2]
    my2 = rects_masks[:, 1] + rects_masks[:, 3]

    # Coins dets (m,)
    dx1 = rects_dets[:, 0]
    dy1 = rects_dets[:, 1]
    dx2 = rects_dets[:, 0] + rects_dets[:, 2]
    dy2 = rects_dets[:, 1] + rects_dets[:, 3]

    # Intersection → (n, m)
    inter_x1 = np.maximum(mx1[:, None], dx1[None, :])
    inter_y1 = np.maximum(my1[:, None], dy1[None, :])
    inter_x2 = np.minimum(mx2[:, None], dx2[None, :])
    inter_y2 = np.minimum(my2[:, None], dy2[None, :])

    inter_w    = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h    = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_masks = rects_masks[:, 2] * rects_masks[:, 3]    # (n,)
    area_dets  = rects_dets[:, 2]  * rects_dets[:, 3]     # (m,)

    union_area = area_masks[:, None] + area_dets[None, :] - inter_area

    iou = np.where(union_area > 0.0, inter_area / union_area, 0.0)
    return (1.0 - iou).astype(np.float32)

# ── Fallback sécurité (mode inconnu) ─────────────────────────────────────────

def _cost_fallback(masks_np, new_rects):
    rects_masks = masks_np[0]                              # (n, 4) — pré-calculé
    n = len(rects_masks)
    m = len(new_rects)
    cost = np.zeros((n, m), dtype=np.float32)
    for i, rect in enumerate(rects_masks):
        for j, det in enumerate(new_rects):
            cost[i, j] = center_distance(rect, det)
    return cost


def _build_cost_matrix(active_masks, new_rects, mode):
    masks_np = _precompute_masks_np(active_masks)          # 1 seule extraction numpy
    if mode == "distance":
        return _cost_distance(masks_np, new_rects)
    elif mode == "iou":
        return _cost_iou(masks_np, new_rects)
    else:
        return _cost_fallback(masks_np, new_rects)



def _apply_threshold(cost_value, mode, iou_thresh, dist_thresh):
    """
    Retourne True si le match est valide selon le seuil.
    mode='iou'      → (1 - cost) >= iou_thresh
    mode='distance' → cost <= dist_thresh
    """
    if mode == "iou":
        return (1.0 - cost_value) >= iou_thresh
    return cost_value <= dist_thresh

def match_hungarian(cost, row_ind, col_ind, n, m, mode, iou_thresh, dist_thresh):
    """
    Matching hongrois basé sur IoU ou distance selon config.
    Reçoit cost + résultat linear_sum_assignment déjà calculés.
    Retourne (matched_pairs, unmatched_masks, unmatched_detections)
    matched_pairs : list[(mask_idx, det_idx)]
    """
    matched       = []
    matched_rows  = set()
    matched_cols  = set()

    for r, c in zip(row_ind, col_ind):
        if _apply_threshold(cost[r, c], mode, iou_thresh, dist_thresh):
            matched.append((int(r), int(c)))   # int() — évite np.int64 dans les listes
            matched_rows.add(int(r))
            matched_cols.add(int(c))

    # set-difference numpy → évite double boucle Python range()
    all_rows        = set(range(n))
    all_cols        = set(range(m))
    unmatched_masks = sorted(all_rows - matched_rows)  # list[int] trié
    unmatched_dets  = sorted(all_cols - matched_cols)  # list[int] trié

    return matched, unmatched_masks, unmatched_dets

def match_hungarian_ambiguity(cost, row_ind, col_ind, n, m, mode, iou_thresh, dist_thresh, ambiguity_ratio):
    """
    Matching hongrois avec rejet des matchs ambigus (ratio top-2).
    Reçoit cost + résultat linear_sum_assignment déjà calculés.
    """
    matched, unmatched_masks, unmatched_dets = match_hungarian(
        cost, row_ind, col_ind, n, m, mode, iou_thresh, dist_thresh
    )

    if ambiguity_ratio <= 0.0 or m < 2:
        return matched, unmatched_masks, unmatched_dets

    ambiguous_rows = set()
    for r, c in matched:
        row_costs = np.sort(cost[r])
        best   = row_costs[0]
        second = row_costs[1]
        if best < 1e-9:
            continue
        if second / best < (1.0 + ambiguity_ratio):
            ambiguous_rows.add(r)

    filtered_matched  = [(r, c) for r, c in matched if r not in ambiguous_rows]
    extra_unmatched_m = [r       for r, c in matched if r in ambiguous_rows]
    extra_unmatched_d = [c       for r, c in matched if r in ambiguous_rows]

    return (
        filtered_matched,
        unmatched_masks + extra_unmatched_m,
        unmatched_dets  + extra_unmatched_d,
    )

def match_and_update(active_masks, new_rects, detect_ts, source, now=None):
    """
    Matching + update/création.
    Retourne la liste des uid assignés (None si nouveau masque créé).
    """

    assigned_uids = [None] * len(new_rects)

    if not active_masks or not new_rects:
        for det_idx in range(len(new_rects)):
            new_m = _new_mask(new_rects[det_idx], detect_ts, now=now)
            active_masks.append(new_m)
            assigned_uids[det_idx] = new_m['uid']
            log.debug(f"Nouveau masque uid={new_m['uid']} rect={new_rects[det_idx]}")
        return assigned_uids

    # ── paramètres config ──────────────────────────────
    mode            = cfg.get("matching.mode",            "distance")
    iou_thresh      = cfg.get("matching.iou_thresh",      0.15)
    dist_thresh     = cfg.get("matching.dist_thresh",     60)
    use_ambiguity   = cfg.get("matching.use_ambiguity",   False)
    ambiguity_ratio = cfg.get("matching.ambiguity_ratio", 0.0)

    n = len(active_masks)
    m = len(new_rects)

    # ── 1 seul calcul matrice ──────────────────────────
    cost = _build_cost_matrix(active_masks, new_rects, mode)

    # ── 1 seul appel hongrois ──────────────────────────
    if n == 1 and m == 1:
        row_ind = np.array([0], dtype=np.intp)
        col_ind = np.array([0], dtype=np.intp)
    else:
        row_ind, col_ind = linear_sum_assignment(cost)

    # ── sélection branche ─────────────────────────────
    if use_ambiguity:
        matched, unmatched_masks, unmatched_dets = match_hungarian_ambiguity(cost, row_ind, col_ind, n, m, mode, iou_thresh, dist_thresh, ambiguity_ratio)
    else:
        matched, unmatched_masks, unmatched_dets = match_hungarian(cost, row_ind, col_ind, n, m, mode, iou_thresh, dist_thresh)

    # ── update masques matchés ─────────────────────────
    for mask_idx, det_idx in matched:
        update_mask(active_masks[mask_idx], new_rects[det_idx], detect_ts, source)
        assigned_uids[det_idx] = active_masks[mask_idx]['uid']

    # ── création nouveaux masques ──────────────────────
    for det_idx in unmatched_dets:
        new_m = _new_mask(new_rects[det_idx], detect_ts, now=now)
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

    """
    def predict_masks(active_masks, updated_uids, now, screen_w, screen_h):

    #Prédit la position des masques non mis à jour.
    #Base = last_detected_rect → ancrage fixe, pas de dérive cumulée.
    #Retourne le nombre de masques prédits.

    count = 0
    for m in active_masks:
        if m['uid'] in updated_uids:
            continue
        count += 1
        dt        = now - m['last_detected_ts']
        dt_capped = min(dt, 0.10)
        damping   = max(0.0, 1.0 - dt * 2.0)

        # ── M3 : ancrage last_detected_rect ──────────────────────
        lx, ly, w, h = m['last_detected_rect']   # ← w,h depuis last_detected
        # ─────────────────────────────────────────────────────────

        x = lx + m['vx'] * dt_capped * damping
        y = ly + m['vy'] * dt_capped * damping
        x = max(0.0, min(x, screen_w - w))
        y = max(0.0, min(y, screen_h - h))
        m['rect'] = (x, y, w, h)
    return count
    """

# ═══════════════════════════════════════════════════════
#  KILL FAST MISS
# ═══════════════════════════════════════════════════════

def kill_fast_miss(active_masks, now):
    fast_miss_thresh  = cfg.get("masks.fast_miss_threshold", 5)
    fast_miss_timeout = cfg.get("masks.fast_miss_timeout_ms", 300) / 1000.0

    alive = []
    for m in active_masks:
        time_since = now - m['last_detected_ts']

        if m['fast_miss_count'] >= fast_miss_thresh:
            continue                        # drop direct, pas de ttl -= 1

        if time_since >= fast_miss_timeout:
            continue                        # drop direct, pas de ttl -= 1

        alive.append(m)

    return alive

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
    return sum(max(0.0, now - m['last_detected_ts']) * 1000
               for m in active_masks) / len(active_masks)
    #return sum((now - m['last_detected_ts']) * 1000 for m in active_masks) / len(active_masks)

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
