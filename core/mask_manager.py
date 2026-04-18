# core/mask_manager.py — v2 (migration Mask dataclass)
import logging
import cv2
from config import cfg

log = logging.getLogger("mask_manager")


# ═══════════════════════════════════════════════════════
#  GÉOMÉTRIE
# ═══════════════════════════════════════════════════════

def _center_distance(r1, r2):
    """Distance euclidienne entre centres (x,y,w,h)."""
    cx1 = r1[0] + r1[2]/2;  cy1 = r1[1] + r1[3]/2
    cx2 = r2[0] + r2[2]/2;  cy2 = r2[1] + r2[3]/2
    return ((cx1-cx2)**2 + (cy1-cy2)**2)**0.5

def _rect_corners(r):
    x, y, w, h = r
    return [(x,y), (x+w,y), (x,y+h), (x+w,y+h)]

def _corners_distance(r1, r2):
    """Distance moyenne entre les 4 coins."""
    c1 = _rect_corners(r1);  c2 = _rect_corners(r2)
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

def _ttl_color(ttl):
    if ttl >= 3:
        return tuple(cfg.get("debug.colors.fresh"))
    elif ttl >= 2:
        return tuple(cfg.get("debug.colors.persist"))
    else:
        return tuple(cfg.get("debug.colors.dying"))

def _ttl_label(ttl, source="S"):
    return f"TTL={ttl} [{source}]"


# ═══════════════════════════════════════════════════════
#  JITTER + AGE
# ═══════════════════════════════════════════════════════

def compute_jitter(active_masks, rects_before):
    """
    Calcule le jitter centre/coins et compte créations/kills.
    Retourne (center_avg, corners_avg, masks_created, masks_killed).
    """
    after_uids       = {m.uid for m in active_masks}
    jitter_c_sum     = 0.0
    jitter_4_sum     = 0.0
    jitter_n         = 0
    masks_created    = 0
    masks_killed     = 0

    for m in active_masks:
        uid = m.uid
        if uid in rects_before:
            jitter_c_sum += _center_distance(rects_before[uid], m.rect)
            jitter_4_sum += _corners_distance(rects_before[uid], m.rect)
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
    return sum(max(0.0, now - m.last_detected_ts) * 1000
               for m in active_masks) / len(active_masks)
    #return sum((now - m['last_detected_ts']) * 1000 for m in active_masks) / len(active_masks)

# ═══════════════════════════════════════════════════════
#  DEBUG DRAW
# ═══════════════════════════════════════════════════════

def draw_debug(frame, active_masks):
    """Dessine les rectangles et labels TTL sur le frame (in-place)."""
    for m in active_masks:
        x, y, w, h = (int(v) for v in m.rect)
        color = _ttl_color(m.ttl)
        source = m.last_source[0].upper() if m.last_source else "?"
        label  = _ttl_label(m.ttl, source)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame, label,
            (x, max(y-5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            cv2.LINE_AA,
        )
