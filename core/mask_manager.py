# core/mask_manager.py — v2 (migration Mask dataclass)
import logging
import cv2
from config import cfg

log = logging.getLogger("mask_manager")

# ═══════════════════════════════════════════════════════
#  GÉOMÉTRIE
# ═══════════════════════════════════════════════════════


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

# ═══════════════════════════════════════════════════════
#  TTL / COULEURS DEBUG
# ═══════════════════════════════════════════════════════

def _ttl_color(ttl):
    if ttl >= 3:
        return tuple(cfg.get("debug.overlay.colors.vert"))
    elif ttl >= 2:
        return tuple(cfg.get("debug.overlay.colors.rouge"))
    else:
        return tuple(cfg.get("debug.overlay.colors.noir"))

def _ttl_label(ttl, source="S"):
    return f"TTL={ttl} [{source}]"