# core/mask_manager.py
import logging
import cv2
from config import cfg
from core.mask import MaskState

log = logging.getLogger("core.mask_manager")

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
#  COULEURS / ÉTAT
# ═══════════════════════════════════════════════════════

# Palette 12 teintes HSV espacées, S=255 V=255 → lisible sur gazon vert,
# ciel bleu et carrosserie rouge. Générée une fois au chargement.
def _build_uid_palette(n=12):
    import numpy as np
    hsv = np.zeros((1, n, 3), dtype=np.uint8)
    hsv[0, :, 0] = (np.arange(n) * (180 // n)).astype(np.uint8)
    hsv[0, :, 1] = 255
    hsv[0, :, 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]
    return [tuple(int(c) for c in px) for px in bgr]

_UID_PALETTE = _build_uid_palette()

# Épaisseur du trait = encodage de l'état (la couleur porte l'uid)
_STATE_THICKNESS = {
    MaskState.PENDING:   3,
    MaskState.CONFIRMED: 4,
    MaskState.LOST:      2,
}

# Lettre d'état, 1 caractère, intégrée au label
_STATE_LETTER = {
    MaskState.PENDING:   "P",
    MaskState.CONFIRMED: "C",
    MaskState.LOST:      "L",
}

def _uid_color(uid):
    return _UID_PALETTE[uid % len(_UID_PALETTE)]

def _state_color(state):
    """Legacy — conservé pour verbosity=0 / color_by_uid=false."""
    if state == MaskState.CONFIRMED:
        return tuple(cfg.get("debug.overlay.colors.vert"))
    elif state == MaskState.PENDING:
        return tuple(cfg.get("debug.overlay.colors.rouge"))
    else:
        return tuple(cfg.get("debug.overlay.colors.noir"))

# ═══════════════════════════════════════════════════════
#  LABELS
# ═══════════════════════════════════════════════════════

_SRC_SHORT = {"slow": "S", "fast": "F", "new": "N", "unknown": "?"}

def _clip_tag(clip):
    """Bitmask sig_map_clip → tag lisible. Gauche/droite seuls décisifs
    (fragment potentiel) ; haut/bas signalés en minuscule."""
    if clip is None:
        return "?"
    tag = ""
    if clip & 1: tag += "L"
    if clip & 2: tag += "R"
    if clip & 4: tag += "t"
    if clip & 8: tag += "b"
    return tag or "-"

def _sig_line(sig):
    """Ligne signature — niveau 2. Distingue NONE (pas de propagation)
    de EMPTY (calcul tenté, dégradé)."""
    if sig is None:
        return "sig=NONE"
    if sig.get("sig_map") is None and sig.get("sig_shape") is None:
        return "sig=EMPTY"
    # sig_hs_mode est un tuple ("otsu", "eroded"|"copy") — seul [1] informe
    hs_mode = sig.get("sig_hs_mode")
    hs = hs_mode[1] if isinstance(hs_mode, (tuple, list)) and len(hs_mode) > 1 else "-"
    mp = sig.get("sig_map_mode") or "-"
    return f"hs={hs} map={mp} clip={_clip_tag(sig.get('sig_map_clip'))}"

def _counters_line(m):
    """Ligne compteurs — niveau 2."""
    return (f"cum={m.total_matches_cumul} fm={m.frames_matched} "
            f"miss={m.fast_miss_count} cf={m.confidence:.2f}")

# ═══════════════════════════════════════════════════════
#  DEBUG DRAW
# ═══════════════════════════════════════════════════════

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FSCALE = 1.25
_FTHICK = 2
_LINE_H = 30

def _put(img, text, org, color):
    """Texte avec liseré noir — indispensable pour rester lisible
    sur ciel clair comme sur gazon sombre."""
    cv2.putText(img, text, org, _FONT, _FSCALE, (0, 0, 0), _FTHICK + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, _FONT, _FSCALE, color, _FTHICK, cv2.LINE_AA)

def draw_debug(frame, masks):
    """
    Overlay debug. Verbosité pilotée par debug.overlay.verbosity :
        0 → legacy (couleur par état, label 'State=...')
        1 → couleur par uid, label '#uid C/S'
        2 → + lignes signature et compteurs (sous la boîte)
    """
    verbosity   = cfg.get("debug.overlay.verbosity", 1)
    color_by_uid = cfg.get("debug.overlay.color_by_uid", True)
    h_img, w_img = frame.shape[:2]

    for m in masks:
        x, y, w, h = (int(v) for v in m.rect)

        if verbosity == 0:
            color = _state_color(m.state)
            thick = 1
            label = f"State={m.state.name} [{m.last_source}]"
        else:
            color = _uid_color(m.uid) if color_by_uid else _state_color(m.state)
            thick = _STATE_THICKNESS.get(m.state, 1)
            src   = _SRC_SHORT.get(m.last_source, m.last_source[:1].upper())
            label = f"#{m.uid} {_STATE_LETTER.get(m.state, '?')}/{src}"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thick)

        # Label au-dessus si la place existe, sinon dans la boîte
        ty = y - 4 if y >= _LINE_H else y + _LINE_H
        _put(frame, label, (x, ty), color)

        # Niveau 2 : sous la boîte (les plaques sont trop plates pour empiler
        # au-dessus sans chevaucher la plaque voisine)
        if verbosity >= 2:
            by = y + h + _LINE_H
            for line in (_sig_line(m.signature), _counters_line(m)):
                if by < h_img:
                    _put(frame, line, (x, by), color)
                    by += _LINE_H