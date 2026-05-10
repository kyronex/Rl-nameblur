# tracker/motion.py — v3 (B-04b.1 + B-04b.2)
"""
Mouvement des masks : mise à jour sur détection + prédiction inertielle.
Fonctions pures (pas de global, pas de cfg singleton).

Sonde `staleness_slow` (ex `motion.dt`) :
    Mesure le délai entre la dernière détection slow d'un mask
    (`last_slow_ts`) et l'instant de prédiction (`now`).
    Alimentée UNIQUEMENT depuis `predict_position` (1×/mask non
    matché/tick) — pas depuis `compute_predicted_rect` qui est
    appelée N×M fois par l'associator pour le gating IoU.

    NB : `staleness_slow` ne reflète PAS la latence de prédiction
    réelle. Le fast tracker rafraîchit `last_seen_ts` mais pas
    `last_slow_ts`. Un mask correctement suivi par le fast peut
    avoir un `staleness_slow` élevé sans que ce soit un problème.
"""

import logging

log = logging.getLogger("motion")

# ── Stats motion (option B : compteurs ad-hoc, thread-safe via GIL) ──
_motion_stats = {"staleness_slow_sum_ms": 0.0, "staleness_slow_max_ms": 0.0, "n": 0, "capped": 0}

def get_and_reset_stats() -> dict:
    """
    Retourne les stats motion accumulées depuis le dernier appel,
    puis remet les compteurs à zéro.
    Appelé depuis main.py dans la branche FPS (thread principal).
    """
    global _motion_stats
    s = _motion_stats
    _motion_stats = {"staleness_slow_sum_ms": 0.0, "staleness_slow_max_ms": 0.0, "n": 0, "capped": 0}
    return s

def apply_detection(mask, new_rect, detect_ts, source, config):
    """
    Met à jour un mask avec une nouvelle détection.
    EMA smoothing + vélocité position ET taille (calculée uniquement sur source 'slow').
    """
    nx, ny, nw, nh = new_rect
    ox, oy, ow, oh = mask.rect
    size_ref = min(ow, oh)
    dead_zone = max(config.dead_zone_min_px, config.dead_zone_rel * size_ref)
    # ── Dead zone : pas de mouvement significatif ──
    if (abs(nx - ox) < dead_zone and abs(ny - oy) < dead_zone
            and abs(nw - ow) < dead_zone and abs(nh - oh) < dead_zone):
        mask.fast_miss_count = 0
        mask.last_detected_ts = detect_ts
        mask.last_source = source
        if source == "slow":
            mask.vx = 0.0
            mask.vy = 0.0
            mask.vw = 0.0
            mask.vh = 0.0
            mask.last_detected_rect = new_rect
            mask.last_slow_ts = detect_ts
        return
    # ── Vélocité : uniquement sur détections slow ──
    if source == "slow":
        lx, ly, lw, lh = mask.last_detected_rect
        if mask.last_slow_ts <= 0.0:
            dt = 0.0
        else:
            dt = detect_ts - mask.last_slow_ts
        if dt > 0.001:
            delta_x = nx - lx
            delta_y = ny - ly
            dist = (delta_x ** 2 + delta_y ** 2) ** 0.5
            if dt > config.dt_slow_max:
                mask.vx = 0.0
                mask.vy = 0.0
                mask.vw = 0.0
                mask.vh = 0.0
            elif dist > config.teleport_thresh:
                mask.vx = 0.0
                mask.vy = 0.0
                mask.vw = 0.0
                mask.vh = 0.0
            else:
                raw_vx = delta_x / dt
                raw_vy = delta_y / dt
                raw_vw = (nw - lw) / dt
                raw_vh = (nh - lh) / dt
                # Dead zone vélocité position
                velocity_dead_zone = config.velocity_dead_zone
                if abs(raw_vx) < velocity_dead_zone:
                    raw_vx = 0.0
                if abs(raw_vy) < velocity_dead_zone:
                    raw_vy = 0.0
                # Dead zone vélocité taille (même seuil)
                if abs(raw_vw) < velocity_dead_zone:
                    raw_vw = 0.0
                if abs(raw_vh) < velocity_dead_zone:
                    raw_vh = 0.0
                # Clamp position
                raw_vx = max(-config.vx_max, min(raw_vx, config.vx_max))
                raw_vy = max(-config.vy_max, min(raw_vy, config.vy_max))
                # Clamp taille
                raw_vw = max(-config.vw_max, min(raw_vw, config.vw_max))
                raw_vh = max(-config.vh_max, min(raw_vh, config.vh_max))
                mask.vx = raw_vx
                mask.vy = raw_vy
                mask.vw = raw_vw
                mask.vh = raw_vh
    # ── EMA smoothing ──
    alpha = config.smooth_alpha
    sx = ox + alpha * (nx - ox)
    sy = oy + alpha * (ny - oy)
    sw = ow + alpha * (nw - ow)
    sh = oh + alpha * (nh - oh)
    mask.rect = (sx, sy, sw, sh)
    mask.fast_miss_count = 0
    mask.last_detected_ts = detect_ts
    mask.last_source = source
    if source == "slow":
        mask.last_detected_rect = new_rect
        mask.last_slow_ts = detect_ts

def compute_predicted_rect(mask, ts, config):
    """
    Version PURE de la prédiction : retourne le rect prédit à `ts`
    sans muter le mask et SANS alimenter de sonde.
    Ancrage = last_detected_rect (pas de dérive cumulative).
    Appelée par l'associator (gating IoU) et par predict_position.
    """
    if mask.last_slow_ts <= 0.0:
        return mask.rect
    dt = ts - mask.last_slow_ts
    dt_capped = max(-config.dt_cap, min(dt, config.dt_cap))
    damping = max(0.0, 1.0 - abs(dt) * config.damping_rate)
    lx, ly, lw, lh = mask.last_detected_rect
    return (lx + mask.vx * dt_capped * damping,
            ly + mask.vy * dt_capped * damping,
            lw + mask.vw * dt_capped * damping,
            lh + mask.vh * dt_capped * damping)

def predict_position(mask, now, screen_w, screen_h, config):
    """
    Prédit la position d'un mask non mis à jour ce frame.
    Ancrage = last_detected_rect (pas de dérive cumulative).
    Alimente la sonde `staleness_slow` (1×/mask/tick).
    """
    # ── Sonde staleness_slow (alimentée UNIQUEMENT ici) ──
    if mask.last_slow_ts > 0.0:
        dt = now - mask.last_slow_ts
        dt_abs_ms = abs(dt) * 1000.0
        _motion_stats["staleness_slow_sum_ms"] += dt_abs_ms
        _motion_stats["n"] += 1
        if dt_abs_ms > _motion_stats["staleness_slow_max_ms"]:
            _motion_stats["staleness_slow_max_ms"] = dt_abs_ms
        if abs(dt) > config.dt_cap:
            _motion_stats["capped"] += 1

    x, y, w, h = compute_predicted_rect(mask, now, config)
    # Taille minimum
    min_size = config.min_mask_size
    w = max(min_size, w)
    h = max(min_size, h)
    # Clamp aux bornes écran
    x = max(0.0, min(x, screen_w - w))
    y = max(0.0, min(y, screen_h - h))
    mask.rect = (x, y, w, h)
