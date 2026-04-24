# tracker/motion.py — v2
"""
Mouvement des masks : mise à jour sur détection + prédiction inertielle.
Fonctions pures (pas de global, pas de cfg singleton).
"""

import logging

log = logging.getLogger("motion")


def apply_detection(mask, new_rect, detect_ts, source, config):
    """
    Met à jour un mask avec une nouvelle détection.
    EMA smoothing + vélocité position ET taille (calculée uniquement sur source 'slow').
    """
    nx, ny, nw, nh = new_rect
    ox, oy, ow, oh = mask.rect

    dead_zone = config.dead_zone

    # ── Dead zone : pas de mouvement significatif ──
    if (abs(nx - ox) < dead_zone and abs(ny - oy) < dead_zone
            and abs(nw - ow) < dead_zone and abs(nh - oh) < dead_zone):
        mask.fast_miss_count = 0
        mask.last_detected_ts = detect_ts
        mask.last_source = source
        if source == "slow":
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
                log.debug(f"uid={mask.uid} reset velocity "f"dt={dt * 1000:.0f}ms > max={config.dt_slow_max * 1000:.0f}ms")
                mask.vx = 0.0
                mask.vy = 0.0
                mask.vw = 0.0
                mask.vh = 0.0
            elif dist > config.teleport_thresh:
                log.debug(f"uid={mask.uid} teleport reset "f"dist={dist:.0f}px > thresh={config.teleport_thresh}px")
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


def predict_position(mask, now, screen_w, screen_h, config):
    """
    Prédit la position d'un mask non mis à jour ce frame.
    Ancrage = last_detected_rect (pas de dérive cumulative).
    Prédit aussi w/h via vw/vh.
    """
    dt = now - mask.last_detected_ts
    dt_capped = min(dt, config.dt_cap)
    damping = max(0.0, 1.0 - dt * config.damping_rate)

    lx, ly, lw, lh = mask.last_detected_rect

    x = lx + mask.vx * dt_capped * damping
    y = ly + mask.vy * dt_capped * damping
    w = lw + mask.vw * dt_capped * damping
    h = lh + mask.vh * dt_capped * damping

    # Taille minimum
    min_size = config.min_mask_size
    w = max(min_size, w)
    h = max(min_size, h)

    # Clamp aux bornes écran
    x = max(0.0, min(x, screen_w - w))
    y = max(0.0, min(y, screen_h - h))

    mask.rect = (x, y, w, h)
