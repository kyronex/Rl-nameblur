# tracker/motion.py — v1
"""
Mouvement des masks : mise à jour sur détection + prédiction inertielle.
Fonctions pures (pas de global, pas de cfg singleton).
"""

import logging

log = logging.getLogger("motion")

def apply_detection(mask, new_rect, detect_ts, source, config):
    """
    Met à jour un mask avec une nouvelle détection.
    EMA smoothing + vélocité (calculée uniquement sur source 'slow').

    Args:
        mask:       Mask dataclass
        new_rect:   (x, y, w, h) détecté
        detect_ts:  float timestamp de la détection
        source:     'slow' | 'fast' | 'new'
        config:     TrackerConfig
    """
    nx, ny, nw, nh = new_rect
    ox, oy, ow, oh = mask.rect

    dead_zone = config.motion_dead_zone

    # ── Dead zone : pas de mouvement significatif ──
    if (abs(nx - ox) < dead_zone and abs(ny - oy) < dead_zone
            and abs(nw - ow) < dead_zone and abs(nh - oh) < dead_zone):
        mask.fast_miss_count = 0
        mask.last_detected_ts = detect_ts
        mask.last_source = source
        if source == "slow":
            mask.last_detected_rect = mask.rect
        return

    # ── Vélocité : uniquement sur détections slow ──
    if source == "slow":
        lx, ly = mask.last_detected_rect[0], mask.last_detected_rect[1]
        dt = detect_ts - mask.last_detected_ts

        if dt > 0.001:
            delta_x = nx - lx
            delta_y = ny - ly
            dist = (delta_x ** 2 + delta_y ** 2) ** 0.5

            if dt > config.motion_dt_slow_max:
                # Cut caméra ou longue absence → reset vélocité
                log.debug(
                    f"uid={mask.uid} reset vx/vy "
                    f"dt={dt * 1000:.0f}ms > max={config.motion_dt_slow_max * 1000:.0f}ms"
                )
                mask.vx = 0.0
                mask.vy = 0.0
            elif dist > config.motion_teleport_thresh:
                # Téléportation → reset vélocité
                log.debug(
                    f"uid={mask.uid} teleport reset "
                    f"dist={dist:.0f}px > thresh={config.motion_teleport_thresh}px"
                )
                mask.vx = 0.0
                mask.vy = 0.0
            else:
                raw_vx = delta_x / dt
                raw_vy = delta_y / dt

                # Dead zone vélocité
                if abs(raw_vx) < config.motion_velocity_dead_zone:
                    raw_vx = 0.0
                if abs(raw_vy) < config.motion_velocity_dead_zone:
                    raw_vy = 0.0

                # Clamp
                raw_vx = max(-config.motion_vx_max, min(raw_vx, config.motion_vx_max))
                raw_vy = max(-config.motion_vy_max, min(raw_vy, config.motion_vy_max))

                mask.vx = raw_vx
                mask.vy = raw_vy

    # ── EMA smoothing ──
    alpha = config.motion_smooth_alpha
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


def predict_position(mask, now, screen_w, screen_h, config):
    """
    Prédit la position d'un mask non mis à jour ce frame.
    Ancrage = last_detected_rect (pas de dérive cumulative).

    Args:
        mask:       Mask dataclass
        now:        float timestamp courant
        screen_w:   int largeur écran
        screen_h:   int hauteur écran
        config:     TrackerConfig
    """
    dt = now - mask.last_detected_ts
    dt_capped = min(dt, config.predict_dt_cap)
    damping = max(0.0, 1.0 - dt * config.predict_damping_rate)

    lx, ly = mask.last_detected_rect[0], mask.last_detected_rect[1]
    w, h = mask.last_detected_rect[2], mask.last_detected_rect[3]

    x = lx + mask.vx * dt_capped * damping
    y = ly + mask.vy * dt_capped * damping

    # Clamp aux bornes écran
    x = max(0.0, min(x, screen_w - w))
    y = max(0.0, min(y, screen_h - h))

    mask.rect = (x, y, w, h)
