# bench/compare/_frame_budget.py
"""
Construction du bloc « frame budget »  — décomposition temporelle de la boucle frame par groupes de sondes.
Rôle : builder de section. Lit les lignes du canal `frame`, agrège par groupe, applique _r() (arrondi présentation), pose les flags (low_presence, unaccounted_warn, conditional). Produit un dict prêt à sérialiser dans le rapport.
"""

from __future__ import annotations

from bench.compare._config import (
    FRAME_BUDGET_REFERENCE,
    FRAME_BUDGET_GROUPS,
    FRAME_BUDGET_CONDITIONAL,
    FRAME_BUDGET_ENABLED,
    FRAME_BUDGET_MIN_PRESENCE_RATE,
    FRAME_BUDGET_UNACCOUNTED_WARN_PCT,
    _r,
)

def build_frame_budget(frame_rows: list[dict]) -> dict | None:
    ref_probe = FRAME_BUDGET_REFERENCE
    # ── Étape 1 : extraction lignes valides + total référence ──
    total_ms = 0.0
    rows_total = 0
    # On stocke pour chaque ligne : (count_ref, dict_probes) pour réutilisation étape 3
    ref_lines: list[tuple[int, dict]] = []
    for row in frame_rows:
        probes = row.get("probes")
        if not isinstance(probes, dict):
            continue
        ref_stats = probes.get(ref_probe)
        if not isinstance(ref_stats, dict):
            continue
        avg = ref_stats.get("avg")
        cnt = ref_stats.get("count")
        if avg is None or cnt is None or cnt <= 0:
            continue
        total_ms += float(avg) * int(cnt)
        rows_total += 1
        ref_lines.append((int(cnt), probes))
    if rows_total == 0 or total_ms == 0:
        return None
    # ── Étape 2 + 3 : accumulation par groupe ──
    groups_out: dict[str, dict] = {}
    sum_pct_present = 0.0
    for group_name, probe_name in FRAME_BUDGET_GROUPS.items():
        sum_group = 0.0
        rows_with = 0
        for cnt_ref, probes in ref_lines:
            stats = probes.get(probe_name)
            if not isinstance(stats, dict):
                continue
            avg_g = stats.get("avg")
            if avg_g is None:
                continue
            sum_group += float(avg_g) * cnt_ref
            rows_with += 1
        presence_rate = rows_with / rows_total
        is_conditional = group_name in FRAME_BUDGET_CONDITIONAL
        low_presence = presence_rate < FRAME_BUDGET_MIN_PRESENCE_RATE
        if rows_with == 0:
            pct = None
            sum_ms_out = None
        else:
            pct = (sum_group / total_ms) * 100.0
            sum_ms_out = sum_group
            sum_pct_present += pct
        groups_out[group_name] = {
            "probe":         probe_name,
            "sum_ms":        _r(sum_ms_out),
            "pct":           _r(pct),
            "presence_rate": _r(presence_rate),
            "rows_with":     rows_with,
            "conditional":   is_conditional,
            "low_presence":  low_presence,
        }
    # ── Étape 6 : unaccounted ──
    unaccounted_pct = 100.0 - sum_pct_present
    unaccounted_warn = unaccounted_pct > FRAME_BUDGET_UNACCOUNTED_WARN_PCT
    return {
        "reference":       ref_probe,
        "total_ms":        _r(total_ms),
        "rows_total":      rows_total,
        "groups":          groups_out,
        "unaccounted_pct": _r(unaccounted_pct),
        "unaccounted_warn": unaccounted_warn,
    }