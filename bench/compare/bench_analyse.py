# bench/compare/bench_analyse.py
"""
Analyse d'un rapport JSON produit par bench_compare.py.

Utilisation :
    python # bench/compare/bench_analyse.py <rapport.json>

Sortie : rapport texte structuré sur stdout.
Aucune dépendance hors stdlib.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Seuils d'analyse (propres à ce script — non issus de config.yaml)
# ---------------------------------------------------------------------------

WARN_P99_OVER_AVG_FACTOR  = 3.0   # p99 > avg × 3  → instabilité sonde
WARN_IQR_OVER_AVG_FACTOR  = 1.0   # iqr > avg × 1  → dispersion élevée
WARN_SPIKE_COUNT          = 3     # spike_count > 3 → régime instable
WARN_DRIFT_SLOPE_MS_S     = 0.5   # |drift_slope| > 0.5 ms/s → dérive notable
GOULOT_TOP_N              = 3     # groupes lourds reportés en passe A
TRIGGER_MIN_ABS_RHO       = 0.7   # |rho| min pour qualifier déclencheur (passe C)

EXPECTED_SCHEMA_VERSION   = 1

# Seuils comparaison (passe E)
DELTA_PROBE_WARN_PCT      = 10.0  # |avg_delta_pct| > 10 % → affiché
DELTA_BUDGET_WARN_PCT     = 5.0   # |pct_delta_pct| > 5 %  → affiché
DELTA_GAUGE_WARN_PCT      = 10.0  # |delta_pct| > 10 %     → affiché

# ---------------------------------------------------------------------------
# Helpers formatage
# ---------------------------------------------------------------------------

def fmt_float(v: float | None, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{digits}f}"

def fmt_pct(v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.1f}%"

def fmt_ratio(v: float | None) -> str:
    """Formate un ratio [0, 1] en pourcentage (ex : 1.0 → '100.0%')."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v * 100:.1f}%"

def fmt_flags(flags: list[str]) -> str:
    if not flags:
        return ""
    return " " + "".join(f"[{f}]" for f in flags)

def _sep(char: str = "─", width: int = 62) -> str:
    return char * width

# ---------------------------------------------------------------------------
# Chargement + validation schéma
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"ERREUR — fichier introuvable : {path}", file=sys.stderr)
        sys.exit(1)
    try:
        with p.open(encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"ERREUR — JSON invalide : {exc}", file=sys.stderr)
        sys.exit(1)

def check_schema(data: dict) -> None:
    version = data.get("schema_version")
    if version != EXPECTED_SCHEMA_VERSION:
        print(
            f"ERREUR — schema_version={version!r}, attendu {EXPECTED_SCHEMA_VERSION}.",
            file=sys.stderr,
        )
        sys.exit(1)

# ---------------------------------------------------------------------------
# Logique flags instabilité (réutilisée passes B et D.2)
# ---------------------------------------------------------------------------

def probe_flags(stats: dict) -> list[str]:
    """
    Calcule les flags d'instabilité d'une sonde à partir de son bloc stats.
    Retourne une liste (vide = sonde propre).
    Protection avg == 0 : ratios non calculés.
    """
    flags: list[str] = []
    avg = stats.get("avg")

    p99 = stats.get("p99_exact")
    if p99 is not None and avg and avg > 0:
        if p99 > avg * WARN_P99_OVER_AVG_FACTOR:
            flags.append("P99_HIGH")

    iqr = stats.get("iqr_exact")
    if iqr is not None and avg and avg > 0:
        if iqr > avg * WARN_IQR_OVER_AVG_FACTOR:
            flags.append("IQR_HIGH")

    spike = stats.get("spike_count")
    if spike is not None:
        if spike > WARN_SPIKE_COUNT:
            flags.append("SPIKES")

    drift = stats.get("drift_slope")
    if drift is not None:
        if abs(drift) > WARN_DRIFT_SLOPE_MS_S:
            flags.append("DRIFT")

    return flags

# ---------------------------------------------------------------------------
# Passe A — Budget frame
# ---------------------------------------------------------------------------

def run_pass_a(bucket: dict) -> dict:
    """
    Analyse le bloc frame_budget du bucket.
    Retourne :
        goulots  : set[str] — sondes (clé `probe`) des top-N groupes retenus
        warnings : list[str] — messages à afficher
        available: bool
    """
    fb = bucket.get("frame_budget")
    if not fb:
        print("  Non disponible (frame_budget absent ou null).")
        return {"goulots": set(), "warnings": [], "available": False}

    groups: dict = fb.get("groups") or {}
    total_ms: float | None = fb.get("total_ms")
    rows_total: int = fb.get("rows_total", 0)
    unaccounted_pct: float | None = fb.get("unaccounted_pct")
    unaccounted_warn: bool = fb.get("unaccounted_warn", False)
    reference: str = fb.get("reference", "?")

    print(f"  Référence : {reference}")
    print(f"  total_ms  : {fmt_float(total_ms)} ms   rows_total : {rows_total}")
    print()

    # Tri par pct décroissant (None en dernier)
    def _sort_key(item):
        pct = item[1].get("pct")
        return (0, -(pct if pct is not None else 0.0)) if pct is not None else (1, 0)

    sorted_groups = sorted(groups.items(), key=_sort_key)

    # En-tête tableau
    col = f"  {'Groupe':<18} {'Sonde':<26} {'Pct':>6}  {'sum_ms':>8}  {'Présence':>8}  Flags"
    print(col)
    print("  " + _sep("─", len(col) - 2))

    goulot_candidates: list[tuple[float, str]] = []  # (pct, probe_name)

    for group_name, gdata in sorted_groups:
        probe       : str         = gdata.get("probe", "?")
        pct         : float|None  = gdata.get("pct")
        sum_ms      : float|None  = gdata.get("sum_ms")
        presence    : float       = gdata.get("presence_rate", 0.0)
        conditional : bool        = gdata.get("conditional", False)
        low_presence: bool        = gdata.get("low_presence", False)

        row_flags: list[str] = []
        if low_presence:
            row_flags.append("LOW_PRESENCE")
        if conditional:
            row_flags.append("CONDITIONNEL")

        print(
            f"  {group_name:<18} {probe:<26} {fmt_pct(pct):>6}  "
            f"{fmt_float(sum_ms):>8}  {fmt_ratio(presence):>8} "
            f"{fmt_flags(row_flags)}"
        )

        # Candidat goulot : pct connu, non conditionnel, non low_presence
        if pct is not None and not conditional and not low_presence:
            goulot_candidates.append((pct, probe))

    print()
    unaccounted_line = f"  Non comptabilisé : {fmt_pct(unaccounted_pct)}"
    if unaccounted_warn:
        unaccounted_line += "  [WARN]"
    print(unaccounted_line)

    # Top-N goulots
    goulot_candidates.sort(key=lambda x: -x[0])
    goulots: set[str] = {probe for _, probe in goulot_candidates[:GOULOT_TOP_N]}

    if goulots:
        print(f"\n  → Goulots top-{GOULOT_TOP_N} : {', '.join(sorted(goulots))}")
    print()
    return {"goulots": goulots, "warnings": [], "available": True}

# ---------------------------------------------------------------------------
# Passe B — Instabilité des sondes
# ---------------------------------------------------------------------------

def run_pass_b(bucket: dict) -> dict:
    """
    Analyse toutes les sondes du bucket.
    Retourne :
        instables : list[dict]  — sondes avec au moins un flag
        total     : int         — nombre total de sondes analysées
    """
    probes: dict = bucket.get("probes") or {}
    instables: list[dict] = []

    for name, stats in probes.items():
        if not isinstance(stats, dict):
            continue
        flags = probe_flags(stats)
        if flags:
            instables.append({
                "name"   : name,
                "flags"  : flags,
                "avg"    : stats.get("avg"),
                "p99"    : stats.get("p99_exact"),
                "iqr"    : stats.get("iqr_exact"),
                "spikes" : stats.get("spike_count"),
                "drift"  : stats.get("drift_slope"),
            })

    # Tri : nombre de flags décroissant, puis nom alphabétique
    instables.sort(key=lambda d: (-len(d["flags"]), d["name"]))

    if not instables:
        print("  Toutes les sondes sont stables.")
    else:
        col = f"  {'Sonde':<30} {'avg':>7}  {'p99':>7}  {'iqr':>7}  {'spikes':>6}  {'drift':>7}  Flags"
        print(col)
        print("  " + _sep("─", len(col) - 2))
        for s in instables:
            print(
                f"  {s['name']:<30} "
                f"{fmt_float(s['avg']):>7}  "
                f"{fmt_float(s['p99']):>7}  "
                f"{fmt_float(s['iqr']):>7}  "
                f"{str(s['spikes']) if s['spikes'] is not None else '—':>6}  "
                f"{fmt_float(s['drift'], 3):>7}  "
                f"{fmt_flags(s['flags'])}"
            )

    print(f"\n  → {len(probes)} sondes analysées, {len(instables)} instable(s).")
    return {"instables": instables, "total": len(probes)}

# ---------------------------------------------------------------------------
# Passe C — Déclencheurs (corrélations → goulots)
# ---------------------------------------------------------------------------

def run_pass_c(bucket: dict, goulots: set[str]) -> dict:
    """
    Croise correlations.pairs avec les sondes goulots identifiées en passe A.
    Retourne :
        declencheurs : list[dict]
    """
    corr = bucket.get("correlations")
    if not corr:
        print("  Non disponible (bloc correlations absent).")
        return {"declencheurs": []}

    pairs: list[dict] = corr.get("pairs") or []
    summary: dict     = corr.get("summary") or {}
    truncated: bool   = summary.get("truncated_by_max_pairs", False)

    if truncated:
        print("  [AVERT] truncated_by_max_pairs=true — certaines paires peuvent être absentes.")

    if not goulots:
        print("  Aucun goulot identifié en passe A — analyse des déclencheurs non applicable.")
        return {"declencheurs": []}

    declencheurs: list[dict] = []
    for pair in pairs:
        a       : str   = pair.get("a", "")
        b       : str   = pair.get("b", "")
        rho     : float = pair.get("rho", 0.0)
        strength: str   = pair.get("strength", "?")
        n_samp  : int   = pair.get("n_samples", 0)

        if abs(rho) < TRIGGER_MIN_ABS_RHO:
            continue

        goulot_match: str | None = None
        other: str | None = None
        if a in goulots:
            goulot_match, other = a, b
        elif b in goulots:
            goulot_match, other = b, a

        if goulot_match is None:
            continue

        declencheurs.append({
            "goulot"  : goulot_match,
            "other"   : other,
            "rho"     : rho,
            "strength": strength,
            "n_samples": n_samp,
        })

    # Tri par |rho| décroissant
    declencheurs.sort(key=lambda d: -abs(d["rho"]))

    if not declencheurs:
        print(
            f"  Aucun déclencheur identifié pour les goulots "
            f"top-{GOULOT_TOP_N} (|rho| ≥ {TRIGGER_MIN_ABS_RHO})."
        )
    else:
        col = f"  {'Goulot':<26} {'← Sonde corrélée':<30} {'rho':>6}  {'strength':<12}  {'n':>5}"
        print(col)
        print("  " + _sep("─", len(col) - 2))
        for d in declencheurs:
            print(
                f"  {d['goulot']:<26} {d['other']:<30} "
                f"{fmt_float(d['rho'], 3):>6}  {d['strength']:<12}  {d['n_samples']:>5}"
            )

    return {"declencheurs": declencheurs}

# ---------------------------------------------------------------------------
# Passe D — Robustesse tracking
# ---------------------------------------------------------------------------

def run_pass_d(bucket: dict, temporal_events: dict) -> dict:
    """
    Analyse :
      D.1 — état tracker (gauges)
      D.2 — stabilité sondes motion_* / associator_* / main_match_ms
      D.3 est délégué à run_pass_f (affichage unique, source target.temporal_events).
      temporal_events reçu ici mais non affiché — retourné dans le résultat pour run_pass_f.
    """
    result: dict = {"tracker": {}, "temporal_events": temporal_events}

    # ── D.1 Tracker gauges ──────────────────────────────────────────────────
    gauges: dict = bucket.get("gauges") or {}
    confirmed = gauges.get("tracker_confirmed")
    pending   = gauges.get("tracker_pending")
    lost      = gauges.get("tracker_lost")

    print("  D.1 État tracker")
    if confirmed is None and pending is None and lost is None:
        print("      (gauges tracker absentes)")
    else:
        print(
            f"      confirmed={fmt_float(confirmed)}  "
            f"pending={fmt_float(pending)}  "
            f"lost={fmt_float(lost)}"
        )
        if None not in (confirmed, pending, lost):
            total_t = confirmed + pending + lost
            if total_t > 0:
                ratio_lost = lost / total_t
                print(f"      ratio_lost = {ratio_lost:.1%}")

    result["tracker"] = {
        "confirmed": confirmed,
        "pending"  : pending,
        "lost"     : lost,
    }

    # ── D.2 Sondes motion_* / associator_* / main_match_ms ─────────────────
    print("\n  D.2 Stabilité motion / associator")
    probes: dict = bucket.get("probes") or {}

    target_probes = {
        name: stats
        for name, stats in probes.items()
        if (
            name.startswith("motion_")
            or name.startswith("associator_")
            or name == "main_match_ms"
        )
        and isinstance(stats, dict)
    }

    if not target_probes:
        print("      (aucune sonde motion_* / associator_* / main_match_ms)")
    else:
        instables_d2: list[dict] = []
        for name, stats in target_probes.items():
            flags = probe_flags(stats)
            if flags:
                instables_d2.append({"name": name, "flags": flags})
        instables_d2.sort(key=lambda d: (-len(d["flags"]), d["name"]))

        if not instables_d2:
            print(f"      {len(target_probes)} sonde(s) analysée(s) — toutes stables.")
        else:
            for s in instables_d2:
                print(f"      {s['name']:<36}{fmt_flags(s['flags'])}")
            print(f"\n      → {len(target_probes)} sondes, {len(instables_d2)} instable(s).")

    return result

# ---------------------------------------------------------------------------
# Passe E — Comparaison (absente si session unique)
# ---------------------------------------------------------------------------

def run_pass_e(data: dict) -> dict:
    """
    Analyse le bloc comparisons (niveau racine du JSON).
    Retourne dict vide si absent.
    """
    comparisons: dict | None = data.get("comparisons")
    if not comparisons:
        return {}

    result: dict = {}

    for comp_type, comp_data in comparisons.items():
        if not isinstance(comp_data, dict):
            continue

        ref_session: str = comp_data.get("reference_session", "?")
        print(f"\n  ── {comp_type} (réf : {ref_session})")

        # Changements structurels
        appeared   : list = comp_data.get("appeared_probes")   or []
        disappeared: list = comp_data.get("disappeared_probes") or []
        if appeared:
            print(f"    Nouvelles sondes    : {', '.join(appeared)}")
        if disappeared:
            print(f"    Sondes disparues    : {', '.join(disappeared)}")

        deltas: dict = comp_data.get("deltas") or {}

        # Deltas probes
        probe_deltas: dict = deltas.get("probes") or {}
        significant_probes = [
            (name, d["avg_delta_pct"])
            for name, d in probe_deltas.items()
            if isinstance(d, dict)
            and d.get("avg_delta_pct") is not None
            and abs(d["avg_delta_pct"]) > DELTA_PROBE_WARN_PCT
        ]
        significant_probes.sort(key=lambda x: -abs(x[1]))

        if significant_probes:
            print(f"\n    Sondes avec delta |avg| > {DELTA_PROBE_WARN_PCT:.0f}% :")
            col = f"    {'Sonde':<32} {'delta_avg%':>10}"
            print(col)
            print("    " + _sep("─", len(col) - 4))
            for name, delta in significant_probes:
                sign = "+" if delta >= 0 else ""
                print(f"    {name:<32} {sign}{fmt_float(delta, 1):>10}%")
        else:
            print(f"    Aucun delta probe > {DELTA_PROBE_WARN_PCT:.0f}%.")
            print()

        # Deltas budget
        fb_deltas: dict = (deltas.get("frame_budget") or {}).get("groups") or {}
        significant_budget = [
            (name, d["pct_delta_pct"])
            for name, d in fb_deltas.items()
            if isinstance(d, dict)
            and d.get("pct_delta_pct") is not None
            and abs(d["pct_delta_pct"]) > DELTA_BUDGET_WARN_PCT
        ]
        significant_budget.sort(key=lambda x: -abs(x[1]))

        if significant_budget:
            print(f"\n    Budget — groupes avec delta pct > {DELTA_BUDGET_WARN_PCT:.0f}% :")
            col = f"    {'Groupe':<20} {'delta_pct%':>10}"
            print(col)
            print("    " + _sep("─", len(col) - 4))
            for name, delta in significant_budget:
                sign = "+" if delta >= 0 else ""
                print(f"    {name:<20} {sign}{fmt_float(delta, 1):>10}%")
        else:
            print(f"    Aucun delta budget > {DELTA_BUDGET_WARN_PCT:.0f}%.")

        # ── Deltas gauges ───────────────────────────────────────────────────
        gauge_deltas: dict = deltas.get("gauges") or {}
        significant_gauges = [
            (name, d["delta_pct"])
            for name, d in gauge_deltas.items()
            if isinstance(d, dict)
            and d.get("delta_pct") is not None
            and abs(d["delta_pct"]) > DELTA_GAUGE_WARN_PCT
        ]
        significant_gauges.sort(key=lambda x: -abs(x[1]))

        if significant_gauges:
            print(f"\n    Gauges avec delta > {DELTA_GAUGE_WARN_PCT:.0f}% :")
            col = f"    {'Gauge':<32} {'delta%':>10}"
            print(col)
            print("    " + _sep("─", len(col) - 4))
            for name, delta in significant_gauges:
                sign = "+" if delta >= 0 else ""
                print(f"    {name:<32} {sign}{fmt_float(delta, 1):>10}%")
        else:
            print(f"    Aucun delta gauge > {DELTA_GAUGE_WARN_PCT:.0f}%.")

        result[comp_type] = {
            "appeared"          : appeared,
            "disappeared"       : disappeared,
            "significant_probes": significant_probes,
            "significant_budget": significant_budget,
            "significant_gauges": significant_gauges,
        }

    return result

# ---------------------------------------------------------------------------
# Passe F — Résumé consolidé
# ---------------------------------------------------------------------------

def run_pass_f(results: list[dict],has_comparison: bool,temporal_events: dict) -> None:
    """
    Agrège les résultats cross-buckets et affiche le résumé final.
    temporal_events : dict lu depuis target.temporal_events dans main()
                      (source unique pour les gaps — pas par bucket).
    Ne pose aucun verdict pass/fail.
    """
    total_instables   : list[str] = []
    all_goulots       : list[str] = []
    total_declencheurs: int       = 0

    for r in results:
        for s in r.get("b", {}).get("instables", []):
            total_instables.append(s["name"])
        for g in r.get("a", {}).get("goulots", set()):
            if g not in all_goulots:
                all_goulots.append(g)
        total_declencheurs += len(r.get("c", {}).get("declencheurs", []))

    # Gaps depuis target.temporal_events (niveau racine, une seule fois)
    total_gaps_stat : int = 0
    total_gaps_fixed: int = 0
    for canal, tevt in temporal_events.items():
        if not isinstance(tevt, dict):
            continue
        gs = tevt.get("gaps_stat")
        gf = tevt.get("gaps_fixed")
        if gs is not None:
            total_gaps_stat  += gs
        if gf is not None:
            total_gaps_fixed += gf

    unique_instables = sorted(set(total_instables))

    print(f"  Buckets analysés       : {len(results)}")
    print(f"  Sondes instables       : {len(unique_instables)}")
    if unique_instables:
        for name in unique_instables:
            print(f"    • {name}")
    print(
        f"  Goulots identifiés     : {len(all_goulots)}"
        f"  ({', '.join(all_goulots) if all_goulots else '—'})"
    )
    print(f"  Déclencheurs corrélés  : {total_declencheurs} paire(s)")
    print(
        f"  Gaps temporels         : {total_gaps_stat} gaps_stat / "
        f"{total_gaps_fixed} gaps_fixed (tous canaux)"
    )

    # ── D.3 affiché une seule fois ici ─────────────────────────────────────
    print()
    print("  D.3 Régularité temporelle (target)")
    if not temporal_events:
        print("      Non disponible (target.temporal_events absent).")
    else:
        col = (
            f"      {'Canal':<10} {'median_itv_s':>13}  "
            f"{'gaps_stat':>9}  {'gaps_fixed':>10}"
        )
        print(col)
        print("      " + _sep("─", len(col) - 6))
        for canal, tevt in temporal_events.items():
            if not isinstance(tevt, dict):
                continue
            med   = tevt.get("median_interval_s")
            gstat = tevt.get("gaps_stat")
            gfix  = tevt.get("gaps_fixed")
            # gaps_fixed structurellement null pour canal frame (event-driven)
            gfix_str = "N/A" if gfix is None else str(gfix)
            print(
                f"      {canal:<10} {fmt_float(med, 4):>13}  "
                f"{str(gstat) if gstat is not None else '—':>9}  "
                f"{gfix_str:>10}"
            )

    print(f"  Comparaison disponible : {'oui' if has_comparison else 'non (session unique)'}")

# ---------------------------------------------------------------------------
# Entête et navigation buckets
# ---------------------------------------------------------------------------

def print_header(data: dict, mode: str) -> None:
    session_id: str = data.get("target_session")
    width = 62
    print("═" * width)
    print(f"  BENCH ANALYSE — {session_id}")
    print(f"  Mode : {mode}")
    print("═" * width)

def print_bucket_header(label: str) -> None:
    print(f"\n{'─' * 62}")
    print(f"  BUCKET : {label}")
    print(f"{'─' * 62}\n")

def print_pass_header(letter: str, title: str) -> None:
    print(f"\n  [{letter}] {title}")
    print("  " + "·" * 58)

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage : python bench/compare/bench_analyse.py <rapport.json>",
            file=sys.stderr,
        )
        sys.exit(1)

    data = load_json(sys.argv[1])
    check_schema(data)

    has_comparison: bool = bool(data.get("comparisons"))
    mode = "comparaison" if has_comparison else "session_unique"

    print_header(data, mode)

    target: dict = data.get("target") or {}

    # temporal_events lu UNE SEULE FOIS ici — transmis à run_pass_d et run_pass_f.
    # N'est PAS lu depuis les buckets individuels.
    temporal_events: dict = target.get("temporal_events") or {}

    buckets: dict = target.get("buckets") or {}

    if not buckets:
        print("\n  (aucun bucket disponible dans target.buckets)")
        print("\n" + "═" * 62)
        print("  Analyse terminée — aucune donnée à traiter.")
        print("═" * 62)
        return

    results: list[dict] = []

    for label, bucket_or_list in buckets.items():
        # hot → liste de dicts ; cold, tail → dict direct.
        # Les deux cas sont traités uniformément.
        if isinstance(bucket_or_list, list):
            items = bucket_or_list
        else:
            items = [bucket_or_list]

        for i, bucket in enumerate(items):
            if not isinstance(bucket, dict):
                print(f"\n  [AVERT] bucket {label}[{i}] n'est pas un dict — ignoré.")
                continue

            # P1 — ignorer les buckets sans probes (ex : sync_metadata)
            if not bucket.get("probes"):
                continue
            sub_label = f"{label}[{i}]" if isinstance(bucket_or_list, list) else label

            print_bucket_header(sub_label)

            print_pass_header("A", "BUDGET FRAME")
            a = run_pass_a(bucket)

            print_pass_header("B", "INSTABILITÉ DES SONDES")
            b = run_pass_b(bucket)

            print_pass_header("C", "DÉCLENCHEURS (corrélations → goulots)")
            c = run_pass_c(bucket, a["goulots"])

            print_pass_header("D", "ROBUSTESSE TRACKING")
            # temporal_events transmis depuis target — pas lu depuis bucket
            d = run_pass_d(bucket, temporal_events)

            results.append({
                "label": sub_label,
                "a": a,
                "b": b,
                "c": c,
                "d": d,
            })

    if has_comparison:
        print(f"\n{'─' * 62}")
        print("  [E] COMPARAISON")
        print("  " + "·" * 58)
        run_pass_e(data)

    print(f"\n{'═' * 62}")
    print("  [F] RÉSUMÉ CONSOLIDÉ")
    print("  " + "·" * 58)
    # temporal_events transmis pour calcul des gaps (source unique)
    run_pass_f(results, has_comparison, temporal_events)
    print("═" * 62)


if __name__ == "__main__":
    main()
