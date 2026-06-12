# bench/compare/_builder.py

from bench.compare._config import (PERCENTILES,FRAME_BUDGET_ENABLED,_r)
from bench.compare._stats import (agg_probes,agg_rates,agg_gauges,session_duration,extract_timeline,compute_frames,compute_duration_mono,compute_temporal_events,collect_frame_samples,collect_fast_approx_samples,collect_frame_exact_pairs,filter_rows_by_mono)
from bench.compare._bucketing import BucketsResult, compute_buckets
from bench.compare._correlations import compute_correlations
from bench.compare._anomalies import (empty_anomalies,compute_anomalies)
from bench.compare._frame_budget import build_frame_budget
from bench.compare._math import (delta_pct,delta_abs,percentile_value, quartile_values,safe_skewness,safe_kurtosis_excess)

def _build_percentile_block(probe_name: str,exact_samples: dict[str, list[float]],approx_samples: dict[str, list[float]],*,channel: str) -> dict:
    """
    Construit le bloc stats descriptives (percentiles + quartiles S4bis) pour une sonde.
    Args:
        probe_name: nom de la sonde.
        exact_samples: échantillons exacts (count==1) issus du canal frame.
        approx_samples: échantillons approximés issus du canal agrégé concerné.
        channel: "agg"  → exact calculable depuis frame_rows.
                 "fast" → pas d'exact (samples_exact=0, *_exact=null).
    Returns:
        dict avec samples_exact, samples_approx,
        p{pct}_exact / p{pct}_approx pour pct ∈ PERCENTILES,
        et q1/q3/iqr déclinés _exact / _approx (S4bis).
    """
    has_exact = (channel == "agg")
    exact_data = exact_samples.get(probe_name, []) if has_exact else []
    approx_data = approx_samples.get(probe_name, [])
    block: dict = {
        "samples_exact": len(exact_data) if has_exact else 0,
        "samples_approx": len(approx_data),
    }
    for pct in PERCENTILES:
        block[f"p{pct}_exact"] = percentile_value(exact_data, pct) if has_exact else None
        block[f"p{pct}_approx"] = percentile_value(approx_data, pct)
    # S4bis — quartiles + IQR
    q1_e, q3_e, iqr_e = quartile_values(exact_data) if has_exact else (None, None, None)
    q1_a, q3_a, iqr_a = quartile_values(approx_data)
    block["q1_exact"]   = q1_e
    block["q1_approx"]  = q1_a
    block["q3_exact"]   = q3_e
    block["q3_approx"]  = q3_a
    block["iqr_exact"]  = iqr_e
    block["iqr_approx"] = iqr_a
    block["skewness_exact"]         = safe_skewness(exact_data) if has_exact else None
    block["skewness_approx"]        = safe_skewness(approx_data)
    block["kurtosis_excess_exact"]  = safe_kurtosis_excess(exact_data) if has_exact else None
    block["kurtosis_excess_approx"] = safe_kurtosis_excess(approx_data)
    return block

def _build_single_bucket(agg_rows: list[dict], frame_rows: list[dict], fast_rows: list[dict], *, bucket_label: str = "unknown") -> dict:
    """
    Agrège un bucket unique. Structure identique à build_session_block
    mais sans duration_s, duration_mono_s, frames, temporal_events.
    """
    base_probes_agg  = agg_probes(agg_rows)
    base_probes_fast = agg_probes(fast_rows)
    rates_agg   = agg_rates(agg_rows)
    rates_fast  = agg_rates(fast_rows)
    gauges_agg  = agg_gauges(agg_rows)
    gauges_fast = agg_gauges(fast_rows)
    exact_samples, frame_approx = collect_frame_samples(frame_rows)
    fast_approx = collect_fast_approx_samples(fast_rows)
    approx_samples: dict[str, list[float]] = {**frame_approx, **fast_approx}
    # ── S5b : extraction des paires (value, mono) triées par mono ──
    exact_pairs = collect_frame_exact_pairs(frame_rows)
    anomalies_by_probe: dict[str, dict] = {}
    for probe_name, pairs in exact_pairs.items():
        pairs_sorted = sorted(pairs, key=lambda p: p[1])
        anomalies_by_probe[probe_name] = compute_anomalies(pairs_sorted)
    probes: dict[str, dict] = {}
    for probe_name, stats in base_probes_agg.items():
        pct_block = _build_percentile_block(probe_name, exact_samples, approx_samples, channel="agg")
        anomalies = anomalies_by_probe.get(probe_name, empty_anomalies())
        probes[probe_name] = {
            "avg": _r(stats["avg"]),
            "min": _r(stats["min"]),
            "max": _r(stats["max"]),
            "count_agg": stats["count_agg"],
            **{k: _r(v) for k, v in pct_block.items()},
            **{k: _r(v) for k, v in anomalies.items()},
        }
    fast_probes: dict[str, dict] = {}
    for probe_name, stats in base_probes_fast.items():
        pct_block = _build_percentile_block(probe_name, exact_samples, approx_samples, channel="fast")
        fast_probes[probe_name] = {
            "avg": _r(stats["avg"]),
            "min": _r(stats["min"]),
            "max": _r(stats["max"]),
            "count_fast": stats["count_agg"],
            **{k: _r(v) for k, v in pct_block.items()},
            **empty_anomalies(),
        }
    fb = build_frame_budget(frame_rows) if FRAME_BUDGET_ENABLED else None
    correlations = compute_correlations(frame_rows, bucket_label=bucket_label)
    return {
        "probes":        probes,
        "rates":         {k: _r(v) for k, v in rates_agg.items()},
        "gauges":        {k: _r(v) for k, v in gauges_agg.items()},
        "fast_probes":   fast_probes,
        "fast_rates":    {k: _r(v) for k, v in rates_fast.items()},
        "fast_gauges":   {k: _r(v) for k, v in gauges_fast.items()},
        "frame_budget":  fb,
        "correlations":  correlations,
    }

def _build_buckets_block(agg_rows:list[dict],frame_rows:list[dict],fast_rows:list[dict],result:BucketsResult) -> dict:
    """
    Itère sur cold / hot_i / tail, filtre les lignes par mono,
    appelle _build_single_bucket pour chaque bucket.
    """
    def _rows(bucket, agg=agg_rows, frame=frame_rows, fast=fast_rows):
        a = filter_rows_by_mono(agg,   bucket.mono_start, bucket.mono_end)
        fr = filter_rows_by_mono(frame, bucket.mono_start, bucket.mono_end)
        fa = filter_rows_by_mono(fast,  bucket.mono_start, bucket.mono_end)
        return a, fr, fa
    # Cold
    ca, cfr, cfa = _rows(result.cold)
    cold_block = {
        "mono_start":        result.cold.mono_start,
        "mono_end":          result.cold.mono_end,
        "duration_s":        _r(result.cold.duration_s),
        "cold_end_target_s": _r(result.cold.cold_end_target_s),
        "cold_end_real_s":   _r(result.cold.cold_end_real_s),
        "cold_drift_s":      _r(result.cold.cold_drift_s),
        "cold_drift_warning": result.cold.cold_drift_warning,
        "cold_truncated":    result.cold.cold_truncated,
        "frames": {
            "agg":   len(ca),
            "frame": len(cfr),
            "fast":  len(cfa),
        },
        **_build_single_bucket(ca, cfr, cfa, bucket_label="cold"),
    }
    # Hot
    hot_list = []
    for h in result.hot:
        ha, hfr, hfa = _rows(h)
        hot_list.append({
            "index":            h.index,
            "mono_start":       h.mono_start,
            "mono_end":         h.mono_end,
            "duration_s":       _r(h.duration_s),
            "is_pivot_snapped": h.is_pivot_snapped,
            "frames": {
                "agg":   len(ha),
                "frame": len(hfr),
                "fast":  len(hfa),
            },
            **_build_single_bucket(ha, hfr, hfa, bucket_label=f"hot_{h.index}"),
        })
    # Tail
    tail_block = None
    if result.tail is not None:
        ta, tfr, tfa = _rows(result.tail)
        tail_block = {
            "mono_start":  result.tail.mono_start,
            "mono_end":    result.tail.mono_end,
            "duration_s":  _r(result.tail.duration_s),
            "is_partial":  True,
            "frames": {
                "agg":   len(ta),
                "frame": len(tfr),
                "fast":  len(tfa),
            },
            **_build_single_bucket(ta, tfr, tfa, bucket_label="tail"),
        }
    return {
        "sync_metadata": {
            "cold_end_target_s":  _r(result.cold.cold_end_target_s),
            "cold_end_real_s":    _r(result.cold.cold_end_real_s),
            "cold_drift_s":       _r(result.cold.cold_drift_s),
            "cold_drift_warning": result.cold.cold_drift_warning,
            "cold_truncated":     result.cold.cold_truncated,
            "fast_enabled":       result.fast_enabled,
        },
        "cold": cold_block,
        "hot":  hot_list,
        "tail": tail_block,
    }

def build_session_block(agg_rows: list[dict],frame_rows: list[dict],fast_rows: list[dict]) -> dict:
    """
    Construit le bloc session complet :
      {
        duration_s, duration_mono_s, frames, temporal_events,
        probes, rates, gauges,
        fast_probes, fast_rates, fast_gauges
      }
    Les blocs `probes`, `rates`, `gauges` agrègent UNIQUEMENT le canal agg.
    Les blocs `fast_probes`, `fast_rates`, `fast_gauges` agrègent UNIQUEMENT
    le canal fast. Les blocs fast_* sont toujours présents (éventuellement {}).
    Les sondes des deux canaux reçoivent un bloc percentiles complet
    (samples_exact, p*_exact, samples_approx, p*_approx) via
    `_build_percentile_block(channel=...)`.
    """
    # ── Agrégations brutes par canal ──
    base_probes_agg  = agg_probes(agg_rows)
    base_probes_fast = agg_probes(fast_rows)
    rates_agg   = agg_rates(agg_rows)
    rates_fast  = agg_rates(fast_rows)
    gauges_agg  = agg_gauges(agg_rows)
    gauges_fast = agg_gauges(fast_rows)
    duration = session_duration(agg_rows)
    # ── Bucketing S4 ──
    timeline_agg   = extract_timeline(agg_rows)
    timeline_fast  = extract_timeline(fast_rows)
    timeline_frame = extract_timeline(frame_rows)
    buckets_result = compute_buckets(timeline_agg, timeline_fast, timeline_frame)
    buckets_block  = (
        _build_buckets_block(agg_rows, frame_rows, fast_rows, buckets_result)
        if buckets_result is not None else None
    )
    # ── Samples pour percentiles ──
    exact_samples, frame_approx = collect_frame_samples(frame_rows)
    fast_approx = collect_fast_approx_samples(fast_rows)
    approx_samples: dict[str, list[float]] = {**frame_approx, **fast_approx}
    # ── Construction probes (canal agg) ──
    probes: dict[str, dict] = {}
    for probe_name, stats in base_probes_agg.items():
        pct_block = _build_percentile_block(probe_name, exact_samples, approx_samples, channel="agg")
        probes[probe_name] = {
            "avg": _r(stats["avg"]),
            "min": _r(stats["min"]),
            "max": _r(stats["max"]),
            "count_agg": stats["count_agg"],
            **{k: _r(v) for k, v in pct_block.items()},
        }
    # ── Construction fast_probes (canal fast) ──
    fast_probes: dict[str, dict] = {}
    for probe_name, stats in base_probes_fast.items():
        pct_block = _build_percentile_block(probe_name, exact_samples, approx_samples, channel="fast")
        fast_probes[probe_name] = {
            "avg": _r(stats["avg"]),
            "min": _r(stats["min"]),
            "max": _r(stats["max"]),
            "count_fast": stats["count_agg"],
            **{k: _r(v) for k, v in pct_block.items()},
        }
    # ── Analyse temporelle (3 canaux) ── 5
    duration_mono = compute_duration_mono(timeline_agg)
    frames = {
        "agg":   compute_frames(timeline_agg),
        "frame": compute_frames(timeline_frame),
        "fast":  compute_frames(timeline_fast),
    }
    temporal_events = {
        "agg":   compute_temporal_events(timeline_agg,   "agg"),
        "frame": compute_temporal_events(timeline_frame, "frame"),
        "fast":  compute_temporal_events(timeline_fast,  "fast"),
    }
    temporal_events_rounded: dict[str, dict] = {}
    for canal, events in temporal_events.items():
        temporal_events_rounded[canal] = {
            "median_interval_s": _r(events["median_interval_s"]),
            "gaps_stat":  events["gaps_stat"],
            "gaps_fixed": events["gaps_fixed"],
        }
    # ── Frame budget S6a (session-level) ──
    frame_budget_block = build_frame_budget(frame_rows) if FRAME_BUDGET_ENABLED else None
    return {
        "duration_s": _r(duration),
        "duration_mono_s": _r(duration_mono),
        "frames": frames,
        "temporal_events": temporal_events_rounded,
        "probes": probes,
        "rates":  {k: _r(v) for k, v in rates_agg.items()},
        "gauges": {k: _r(v) for k, v in gauges_agg.items()},
        "fast_probes": fast_probes,
        "fast_rates":  {k: _r(v) for k, v in rates_fast.items()},
        "fast_gauges": {k: _r(v) for k, v in gauges_fast.items()},
        "buckets": buckets_block,
        "frame_budget": frame_budget_block,
    }

def _build_temporal_deltas(ref_block: dict, target_block: dict) -> dict:
    """
    Construit les deltas temporels entre target et référence.
    Couvre duration_mono_s, frames, et les sous-clés temporal_events par canal.
    Structure retournée :
      {
        "duration_mono_s": {"delta_pct": float|None},
        "agg":   {"frames": {...}, "median_interval_s": {...}, "gaps_stat": {...}, "gaps_fixed": {...}},
        "frame": {...},
        "fast":  {...},
      }
    """
    deltas: dict = {"duration_mono_s": {"delta_pct": delta_pct(target_block.get("duration_mono_s"),ref_block.get("duration_mono_s"))}}
    target_events = target_block.get("temporal_events", {})
    ref_events = ref_block.get("temporal_events", {})
    target_frames = target_block.get("frames", {})
    ref_frames = ref_block.get("frames", {})
    for canal in ("agg", "frame", "fast"):
        t_canal = target_events.get(canal, {})
        r_canal = ref_events.get(canal, {})
        deltas[canal] = {
            "frames": {"delta_pct": delta_pct(target_frames.get(canal), ref_frames.get(canal))},
            "median_interval_s": {"delta_pct": delta_pct(t_canal.get("median_interval_s"),r_canal.get("median_interval_s"),)},
            "gaps_stat": {"delta_pct": delta_pct(t_canal.get("gaps_stat"), r_canal.get("gaps_stat"))},
            "gaps_fixed": {"delta_pct": delta_pct(t_canal.get("gaps_fixed"), r_canal.get("gaps_fixed"))},
        }
    return deltas

def _build_probe_deltas(target_probes: dict, ref_probes: dict, *, include_anomalies: bool = False) -> dict:
    """
    Construit les deltas pour toutes les sondes présentes dans target ou ref.
    Couvre avg, min, max, tous les percentiles (exact + approx),
    et les quartiles q1/q3/iqr (exact + approx) — S4bis.
    """
    all_keys = set(target_probes) | set(ref_probes)
    deltas: dict[str, dict] = {}
    for key in sorted(all_keys):
        t = target_probes.get(key, {})
        r = ref_probes.get(key, {})
        entry: dict = {}
        for field in ("avg", "min", "max"):
            entry[f"{field}_delta_pct"] = delta_pct(t.get(field), r.get(field))
        for pct in PERCENTILES:
            for method in ("exact", "approx"):
                fname = f"p{pct}_{method}"
                entry[f"{fname}_delta_pct"] = delta_pct(t.get(fname), r.get(fname))
        # S4bis — quartiles + IQR
        for stat in ("q1", "q3", "iqr"):
            for method in ("exact", "approx"):
                fname = f"{stat}_{method}"
                entry[f"{fname}_delta_pct"] = delta_pct(t.get(fname), r.get(fname))
         # S5a — Skewness + Kurtosis excess (deltas absolus, D8)
        for stat in ("skewness", "kurtosis_excess"):
            for method in ("exact", "approx"):
                fname = f"{stat}_{method}"
                entry[f"{fname}_delta"] = delta_abs(t.get(fname), r.get(fname))
        # S5b — Spikes + Drift (deltas absolus, E13)
        # spike_max_value et drift_intercept exclus (Q-Patch-3 A)
        if include_anomalies:
            for fname in ("spike_count", "spike_max_deviation", "drift_slope", "drift_r2"):
                entry[f"{fname}_delta"] = delta_abs(t.get(fname), r.get(fname))
        deltas[key] = entry
    return deltas

def _build_scalar_deltas(target: dict, ref: dict) -> dict:
    """Construit les deltas pour rates ou gauges (valeurs scalaires)."""
    all_keys = set(target) | set(ref)
    return {
        key: {"delta_pct": delta_pct(target.get(key), ref.get(key))}
        for key in sorted(all_keys)
    }

def _appeared_disappeared(target_map: dict, ref_map: dict) -> tuple[list, list]:
    """
    Retourne (appeared_in_target, disappeared_in_target).

    Générique : opère sur n'importe quel mapping {clé: ...} — utilisé pour
    les sondes (`probes`), les rates (`rates`) et les gauges (`gauges`).
    Seules les clés sont comparées ; les valeurs sont ignorées.
    """
    t_keys = set(target_map)
    r_keys = set(ref_map)
    return sorted(t_keys - r_keys), sorted(r_keys - t_keys)

def _build_frame_budget_deltas(target_fb: dict | None, ref_fb: dict | None) -> dict | None:
    """
    Deltas S6a — frame budget aligné target vs référence.
    Returns:
        None si l'un des deux blocs est None (frame_budget désactivé d'un côté).
    """
    if target_fb is None or ref_fb is None:
        return None
    t_groups = target_fb.get("groups", {})
    r_groups = ref_fb.get("groups", {})
    appeared = sorted(set(t_groups) - set(r_groups))
    disappeared = sorted(set(r_groups) - set(t_groups))
    aligned = sorted(set(t_groups) & set(r_groups))
    group_deltas: dict[str, dict] = {}
    for name in aligned:
        t_g = t_groups[name]
        r_g = r_groups[name]
        group_deltas[name] = {
            "pct_delta":     _r(delta_abs(t_g.get("pct"), r_g.get("pct"))),
            "sum_ms_delta_pct": delta_pct(t_g.get("sum_ms"), r_g.get("sum_ms")),
            "presence_rate_delta": _r(delta_abs(
                t_g.get("presence_rate"), r_g.get("presence_rate")
            )),
        }
    return {
        "total_ms_delta_pct":    delta_pct(target_fb.get("total_ms"), ref_fb.get("total_ms")),
        "unaccounted_pct_delta": _r(delta_abs(
            target_fb.get("unaccounted_pct"), ref_fb.get("unaccounted_pct")
        )),
        "groups":              group_deltas,
        "appeared_groups":     appeared,
        "disappeared_groups":  disappeared,
    }

def _build_buckets_deltas(target_buckets:dict | None,ref_buckets:dict | None) -> dict | None:
    """
    Deltas P1 — par bucket aligné : cold vs cold, hot_i vs hot_i.
    Retourne None si l'un ou l'autre est absent (session pré-S4).
    """
    if target_buckets is None or ref_buckets is None:
        return None
    # Cold
    cold_delta = {
        "duration_delta_pct": delta_pct(target_buckets["cold"].get("duration_s"),ref_buckets["cold"].get("duration_s")),
        "probes":      _build_probe_deltas(target_buckets["cold"].get("probes", {}),ref_buckets["cold"].get("probes", {}), include_anomalies=True),
        "rates":       _build_scalar_deltas(target_buckets["cold"].get("rates", {}),ref_buckets["cold"].get("rates", {})),
        "gauges":      _build_scalar_deltas(target_buckets["cold"].get("gauges", {}),ref_buckets["cold"].get("gauges", {})),
        "fast_probes": _build_probe_deltas(target_buckets["cold"].get("fast_probes", {}),ref_buckets["cold"].get("fast_probes", {}), include_anomalies=True),
        "fast_rates":  _build_scalar_deltas(target_buckets["cold"].get("fast_rates", {}),ref_buckets["cold"].get("fast_rates", {})),
        "fast_gauges": _build_scalar_deltas(target_buckets["cold"].get("fast_gauges", {}),ref_buckets["cold"].get("fast_gauges", {})),
        "frame_budget": _build_frame_budget_deltas(target_buckets["cold"].get("frame_budget"),ref_buckets["cold"].get("frame_budget")),
    }
    # Hot alignés
    t_hot = target_buckets.get("hot", [])
    r_hot = ref_buckets.get("hot", [])
    n_aligned = min(len(t_hot), len(r_hot))
    hot_deltas = []
    for i in range(n_aligned):
        th, rh = t_hot[i], r_hot[i]
        hot_deltas.append({
            "index": i,
            "duration_delta_pct": delta_pct(th.get("duration_s"), rh.get("duration_s")),
            "probes":      _build_probe_deltas(th.get("probes",{}), rh.get("probes",{}), include_anomalies=True),
            "rates":       _build_scalar_deltas(th.get("rates",{}), rh.get("rates",{})),
            "gauges":      _build_scalar_deltas(th.get("gauges",{}), rh.get("gauges",{})),
            "fast_probes": _build_probe_deltas(th.get("fast_probes",{}), rh.get("fast_probes",{}), include_anomalies=True),
            "fast_rates":  _build_scalar_deltas(th.get("fast_rates",{}), rh.get("fast_rates",{})),
            "fast_gauges": _build_scalar_deltas(th.get("fast_gauges",{}), rh.get("fast_gauges",{})),
            "frame_budget": _build_frame_budget_deltas(th.get("frame_budget"),rh.get("frame_budget")),
        })
    unaligned_hot = list(range(n_aligned, max(len(t_hot), len(r_hot))))
    # Tail
    t_tail = target_buckets.get("tail")
    r_tail = ref_buckets.get("tail")
    if t_tail is not None and r_tail is not None:
        tail_status = "aligned"
        tail_delta = {"duration_delta_pct": delta_pct(t_tail.get("duration_s"), r_tail.get("duration_s"))}
    elif t_tail is None and r_tail is None:
        tail_status = "both_absent"
        tail_delta = None
    elif t_tail is None:
        tail_status = "target_absent"
        tail_delta = None
    else:
        tail_status = "ref_absent"
        tail_delta = None
    return {
        "cold":          cold_delta,
        "hot":           hot_deltas,
        "unaligned_hot": unaligned_hot,
        "tail_status":   tail_status,
        "tail":          tail_delta,
    }

def build_comparison(ref_session_id: str, ref_block: dict, target_block: dict) -> dict:
    """Construit un bloc de comparaison complet target vs référence.
    Ventilation par canal :
      - agg  : probes / rates / gauges (clés top-level non préfixées)
      - fast : fast_probes / fast_rates / fast_gauges (clés top-level préfixées 'fast_')
    """
    # ── Canal agg ──
    appeared_probes, disappeared_probes = _appeared_disappeared(target_block["probes"], ref_block["probes"])
    appeared_rates, disappeared_rates = _appeared_disappeared(target_block["rates"], ref_block["rates"])
    appeared_gauges, disappeared_gauges = _appeared_disappeared(target_block["gauges"], ref_block["gauges"])
    # ── Canal fast ──
    appeared_fast_probes, disappeared_fast_probes = _appeared_disappeared(target_block["fast_probes"], ref_block["fast_probes"])
    appeared_fast_rates, disappeared_fast_rates = _appeared_disappeared(target_block["fast_rates"], ref_block["fast_rates"])
    appeared_fast_gauges, disappeared_fast_gauges = _appeared_disappeared(target_block["fast_gauges"], ref_block["fast_gauges"])
    return {
        "reference_session": ref_session_id,
        "reference": ref_block,
        "deltas": {
            "temporal": _build_temporal_deltas(ref_block, target_block),
            # Canal agg
            "probes": _build_probe_deltas(target_block["probes"], ref_block["probes"]),
            "rates": _build_scalar_deltas(target_block["rates"], ref_block["rates"]),
            "gauges": _build_scalar_deltas(target_block["gauges"], ref_block["gauges"]),
            # Canal fast
            "fast_probes": _build_probe_deltas(target_block["fast_probes"], ref_block["fast_probes"]),
            "fast_rates": _build_scalar_deltas(target_block["fast_rates"], ref_block["fast_rates"]),
            "fast_gauges": _build_scalar_deltas(target_block["fast_gauges"], ref_block["fast_gauges"]),
             # S6a — Frame budget (session-level)
            "frame_budget": _build_frame_budget_deltas(target_block.get("frame_budget"),ref_block.get("frame_budget")),
        },
        "buckets": _build_buckets_deltas(target_block.get("buckets"),ref_block.get("buckets")),
        # Canal agg
        "appeared_probes": appeared_probes,
        "disappeared_probes": disappeared_probes,
        "appeared_rates": appeared_rates,
        "disappeared_rates": disappeared_rates,
        "appeared_gauges": appeared_gauges,
        "disappeared_gauges": disappeared_gauges,
        # Canal fast
        "appeared_fast_probes": appeared_fast_probes,
        "disappeared_fast_probes": disappeared_fast_probes,
        "appeared_fast_rates": appeared_fast_rates,
        "disappeared_fast_rates": disappeared_fast_rates,
        "appeared_fast_gauges": appeared_fast_gauges,
        "disappeared_fast_gauges": disappeared_fast_gauges,
    }
