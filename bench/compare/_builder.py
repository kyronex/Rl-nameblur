# bench/compare/_builder.py

from bench.compare._config import (PERCENTILES,_r)
from bench.compare._stats import (_agg_probes,_agg_rates,_agg_gauges,_session_duration,_extract_timeline,_compute_frames,_compute_duration_mono,_compute_temporal_events,_collect_frame_samples,_collect_fast_approx_samples,_percentile_value,_delta_pct)


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
    base_probes_agg  = _agg_probes(agg_rows)
    base_probes_fast = _agg_probes(fast_rows)
    rates_agg   = _agg_rates(agg_rows)
    rates_fast  = _agg_rates(fast_rows)
    gauges_agg  = _agg_gauges(agg_rows)
    gauges_fast = _agg_gauges(fast_rows)

    duration = _session_duration(agg_rows)

    # ── Samples pour percentiles ──
    exact_samples, frame_approx = _collect_frame_samples(frame_rows)
    fast_approx = _collect_fast_approx_samples(fast_rows)
    approx_samples: dict[str, list[float]] = {**frame_approx, **fast_approx}

    # ── Construction probes (canal agg) ──
    probes: dict[str, dict] = {}
    for probe_name, stats in base_probes_agg.items():
        pct_block = _build_percentile_block(
            probe_name, exact_samples, approx_samples, channel="agg"
        )
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
        pct_block = _build_percentile_block(
            probe_name, exact_samples, approx_samples, channel="fast"
        )
        fast_probes[probe_name] = {
            "avg": _r(stats["avg"]),
            "min": _r(stats["min"]),
            "max": _r(stats["max"]),
            "count_fast": stats["count_agg"],
            **{k: _r(v) for k, v in pct_block.items()},
        }

    # ── Analyse temporelle (3 canaux) ──
    timeline_agg   = _extract_timeline(agg_rows)
    timeline_frame = _extract_timeline(frame_rows)
    timeline_fast  = _extract_timeline(fast_rows)

    duration_mono = _compute_duration_mono(timeline_agg)

    frames = {
        "agg":   _compute_frames(timeline_agg),
        "frame": _compute_frames(timeline_frame),
        "fast":  _compute_frames(timeline_fast),
    }

    temporal_events = {
        "agg":   _compute_temporal_events(timeline_agg,   "agg"),
        "frame": _compute_temporal_events(timeline_frame, "frame"),
        "fast":  _compute_temporal_events(timeline_fast,  "fast"),
    }

    temporal_events_rounded: dict[str, dict] = {}
    for canal, events in temporal_events.items():
        temporal_events_rounded[canal] = {
            "median_interval_s": _r(events["median_interval_s"]),
            "gaps_stat":  events["gaps_stat"],
            "gaps_fixed": events["gaps_fixed"],
        }

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
    }

def _build_percentile_block(probe_name: str,exact_samples: dict[str, list[float]],approx_samples: dict[str, list[float]],*,channel: str) -> dict:
    """
    Construit le bloc percentiles pour une sonde.
    Args:
        probe_name: nom de la sonde.
        exact_samples: échantillons exacts (count==1) issus du canal frame.
        approx_samples: échantillons approximés issus du canal agrégé concerné.
        channel: "agg" → exact calculable depuis frame_rows.
                 "fast" → pas d'exact (samples_exact=0, *_exact=null).
    Returns:
        dict avec samples_exact, samples_approx, et p{pct}_exact / p{pct}_approx.
    """
    has_exact = (channel == "agg")

    exact_data = exact_samples.get(probe_name, []) if has_exact else []
    approx_data = approx_samples.get(probe_name, [])

    block: dict = {
        "samples_exact": len(exact_data) if has_exact else 0,
        "samples_approx": len(approx_data),
    }

    for pct in PERCENTILES:
        block[f"p{pct}_exact"] = _percentile_value(exact_data, pct) if has_exact else None
        block[f"p{pct}_approx"] = _percentile_value(approx_data, pct)

    return block

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
    deltas: dict = {
        "duration_mono_s": {
            "delta_pct": _delta_pct(
                target_block.get("duration_mono_s"),
                ref_block.get("duration_mono_s"),
            )
        }
    }

    target_events = target_block.get("temporal_events", {})
    ref_events = ref_block.get("temporal_events", {})
    target_frames = target_block.get("frames", {})
    ref_frames = ref_block.get("frames", {})

    for canal in ("agg", "frame", "fast"):
        t_canal = target_events.get(canal, {})
        r_canal = ref_events.get(canal, {})
        deltas[canal] = {
            "frames": {
                "delta_pct": _delta_pct(
                    target_frames.get(canal), ref_frames.get(canal)
                )
            },
            "median_interval_s": {
                "delta_pct": _delta_pct(
                    t_canal.get("median_interval_s"),
                    r_canal.get("median_interval_s"),
                )
            },
            "gaps_stat": {
                "delta_pct": _delta_pct(t_canal.get("gaps_stat"), r_canal.get("gaps_stat"))
            },
            "gaps_fixed": {
                "delta_pct": _delta_pct(t_canal.get("gaps_fixed"), r_canal.get("gaps_fixed"))
            },
        }

    return deltas


def _build_probe_deltas(target_probes: dict, ref_probes: dict) -> dict:
    """
    Construit les deltas pour toutes les sondes présentes dans target ou ref.
    Couvre avg, min, max et tous les percentiles (exact + approx).
    """
    all_keys = set(target_probes) | set(ref_probes)
    deltas: dict[str, dict] = {}

    for key in sorted(all_keys):
        t = target_probes.get(key, {})
        r = ref_probes.get(key, {})
        entry: dict = {}

        for field in ("avg", "min", "max"):
            entry[f"{field}_delta_pct"] = _delta_pct(t.get(field), r.get(field))

        for pct in PERCENTILES:
            for method in ("exact", "approx"):
                fname = f"p{pct}_{method}"
                entry[f"{fname}_delta_pct"] = _delta_pct(t.get(fname), r.get(fname))

        deltas[key] = entry

    return deltas

def _build_scalar_deltas(target: dict, ref: dict) -> dict:
    """Construit les deltas pour rates ou gauges (valeurs scalaires)."""
    all_keys = set(target) | set(ref)
    return {
        key: {"delta_pct": _delta_pct(target.get(key), ref.get(key))}
        for key in sorted(all_keys)
    }


def build_comparison(ref_session_id: str, ref_block: dict, target_block: dict) -> dict:
    """Construit un bloc de comparaison complet target vs référence.

    Ventilation par canal :
      - agg  : probes / rates / gauges (clés top-level non préfixées)
      - fast : fast_probes / fast_rates / fast_gauges (clés top-level préfixées 'fast_')
    """
    # ── Canal agg ──
    appeared_probes, disappeared_probes = _appeared_disappeared(
        target_block["probes"], ref_block["probes"]
    )
    appeared_rates, disappeared_rates = _appeared_disappeared(
        target_block["rates"], ref_block["rates"]
    )
    appeared_gauges, disappeared_gauges = _appeared_disappeared(
        target_block["gauges"], ref_block["gauges"]
    )

    # ── Canal fast ──
    appeared_fast_probes, disappeared_fast_probes = _appeared_disappeared(
        target_block["fast_probes"], ref_block["fast_probes"]
    )
    appeared_fast_rates, disappeared_fast_rates = _appeared_disappeared(
        target_block["fast_rates"], ref_block["fast_rates"]
    )
    appeared_fast_gauges, disappeared_fast_gauges = _appeared_disappeared(
        target_block["fast_gauges"], ref_block["fast_gauges"]
    )

    return {
        "reference_session": ref_session_id,
        "reference": ref_block,
        "deltas": {
            "temporal": _build_temporal_deltas(ref_block, target_block),
            # Canal agg
            "probes": _build_probe_deltas(
                target_block["probes"], ref_block["probes"]
            ),
            "rates": _build_scalar_deltas(
                target_block["rates"], ref_block["rates"]
            ),
            "gauges": _build_scalar_deltas(
                target_block["gauges"], ref_block["gauges"]
            ),
            # Canal fast
            "fast_probes": _build_probe_deltas(
                target_block["fast_probes"], ref_block["fast_probes"]
            ),
            "fast_rates": _build_scalar_deltas(
                target_block["fast_rates"], ref_block["fast_rates"]
            ),
            "fast_gauges": _build_scalar_deltas(
                target_block["fast_gauges"], ref_block["fast_gauges"]
            ),
        },
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