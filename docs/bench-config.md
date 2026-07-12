# Bench — Configuration

Instrumentation centralisée via `core/bench.py`.

Cinq types de sondes :

- `timer` — context manager mesurant un bloc, écrit via `probe()` en sortie (`with bench.timer(name)`)
- `probe` — valeur scalaire échantillonnée (durée, score, aire…)
- `count` — compteur cumulatif incrémental
- `gauge` — valeur instantanée écrasée à chaque mesure
- `event` — enregistrement structuré (lifecycle / détection)

## Configuration ([`config.yaml`](./../config/config.yaml))

> Schéma complet des fichiers JSONL produits : [`bench-jsonl-schema.md`](bench-jsonl-schema.md).
> Catalogue des sondes émises : [`bench-probes.md`](bench-probes.md).

```yaml
debug:
  bench:
    enabled: true # Active BenchRegistry + démarre les writers si writer.enabled=true
    history_window_s: 60.0 # fenêtre de rétention pour les snapshots agg/fast

    writer:
      enabled: true # Maître : false = aucun writer démarré
      queue_maxsize: 10000 # Drop + bench.count("bench_writer_dropped") au-delà
      shutdown_timeout_s: 2.0 # Délai max accordé à chaque writer pour vider sa queue
      max_chars: 3000000 # null = illimité (pas de rotation). Sinon : rotation par fichier au-delà de N caractères écrits (\n inclus)
      session_id_format: "%Y%m%d_%H%M%S" # Inséré dans le nom de fichier avant l'extension

    agg:
      enabled: true
      path: "logs/json/bench_agg.jsonl" # → bench_agg_{session_id}.jsonl
      interval_s: 1.0

    frame:
      enabled: true
      path: "logs/json/bench_frame.jsonl" # → bench_frame_{session_id}.jsonl

    fast:
      enabled: true
      path: "logs/json/bench_fast.jsonl" # → bench_fast_{session_id}.jsonl
      interval_s: 1.0

    # ── Canal LifeCycle (snapshot_events, 1 ligne / events capturée)
    events:
      enabled: true
      path: "logs/json/bench_events.jsonl"
    detections:
      enabled: true
      path: logs/json/bench_detections.jsonl

    frame_dumper:
      enabled: true
      path: "logs/frames"
      jpeg_quality: 75 # configurable
      tail_frames: 2 # 0 (couvertes) | 2 (étendu)
      ring_size: 240 # Stratégie B : survie_tracker_max + tail + marge
      queue_maxsize: 256 # DROP au-delà (put_nowait + except Full)
```

## Hiérarchie d'activation

```yaml
bench.enabled: false  → bench désactivé (noop)
bench.enabled: true   → bench actif
  debug.bench.writer.enabled: false  → writers inactifs (bench.timer/count/gauge still work)
  debug.bench.writer.enabled: true   → writers actifs (5 canaux : agg / frame / fast / events / detections)
```

## Canaux JSONL

| Canal        | Cadence                     | Sections                 | Filtre sondes exclues              |
| ------------ | --------------------------- | ------------------------ | ---------------------------------- |
| `agg`        | 1 ligne / `interval_s`      | probes / gauges / rates  | `fast_*` et `bench_writer_*`       |
| `fast`       | 1 ligne / `interval_s`      | probes / gauges / rates  | **uniquement** les sondes `fast_*` |
| `frame`      | 1 ligne / frame (push main) | probes / gauges / counts | `fast_*` et `bench_writer_*`       |
| `events`     | 1 ligne / batch (push main) | events.records           | aucune (lifecycle)                 |
| `detections` | 1 ligne / batch (push main) | detections.records       | aucune (détections raw)            |

**Note sur les sondes `F_*` (variantes exactes des sondes fast)** : les sondes `F_*` ne sont pas filtrées par `_is_fast_probe` ni par `_is_writer_probe`. Elles apparaissent donc dans les snapshots `frame` (canal différentiel par frame) pour fournir des mesures **exactes non agrégées par frame**, en complément des `fast_*` agrégés dans le canal `fast`. Ce comportement est intentionnel..

> Voir [`bench-jsonl-schema.md`](bench-jsonl-schema.md) pour la structure exacte de chaque ligne

---

## Rotation des fichiers JSONL

Quand `debug.bench.writer.max_chars` est configuré, chaque writer effectue une rotation :

- le compteur `_char_count` est incrémenté de la taille de chaque ligne écrite
- quand le seuil est atteint → fermeture du fichier courant, ouverture de `_<session_id>_<index>` (index à partir de 0) garantie : au moins 1 ligne par fichier (rotation après écriture complète)
