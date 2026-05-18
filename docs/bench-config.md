# Bench — Configuration

Instrumentation centralisée via `core/bench.py`.

Quatre types de sondes :

- `timer` — context manager mesurant un bloc, écrit via `probe()` en sortie (`with bench.timer(name)`)
- `probe` — valeur scalaire échantillonnée (durée, score, aire…)
- `count` — compteur cumulatif incrémental
- `gauge` — valeur instantanée écrasée à chaque mesure

## Configuration ([`config.yaml`](./../config/config.yaml))

> Schéma complet des fichiers JSONL produits : [`bench-jsonl-schema.md`](bench-jsonl-schema.md).
> Catalogue des sondes émises : [`bench-probes.md`](bench-probes.md).

```yaml
debug:
  bench:
    enabled: true # Active BenchRegistry + démarre les writers si writer.enabled=true
    history_window_s: 60 # Fenêtre glissante conservée en mémoire pour summary_window() (s)

    writer:
      enabled: true # Maître : false = aucun writer démarré
      queue_maxsize: 10000 # Drop + bench.count("bench_writer_dropped") au-delà
      shutdown_timeout_s: 2.0 # Délai max accordé à chaque writer pour vider sa queue
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
```

**Hiérarchie d'activation** :

- `debug.bench.enabled: false` → `BenchRegistry` désactivé, aucun writer démarré, toutes les sondes sont des no-ops.
- `debug.bench.writer.enabled: false` → sondes actives en mémoire, aucun fichier écrit.
- Chaque canal (`agg` / `frame` / `fast`) peut être désactivé indépendamment via son propre `enabled`.

**Canaux JSONL** :

| Canal   | Fichier                          | Cadence                     |
| ------- | -------------------------------- | --------------------------- |
| `frame` | `bench_frame_{session_id}.jsonl` | 1 ligne / frame capturée    |
| `agg`   | `bench_agg_{session_id}.jsonl`   | 1 ligne / `agg.interval_s`  |
| `fast`  | `bench_fast_{session_id}.jsonl`  | 1 ligne / `fast.interval_s` |

> Voir [`bench-jsonl-schema.md`](bench-jsonl-schema.md) pour la structure exacte de chaque ligne
