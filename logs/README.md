# logs/

Sorties d'exécution de l'application et de ses outils d'analyse.

## Sous-dossiers

| Dossier    | Producteur         | Contenu                                                   |
| ---------- | ------------------ | --------------------------------------------------------- |
| `json/`    | `core/bench.py`    | Sessions de bench fraîches (3 fichiers JSONL par session) |
| `results/` | `bench_compare.py` | Sessions archivées + rapports comparatifs `.json`         |

## Convention de nommage

- `session_id` : format `YYYYMMDD_HHMMSS` (tri lexicographique = tri chronologique).
- Fichiers JSONL : `bench_{frame|agg|fast}_<session_id>.jsonl`.
- Dossier archive : `results/<session_id>/`.
- Rapport comparatif : `results/<session_id>/<session_id>.json`.

## Documentation associée

- Format des lignes JSONL → [`docs/bench-jsonl-schema.md`](../docs/bench-jsonl-schema.md)
- Outil d'analyse comparative → [`docs/bench-compare.md`](../docs/bench-compare.md)
