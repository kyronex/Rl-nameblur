# [`bench_compare.py`](./../logs/bench_compare.py)

Script d'analyse comparative des sessions de benchmark de l'application RL-NameBlur.

À chaque exécution, sélectionne la session la plus récente disponible
(toutes sources confondues : `logs/json/` et `logs/results/`) comme cible,
et la compare à deux références :

- **Référence absolue** : la session la plus ancienne (toutes sources confondues).
- **Référence relative** : l'avant-dernière session (cible précédente).

Produit un JSON structuré contenant les deux comparaisons.

Si une seule session est disponible au total, le script produit un rapport
en **mode session unique** : seuls les agrégats de la cible sont calculés,
les deux comparaisons valent `null`.

---

## Prérequis

- Python 3.10+
- Dépendances : stdlib uniquement (`json`, `pathlib`, `datetime`, `statistics`,
  `shutil`, `sys`, `logging`)
- Fichiers JSONL produits par `core/bench.py` (canaux `frame`, `agg`, `fast`)

---

## Structure des dossiers

### Avant exécution

```text
Rl-nameblur/
└── logs/
    ├── json/
    │   ├── bench_agg_20260519_091540.jsonl
    │   ├── bench_fast_20260519_091540.jsonl
    │   └── bench_frame_20260519_091540.jsonl
    └── results/
        ├── 20260517_205106/
        │   ├── bench_agg_20260517_205106.jsonl
        │   ├── bench_fast_20260517_205106.jsonl
        │   └── bench_frame_20260517_205106.jsonl
        └── 20260518_103022/
            ├── bench_agg_20260518_103022.jsonl
            ├── bench_fast_20260518_103022.jsonl
            ├── bench_frame_20260518_103022.jsonl
            └── 20260518_103022.json
```

### Après exécution

```text
Rl-nameblur/
└── logs/
    ├── json/                          # Vidé des sessions traitées
    └── results/
        ├── 20260517_205106/           # Référence absolue (inchangée)
        │   └── ...
        ├── 20260518_103022/           # Référence relative (inchangée)
        │   └── ...
        └── 20260519_091540/           # Cible — déplacée depuis logs/json/
            ├── bench_agg_20260519_091540.jsonl
            ├── bench_fast_20260519_091540.jsonl
            ├── bench_frame_20260519_091540.jsonl
            └── 20260519_091540.json   # Rapport produit
```

---

## Sources de sessions

Le script lit **les deux répertoires** pour constituer l'ensemble des sessions
disponibles. La cible et les références peuvent provenir de l'un comme de l'autre.

| Répertoire      | Rôle               | Mouvement de fichiers                                      |
| --------------- | ------------------ | ---------------------------------------------------------- |
| `logs/json/`    | Sessions neuves    | Déplacées vers `results/` après succès                     |
| `logs/results/` | Sessions archivées | Lues ; modifiées uniquement dans les cas listés ci-dessous |

### Cas de modification de `logs/results/`

Trois cas — et trois seulement — où des fichiers de `logs/results/` sont modifiés :

1. **Doublon de `session_id`** entre `logs/json/` et `logs/results/`
   `logs/json/` est prioritaire (cas attendu uniquement après renommage manuel ou rejeu).
   Un avertissement est émis à l'ingestion.
   Le dossier `logs/results/<session_id>/` est **vidé de ses fichiers de premier niveau** (JSONL **et** rapport préexistant éventuel) avant déplacement des fichiers venant de `logs/json/`. Le rapport est ensuite régénéré.

   > **Note d'implémentation (v1)** : le vidage est non récursif (`Path.iterdir()` + `unlink()`). Le dossier n'est pas censé contenir de sous-dossier ; si tel était le cas, l'opération échouerait avec `OSError`. Aucun sous-dossier n'est créé par le pipeline actuel.

2. **Cible déjà présente dans `logs/results/`**
   Si la session la plus récente se trouve dans `logs/results/` (aucune session neuve dans `logs/json/` portant le même `session_id`), aucun JSONL n'est déplacé ni supprimé. Seul le rapport JSON `<target_session>.json` est (re)généré dans le dossier existant, écrasant atomiquement tout rapport préexistant du même nom (écriture `.tmp` + `replace`).

3. **Création du dossier cible**
   Si la cible vient de `logs/json/` et que `logs/results/<session_id>/` n'existe pas, il est créé.

Dans tous les autres cas, `logs/results/` est en lecture seule.

---

## Logique de sélection

Soit **N** = nombre total de sessions disponibles (union `logs/json/` + `logs/results/`,dédoublonnée).

| Condition | Comportement                                                                                          |
| --------- | ----------------------------------------------------------------------------------------------------- |
| N == 0    | Sortie : aucune session disponible, message explicite                                                 |
| N == 1    | Mode session unique : rapport produit, `comparisons.absolute` et `comparisons.relative` valent `null` |
| N == 2    | Cible + référence absolue uniquement ; `comparisons.relative` vaut `null`                             |
| N >= 3    | Cible + référence absolue + référence relative                                                        |

### Rôles attribués

- **Cible** : session avec le `session_id` le plus récent (tri lexicographique sur le format `YYYYMMDD_HHMMSS`).
- **Référence absolue** : session avec le `session_id` le plus ancien.
- **Référence relative** : session immédiatement antérieure à la cible (avant-dernière dans l'ordre chronologique).
- En mode N==1 : la cible est l'unique session disponible, aucune référence n'est attribuée.
- En mode N==2 : seule la référence absolue est attribuée (`sorted_ids[0]`). `comparisons.relative` vaut `null` : l'avant-dernière session étant alors identique à la référence absolue, la comparaison serait redondante. Cette redondance est évitée par construction — `relative_id` n'est attribué que si **N >= 3** (`relative_id = sorted_ids[-2] if len(sorted_ids) >= 3 else None`).

---

## Utilisation

```bash
python bench_compare.py
```

Exécution non interactive. Aucune option CLI en v1.

---

## Format du JSON de sortie

Fichier : `logs/results/<target_session>/<target_session>.json`

> Représentation typée du schéma. Les clés entre `<...>` sont dynamiques.

```json
{
  "schema_version": "int",
  "generated_at": "datetime_iso8601",
  "target_session": "session_id",
  "target": {
    "duration_s": "float",
    "duration_mono_s": "float",
    "frames": {
      "agg": "int",
      "frame": "int",
      "fast": "int"
    },
    "temporal_events": {
      "agg": { "median_interval_s": "float | null", "gaps_stat": "int", "gaps_fixed": "int | null" },
      "frame": { "median_interval_s": "float | null", "gaps_stat": "int", "gaps_fixed": "int | null" },
      "fast": { "median_interval_s": "float | null", "gaps_stat": "int", "gaps_fixed": "int | null" }
    },
    "probes": {
      "<probe_name>": {
        "avg": "float",
        "min": "float",
        "max": "float",
        "count_agg": "int",
        "samples_exact": "int",
        "samples_approx": "int",
        "p90_exact": "float | null",
        "p95_exact": "float | null",
        "p99_exact": "float | null",
        "p90_approx": "float | null",
        "p95_approx": "float | null",
        "p99_approx": "float | null"
      }
    },
    "rates": {"<rate_name>": "float"},
    "gauges": {"<gauge_name>": "float"},
    "fast_probes": {
      "<probe_name>": {
        "avg": "float",
        "min": "float",
        "max": "float",
        "count_fast": "int",
        "samples_exact": "int",
        "samples_approx": "int",
        "p90_exact": "float | null",
        "p95_exact": "float | null",
        "p99_exact": "float | null",
        "p90_approx": "float | null",
        "p95_approx": "float | null",
        "p99_approx": "float | null"
      }
    },
    "fast_rates": {"<rate_name>": "float"},
    "fast_gauges": {"<gauge_name>": "float"},
    "buckets": {
      "sync_metadata": {
        "cold_end_target_s":  "float",
        "cold_end_real_s":    "float",
        "cold_drift_s":       "float",
        "cold_drift_warning": "bool",
        "cold_truncated":     "bool",
        "fast_enabled":       "bool"
      },
      "cold": {
        "mono_start":  "float",
        "mono_end":    "float",
        "duration_s":  "float",
        "frames": { "agg": "int", "frame": "int", "fast": "int" },
        "probes":      { "<probe_name>": { "...même structure que target.probes..." } },
        "rates":       { "<rate_name>":  "float" },
        "gauges":      { "<gauge_name>": "float" },
        "fast_probes": { "<probe_name>": { "...même structure que target.fast_probes..." } },
        "fast_rates":  { "<rate_name>":  "float" },
        "fast_gauges": { "<gauge_name>": "float" }
      },
      "hot": [
        {
          "index":           "int",
          "mono_start":      "float",
          "mono_end":        "float",
          "duration_s":      "float",
          "is_pivot_snapped":"bool",
          "frames": { "agg": "int", "frame": "int", "fast": "int" },
          "probes":      { "<probe_name>": { "..." } },
          "rates":       { "<rate_name>":  "float" },
          "gauges":      { "<gauge_name>": "float" },
          "fast_probes": { "<probe_name>": { "..." } },
          "fast_rates":  { "<rate_name>":  "float" },
          "fast_gauges": { "<gauge_name>": "float" }
        }
      ],
      "tail": {
        "mono_start":  "float",
        "mono_end":    "float",
        "duration_s":  "float",
        "is_partial":  true,
        "frames": { "agg": "int", "frame": "int", "fast": "int" },
        "probes":      { "<probe_name>": { "..." } },
        "rates":       { "<rate_name>":  "float" },
        "gauges":      { "<gauge_name>": "float" },
        "fast_probes": { "<probe_name>": { "..." } },
        "fast_rates":  { "<rate_name>":  "float" },
        "fast_gauges": { "<gauge_name>": "float" }
      }
    }
  },
  "comparisons": {
    "<comparison_type>": {
      "reference_session": "session_id",
      "reference": { "...même structure que target..." },
      "deltas": {
        "temporal": {
          "agg":   { "frames": {}, "median_interval_s": {}, "gaps_stat": {}, "gaps_fixed": {} },
          "frame": { "frames": {}, "median_interval_s": {}, "gaps_stat": {}, "gaps_fixed": {} },
          "fast":  { "frames": {}, "median_interval_s": {}, "gaps_stat": {}, "gaps_fixed": {} }
        },
        "probes":      { "<probe_name>": { "avg_delta_pct": "float | null", "...percentile_delta_pct..." } },
        "rates":       { "<rate_name>":  { "delta_pct": "float | null" } },
        "gauges":      { "<gauge_name>": { "delta_pct": "float | null" } },
        "fast_probes": { "<probe_name>": { "avg_delta_pct": "float | null", "...percentile_delta_pct..." } },
        "fast_rates":  { "<rate_name>":  { "delta_pct": "float | null" } },
        "fast_gauges": { "<gauge_name>": { "delta_pct": "float | null" } },
        "buckets": {
          "cold": {
            "duration_delta_pct": "float | null",
            "probes":      { "<probe_name>": { "avg_delta_pct": "float | null", "...percentile_delta_pct..." } },
            "rates":       { "<rate_name>":  { "delta_pct": "float | null" } },
            "gauges":      { "<gauge_name>": { "delta_pct": "float | null" } },
            "fast_probes": { "<probe_name>": { "..." } },
            "fast_rates":  { "<rate_name>":  { "delta_pct": "float | null" } },
            "fast_gauges": { "<gauge_name>": { "delta_pct": "float | null" } }
          },
          "hot": [
            {
              "index":              "int",
              "duration_delta_pct": "float | null",
              "is_pivot_snapped_ref": "bool | null",
              "probes":      { "<probe_name>": { "..." } },
              "rates":       { "<rate_name>":  { "delta_pct": "float | null" } },
              "gauges":      { "<gauge_name>": { "delta_pct": "float | null" } },
              "fast_probes": { "<probe_name>": { "..." } },
              "fast_rates":  { "<rate_name>":  { "delta_pct": "float | null" } },
              "fast_gauges": { "<gauge_name>": { "delta_pct": "float | null" } }
            }
          ],
          "unaligned_hot": ["int"],
          "tail_status": "aligned | both_absent | target_absent | ref_absent",
          "tail": {
            "duration_delta_pct": "float | null",
            "probes":      { "<probe_name>": { "..." } },
            "rates":       { "<rate_name>":  { "delta_pct": "float | null" } },
            "gauges":      { "<gauge_name>": { "delta_pct": "float | null" } },
            "fast_probes": { "<probe_name>": { "..." } },
            "fast_rates":  { "<rate_name>":  { "delta_pct": "float | null" } },
            "fast_gauges": { "<gauge_name>": { "delta_pct": "float | null" } }
          }
        }
      },
      "appeared_probes":           ["string"],
      "disappeared_probes":        ["string"],
      "appeared_rates":            ["string"],
      "disappeared_rates":         ["string"],
      "appeared_gauges":           ["string"],
      "disappeared_gauges":        ["string"],
      "appeared_fast_probes":      ["string"],
      "disappeared_fast_probes":   ["string"],
      "appeared_fast_rates":       ["string"],
      "disappeared_fast_rates":    ["string"],
      "appeared_fast_gauges":      ["string"],
      "disappeared_fast_gauges":   ["string"]
    }
  }
}
```

> ⚠️ `comparisons.<comparison_type>` (le bloc entier) vaut `null` en mode session unique — voir « Invariants garantis ».

---

### Légende des types

| Placeholder         | Signification                                                                     |
| ------------------- | --------------------------------------------------------------------------------- |
| `session_id`        | Identifiant de session, format `YYYYMMDD_HHmmss`                                  |
| `datetime_iso8601`  | Horodatage ISO 8601 avec microsecondes et offset UTC local                        |
| `<probe_name>`      | Clé dynamique — nom d'une sonde (cf. `bench-probes.md`)                           |
| `<rate_name>`       | Clé dynamique — nom d'un compteur de taux                                         |
| `<gauge_name>`      | Clé dynamique — nom d'une jauge                                                   |
| `<comparison_type>` | Clé dynamique — `absolute` ou `relative`                                          |
| `float`             | Nombre décimal signé                                                              |
| `int`               | Entier ≥ 0                                                                        |
| `string`            | Chaîne de caractères                                                              |
| `["string"]`        | Tableau de chaînes (peut être vide `[]`)                                          |
| `T \| null`         | Champ pouvant valoir `null` — conditions détaillées section « Sémantique `null` » |

Si N == 2, `comparisons.relative` vaut `null`.
Si N == 1, `comparisons.absolute` et `comparisons.relative` valent `null`.

---

## Règles de calcul

### Convention de nommage des sondes

Les noms de sondes exposés dans le rapport JSON correspondent **exactement** aux clés présentes dans les sections `probes`, `rates` et `gauges` des fichiers JSONL sources, sans préfixe ni transformation.

| Catégorie       | Source JSONL        | Exemple de clé dans le rapport     |
| --------------- | ------------------- | ---------------------------------- |
| Probes (timers) | `row.probes.<name>` | `main_blur_ms`, `capture_frame_ms` |
| Rates           | `row.rates.<name>`  | `main_frames_total`                |
| Gauges          | `row.gauges.<name>` | `registry_confirmed`               |

**Rationale** : correspondance 1:1 stricte avec les clés JSONL d'entrée. Une sonde peut être tracée sans ambiguïté de son point d'émission (`bench.timer("main_blur_ms")`) jusqu'au rapport final, sans renommage intermédiaire par `bench_compare.py`.

### Convention de préfixage des canaux

| Canal source | Préfixage des noms de sondes     | Sections du rapport JSON                   | Champ de comptage               |
| ------------ | -------------------------------- | ------------------------------------------ | ------------------------------- |
| `agg`        | Non préfixé (`main_blur_ms`)     | `probes`, `rates`, `gauges`                | `count_agg`                     |
| `fast`       | Préfixé `fast_*` (`fast_ncc_ms`) | `fast_probes`, `fast_rates`, `fast_gauges` | `count_fast`                    |
| `frame`      | Hérité de la sonde émettrice     | `probes` ou `fast_probes` selon la sonde   | — (ne produit pas de `count_*`) |

**Règles induites** :

- Une sonde dont le nom commence par `fast_` est **toujours** rattachée au canal `fast`.
- Une sonde dont le nom ne commence pas par `fast_` est **toujours** rattachée au canal `agg`.
- Les listes `appeared_*` / `disappeared_*` suivent la même ventilation : `appeared_probes` liste les sondes agg apparues, `appeared_fast_probes` liste les sondes fast apparues. Aucun chevauchement possible.
- Le canal `frame` n'introduit pas de préfixe propre.

> ⚠️ **Conséquence pour l'ajout d'une nouvelle sonde** — le choix du préfixe (`fast_` ou non) à l'émission détermine **irréversiblement** le canal de destination.

### Probes (canaux `agg` et `fast`)

Chaque ligne JSONL du canal `agg` expose `{avg, min, max, count}` par sonde.
Chaque ligne JSONL du canal `fast` expose `{avg, min, max, count}` par sonde.
Chaque ligne JSONL du canal `frame` expose `{avg, max, min, count}` par sonde (cf. bench-jsonl-schema.md §6.1). Sur ce canal, chaque ligne agrège exactement 1 échantillon (1 ligne = 1 frame), donc min == max == avg par construction.

**Agrégats de base** :

| Champ produit              | Calcul                                                |
| -------------------------- | ----------------------------------------------------- |
| `avg`                      | Moyenne pondérée des `avg` par `count`                |
| `min`                      | Minimum des `min` sur toutes les lignes               |
| `max`                      | Maximum des `max` sur toutes les lignes               |
| `count_agg` / `count_fast` | Somme des `count` JSONL du canal source (agg ou fast) |

**Méthodes percentiles** :

- **Méthode `exact`** : collecte des valeurs des lignes du canal `frame` où `count == 1`.
- **Méthode `approx`** : collecte de `avg` de toutes les lignes du canal source.

**Seuil minimal** : percentile calculé uniquement si **`samples >= 20`**. Sinon → `null`.

**Cas particulier `fast_*`** : `p90_exact` / `p95_exact` / `p99_exact` toujours `null`.

### Rates

Moyenne arithmétique simple de toutes les valeurs `rates.<nom>` sur les lignes de la session.

### Gauges

Moyenne arithmétique simple de toutes les valeurs `gauges.<nom>` sur les lignes de la session.

### Durée session

`duration_s` = timestamp de la dernière ligne du canal `agg` − timestamp de la première ligne (horloge wall, canal `agg` uniquement).

`duration_mono_s` = `max(ts_mono)` − `min(ts_mono)` sur la timeline unifiée des trois canaux (`agg` + `frame` + `fast`), horloge monotone (`perf_counter`). Robuste aux décalages d'horloge système. Vaut `0.0` si aucune ligne ingérée.

### Analyse temporelle

Pour chaque canal (`agg`, `frame`, `fast`):

| Champ               | Définition                                                    | Valeur si `frames < 2` |
| ------------------- | ------------------------------------------------------------- | ---------------------- |
| `median_interval_s` | Médiane des intervalles consécutifs `ts[i+1] − ts[i]`         | `null`                 |
| `gaps_stat`         | Nombre d'intervalles dépassant **3× la médiane**              | `0`                    |
| `gaps_fixed`        | Nombre d'intervalles dépassant `2 × EXPECTED_PERIOD_S[canal]` | `null` si event-driven |

**Constantes :**

```python
EXPECTED_PERIOD_S = {
    "agg":   cfg.get("debug.bench.agg.interval_s",  1.0),
    "frame": None,   # event-driven
    "fast":  cfg.get("debug.bench.fast.interval_s", 1.0),
}
GAP_STAT_FACTOR  = 3.0  # figé en v1
GAP_FIXED_FACTOR = 2.0  # figé en v1
```

### Delta (%)

```text
delta_pct = ((target - reference) / reference) × 100
```

Valeur positive = target plus élevé que reference.
Valeur `null` si reference = 0, ou si `target` / `reference` est `null`.

---

### Deltas temporels (`deltas.temporal`)

Ventilé par canal (`agg`, `frame`, `fast`), chaque canal expose les 4 sous-clés :
`frames`, `median_interval_s`, `gaps_stat`, `gaps_fixed`.

---

## Bucketing adaptatif S4

### Vue d'ensemble

Le bloc `target.buckets` (et `reference.buckets` dans les comparaisons) découpe chaque
session en phases temporelles distinctes afin d'isoler le comportement en régime établi
des artefacts de démarrage.

| Phase    | Description                                                            |
| -------- | ---------------------------------------------------------------------- |
| `cold`   | Phase de démarrage — durée variable, synchro wait-for-all              |
| `hot[i]` | Phases de régime établi — durée nominale `hot_duration_s`, répétées    |
| `tail`   | Résidu final si durée restante < `hot_duration_s`, marqué `is_partial` |

### Configuration (`config.yaml`)

```yaml
debug:
  bench:
    compare:
      buckets:
        cold_target_s: 5.0 # cible théorique fin cold
        hot_duration_s: 10.0 # durée nominale hot_i
        max_cold_drift_s: 3.0 # garde-fou drift cold (warning si dépassé)
        boundary_guard_s: 0.5 # demi-largeur fenêtre recherche pivot hot_i
        min_gap_s: 0.1 # écart minimal pour qu'un instant soit pivot valide
        epsilon_s: 0.001 # ε de sécurité synchro wait-for-all
```

### Synchro fin de cold (wait-for-all)

La frontière `cold_end_real` est calculée comme suit :

```text
cold_end_real = max(next_agg_after_target, next_fast_after_target) + epsilon_s
             — si fast désactivé : next_agg_after_target + epsilon_s
```

- `cold_drift_s = cold_end_real - cold_end_target_s`
- Si `cold_drift_s > max_cold_drift_s` → `cold_drift_warning: true` + log WARNING
- Si `cold_end_real > t_max` → `cold_truncated: true`, pas de `hot`, pas de `tail`

### Frontières hot_i (snap pivot)

Pour chaque `hot_i`, la frontière théorique `T = t_cursor + hot_duration_s` est
ajustée par recherche d'un **pivot** dans la fenêtre `[T - boundary_guard_s, T + boundary_guard_s]` :

- Algorithme **D2 analytique** : intervalles vides entre événements agg/fast dans la
  fenêtre, recherche de l'instant le plus proche de T dans un intervalle ≥ `2 × min_gap_s`.
- Si pivot trouvé → `is_pivot_snapped: true`, frontière = instant pivot.
- Si non → `is_pivot_snapped: false` (silencieux), frontière = T stricte.

### Métadonnées `sync_metadata`

| Champ                | Type  | Description                                           |
| -------------------- | ----- | ----------------------------------------------------- |
| `cold_end_target_s`  | float | Cible théorique depuis config                         |
| `cold_end_real_s`    | float | Frontière réelle après wait-for-all                   |
| `cold_drift_s`       | float | Écart `cold_end_real - cold_end_target`               |
| `cold_drift_warning` | bool  | `true` si `cold_drift_s > max_cold_drift_s`           |
| `cold_truncated`     | bool  | `true` si session trop courte pour produire des `hot` |
| `fast_enabled`       | bool  | `false` si canal fast désactivé ou vide               |

> `cold_truncated: true` implique `hot: []` et `tail: null`.
> `buckets` vaut `null` si la session ne contient pas assez d'événements agg (< 2 lignes).

### Deltas buckets (`deltas.buckets`) — stratégie P1

Les deltas entre sessions sont calculés **par bucket aligné** (P1) :

- `cold` vs `cold` — toujours présent si les deux sessions ont un bloc `buckets` non null.
- `hot[i]` vs `hot[i]` — jusqu'à `min(N_hot_target, N_hot_ref)`.
- Buckets `hot` non alignés (index > min) listés dans `unaligned_hot`.
- `tail` comparé si les deux sessions ont un `tail` (`tail_status: aligned`).

**Valeurs de `tail_status`** :

| Valeur          | Signification                         |
| --------------- | ------------------------------------- |
| `aligned`       | Tail présent dans target et reference |
| `both_absent`   | Absent des deux côtés                 |
| `target_absent` | Absent de la cible uniquement         |
| `ref_absent`    | Absent de la référence uniquement     |

> Si `target.buckets` ou `reference.buckets` est `null`, `deltas.buckets` vaut `null`.

---

## Sémantique des valeurs `null`

| Contexte                                        | Signification                                                  |
| ----------------------------------------------- | -------------------------------------------------------------- |
| Sonde absente de la session                     | Branche de code non atteinte                                   |
| Percentile sous seuil `samples >= 20`           | Échantillon insuffisant pour calcul statistique fiable         |
| Delta impossible (référence = 0 ou valeur null) | `null`                                                         |
| Mode session unique (N==1)                      | `comparisons.absolute` et `comparisons.relative` valent `null` |
| `target.buckets` null                           | Session trop courte (< 2 événements agg) pour bucketing        |
| `deltas.buckets` null                           | Au moins un des deux blocs `buckets` est null                  |

`null` signifie **"donnée non disponible ou non calculable"**, jamais zéro implicite.

### Sondes conditionnelles notables

| Sonde                                       | Condition d'émission                              |
| ------------------------------------------- | ------------------------------------------------- |
| `mask_lost_latency_ms`                      | Uniquement si un mask passe en état LOST          |
| `mask_revive_latency_ms`                    | Uniquement si un mask est revitalisé              |
| `motion_staleness_slow_ms`                  | Uniquement si staleness dépasse le seuil          |
| `fast_stale_used`                           | Uniquement si fallback stale déclenché            |
| `selector_source_<name>`                    | Émise une fois — présente dans `frame` uniquement |
| `temporal_events.<canal>.median_interval_s` | Canal avec moins de 2 lignes ingérées             |
| `temporal_events.frame.gaps_fixed`          | Toujours `null` (canal event-driven)              |
| `deltas.temporal.*.delta_pct`               | Référence à 0 ou null                             |

---

## Limites v1

| Limite                                              | Statut                                             |
| --------------------------------------------------- | -------------------------------------------------- |
| Canal `frame` lu uniquement pour percentiles probes | Reste du contenu archivé, exploitable manuellement |
| Sélection interactive de session                    | Hors scope v1 — prévu v2                           |
| Génération automatique de `analyse.md`              | Hors scope — rédigé manuellement par le dev        |
| Comparaison N cibles simultanées                    | Hors scope — une cible par exécution               |
| Seuils de régression configurables                  | Hors scope                                         |
| Détection statistique (p-values)                    | Hors scope                                         |
| Seuil minimal d'échantillons percentiles            | Figé à `20` en v1 — non configurable               |
| Facteurs `GAP_STAT_FACTOR` / `GAP_FIXED_FACTOR`     | Figés à `3.0` / `2.0` en v1 — non configurables    |
| IQR                                                 | Prévu S4-bis — non implémenté en v1                |
| Skewness / Kurtosis par bucket                      | Prévu S5 — non implémenté en v1                    |

---

## Invariants garantis

- Une session sans fichier `agg` est ignorée avec avertissement.
- Une session sans fichier `frame` est traitée — tous les percentiles `*_exact` valent `null`.
- Une session sans fichier `fast` est traitée — sondes `fast_*` absentes = `null`.
- Si une seule session est disponible, rapport en mode single (`comparisons.absolute` et `comparisons.relative` valent `null`).
- Si aucune session n'est disponible, le script s'arrête avec un message explicite.
- Les fichiers JSONL sont déplacés **avant** l'écriture du rapport JSON.
- `target.buckets` vaut `null` si la session contient moins de 2 événements agg (bucketing impossible).
- `cold_truncated: true` est émis si `cold_end_real > t_max` (session trop courte).
- `cold_drift_warning: true` est émis si `cold_drift_s > max_cold_drift_s` — le bucketing continue sans interruption.
- Le champ `schema_version` identifie la version du schéma. Toute évolution non rétro-compatible incrémente ce champ.
- Le champ `generated_at` est un timestamp ISO 8601 avec fuseau horaire local (`datetime.now().astimezone().isoformat()`).
