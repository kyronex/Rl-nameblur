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
    "rates": {
      "<rate_name>": "float"
    },
    "gauges": {
      "<gauge_name>": "float"
    },
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
    "fast_rates": {
      "<rate_name>": "float"
    },
    "fast_gauges": {
      "<gauge_name>": "float"
    }
  },
  "comparisons": {
    "<comparison_type>": {
      "reference_session": "session_id",
      "reference": {
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
        "rates": {
          "<rate_name>": "float"
        },
        "gauges": {
          "<gauge_name>": "float"
        },
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
        "fast_rates": {
          "<rate_name>": "float"
        },
        "fast_gauges": {
          "<gauge_name>": "float"
        }
      },
      "deltas": {
        "duration_s": { "delta_pct": "float | null" },
        "duration_mono_s": { "delta_pct": "float | null" },
        "temporal": {
          "agg": {
            "frames": { "delta_pct": "float | null" },
            "median_interval_s": { "delta_pct": "float | null" },
            "gaps_stat": { "delta_pct": "float | null" },
            "gaps_fixed": { "delta_pct": "float | null" }
          },
          "frame": {
            "frames": { "delta_pct": "float | null" },
            "median_interval_s": { "delta_pct": "float | null" },
            "gaps_stat": { "delta_pct": "float | null" },
            "gaps_fixed": { "delta_pct": "float | null" }
          },
          "fast": {
            "frames": { "delta_pct": "float | null" },
            "median_interval_s": { "delta_pct": "float | null" },
            "gaps_stat": { "delta_pct": "float | null" },
            "gaps_fixed": { "delta_pct": "float | null" }
          }
        },
        "probes": {
          "<probe_name>": {
            "avg_delta_pct": "float | null",
            "min_delta_pct": "float | null",
            "max_delta_pct": "float | null",
            "p90_exact_delta_pct": "float | null",
            "p95_exact_delta_pct": "float | null",
            "p99_exact_delta_pct": "float | null",
            "p90_approx_delta_pct": "float | null",
            "p95_approx_delta_pct": "float | null",
            "p99_approx_delta_pct": "float | null"
          }
        },
        "rates": {
          "<rate_name>": {
            "delta_pct": "float | null"
          }
        },
        "gauges": {
          "<gauge_name>": {
            "delta_pct": "float | null"
          }
        },
        "fast_probes": {
          "<probe_name>": {
            "avg_delta_pct": "float | null",
            "min_delta_pct": "float | null",
            "max_delta_pct": "float | null",
            "p90_exact_delta_pct": "float | null",
            "p95_exact_delta_pct": "float | null",
            "p99_exact_delta_pct": "float | null",
            "p90_approx_delta_pct": "float | null",
            "p95_approx_delta_pct": "float | null",
            "p99_approx_delta_pct": "float | null"
          }
        },
        "fast_rates": {
          "<rate_name>": {
            "delta_pct": "float | null"
          }
        },
        "fast_gauges": {
          "<gauge_name>": {
            "delta_pct": "float | null"
          }
        }
      },
      "appeared_probes": ["string"],
      "disappeared_probes": ["string"],
      "appeared_rates": ["string"],
      "disappeared_rates": ["string"],
      "appeared_gauges": ["string"],
      "disappeared_gauges": ["string"],
      "appeared_fast_probes": ["string"],
      "disappeared_fast_probes": ["string"],
      "appeared_fast_rates": ["string"],
      "disappeared_fast_rates": ["string"],
      "appeared_fast_gauges": ["string"],
      "disappeared_fast_gauges": ["string"]
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

Le rapport JSON sépare strictement les sondes selon leur canal d'origine. Cette séparation s'exprime par une **convention de préfixage asymétrique** : le canal `fast` est préfixé `fast_*`, le canal `agg` ne l'est pas.

| Canal source | Préfixage des noms de sondes                                        | Sections du rapport JSON                   | Champ de comptage                           |
| ------------ | ------------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------- |
| `agg`        | Non préfixé (`main_blur_ms`)                                        | `probes`, `rates`, `gauges`                | `count_agg`                                 |
| `fast`       | Préfixé `fast_*` (`fast_ncc_ms`)                                    | `fast_probes`, `fast_rates`, `fast_gauges` | `count_fast`                                |
| `frame`      | Hérité de la sonde émettrice (préfixé ou non selon son canal cible) | `probes` ou `fast_probes` selon la sonde   | — (canal frame ne produit pas de `count_*`) |

**Règles induites** :

- Une sonde dont le nom commence par `fast_` est **toujours** rattachée au canal `fast` (production via `bench.timer("fast_*")`, lecture côté `fast_probes` du rapport).
- Une sonde dont le nom ne commence pas par `fast_` est **toujours** rattachée au canal `agg` (lecture côté `probes` du rapport).
- Les listes `appeared_*` / `disappeared_*` suivent la même ventilation : `appeared_probes` liste les sondes agg apparues, `appeared_fast_probes` liste les sondes fast apparues. Aucun chevauchement possible.
- Le canal `frame` n'introduit pas de préfixe propre : il alimente uniquement les percentiles (`*_exact` / `*_approx`) des sondes déjà classées par leur préfixe d'origine.

**Rationale** : la convention de préfixage rend la classification d'une sonde **déterministe par lecture du nom seul**, sans avoir à consulter le code émetteur ou le schéma JSONL. Elle reflète la séparation physique des fichiers (`bench_agg.jsonl` vs `bench_fast.jsonl`) au niveau du rapport final.

> ⚠️ **Conséquence pour l'ajout d'une nouvelle sonde** — le choix du préfixe (`fast_` ou non) à l'émission (`bench.timer("...")`) détermine **irréversiblement** le canal de destination et la section du rapport où elle apparaîtra. Renommer une sonde existante en ajoutant/retirant `fast_` la fait migrer entre canaux et apparaîtra comme `disappeared_*` côté ancien canal et `appeared_*` côté nouveau canal lors de la prochaine comparaison.

### Probes (canaux `agg` et `fast`)

Chaque ligne JSONL du canal `agg` expose `{avg, min, max, count}` par sonde.
Chaque ligne JSONL du canal `fast` expose `{avg, min, max, count}` par sonde.
Chaque ligne JSONL du canal `frame` expose `{avg, max, min, count}` par sonde (cf. bench-jsonl-schema.md §6.1). Sur ce canal, chaque ligne agrège exactement 1 échantillon (1 ligne = 1 frame), donc min == max == avg par construction. `bench_compare.py` ne lit que `avg` et `count` pour calculer les percentiles exacts (cf. règle d'éligibilité samples_exact plus bas).

**Agrégats de base** :

Le champ de comptage est nommé différemment selon le canal source, pour refléter la séparation stricte entre sondes agg et sondes `fast_*` :

- Sondes hors `fast_*` → agrégées depuis le canal `agg` → champ produit `count_agg`
- Sondes `fast_*` → agrégées depuis le canal `fast` → champ produit `count_fast`

| Champ produit              | Calcul                                                |
| -------------------------- | ----------------------------------------------------- |
| `avg`                      | Moyenne pondérée des `avg` par `count`                |
| `min`                      | Minimum des `min` sur toutes les lignes               |
| `max`                      | Maximum des `max` sur toutes les lignes               |
| `count_agg` / `count_fast` | Somme des `count` JSONL du canal source (agg ou fast) |

#### Sémantique de count_agg / count_fast, samples_exact, samples_approx

Ces champs **ne sont pas redondants** : ils mesurent des grandeurs distinctes et répondent à des questions différentes. `count_agg` et `count_fast` sont mutuellement exclusifs : une sonde produit l'un **ou** l'autre selon son canal source, jamais les deux.

| Champ            | Question à laquelle il répond                                                          | Source (canal JSONL)                                | Règle d'éligibilité d'une ligne       |
| ---------------- | -------------------------------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------- |
| `count_agg`      | Combien d'échantillons bruts alimentent `avg` / `min` / `max` (sondes hors `fast_*`) ? | `agg`                                               | Toute ligne contenant la sonde        |
| `count_fast`     | Combien d'échantillons bruts alimentent `avg` / `min` / `max` (sondes `fast_*`) ?      | `fast`                                              | Toute ligne contenant la sonde        |
| `samples_exact`  | Sur combien d'échantillons reposent les percentiles `*_exact` ?                        | `frame`                                             | Ligne avec `probes.<name>.count == 1` |
| `samples_approx` | Sur combien de lignes agrégées reposent les percentiles `*_approx` ?                   | `frame` (hors `fast_*`) ou `fast` (sondes `fast_*`) | Toute ligne contenant la sonde        |

**Conséquences directes** :

- `count_agg` / `count_fast` et `samples_approx` ne sont pas comparables : les champs de comptage agg/fast somment les `count` JSONL bruts (= total d'échantillons), tandis que `samples_approx` compte les lignes du canal source. Les unités diffèrent (échantillons vs lignes) ; aucune égalité n'est attendue, même par construction.
- `samples_exact` est **indépendant** des autres : il dépend exclusivement de la présence du canal `frame` et du filtre `count == 1`. Une sonde présente uniquement dans `agg` ou `fast` aura `samples_exact = 0` et tous ses `p*_exact = null`.
- Aucune relation arithmétique n'est garantie entre ces champs — voir `bench-jsonl-schema.md` §6.1 pour la sémantique amont.

> ⚠️ **Note terminologique** — les champs `count_agg` (sondes hors `fast_*`) et `count_fast` (sondes `fast_*`) sont des totaux d'échantillons agrégés issus de leur canal source respectif, pas des nombres de lignes JSONL ni de frames. Ils ne sont pas directement comparables à `samples_approx` (qui compte les lignes du canal source). Pour le nombre de frames consommées par la sonde, voir `samples_exact` (canal `frame` uniquement, jamais peuplé pour les sondes `fast_*`).

- **Méthode `exact`** : collecte des valeurs des lignes du canal `frame` (sondes hors `fast_*`) où `count == 1`.
  Chaque ligne retenue contribue 1 échantillon = sa valeur `avg`. Percentile calculé via `statistics.quantiles(data, n=100, method='inclusive')`.
- **Méthode `approx`** : collecte de `avg` de toutes les lignes du canal `frame` (sondes hors `fast_*`) ou du canal `fast` (sondes `fast_*`).
  Chaque ligne contribue 1 échantillon. Percentile calculé de la même manière.

**Champs de comptage associés** :

- `samples_exact` : nombre d'échantillons utilisés par la méthode exact. Toujours `0` pour les sondes `fast_*` (canal `frame` ne les expose pas).
- `samples_approx` : nombre de lignes du canal source contenant la sonde, utilisées comme échantillons par la méthode `approx` (chaque ligne contribue sa valeur `avg` comme 1 point de donnée, indépendamment du `count` JSONL sous-jacent).

**Seuil minimal** : un percentile (`exact` ou `approx`) n'est calculé que si **`samples >= 20`** pour la méthode considérée. Sinon → `null`.

**Cas particulier `fast_*`** :

- `p90_exact` / `p95_exact` / `p99_exact` toujours `null`.
- `p90_approx` / `p95_approx` / `p99_approx` calculés depuis `bench_fast.jsonl`.

### Rates

Moyenne arithmétique simple de toutes les valeurs `rates.<nom>` sur les lignes de la session.

### Gauges

Moyenne arithmétique simple de toutes les valeurs `gauges.<nom>` sur les lignes de la session.

### Durée session

`duration_s` = timestamp de la dernière ligne du canal `agg` − timestamp de la première ligne (horloge wall, canal `agg` uniquement).

`duration_mono_s` = `max(ts_mono)` − `min(ts_mono)` sur la timeline unifiée des trois canaux (`agg` + `frame` + `fast`), horloge monotone (`perf_counter`). Robuste aux décalages d'horloge système. Vaut `0.0` si aucune ligne ingérée.

### Analyse temporelle

Pour chaque canal (`agg`, `frame`, `fast`), trois métriques sont calculées à partir des timestamps monotones :

| Champ               | Définition                                                                                    | Valeur si `frames < 2` |
| ------------------- | --------------------------------------------------------------------------------------------- | ---------------------- |
| `median_interval_s` | Médiane statistique des intervalles consécutifs `ts[i+1] − ts[i]`                             | `null`                 |
| `gaps_stat`         | Nombre d'intervalles dépassant **2× la médiane** (seuil statistique relatif)                  | `0`                    |
| `gaps_fixed`        | Nombre d'intervalles dépassant `2 × EXPECTED_PERIOD_S[canal]` (seuil relatif à `config.yaml`) | `null` si event-driven |

**Constantes :**

```python
EXPECTED_PERIOD_S = {
    "agg":   cfg.get("debug.bench.agg.interval_s",  1.0),  # configurable
    "frame": None,                                         # event-driven
    "fast":  cfg.get("debug.bench.fast.interval_s", 1.0),  # configurable
}
GAP_STAT_FACTOR  = 3.0  # figé en v1
GAP_FIXED_FACTOR = 2.0  # figé en v1
```

> Les cadences attendues `agg` et `fast` sont lues depuis `config.yaml` (clés `debug.bench.agg.interval_s` et `debug.bench.fast.interval_s`, défaut `1.0`).
> Le canal `frame` est **event-driven** : pas de cadence attendue, `gaps_fixed = null`.
> Les facteurs `GAP_STAT_FACTOR` et `GAP_FIXED_FACTOR` sont figés en v1.
> `frames.<canal>` = nombre de lignes ingérées sur le canal.

### Delta (%)

```text
delta_pct = ((target - reference) / reference) × 100
```

Valeur positive = target plus élevé que reference.

Règles de nullité :

- Valeur `null` si reference = 0 (division impossible).
- Valeur `null` si `target` ou `reference` est `null` (donnée manquante d'un côté).
- Valeur `null` pour les percentiles si la méthode (`exact` ou `approx`)
  est sous le seuil minimal côté target **ou** côté reference.

---

### Deltas temporels (`deltas.temporal`)

Le bloc `deltas.temporal` est ventilé par **canal source** (`agg`, `frame`, `fast`), reflétant la structure de `target.temporal_events` et `reference.temporal_events`. Chaque canal expose les mêmes 4 sous-clés :

| Sous-clé            | Source comparée                             | Calcul du delta                                                                      |
| ------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------ |
| `frames`            | `temporal_events.<canal>.frames`            | `(target − reference) / reference × 100`                                             |
| `median_interval_s` | `temporal_events.<canal>.median_interval_s` | Idem                                                                                 |
| `gaps_stat`         | `temporal_events.<canal>.gaps_stat`         | Idem                                                                                 |
| `gaps_fixed`        | `temporal_events.<canal>.gaps_fixed`        | Idem — toujours `null` pour le canal `frame` (event-driven, cf. §Analyse temporelle) |

**Règle de nullité spécifique** : `gaps_fixed.delta_pct` vaut systématiquement `null` pour le canal `frame`, car la grandeur source est `null` côté target ET reference (canal event-driven, pas de cadence attendue). Pour les canaux `agg` et `fast`, le delta suit la règle générale (`null` si référence vaut `0` ou `null`).

**Structure JSON** :

```json
"deltas": {
  "temporal": {
    "agg":   { "frames": {...}, "median_interval_s": {...}, "gaps_stat": {...}, "gaps_fixed": {...} },
    "frame": { "frames": {...}, "median_interval_s": {...}, "gaps_stat": {...}, "gaps_fixed": {...} },
    "fast":  { "frames": {...}, "median_interval_s": {...}, "gaps_stat": {...}, "gaps_fixed": {...} }
  }
}
```

## Sémantique des valeurs `null`

| Contexte                                        | Signification                                                                |
| ----------------------------------------------- | ---------------------------------------------------------------------------- |
| Sonde absente de la session                     | Branche de code non atteinte                                                 |
| Percentile sous seuil `samples >= 20`           | Échantillon insuffisant pour calcul statistique fiable                       |
| Delta impossible (référence = 0 ou valeur null) | `null`                                                                       |
| Mode session unique (N==1)                      | `comparisons.absolute` et `comparisons.relative` valent `null` (cible seule) |

`null` signifie **"donnée non disponible ou non calculable"**, jamais zéro implicite.
Une sonde absente indique que la branche de code correspondante
n'a pas été atteinte pendant la session (voir sondes conditionnelles ci-dessous).
Un percentile `null` avec `samples_*` renseigné indique que l'échantillon
est sous le seuil statistique minimal.

---

### Sondes conditionnelles notables

Certaines sondes ne sont émises que dans des conditions spécifiques. Leur absence dans un rapport est **normale** et ne constitue pas une régression.

| Sonde                                       | Condition d'émission                                                                                  |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `mask_lost_latency_ms`                      | Uniquement si un mask passe en état LOST                                                              |
| `mask_revive_latency_ms`                    | Uniquement si un mask est revitalisé                                                                  |
| `motion_staleness_slow_ms`                  | Uniquement si staleness dépasse le seuil                                                              |
| `fast_stale_used`                           | Uniquement si fallback stale déclenché                                                                |
| `selector_source_<name>`                    | Émise une fois — présente dans `frame` uniquement                                                     |
| `temporal_events.<canal>.median_interval_s` | Canal avec moins de 2 lignes ingérées                                                                 |
| `temporal_events.frame.gaps_fixed`          | Canal `frame` event-driven (`EXPECTED_PERIOD_S["frame"] = None`) — toujours `null`.                   |
|                                             | Les canaux `agg` et `fast` ont une cadence attendue configurée et calculent normalement `gaps_fixed`. |
| `deltas.temporal.*.delta_pct`               | Référence à `0` ou `null` (division impossible)                                                       |

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
| Facteurs de `GAP_STAT_FACTOR`                       | Figés à `3.0` en v1 — non configurables            |
| Facteurs de `GAP_FIXED_FACTOR`                      | Figés à `2.0` en v1 — non configurables            |

---

## Fichiers produits par exécution

| Fichier                      | Emplacement                      | Description                |
| ---------------------------- | -------------------------------- | -------------------------- |
| `<target_session>.json`      | `logs/results/<target_session>/` | Rapport comparatif complet |
| `bench_*_<session_id>.jsonl` | `logs/results/<session_id>/`     | Sources archivées          |

Les fichiers sources présents dans `logs/json/` sont **déplacés** (pas copiés) vers `logs/results/<session_id>/` **avant** l'écriture du rapport JSON. `logs/json/` est vidé des sessions traitées après chaque exécution.

Si la cible provient déjà de `logs/results/`, aucun déplacement n'est effectué pour elle ; seul le rapport JSON est (re)généré dans son dossier existant.

---

## Invariants garantis

- Une session sans fichier `agg` est ignorée avec avertissement.
- Une session sans fichier `frame` est traitée — tous les percentiles `*_exact` valent `null`.
- Une session sans fichier `fast` est traitée — sondes `fast_*` absentes = `null`.
- Si une seule session est disponible, le script produit un rapport en mode single
  (`comparisons.absolute` et `comparisons.relative` valent `null`). Si aucune session n'est disponible, le script s'arrête avec un message explicite.
- Les fichiers JSONL sont déplacés **avant** l'écriture du rapport JSON.
  En cas d'échec de l'écriture du rapport (`OSError`), les JSONL sont déjà archivés dans `logs/results/<session_id>/` mais le rapport `.json` est absent.
  Une nouvelle exécution traitera alors la session comme « cible déjà dans results » (cas 2 ci-dessus) et régénérera le rapport sans re-déplacement.
- En cas d'échec du déplacement lui-même (`OSError` levée par `_move_session_to_results`),le rapport n'est pas écrit et l'état filesystem peut être partiellement modifié (déplacement non atomique fichier par fichier).
- `logs/results/` est en lecture seule, **sauf** dans les deux cas listés à la section « Cas de modification de `logs/results/` » (doublon de `session_id`, ou cible déjà archivée dont le rapport JSON est régénéré).
- Le champ `schema_version` au sommet du JSON identifie la version du schéma de sortie. Toute évolution non rétro-compatible du format incrémente ce champ.
- Le champ `generated_at` au sommet du JSON est un timestamp ISO 8601 **avec fuseau horaire local** (offset UTC inclus, ex. `2026-05-19T09:15:40.123456+02:00`). Produit par `datetime.now().astimezone().isoformat()`.
- Une session sans fichier `frame` est traitée — tous les percentiles `*_exact` valent `null`, `frames.frame` vaut `0`, et `temporal_events.frame.{median_interval_s, gaps_stat, gaps_fixed}` valent respectivement `null`, `null`, `null` (timeline vide → `_compute_temporal_events` retourne `None` pour les trois champs, cf. branche `len(timeline) < 2`).
- Le même principe s'applique aux canaux `agg` et `fast` absents : `frames.<canal>` vaut `0` et `temporal_events.<canal>.{median_interval_s, gaps_stat, gaps_fixed}` valent tous `null` (timeline vide → branche `len(timeline) < 2` de `_compute_temporal_events`). Note : pour le canal `frame`, `gaps_fixed` vaut `null` **dans tous les cas** (event-driven), indépendamment de la présence du fichier ; pour `agg` et `fast`, `gaps_fixed` est calculable dès que la timeline contient ≥ 2 événements.
- Les constantes `EXPECTED_PERIOD_S`, `GAP_STAT_FACTOR` et `GAP_FIXED_FACTOR` sont figées en v1 (cf. section « Analyse temporelle »). Toute modification fait évoluer la sémantique de `gaps_stat` / `gaps_fixed`.
