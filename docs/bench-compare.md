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
    }
  },
  "comparisons": {
    "<comparison_type>": {
      "reference_session": "session_id",
      "reference": {
        "duration_s": "float",
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
        }
      },
      "deltas": {
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
        }
      },
      "appeared_probes": ["string"],
      "disappeared_probes": ["string"],
      "appeared_rates": ["string"],
      "disappeared_rates": ["string"],
      "appeared_gauges": ["string"],
      "disappeared_gauges": ["string"]
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

### Probes (canaux `agg` et `fast`)

Chaque ligne JSONL du canal `agg` expose `{avg, min, max, count}` par sonde.
Chaque ligne JSONL du canal `fast` expose `{avg, min, max, count}` par sonde.
Chaque ligne JSONL du canal `frame` expose `{avg, max, min, count}` par sonde (cf. bench-jsonl-schema.md §6.1). Sur ce canal, chaque ligne agrège exactement 1 échantillon (1 ligne = 1 frame), donc min == max == avg par construction. `bench_compare.py` ne lit que `avg` et `count` pour calculer les percentiles exacts (cf. règle d'éligibilité samples_exact plus bas).

**Agrégats de base** (depuis canal `agg`, ou `fast` pour sondes `fast_*`) :

| Champ produit | Calcul                                               |
| ------------- | ---------------------------------------------------- |
| `avg`         | Moyenne pondérée des `avg` par `count`               |
| `min`         | Minimum des `min` sur toutes les lignes              |
| `max`         | Maximum des `max` sur toutes les lignes              |
| `count_agg`   | Somme des `count` JSONL (échantillons agrégés bruts) |

#### Sémantique de count_agg, samples_exact, samples_approx

Ces trois champs **ne sont pas redondants** : ils mesurent des grandeurs distinctes et répondent à des questions différentes.

| Champ            | Question à laquelle il répond                                        | Source (canal JSONL) | Règle d'éligibilité d'une ligne               |
| ---------------- | -------------------------------------------------------------------- | -------------------- | --------------------------------------------- |
| `count_agg`      | Cb samples bruts ont alimenté `avg`/`min`/`max`(canaux agg + fast) ? | `agg` + `fast`       | Toute ligne contenant la sonde                |
| `samples_exact`  | Sur cb samples reposent les percentiles `*_exact` ?                  | `frame`              | Ligne avec `probes.<name>.count == 1`         |
| `samples_approx` | Sur cb de lignes agrégées reposent les percentiles `*_approx` ?      | `agg` + `fast`       | Toute ligne contenant la sonde (idem `count`) |

**Conséquences directes** :

- `count_agg` et `samples_approx` ne sont pas comparables : `count_agg` agrège les canaux `agg` + `fast` (somme des count JSONL = total d'échantillons bruts), tandis que `samples_approx` compte les lignes du canal `frame` (sondes hors `fast_*`) ou `fast` (sondes `fast_*`). Les domaines de canaux diffèrent ; aucune égalité n'est attendue, même par construction.
- `samples_exact` est **indépendant** des deux autres : il dépend exclusivement de la présence du canal `frame` et du filtre `count == 1`. Une sonde présente uniquement dans `agg`/`fast` aura `samples_exact = 0` et tous ses `p*_exact = null`.
- Aucune relation arithmétique n'est garantie entre les trois — voir `bench-jsonl-schema.md` §6.1 pour la sémantique amont.

> ⚠️ **Note terminologique** — le champ `count_agg` est un total d'échantillons agrégés issus des canaux `agg` + `fast`, pas un nombre de lignes JSONL ni un nombre de frames. Il n'est pas directement comparable à `samples_approx` (qui agrège les canaux frame / fast selon la sonde). Pour le nombre de frames consommées par la sonde, voir samples_exact (canal frame uniquement).

- **Méthode `exact`** : collecte des valeurs des lignes du canal `frame` (sondes hors `fast_*`) où `count == 1`.
  Chaque ligne retenue contribue 1 échantillon = sa valeur `avg`. Percentile calculé via `statistics.quantiles(data, n=100, method='inclusive')`.
- **Méthode `approx`** : collecte de `avg` de toutes les lignes du canal `frame` (sondes hors `fast_*`) ou du canal `fast` (sondes `fast_*`).
  Chaque ligne contribue 1 échantillon. Percentile calculé de la même manière.

**Champs de comptage associés** :

- `samples_exact` : nombre d'échantillons utilisés par la méthode exact.Toujours `0` pour les sondes `fast_*` (canal `frame` ne les expose pas).
- `samples_approx` : nombre total d'échantillons utilisés par la méthode approx (= nombre de lignes du canal source contenant la sonde).

**Seuil minimal** : un percentile (`exact` ou `approx`) n'est calculé que si **`samples >= 20`** pour la méthode considérée. Sinon → `null`.

**Cas particulier `fast_*`** :

- `p90_exact` / `p95_exact` / `p99_exact` toujours `null`.
- `p90_approx` / `p95_approx` / `p99_approx` calculés depuis `bench_fast.jsonl`.

### Rates

Moyenne arithmétique simple de toutes les valeurs `rates.<nom>` sur les lignes de la session.

### Gauges

Moyenne arithmétique simple de toutes les valeurs `gauges.<nom>` sur les lignes de la session.

### Durée session

`ts` de la dernière ligne `agg` − `ts` de la première ligne `agg`.

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

| Sonde                      | Condition d'émission                              |
| -------------------------- | ------------------------------------------------- |
| `mask_lost_latency_ms`     | Uniquement si un mask passe en état LOST          |
| `mask_revive_latency_ms`   | Uniquement si un mask est revitalisé              |
| `motion_staleness_slow_ms` | Uniquement si staleness dépasse le seuil          |
| `fast_stale_used`          | Uniquement si fallback stale déclenché            |
| `selector_source_<name>`   | Émise une fois — présente dans `frame` uniquement |

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
