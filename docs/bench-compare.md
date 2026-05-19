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
  `shutil`, `sys`)
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

Deux cas — et deux seulement — où des fichiers de `logs/results/` sont modifiés :

1. **Doublon de `session_id`** entre `logs/json/` et `logs/results/`
   `logs/json/` est prioritaire (cas attendu uniquement après renommage manuel).
   Un avertissement est émis. Le dossier `logs/results/<session_id>/` est
   **vidé puis remplacé** par les fichiers venant de `logs/json/`.

2. **Cible déjà présente dans `logs/results/`**
   Si la session la plus récente se trouve dans `logs/results/` (aucune session
   neuve dans `logs/json/`, ou cible exclusivement archivée), le rapport JSON
   `<target_session>.json` est régénéré dans le dossier existant.
   Tout rapport JSON préexistant du même nom est **écrasé** après validation
   de l'écriture du nouveau.

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
- En mode N==2 : référence absolue == avant-dernière. Pour éviter une comparaison redondante avec elle-même, `comparisons.relative` vaut `null`.

---

## Utilisation

```bash
python bench_compare.py
```

Exécution non interactive. Aucune option CLI en v1.

---

## Format du JSON de sortie

Fichier : `logs/results/<target_session>/<target_session>.json`

```json
{
  "format_version": "1.0",
  "generated_at": "2026-05-19T09:45:00.123456",
  "target_session": "20260519_091540",
  "target": {
    "duration_s": 18.3,
    "probes": {
      "main_blur_ms": {
        "avg": 6.91,
        "min": 0.98,
        "max": 28.4,
        "count": 161,
        "samples_exact": 158,
        "samples_approx": 161,
        "p90_exact": 12.3,
        "p95_exact": 18.7,
        "p99_exact": 24.1,
        "p90_approx": 12.5,
        "p95_approx": 18.9,
        "p99_approx": 24.3
      }
    },
    "rates": {
      "main_frames_total": 46.1
    },
    "gauges": {
      "registry_confirmed": 2.1
    }
  },
  "comparisons": {
    "absolute": {
      "reference_session": "20260517_205106",
      "reference": {
        "duration_s": 18.1,
        "probes": {
          "main_blur_ms": {
            "avg": 7.42,
            "min": 1.12,
            "max": 31.8,
            "count": 156,
            "samples_exact": 152,
            "samples_approx": 156,
            "p90_exact": 13.4,
            "p95_exact": 20.1,
            "p99_exact": 25.5,
            "p90_approx": 13.6,
            "p95_approx": 20.4,
            "p99_approx": 25.7
          }
        },
        "rates": { "main_frames_total": 44.2 },
        "gauges": { "registry_confirmed": 1.8 }
      },
      "deltas": {
        "probes": {
          "main_blur_ms": {
            "avg_delta_pct": -6.9,
            "min_delta_pct": -12.5,
            "max_delta_pct": -10.7,
            "p90_exact_delta_pct": -8.2,
            "p95_exact_delta_pct": -7.0,
            "p99_exact_delta_pct": -5.5,
            "p90_approx_delta_pct": -8.1,
            "p95_approx_delta_pct": -7.4,
            "p99_approx_delta_pct": -5.4
          }
        },
        "rates": { "main_frames_total": { "delta_pct": 4.3 } },
        "gauges": { "registry_confirmed": { "delta_pct": 16.7 } }
      },
      "appeared_in_target": [],
      "disappeared_in_target": []
    },
    "relative": {
      "reference_session": "20260518_103022",
      "reference": { "duration_s": 18.4, "probes": {}, "rates": {}, "gauges": {} },
      "deltas": { "probes": {}, "rates": {}, "gauges": {} },
      "appeared_in_target": [],
      "disappeared_in_target": []
    }
  }
}
```

Si N == 2, `comparisons.relative` vaut `null`.
Si N == 1, `comparisons.absolute` et `comparisons.relative` valent `null`.

---

## Règles de calcul

### Probes (canaux `agg` et `fast`)

Chaque ligne JSONL du canal `agg` expose `{avg, min, max, count}` par sonde.
Chaque ligne JSONL du canal `frame` expose `{avg, count}` par sonde (1 ligne / frame).
Chaque ligne JSONL du canal `fast` expose `{avg, min, max, count}` par sonde.

**Agrégats de base** (depuis canal `agg`, ou `fast` pour sondes `fast_*`) :

| Champ produit | Calcul                                  |
| ------------- | --------------------------------------- |
| `avg`         | Moyenne pondérée des `avg` par `count`  |
| `min`         | Minimum des `min` sur toutes les lignes |
| `max`         | Maximum des `max` sur toutes les lignes |
| `count`       | Somme des `count`                       |

**Percentiles** (`p90`, `p95`, `p99`) — double émission systématique :

- **Méthode `exact`** : collecte des valeurs des lignes du canal `frame`
  (sondes hors `fast_*`) où `count == 1`. Chaque ligne retenue contribue
  1 échantillon = sa valeur `avg`. Percentile calculé via
  `statistics.quantiles(data, n=100, method='inclusive')`.
- **Méthode `approx`** : collecte de `avg` de toutes les lignes du canal
  `frame` (sondes hors `fast_*`) ou du canal `fast` (sondes `fast_*`).
  Chaque ligne contribue 1 échantillon. Percentile calculé de la même
  manière.

**Champs de comptage associés** :

- `samples_exact` : nombre d'échantillons utilisés par la méthode exact.
  Toujours `0` pour les sondes `fast_*` (canal `frame` ne les expose pas).
- `samples_approx` : nombre total d'échantillons utilisés par la méthode
  approx (= nombre de lignes du canal source contenant la sonde).

**Seuil minimal** : un percentile (`exact` ou `approx`) n'est calculé que
si **`samples >= 20`** pour la méthode considérée. Sinon → `null`.

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

## Sondes conditionnelles notables

Ces sondes peuvent être absentes (`null`) sans que ce soit une anomalie :

| Sonde                          | Condition d'émission                              |
| ------------------------------ | ------------------------------------------------- |
| `capture_drop`                 | Uniquement si `source.grab()` retourne `None`     |
| `detect_slow_candidates_total` | Présente mais peut être 0 sans détection          |
| `mask_lost_latency_ms`         | Uniquement si un mask passe en état LOST          |
| `mask_revive_latency_ms`       | Uniquement si un mask est revitalisé              |
| `motion_staleness_slow_ms`     | Uniquement si staleness dépasse le seuil          |
| `fast_stale_used`              | Uniquement si fallback stale déclenché            |
| `selector_source_<name>`       | Émise une fois — présente dans `frame` uniquement |

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

Les fichiers sources présents dans `logs/json/` sont **déplacés** (pas copiés)
vers `logs/results/<session_id>/` à la fin du traitement.
`logs/json/` est vidé des sessions traitées après chaque exécution.

Si la cible provient déjà de `logs/results/`, aucun déplacement n'est effectué pour
elle ; seul le rapport JSON est (re)généré dans son dossier existant.

---

## Invariants garantis

- Une session sans fichier `agg` est ignorée avec avertissement.
- Une session sans fichier `frame` est traitée — tous les percentiles `*_exact` valent `null`.
- Une session sans fichier `fast` est traitée — sondes `fast_*` absentes = `null`.
- Si une seule session est disponible, le script produit un rapport en mode single
  (`comparisons.absolute` et `comparisons.relative` valent `null`). Si aucune
  session n'est disponible, le script s'arrête avec un message explicite.
- Les fichiers sont déplacés **après** écriture réussie du JSON de sortie.
  En cas d'erreur, aucun fichier source n'est déplacé.
- `logs/results/` est en lecture seule, **sauf** dans les deux cas listés à la section
  « Cas de modification de `logs/results/` » (doublon de `session_id`, ou cible déjà
  archivée dont le rapport JSON est régénéré).
- Le champ `format_version` au sommet du JSON identifie la version du schéma de sortie.
  Toute évolution non rétro-compatible du format incrémente ce champ.
