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

## Sommaire

- [Prérequis](#prérequis)
- [Structure des dossiers](#structure-des-dossiers)
  - [Avant exécution](#avant-exécution)
  - [Après exécution](#après-exécution)
- [Sources de sessions](#sources-de-sessions)
  - [Cas de modification de `logs/results/`](#cas-de-modification-de-logsresults)
- [Logique de sélection](#logique-de-sélection)
  - [Rôles attribués](#rôles-attribués)
- [Utilisation](#utilisation)
- [Format du rapport JSON](#format-du-rapport-json)
- [Forme de distribution](#forme-de-distribution)
- [Détection d'anomalies](#détection-danomalies)
- [Configuration (`config.yaml`)](#configuration-configyaml)
- [Invariants garantis](#invariants-garantis)
- [Limites v1](#limites-v1)

---

## Prérequis

- Python 3.10+
- Dépendances :
  - **stdlib** : `json`, `pathlib`, `datetime`, `statistics`, `shutil`, `sys`, `logging`
  - **tiers** : `numpy` (régression linéaire pour la détection de drift), `scipy` (skewness / kurtosis excess), `PyYAML` (lecture `config.yaml`)
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

   > **Note d'implémentation (v1)** : le vidage est non récursif (`Path.iterdir()` + `Path.unlink()`). Le dossier n'est pas censé contenir de sous-dossier ; si tel était le cas, `unlink()` lèverait `IsADirectoryError` (Linux) / `PermissionError` (Windows) **non catchée**, interrompant le déplacement. Aucun sous-dossier n'est créé par le pipeline actuel — cette limite est documentée mais non bloquante en exploitation normale.

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

- **Cible** : session avec le `session_id` le plus récent (tri lexicographique sur le format `YYYYMMDD_HHMMSS`, équivalent au strftime Python `%Y%m%d_%H%M%S`).
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

## Format du rapport JSON

Fichier : `logs/results/<target_session>/<target_session>.json`

> Schéma complet des fichiers JSONL produits : [`bench-compare-jsonl-schema.md`](bench-compare-jsonl-schema.md).

---

### Légende des types

| Placeholder         | Signification                                                                     |
| ------------------- | --------------------------------------------------------------------------------- |
| `session_id`        | Identifiant de session, format `YYYYMMDD_HHMMSS` (strftime `%Y%m%d_%H%M%S`)       |
| `datetime_iso8601`  | Horodatage ISO 8601 avec microsecondes et offset timezone local                   |
| `<probe_name>`      | Clé dynamique — nom d'une sonde (cf. `bench-probes.md`)                           |
| `<rate_name>`       | Clé dynamique — nom d'un compteur de taux                                         |
| `<gauge_name>`      | Clé dynamique — nom d'une jauge                                                   |
| `<comparison_type>` | Clé dynamique — `absolute` ou `relative`                                          |
| `float`             | Nombre décimal signé                                                              |
| `int`               | Entier ≥ 0                                                                        |
| `string`            | Chaîne de caractères                                                              |
| `["string"]`        | Tableau de chaînes (peut être vide `[]`)                                          |
| `["int"]`           | Tableau d'entiers (peut être vide `[]`)                                           |
| `bool`              | Booléen `true` / `false`                                                          |
| `bool \| null`      | Booléen pouvant valoir `null`                                                     |
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

**Quartiles & IQR** :

Pour chaque sonde, en complément des percentiles, sont calculés :

| Champ produit              | Définition                                           |
| -------------------------- | ---------------------------------------------------- |
| `q1_exact` / `q1_approx`   | Premier quartile (25e percentile)                    |
| `q3_exact` / `q3_approx`   | Troisième quartile (75e percentile)                  |
| `iqr_exact` / `iqr_approx` | Écart interquartile `q3 - q1` — mesure de dispersion |

Calcul via `statistics.quantiles(data, n=4, method="inclusive")`. Même seuil minimal que les percentiles : si `samples < 20` → les trois champs (`q1`, `q3`, `iqr`) valent `null`.

**Cas particulier `fast_*`** : `q1_exact` / `q3_exact` / `iqr_exact` toujours `null` (pas d'échantillons exacts disponibles sur le canal fast).

### Forme de distribution

Pour chaque sonde, en complément des quartiles, sont calculés les indicateurs de forme de distribution. Ils permettent de détecter des distributions asymétriques (queue de latence longue) ou à queues lourdes (pics récurrents), invisibles dans les agrégats `avg` / `min` / `max`.

| Champ produit            | Définition                                                                                             |
| ------------------------ | ------------------------------------------------------------------------------------------------------ |
| `skewness_exact`         | Asymétrie Fisher-Pearson calculée sur échantillons exacts (`scipy.stats.skew`, `bias=False`).          |
| `skewness_approx`        | Même calcul sur échantillons approchés (moyennes agrégées des lignes `agg`).                           |
| `kurtosis_excess_exact`  | Kurtosis excess calculé sur échantillons exacts (`scipy.stats.kurtosis`, `fisher=True`, `bias=False`). |
| `kurtosis_excess_approx` | Même calcul sur échantillons approchés (moyennes agrégées des lignes `agg`).                           |

**Interprétation** :

- `skewness_*` — valeur de référence : 0 (distribution symétrique). Positif = queue droite longue (latences hautes rares).
- `kurtosis_excess_*` — valeur de référence : 0 (loi normale). Positif = queues plus lourdes que la normale (pics extrêmes fréquents).

**Seuils minimaux distincts** :

| Champ               | Seuil minimal    | Constante              | Sous seuil        |
| ------------------- | ---------------- | ---------------------- | ----------------- |
| `skewness_*`        | `samples >= 50`  | `SKEWNESS_MIN_SAMPLES` | `null` silencieux |
| `kurtosis_excess_*` | `samples >= 100` | `KURTOSIS_MIN_SAMPLES` | `null` silencieux |

**Constantes (configurables via `config.yaml`)** :

```yaml
debug:
  bench:
    shape:
      skewness_min_samples: 50 # Seuil min échantillons pour calculer skewness (D2)
      kurtosis_min_samples: 100 # Seuil min échantillons pour calculer kurtosis excess (D3)
```

Aucun flag interprétatif n'est ajouté — les valeurs brutes sont exposées telles quelles pour laisser l'analyse à l'opérateur.

**Garde défensive — variance nulle** : si l'écart-type des échantillons est nul (toutes les valeurs identiques), `skewness_*` et `kurtosis_excess_*` valent `null` — scipy produirait une division par zéro.

**Cas particulier `fast_*`** : `skewness_exact` et `kurtosis_excess_exact` valent toujours `null` — le canal fast ne dispose pas d'échantillons exacts (cf. convention de préfixage des canaux). Les variantes `_approx` sont calculées normalement si le seuil est atteint.

**Périmètre d'application** : calculé au niveau session (`target.probes`) et dans chaque bucket (`cold` / `hot[i]` / `tail`). Pas de restriction de périmètre contrairement aux anomalies.

### Détection d'anomalies

Pour chaque sonde, dans chaque bucket (`cold` / `hot[i]` / `tail`), sont calculés des indicateurs de **spikes** (valeurs aberrantes ponctuelles) et de **drift** (tendance linéaire intra-bucket). Ces indicateurs complètent la forme de distribution (S5a) en localisant les anomalies dans le temps.

| Champ produit         | Définition                                                                                  |
| --------------------- | ------------------------------------------------------------------------------------------- |
| `spike_count`         | Nombre de valeurs telles que `\|v − median\| > SPIKE_MAD_FACTOR × MAD`                      |
| `spike_max_value`     | Valeur brute du plus gros spike détecté (`null` si `spike_count == 0`)                      |
| `spike_max_deviation` | Déviation du plus gros spike, exprimée en multiples de MAD (`null` si `spike_count == 0`)   |
| `drift_slope`         | Pente OLS (`numpy.polyfit`, deg=1) sur la série filtrée des spikes, en unité/sec            |
| `drift_intercept`     | Ordonnée à l'origine de la régression (valeur brute, jamais exposée en delta)               |
| `drift_r2`            | Coefficient de détermination de l'ajustement linéaire (`∈ [0, 1]`, qualité de l'ajustement) |

**Méthode spikes — MAD robuste** :

```text
median = median(samples)
MAD    = median(|samples − median|)
spike  ⇔  |value − median| > SPIKE_MAD_FACTOR × MAD
```

La MAD (Median Absolute Deviation) est préférée à l'écart-type pour sa robustesse aux valeurs extrêmes : un spike isolé ne gonfle pas le seuil de détection, contrairement à `mean ± k·stdev`.

> ⚠️ **Limite connue** : lorsque la MAD est très faible (distribution quasi-dégénérée mais non strictement constante), `spike_max_deviation` peut atteindre des valeurs très élevées (plusieurs milliers d'unités MAD). Ce comportement est algorithmiquement correct mais signale une distribution où l'analyse de spikes perd en pertinence — à interpréter conjointement avec `iqr_exact` et `stdev` implicite.

**Méthode drift — OLS sur série filtrée** :

1. Retrait des points identifiés comme spike (préfiltrage par masquage booléen).
2. Re-vérification du seuil minimal sur la série filtrée.
3. Si série filtrée ≥ `DRIFT_MIN_SAMPLES` → `numpy.polyfit(t_mono, values, deg=1)`.
4. `drift_r2` calculé via résiduels (`1 − SS_res / SS_tot`).

Le domaine temporel `t_mono` est exprimé en **secondes relatives** au premier échantillon du bucket, depuis le champ `mono` du canal frame.

> ℹ️ **Interprétation `drift_r2`** : un `drift_slope` non-nul accompagné d'un `drift_r2` très faible (typiquement < 0.05) indique une pente statistiquement non significative — les deux valeurs sont exposées brutes sans flag, l'interprétation revient à l'opérateur (cf. invariant « valeurs brutes seules, aucun flag »).

**Source des échantillons** : canal `frame` avec `count == 1` uniquement (échantillons exacts).

**Seuils minimaux distincts** :

| Champ     | Seuil minimal           | Constante           | Sous seuil        |
| --------- | ----------------------- | ------------------- | ----------------- |
| `spike_*` | `samples >= 20`         | `SPIKE_MIN_SAMPLES` | `null` silencieux |
| `drift_*` | `samples_filtrés >= 30` | `DRIFT_MIN_SAMPLES` | `null` silencieux |

**Garde défensive — MAD nul** :

Si `MAD == 0` (toutes les valeurs identiques à la médiane), `spike_count` vaut `0`, `spike_max_value` et `spike_max_deviation` valent `null` — aucun spike détectable sans dispersion mesurable. Le calcul drift est tenté sur la série brute non filtrée si le seuil `DRIFT_MIN_SAMPLES` est atteint.

**Cas particulier `fast_*`** : tous les champs S5b (`spike_*`, `drift_*`) valent **toujours `null`** sur les sondes du canal fast — le canal fast ne dispose pas d'échantillons exacts (cf. convention de préfixage des canaux).

**Périmètre d'application** :

| Niveau                              | Calculé ?       | Rationale                                                                                |
| ----------------------------------- | --------------- | ---------------------------------------------------------------------------------------- |
| `target.probes` (session globale)   | ❌ Non          | Anomalies utiles only en phase stable,session globale mélange warm-up et régime nominal. |
| `target.buckets.cold.probes`        | ✅ Oui          | Phase de warm-up, détecte les pics initiaux et les dérives de stabilisation.             |
| `target.buckets.hot[i].probes`      | ✅ Oui          | Régime nominal, détecte les anomalies en charge.                                         |
| `target.buckets.tail.probes`        | ✅ Oui          | Fin de session, détecte les dérives de fin (fuite mémoire, dégradation progressive).     |
| `target.fast_probes` (tous niveaux) | ❌ Forcé `null` | Pas d'échantillons exacts disponibles sur le canal fast.                                 |

**Périmètre deltas (mode comparaison)** :

| Bucket   | Deltas S5b exposés ? | Champs exposés                                                                          |
| -------- | -------------------- | --------------------------------------------------------------------------------------- |
| `cold`   | ✅ Oui               | `spike_count_delta`, `spike_max_deviation_delta`, `drift_slope_delta`, `drift_r2_delta` |
| `hot[i]` | ✅ Oui               | Idem cold                                                                               |
| `tail`   | ❌ Non               | Valeurs brutes disponibles côté `target`/`reference`, comparaison manuelle              |

**Champs jamais exposés en delta** : `spike_max_value` (valeur brute non comparable inter-sessions) et `drift_intercept` (dépendant du repère temporel relatif au bucket). Ces deux champs apparaissent uniquement côté `target` / `reference`, jamais sous forme de delta calculé.

**Constantes (configurables via `config.yaml`)** :

```yaml
debug:
  bench:
    compare:
      anomalies:
        spike_min_samples: 20 # SPIKE_MIN_SAMPLES
        spike_mad_factor: 3.5 # SPIKE_MAD_FACTOR
        drift_min_samples: 30 # DRIFT_MIN_SAMPLES
```

### Rates

Moyenne arithmétique simple de toutes les valeurs `rates.<nom>` sur les lignes de la session.

### Gauges

Moyenne arithmétique simple de toutes les valeurs `gauges.<nom>` sur les lignes de la session.

### Durée session

Deux champs exposés au niveau `target` :

| Champ             | Définition                                                                                        | Source                    |
| ----------------- | ------------------------------------------------------------------------------------------------- | ------------------------- |
| `duration_s`      | `ts_wall_last − ts_wall_first` sur le canal `agg` uniquement (horloge wall)                       | Canal `agg`, champ `ts`   |
| `duration_mono_s` | `max(ts_mono) − min(ts_mono)` sur la timeline unifiée des trois canaux (`agg` + `frame` + `fast`) | Tous canaux, champ `mono` |

**Rationale du double champ** :

- `duration_s` (wall clock) : valeur lisible humainement, utile pour corréler avec des événements externes (logs système, captures). Sensible aux décalages d'horloge (NTP, hibernation).
- `duration_mono_s` (monotonic, via `time.perf_counter`) : **valeur de référence pour tous les calculs internes** (buckets, drift, intervalles). Robuste aux décalages d'horloge système, jamais en recul.

**Cas limite** :

| Condition                              | `duration_s`    | `duration_mono_s` |
| -------------------------------------- | --------------- | ----------------- |
| Aucune ligne ingérée (toutes sources)  | `0.0`           | `0.0`             |
| Lignes uniquement sur `frame` / `fast` | `0.0`           | Valeur calculée   |
| Lignes uniquement sur `agg`            | Valeur calculée | Valeur calculée   |

> ⚠️ Une session avec `duration_s == 0.0` mais `duration_mono_s > 0` indique qu'aucun flush `agg` n'a eu lieu — typiquement une session très courte (< intervalle agg). Les blocs `target.probes`, `target.rates`, `target.gauges` seront alors vides.

### Analyse temporelle

Pour chaque canal (`agg`, `frame`, `fast`), trois indicateurs caractérisent la régularité temporelle du flux ingéré. Ces indicateurs sont exposés sous **`target.temporal_events.<canal>`** (le nombre de frames brutes par canal étant exposé séparément sous `target.frames.<canal>`).

| Champ               | Définition                                                                   | Valeur si `frames < 2`       |
| ------------------- | ---------------------------------------------------------------------------- | ---------------------------- |
| `median_interval_s` | Médiane des intervalles consécutifs `ts_mono[i+1] − ts_mono[i]`              | `null`                       |
| `gaps_stat`         | Nombre d'intervalles dépassant `GAP_STAT_FACTOR × median_interval_s`         | `0`                          |
| `gaps_fixed`        | Nombre d'intervalles dépassant `GAP_FIXED_FACTOR × EXPECTED_PERIOD_S[canal]` | `null` si canal event-driven |

**Rationale des deux familles de gaps** :

- `gaps_stat` : détection **relative** — repère les gaps anormaux par rapport au rythme effectivement observé sur la session. Adapté aux canaux event-driven ou à fréquence variable.
- `gaps_fixed` : détection **absolue** — repère les gaps par rapport à la fréquence théorique attendue. Adapté aux canaux périodiques (`agg`, `fast`). Vaut `null` constant sur `frame` (event-driven, pas de période cible).

**Constantes** :

```python
EXPECTED_PERIOD_S = {
    "agg":   cfg.get("debug.bench.agg.interval_s",  1.0),  # période flush agrégat
    "frame": None,                                          # event-driven (par frame)
    "fast":  cfg.get("debug.bench.fast.interval_s", 1.0),  # période flush canal fast
}
GAP_STAT_FACTOR  = 3.0  # figé en v1 — gap statistique si > 3× médiane observée
GAP_FIXED_FACTOR = 2.0  # figé en v1 — gap absolu si > 2× période théorique
```

**Exposition dans le rapport** :

- `target.frames.<canal>` — compteur brut de lignes ingérées par canal (clé top-level distincte).
- `target.temporal_events.<canal>.{median_interval_s, gaps_stat, gaps_fixed}` — indicateurs de régularité.

### Delta (%)

Mode comparaison uniquement (option `--reference`). Pour chaque indicateur scalaire des blocs `target.probes`, `target.rates`, `target.gauges`, la doc expose un **delta relatif** :

```text
delta_pct = ((target - reference) / reference) × 100
```

**Convention de signe** : valeur positive ⇔ `target` plus élevé que `reference`.
**Cas `null`** :

| Condition                                      | Résultat        |
| ---------------------------------------------- | --------------- |
| `reference == 0`                               | `null`          |
| `target` ou `reference` vaut `null`            | `null`          |
| Les deux valeurs présentes et `reference != 0` | Valeur calculée |

**Champs concernés (delta relatif `_delta_pct`)** :

- Bloc `deltas.probes` : `avg`, `min`, `max`, percentiles (`p90`/`p95`/`p99` × `_exact`/`_approx`), quartiles (`q1`/`q3`/`iqr` × `_exact`/`_approx`).
- Bloc `deltas.rates` : chaque rate.
- Bloc `deltas.gauges` : chaque gauge.
- Bloc `deltas.buckets.<phase>` : `duration_s`, plus tous les champs probes ci-dessus.

### Exception sondes — Deltas absolus

Les indicateurs de **forme de distribution** (S5a) et **d'anomalies** (S5b) utilisent un **delta absolu** (suffixe `_delta`, sans `_pct`) plutôt qu'un pourcentage :

```text
delta = target − reference
```

**Rationale** : ces indicateurs ne sont pas des grandeurs proportionnelles. Un skewness passant de `0.1` à `0.3` représente un changement qualitatif (distribution plus asymétrique) — exprimer ce changement en `+200 %` serait trompeur. De même, un `spike_count` passant de 0 à 5 ne peut pas être exprimé en pourcentage (`reference == 0`).

| Sonde                                              | Type delta       |
| -------------------------------------------------- | ---------------- |
| `skewness_exact` / `skewness_approx`               | `_delta` absolu  |
| `kurtosis_excess_exact` / `kurtosis_excess_approx` | `_delta` absolu  |
| `spike_count`                                      | `_delta` absolu  |
| `spike_max_deviation`                              | `_delta` absolu  |
| `drift_slope`                                      | `_delta` absolu  |
| `drift_r2`                                         | `_delta` absolu  |
| `spike_max_value`                                  | **Pas de delta** |
| `drift_intercept`                                  | **Pas de delta** |

---

**Champs sans delta — rationale** :

- `spike_max_value` : valeur brute exprimée dans l'unité de la sonde (ms, count, etc.). Comparer deux valeurs maximales isolées entre sessions n'a pas de sens analytique (chaque spike est un événement ponctuel). Les valeurs restent disponibles côté `target.buckets.<phase>.probes.<nom>` et `reference.buckets.<phase>.probes.<nom>` pour analyse manuelle.
- `drift_intercept` : ordonnée à l'origine de la régression linéaire, exprimée dans le repère temporel **relatif au bucket** (`t = 0` correspond au début du bucket). Comparer deux intercepts entre sessions n'est pas interprétable car le repère temporel diffère pour chaque session.

**Cas `null`** : si l'une des deux valeurs source est `null`, le delta vaut `null`.

---

### Périmètre delta

| Niveau                                    | Deltas (skew/kurt)     | Deltas (spike/drift)                     |
| ----------------------------------------- | ---------------------- | ---------------------------------------- |
| `deltas.probes.<nom>` (session)           | ✅ Oui                 | ❌ Non (= non calculé au niveau session) |
| `deltas.buckets.cold.probes.<nom>`        | ✅ Oui                 | ✅ Oui                                   |
| `deltas.buckets.hot[i].probes.<nom>`      | ✅ Oui                 | ✅ Oui                                   |
| `deltas.buckets.tail.probes.<nom>`        | ❌ Non                 | ❌ Non                                   |
| `deltas.fast_probes.<nom>` (tous niveaux) | ❌ Non (forcés `null`) | ❌ Non (forcés `null`)                   |

**Cas `tail`** : aucun delta de forme ou d'anomalie n'est exposé. Les valeurs brutes restent disponibles dans `reference.buckets.tail` / `target.buckets.tail` pour analyse manuelle. Rationale : la `tail` représente un échantillon non synchronisé entre sessions (durée variable post-`hot`), une comparaison directe serait peu fiable.

> ℹ️ **Cohérence avec (forme de distribution)** : sur le bucket `tail`, les **valeurs brutes** `skewness_*` et `kurtosis_excess_*` sont calculées et exposées côté `target.buckets.tail.probes` (cf. section « Forme de distribution (S5a) », périmètre d'application). Seuls leurs **deltas** sont supprimés du bloc `deltas.buckets.tail`, pour la raison de non-synchronisation évoquée ci-dessus.

---

## Deltas temporels (`deltas.temporal`)

Ventilation par canal (`agg`, `frame`, `fast`). Chaque canal expose les 4 sous-clés issues de l'analyse temporelle :

| Champ                         | Type delta       | Source brute                     | Cas `null`                                                 |
| ----------------------------- | ---------------- | -------------------------------- | ---------------------------------------------------------- |
| `frames_delta_pct`            | Relatif (`_pct`) | `target.frames.<canal>`          | `reference.frames == 0`                                    |
| `median_interval_s_delta_pct` | Relatif (`_pct`) | `target.temporal_events.<canal>` | `reference.median_interval_s == null` ou `0`               |
| `gaps_stat_delta_pct`         | Relatif (`_pct`) | `target.temporal_events.<canal>` | `reference.gaps_stat == 0`                                 |
| `gaps_fixed_delta_pct`        | Relatif (`_pct`) | `target.temporal_events.<canal>` | `reference.gaps_fixed == null` (canal event-driven) ou `0` |

**Cas particulier `frame` event-driven** : `gaps_fixed_delta_pct` vaut `null` constant (la valeur brute est `null` des deux côtés par construction — cf. section « Analyse temporelle »).

---

## Bucketing adaptatif S4

### Vue d'ensemble

Le bloc `target.buckets` (et `reference.buckets` dans les comparaisons) découpe chaque session en phases temporelles distinctes afin d'isoler le comportement en régime établi des artefacts de démarrage.

| Phase    | Description                                                                  |
| -------- | ---------------------------------------------------------------------------- |
| `cold`   | Phase de démarrage — durée variable, synchro wait-for-all                    |
| `hot[i]` | Phases de régime établi — durée nominale `hot_duration_s`, répétées          |
| `tail`   | Résidu final si durée restante < `hot_duration_s`, marqué `is_partial: true` |

**Périmètre statistique par bucket** : chaque bucket expose son propre bloc `probes` (avec percentiles, quartiles, S5a forme, S5b anomalies), `fast_probes`, et `duration_s`. Rates et gauges ne sont **pas** ventilés par bucket (cf. section « Rates / Gauges »).

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

La frontière `cold_end_real` garantit qu'**aucune ligne ingérée sur les canaux périodiques** (`agg` et `fast` si actif) n'est coupée en deux entre `cold` et le premier `hot`. Elle est calculée comme suit :

```text
si fast actif :
    cold_end_real = max(next_agg_after_target, next_fast_after_target) + epsilon_s
sinon :
    cold_end_real = next_agg_after_target + epsilon_s
```

où `next_X_after_target` désigne le premier `mono` du canal X strictement postérieur à `cold_target_s`.

**Indicateurs dérivés** :

- `cold_drift_s = cold_end_real − cold_end_target_s` (toujours ≥ 0)
- Si `cold_drift_s > max_cold_drift_s` → `cold_drift_warning: true` + log WARNING
- Si `cold_end_real > t_max` → `cold_truncated: true`, **pas de `hot`, pas de `tail`** (cf. cas limite ci-dessous)

**Écrêtage cas dégénéré** : si la session est si courte que `cold_end_real` calculé dépasserait `t_max`, l'algorithme retourne `cold_end_real = t_max` et active `cold_truncated`. Aucune exception levée.

### Frontières hot_i (snap pivot)

Pour chaque `hot_i`, la frontière théorique `T = t_cursor + hot_duration_s` est ajustée par recherche d'un **pivot** dans la fenêtre `[T − boundary_guard_s, T + boundary_guard_s]`.

**Algorithme D2 analytique** :

1. Lister les événements `agg` + `fast` (timestamps `mono`) tombant dans la fenêtre.
2. Identifier les intervalles vides (gaps) entre événements consécutifs dans la fenêtre.
3. Sélectionner l'intervalle dont la largeur ≥ `2 × min_gap_s` et dont le centre est le plus proche de `T`.
4. Le pivot est l'instant le plus proche de `T` à l'intérieur de cet intervalle.

**Résultats possibles** :

| Résultat             | Champ `is_pivot_snapped` | Frontière retenue         |
| -------------------- | ------------------------ | ------------------------- |
| Pivot trouvé         | `true`                   | Instant pivot             |
| Aucun pivot éligible | `false` (silencieux)     | `T` stricte (non snappée) |

### Métadonnées `sync_metadata`

Bloc exposé sous `target.buckets.sync_metadata`

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

Les deltas entre sessions sont calculés **par bucket aligné par index** :

| Alignement                                           | Comportement                                                          |
| ---------------------------------------------------- | --------------------------------------------------------------------- |
| `cold` vs `cold`                                     | Toujours présent si les deux sessions ont un bloc `buckets` non null  |
| `hot[i]` vs `hot[i]` pour `i < min(N_target, N_ref)` | Comparaison directe                                                   |
| `hot[i]` non aligné (i ≥ min)                        | Listé dans `unaligned_hot` avec son origine (`target` / `reference`)  |
| `tail` vs `tail`                                     | Comparé uniquement si présent des deux côtés (`tail_status: aligned`) |

**Valeurs de `tail_status`** :

| Valeur          | Signification                         |
| --------------- | ------------------------------------- |
| `aligned`       | Tail présent dans target et reference |
| `both_absent`   | Absent des deux côtés                 |
| `target_absent` | Absent de la cible uniquement         |
| `ref_absent`    | Absent de la référence uniquement     |

**Cas dégénéré** : si `target.buckets` **ou** `reference.buckets` vaut `null`, alors `deltas.buckets` vaut `null` (aucun alignement possible).

> ℹ️ **Rappel périmètre deltas par phase** : les deltas (skew/kurt) sont exposés sur `cold` + `hot[i]`, les deltas (spike/drift) sur `cold` + `hot[i]` uniquement. Le bloc `tail` n'expose **aucun delta de forme ou d'anomalie** (cf. section « Périmètre delta »).

---

## Sémantique des valeurs `null`

Dans toute la sortie JSON, `null` signifie **« donnée non disponible ou non calculable »** — jamais zéro implicite.

| Contexte                                              | Signification                                                            |
| ----------------------------------------------------- | ------------------------------------------------------------------------ |
| Sonde absente de la session                           | Branche de code non atteinte pendant l'exécution                         |
| Percentile / quartile sous seuil (< 20)               | Échantillon insuffisant pour calcul statistique fiable                   |
| `skewness_*` sous seuil `SKEWNESS_MIN_SAMPLES`        | Échantillon insuffisant pour forme de distribution (S5a)                 |
| `kurtosis_excess_*` sous seuil `KURTOSIS_MIN_SAMPLES` | Échantillon insuffisant pour forme de distribution (S5a)                 |
| `spike_*` sous seuil `SPIKE_MIN_SAMPLES`              | Échantillon insuffisant pour détection MAD (S5b)                         |
| `drift_*` sous seuil `DRIFT_MIN_SAMPLES`              | Série filtrée trop courte ou MAD nul / quasi-nul (S5b)                   |
| Variance nulle (`stdev == 0`)                         | `skewness_*` / `kurtosis_excess_*` → `null` (garde défensive)            |
| MAD nul (`MAD == 0`)                                  | `spike_*` → `null` (distribution dégénérée, garde défensive)             |
| Delta impossible (référence = 0 ou valeur null)       | `null`                                                                   |
| Mode session unique (`N == 1`)                        | `comparisons.absolute` et `comparisons.relative` valent `null`           |
| `target.buckets` null                                 | Session trop courte (< 2 événements `agg`) pour bucketing                |
| `deltas.buckets` null                                 | Au moins un des deux blocs `buckets` est null                            |
| `cold_truncated: true`                                | `hot: []` et `tail: null` — session trop courte pour phase régime établi |

### Sondes conditionnelles notables

| Sonde                                              | Condition d'émission                                           |
| -------------------------------------------------- | -------------------------------------------------------------- |
| `mask_lost_latency_ms`                             | Uniquement si un mask passe en état LOST                       |
| `mask_revive_latency_ms`                           | Uniquement si un mask est revitalisé                           |
| `motion_staleness_slow_ms`                         | Uniquement si staleness dépasse le seuil                       |
| `fast_stale_used`                                  | Uniquement si fallback stale déclenché                         |
| `selector_source_<name>`                           | Émise une fois — présente dans `frame` uniquement              |
| `temporal_events.<canal>.median_interval_s`        | `null` si canal avec moins de 2 lignes ingérées                |
| `temporal_events.frame.gaps_fixed`                 | `null` constant (canal event-driven, pas de période théorique) |
| `deltas.temporal.*.delta_pct`                      | `null` si référence à 0 ou null                                |
| `skewness_*` / `kurtosis_excess_*` (session)       | Calculés sur `target.probes` si seuil atteint                  |
| `skewness_*` / `kurtosis_excess_*` (bucket)        | Calculés sur cold / hot[i] / tail si seuil atteint             |
| `skewness_*` / `kurtosis_excess_*` (`fast_probes`) | `null` constant à tous niveaux (cohérence E10)                 |
| `spike_*` / `drift_*` (session)                    | **Non calculés** — hors périmètre S5b (buckets uniquement)     |
| `spike_*` (bucket)                                 | `null` si bucket < `SPIKE_MIN_SAMPLES` échantillons exacts     |
| `drift_*` (bucket)                                 | `null` si série filtrée < `DRIFT_MIN_SAMPLES` ou MAD nul       |
| `spike_max_value` / `spike_max_deviation`          | `null` si `spike_count == 0` (pas `0.0`)                       |
| `spike_*` / `drift_*` (`fast_probes`)              | `null` constant (pas d'échantillons `exact` sur canal `fast`)  |

---

## Limites v1

| Limite                                                   | Statut                                                        |
| -------------------------------------------------------- | ------------------------------------------------------------- |
| Canal `frame` lu uniquement pour percentiles probes      | Reste du contenu archivé, exploitable manuellement            |
| Sélection interactive de session                         | Hors scope v1 — prévu v2                                      |
| Génération automatique de `analyse.md`                   | Hors scope — rédigé manuellement                              |
| Comparaison N cibles simultanées                         | Hors scope — une cible par exécution                          |
| Seuils de régression configurables                       | Hors scope                                                    |
| Détection statistique (p-values)                         | Hors scope                                                    |
| Seuil minimal d'échantillons percentiles                 | `20` en v1                                                    |
| Facteurs `GAP_STAT_FACTOR` / `GAP_FIXED_FACTOR`          | `3.0` / `2.0` en v1                                           |
| Seuils (`SKEWNESS_MIN_SAMPLES` / `KURTOSIS_MIN_SAMPLES`) | `50` / `100` en v1                                            |
| Rates / gauges ventilés par bucket                       | Hors scope v1 — moyenne session uniquement (choix volontaire) |
| IQR (Q1 / Q3 / IQR par sonde)                            | ✅ Implémenté                                                 |
| Skewness / Kurtosis (forme de distribution)              | ✅ Implémenté (deltas absolus)                                |
| Détection spikes & drift par bucket                      | ✅ Implémenté (MAD + OLS, deltas absolus)                     |
| Corrélations inter-sondes & budget frame                 | Prévu S6 — non implémenté en v1                               |

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
- `tail_status` est **toujours présent** dans la sortie ; le bloc `tail` est absent quand `tail_status != "aligned"`.
- Le champ `schema_version` identifie la version du schéma. Toute évolution non rétro-compatible incrémente ce champ.
- Le champ `generated_at` est un timestamp ISO 8601 avec fuseau horaire local (`datetime.now().astimezone().isoformat()`).
