# Schéma JSON bench-compare — Contrat normatif L0.5

> 🔒 **Statut** : figé.
> Toute modification du schéma (ajout/suppression/renommage de champ, changement de type, restructuration) requiert :
>
> 1. Incrément de `schema_version`.
> 2. Ouverture d'un nouveau ticket dédié.
> 3. Mise à jour du présent document.

---

## Sommaire

1. [Portée](#1-portée)
2. [Conventions communes](#2-conventions-communes)
   - 2.1 [Méta-champs obligatoires (racine)](#21-méta-champs-obligatoires-racine)
   - 2.2 [Valeurs nulles](#22-valeurs-nulles)
   - 2.3 [Unités et formule delta](#23-unités-et-formule-delta)
   - 2.4 [Seuils statistiques minimaux](#24-seuils-statistiques-minimaux)
   - 2.5 [Source des échantillons (`_exact` / `_approx`)](#25-source-des-échantillons-_exact--_approx)
3. [Structure racine](#3-structure-racine)
4. [Bloc `target`](#4-bloc-target)
   - 4.1 [Structure](#41-structure)
   - 4.2 [`temporal_events`](#42-temporal_events)
5. [Contrat des blocs `probe_stats`](#5-contrat-des-blocs-probe_stats)
   - 5.1 [`probe_stats_global`](#51-probe_stats_global--sondes-globales-probes-fast_probes-hors-buckets)
   - 5.2 [`probe_stats_bucket`](#52-probe_stats_bucket--sondes-dans-un-bucket-cold-hoti-tail)
   - 5.3 [`probe_stats_fast_bucket`](#53-probe_stats_fast_bucket--fast_probes-dans-un-bucket)
6. [Bloc `buckets`](#6-bloc-buckets)
   - 6.1 [`sync_metadata`](#61-sync_metadata)
   - 6.2 [Bucket `cold`](#62-bucket-cold)
   - 6.3 [Bucket `hot` (tableau)](#63-bucket-hot-tableau)
   - 6.4 [Bucket `tail`](#64-bucket-tail)
7. [Bloc `comparisons`](#7-bloc-comparisons)
   - 7.1 [Structure](#71-structure)
   - 7.2 [Bloc `deltas`](#72-bloc-deltas)
   - 7.3 [Bloc `buckets` (deltas par bucket)](#73-bloc-buckets-deltas-par-bucket)
8. [Matrice des sections par zone](#8-matrice-des-sections-par-zone)
9. [Règles d'évolution](#9-règles-dévolution)
10. [Référence d'implémentation](#10-référence-dimplémentation)

## 1. Portée

Ce document décrit le format du fichier JSON produit par `bench/bench_compare.py`.

Un seul fichier par exécution de comparaison :

| Fichier                           | Cadence de production             |
| --------------------------------- | --------------------------------- |
| `bench_compare_{session_id}.json` | 1 fichier / appel `bench_compare` |

Le fichier est un objet JSON autonome et auto-descriptif. Il contient :

- Les statistiques complètes de la **session cible** (`target`).
- Pour chaque type de comparaison, les statistiques de la **session de référence** (`reference`) et les **deltas** calculés.

---

## 2. Conventions communes

### 2.1 Méta-champs obligatoires (racine)

| Champ            | Type   | Description                                                                                |
| ---------------- | ------ | ------------------------------------------------------------------------------------------ |
| `schema_version` | int    | Version du contrat (valeur courante : `1`).                                                |
| `generated_at`   | string | Timestamp ISO 8601 avec offset timezone local (`datetime.now().astimezone().isoformat()`). |
| `target_session` | string | Identifiant de la session analysée (session cible).                                        |

### 2.2 Valeurs nulles

Un champ marqué `float | null` ou `int | null` vaut `null` (JSON) lorsque le calcul
est impossible (échantillons insuffisants, division par zéro, données absentes).

### 2.3 Unités et formule delta

Le schéma ne normalise **pas** les unités des sondes. La sémantique relève du producteur
de la sonde. Les deltas en pourcentage (`_delta_pct`) sont calculés comme :

```text
delta_pct = (target − reference) / abs(reference) × 100
```

`null` si `reference == 0` ou si l'une des valeurs source est `null`.

### 2.4 Seuils statistiques minimaux

En-dessous des seuils ci-dessous, les champs concernés valent `null` (silencieux, sans warning).

| Statistique                 | Seuil minimum       | Constante              | Source de configuration                           |
| --------------------------- | ------------------- | ---------------------- | ------------------------------------------------- |
| `p90_*` / `p95_*` / `p99_*` | 20 échantillons     | _(figé v1)_            | —                                                 |
| `q1_*` / `q3_*` / `iqr_*`   | 20 échantillons     | _(figé v1)_            | —                                                 |
| `skewness_*`                | 50 échantillons     | `SKEWNESS_MIN_SAMPLES` | `debug.bench.compare.shape.skewness_min_samples`  |
| `kurtosis_excess_*`         | 100 échantillons    | `KURTOSIS_MIN_SAMPLES` | `debug.bench.compare.shape.kurtosis_min_samples`  |
| `spike_*`                   | `SPIKE_MIN_SAMPLES` | `SPIKE_MIN_SAMPLES`    | `debug.bench.compare.anomalies.spike_min_samples` |
| `drift_*`                   | `DRIFT_MIN_SAMPLES` | `DRIFT_MIN_SAMPLES`    | `debug.bench.compare.anomalies.drift_min_samples` |

**Garde défensive supplémentaire** : `skewness_*` et `kurtosis_excess_*` valent également `null` si la variance des échantillons est nulle (toutes valeurs identiques), pour éviter une division par zéro dans `scipy.stats`.

### 2.5 Source des échantillons (`_exact` / `_approx`)

Les statistiques se déclinent en deux variantes selon le canal source des échantillons :

| Suffixe   | Canal source                                  | Nature des échantillons                       |
| --------- | --------------------------------------------- | --------------------------------------------- |
| `_exact`  | Canal `frame` (lignes `count == 1`)           | Valeurs individuelles par frame               |
| `_approx` | Canal d'origine de la sonde (`agg` ou `fast`) | Moyennes pré-agrégées (`avg` de chaque ligne) |

**Champs concernés par cette dichotomie** :

`p90_*`, `p95_*`, `p99_*`, `q1_*`, `q3_*`, `iqr_*`, `skewness_*`, `kurtosis_excess_*`.

**Cas particulier `samples_exact` / `samples_approx`** : nombre d'échantillons effectivement collectés par variante (sert au contrôle des seuils §2.4).

**Cas particulier `fast_*`** : les variantes `_exact` valent **toujours** `null` — le canal `fast` ne produit pas de lignes `count == 1` exploitables, et `samples_exact` vaut systématiquement `0`. Seules les variantes `_approx` sont calculées si le seuil est atteint ; en pratique, elles sont fréquemment `null` sur sessions courtes faute d'échantillons suffisants (comportement émergent, pas de forçage côté code).

**Champs sans variante (frame uniquement)** : `spike_*` et `drift_*` sont calculés **exclusivement** depuis le canal `frame` et n'ont pas de variante `_approx`.

**Périmètre d'application** :

| Famille de champs                  | Session (`target.probes`) | Buckets `cold` / `hot[i]` | Bucket `tail` |
| ---------------------------------- | ------------------------- | ------------------------- | ------------- |
| Percentiles, quartiles, IQR        | ✅                        | ✅                        | ✅            |
| `skewness_*` / `kurtosis_excess_*` | ✅                        | ✅                        | ✅            |
| `spike_*` / `drift_*`              | ❌                        | ✅                        | ✅            |

> Le périmètre phase-only des indicateurs d'anomalie (`spike_*` / `drift_*`) est
> volontaire — cf. `bench-compare.md` section « Détection d'anomalies ».

---

## 3. Structure racine

```json
{
  "schema_version": <int>,
  "generated_at":   <string>,
  "target_session": <string>,
  "target":         { ... },
  "comparisons":    { "<comparison_type>": { ... }, ... }
}
```

---

## 4. Bloc `target`

Contient l'intégralité des statistiques de la session cible.

### 4.1 Structure

```json
"target": {
  "duration_s":       <float>,
  "duration_mono_s":  <float>,
  "frames":           { "agg": <int>, "frame": <int>, "fast": <int> },
  "temporal_events":  { ... },
  "probes":           { "<probe_name>": <probe_stats_global>, ... },
  "rates":            { "<rate_name>":  <float>, ... },
  "gauges":           { "<gauge_name>": <float>, ... },
  "fast_probes":      { "<probe_name>": <probe_stats_fast_global>, ... },
  "fast_rates":       { "<rate_name>":  <float>, ... },
  "fast_gauges":      { "<gauge_name>": <float>, ... },
  "buckets":          { ... } | null
}
```

> `target.buckets` peut valoir `null` si la session contient moins de 2 lignes `agg` (bucketing impossible — cf. `bench-compare.md` invariants).

### 4.2 `temporal_events`

Statistiques sur la régularité temporelle des lignes JSONL par canal.

```json
"temporal_events": {
  "agg":   { "median_interval_s": <float | null>, "gaps_stat": <int>, "gaps_fixed": <int | null> },
  "frame": { "median_interval_s": <float | null>, "gaps_stat": <int>, "gaps_fixed": <int | null> },
  "fast":  { "median_interval_s": <float | null>, "gaps_stat": <int>, "gaps_fixed": <int | null> }
}
```

| Champ               | Type          | Description                                                         |
| ------------------- | ------------- | ------------------------------------------------------------------- |
| `median_interval_s` | float \| null | Médiane des intervalles entre lignes consécutives (secondes).       |
| `gaps_stat`         | int           | Nombre de gaps détectés (intervalles > seuil configurable).         |
| `gaps_fixed`        | int \| null   | Nombre de gaps corrigés par interpolation (null si non applicable). |

> **Note canal `frame`** : `gaps_fixed` vaut systématiquement `null` sur le canal `frame` (canal event-driven sans interpolation possible).

---

## 5. Contrat des blocs `probe_stats`

### 5.1 `probe_stats_global` — sondes globales (`probes`, `fast_probes` hors buckets)

```json
"<probe_name>": {
  "avg":             <float>,
  "min":             <float>,
  "max":             <float>,
  "count_agg":       <int>,
  "samples_exact":   <int>,
  "samples_approx":  <int>,
  "p90_exact":       <float | null>,
  "p95_exact":       <float | null>,
  "p99_exact":       <float | null>,
  "p90_approx":      <float | null>,
  "p95_approx":      <float | null>,
  "p99_approx":      <float | null>,
  "q1_exact":        <float | null>,
  "q1_approx":       <float | null>,
  "q3_exact":        <float | null>,
  "q3_approx":       <float | null>,
  "iqr_exact":       <float | null>,
  "iqr_approx":      <float | null>,
  "skewness_exact":        <float | null>,
  "skewness_approx":       <float | null>,
  "kurtosis_excess_exact": <float | null>,
  "kurtosis_excess_approx": <float | null>,
}
```

> **Pour `fast_probes` globaux (`probe_stats_fast_global`)** :
> `count_agg` est remplacé par `count_fast`.
> `samples_exact` vaut toujours `0`.
> Toutes les variantes `_exact` (percentiles, quartiles, IQR, skewness, kurtosis_excess) valent toujours `null`.
> Les variantes `_approx` de `skewness` et `kurtosis_excess` sont calculées si le seuil §2.4 est atteint (pas de forçage `null`).

### 5.2 `probe_stats_bucket` — sondes dans un bucket (`cold`, `hot[i]`, `tail`)

Hérite de `probe_stats_global` et ajoute les champs d'anomalies S5b :

```json
"<probe_name>": {
  "avg":             <float>,
  "min":             <float>,
  "max":             <float>,
  "count_agg":       <int>,
  "samples_exact":   <int>,
  "samples_approx":  <int>,
  "p90_exact":       <float | null>,
  "p95_exact":       <float | null>,
  "p99_exact":       <float | null>,
  "p90_approx":      <float | null>,
  "p95_approx":      <float | null>,
  "p99_approx":      <float | null>,
  "q1_exact":        <float | null>,
  "q1_approx":       <float | null>,
  "q3_exact":        <float | null>,
  "q3_approx":       <float | null>,
  "iqr_exact":       <float | null>,
  "iqr_approx":      <float | null>,
  "skewness_exact":         <float | null>,
  "skewness_approx":        <float | null>,
  "kurtosis_excess_exact":  <float | null>,
  "kurtosis_excess_approx": <float | null>,
  "spike_count":         <int | null>,
  "spike_max_value":     <float | null>,
  "spike_max_deviation": <float | null>,
  "drift_slope":         <float | null>,
  "drift_intercept":     <float | null>,
  "drift_r2":            <float | null>
}
```

### 5.3 `probe_stats_fast_bucket` — fast_probes dans un bucket

Identique à `probe_stats_bucket` avec les restrictions suivantes :

| Champ                                   | Valeur produite                                        |
| --------------------------------------- | ------------------------------------------------------ |
| `samples_exact`                         | `0` (constant)                                         |
| `p90_exact` / `p95_exact` / `p99_exact` | `null` (constant)                                      |
| `q1_exact` / `q3_exact` / `iqr_exact`   | `null` (constant)                                      |
| `skewness_exact`                        | `null` (constant)                                      |
| `skewness_approx`                       | calculé si seuil §2.4 atteint, sinon `null` (émergent) |
| `kurtosis_excess_exact`                 | `null` (constant)                                      |
| `kurtosis_excess_approx`                | calculé si seuil §2.4 atteint, sinon `null` (émergent) |
| `spike_count`                           | `null` forcé (pas de source `frame` côté fast)         |
| `spike_max_value`                       | `null` forcé                                           |
| `spike_max_deviation`                   | `null` forcé                                           |
| `drift_slope`                           | `null` forcé                                           |
| `drift_intercept`                       | `null` forcé                                           |
| `drift_r2`                              | `null` forcé                                           |

> Tous ces champs sont **présents dans la sortie JSON** (jamais omis).
> Distinction importante :
> Champs « constant `null` » / « forcé `null` » : structurellement impossibles à calculer pour fast.
> Champs « émergent » : calculables en théorie mais souvent `null` faute d'échantillons.

---

## 6. Bloc `buckets`

### 6.1 `sync_metadata`

Métadonnées de synchronisation cold/hot entre sessions.

```json
"sync_metadata": {
  "cold_end_target_s":  <float>,
  "cold_end_real_s":    <float>,
  "cold_drift_s":       <float>,
  "cold_drift_warning": <bool>,
  "cold_truncated":     <bool>,
  "fast_enabled":       <bool>
}
```

| Champ                | Type  | Description                                                         |
| -------------------- | ----- | ------------------------------------------------------------------- |
| `cold_end_target_s`  | float | Durée cible du bucket cold (config).                                |
| `cold_end_real_s`    | float | Durée réelle du bucket cold après snap pivot.                       |
| `cold_drift_s`       | float | Dérive absolue `cold_end_real_s − cold_end_target_s`.               |
| `cold_drift_warning` | bool  | `true` si dérive > seuil configurable.                              |
| `cold_truncated`     | bool  | `true` si la session est plus courte que `cold_end_target_s`.       |
| `fast_enabled`       | bool  | `true` si le canal `fast` est présent et activé pour cette session. |

### 6.2 Bucket `cold`

```json
"cold": {
  "mono_start":         <float>,
  "mono_end":           <float>,
  "duration_s":         <float>,
  "cold_end_target_s":  <float>,
  "cold_end_real_s":    <float>,
  "cold_drift_s":       <float>,
  "cold_drift_warning": <bool>,
  "cold_truncated":     <bool>,
  "frames":     { "agg": <int>, "frame": <int>, "fast": <int> },
  "frame_budget": { "groups": <int>, "reference": <int | null>, "rows_total": <int>, "total_ms": <float>, "unaccounted_pct": <float>, "unaccounted_warn": <bool> },
  "probes":      { "<probe_name>": <probe_stats_bucket>, ... },
  "rates":       { "<rate_name>":  <float>, ... },
  "gauges":      { "<gauge_name>": <float>, ... },
  "fast_probes": { "<probe_name>": <probe_stats_fast_bucket>, ... },
  "fast_rates":  { "<rate_name>":  <float>, ... },
  "fast_gauges": { "<gauge_name>": <float>, ... },
  "correlations": {
    "pairs": [
      {
        "probe_a":         <string>,
        "probe_b":         <string>,
        "rho":             <float>,
        "strength":        <string>,         // "moderate" | "strong" | "very_strong"
        "n_samples":       <int>
      }
    ],
    "summary": {
      "n_metrics_excluded_blacklist":  <int>,
      "n_metrics_excluded_zero_var":  <int>,
      "n_metrics_total":              <int>,
      "n_pairs_below_threshold":      <int>,
      "n_pairs_evaluated":            <int>,
      "n_pairs_low_samples":          <int>,
      "n_pairs_reported":             <int>,
      "n_rows":                       <int>,
      "truncated_by_max_pairs":       <bool>
    }
  }
}
```

> **Note duplication** : les 5 champs `cold_end_target_s`, `cold_end_real_s`, `cold_drift_s`, `cold_drift_warning`, `cold_truncated` sont dupliqués depuis `sync_metadata` (§6.1) pour faciliter l'accès local au bucket. Les valeurs sont identiques entre les deux emplacements.

### 6.3 Bucket `hot` (tableau)

```json
"hot": [
  {
    "index":            <int>,
    "mono_start":       <float>,
    "mono_end":         <float>,
    "duration_s":       <float>,
    "is_pivot_snapped": <bool>,
    "frames":      { "agg": <int>, "frame": <int>, "fast": <int> },
    "probes":      { "<probe_name>": <probe_stats_bucket>, ... },
    "rates":       { "<rate_name>":  <float>, ... },
    "gauges":      { "<gauge_name>": <float>, ... },
    "fast_probes": { "<probe_name>": <probe_stats_fast_bucket>, ... },
    "fast_rates":  { "<rate_name>":  <float>, ... },
    "fast_gauges": { "<gauge_name>": <float>, ... },
    "correlations": {
      "pairs": [
        {
          "probe_a":   <string>,
          "probe_b":   <string>,
          "rho":       <float>,
          "strength":  <string>,
          "n_samples": <int>
        }
      ],
      "summary": {
        "n_metrics_excluded_blacklist":  <int>,
        "n_metrics_excluded_zero_var":  <int>,
        "n_metrics_total":              <int>,
        "n_pairs_below_threshold":      <int>,
        "n_pairs_evaluated":            <int>,
        "n_pairs_low_samples":          <int>,
        "n_pairs_reported":             <int>,
        "n_rows":                       <int>,
        "truncated_by_max_pairs":       <bool>
      }
    }
  },
  ...
]
```

| Champ              | Type | Description                                             |
| ------------------ | ---- | ------------------------------------------------------- |
| `index`            | int  | Indice du bucket hot (commence à `0`).                  |
| `is_pivot_snapped` | bool | `true` si la borne de début a été snappée sur un pivot. |

### 6.4 Bucket `tail`

```json
"tail": {
  "mono_start": <float>,
  "mono_end":   <float>,
  "duration_s": <float>,
  "is_partial": true,
  "frames":      { "agg": <int>, "frame": <int>, "fast": <int> },
  "frame_budget": { "groups": <int>, "reference": <int | null>, "rows_total": <int>, "total_ms": <float>, "unaccounted_pct": <float>, "unaccounted_warn": <bool> },
  "probes":      { "<probe_name>": <probe_stats_bucket>, ... },
  "rates":       { "<rate_name>":  <float>, ... },
  "gauges":      { "<gauge_name>": <float>, ... },
  "fast_probes": { "<probe_name>": <probe_stats_fast_bucket>, ... },
  "fast_rates":  { "<rate_name>":  <float>, ... },
  "fast_gauges": { "<gauge_name>": <float>, ... },
  "correlations": {
    "pairs": [
      {
        "probe_a":         <string>,
        "probe_b":         <string>,
        "rho":             <float>,
        "strength":        <string>,         // "moderate" | "strong" | "very_strong"
        "n_samples":       <int>
      }
    ],
    "summary": {
      "n_metrics_excluded_blacklist":  <int>,
      "n_metrics_excluded_zero_var":  <int>,
      "n_metrics_total":              <int>,
      "n_pairs_below_threshold":      <int>,
      "n_pairs_evaluated":            <int>,
      "n_pairs_low_samples":          <int>,
      "n_pairs_reported":             <int>,
      "n_rows":                       <int>,
      "truncated_by_max_pairs":       <bool>
    }
  }
}
```

> `is_partial` est **toujours `true`** — le tail représente la fin de session potentiellement incomplète par construction.
> Le bloc `tail` peut valoir `null` si la session se termine exactement sur la frontière d'un bucket `hot` (pas de résidu à représenter).

---

## 7. Bloc `comparisons`

### 7.1 Structure

```json
"comparisons": {
  "<comparison_type>": {
    "reference_session": <string>,
    "reference": { ... },
    "deltas":    { ... },
    "buckets":    { ... },
    "appeared_probes":         [<string>, ...],
    "disappeared_probes":      [<string>, ...],
    "appeared_rates":          [<string>, ...],
    "disappeared_rates":       [<string>, ...],
    "appeared_gauges":         [<string>, ...],
    "disappeared_gauges":      [<string>, ...],
    "appeared_fast_probes":    [<string>, ...],
    "disappeared_fast_probes": [<string>, ...],
    "appeared_fast_rates":     [<string>, ...],
    "disappeared_fast_rates":  [<string>, ...],
    "appeared_fast_gauges":    [<string>, ...],
    "disappeared_fast_gauges": [<string>, ...]
  }
}
```

| Champ               | Type   | Description                                                            |
| ------------------- | ------ | ---------------------------------------------------------------------- |
| `reference_session` | string | Identifiant de la session de référence.                                |
| `reference`         | object | Même structure que `target` — statistiques de la session de référence. |
| `deltas`            | object | Différences calculées scalaires (voir §7.2).                           |
| `buckets`           | object | Deltas ventilés par bucket cold/hot/tail (voir §7.3).                  |
| `appeared_*`        | array  | Sondes/rates/gauges présentes dans target, absentes de reference.      |
| `disappeared_*`     | array  | Sondes/rates/gauges présentes dans reference, absentes de target.      |

### 7.2 Bloc `deltas`

```json
"deltas": {
  "temporal":    { ... },
  "probes":      { "<probe_name>": <delta_probe_global>, ... },
  "rates":       { "<rate_name>":  { "delta_pct": <float | null> }, ... },
  "gauges":      { "<gauge_name>": { "delta_pct": <float | null> }, ... },
  "fast_probes": { "<probe_name>": <delta_probe_global>, ... },
  "fast_rates":  { "<rate_name>":  { "delta_pct": <float | null> }, ... },
  "fast_gauges": { "<gauge_name>": { "delta_pct": <float | null> }, ... },
  "frame_budget": { "appeared_groups": <int>, "disappeared_groups": <int>, "groups": <int>, "total_ms_delta_pct": <float | null>, "unaccounted_pct_delta": <float | null> }
}
```

#### 7.2.1 `deltas.temporal`

```json
"temporal": {
  "duration_mono_s": { "delta_pct": <float | null> },
  "agg":   {
    "frames":             { "reference": <int>, "target": <int>, "delta": <int> },
    "median_interval_s":  { "reference": <float | null>, "target": <float | null>, "delta": <float | null> },
    "gaps_stat":          { "reference": <int>, "target": <int>, "delta": <int> },
    "gaps_fixed":         { "reference": <int | null>, "target": <int | null>, "delta": <int | null> }
  },
  "frame": { "...même structure que agg..." },
  "fast":  { "...même structure que agg..." }
}
```

> **Note canal `frame`** : `gaps_fixed.delta` vaut `null` constant (les deux valeurs source sont `null` par construction event-driven).

#### 7.2.2 `delta_probe_global` — deltas sondes hors buckets

```json
"<probe_name>": {
  "avg_delta_pct":                <float | null>,
  "min_delta_pct":                <float | null>,
  "max_delta_pct":                <float | null>,
  "p90_exact_delta_pct":          <float | null>,
  "p90_approx_delta_pct":         <float | null>,
  "p95_exact_delta_pct":          <float | null>,
  "p95_approx_delta_pct":         <float | null>,
  "p99_exact_delta_pct":          <float | null>,
  "p99_approx_delta_pct":         <float | null>,
  "q1_exact_delta_pct":           <float | null>,
  "q1_approx_delta_pct":          <float | null>,
  "q3_exact_delta_pct":           <float | null>,
  "q3_approx_delta_pct":          <float | null>,
  "iqr_exact_delta_pct":          <float | null>,
  "iqr_approx_delta_pct":         <float | null>,
  "skewness_exact_delta":         <float | null>,
  "skewness_approx_delta":        <float | null>,
  "kurtosis_excess_exact_delta":  <float | null>,
  "kurtosis_excess_approx_delta": <float | null>
}
```

> **Pas de delta sur compteurs** : `count_agg` / `count_fast` / `samples_exact` / `samples_approx` ne génèrent pas de champ delta (valeurs brutes seules dans `target` et `reference`).
> **Pas de delta sur anomalies au niveau global** : `spike_*` et `drift_*` n'ont pas de delta global (ces champs ne sont présents que dans les deltas de buckets — voir §7.2.3).

#### 7.2.3 `delta_probe_bucket` — deltas sondes dans un bucket

Hérite de `delta_probe_global` et ajoute les deltas anomalies :

```json
"<probe_name>": {
  "avg_delta_pct":                <float | null>,
  "min_delta_pct":                <float | null>,
  "max_delta_pct":                <float | null>,
  "p90_exact_delta_pct":          <float | null>,
  "p90_approx_delta_pct":         <float | null>,
  "p95_exact_delta_pct":          <float | null>,
  "p95_approx_delta_pct":         <float | null>,
  "p99_exact_delta_pct":          <float | null>,
  "p99_approx_delta_pct":         <float | null>,
  "q1_exact_delta_pct":           <float | null>,
  "q1_approx_delta_pct":          <float | null>,
  "q3_exact_delta_pct":           <float | null>,
  "q3_approx_delta_pct":          <float | null>,
  "iqr_exact_delta_pct":          <float | null>,
  "iqr_approx_delta_pct":         <float | null>,
  "skewness_exact_delta":         <float | null>,
  "skewness_approx_delta":        <float | null>,
  "kurtosis_excess_exact_delta":  <float | null>,
  "kurtosis_excess_approx_delta": <float | null>,
  "spike_count_delta":            <int | null>,
  "spike_max_deviation_delta":    <float | null>,
  "drift_slope_delta":            <float | null>,
  "drift_r2_delta":               <float | null>
}
```

> **Sémantique delta** : si l'une des deux valeurs source est `null` → delta `null`.
> `spike_max_value` et `drift_intercept` n'ont **jamais** de clé `delta` — valeurs brutes seules.

### 7.3 Bloc `buckets` (deltas par bucket)

```json
"buckets": {
  "cold": {
    "duration_delta_pct": <float | null>,
    "frame_budget": { "appeared_groups": <int>, "disappeared_groups": <int>, "groups": <int>, "total_ms_delta_pct": <float | null>, "unaccounted_pct_delta": <float | null> },
    "probes":      { "<probe_name>": <delta_probe_bucket>, ... },
    "rates":       { "<rate_name>":  { "delta_pct": <float | null> }, ... },
    "gauges":      { "<gauge_name>": { "delta_pct": <float | null> }, ... },
    "fast_probes": { "<probe_name>": <delta_probe_bucket>, ... },
    "fast_rates":  { "<rate_name>":  { "delta_pct": <float | null> }, ... },
    "fast_gauges": { "<gauge_name>": { "delta_pct": <float | null> }, ... }
  },
  "hot": [
    {
      "index":                <int>,
      "duration_delta_pct":   <float | null>,
      "frame_budget": { "appeared_groups": <int>, "disappeared_groups": <int>, "groups": <int>, "total_ms_delta_pct": <float | null>, "unaccounted_pct_delta": <float | null> },
      "probes":      { "<probe_name>": <delta_probe_bucket>, ... },
      "rates":       { "<rate_name>":  { "delta_pct": <float | null> }, ... },
      "gauges":      { "<gauge_name>": { "delta_pct": <float | null> }, ... },
      "fast_probes": { "<probe_name>": <delta_probe_bucket>, ... },
      "fast_rates":  { "<rate_name>":  { "delta_pct": <float | null> }, ... },
      "fast_gauges": { "<gauge_name>": { "delta_pct": <float | null> }, ... }
    }
  ],
  "unaligned_hot": [<int>, ...],
  "tail_status": "<aligned | both_absent | target_absent | ref_absent>",
  "tail": {
    "duration_delta_pct": <float | null>,
    "probes":      { "<probe_name>": <delta_probe_bucket>, ... },
    "rates":       { "<rate_name>":  { "delta_pct": <float | null> }, ... },
    "gauges":      { "<gauge_name>": { "delta_pct": <float | null> }, ... },
    "fast_probes": { "<probe_name>": <delta_probe_bucket>, ... },
    "fast_rates":  { "<rate_name>":  { "delta_pct": <float | null> }, ... },
    "fast_gauges": { "<gauge_name>": { "delta_pct": <float | null> }, ... }
  }
}
```

| Champ                | Type          | Description                                                             |
| -------------------- | ------------- | ----------------------------------------------------------------------- |
| `duration_delta_pct` | float \| null | Delta de durée en % (target − reference) / \|reference\| × 100.         |
| `unaligned_hot`      | array\[int\]  | Indices des buckets hot sans correspondance dans la session opposée.    |
| `tail_status`        | string        | Statut d'alignement du tail entre les deux sessions (toujours présent). |

#### Valeurs de `tail_status`

| Valeur          | Signification                                             | Bloc `tail` présent ? |
| --------------- | --------------------------------------------------------- | --------------------- |
| `aligned`       | Les deux sessions ont un tail — deltas calculés.          | ✅ Oui                |
| `both_absent`   | Aucune session n'a de tail.                               | ❌ Absent             |
| `target_absent` | Le tail est absent de la session cible uniquement.        | ❌ Absent             |
| `ref_absent`    | Le tail est absent de la session de référence uniquement. | ❌ Absent             |

> Le champ `tail_status` est **toujours présent** dans le JSON (jamais omis).
> Quand `tail_status != "aligned"`, le bloc `tail` est **absent** du JSON.

---

## 8. Matrice des sections par zone

| Section             | `target` global | `target.buckets.cold/hot/tail` | `deltas` global | `deltas.buckets.cold/hot/tail` |
| ------------------- | --------------- | ------------------------------ | --------------- | ------------------------------ |
| `probes`            | ✅              | ✅                             | ✅              | ✅                             |
| `fast_probes`       | ✅              | ✅                             | ✅              | ✅                             |
| `rates`             | ✅              | ✅                             | ✅              | ✅                             |
| `fast_rates`        | ✅              | ✅                             | ✅              | ✅                             |
| `gauges`            | ✅              | ✅                             | ✅              | ✅                             |
| `fast_gauges`       | ✅              | ✅                             | ✅              | ✅                             |
| `skewness_*`        | ✅ (probe)      | ✅ (probe)                     | ✅              | ✅                             |
| `kurtosis_excess_*` | ✅ (probe)      | ✅ (probe)                     | ✅              | ✅                             |
| `spike_* / drift_*` | ❌              | ✅ (probe bucket uniquement)   | ❌              | ✅                             |
| `sync_metadata`     | ❌              | ✅ (racine buckets)            | ❌              | ❌                             |
| `tail_status`       | ❌              | ❌                             | ❌              | ✅ (racine `buckets`)          |
| `temporal_events`   | ✅              | ❌                             | ✅ (`temporal`) | ❌                             |

---

## 9. Règles d'évolution

| Type de changement                              | Action                                  |
| ----------------------------------------------- | --------------------------------------- |
| Ajout d'une sonde (nouvelle clé dans `probes`)  | Aucun bump (open-set par construction). |
| Ajout d'un champ méta (ex. `host`)              | Bump `schema_version`.                  |
| Suppression ou renommage d'un champ méta        | Bump `schema_version`.                  |
| Modification du contrat d'une section imbriquée | Bump `schema_version`.                  |
| Ajout d'une nouvelle section imbriquée          | Bump `schema_version`.                  |
| Ajout d'un nouveau type de comparaison          | Aucun bump (open-set par construction). |

---

## 10. Référence d'implémentation

Producteur unique : `bench/bench_compare.py`.
Toute divergence entre ce document et l'implémentation est un bug de l'un ou de l'autre ,la résolution est arbitrée par l'équipe avant merge.
