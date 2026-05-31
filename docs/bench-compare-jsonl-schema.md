# Schéma JSON bench-compare — Contrat normatif L0.5

> 🔒 **Statut** : figé.
> Toute modification du schéma (ajout/suppression/renommage de champ, changement de type, restructuration) requiert :
>
> 1. Incrément de `schema_version`.
> 2. Ouverture d'un nouveau ticket dédié.
> 3. Mise à jour du présent document.

---

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

| Champ            | Type   | Description                                                         |
| ---------------- | ------ | ------------------------------------------------------------------- |
| `schema_version` | int    | Version du contrat (valeur courante : `1`).                         |
| `generated_at`   | string | Timestamp de génération ISO 8601 (`datetime.utcnow().isoformat()`). |
| `target_session` | string | Identifiant de la session analysée (session cible).                 |

### 2.2 Valeurs nulles

Un champ marqué `float | null` ou `int | null` vaut `null` (JSON) lorsque le calcul
est impossible (échantillons insuffisants, division par zéro, données absentes).

### 2.3 Unités

Le schéma ne normalise **pas** les unités des sondes. La sémantique relève du producteur
de la sonde. Les deltas en pourcentage (`_delta_pct`) sont calculés comme :

```text
delta_pct = (target − reference) / abs(reference) × 100
```

`null` si `reference == 0` ou si l'une des valeurs source est `null`.

### 2.4 Seuils statistiques minimaux

| Statistique       | Seuil minimum                | Comportement sous seuil |
| ----------------- | ---------------------------- | ----------------------- |
| `skewness`        | 50 échantillons              | `null` silencieux       |
| `kurtosis_excess` | 100 échantillons             | `null` silencieux       |
| `spike_*`         | `SPIKE_MIN_SAMPLES` (config) | `null` silencieux       |
| `drift_*`         | `DRIFT_MIN_SAMPLES` (config) | `null` silencieux       |

### 2.5 Source des échantillons exacts

Les champs `samples_exact`, `p90_exact`, `p95_exact`, `p99_exact`, `q1_exact`, `q3_exact`,
`iqr_exact`, `skewness`, `kurtosis_excess`, `spike_*`, `drift_*` sont calculés
**exclusivement** depuis le canal `frame` (lignes `count == 1`).

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
  "buckets":          { ... }
}
```

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
  "skewness":        <float | null>,
  "kurtosis_excess": <float | null>
}
```

> Pour `fast_probes` globaux : `count_agg` est remplacé par `count_fast`. Les champs
> `skewness` et `kurtosis_excess` sont forcés à `null` (source exacte non disponible
> sur le canal `fast`).

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
  "skewness":        <float | null>,
  "kurtosis_excess": <float | null>,
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

| Champ                 | Valeur forcée |
| --------------------- | ------------- |
| `skewness`            | `null`        |
| `kurtosis_excess`     | `null`        |
| `spike_count`         | `null`        |
| `spike_max_value`     | `null`        |
| `spike_max_deviation` | `null`        |
| `drift_slope`         | `null`        |
| `drift_intercept`     | `null`        |
| `drift_r2`            | `null`        |

> Ces champs sont présents dans la sortie JSON (jamais omis) mais toujours à `null`.
> Cohérence avec la décision D10 : `fast_probes` forcés à `null` pour les anomalies S5b.

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
  "mono_start": <float>,
  "mono_end":   <float>,
  "duration_s": <float>,
  "frames":     { "agg": <int>, "frame": <int>, "fast": <int> },
  "probes":      { "<probe_name>": <probe_stats_bucket>, ... },
  "rates":       { "<rate_name>":  <float>, ... },
  "gauges":      { "<gauge_name>": <float>, ... },
  "fast_probes": { "<probe_name>": <probe_stats_fast_bucket>, ... },
  "fast_rates":  { "<rate_name>":  <float>, ... },
  "fast_gauges": { "<gauge_name>": <float>, ... }
}
```

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
    "fast_gauges": { "<gauge_name>": <float>, ... }
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
  "probes":      { "<probe_name>": <probe_stats_bucket>, ... },
  "rates":       { "<rate_name>":  <float>, ... },
  "gauges":      { "<gauge_name>": <float>, ... },
  "fast_probes": { "<probe_name>": <probe_stats_fast_bucket>, ... },
  "fast_rates":  { "<rate_name>":  <float>, ... },
  "fast_gauges": { "<gauge_name>": <float>, ... }
}
```

> `is_partial` est **toujours `true`** — le tail représente la fin de session
> potentiellement incomplète par construction.

---

## 7. Bloc `comparisons`

### 7.1 Structure

```json
"comparisons": {
  "<comparison_type>": {
    "reference_session": <string>,
    "reference": { ... },
    "deltas":    { ... },
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
| `deltas`            | object | Différences calculées (voir §7.2).                                     |
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
  "buckets":     { ... }
}
```

#### 7.2.1 `deltas.temporal`

```json
"temporal": {
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

#### 7.2.2 `delta_probe_global` — deltas sondes hors buckets

```json
"<probe_name>": {
  "avg_delta_pct":   <float | null>,
  "p90_delta_pct":   <float | null>,
  "p95_delta_pct":   <float | null>,
  "p99_delta_pct":   <float | null>,
  "q1_delta_pct":    <float | null>,
  "q3_delta_pct":    <float | null>,
  "iqr_delta_pct":   <float | null>
}
```

> Pas de champs `skewness`/`kurtosis_excess`/`spike_*`/`drift_*` au niveau global
> (ces champs ne sont présents que dans les deltas de buckets — voir §7.2.3).

#### 7.2.3 `delta_probe_bucket` — deltas sondes dans un bucket

Hérite de `delta_probe_global` et ajoute :

```json
"<probe_name>": {
  "avg_delta_pct":   <float | null>,
  "p90_delta_pct":   <float | null>,
  "p95_delta_pct":   <float | null>,
  "p99_delta_pct":   <float | null>,
  "q1_delta_pct":    <float | null>,
  "q3_delta_pct":    <float | null>,
  "iqr_delta_pct":   <float | null>,
  "skewness":        { "reference": <float | null>, "target": <float | null>, "delta": <float | null> },
  "kurtosis_excess": { "reference": <float | null>, "target": <float | null>, "delta": <float | null> },
  "spike_count":         { "reference": <int | null>,   "target": <int | null>,   "delta": <int | null> },
  "spike_max_value":     { "reference": <float | null>, "target": <float | null> },
  "spike_max_deviation": { "reference": <float | null>, "target": <float | null>, "delta": <float | null> },
  "drift_slope":         { "reference": <float | null>, "target": <float | null>, "delta": <float | null> },
  "drift_intercept":     { "reference": <float | null>, "target": <float | null> },
  "drift_r2":            { "reference": <float | null>, "target": <float | null>, "delta": <float | null> }
}
```

> **Sémantique delta** : si l'une des deux valeurs source est `null` → delta `null`.
> `spike_max_value` et `drift_intercept` n'ont **jamais** de clé `delta` — valeurs brutes seules.

#### 7.2.4 `deltas.buckets`

```json
"buckets": {
  "cold": {
    "duration_delta_pct": <float | null>,
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
      "is_pivot_snapped_ref": <bool | null>,
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

| Champ                  | Type          | Description                                                              |
| ---------------------- | ------------- | ------------------------------------------------------------------------ |
| `duration_delta_pct`   | float \| null | Delta de durée en % (target − reference) / \|reference\| × 100.          |
| `is_pivot_snapped_ref` | bool \| null  | Valeur `is_pivot_snapped` du bucket hot de référence (`null` si absent). |
| `unaligned_hot`        | array\[int\]  | Indices des buckets hot sans correspondance dans la session opposée.     |
| `tail_status`          | string        | Statut d'alignement du tail entre les deux sessions.                     |

##### Valeurs de `tail_status`

| Valeur          | Signification                                             |
| --------------- | --------------------------------------------------------- |
| `aligned`       | Les deux sessions ont un tail — deltas calculés.          |
| `both_absent`   | Aucune session n'a de tail — bloc `tail` absent.          |
| `target_absent` | Le tail est absent de la session cible uniquement.        |
| `ref_absent`    | Le tail est absent de la session de référence uniquement. |

> Quand `tail_status != "aligned"`, le bloc `tail` dans `deltas.buckets` est **absent** du JSON.

---

## 8. Matrice des sections par zone

| Section             | `target` global | `target.buckets.cold/hot/tail` | `deltas` global | `deltas.buckets.cold/hot/tail` |
| ------------------- | :-------------: | :----------------------------: | :-------------: | :----------------------------: |
| `probes`            |       ✅        |               ✅               |       ✅        |               ✅               |
| `fast_probes`       |       ✅        |               ✅               |       ✅        |               ✅               |
| `rates`             |       ✅        |               ✅               |       ✅        |               ✅               |
| `fast_rates`        |       ✅        |               ✅               |       ✅        |               ✅               |
| `gauges`            |       ✅        |               ✅               |       ✅        |               ✅               |
| `fast_gauges`       |       ✅        |               ✅               |       ✅        |               ✅               |
| `skewness`          |   ✅ (probe)    |           ✅ (probe)           |       ❌        |    ✅ (delta_probe_bucket)     |
| `kurtosis_excess`   |   ✅ (probe)    |           ✅ (probe)           |       ❌        |    ✅ (delta_probe_bucket)     |
| `spike_* / drift_*` |       ❌        |  ✅ (probe bucket uniquement)  |       ❌        |    ✅ (delta_probe_bucket)     |
| `sync_metadata`     |       ❌        |      ✅ (racine buckets)       |       ❌        |               ❌               |
| `temporal_events`   |       ✅        |               ❌               | ✅ (`temporal`) |               ❌               |

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
Toute divergence entre ce document et l'implémentation est un bug de l'un ou de l'autre —
la résolution est arbitrée par l'équipe avant merge.

---

## Historique des versions

| Version | Date       | Motif                                                                                               |
| ------- | ---------- | --------------------------------------------------------------------------------------------------- |
| 1       | 2026-05-20 | Version initiale — target + comparisons, buckets cold/hot/tail, S5a (skew/kurt), S5b (spike/drift). |
