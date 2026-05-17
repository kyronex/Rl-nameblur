# Schéma JSONL bench — Contrat normatif L0.4

> 🔒 **Statut** : figé.
> Toute modification du schéma (ajout/suppression/renommage de champ, changement de type, restructuration) requiert :
>
> 1. Incrément de `schema_version`.
> 2. Ouverture d'un nouveau ticket dédié.
> 3. Mise à jour du présent document.

---

## 1. Portée

Ce document décrit le format des fichiers JSONL produits par `bench/jsonl_writer.py`.

Trois canaux indépendants, un fichier par canal et par session :

| Canal   | Fichier                          | Cadence d'écriture                 |
| ------- | -------------------------------- | ---------------------------------- |
| `frame` | `bench_frame_{session_id}.jsonl` | 1 ligne / frame capturée           |
| `agg`   | `bench_agg_{session_id}.jsonl`   | 1 ligne / `agg_interval` (config)  |
| `fast`  | `bench_fast_{session_id}.jsonl`  | 1 ligne / `fast_interval` (config) |

Chaque ligne est un objet JSON autonome, sans dépendance inter-lignes.

---

## 2. Conventions communes

### 2.1 Méta-champs obligatoires

Tout objet JSONL, quel que soit le canal, expose les 5 méta-champs suivants en tête :

| Champ            | Type   | Description                                              |
| ---------------- | ------ | -------------------------------------------------------- |
| `schema_version` | int    | Version du contrat (valeur courante : `1`).              |
| `ts`             | float  | Timestamp wall-clock UNIX (`time.time()`), secondes.     |
| `mono`           | float  | Horloge monotone (`time.perf_counter()`), secondes.      |
| `session_id`     | string | Identifiant unique de session, propagé sur les 3 canaux. |
| `mode`           | string | Canal d'origine : `"frame"` / `"agg"` / `"fast"`.        |

### 2.2 Compatibilité ascendante

Une ligne JSONL **sans** champ `schema_version` est conventionnellement interprétée comme `schema_version = 1` (sessions antérieures à l'introduction du champ).

### 2.3 Unités

Le schéma ne normalise **pas** les unités des sondes. La sémantique (secondes, pixels, ratio, compteur, etc.) relève du producteur de la sonde et doit être documentée séparément dans l'inventaire des sondes (hors périmètre L0.4).

---

## 3. Canal `agg`

### 3.1 Structure

```json
{
  "schema_version": 1,
  "ts": <float>,
  "mono": <float>,
  "session_id": <string>,
  "mode": "agg",
  "probes": { "<probe_name>": <probe_stats>, ... },
  "gauges": { "<gauge_name>": <float>, ... },
  "rates":  { "<rate_name>": <float>, ... }
}
```

### 3.2 Sections de données

| Section  | Obligatoire | Peut être vide `{}` | Description                                      |
| -------- | ----------- | ------------------- | ------------------------------------------------ |
| `probes` | oui         | oui                 | Statistiques agrégées sur la fenêtre temporelle. |
| `gauges` | oui         | oui                 | Dernière valeur observée dans la fenêtre.        |
| `rates`  | oui         | oui                 | Taux dérivés (`count_delta / interval_s`).       |

---

## 4. Canal `fast`

### 4.1 Structure

```json
{
  "schema_version": 1,
  "ts": <float>,
  "mono": <float>,
  "session_id": <string>,
  "mode": "fast",
  "probes": { "<probe_name>": <probe_stats>, ... },
  "gauges": { "<gauge_name>": <float>, ... },
  "rates":  { "<rate_name>": <float>, ... }
}
```

### 4.2 Sections de données

| Section  | Obligatoire | Peut être vide `{}` | Description                                      |
| -------- | ----------- | ------------------- | ------------------------------------------------ |
| `probes` | oui         | oui                 | Statistiques agrégées sur la fenêtre temporelle. |
| `gauges` | oui         | oui                 | Dernière valeur observée dans la fenêtre.        |
| `rates`  | oui         | oui                 | Taux dérivés (`count_delta / interval_s`).       |

---

## 5. Canal `frame`

### 5.1 Structure

```json
{
  "schema_version": 1,
  "ts": <float>,
  "mono": <float>,
  "session_id": <string>,
  "mode": "frame",
  "probes": { "<probe_name>": <probe_stats>, ... },
  "gauges": { "<gauge_name>": <float>, ... },
  "counts": { "<count_name>": <int>, ... }
}
```

### 5.2 Sections de données

| Section  | Obligatoire | Peut être vide `{}` | Description                                                                   |
| -------- | ----------- | ------------------- | ----------------------------------------------------------------------------- |
| `probes` | oui         | oui                 | Statistiques de la frame courante (timing, pixels, scores, etc.).             |
| `gauges` | oui         | oui                 | Dernière valeur scalaire observée dans la frame (ex. nombre de masks actifs). |
| `counts` | oui         | oui                 | Compteurs bruts cumulatifs par frame (ex. `main_frames_total`).               |

> ⚠️ **`counts` ≠ `rates`** : les valeurs de `counts` sont des entiers bruts cumulatifs depuis
> le démarrage de la session, **pas** des taux normalisés par seconde. Cette sémantique diffère
> volontairement des sections `rates` présentes sur les canaux `agg` et `fast`.
> 📋 **Note d'évolution** : le canal `frame` n'expose pas de section `rates`. Si un besoin de
> taux par frame émerge, il constitue un nouveau ticket avec bump `schema_version`.

---

## 6. Contrat des sections imbriquées

### 6.1 `probes`

Chaque entrée est un objet de statistiques agrégées sur la fenêtre temporelle du canal.

```json
"<probe_name>": {
  "avg":   <float>,
  "max":   <float>,
  "min":   <float>,
  "count": <int>
}
```

| Clé     | Type  | Obligatoire | Description                          |
| ------- | ----- | ----------- | ------------------------------------ |
| `avg`   | float | oui         | Moyenne arithmétique sur la fenêtre. |
| `max`   | float | oui         | Maximum observé sur la fenêtre.      |
| `min`   | float | oui         | Minimum observé sur la fenêtre.      |
| `count` | int   | oui         | Nombre d'échantillons agrégés.       |

> Les 4 clés sont **toujours présentes**, même si `count == 0` (auquel cas `avg`/`max`/`min` peuvent valoir `0.0` par convention producteur).

### 6.2 `gauges`

Chaque entrée est un scalaire flottant représentant la dernière valeur observée dans la fenêtre.

```json
"<gauge_name>": <float>
```

### 6.3 `rates`

Chaque entrée est un scalaire flottant représentant un taux dérivé (typiquement `count_delta / interval_s`).
Présent sur les canaux `agg` et `fast` uniquement.

```json
"<rate_name>": <float>
```

---

### 6.4 `counts`

Chaque entrée est un entier représentant un compteur brut cumulatif depuis le démarrage
de la session. Présent sur le canal `frame` uniquement.

```json
"<count_name>": <int>
```

---

## 7. Matrice des sections par canal

| Section  | `frame` | `agg` | `fast` |
| -------- | ------- | ----- | ------ |
| `probes` | ✅      | ✅    | ✅     |
| `gauges` | ✅      | ✅    | ✅     |
| `counts` | ✅      | ❌    | ❌     |
| `rates`  | ❌      | ✅    | ✅     |

---

## 8. Règles d'évolution

| Type de changement                              | Action                                  |
| ----------------------------------------------- | --------------------------------------- |
| Ajout d'une sonde (nouvelle clé dans `probes`)  | Aucun bump (open-set par construction). |
| Ajout d'un champ méta (ex. `host`)              | Bump `schema_version`.                  |
| Suppression ou renommage d'un champ méta        | Bump `schema_version`.                  |
| Modification du contrat d'une section imbriquée | Bump `schema_version`.                  |
| Ajout d'une nouvelle section imbriquée          | Bump `schema_version`.                  |
| Ajout d'un nouveau canal                        | Bump `schema_version`.                  |

---

## 9. Référence d'implémentation

Producteur unique : `bench/jsonl_writer.py`, méthode `_enqueue()`.
Toute divergence entre ce document et l'implémentation est un bug de l'un ou de l'autre — la résolution est arbitrée par l'équipe avant merge.

---

## Historique des versions

| Version | Date       | Motif                                                                           |
| ------- | ---------- | ------------------------------------------------------------------------------- |
| 1       | 2026-05-15 | Version initiale — 3 canaux, sections `probes`/`gauges`/`rates`.                |
| 1       | 2026-05-17 | Option A : ajout section `counts` canal `frame`, retrait `rates` frame .        |
|         |            | Alignement sur tests réels. Pas de bump correction intra-v1 pré-livraison L0.4. |
