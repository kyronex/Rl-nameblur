# Schéma JSONL bench — Contrat normatif L0.4

> 🔒 **Statut** : figé.
> Toute modification du schéma (ajout/suppression/renommage de champ, changement de type, restructuration) requiert :
>
> 1. Incrément de `schema_version`.
> 2. Ouverture d'un nouveau ticket dédié.
> 3. Mise à jour du présent document.

---

## Sommaire

1. [Portée](#1-portée)
2. [Méta-champs communs](#2-méta-champs-communs)
   - 2.1 [Champs obligatoires](#21-méta-champs-obligatoires)
   - 2.2 [Règles d'évolution du `schema_version`](#22-règles-dévolution-du-schema_version)
3. [Canal `agg`](#3-canal-agg)
4. [Canal `fast`](#4-canal-fast)
5. [Canal `frame`](#5-canal-frame)
6. [Contrat des sections imbriquées](#6-contrat-des-sections-imbriquées)
   - 6.1 [`probes`](#61-probes)
   - 6.2 [`gauges`](#62-gauges)
   - 6.3 [`rates`](#63-rates)
   - 6.4 [`counts`](#64-counts)
7. [Matrice des sections par canal](#7-matrice-des-sections-par-canal)
8. [Règles d'évolution](#8-règles-dévolution)
9. [Référence d'implémentation](#9-référence-dimplémentation)
10. [Paramètres de configuration](#10-paramètres-de-configuration)

---

## 1. Portée

Ce document décrit le format des fichiers JSONL produits par `bench/jsonl_writer.py`.

Trois canaux indépendants, un fichier par canal et par session :

| Canal   | Fichier                          | Cadence d'écriture                 |
| ------- | -------------------------------- | ---------------------------------- |
| `frame` | `bench_frame_{session_id}.jsonl` | 1 ligne / appel `push_frame()`     |
| `agg`   | `bench_agg_{session_id}.jsonl`   | 1 ligne / `agg_interval` (config)  |
| `fast`  | `bench_fast_{session_id}.jsonl`  | 1 ligne / `fast_interval` (config) |

> **Note canal `frame`** : `push_frame()` est invoqué par la boucle principale (`main.py`) à chaque frame capturée. La cadence est donc gouvernée par le pipeline de capture, pas par un intervalle configurable. Le paramètre `debug.bench.frame.interval_s` est **ignoré** pour ce canal.

---

## 2. Méta-champs communs

### 2.1 Méta-champs obligatoires

Chaque ligne JSONL, tous canaux confondus, contient au minimum :

| Champ            | Type   | Description                                              |
| ---------------- | ------ | -------------------------------------------------------- |
| `schema_version` | int    | Version du contrat (valeur courante : `1`).              |
| `ts`             | float  | Timestamp wall-clock UNIX (`time.time()`), secondes.     |
| `mono`           | float  | Horloge monotone (`time.perf_counter()`), secondes.      |
| `session_id`     | string | Identifiant unique de session, propagé sur les 3 canaux. |
| `mode`           | string | Canal d'origine : `"frame"` / `"agg"` / `"fast"`.        |

### 2.2 Règles d'évolution du `schema_version`

- Toute modification **non rétro-compatible** des sections imbriquées (suppression de champ, renommage, changement de type) **doit** incrémenter `schema_version`.
- L'ajout d'une nouvelle section optionnelle ou d'un nouveau champ optionnel **n'impose pas** d'incrément, à condition que les consommateurs existants puissent l'ignorer sans erreur.

---

## 3. Canal `agg`

### 3.1 Structure

```json
{
  "schema_version": <int>,
  "ts": <float>,
  "mono": <float>,
  "session_id": <string>,
  "mode": "agg",
  "probes": { "<probe_name>": <probe_stats>, ... },
  "gauges": { "<gauge_name>": <float>, ... },
  "rates":  { "<rate_name>": <float>, ... }
}
```

### 3.2 Sections — présence

Les sections `probes`, `gauges`, `rates` sont **conditionnellement présentes** :

- Une section est émise **si et seulement si** elle contient au moins une entrée non vide sur la fenêtre temporelle (`interval_s`).
- Une ligne JSONL est émise uniquement si **au moins une** des trois sections est non vide. Sinon le producteur (`snapshot_all`) retourne `{}` et aucune ligne n'est écrite.

> **Conséquence côté consommateur** : tester systématiquement la présence de chaque section avant lecture. L'absence d'une section signifie _« aucune donnée sur la fenêtre »_, pas une erreur.

### 3.3 Sondes exclues

Le canal `agg` exclut :

- Les sondes commençant par `fast_` (réservées au canal `fast` — cf. §4).
- Les sondes commençant par `bench_writer_` (auto-instrumentation interne des writers — cf. §9).

---

## 4. Canal `fast`

### 4.1 Structure

```json
{
  "schema_version": <int>,
  "ts": <float>,
  "mono": <float>,
  "session_id": <string>,
  "mode": "fast",
  "probes": { "<probe_name>": <probe_stats>, ... },
  "gauges": { "<gauge_name>": <float>, ... },
  "rates":  { "<rate_name>": <float>, ... }
}
```

### 4.2 Sections — présence

Mêmes règles que §3.2 : présence conditionnelle, ligne émise uniquement si au moins une section non vide.

### 4.3 Sondes incluses / exclues

Le canal `fast` inclut **exclusivement** les sondes, gauges et compteurs dont le nom commence par le préfixe `fast_`.

> ⚠ Le choix du préfixe à l'émission (`bench.probe("fast_xxx", ...)`) détermine **irréversiblement** le canal de destination. Cf. `bench-compare.md` §_Convention de préfixage des canaux_.

---

## 5. Canal `frame`

### 5.1 Structure

```json
{
  "schema_version": <int>,
  "ts": <float>,
  "mono": <float>,
  "session_id": <string>,
  "mode": "frame",
  "probes": { "<probe_name>": <probe_stats>, ... },
  "gauges": { "<gauge_name>": <float>, ... },
  "counts": { "<count_name>": <int>, ... }
}
```

### 5.2 Sections — présence

Mêmes règles que §3.2.

### 5.3 Sémantique différentielle

Contrairement aux canaux `agg` et `fast` qui agrègent sur une fenêtre temporelle, le canal `frame` est **différentiel** :

- `probes` et `counts` sont vidés après chaque appel à `snapshot_frame()`.
- `gauges` reste un état instantané (pas de vidage).
- Pour reconstituer un cumulatif depuis le début de session, sommer les `counts` ligne à ligne en post-analyse.

### 5.4 Sondes exclues

Mêmes exclusions que §3.3 (`fast_*` et `bench_writer_*`).

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

| Champ   | Description                                                      |
| ------- | ---------------------------------------------------------------- |
| `avg`   | Moyenne arithmétique des mesures sur la fenêtre.                 |
| `max`   | Valeur maximale observée.                                        |
| `min`   | Valeur minimale observée.                                        |
| `count` | Nombre de mesures agrégées (toujours ≥ 1 si la sonde est émise). |

**Invariant** : une probe avec `count == 0` n'est **jamais** émise — elle est filtrée à la source par les méthodes `snapshot_*()`.

### 6.2 `gauges`

État instantané — dernière valeur posée via `bench.gauge(name, value)`.

```json
"<gauge_name>": <float>
```

Aucun agrégat : la valeur publiée est celle au moment du snapshot.

### 6.3 `rates`

Débit moyen sur la fenêtre du canal.

```json
"<rate_name>": <float>
```

- Pour `agg` : `rate = total_count / agg.interval_s`.
- Pour `fast` : `rate = total_count / history_window_s`.

**Invariant** : un rate de valeur `0.0` n'est **jamais** émis — filtré à la source.

### 6.4 `counts`

Compteur différentiel sur la fenêtre entre deux appels à `snapshot_frame()`.

```json
"<count_name>": <int>
```

La valeur est remise à zéro après chaque appel à `snapshot_frame()` — elle n'est donc **pas** cumulative depuis le démarrage de session. Pour reconstituer un cumulatif en post-analyse, sommer les `counts` ligne à ligne dans le fichier `bench_frame_*.jsonl`.

**Invariant** : un count de valeur `0` n'est **jamais** émis — filtré à la source.

---

## 7. Matrice des sections par canal

| Section  | `frame` | `agg` | `fast` |
| -------- | ------- | ----- | ------ |
| `probes` | ✅      | ✅    | ✅     |
| `gauges` | ✅      | ✅    | ✅     |
| `counts` | ✅      | ❌    | ❌     |
| `rates`  | ❌      | ✅    | ✅     |

Légende :

- ✅ = section autorisée (présente si non vide).
- ❌ = section interdite (rejetée par `_validate_snap` côté writer).

---

## 8. Règles d'évolution

1. **Ajout d'un champ** dans une section existante (`probes`, `gauges`, `rates`, `counts`) → autorisé sans bump si les consommateurs existants l'ignorent silencieusement.
2. **Renommage / suppression** d'un champ → bump `schema_version` obligatoire.
3. **Ajout d'une nouvelle section** dans un canal → mise à jour de §7 (matrice) + §9 (référence d'implémentation) + `_ALLOWED_SECTIONS` dans `jsonl_writer.py`.
4. **Ajout d'un nouveau canal** → nouveau bloc §X dédié + mise à jour §1 et §7.
5. **Changement de sémantique** (ex. `counts` cumulatif au lieu de différentiel) → bump `schema_version` obligatoire + note explicite dans §11.

---

## 9. Référence d'implémentation

- **Producteur unique des lignes JSONL** : `bench/jsonl_writer.py`, méthode `_enqueue()`.
- **Producteurs des snapshots** : `bench/bench.py`, méthodes `snapshot_all()` (canal `agg`), `snapshot_frame()` (canal `frame`), `snapshot_fast()` (canal `fast`).
- **Filtre défensif §7** : `jsonl_writer.py::_validate_snap()` rejette toute section non autorisée pour le canal + tout snap non conforme structurellement.

### Conventions d'auto-instrumentation

- Les sondes, compteurs et **gauges** dont le nom commence par `bench_writer_` sont des auto-mesures internes des writers JSONL.
- **Convention** : les writers JSONL n'émettent **jamais** de gauge. Seules des `probe()` (taille queue) et des `count()` (drops, rejets) sont autorisées sous ce préfixe.
- Conséquence : le filtre `_is_writer_probe` n'est appliqué qu'aux sections `probes`, `rates` et `counts`. La section `gauges` n'a pas de filtre explicite car aucune gauge writer ne doit exister par construction.
- **Toute évolution introduisant une gauge writer requiert** : (a) mise à jour de cette convention, (b) ajout du filtre `_is_writer_probe` aux blocs `gauges` de `snapshot_all` et `snapshot_frame`.

Toute divergence entre ce document et l'implémentation est un bug de l'un ou de l'autre — la résolution est arbitrée par l'équipe avant merge.

---

## 10. Paramètres de configuration

Tous les paramètres ci-dessous sont définis dans `config/config.yaml` sous la clé racine `debug.bench`.

### 10.1 Activation globale

| Clé YAML                       | Type  | Défaut | Effet sur le JSONL                                                |
| ------------------------------ | ----- | ------ | ----------------------------------------------------------------- |
| `debug.bench.enabled`          | bool  | —      | Si `false` : `BenchRegistry` désactivé, aucune ligne JSONL émise. |
| `debug.bench.history_window_s` | float | `60.0` | Fenêtre glissante en mémoire pour `snapshot_fast` et `summary()`. |

### 10.2 Writers JSONL — paramètres communs

| Clé YAML                                | Type   | Défaut          | Effet sur le JSONL                                                    |
| --------------------------------------- | ------ | --------------- | --------------------------------------------------------------------- |
| `debug.bench.writer.enabled`            | bool   | —               | Maître : si `false`, aucun fichier `.jsonl` créé.                     |
| `debug.bench.writer.queue_maxsize`      | int    | `10000`         | Taille max de la queue producteur→fichier.                            |
| `debug.bench.writer.shutdown_timeout_s` | float  | `2.0`           | Délai max accordé à chaque writer pour drain à l'arrêt.               |
| `debug.bench.writer.session_id_format`  | string | `%Y%m%d_%H%M%S` | Format `session_id` injecté dans chaque ligne et dans le nom fichier. |

### 10.3 Canaux

| Clé YAML                       | Type   | Défaut                        | Effet sur le JSONL                                                    |
| ------------------------------ | ------ | ----------------------------- | --------------------------------------------------------------------- |
| `debug.bench.agg.enabled`      | bool   | —                             | Active le canal `agg`.                                                |
| `debug.bench.agg.path`         | string | `logs/json/bench_agg.jsonl`   | Chemin de base — `session_id` inséré avant l'extension.               |
| `debug.bench.agg.interval_s`   | float  | `1.0`                         | Période entre deux snapshots agrégés. Définit la fenêtre des `rates`. |
| `debug.bench.frame.enabled`    | bool   | —                             | Active le canal `frame`.                                              |
| `debug.bench.frame.path`       | string | `logs/json/bench_frame.jsonl` | Chemin de base — `session_id` inséré avant l'extension.               |
| `debug.bench.frame.interval_s` | float  | (ignoré)                      | **Inutilisé** — le canal `frame` est piloté par `push_frame()`.       |
| `debug.bench.fast.enabled`     | bool   | —                             | Active le canal `fast`.                                               |
| `debug.bench.fast.path`        | string | `logs/json/bench_fast.jsonl`  | Chemin de base — `session_id` inséré avant l'extension.               |
| `debug.bench.fast.interval_s`  | float  | `1.0`                         | Période entre deux snapshots fast.                                    |

> **Note** : les clés `debug.bench.compare.*` (`buckets`, `shape`, `anomalies`, `frame_budget`) ne sont **pas** lues par le runtime bench. Elles sont la propriété exclusive de l'outil de post-analyse `bench_compare.py` et n'ont aucun impact sur la production des fichiers JSONL. Voir `bench-compare.md`.

---
