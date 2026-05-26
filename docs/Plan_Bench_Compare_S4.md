# 📋 Plan séquentiel autoporté S4 — Bucketing adaptatif cold/hot avec synchro coulante

> Document de référence consolidé (**rev 3**). À utiliser comme point d'entrée unique pour reprendre l'implémentation S4 sans relire l'historique.
>
> **Évolution prévue** : ajout ultérieur de calculs d'Interquartile Range (S4bis), Skewness et Kurtosis (S5).

---

## 🎯 Objectif

**Introduire** un bucketing **adaptatif** qui distingue :

- Une phase **cold** (démarrage, montée en charge) — durée variable, sync coulante
- N phases **hot** (régime établi) — durée nominale 10 s, frontières flexibles avec snap pivot
- Un éventuel **tail** (résidu) — fin de session incomplète

Garantir l'**absence de pollution croisée** entre buckets sans exclure de données.

> **Note de cadrage** : aucun bucketing n'existe actuellement dans `bench/compare/`. `build_session_block` agrège sur la session entière. S4 ajoute donc une **nouvelle dimension d'analyse** (clé `buckets` dans le bloc session), sans remplacer ni rompre la rétro-compatibilité du schéma existant.

---

## 🔑 Décisions verrouillées (rappel synthétique)

| Aspect                 | Décision                                                                                                           |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Synchro fin de cold    | **Wait-for-all** : `max(next_agg, next_fast) + ε` après la cible théorique                                         |
| Cascade                | Cold coulant, hot_i rigides avec zone tampon ±0.5 s                                                                |
| Drift cold             | Garde-fou 3.0 s → warning si dépassé, on continue                                                                  |
| Fast désactivé         | Détection auto (`timeline_fast` vide), sync sur agg uniquement                                                     |
| Frontières hot_i       | Flexibles ±0.5 s, snap pivot si trouvé, sinon coupe stricte à `T_theorique`                                        |
| Pivot — définition     | Option γ : instant le plus proche de T avec écart ≥ `min_gap_s`                                                    |
| Pivot — génération     | **D2** — Analytique : intervalles vides ≥ `2 × min_gap_s`, instant le plus proche de T dans chacun                 |
| `min_gap_s`            | 0.1 s                                                                                                              |
| Configurabilité        | Niveau B — tout dans `config.yaml` sous `debug.bench.compare.buckets.*`                                            |
| **Convention conf**    | **Constantes module-level** `BUCKET_*` dans `_config.py` via `_get(...)` (aligné sur l'existant, pas de dataclass) |
| **Timeline**           | `list[dict]` avec clés `{ts, mono}` (sortie directe de `_extract_timeline` — pas de `frame_idx`)                   |
| **Borne bucket**       | `[t_start, t_end)` exclusive à droite (pas de double-comptage frontière)                                           |
| **Stratégie agrég.**   | **_Filter + reuse_** : helper `_filter_rows_by_mono` + agrégateurs `_agg_*` existants inchangés                    |
| Échec global pivot     | Silencieux (cas nominal généralisé)                                                                                |
| Deltas inter-sess.     | **P1** — Par bucket aligné (`cold` vs `cold`, `hot_i` vs `hot_i` jusqu'à `min(N_A, N_B)`)                          |
| `unaligned_hot`        | Indices seuls (pas de stats répétées — déjà présentes dans `target`/`reference`)                                   |
| `appeared/disappeared` | Maintenus au niveau session uniquement (pas de duplication par bucket)                                             |
| Métadonnées cold       | `cold_end_target_s`, `cold_end_real_s`, `cold_drift_s`, `cold_drift_warning`, `cold_truncated`, `fast_enabled`     |
| Marqueur hot_i         | `is_pivot_snapped: true/false` par hot_i                                                                           |
| Cas pathologique       | Voir §Comportements aux bornes ci-dessous                                                                          |

---

## 🧭 Comportements aux bornes (Q-Détail-2 consolidé)

Trois cas distincts, traités séparément :

### Cas A — Échec pivot sur frontière hot*i / hot*{i+1}

- Aucun candidat ne respecte `min_gap_s` dans `[T-0.5, T+0.5]`
- → **Cut strict à `T_theorique`**, `is_pivot_snapped = false`
- Traité dans l'algo §Étape 3

### Cas B — Fin de fichier atteinte

- `t_cursor + hot_duration_s > t_max` (matière insuffisante pour un hot_i complet)
- → Reliquat devient **`tail`** avec `is_partial = true`
- Si `t_cursor == t_max` (frontière pivot tombée exactement sur la fin) → pas de tail
- Traité dans l'algo §Étape 4

### Cas C — `cold_end_real` indéterminable

- `next_agg == +∞` après `t_target`, OU (`fast_enabled` ET `next_fast == +∞`)
- → `cold_end_real = t_max`
- → **Toute la session = `cold`** (partiel), pas de `hot[]`, pas de `tail`
- → `sync_metadata.cold_truncated = true` + `cold_drift_warning = true`
- → Warning loggé : `"cold_truncated: session too short or sparse"`

> **Règle invariante** : le 1er bucket est **toujours** `cold` (jamais `hot_0` direct), même si Cas C.

---

## 📐 Configuration cible (`config.yaml`)

Section à ajouter sous `debug.bench`, au même niveau que `agg` / `frame` / `fast` / `writer` :

```yaml
debug:
  bench:
    # ... agg / frame / fast / writer existants ...

    # ── Bucketing adaptatif cold/hot (S4) ──────────────────────
    compare:
      buckets:
        cold_target_s: 5.0 # cible théorique fin cold
        hot_duration_s: 10.0 # durée nominale hot_i
        max_cold_drift_s: 3.0 # garde-fou drift cold (warning si dépassé)
        boundary_guard_s: 0.5 # demi-largeur fenêtre recherche pivot hot_i
        min_gap_s: 0.1 # écart minimal pour qu'un instant soit pivot valide
        epsilon_s: 0.001 # ε de sécurité synchro wait-for-all
```

---

## 🧮 Algorithme complet (pseudocode autoporté)

```text
ENTRÉES :
  - timeline_agg   : list[dict] {ts, mono} triée par mono (sortie _extract_timeline)
  - timeline_fast  : list[dict] {ts, mono} triée par mono (vide si fast désactivé)
  - timeline_frame : list[dict] {ts, mono} triée par mono (dense)
  - constantes BUCKET_* importées depuis bench.compare._config

SORTIE :
  - cold        : { mono_range_s, duration_s }
  - hot         : liste [{ index, mono_range_s, duration_s, is_pivot_snapped }]
  - tail        : { mono_range_s, duration_s, is_partial: true } | None
  - sync_meta   : { cold_end_target_s, cold_end_real_s, cold_drift_s,
                    cold_drift_warning, cold_truncated, fast_enabled }

ÉTAPE 1 — Bornes globales
  t_min = min(timeline_frame[0].mono, timeline_agg[0].mono, [timeline_fast[0].mono])
  t_max = max(timeline_frame[-1].mono, timeline_agg[-1].mono, [timeline_fast[-1].mono])
  fast_enabled = (timeline_fast non vide)

ÉTAPE 2 — Calcul cold_end_réel (wait-for-all)
  t_target = t_min + BUCKET_COLD_TARGET_S
  next_agg  = premier événement agg  avec mono > t_target  (sinon +∞)
  next_fast = premier événement fast avec mono > t_target  (sinon +∞ si fast_enabled, ignoré sinon)

  # Détection Cas C — cold_end indéterminable
  cold_truncated = false
  IF next_agg == +∞ OR (fast_enabled AND next_fast == +∞):
      cold_end_real = t_max
      cold_truncated = true
      cold_drift = t_max - t_target
      cold_drift_warning = true
      log.warning("cold_truncated: session too short or sparse")
      RETURN { cold: [t_min, t_max], hot: [], tail: None, sync_meta: {..., cold_truncated: true} }

  IF fast_enabled:
      cold_end_real = max(next_agg, next_fast) + BUCKET_EPSILON_S
  ELSE:
      cold_end_real = next_agg + BUCKET_EPSILON_S

  cold_drift = cold_end_real - t_target
  cold_drift_warning = (cold_drift > BUCKET_MAX_COLD_DRIFT_S)
  IF cold_drift_warning:
      log.warning("cold_drift_exceeded: drift=%.3fs > max=%.3fs", cold_drift, BUCKET_MAX_COLD_DRIFT_S)

  bucket_cold = [t_min, cold_end_real)

ÉTAPE 3 — Génération hot_i (i = 0, 1, 2, ...)
  t_cursor = cold_end_real
  i = 0
  hot_list = []
  merged_events = merge_sorted(   # fusion mono agg + fast, triés
      [e["mono"] for e in timeline_agg],
      [e["mono"] for e in timeline_fast]
  )

  WHILE t_cursor + BUCKET_HOT_DURATION_S <= t_max:
      T_theorique = t_cursor + BUCKET_HOT_DURATION_S

      # Recherche pivot — Option γ / méthode D2 analytique
      window_start = T_theorique - BUCKET_BOUNDARY_GUARD_S
      window_end   = T_theorique + BUCKET_BOUNDARY_GUARD_S

      # D2 — Énumération analytique des intervalles vides ≥ 2 × min_gap_s
      # Borne virtuelle "avant" = dernier événement < window_start (ou -∞)
      # Borne virtuelle "après" = premier événement > window_end (ou +∞)
      events_in_window = [e for e in merged_events if window_start <= e <= window_end]
      ev_before = max([e for e in merged_events if e < window_start], default=-∞)
      ev_after  = min([e for e in merged_events if e > window_end],   default=+∞)
      bornes = [ev_before] + events_in_window + [ev_after]

      candidates = []
      FOR k in range(len(bornes) - 1):
          gap_start = bornes[k]
          gap_end   = bornes[k+1]
          # Intervalle utile = [gap_start + min_gap_s, gap_end - min_gap_s] ∩ window
          valid_lo = max(gap_start + BUCKET_MIN_GAP_S, window_start)
          valid_hi = min(gap_end   - BUCKET_MIN_GAP_S, window_end)
          IF valid_lo <= valid_hi:
              # Instant le plus proche de T_theorique dans cet intervalle valide
              t_star = clamp(T_theorique, valid_lo, valid_hi)
              candidates.append(t_star)

      IF candidates non vide:
          pivot = candidat minimisant |pivot - T_theorique|
          frontière = pivot
          is_pivot_snapped = true
      ELSE:
          frontière = T_theorique
          is_pivot_snapped = false

      hot_list.append({
          index: i,
          mono_range_s: [t_cursor, frontière],
          duration_s: frontière - t_cursor,
          is_pivot_snapped: is_pivot_snapped
      })
      t_cursor = frontière
      i += 1

ÉTAPE 4 — Tail (résidu)
  IF t_max - t_cursor > 0:
      tail = {
          mono_range_s: [t_cursor, t_max],
          duration_s: t_max - t_cursor,
          is_partial: true
      }
  ELSE:
      tail = None

ÉTAPE 5 — Assemblage sync_metadata
  sync_meta = {
      cold_end_target_s:  t_target - t_min,    # relatif au début session
      cold_end_real_s:    cold_end_real - t_min,
      cold_drift_s:       cold_drift,
      cold_drift_warning: cold_drift_warning,
      cold_truncated:     cold_truncated,
      fast_enabled:       fast_enabled
  }
```

---

## 📦 Structure JSON cible (par session)

```json
{
  "session_id": "...",
  "frames": { "first": ..., "last": ..., "count": ... },
  "buckets": {
    "sync_metadata": {
      "cold_end_target_s": 5.0,
      "cold_end_real_s": 5.732,
      "cold_drift_s": 0.732,
      "cold_drift_warning": false,
      "cold_truncated": false,
      "fast_enabled": true
    },
    "cold": {
      "mono_range_s": [0.0, 5.732],
      "duration_s": 5.732,
      "probes": { ... },
      "fast_probes": { ... },
      "rates": { ... },
      "fast_rates": { ... },
      "gauges": { ... },
      "fast_gauges": { ... }
    },
    "hot": [
      {
        "index": 0,
        "mono_range_s": [5.732, 15.689],
        "duration_s": 9.957,
        "is_pivot_snapped": true,
        "probes": { ... },
        "fast_probes": { ... },
        "rates": { ... },
        "fast_rates": { ... },
        "gauges": { ... },
        "fast_gauges": { ... }
      }
    ],
    "tail": {
      "mono_range_s": [55.732, 58.120],
      "duration_s": 2.388,
      "is_partial": true,
      "probes": { ... },
      "fast_probes": { ... },
      "rates": { ... },
      "fast_rates": { ... },
      "gauges": { ... },
      "fast_gauges": { ... }
    }
  }
}
```

### Structure JSON deltas (rapport comparatif, P1)

```json
{
  "deltas": {
    "buckets": {
      "cold": {
        "duration_delta_s": 0.123,
        "probes": { ... },
        "rates": { ... },
        "gauges": { ... },
        "fast_probes": { ... },
        "fast_rates": { ... },
        "fast_gauges": { ... }
      },
      "hot": [
        { "index": 0, "duration_delta_s": 0.05, "probes": {...}, ... },
        { "index": 1, "duration_delta_s": -0.02, "probes": {...}, ... }
      ],
      "unaligned_hot": {
        "target_only": [2, 3],
        "ref_only": []
      },
      "tail": { ... } | null,
      "tail_status": "aligned" | "appeared" | "disappeared" | "absent_both"
    }
  }
}
```

---

## 🪜 Plan d'implémentation séquentiel

### Étape 1 — Configuration

**Fichier** : `config/config.yaml`

- Ajouter la section `debug.bench.compare.buckets` avec les 6 constantes (cf. §Configuration cible).
- Documenter chaque clé en commentaire YAML (style aligné `# ── … ──`).

**Fichier** : `bench/compare/_config.py`

- Ajouter, **après** la section "Seuils détection des gaps temporels", une nouvelle section :

  ```python
  # ---------------------------------------------------------------------------
  # Bucketing adaptatif cold/hot (S4)
  # ---------------------------------------------------------------------------

  BUCKET_COLD_TARGET_S    = _get("debug.bench.compare.buckets.cold_target_s",    5.0)
  BUCKET_HOT_DURATION_S   = _get("debug.bench.compare.buckets.hot_duration_s",  10.0)
  BUCKET_MAX_COLD_DRIFT_S = _get("debug.bench.compare.buckets.max_cold_drift_s", 3.0)
  BUCKET_BOUNDARY_GUARD_S = _get("debug.bench.compare.buckets.boundary_guard_s", 0.5)
  BUCKET_MIN_GAP_S        = _get("debug.bench.compare.buckets.min_gap_s",        0.1)
  BUCKET_EPSILON_S        = _get("debug.bench.compare.buckets.epsilon_s",        0.001)
  ```

- **Validation au chargement** : ajouter à la fin du fichier un bloc `assert` :

  ```python
  assert BUCKET_COLD_TARGET_S > 0,        "cold_target_s doit être > 0"
  assert BUCKET_HOT_DURATION_S > 0,       "hot_duration_s doit être > 0"
  assert BUCKET_MAX_COLD_DRIFT_S >= 0,    "max_cold_drift_s doit être >= 0"
  assert BUCKET_BOUNDARY_GUARD_S >= 0,    "boundary_guard_s doit être >= 0"
  assert 0 <= BUCKET_MIN_GAP_S < BUCKET_BOUNDARY_GUARD_S, \
      "min_gap_s doit être dans [0, boundary_guard_s["
  assert BUCKET_EPSILON_S >= 0,           "epsilon_s doit être >= 0"
  ```

> ❌ **Pas de dataclass `BucketConfig`** — incohérent avec la convention de `_config.py` (constantes module-level).

### Étape 2 — Module de bucketing (cœur, nouveau)

**Fichier** : `bench/compare/_bucketing.py` (à créer)

Module **purement fonctionnel**, sans I/O. Une seule fonction publique.

**Signature publique** :

```python
def compute_buckets(
    timeline_agg:   list[dict],   # [{ts, mono}, ...] triée par mono
    timeline_fast:  list[dict],   # idem, vide si fast désactivé
    timeline_frame: list[dict],   # idem, dense
) -> BucketsResult
```

**Type de retour** : `BucketsResult` = dataclass `frozen` exposant :

- `cold: BucketSpec`
- `hot: list[BucketSpec]`
- `tail: BucketSpec | None`
- `sync_metadata: dict`

Où `BucketSpec = dataclass(frozen=True)` avec champs :

- `mono_range_s: tuple[float, float]`
- `duration_s: float`
- `index: int | None` (None pour cold/tail, valeur entière pour hot_i)
- `is_pivot_snapped: bool | None` (None pour cold/tail)
- `is_partial: bool` (True uniquement pour tail)

**Sous-fonctions privées** :

| Fonction                                                                                          | Rôle                                                                           |
| ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `_compute_cold_end(t_min, t_max, timeline_agg, timeline_fast) -> tuple[float, float, bool, bool]` | Retourne `(cold_end_real, drift, warning, truncated)`.                         |
|                                                                                                   | Implémente wait-for-all + ε + garde-fou drift + Cas C.                         |
| `_find_pivot(T_theorique, merged_events) -> tuple[float, bool]`                                   | Implémente la stratégie **D2 analytique**. Retourne `(frontière, is_snapped)`. |
| `_generate_hot_buckets(cold_end, t_max, merged_events) -> list[BucketSpec]`                       | Itère hot_i avec pivot snap.                                                   |
| `_generate_tail(t_cursor, t_max) -> BucketSpec \| None`                                           | Génère tail si reste > 0.                                                      |
| `_merged_events_sorted(timeline_agg, timeline_fast) -> list[float]`                               | Helper : fusion des `mono` agg+fast triés.                                     |

**Constantes** : import depuis `_config.py` (`BUCKET_*`).

### Étape 3 — Helper de filtrage par bucket

**Fichier** : `bench/compare/_stats.py` (extension)

Ajouter **une seule fonction** :

```python
def _filter_rows_by_mono(rows: list[dict], t_start: float, t_end: float) -> list[dict]:
    """
    Retourne le sous-ensemble de `rows` dont le champ `mono` est dans
    [t_start, t_end). Borne droite exclusive pour garantir l'absence
    de double-comptage entre buckets adjacents.

    Les lignes sans `mono` numérique sont ignorées.
    """
    return [
        r for r in rows
        if isinstance(r.get("mono"), (int, float))
        and t_start <= r["mono"] < t_end
    ]
```

> ✅ Stratégie _filter + reuse_ : les agrégateurs `_agg_probes`, `_agg_rates`, `_agg_gauges`, `_collect_frame_samples`, `_collect_fast_approx_samples` restent **inchangés**. Ils sont appelés sur le sous-ensemble filtré.

### Étape 4 — Construction du bloc `buckets` dans la session

**Fichier** : `bench/compare/_builder.py` (modification de `build_session_block`)

**Stratégie** : étendre le `return` de `build_session_block` avec une clé `buckets`, **sans toucher** aux clés existantes (`probes`, `rates`, `gauges`, `fast_*`, `duration_s`, etc.) → préservation totale de la rétro-compatibilité du schéma session.

**Pseudocode du patch** :

```python
def build_session_block(agg_rows, frame_rows, fast_rows) -> dict:
    # ... code existant inchangé jusqu'au return ...

    # ➕ NOUVEAU : calcul des buckets
    timeline_agg   = _extract_timeline(agg_rows)
    timeline_fast  = _extract_timeline(fast_rows)
    timeline_frame = _extract_timeline(frame_rows)

    buckets_result = compute_buckets(timeline_agg, timeline_fast, timeline_frame)
    buckets_block = _build_buckets_block(
        buckets_result, agg_rows, frame_rows, fast_rows
    )

    return {
        # ... clés existantes ...
        "buckets": buckets_block,
    }
```

**Nouvelles fonctions privées dans `_builder.py`** :

```python
def _build_buckets_block(
    buckets_result: BucketsResult,
    agg_rows:   list[dict],
    frame_rows: list[dict],
    fast_rows:  list[dict],
) -> dict:
    """
    Construit le bloc `buckets` final avec sync_metadata + cold + hot[] + tail.
    Pour chaque bucket, filtre les rows par mono_range_s et applique les
    agrégateurs existants.
    """

def _build_single_bucket(
    spec: BucketSpec,
    agg_rows: list[dict],
    frame_rows: list[dict],
    fast_rows: list[dict],
) -> dict:
    """
    Pour un bucket donné, filtre les 3 canaux par mono_range_s et produit
    un dict {mono_range_s, duration_s, probes, rates, gauges,
    fast_probes, fast_rates, fast_gauges, + méta spécifiques}.
    Réutilise _build_percentile_block pour les sondes.
    """
```

> 💡 **Réutilisation maximale** : `_build_percentile_block` existant fonctionne tel quel pourvu qu'on lui passe les samples exact/approx filtrés par bucket.

### Étape 5 — Adaptation rapport comparatif (Q-Cadrage-1 → P1)

**Fichier** : `bench/compare/_builder.py` (modification de `build_comparison`)

**Granularité P1** : deltas par bucket aligné.

**Patch** :

```python
def build_comparison(ref_session_id, ref_block, target_block) -> dict:
    # ... code existant inchangé ...

    return {
        # ... clés existantes inchangées ...
        "deltas": {
            # ... clés existantes ...
            "buckets": _build_buckets_deltas(  # ➕ NOUVEAU
                target_block.get("buckets"),
                ref_block.get("buckets"),
            ),
        },
    }
```

**Nouvelle fonction** :

```python
def _build_buckets_deltas(target_buckets: dict, ref_buckets: dict) -> dict:
    """
    Compare buckets aligné par aligné (P1) :
      - cold vs cold (toujours présent des deux côtés)
      - hot[i] vs hot[i] pour i ∈ [0, min(N_target, N_ref))
      - hot non alignés listés dans `unaligned_hot` (indices seulement)
      - tail vs tail si les deux existent ; sinon `tail_status`:
        "aligned" | "appeared" | "disappeared" | "absent_both"

    Réutilise _build_probe_deltas et _build_scalar_deltas pour chaque
    bucket aligné, exactement comme au niveau session.
    Ajoute `duration_delta_s` par bucket aligné.
    """
```

**Conventions de sortie** :

- `appeared_probes` / `disappeared_probes` restent au niveau session (pas dupliqués par bucket)
- `unaligned_hot` ne contient que des indices, pas de stats (déjà présentes dans `target.buckets.hot` / `reference.buckets.hot`)
- Affichage en sortie texte (rapport humain) : flag `is_pivot_snapped` par hot_i + warnings `cold_drift_warning` / `cold_truncated` en en-tête de session

### Étape 6 — Tests / validation manuelle

- Lancer sur une session existante et vérifier :
  - `cold_drift` reste < `max_cold_drift_s` dans le cas nominal
  - `is_pivot_snapped` est `true` pour la majorité des hot_i
  - Aucune pollution croisée visible (sondes cohérentes par bucket)
- Cas limites à tester :
  - Session très courte (< 5 s) → uniquement `cold` partiel, `cold_truncated = true`
  - Session de 7 s → `cold` + `tail`, pas de `hot`
  - Fast désactivé (config) → sync sur agg uniquement, blocs `fast_*` à `{}` mais clés présentes
  - Session ultra-dense → frontières hot_i toutes rigides, `is_pivot_snapped = false` partout (silencieux)
  - Sessions A/B de longueurs différentes → deltas alignés + `unaligned_hot` rempli

---

## ❓ Questionnements résiduels

### ✅ Tous les questionnements précédents tranchés

- **Q-Cadrage-1** → **P1** (deltas par bucket aligné) — schéma détaillé §Structure JSON deltas
- **Q-Détail-1** → **D2** (analytique)
- **Q-Détail-2** → 3 cas séparés (A/B/C) avec flag `cold_truncated`
- **Q-Détail-3** → Borne `[t_start, t_end)` exclusive à droite
- **Q-Détail-4** → `unaligned_hot` en indices seuls
- **Q-Fichiers** → Tous les fichiers nécessaires lus (`_builder.py`, `_stats.py`, `_config.py`, section `config.yaml`). Pas de modules `_session.py` ni `_report.py` (rôles absorbés par `_builder.py`).

### Questionnements ouverts mineurs

- **Q-Détail-5** (à valider à l'implémentation) : faut-il logger un INFO récapitulatif en fin de `compute_buckets` (ex. `"buckets: cold=5.7s, hot=4 (snapped=3/4), tail=2.4s"`) ? Recommandation : oui, niveau INFO, utile au débogage.
- **Q-Détail-6** (à valider à l'implémentation) : `tail_status` du delta — si `target.tail` existe et `ref.tail` aussi mais leurs `duration_s` diffèrent fortement (ex. ratio > 2x), faut-il flaguer `tail_status: "aligned_warning"` ? Recommandation : non pour S4, simple delta sur `duration_delta_s` suffit.

---

## ✅ Critère de complétion S4

L'étape S4 est considérée terminée quand :

1. `config.yaml` contient la section `debug.bench.compare.buckets` documentée.
2. `_config.py` expose les 6 constantes `BUCKET_*` avec validation par `assert`.
3. `_bucketing.py` produit un `BucketsResult` correct pour une session réelle.
4. `_builder.py` injecte le bloc `buckets` dans la session et les deltas dans la comparaison (P1).
5. Les 5 cas limites listés en Étape 6 sont vérifiés manuellement sans erreur.
6. Aucune régression sur les sessions déjà analysées (clés existantes inchangées au niveau session).

---

## 📜 Historique des révisions

| Rev | Date    | Changements                                                                                                                                                     |
| --- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | initial | Document initial avec 3 questionnements ouverts (Q-Cadrage-1, Q-Détail-1, Q-Détail-2)                                                                           |
| 2   | —       | Q-Cadrage-1 → **P1** / Q-Détail-1 → **D2** / Q-Détail-2 → 3 cas séparés (A/B/C) avec flag `cold_truncated`. Algo enrichi (cas C en early return, D2 explicité). |
|     | —       | Étape 5 détaillée selon P1. Cas limite session A/B asymétriques ajouté.                                                                                         |
| 3   | actuel  | Analyse code (`_builder.py`, `_stats.py`, `_config.py`) → **objectif reformulé "Introduire" (pas "Remplacer")**.                                                |
|     |         | Convention conf = **constantes module-level** (pas dataclass). Timeline = `list[dict]{ts,mono}` sans `frame_idx`. Borne bucket `[t_start, t_end)` actée.        |
|     |         | Stratégie **filter + reuse** des agrégateurs existants. Structure deltas explicitée (`unaligned_hot` indices seuls, `tail_status` 4 valeurs).                   |
|     |         | Étapes 1→5 mises à jour avec noms réels de fichiers (`_builder.py` au lieu de `_session.py`/`_report.py` qui n'existent pas).                                   |
|     |         | Q-Détail-3/4 ajoutés et tranchés. Q-Détail-5/6 (mineurs) ouverts.                                                                                               |

---

**Prochaine action attendue** : implémentation séquentielle Étape 1 → 6, fichier par fichier, en commençant par `config.yaml` puis `_config.py`.
