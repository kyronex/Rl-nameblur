# 📋 Plan séquentiel autoporté S4 — Bucketing adaptatif cold/hot avec synchro coulante ✅ **Livré**

> Document de référence consolidé (**rev 4**). À utiliser comme point d'entrée unique pour reprendre l'implémentation S4 sans relire l'historique.
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

| Aspect                       | Décision                                                                                                           |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Synchro fin de cold          | **Wait-for-all** : `max(next_agg, next_fast) + ε` après la cible théorique                                         |
| Cascade                      | Cold coulant, hot_i rigides avec zone tampon ±0.5 s                                                                |
| Drift cold                   | Garde-fou 3.0 s → warning si dépassé, on continue                                                                  |
| Fast désactivé               | Détection auto (`timeline_fast` vide), sync sur agg uniquement                                                     |
| Frontières hot_i             | Flexibles ±0.5 s, snap pivot si trouvé, sinon coupe stricte à `T_theorique`                                        |
| Pivot — définition           | Option γ : instant le plus proche de T avec écart ≥ `min_gap_s`                                                    |
| Pivot — génération           | **D2** — Analytique : intervalles vides ≥ `2 × min_gap_s`, instant le plus proche de T dans chacun                 |
| `min_gap_s`                  | 0.1 s                                                                                                              |
| Configurabilité              | Niveau B — tout dans `config.yaml` sous `debug.bench.compare.buckets.*`                                            |
| **Convention conf**          | **Constantes module-level** `BUCKET_*` dans `_config.py` via `_get(...)` (aligné sur l'existant, pas de dataclass) |
| **Timeline**                 | `list[dict]` avec clés `{ts, mono}` (sortie directe de `_extract_timeline` — pas de `frame_idx`)                   |
| **Borne bucket**             | `[t_start, t_end)` exclusive à droite (pas de double-comptage frontière)                                           |
| **Stratégie agrég.**         | **_Filter + reuse_** : helper `_filter_rows_by_mono` + agrégateurs `_agg_*` existants inchangés                    |
| Échec global pivot           | Silencieux (cas nominal généralisé)                                                                                |
| Deltas inter-sess.           | **P1** — Par bucket aligné (`cold` vs `cold`, `hot_i` vs `hot_i` jusqu'à `min(N_A, N_B)`)                          |
| `unaligned_hot`              | Indices seuls (pas de stats répétées — déjà présentes dans `target`/`reference`)                                   |
| `appeared/disappeared`       | Maintenus au niveau session uniquement (pas de duplication par bucket)                                             |
| Métadonnées cold             | `cold_end_target_s`, `cold_end_real_s`, `cold_drift_s`, `cold_drift_warning`, `cold_truncated`, `fast_enabled`     |
| Marqueur hot_i               | `is_pivot_snapped: true/false` par hot_i                                                                           |
| `temporal_events` par bucket | **Absent** — trop bruité à l'échelle d'un bucket                                                                   |
| `duration_s` par bucket      | Fourni par `BucketsResult`, **pas recalculé** depuis les lignes                                                    |
| `frames` par bucket          | Présent : `{"agg": int, "frame": int, "fast": int}` — compte des lignes filtrées par canal                         |
| `include_buckets`            | **Non** — `buckets` toujours calculé et injecté (pas de paramètre optionnel)                                       |
| `tail_status` valeurs        | `"aligned"` \| `"appeared"` \| `"disappeared"` \| `"absent_both"`                                                  |
| Cas pathologique             | Voir §Comportements aux bornes ci-dessous                                                                          |

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
        cold_target_s: 5.0 # Cible théorique fin de phase cold (s)
        hot_duration_s: 10.0 # Durée nominale d'un bucket hot_i (s)
        max_cold_drift_s: 3.0 # Seuil warning si cold_end_real dépasse cold_target_s de plus de N s
        boundary_guard_s: 0.5 # Demi-largeur fenêtre recherche pivot hot_i (s)
        min_gap_s: 0.1 # Écart minimal entre un événement et la frontière pour être pivot valide (s)
        epsilon_s: 0.001 # ε de sécurité synchro wait-for-all (s)
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
  - cold      : BucketSpec
  - hot       : list[BucketSpec]
  - tail      : BucketSpec | None
  - sync_meta : dict {cold_end_target_s, cold_end_real_s, cold_drift_s,
                      cold_drift_warning, cold_truncated, fast_enabled}

ÉTAPE 1 — Bornes globales
  Fusionner les trois timelines pour obtenir t_min et t_max :
    all_monos = [e["mono"] for e in timeline_agg + timeline_fast + timeline_frame]
    t_min = min(all_monos)
    t_max = max(all_monos)
  fast_enabled = (timeline_fast non vide)

ÉTAPE 2 — Calcul cold_end_réel (wait-for-all)
  t_target = t_min + BUCKET_COLD_TARGET_S
  next_agg  = premier e["mono"] dans timeline_agg  avec mono > t_target  (sinon +∞)
  next_fast = premier e["mono"] dans timeline_fast avec mono > t_target  (sinon +∞ si fast_enabled, ignoré sinon)

  # Détection Cas C — cold_end indéterminable → early return
  cold_truncated = false
  IF next_agg == +∞ OR (fast_enabled AND next_fast == +∞):
      cold_end_real = t_max
      cold_truncated = true
      cold_drift = t_max - t_target
      cold_drift_warning = true
      log.warning("cold_truncated: session too short or sparse")
      RETURN BucketsResult(
          cold = BucketSpec(mono_range_s=(t_min, t_max), duration_s=t_max-t_min, ...),
          hot  = [],
          tail = None,
          sync_metadata = { cold_end_target_s: t_target-t_min, cold_end_real_s: t_max-t_min,
                             cold_drift_s: cold_drift, cold_drift_warning: true,
                             cold_truncated: true, fast_enabled: fast_enabled }
      )

  # Cas nominal
  IF fast_enabled:
      cold_end_real = max(next_agg, next_fast) + BUCKET_EPSILON_S
  ELSE:
      cold_end_real = next_agg + BUCKET_EPSILON_S

  cold_drift = cold_end_real - t_target
  cold_drift_warning = (cold_drift > BUCKET_MAX_COLD_DRIFT_S)
  IF cold_drift_warning:
      log.warning("cold_drift_exceeded: drift=%.3fs > max=%.3fs", cold_drift, BUCKET_MAX_COLD_DRIFT_S)

  bucket_cold = BucketSpec(mono_range_s=(t_min, cold_end_real), duration_s=cold_end_real-t_min, ...)

ÉTAPE 3 — Génération hot_i (i = 0, 1, 2, ...)
  t_cursor = cold_end_real
  i = 0
  hot_list = []
  # Fusion mono agg + fast, triés — calculée une seule fois
  merged_events = sorted(
      [e["mono"] for e in timeline_agg] + [e["mono"] for e in timeline_fast]
  )

  WHILE t_cursor + BUCKET_HOT_DURATION_S <= t_max:
      T_theorique = t_cursor + BUCKET_HOT_DURATION_S

      # Recherche pivot — Option γ / méthode D2 analytique
      window_start = T_theorique - BUCKET_BOUNDARY_GUARD_S
      window_end   = T_theorique + BUCKET_BOUNDARY_GUARD_S

      # D2 — Recherche pivot analytique dans [window_start, window_end]
      # Bornes virtuelles : dernier événement avant la fenêtre, premier après
      ev_before = max([e for e in merged_events if e < window_start], default=-inf)
      ev_after  = min([e for e in merged_events if e > window_end],   default=+inf)
      events_in_window = [e for e in merged_events if window_start <= e <= window_end]
      bornes = [ev_before] + events_in_window + [ev_after]

      # Énumération des intervalles vides suffisants
      candidates = []
      FOR k in range(len(bornes) - 1):
          gap_start = bornes[k]
          gap_end   = bornes[k+1]
          valid_lo  = max(gap_start + BUCKET_MIN_GAP_S, window_start)
          valid_hi  = min(gap_end   - BUCKET_MIN_GAP_S, window_end)
          IF valid_lo <= valid_hi:
              t_star = clamp(T_theorique, valid_lo, valid_hi)
              candidates.append(t_star)

      IF candidates non vide:
          frontière       = candidat minimisant |candidat - T_theorique|
          is_pivot_snapped = true
      ELSE:
          frontière        = T_theorique
          is_pivot_snapped = false

      hot_list.append(BucketSpec(
          mono_range_s     = (t_cursor, frontière),
          duration_s       = frontière - t_cursor,
          index            = i,
          is_pivot_snapped = is_pivot_snapped,
      ))
      t_cursor = frontière
      i += 1

ÉTAPE 4 — Tail (résidu)
  IF t_max - t_cursor > 0:
      tail = BucketSpec(
          mono_range_s = (t_cursor, t_max),
          duration_s   = t_max - t_cursor,
          is_partial   = true,
      )
  ELSE:
      tail = None

ÉTAPE 5 — Assemblage sync_metadata
  sync_meta = {
      "cold_end_target_s":  t_target - t_min,
      "cold_end_real_s":    cold_end_real - t_min,
      "cold_drift_s":       cold_drift,
      "cold_drift_warning": cold_drift_warning,
      "cold_truncated":     cold_truncated,
      "fast_enabled":       fast_enabled,
  }

ÉTAPE 6 — Log récapitulatif (Q-Détail-5 → oui, niveau INFO)
  log.info("buckets: cold=%.2fs, hot=%d (snapped=%d/%d), tail=%s",
      cold_end_real - t_min,
      len(hot_list),
      sum(b.is_pivot_snapped for b in hot_list),
      len(hot_list),
      f"{t_max - t_cursor:.2f}s" if tail else "none"
  )
```

---

## 📦 Structure JSON cible

### Bloc session (par session)

```json
{
  "duration_s": 58.3,
  "duration_mono_s": 58.1,
  "frames": { "agg": 58, "frame": 1748, "fast": 58 },
  "temporal_events": { "...": "..." },
  "probes": { "...": "..." },
  "rates": { "...": "..." },
  "gauges": { "...": "..." },
  "fast_probes": { "...": "..." },
  "fast_rates": { "...": "..." },
  "fast_gauges": { "...": "..." },
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
      "frames": { "agg": 5, "frame": 172, "fast": 5 },
      "probes": { "...": "..." },
      "rates": { "...": "..." },
      "gauges": { "...": "..." },
      "fast_probes": { "...": "..." },
      "fast_rates": { "...": "..." },
      "fast_gauges": { "...": "..." }
    },
    "hot": [
      {
        "index": 0,
        "mono_range_s": [5.732, 15.689],
        "duration_s": 9.957,
        "is_pivot_snapped": true,
        "frames": { "agg": 10, "frame": 300, "fast": 10 },
        "probes": { "...": "..." },
        "rates": { "...": "..." },
        "gauges": { "...": "..." },
        "fast_probes": { "...": "..." },
        "fast_rates": { "...": "..." },
        "fast_gauges": { "...": "..." }
      }
    ],
    "tail": {
      "mono_range_s": [55.732, 58.12],
      "duration_s": 2.388,
      "is_partial": true,
      "frames": { "agg": 2, "frame": 72, "fast": 2 },
      "probes": { "...": "..." },
      "rates": { "...": "..." },
      "gauges": { "...": "..." },
      "fast_probes": { "...": "..." },
      "fast_rates": { "...": "..." },
      "fast_gauges": { "...": "..." }
    }
  }
}
```

### Bloc deltas (rapport comparatif, P1)

```json
{
  "deltas": {
    "buckets": {
      "cold": {
        "duration_delta_s": 0.123,
        "probes": { "...": "..." },
        "rates": { "...": "..." },
        "gauges": { "...": "..." },
        "fast_probes": { "...": "..." },
        "fast_rates": { "...": "..." },
        "fast_gauges": { "...": "..." }
      },
      "hot": [
        {
          "index": 0,
          "duration_delta_s": 0.05,
          "probes": { "...": "..." },
          "rates": { "...": "..." },
          "gauges": { "...": "..." },
          "fast_probes": { "...": "..." },
          "fast_rates": { "...": "..." },
          "fast_gauges": { "...": "..." }
        }
      ],
      "unaligned_hot": {
        "target_only": [2, 3],
        "ref_only": []
      },
      "tail_status": "aligned",
      "tail": {
        "duration_delta_s": -0.41,
        "probes": { "...": "..." },
        "rates": { "...": "..." },
        "gauges": { "...": "..." },
        "fast_probes": { "...": "..." },
        "fast_rates": { "...": "..." },
        "fast_gauges": { "...": "..." }
      }
    }
  }
}
```

> `tail` est présent dans les deltas uniquement si `tail_status == "aligned"`. Pour les autres valeurs (`"appeared"`, `"disappeared"`, `"absent_both"`), la clé `tail` est absente.

---

## 🪜 Plan d'implémentation séquentiel

### Étape 1 — Configuration

**Fichier** : `config/config.yaml`

Ajouter après le bloc `fast:`, avant la fermeture de `bench:` :

```yaml
# ── Analyse comparative (bench_compare.py) ───────────────────
compare:
  buckets:
    cold_target_s: 5.0 # Cible théorique fin de phase cold (s)
    hot_duration_s: 10.0 # Durée nominale d'un bucket hot_i (s)
    max_cold_drift_s: 3.0 # Seuil warning si cold_end_real dépasse cold_target_s de plus de N s
    boundary_guard_s: 0.5 # Demi-largeur fenêtre recherche pivot hot_i (s)
    min_gap_s: 0.1 # Écart minimal entre un événement et la frontière pour être pivot valide (s)
    epsilon_s: 0.001 # ε de sécurité synchro wait-for-all (s)
```

**Fichier** : `bench/compare/_config.py`

Ajouter après la section "Seuils détection des gaps temporels" :

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

assert BUCKET_COLD_TARGET_S > 0,     "cold_target_s doit être > 0"
assert BUCKET_HOT_DURATION_S > 0,    "hot_duration_s doit être > 0"
assert BUCKET_MAX_COLD_DRIFT_S >= 0, "max_cold_drift_s doit être >= 0"
assert BUCKET_BOUNDARY_GUARD_S >= 0, "boundary_guard_s doit être >= 0"
assert 0 <= BUCKET_MIN_GAP_S < BUCKET_BOUNDARY_GUARD_S, \
    "min_gap_s doit être dans [0, boundary_guard_s["
assert BUCKET_EPSILON_S >= 0,        "epsilon_s doit être >= 0"
```

> ❌ **Pas de dataclass `BucketConfig`** — incohérent avec la convention de `_config.py` (constantes module-level).

---

### Étape 2 — Module de bucketing (cœur, nouveau)

**Fichier** : `bench/compare/_bucketing.py` (à créer)

Module **purement fonctionnel**, sans I/O.

**Dataclasses** (définies dans ce module) :

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class BucketSpec:
    mono_range_s:     tuple[float, float]
    duration_s:       float
    index:            int | None  = None   # None pour cold/tail
    is_pivot_snapped: bool | None = None   # None pour cold/tail
    is_partial:       bool        = False  # True uniquement pour tail
    cold_truncated:   bool        = False  # True uniquement pour cold (Cas C)

@dataclass(frozen=True)
class BucketsResult:
    cold:          BucketSpec
    hot:           tuple[BucketSpec, ...]   # tuple pour frozen
    tail:          BucketSpec | None
    sync_metadata: dict                     # non-frozen toléré : dict immuable en pratique
    fast_enabled:  bool
```

**Signature publique** :

```python
def compute_buckets(
    timeline_agg:   list[dict],   # [{ts, mono}, ...] triée par mono
    timeline_fast:  list[dict],   # idem, vide si fast désactivé
    timeline_frame: list[dict],   # idem, dense
) -> BucketsResult:
```

**Sous-fonctions privées** :

| Fonction                | Signature                                                                                                | Rôle                                                                                          |
| ----------------------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | --------------------------- |
| `_merged_events_sorted` | `(timeline_agg, timeline_fast) -> list[float]`                                                           | Fusion mono agg+fast triés, calculée une fois                                                 |
| `_compute_cold_end`     | `(t_min, t_max, t_target, timeline_agg, timeline_fast, fast_enabled) -> tuple[float, float, bool, bool]` | Retourne `(cold_end_real, drift, drift_warning, truncated)` — implémente wait-for-all + Cas C |
| `_find_pivot`           | `(T_theorique, merged_events) -> tuple[float, bool]`                                                     | Stratégie D2 analytique. Retourne `(frontière, is_snapped)`                                   |
| `_generate_hot_buckets` | `(cold_end, t_max, merged_events) -> list[BucketSpec]`                                                   | Itère hot_i avec pivot snap                                                                   |
| `_generate_tail`        | `(t_cursor, t_max) -> BucketSpec                                                                         | None`                                                                                         | Génère tail si reliquat > 0 |

**Constantes** : importées depuis `bench.compare._config` (`BUCKET_*`).

---

### Étape 3 — Helper de filtrage par bucket

**Fichier** : `bench/compare/_stats.py` (extension)

Ajouter **une seule fonction** en fin de fichier, avant les éventuels exports :

```python
def _filter_rows_by_mono(
    rows: list[dict],
    t_start: float,
    t_end: float,
) -> list[dict]:
    """
    Retourne le sous-ensemble de `rows` dont le champ `mono` est dans
    [t_start, t_end). Borne droite exclusive — cohérent avec BucketSpec.
    Les lignes sans champ `mono` numérique sont ignorées silencieusement.
    rows doit être trié par mono croissant (invariant garanti par les writers).
    """
    return [
        r for r in rows
        if isinstance(r.get("mono"), (int, float))
        and t_start <= r["mono"] < t_end
    ]
```

> ✅ Les agrégateurs `_agg_probes`, `_agg_rates`, `_agg_gauges`, `_collect_frame_samples`, `_collect_fast_approx_samples` restent **inchangés**. Ils sont appelés sur les sous-ensembles filtrés.

---

### Étape 4 — Construction du bloc `buckets` dans `_builder.py`

**Fichier** : `bench/compare/_builder.py`

**Imports à ajouter** :

```python
from bench.compare._bucketing import BucketsResult, BucketSpec, compute_buckets
from bench.compare._stats     import _filter_rows_by_mono
```

**Modification de `build_session_block`** — ajouter `"buckets"` au `return` existant, sans toucher aux autres clés :

```python
# ➕ Calcul buckets (ajouté après le code existant, avant return)
buckets_result = compute_buckets(timeline_agg, timeline_fast, timeline_frame)
buckets_block  = _build_buckets_block(buckets_result, agg_rows, frame_rows, fast_rows)

return {
    # ... toutes les clés existantes inchangées ...
    "buckets": buckets_block,   # ➕ NOUVEAU
}
```

> `timeline_agg`, `timeline_fast`, `timeline_frame` sont déjà calculés dans `build_session_block` — pas de recalcul.

**Nouvelle fonction `_build_buckets_block`** :

```python
def _build_buckets_block(
    result:     BucketsResult,
    agg_rows:   list[dict],
    frame_rows: list[dict],
    fast_rows:  list[dict],
) -> dict:
    """
    Construit le bloc `buckets` : sync_metadata + cold + hot[] + tail.
    Filtre les 3 canaux par mono_range_s pour chaque BucketSpec.
    """
    cold_dict = _build_single_bucket(result.cold, agg_rows, frame_rows, fast_rows)

    hot_list = []
    for spec in result.hot:
        b = _build_single_bucket(spec, agg_rows, frame_rows, fast_rows)
        b["index"]            = spec.index
        b["is_pivot_snapped"] = spec.is_pivot_snapped
        hot_list.append(b)

    tail_dict = None
    if result.tail is not None:
        tail_dict = _build_single_bucket(result.tail, agg_rows, frame_rows, fast_rows)
        tail_dict["is_partial"] = True

    return {
        "sync_metadata": result.sync_metadata,
        "cold": cold_dict,
        "hot":  hot_list,
        "tail": tail_dict,
    }
```

**Nouvelle fonction `_build_single_bucket`** :

```python
def _build_single_bucket(
    spec:       BucketSpec,
    agg_rows:   list[dict],
    frame_rows: list[dict],
    fast_rows:  list[dict],
) -> dict:
    """
    Agrège un bucket unique.
    Structure : {mono_range_s, duration_s, frames,
                 probes, rates, gauges,
                 fast_probes, fast_rates, fast_gauges}
    Pas de temporal_events (trop bruité à l'échelle d'un bucket).
    duration_s provient de spec.duration_s (pas recalculé depuis les lignes).
    """
    t_start, t_end = spec.mono_range_s

    f_agg   = _filter_rows_by_mono(agg_rows,   t_start, t_end)
    f_frame = _filter_rows_by_mono(frame_rows, t_start, t_end)
    f_fast  = _filter_rows_by_mono(fast_rows,  t_start, t_end)

    base_probes_agg  = _agg_probes(f_agg)
    base_probes_fast = _agg_probes(f_fast)
    rates_agg   = _agg_rates(f_agg)
    rates_fast  = _agg_rates(f_fast)
    gauges_agg  = _agg_gauges(f_agg)
    gauges_fast = _agg_gauges(f_fast)

    exact_samples, frame_approx = _collect_frame_samples(f_frame)
    fast_approx   = _collect_fast_approx_samples(f_fast)
    approx_samples = {**frame_approx, **fast_approx}

    probes: dict[str, dict] = {}
    for probe_name, stats in base_probes_agg.items():
        pct_block = _build_percentile_block(probe_name, exact_samples, approx_samples, channel="agg")
        probes[probe_name] = {
            "avg": _r(stats["avg"]),
            "min": _r(stats["min"]),
            "max": _r(stats["max"]),
            "count_agg": stats["count_agg"],
            **{k: _r(v) for k, v in pct_block.items()},
        }

    fast_probes: dict[str, dict] = {}
    for probe_name, stats in base_probes_fast.items():
        pct_block = _build_percentile_block(probe_name, exact_samples, approx_samples, channel="fast")
        fast_probes[probe_name] = {
            "avg": _r(stats["avg"]),
            "min": _r(stats["min"]),
            "max": _r(stats["max"]),
            "count_fast": stats["count_agg"],
            **{k: _r(v) for k, v in pct_block.items()},
        }

    return {
        "mono_range_s": list(spec.mono_range_s),
        "duration_s":   _r(spec.duration_s),
        "frames": {
            "agg":   len(f_agg),
            "frame": len(f_frame),
            "fast":  len(f_fast),
        },
        "probes":      probes,
        "rates":       {k: _r(v) for k, v in rates_agg.items()},
        "gauges":      {k: _r(v) for k, v in gauges_agg.items()},
        "fast_probes": fast_probes,
        "fast_rates":  {k: _r(v) for k, v in rates_fast.items()},
        "fast_gauges": {k: _r(v) for k, v in gauges_fast.items()},
    }
```

---

### Étape 5 — Adaptation rapport comparatif (P1)

**Fichier** : `bench/compare/_builder.py`

**Modification de `build_comparison`** — ajouter `"buckets"` dans `result["deltas"]`, sans toucher aux autres clés :

```python
def build_comparison(ref_session_id, ref_block, target_block) -> dict:
    # ... code existant inchangé ...
    return {
        # ... toutes les clés existantes inchangées ...
        "deltas": {
            # ... clés existantes inchangées ...
            "buckets": _build_buckets_deltas(   # ➕ NOUVEAU
                target_block.get("buckets"),
                ref_block.get("buckets"),
            ),
        },
        # ... reste inchangé ...
    }
```

**Nouvelle fonction `_build_buckets_deltas`** :

```python
def _build_buckets_deltas(
    target_buckets: dict | None,
    ref_buckets:    dict | None,
) -> dict | None:
    """
    Deltas P1 : cold vs cold, hot[i] vs hot[i] jusqu'à min(N_target, N_ref).
    Réutilise _build_probe_deltas et _build_scalar_deltas par bucket aligné.
    Ajoute duration_delta_s par bucket aligné.

    Retourne None si l'un ou l'autre bloc buckets est absent
    (session pré-S4 ou Cas C côté référence).
    """
    if target_buckets is None or ref_buckets is None:
        return None

    def _bucket_deltas(t: dict, r: dict) -> dict:
        return {
            "duration_delta_s": _r((t["duration_s"] or 0) - (r["duration_s"] or 0)),
            "probes":      _build_probe_deltas(t["probes"],      r["probes"]),
            "rates":       _build_scalar_deltas(t["rates"],      r["rates"]),
            "gauges":      _build_scalar_deltas(t["gauges"],     r["gauges"]),
            "fast_probes": _build_probe_deltas(t["fast_probes"], r["fast_probes"]),
            "fast_rates":  _build_scalar_deltas(t["fast_rates"], r["fast_rates"]),
            "fast_gauges": _build_scalar_deltas(t["fast_gauges"],r["fast_gauges"]),
        }

    # cold — toujours présent
    cold_deltas = _bucket_deltas(target_buckets["cold"], ref_buckets["cold"])

    # hot — aligné par index
    t_hot = target_buckets.get("hot", [])
    r_hot = ref_buckets.get("hot",    [])
    n_aligned = min(len(t_hot), len(r_hot))

    hot_deltas = []
    for i in range(n_aligned):
        d = _bucket_deltas(t_hot[i], r_hot[i])
        d["index"] = i
        hot_deltas.append(d)

    unaligned_hot = {
        "target_only": list(range(n_aligned, len(t_hot))),
        "ref_only":    list(range(n_aligned, len(r_hot))),
    }

    # tail — statut + deltas si aligné
    t_tail = target_buckets.get("tail")
    r_tail = ref_buckets.get("tail")

    if t_tail is not None and r_tail is not None:
        tail_status = "aligned"
        tail_deltas = _bucket_deltas(t_tail, r_tail)
    elif t_tail is not None:
        tail_status = "appeared"
        tail_deltas = None
    elif r_tail is not None:
        tail_status = "disappeared"
        tail_deltas = None
    else:
        tail_status = "absent_both"
        tail_deltas = None

    result = {
        "cold":          cold_deltas,
        "hot":           hot_deltas,
        "unaligned_hot": unaligned_hot,
        "tail_status":   tail_status,
    }
    if tail_deltas is not None:
        result["tail"] = tail_deltas

    return result
```

---

### Étape 6 — Tests / validation manuelle

Cas limites à vérifier :

| #   | Scénario                              | Résultat attendu                                                |
| --- | ------------------------------------- | --------------------------------------------------------------- |
| 1   | Session < 5 s                         | `cold_truncated = true`, `hot = []`, `tail = null`              |
| 2   | Session = 7 s                         | `cold` + `tail`, `hot = []`                                     |
| 3   | Fast désactivé                        | Sync sur agg uniquement, blocs `fast_*` à `{}` (clés présentes) |
| 4   | Session ultra-dense                   | `is_pivot_snapped = false` partout (silencieux)                 |
| 5   | Sessions A/B longueurs différentes    | `unaligned_hot` rempli, deltas alignés sur `min(N_A, N_B)`      |
| 6   | Session pré-S4 (pas de clé `buckets`) | `_build_buckets_deltas` retourne `None` sans erreur             |

---

## ❓ Questionnements résiduels

### ✅ Tous tranchés

| Q           | Décision                                                                        |
| ----------- | ------------------------------------------------------------------------------- |
| Q-Cadrage-1 | **P1** — deltas par bucket aligné                                               |
| Q-Détail-1  | **D2** — analytique                                                             |
| Q-Détail-2  | 3 cas A/B/C avec `cold_truncated`                                               |
| Q-Détail-3  | Borne `[t_start, t_end)` exclusive à droite                                     |
| Q-Détail-4  | `unaligned_hot` en indices seuls                                                |
| Q-Détail-5  | Log INFO récapitulatif dans `compute_buckets` → **oui**                         |
| Q-Détail-6  | Pas de `tail_status: "aligned_warning"` en S4 — delta `duration_delta_s` suffit |
| Q-Fichiers  | Tous lus. Pas de `_session.py` ni `_report.py` — rôles dans `_builder.py`       |

---

## ✅ Critère de complétion S4

1. `config.yaml` contient la section `debug.bench.compare.buckets` (6 clés documentées).
2. `_config.py` expose les 6 constantes `BUCKET_*` avec validation `assert`.
3. `_bucketing.py` produit un `BucketsResult` correct pour une session réelle.
4. `_stats.py` expose `_filter_rows_by_mono`.
5. `_builder.py` injecte `"buckets"` dans `build_session_block` et `"buckets"` dans `build_comparison["deltas"]`.
6. Les 6 cas limites de l'Étape 6 sont vérifiés manuellement sans erreur.
7. Aucune régression : les clés existantes au niveau session sont inchangées.

---

## 📜 Historique des révisions

| Rev | Changements                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Document initial — 3 questionnements ouverts                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| 2   | Q-Cadrage-1 → P1 / Q-Détail-1 → D2 / Q-Détail-2 → 3 cas A/B/C                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 3   | Objectif reformulé "Introduire". Convention constantes module-level. Timeline `{ts,mono}`. Borne `[start,end)`. Filter + reuse. `unaligned_hot` indices seuls. Noms réels de fichiers. Q-Détail-3/4 tranchés.                                                                                                                                                                                                                                                                               |
| 4   | Analyse `_builder.py` complète. Fonctions réutilisables identifiées. `temporal_events` absent par bucket. `duration_s` depuis `BucketsResult`. `include_buckets` supprimé (toujours calculé). `BucketSpec`/`BucketsResult` dataclasses définies. Signatures sous-fonctions `_bucketing.py` complètes. Corps de `_build_single_bucket` et `_build_buckets_deltas` fournis. Cas 6 (session pré-S4) ajouté. Q-Détail-5/6 tranchés. `tail` absent des deltas si `tail_status ≠ "aligned"` acté. |

---

**Prochaine action attendue** : implémentation séquentielle Étape 1 → 6, fichier par fichier, en commençant par `config.yaml` puis `_config.py`.
