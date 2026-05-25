# 📋 Plan séquentiel autoporté S4 — Bucketing adaptatif cold/hot avec synchro coulante

> Document de référence consolidé. À utiliser comme point d'entrée unique pour reprendre l'implémentation S4 sans relire l'historique.

--- Je compte ajouter des calculs d'Interquartile Range de Skewness et du Kurtosis

## 🎯 Objectif

Remplacer le bucketing actuel (tiers fixes) par un bucketing **adaptatif** qui distingue :

- Une phase **cold** (démarrage, montée en charge) — durée variable, sync coulante
- N phases **hot** (régime établi) — durée nominale 10 s, frontières flexibles avec snap pivot
- Un éventuel **tail** (résidu) — fin de session incomplète

Garantir l'**absence de pollution croisée** entre buckets sans exclure de données.

---

## 🔑 Décisions verrouillées (rappel synthétique)

| Aspect              | Décision                                                                                               |
| ------------------- | ------------------------------------------------------------------------------------------------------ |
| Synchro fin de cold | **Wait-for-all** : `max(next_agg, next_fast) + ε` après la cible théorique                             |
| Cascade             | Cold coulant, hot_i rigides avec zone tampon ±0.5 s                                                    |
| Drift cold          | Garde-fou 3.0 s → warning si dépassé, on continue                                                      |
| Fast désactivé      | Détection auto, sync sur agg uniquement                                                                |
| Frontières hot_i    | Flexibles ±0.5 s, snap pivot si trouvé, sinon coupe stricte                                            |
| Pivot — définition  | Option γ : instant le plus proche de T avec écart ≥ `min_gap_s`                                        |
| `min_gap_s`         | 0.1 s                                                                                                  |
| Configurabilité     | Niveau B — tout dans `config.yaml` sous `debug.bench.compare.buckets.*`                                |
| Échec global pivot  | Silencieux (cas nominal généralisé)                                                                    |
| Métadonnées cold    | Minimal : `cold_end_target_s`, `cold_end_real_s`, `cold_drift_s`, `cold_drift_warning`, `fast_enabled` |
| Marqueur hot_i      | `is_pivot_snapped: true/false` par hot_i                                                               |

---

## 📐 Configuration cible (`config.yaml`)

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

---

## 🧮 Algorithme complet (pseudocode autoporté)

```text
ENTRÉES :
  - timeline_agg   : liste [mono, frame_idx] triée par mono
  - timeline_fast  : liste [mono, frame_idx] triée par mono (vide si fast désactivé)
  - timeline_frame : liste [mono, frame_idx] triée par mono (dense)
  - cfg.buckets.*  : constantes depuis config.yaml

SORTIE :
  - cold        : { mono_range_s, duration_s }
  - hot         : liste [{ index, mono_range_s, duration_s, is_pivot_snapped }]
  - tail        : { mono_range_s, duration_s, is_partial: true } | None
  - sync_meta   : { cold_end_target_s, cold_end_real_s, cold_drift_s,
                    cold_drift_warning, fast_enabled }

ÉTAPE 1 — Bornes globales
  t_min = min(timeline_frame[0].mono, timeline_agg[0].mono, [timeline_fast[0].mono])
  t_max = max(timeline_frame[-1].mono, timeline_agg[-1].mono, [timeline_fast[-1].mono])
  fast_enabled = (timeline_fast non vide)

ÉTAPE 2 — Calcul cold_end_réel (wait-for-all)
  t_target = t_min + cold_target_s
  next_agg  = premier événement agg  avec mono > t_target  (sinon +∞)
  next_fast = premier événement fast avec mono > t_target  (sinon +∞ si fast_enabled, ignoré sinon)

  IF fast_enabled:
      cold_end_real = max(next_agg, next_fast) + epsilon_s
  ELSE:
      cold_end_real = next_agg + epsilon_s

  cold_drift = cold_end_real - t_target
  cold_drift_warning = (cold_drift > max_cold_drift_s)
  IF cold_drift_warning:
      log.warning("cold_drift_exceeded: drift=%.3fs > max=%.3fs", cold_drift, max_cold_drift_s)

  bucket_cold = [t_min, cold_end_real)

ÉTAPE 3 — Génération hot_i (i = 0, 1, 2, ...)
  t_cursor = cold_end_real
  i = 0
  hot_list = []

  WHILE t_cursor + hot_duration_s <= t_max:
      T_theorique = t_cursor + hot_duration_s

      # Recherche pivot — Option γ
      window_start = T_theorique - boundary_guard_s
      window_end   = T_theorique + boundary_guard_s

      # Fusion ordonnée des événements agg + fast dans la fenêtre
      events_in_window = merge_sorted(
          [e for e in timeline_agg  if window_start <= e.mono <= window_end],
          [e for e in timeline_fast if window_start <= e.mono <= window_end]
      )

      # Recherche candidats : instants respectant min_gap_s avec voisins
      candidates = []
      FOR chaque instant t* candidat dans window:
          # Critère γ : écart >= min_gap_s avec événement précédent ET suivant
          gap_prev = t* - (dernier événement agg/fast avant t*, ou -∞)
          gap_next = (premier événement agg/fast après t*, ou +∞) - t*
          IF gap_prev >= min_gap_s AND gap_next >= min_gap_s:
              candidates.append(t*)

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
      cold_end_target_s: t_target - t_min,    # relatif au début session
      cold_end_real_s:   cold_end_real - t_min,
      cold_drift_s:      cold_drift,
      cold_drift_warning: cold_drift_warning,
      fast_enabled:      fast_enabled
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
      "fast_enabled": true
    },
    "cold": {
      "mono_range_s": [0.0, 5.732],
      "duration_s": 5.732,
      "probes": { ... },
      "fast_probes": { ... },
      "rates": { ... },
      "fast_rates": { ... },
      "gauges": { ... }
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
        "gauges": { ... }
      }
    ],
    "tail": {
      "mono_range_s": [55.732, 58.120],
      "duration_s": 2.388,
      "is_partial": true,
      "probes": { ... }
    }
  }
}
```

---

## 🪜 Plan d'implémentation séquentiel

### Étape 1 — Configuration (préalable)

**Fichier** : `config/config.yaml`

- Ajouter la section `debug.bench.compare.buckets` avec les 6 constantes.
- Documenter chaque clé en commentaire YAML.

**Fichier** : `bench/compare/_config.py` (ou équivalent)

- Ajouter une fonction `load_bucket_config()` qui lit la section et retourne un dataclass `BucketConfig` (frozen).
- Validations dans `__post_init__` :
  - `cold_target_s > 0`
  - `hot_duration_s > 0`
  - `max_cold_drift_s >= 0`
  - `boundary_guard_s >= 0`
  - `min_gap_s >= 0` et `min_gap_s < boundary_guard_s` (sinon aucun pivot ne peut être trouvé)
  - `epsilon_s >= 0`

### Étape 2 — Module de bucketing (cœur)

**Fichier** : `bench/compare/_bucketing.py` (à créer ou refactorer)

- Fonction pure `compute_buckets(timeline_agg, timeline_fast, timeline_frame, cfg) -> BucketsResult`
- `BucketsResult` = dataclass avec : `cold`, `hot` (list), `tail` (optional), `sync_metadata`
- Sous-fonctions internes :
  - `_compute_cold_end(t_min, timeline_agg, timeline_fast, cfg) -> (cold_end, drift, warning)`
  - `_find_pivot(T_theorique, timeline_agg, timeline_fast, cfg) -> (frontière, is_snapped)`
  - `_generate_hot_buckets(cold_end, t_max, timelines, cfg) -> list`

### Étape 3 — Agrégation par bucket

**Fichier** : `bench/compare/_session.py` (ou équivalent)

- Adapter la fonction qui agrège les sondes/rates/gauges par bucket pour :
  - Itérer sur `cold`, chaque `hot_i`, et `tail` si présent
  - Filtrer les lignes JSONL par `mono_range_s` de chaque bucket
  - Produire les sous-blocs `probes` / `fast_probes` / `rates` / `fast_rates` / `gauges`

### Étape 4 — Construction du bloc session

**Fichier** : `bench/compare/_session.py` (suite)

- Assembler le bloc `buckets` selon la structure JSON cible.
- Injecter `sync_metadata` au niveau approprié.

### Étape 5 — Adaptation rapport comparatif

**Fichier** : `bench/compare/_report.py` (ou équivalent)

- ⚠️ **Questionnement subsistant Q-Cadrage-1** : granularité des deltas entre sessions à trancher (voir §Questions ci-dessous).

### Étape 6 — Tests / validation manuelle

- Lancer sur une session existante et vérifier :
  - `cold_drift` reste < `max_cold_drift_s` dans le cas nominal
  - `is_pivot_snapped` est `true` pour la majorité des hot_i
  - Aucune pollution croisée visible (sondes cohérentes par bucket)
- Cas limites à tester :
  - Session très courte (< 5 s) → uniquement `cold` partiel
  - Session de 7 s → `cold` + `tail`, pas de `hot`
  - Fast désactivé → sync sur agg uniquement, pas de `fast_probes` ni `fast_rates`
  - Session ultra-dense → frontières hot_i toutes rigides (silencieux)

---

## ❓ Questionnements subsistants

### Q-Cadrage-1 — Granularité des deltas inter-sessions (non tranchée)

Comment comparer les buckets entre deux sessions A et B dans le rapport ?

- **P1** — Deltas par bucket aligné (`cold` vs `cold`, `hot_0` vs `hot_0`, …, jusqu'à `min(N_A, N_B)`). Buckets non alignés listés à part.
- **P2** — Deltas sur `cold` + agrégat "tous les hot_i confondus" (moyenne pondérée par durée).
- **P3** — Hybride : `cold` + `hot_0` + `hot_last` + agrégat "hot_middle".

**Recommandation** : P1 (granularité fine cohérente avec l'esprit du projet).

### Q-Fichiers — Contenu actuel des modules `bench/compare/`

Pour produire le patch sans hallucination, contenu actuel nécessaire de :

1. `bench/compare/_config.py`
2. `bench/compare/_bucketing.py` (ou module équivalent gérant les tiers actuels)
3. `bench/compare/_session.py`
4. `bench/compare/_report.py`
5. Section `debug.bench.compare` actuelle de `config/config.yaml`

Si l'arborescence diffère, fournir la structure réelle de `bench/compare/`.

### Q-Détail-1 — Génération des candidats pivot (point d'implémentation)

Dans l'algorithme, l'ensemble des "instants candidats t\*" dans la fenêtre `[T-0.5, T+0.5]` doit être défini concrètement. Deux approches :

- **D1** — Échantillonnage discret : on teste t\* = T, T±0.05, T±0.1, …, T±0.5 (pas configurable, ex. 0.05 s) et on garde ceux respectant le critère γ.
- **D2** — Analytique : on calcule directement les intervalles vides entre événements agg/fast dans la fenêtre, et on cherche dans chaque intervalle ≥ `2 × min_gap_s` l'instant le plus proche de T.

**Recommandation** : D2 (exact, sans paramètre supplémentaire, performances équivalentes).

À trancher avant implémentation de `_find_pivot`.

### Q-Détail-2 — Comportement si `cold_end_real > t_max`

Cas pathologique : session très courte ou trous massifs → `next_agg` ou `next_fast` n'existe pas après `t_target`.

- Comportement actuel non spécifié.
- **Proposition** : si `next_agg == +∞` ou (`fast_enabled` et `next_fast == +∞`) → `cold_end_real = t_max`, pas de `hot`, pas de `tail`, warning émis.

À valider.

---

## ✅ Critère de complétion S4

L'étape S4 est considérée terminée quand :

1. `config.yaml` contient la section `debug.bench.compare.buckets` documentée.
2. Le module de bucketing produit la structure JSON cible pour une session réelle.
3. Le rapport comparatif affiche les deltas selon la granularité tranchée en Q-Cadrage-1.
4. Les 4 cas limites listés en Étape 6 sont vérifiés manuellement sans erreur.
5. Aucune régression sur les sessions déjà analysées (snapshot avant/après comparable).

---

**Prochaine action attendue** : trancher Q-Cadrage-1, Q-Détail-1, Q-Détail-2 et fournir les fichiers listés en Q-Fichiers pour démarrer le patch.
