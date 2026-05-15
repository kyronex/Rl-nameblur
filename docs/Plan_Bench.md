# 📋 Plan v5 — Post-audit B-04b

> **Objectif** : passer du backlog audité à un système instrumenté, propre et prêt pour B-04c.
>
> **Principe directeur** : séquentialité stricte, chaque lot indépendant testable, rollback possible à chaque borne. Aucun lot ne démarre tant que le précédent n'est pas validé.
>
> **Arbitrages v4 → v5** :
>
> - **3 canaux JSONL** (`bench_frame` + `bench_agg` + `bench_fast`) — R6 acté.
> - Config `debug.bench.*` restructurée avec `writer.*`, `queue_maxsize`, `shutdown_timeout_s`, `session_id_format` — R6 acté.
> - Schéma JSONL enrichi : nommage `session_id` + schéma `bench_fast` — R9-3 acté.
> - Sondes L3.8 / L3.9 conditionnées à un audit code base — R9-4 / R9-5 actés.
> - Harmonisation noms sondes `registry_*` — R9-6 acté.

---

## 🎯 Vue d'ensemble

| Lot    | Scope                                                               | Effort  | Précondition           |
| ------ | ------------------------------------------------------------------- | ------- | ---------------------- |
| **L0** | Fondations transverses (API bench + schéma JSON + migration config) | ~2 h 30 | Audit B-04b clôturé ✅ |
| **L1** | Arbitrage structurel : `created_ts` natif sur `Mask`                | ~30 min | L0 ✅                  |
| **L2** | Purge dettes legacy (3 blocs + arbitrage `tracker.stats()`)         | ~1 h    | L1                     |
| **L3** | Déploiement sondes différées (par fichier) + audit R9-4/R9-5        | ~4 h    | L2                     |
| **L4** | Validation intégration (smoke test 60 s)                            | ~30 min | L3                     |
| **L5** | Livraison B-04b : récap consolidé + bascule B-04c                   | ~30 min | L4                     |

**Effort total estimé** : ~9 h, étalable sur 2–3 sessions.

---

## 🔵 Lot L0 — Fondations transverses

> ⚠️ **Lot critique** : modifie l'API `bench`, la structure config et fige le schéma JSONL. Tout ce qui suit dépend de ces fondations. À traiter en premier, en un seul commit atomique.

### L0.1 — API `bench` : `snapshot_frame()` + `snapshot_all()` + writer JSONL ✅ **Livré**

- **Fichier** : `bench.py`
- **Actions** :
  - `snapshot_frame() -> dict[str, dict]` : snapshot **par frame** → `bench_frame_{session_id}.jsonl`.
  - `snapshot_all() -> dict[str, dict]` : snapshot **cumulatif glissant** → `bench_agg_{session_id}.jsonl`.
  - `snapshot_fast() -> dict[str, dict]` : snapshot **dédié FastTrackThread** → `bench_fast_{session_id}.jsonl`.
  - **3 writers JSONL** : threads dédiés, queue partagée bornée (`queue_maxsize`), flush ligne par ligne, rotation par `session_id`, shutdown propre (`shutdown_timeout_s`).
- **Contrat** :
  - Thread-safe (lock interne).
  - Idempotent sur snapshot vide (retourne `{}` ou domaines vides explicitement).
  - Pas de reset implicite — sémantique cumulative explicite pour `snapshot_all()`.
  - Groupement par domaine dérivé du **préfixe de nommage T-2** (`<domaine>_<action>[_<qualifieur>]` → domaine = premier segment).
  - Drop JSONL loggué WARNING si queue pleine.
  - Snapshot dynamique : liste sondes figée au premier `flush()` de la session (R1).
  - Sondes sans données dans la fenêtre → clé omise de la ligne JSONL (R2).
- **Validation** : tests unitaires (snapshot vide, snapshot non-vide, concurrence basique, cohérence groupement par domaine, drop queue pleine, shutdown propre).

---

### L0.2 — Décisions transverses actées ✅ **Livré**

- **T-1 (R6)** : 3 canaux JSONL :
  - `bench_frame_{session_id}.jsonl` (1 ligne / frame).
  - `bench_agg_{session_id}.jsonl` (1 ligne / `agg_interval`).
  - `bench_fast_{session_id}.jsonl` (1 ligne / `fast.interval_s`).
- **T-2 (nommage sondes)** : convention verrouillée `<domaine>_<action>[_<qualifieur>]`. Conditionne le groupement JSON.
- **T-3 (rétention)** : cumulatif depuis start pour `snapshot_all()` et `snapshot_fast()` ; snapshot frame indépendant pour `snapshot_frame()`.
- **T-4 (writers)** : 3 threads dédiés, `queue_maxsize` et `shutdown_timeout_s` communs, issus de `debug.bench.writer.*`.
- **Fichiers touchés** : `bench.py`, `main.py`.

---

### L0.3 — Migration config `debug.csv.*` → `debug.bench.*` ✅ **Livré**

> **R9-2 acté** : remplacement complet de la structure v4. `debug.csv.*` supprimé (hard cut).

- **Fichier** : `config.yaml`
- **Avant** : `debug.csv.*` (legacy mono-fichier).
- **Après** :

```yaml
debug:
  bench:
    writer:
      path: "logs/"
      session_id_format: "%Y%m%d_%H%M%S"
      queue_maxsize: 10000
      shutdown_timeout_s: 2.0
    frame:
      enabled: true
      include_masks: true
    agg:
      enabled: true
      interval_s: 1.0
    fast:
      enabled: true
      interval_s: 1.0
```

- **Validation** :
  - Démarrage sans crash.
  - 3 fichiers `bench_{frame|agg|fast}_{session_id}.jsonl` créés dans `writer.path`.
  - Aucune écriture CSV résiduelle.

---

### L0.4 — Schéma JSONL figé ✅ **Livré**

> **R9-3 acté** : ajout schéma `bench_fast` + nommage `session_id`.
> ⚠️ **Schéma immuable** : toute évolution post-L0 = nouveau ticket.

#### `bench_frame_{session_id}.jsonl` — 1 ligne / frame

```json
{
  "ts": 1713000000.123,
  "frame_id": 42,
  "main": {
    "capture_wait_ms": 4.2,
    "slow_poll_ms": 12.1,
    "match_ms": 0.8,
    "fast_poll_ms": 0.3,
    "predict_ms": 0.5,
    "blur_ms": 1.2,
    "send_ms": 0.4,
    "frames_total": 42
  },
  "tracker": {
    "masks_total": 3,
    "masks_confirmed": 2,
    "masks_pending": 1,
    "masks_lost": 0
  },
  "detect": {
    "slow_detections": 4,
    "slow_duration_ms": 11.8
  },
  "masks": [
    {
      "mask_id": "abc123",
      "state": "CONFIRMED",
      "created_ts": 1713000000.0,
      "lifetime_s": 0.123
    }
  ]
}
```

#### `bench_agg_{session_id}.jsonl` — 1 ligne / `interval_s`

```json
{
  "ts": 1713000001.0,
  "window_s": 1.0,
  "main": {
    "fps_mean": 62.3,
    "fps_p5": 48.1,
    "fps_p95": 74.2,
    "capture_wait_ms_mean": 4.1,
    "blur_ms_mean": 1.3
  },
  "tracker": {
    "registry_create_total": 2,
    "registry_expire_total": 1,
    "masks_confirmed_mean": 1.8
  },
  "motion": {
    "motion_dt_ms_mean": 16.2,
    "staleness_capped_total": 0
  },
  "detect": {
    "slow_detections_total": 18,
    "slow_duration_ms_mean": 11.5
  }
}
```

#### `bench_fast_{session_id}.jsonl` — 1 ligne / `fast.interval_s`

```json
{
  "ts": 1713000001.0,
  "window_s": 1.0,
  "fast": {
    "fast_wakeup_lag_ms_mean": 2.1,
    "fast_tick_ms_mean": 3.4,
    "fast_of_total": 120,
    "fast_ncc_total": 118,
    "fast_ncc_confirmed": 110,
    "fast_stale_used": 3,
    "fast_mask_lost": 1,
    "fast_poll_ms_mean": 0.3
  }
}
```

- **Validation** : parsing de chaque schéma par un script de test dédié, vérification clés obligatoires présentes.

---

### 🔒 Borne L0

- ✅ `snapshot_frame()` / `snapshot_all()` / `snapshot_fast()` implémentées et testées.
- ✅ 3 writers JSONL opérationnels (queue bornée, drop loggué, shutdown propre).
- ✅ Config `debug.bench.*` en place, `debug.csv.*` absent.
- ✅ Schéma L0.4 validé par script de parsing.
- ✅ Aucune régression démarrage application.
- ✅ Tag git `b04b-L0`.

---

## 🟢 Lot L1 — Arbitrage structurel `created_ts` ✅ **Livré**

> **R7 acté** : `created_ts: float` natif sur `Mask`, set unique dans `MaskRegistry.create()`, exposé dans `to_dict()`.

### L1.1 — Champ `created_ts` sur `Mask`

- **Fichier** : `core/mask.py`
- **Action** :

```python
@dataclass
class Mask:
    # ... champs existants ...
    created_ts: float = field(default_factory=time.perf_counter)
```

- **Contrat** :
  - Set **une seule fois** à la création dans `MaskRegistry.create()` via `perf_counter()`.
  - Jamais modifié après création.
  - Exposé dans `to_dict()` : `"created_ts": self.created_ts`.
- **Validation** : test unitaire (valeur non nulle, non modifiable après création, présente dans `to_dict()`).

### L1.2 — Intégration `MaskRegistry.create()`

- **Fichier** : `tracker/registry.py`
- **Action** : s'assurer que `create()` passe `created_ts=time.perf_counter()` à la construction du `Mask`. Supprimer tout calcul de `created_ts` hors de ce point.
- **Validation** : test unitaire (deux masks créés à des ts différents → `created_ts` distincts et ordonnés).

---

### 🔒 Borne L1

- ✅ `created_ts` présent sur `Mask`, set dans `registry.create()` uniquement.
- ✅ `to_dict()` expose `created_ts`.
- ✅ Tests unitaires L1.1 + L1.2 verts.
- ✅ Tag git `b04b-L1`.

---

## 🟡 Lot L2 — Purge dettes legacy ✅ **Livré**

> ⚠️ **Purge destructive** : 4 blocs supprimés. Commit atomique par dette, rollback par tag L1.

### L2.1 — Purge `_motion_stats` + `get_and_reset_stats()`

- **Fichier** : `tracker/motion.py`
- **Action** : supprimer `_motion_stats` dict interne et la méthode `get_and_reset_stats()`. Les sondes motion sont désormais portées par `bench.py` (L3.6).
- **Validation** : aucun appel résiduel dans le codebase (`grep -r "get_and_reset_stats"`).

### L2.2 — Purge `frame_count` / `fps_timer` legacy

- **Fichier** : `main.py`
- **Action** : supprimer les variables `frame_count`, `fps_timer` et leurs usages. Remplacer par lectures directes des sondes bench (`frames_total`, `fps_mean`).
- **Validation** : `grep -r "frame_count\|fps_timer"` → zéro résultat hors commentaires.

### L2.3 — Purge `_b01_stats`

- **Fichier** : à confirmer par audit (probable `tracker/registry.py`).
- **Action** : supprimer le bloc `_b01_stats` et ses points d'injection. Dette tracée explicitement si localisation incertaine.
- **Validation** : `grep -r "_b01_stats"` → zéro résultat.

### L2.4 — Arbitrage `tracker.stats()`

- **Fichier** : `tracker/tracker.py`
- **Action** : supprimer `tracker.stats()` et ses appels dans `main.py`. Les métriques exposées sont désormais portées par les sondes `registry_*` et `motion_*` de L3.
- **Validation** : `grep -r "tracker\.stats\(\)"` → zéro résultat.

---

### 🔒 Borne L2

- ✅ 4 dettes legacy purgées.
- ✅ Zéro appel résiduel (grep validé pour chaque dette).
- ✅ Démarrage application sans crash post-purge.
- ✅ Tag git `b04b-L2`.

---

## 🔴 Lot L3 — Déploiement sondes + audit R9-4/R9-5

> **L3 atomique vis-à-vis de L4** : L3.1 → L3.9 tous livrés avant L4.
> **L3.8 prioritaire** : lève 3 suspensions (L3.3, L3.4, L3.6).

---

### 🔍 Audit R9-4 / R9-5 — Pré-requis bloquant L3.8 et L3.9

> **Déclencheur** : avant toute implémentation de L3.8 (`fast_track_thread.py`) et L3.9 (`core/mask.py`), un audit code base est **obligatoire**.

#### Périmètre audit R9-4 (`fast_track_thread.py`)

- Lister exhaustivement toutes les fonctions du fichier.
- Pour chaque fonction : identifier les points de mesure pertinents (entrée, sortie, branche critique).
- Produire la liste des 8 sondes avec : nom (`<domaine>_<action>[_<qualifieur>]`), type (`probe`/`count`/`gauge`), emplacement exact (fonction + ligne).
- Vérifier l'absence de doublon avec les sondes `fast_*` déjà présentes dans le schéma L0.4.

#### Périmètre audit R9-5 (`core/mask.py`)

- Lister toutes les méthodes publiques et propriétés calculées.
- Identifier les métriques cycle de vie exposables (transitions d'état, durées).
- Produire la liste des 6 sondes avec : nom, type, emplacement exact.
- Vérifier cohérence avec `created_ts` L1 et `mask_lifetime_s` déjà dans le schéma L0.4.

#### Livrable audit

```text
┌─────────────────────────────────────────────────────┐
│ AUDIT R9-4 : fast_track_thread.py                   │
│ Sondes retenues : 8                                 │
│ [liste nom / type / emplacement]                    │
├─────────────────────────────────────────────────────┤
│ AUDIT R9-5 : core/mask.py                           │
│ Sondes retenues : 6                                 │
│ [liste nom / type / emplacement]                    │
└─────────────────────────────────────────────────────┘
```

> ⚠️ **L3.8 et L3.9 ne démarrent pas tant que ce livrable n'est pas validé.**

---

### L3.1 — Sondes `main.py` (8 sondes)

| Sonde                  | Type  | Emplacement                             |
| ---------------------- | ----- | --------------------------------------- |
| `main_capture_wait_ms` | probe | boucle principale — attente frame       |
| `main_slow_poll_ms`    | probe | boucle principale — poll detect thread  |
| `main_match_ms`        | probe | boucle principale — appel associator    |
| `main_fast_poll_ms`    | probe | boucle principale — poll fast thread    |
| `main_predict_ms`      | probe | boucle principale — predict positions   |
| `main_blur_ms`         | probe | boucle principale — rendu flou          |
| `main_send_ms`         | probe | boucle principale — envoi frame         |
| `main_frames_total`    | count | boucle principale — incrément par frame |

---

### L3.2 — Sondes `tracker/registry.py` (5 sondes)

> **R9-6 acté** : noms harmonisés `registry_*`.

| Sonde                | Type  | Emplacement         |
| -------------------- | ----- | ------------------- |
| `registry_create`    | count | `create()`          |
| `registry_expire`    | count | `expire()`          |
| `registry_confirmed` | gauge | `tick_and_expire()` |
| `registry_lost`      | gauge | `tick_and_expire()` |
| `registry_pending`   | gauge | `tick_and_expire()` |

---

### L3.3 — Sondes `tracker/tracker.py` (6 sondes)

| Sonde                     | Type  | Emplacement                      |
| ------------------------- | ----- | -------------------------------- |
| `tracker_tick_ms`         | probe | `tick()` — durée totale          |
| `tracker_detections_in`   | count | `tick()` — nb détections reçues  |
| `tracker_masks_total`     | gauge | `tick()` — total masks actifs    |
| `tracker_confirmed_total` | gauge | `get_confirmed_masks()`          |
| `tracker_lost_total`      | gauge | `tick()` — masks en état LOST    |
| `tracker_pending_total`   | gauge | `tick()` — masks en état PENDING |

---

### L3.4 — Sondes `detect/detect.py` (5 sondes)

| Sonde                     | Type  | Emplacement                                  |
| ------------------------- | ----- | -------------------------------------------- |
| `detect_slow_duration_ms` | probe | `detect_plates()` — durée totale             |
| `detect_slow_count`       | count | `detect_plates()` — nb détections retournées |
| `detect_ncc_match_ms`     | probe | `ncc_match()` — durée                        |
| `detect_ncc_score`        | probe | `ncc_match()` — score retourné               |
| `detect_boxes_filtered`   | count | pipeline filtrage — boxes rejetées           |

---

### L3.5 — Sondes `tracker/associator.py` (6 sondes)

| Sonde                       | Type  | Emplacement                                       |
| --------------------------- | ----- | ------------------------------------------------- |
| `associator_tick_ms`        | probe | `associate()` — durée totale                      |
| `associator_matched`        | count | `associate()` — nb matches retenus                |
| `associator_unmatched_det`  | count | `associate()` — détections non matchées           |
| `associator_unmatched_mask` | count | `associate()` — masks non matchés                 |
| `associator_score`          | probe | commit match — score final                        |
| `associator_source`         | probe | commit match — labellisé `source` (`slow`/`fast`) |

---

### L3.6 — Sondes `tracker/motion.py` (3 sondes)

| Sonde                      | Type  | Emplacement                                     |
| -------------------------- | ----- | ----------------------------------------------- |
| `motion_dt_ms`             | probe | `predict_position()` — `dt` brut                |
| `motion_staleness_slow_ms` | probe | `predict_position()` — staleness détection slow |
| `motion_staleness_capped`  | count | `predict_position()` — branche dt saturé        |

---

### L3.7 — Sondes `capture/` (3 sondes)

| Sonde              | Type  | Emplacement                             |
| ------------------ | ----- | --------------------------------------- |
| `capture_frame_ms` | probe | source active — durée acquisition frame |
| `capture_drop`     | count | source active — frame nulle / timeout   |
| `capture_source`   | gauge | `selector.py` — index source active     |

---

### L3.8 — Sondes `fast_track_thread.py` (8 sondes — **conditionnées à audit R9-4**)

> ⚠️ **Ne pas implémenter avant livraison audit R9-4 validé.**

| Sonde                | Type  | Emplacement                                |
| -------------------- | ----- | ------------------------------------------ |
| `fast_wakeup_lag_ms` | probe | `_worker()` — délai réveil thread          |
| `fast_tick_ms`       | probe | `_worker()` — durée tick complet           |
| `fast_of_total`      | count | `_worker()` — appels optical flow          |
| `fast_ncc_total`     | count | `_ncc_on_roi()` — appels NCC               |
| `fast_ncc_confirmed` | count | `_ncc_on_roi()` — NCC au-dessus seuil      |
| `fast_stale_used`    | count | `_worker()` — masks stale utilisés         |
| `fast_mask_lost`     | count | `_worker()` — masks perdus (stale dépassé) |
| `fast_poll_ms`       | probe | `_worker()` — durée poll résultat          |

> ✅ Vérification anti-doublon avec schéma L0.4 `bench_fast` requise lors de l'audit.

---

### L3.9 — Sondes `core/mask.py` (6 sondes — **conditionnées à audit R9-5**)

> ⚠️ **Ne pas implémenter avant livraison audit R9-5 validé.**

| Sonde                     | Type  | Emplacement                                                     |
| ------------------------- | ----- | --------------------------------------------------------------- |
| `mask_lifetime_s`         | probe | `to_dict()` ou point de lecture — durée vie depuis `created_ts` |
| `mask_state_transition`   | count | setter état — transition PENDING→CONFIRMED→LOST                 |
| `mask_confirm_latency_ms` | probe | transition CONFIRMED — `ts_confirmed - created_ts`              |
| `mask_lost_latency_ms`    | probe | transition LOST — `ts_lost - created_ts`                        |
| `mask_area_px`            | probe | création / update — aire ROI en pixels                          |
| `mask_iou_history`        | probe | update — IoU avec prédiction motion                             |

> ✅ Cohérence avec `created_ts` L1 et `mask_lifetime_s` schéma L0.4 requise lors de l'audit.

---

### 🔒 Borne L3

- ✅ Audit R9-4 + R9-5 livrés et validés avant L3.8 / L3.9.
- ✅ 27 sondes actives L3.1→L3.7 implémentées.
- ✅ 14 sondes L3.8 + L3.9 implémentées post-audit.
- ✅ 3 fichiers JSONL produits par session, nommage `session_id` conforme.
- ✅ `debug.csv.*` absent — aucune écriture CSV résiduelle.
- ✅ Drop JSONL loggué WARNING si queue pleine.
- ✅ Shutdown propre testé (SIGINT + fin normale).
- ✅ Nommage T-2 respecté sur toutes les sondes.
- ✅ Aucune régression smoke test.
- ✅ Tag git `b04b-L3`.

---

## ⚫ Lot L4 — Validation intégration

### L4.1 — Smoke test 60 s

- Lancer l'application sur session de référence (60 s).
- Vérifier :
  - 3 fichiers JSONL créés et alimentés en continu.
  - Aucune ligne malformée (parser JSON ligne par ligne).
  - Schéma L0.4 respecté (clés obligatoires présentes).
  - Aucun WARNING drop queue.
  - FPS stable (variance < 15% — critère B-04c).
  - Aucune exception non gérée.

### L4.2 — Vérification non-régression

- Comparer FPS moyen session L4 vs baseline pré-L0 (delta toléré < 5%).
- Vérifier sondes `main_frames_total`, `main_blur_ms`, `main_capture_wait_ms` cohérentes avec baseline.

---

### 🔒 Borne L4

- ✅ 3 fichiers JSONL valides produits sur 60 s.
- ✅ Schéma L0.4 validé parsing automatique.
- ✅ FPS delta < 5% vs baseline.
- ✅ Zéro WARNING drop queue sur session nominale.
- ✅ Tag git `b04b-L4`.

---

## ⚪ Lot L5 — Livraison B-04b

### L5.1 — Récap consolidé

- Produire le livrable markdown final de B-04b dans `Plan_Tracker.md` :
  - Liste exhaustive des 41 sondes finales (nom, type, fichier, domaine JSON, justification).
  - Décisions transverses T-1→T-4 appliquées.
  - Schéma JSONL L0.4 en annexe normative.
  - Dettes legacy purgées (4 : `_motion_stats`, `frame_count/fps_timer`, `_b01_stats`, `tracker.stats()`).
  - Arbitrage structurel tranché (1 : `created_ts` Option A).
  - Suspensions levées (2 : `apply_detections_fast` + branches `source="fast"` associator).
  - Sondes écartées avec motifs (référence audit R9-4 / R9-5).

### L5.2 — Mise à jour `Plan_Tracker.md`

- Marquer B-04b ✅ livré.
- Lever blocage B-04c, B-04d, B-05.

### L5.3 — Bascule B-04c

- Précondition activée : _« Session > 30 s avec variance FPS mesurée séparément scène avec/sans masks actifs. »_
- Démarrer la session de référence.

---

### 🔒 Borne L5

- ✅ Récap consolidé écrit et archivé dans `Plan_Tracker.md`.
- ✅ B-04b marqué ✅ livré.
- ✅ B-04c prêt à démarrer.
- ✅ Tag git `b04b-L5`.

---

## 🛑 Règles transverses non-négociables

1. **Aucun lot ne démarre tant que le précédent n'est pas validé** (borne 🔒 cochée).
2. **Aucune sonde ajoutée hors plan** → backlog B-04c.
3. **Rollback** : tag git à chaque borne 🔒. Granularité commits intra-lot libre.
4. **Pas de mise en œuvre avant validation du plan v5**.
5. **Aucune optimisation opportuniste** → P-04.
6. **L3 atomique vis-à-vis de L4** : L3.1 → L3.9 tous livrés avant L4.
7. **L3.8 prioritaire dans L3** : lève 3 suspensions (L3.3, L3.4, L3.6).
8. **Schéma L0.4 immuable** : toute évolution post-L0 = nouveau ticket.
9. **Nommage T-2 = contrat de groupement JSON** : toute nouvelle sonde doit respecter `<domaine>_<action>[_<qualifieur>]`.
10. **Audit R9-4 / R9-5 bloquants** : L3.8 et L3.9 ne démarrent pas sans livrable audit validé.

---

## 📊 Synthèse finale post-L5

| Métrique                         | Valeur cible                                                                     |
| -------------------------------- | -------------------------------------------------------------------------------- |
| Sondes actives totales           | 41 (27 L3.1→L3.7 + 8 L3.8 + 6 L3.9)                                              |
| Dettes legacy purgées            | 4 / 4                                                                            |
| Décisions transverses appliquées | 4 / 4 (T-1→T-4)                                                                  |
| Arbitrages structurels tranchés  | 1 / 1 (`created_ts` Option A)                                                    |
| Suspensions levées               | 2 / 2                                                                            |
| Fichiers JSONL en sortie         | 3 (`frame` + `agg` + `fast`)                                                     |
| Writers JSONL                    | 3 threads dédiés                                                                 |
| Domaines JSON                    | 8 (`main`, `tracker`, `registry`, `motion`, `detect`, `fast`, `capture`, `mask`) |
| `agg_interval` par défaut        | 1.0 s                                                                            |
| `fast.interval_s` par défaut     | 1.0 s                                                                            |
| Régression FPS tolérée           | < 5%                                                                             |
| Durée smoke test L4              | 60 s                                                                             |
| Audit R9-4 / R9-5                | Bloquants L3.8 / L3.9                                                            |
| État B-04b                       | ✅ Livré post-L5                                                                 |
| État B-04c                       | 🟢 Démarrable post-L5                                                            |
