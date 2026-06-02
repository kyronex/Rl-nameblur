# 📐 Plan séquentiel autoporté — S5a & S5b ✅ **Livré**

**Document de référence** pour l'implémentation des familles statistiques **forme de distribution** (S5a — ✅ livré) et **détection d'anomalies temporelles** (S5b — prêt à patcher) dans `bench_compare.py`.

À consulter **avant tout patch**. Toute décision verrouillée ici fait foi.

---

## 🎯 Objectifs globaux S5

| Famille                   | Tronçon | Statut           | Signal produit                                                      | Critère audit impacté    |
| ------------------------- | ------- | ---------------- | ------------------------------------------------------------------- | ------------------------ |
| A — Forme distribution    | **S5a** | ✅ Livré & testé | Skewness + Kurtosis excess par sonde                                | #5 (dispersion : 8→9/10) |
| B — Anomalies temporelles | **S5b** | 🟢 Spec complète | Spikes (count, amplitude, deviation) + Drift (slope, intercept, R²) | #6 (anomalies : 3→8/10)  |

---

## 🔗 Dépendances dures

| Tronçon | Dépend de       | Raison                                                                                                 |
| ------- | --------------- | ------------------------------------------------------------------------------------------------------ |
| S5a     | S2 + S4 + S4bis | ✅ satisfait                                                                                           |
| S5b     | S5a             | ✅ satisfait — réutilise infra Skew/Kurt (échantillons exact, gestion null, prédicat sondes éligibles) |

---

## ✅ TRONÇON S5a — Skewness + Kurtosis excess — **LIVRÉ**

### Décisions verrouillées (rappel, toutes implémentées)

| #   | Décision              | Valeur figée                                                 |
| --- | --------------------- | ------------------------------------------------------------ |
| D1  | Type sortie           | Valeurs brutes seules, aucun flag                            |
| D2  | Seuil min Skewness    | `SKEWNESS_MIN_SAMPLES = 50`                                  |
| D3  | Seuil min Kurtosis    | `KURTOSIS_MIN_SAMPLES = 100`                                 |
| D4  | Sous seuil            | `null` silencieux                                            |
| D5  | Définition Kurtosis   | Excess (`scipy.stats.kurtosis`, `fisher=True`, `bias=False`) |
| D6  | Définition Skewness   | Fisher-Pearson (`scipy.stats.skew`, `bias=False`)            |
| D7  | Source échantillons   | `exact` uniquement (canal frame, count==1)                   |
| D8  | Deltas inter-sessions | Absolus (target − reference)                                 |
| D9  | `schema_version`      | Inchangé (1)                                                 |
| D10 | Périmètre             | `probes.*` + `fast_probes.*` forcés à `null`                 |
| D11 | Niveaux               | Session + cold + hot[i] + tail                               |
| D12 | scipy                 | Présent (1.17.1)                                             |
| D13 | Variance nulle        | Wrapper défensif `stdev == 0` → `null`                       |
| D14 | Ordre champs JSON     | Après `iqr_exact` / `iqr_approx`                             |

### Questionnements S5a — tous tranchés ✅

| ID      | Décision actée                             |
| ------- | ------------------------------------------ |
| Q-S5a-1 | scipy déjà déclaré → usage direct          |
| Q-S5a-2 | `bias=False` pour skew + kurtosis          |
| Q-S5a-3 | Variance nulle → wrapper `null`            |
| Q-S5a-4 | Insertion après `iqr_exact` / `iqr_approx` |

### ✅ Critère de fin S5a — **VALIDÉ**

- Rapports JSON exposent `skewness` + `kurtosis_excess` aux 4 niveaux. ✅
- Seuils min respectés (50 / 100). ✅
- Variance nulle → `null`. ✅
- Deltas absolus présents et corrects. ✅
- `bench-compare.md` mis à jour (sous-section « Forme de distribution »). ✅

---

## 🟢 TRONÇON S5b — Spikes + Drift — **SPEC COMPLÈTE, PRÊT À PATCHER**

### Objectif

Détecter dans chaque bucket (`cold` / `hot[i]` / `tail`) :

- **Spikes** : valeurs anormalement élevées via MAD robuste → `spike_count`, `spike_max_value`, `spike_max_deviation`.
- **Drift** : tendance linéaire intra-bucket via OLS sur série **filtrée des spikes** → `drift_slope`, `drift_intercept`, `drift_r2`.

### ✅ Décisions verrouillées S5b (spec complète)

| #   | Décision                    | Valeur figée                                                                                                                          |
| --- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| E1  | Définition spike            | **MAD robuste**, `SPIKE_MAD_FACTOR = 3.5`. Critère :                                                                                  |
| E2  | Garde MAD nul               | Si `MAD == 0` (bucket) → `spike_count`, `spike_max_value`, `spike_max_deviation` = `null` ; drift calculé sur série brute si éligible |
| E3  | Granularité spikes          | `spike_count` (int), `spike_max_value` (float, unité ms), `spike_max_deviation` (float, unités MAD)                                   |
| E4  | Méthode drift               | OLS via `numpy.polyfit(t_rel, y, 1)` → `slope`, `intercept` ; `r²` calculé manuellement                                               |
| E5  | Domaine temps drift         | `t_rel` en **secondes** (relatif au premier échantillon du bucket, depuis `mono` du canal frame)                                      |
| E6  | Pré-filtrage drift          | Retirer points spike (masquage booléen) avant `polyfit`. Si `MAD == 0` → série brute non filtrée.                                     |
| E7  | Seuil significativité drift | Valeurs brutes systématiques, **aucun flag** (cohérent D1 S5a)                                                                        |
| E8  | Min échantillons spikes     | `SPIKE_MIN_SAMPLES = 20` (sur série brute du bucket)                                                                                  |
| E9  | Min échantillons drift      | `DRIFT_MIN_SAMPLES = 30`, **appliqué post-filtrage spikes**                                                                           |
| E10 | Niveaux d'application       | `cold` + `hot[i]` + `tail` uniquement. **Pas au niveau session**.                                                                     |
| E11 | Périmètre sondes            | **Identique S5a** : canal `frame` + source `exact` + `count == 1`. Prédicat partagé.                                                  |
| E12 | `fast_probes`               | Forcés à `null` (cohérence stricte avec S5a, pas d'asymétrie de traitement)                                                           |
| E13 | Deltas inter-sessions       | Absolus (target − reference), sauf `spike_max_value` ET `drift_intercept` exclus des deltas                                           |
| E14 | Dépendance OLS              | `numpy.polyfit(deg=1)` (numpy déjà en deps)                                                                                           |
| E15 | Source échantillons         | `exact` uniquement (timestamps `mono` réels requis pour drift)                                                                        |
| E16 | Sous seuil                  | `null` silencieux (cohérent D4 S5a)                                                                                                   |
| E17 | Ordre champs JSON           | Insertion **après** `kurtosis_excess` (suite logique : forme → anomalies)                                                             |
| E18 | `schema_version`            | Inchangé (1) — ajouts additifs rétro-compatibles                                                                                      |

Critère E1 : ✅ MAD robuste, SPIKE_MAD_FACTOR = 3.5 figé en v1. Formule : x est spike si |x − median| > 3.5 × MAD. Garde défensive : MAD == 0 → spike : null. Doc bench-compare.md : note succincte (3-4 lignes) expliquant MAD + seuil + rationale robustesse, sans démonstration mathématique.

### ❓ Questionnements résiduels S5b

**→ Aucun. Tous tranchés (Q-S5b-1 à Q-S5b-11).**

| ID       | Sujet                       | Décision actée                                                              |
| -------- | --------------------------- | --------------------------------------------------------------------------- |
| Q-S5b-1  | Méthode spike               | ✅ MAD robuste, k=3.5. `MAD == 0` → null.                                   |
| Q-S5b-2  | Granularité spikes          | ✅ `spike_count` + `spike_max_value` + `spike_max_deviation`                |
| Q-S5b-3  | Méthode drift               | ✅ OLS `numpy.polyfit(t_rel, y, 1)`                                         |
| Q-S5b-4  | Seuil significativité drift | ✅ Valeurs brutes, aucun flag                                               |
| Q-S5b-5  | Min échantillons drift      | ✅ 30, **post-filtrage spikes**                                             |
| Q-S5b-6  | Niveaux drift               | ✅ cold + hot[i] + tail                                                     |
| Q-S5b-7  | Niveaux spikes              | ✅ cold + hot[i] + tail. Garde `MAD == 0` par bucket.                       |
| Q-S5b-8  | Min échantillons spikes     | ✅ `SPIKE_MIN_SAMPLES = 20`                                                 |
| Q-S5b-9  | Périmètre sondes            | ✅ Identique S5a (frame + exact + count==1, prédicat partagé)               |
| Q-S5b-10 | Deltas inter-sessions       | ✅ Absolus sauf `spike_max_value` ET `drift_intercept`                      |
| Q-S5b-11 | Pré-filtrage drift          | ✅ Retrait points spike avant `polyfit` ; re-check `DRIFT_MIN_SAMPLES` post |

### 📋 Champs ajoutés au schéma JSON (S5b)

Dans `target.buckets.cold.probes.<name>`, `hot[i].probes.<name>`, `tail.probes.<name>` — **insérés après `kurtosis_excess`** :

```json
{
  "...champs S5a (skewness, kurtosis_excess)...",
  "spike_count":         "int | null",
  "spike_max_value":     "float | null",
  "spike_max_deviation": "float | null",
  "drift_slope":         "float | null",
  "drift_intercept":     "float | null",
  "drift_r2":            "float | null"
}
```

**Niveau session** : aucun champ S5b. **`fast_probes`** : forcés à `null` (cohérence S5a).

Dans `deltas.buckets.cold.probes.<name>`, `hot[i].probes.<name>`, `tail.probes.<name>` :

```json
{
  "spike_count":         {"reference": <int>, "target": <int>, "delta": <int>},
  "spike_max_value":     {"reference": <float>, "target": <float>},
  "spike_max_deviation": {"reference": <float>, "target": <float>, "delta": <float>},
  "drift_slope":         {"reference": <float>, "target": <float>, "delta": <float>},
  "drift_intercept":     {"reference": <float>, "target": <float>},
  "drift_r2":            {"reference": <float>, "target": <float>, "delta": <float>}
}
```

**Sémantique delta** : si l'une des deux valeurs source est `null` → delta `null`. Champs sans `delta` (`spike_max_value`, `drift_intercept`) = valeurs brutes seules, jamais de clé `delta`.

### 🛡️ Contrat algorithmique S5b (ordre opératoire par bucket éligible)

```python
def compute_anomalies(samples: list[float], timestamps_mono: list[float]) -> dict:
    """
    samples : valeurs ms, ordre temporel mono préservé
    timestamps_mono : timestamps mono alignés sur samples
    """
    n = len(samples)
    out = {
        "spike_count": None, "spike_max_value": None, "spike_max_deviation": None,
        "drift_slope": None, "drift_intercept": None, "drift_r2": None,
    }

    # --- SPIKES ---
    spike_mask = None
    if n >= SPIKE_MIN_SAMPLES:                                    # E8
        median = statistics.median(samples)
        mad = statistics.median([abs(x - median) for x in samples])
        if mad > 0:                                               # E2
            arr = np.asarray(samples)
            spike_mask = np.abs(arr - median) > SPIKE_MAD_FACTOR * mad
            count = int(spike_mask.sum())
            out["spike_count"] = count
            if count > 0:
                spikes = arr[spike_mask]
                out["spike_max_value"] = float(spikes.max())
                out["spike_max_deviation"] = float(
                    np.max(np.abs(spikes - median) / mad)
                )
            else:
                out["spike_max_value"] = None         # pas de spike → null
                out["spike_max_deviation"] = None
        # MAD == 0 → spike_* restent null (E2)

    # --- DRIFT ---
    arr = np.asarray(samples)
    t = np.asarray(timestamps_mono)
    if spike_mask is not None:
        t_clean = t[~spike_mask]                                  # E6
        y_clean = arr[~spike_mask]
    else:
        t_clean = t                                               # MAD==0 ou n<SPIKE_MIN
        y_clean = arr

    if len(y_clean) >= DRIFT_MIN_SAMPLES:                         # E9 post-filtrage
        if statistics.stdev(y_clean.tolist()) == 0:               # garde variance nulle
            pass  # drift_* restent null
        else:
            t_rel = t_clean - t_clean[0]                          # E5 (secondes)
            slope, intercept = np.polyfit(t_rel, y_clean, 1)      # E4
            y_pred = slope * t_rel + intercept
            ss_res = float(np.sum((y_clean - y_pred) ** 2))
            ss_tot = float(np.sum((y_clean - y_clean.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
            out["drift_slope"] = float(slope)
            out["drift_intercept"] = float(intercept)
            out["drift_r2"] = r2

    return out
```

**Ordre des gardes obligatoire** :

1. `n >= SPIKE_MIN_SAMPLES` → sinon tous champs `null`
2. `MAD == 0` → spikes `null` (drift peut continuer sur série brute)
3. `len(filtré) >= DRIFT_MIN_SAMPLES` → sinon drift `null`
4. Variance nulle sur série filtrée → drift `null`

### 📦 Zones à fournir pour patch S5b

| #   | Fichier                                                          | Portée                                                                                 | Priorité |
| --- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------------- | -------- |
| 1   | `bench/compare/_stats.py`                                        | **Intégralité** (état post-S5a)                                                        | Haute    |
| 2   | `bench/compare/_builder.py`                                      | **Intégralité** (état post-S5a)                                                        | Haute    |
| 3   | `bench/compare/_config.py`                                       | **Intégralité** (état post-S5a)                                                        | Haute    |
| 4   | `bench/compare/_bucketing.py` (ou équivalent S4)                 | **Intégralité** — vérifier extraction `(samples, timestamps_mono)` ordonnée par bucket | Haute    |
| 5   | 1 rapport JSON exemple post-S5a (≥ 2 sessions, mode comparaison) | Validation schéma cible                                                                | Haute    |
| 6   | `bench-compare.md` — sections « Buckets » + « Probes »           | Pour patch doc                                                                         | Moyenne  |

### 🔬 Audits à mener pendant S5b

- **Audit B1** : confirmer que les échantillons `exact` par bucket conservent l'**ordre temporel `mono`** (prérequis OLS drift). Si tri par valeur en amont → corriger pour préserver ordre d'arrivée.
- **Audit B2** : confirmer disponibilité des timestamps `mono` alignés sur chaque échantillon `exact` (canal frame). Si pas exposés → ajout extraction dans `_bucketing.py`.
- **Audit B3** : mesurer impact perf rapport sur session typique (~10 sondes × 5 buckets × OLS) — cible < 100 ms additionnels.
- **Audit B4** : vérifier comportement sur bucket à valeurs strictement constantes (`MAD == 0` ET `stdev == 0`) → tous champs S5b `null`, pas de division par zéro.
- **Audit B5** : vérifier interaction avec `cold_truncated: true` (pas de hot/tail) → S5b sur cold uniquement, null silencieux pour hot/tail absents.
- **Audit B6** : vérifier cas `count > 0` mais aucun spike détecté (`spike_count == 0`) → `spike_max_value` et `spike_max_deviation` doivent être `null`, **pas `0.0`**.

### ✅ Critère de fin S5b

- Tous les rapports JSON exposent `spike_count`, `spike_max_value`, `spike_max_deviation`, `drift_slope`, `drift_r2` aux 3 niveaux bucket (cold + hot[i] + tail).
- Sondes/buckets sous seuil min → tous champs `null`.
- Deltas absolus présents sur cas de test à 2+ sessions.
- Doc `bench-compare.md` mise à jour : nouvelle sous-section « Détection d'anomalies », table « Limites v1 » marque S5b ✅, sous-section dédiée pour méthode MAD + OLS.

---

## 🗺️ Vue d'ensemble séquentielle

```text
S5a — Skewness + Kurtosis ✅ LIVRÉ & TESTÉ
       ↓
S5b — Spikes + Drift (≈3-4h) — SPEC COMPLÈTE ✅
 ├─ Q-S5b-1 à Q-S5b-11 tous tranchés
 ├─ Audit B1 à B6
 ├─ Patch _stats.py + _builder.py + _config.py + _bucketing.py
 ├─ Patch bench-compare.md
 └─ Validation : rapport JSON contient spikes/drift aux 3 niveaux bucket
       ↓
S5 complet ✅ → critères #5 (9/10) et #6 (8/10) atteints
       ↓
S6 — Budget frame & corrélations
```

## 📌 Récapitulatif des questionnements

### S5a (0 ouvert / 4 tranchés ✅)

| ID      | Sujet                  | Décision actée                            |
| ------- | ---------------------- | ----------------------------------------- |
| Q-S5a-1 | scipy en deps          | ✅ Présent (1.17.1) — utilisation directe |
| Q-S5a-2 | `bias=False` pour skew | ✅ Oui (skew + kurtosis, cohérence n−1)   |
| Q-S5a-3 | Variance nulle → null  | ✅ Wrapper défensif `stdev == 0` → `null` |
| Q-S5a-4 | Ordre des champs JSON  | ✅ Après `iqr_exact` / `iqr_approx`       |

### S5b (11 ouverts)

| ID       | Sujet                       | Décision actée                                          |
| -------- | --------------------------- | ------------------------------------------------------- |
| Q-S5b-1  | Méthode spike               | MAD robuste, k=3.5, garde `MAD == 0` → null             |
| Q-S5b-2  | Granularité spikes          | count + max_value + max_deviation                       |
| Q-S5b-3  | Méthode drift               | OLS `numpy.polyfit(t_rel, y, 1)`                        |
| Q-S5b-4  | Seuil significativité drift | Valeurs brutes, aucun flag                              |
| Q-S5b-5  | Min échantillons drift      | 30, post-filtrage spikes                                |
| Q-S5b-6  | Niveaux drift               | cold + hot[i] + tail (pas session)                      |
| Q-S5b-7  | Niveaux spikes              | cold + hot[i] + tail                                    |
| Q-S5b-8  | Min échantillons spikes     | 20                                                      |
| Q-S5b-9  | Périmètre sondes            | frame + exact + count==1 (identique S5a)                |
| Q-S5b-10 | Deltas                      | Absolus sauf `spike_max_value` ET `drift_intercept`     |
| Q-S5b-11 | Pré-filtrage drift          | Retrait points spike avant `polyfit`, re-check min post |

---

## 🚦 Prochaine action

**S5b entièrement spécifié — démarrage patch immédiat possible.**

1. ✅ Tous les questionnements S5b tranchés.
2. **Fournir zones 1 à 5** listées au tronçon S5b (`_stats.py`, `_builder.py`, `_config.py`, `_bucketing.py`, 1 rapport JSON post-S5a).
3. Production patch S5b (code + doc) + validation sur cas de test multi-sessions.

---
