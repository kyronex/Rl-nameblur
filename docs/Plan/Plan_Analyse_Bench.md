# Plan — Script d'analyse `bench_analyse.py`✅ **Livré**

---

## 0. Identité du script

**Fichier** : `bench/compare/bench_analyse.py`
**Entrée** : un fichier JSON produit par `bench_compare.py` (rapport de session unique **ou** comparaison)
**Sortie** : rapport texte structuré sur `stdout`, exploitable sans outillage
**Invocation** : `python bench/compare/bench_analyse.py <rapport.json>`
**Aucune dépendance** hors stdlib — le JSON est déjà produit, `_config.py` n'est pas importé (les valeurs sont lues dans le JSON, les seuls seuils inventoriés ci-dessous sont documentés comme tels)

---

## 1. Seuils et constantes internes

Aucune magic number silencieuse. Tout seuil est nommé en tête de fichier avec sa justification :

```python
# ── Seuils d'analyse (non issus de config.yaml — propres à ce script) ──
WARN_P99_OVER_AVG_FACTOR  = 3.0   # p99 > avg × 3 → instabilité sonde
WARN_IQR_OVER_AVG_FACTOR  = 1.0   # iqr > avg × 1 → dispersion élevée
WARN_SPIKE_COUNT          = 3     # spike_count > 3 → régime instable
WARN_DRIFT_SLOPE_MS_S     = 0.5   # |drift_slope| > 0.5 ms/s → dérive notable
WARN_PRESENCE_RATE_MIN    = 0.5   # reprise de FRAME_BUDGET_MIN_PRESENCE_RATE (lu JSON)
GOULOT_TOP_N              = 3     # nombre de groupes lourds reportés
TRIGGER_MIN_ABS_RHO       = 0.7   # corrélation minimale pour qualifier déclencheur
DELTA_PROBE_WARN_PCT   = 10.0  # |avg_delta_pct| > 10 % → affiché en passe E
DELTA_BUDGET_WARN_PCT  =  5.0  # |pct_delta_pct| > 5 %  → affiché en passe E
DELTA_GAUGE_WARN_PCT   = 10.0  # |delta_pct| > 10 %     → affiché en passe E
```

---

## 2. Structure du rapport produit (stdout)

```text
══════════════════════════════════════════════════
  BENCH ANALYSE — <session_id>
  Mode : [session_unique | comparaison]
══════════════════════════════════════════════════

[A] BUDGET FRAME
[B] INSTABILITÉ DES SONDES
[C] DÉCLENCHEURS (corrélations → goulots)
[D] ROBUSTESSE TRACKING
[E] COMPARAISON        ← absent si mode session_unique
[F] RÉSUMÉ CONSOLIDÉ
```

---

## 3. Données lues dans le JSON — inventaire exhaustif

### 3.1 Racine

| Clé JSON                 | Type         | Utilisation                                                              |
| ------------------------ | ------------ | ------------------------------------------------------------------------ |
| `schema_version`         | int          | Garde — abort si ≠ 1                                                     |
| `target_session`         | str          | Entête                                                                   |
| `target.buckets`         | dict         | Itération sur les buckets (voir §3.2 structure)                          |
| `target.temporal_events` | dict         | D.3 — lu **une seule fois** au niveau `target`, transmis à chaque bucket |
| `comparisons`            | dict\|absent | Détection mode comparaison                                               |

> **Note structure `target.buckets`** : la clé `hot` contient une **liste** de dicts (buckets indexés), pas un dict unique. Les clés `cold` et `tail` sont des dicts directs. L'itération doit gérer les deux cas (voir §5 Architecture).

### 3.2 Par bucket (`target.buckets.<label>`)

| Clé JSON                                      | Type        | Passe                                |
| --------------------------------------------- | ----------- | ------------------------------------ |
| `frame_budget`                                | dict\|null  | A                                    |
| `frame_budget.reference`                      | str         | A — libellé sonde totale             |
| `frame_budget.total_ms`                       | float       | A                                    |
| `frame_budget.rows_total`                     | int         | A                                    |
| `frame_budget.groups.<g>.probe`               | str         | A                                    |
| `frame_budget.groups.<g>.pct`                 | float\|null | A — tri goulots                      |
| `frame_budget.groups.<g>.sum_ms`              | float\|null | A                                    |
| `frame_budget.groups.<g>.presence_rate`       | float       | A                                    |
| `frame_budget.groups.<g>.conditional`         | bool        | A — pas d'alarme si absent           |
| `frame_budget.groups.<g>.low_presence`        | bool        | A — flag fiabilité                   |
| `frame_budget.unaccounted_pct`                | float       | A                                    |
| `frame_budget.unaccounted_warn`               | bool        | A — booléen pré-calculé, lu tel quel |
| `probes.<name>.avg`                           | float       | B                                    |
| `probes.<name>.p99_exact`                     | float\|null | B                                    |
| `probes.<name>.p95_exact`                     | float\|null | B                                    |
| `probes.<name>.iqr_exact`                     | float\|null | B                                    |
| `probes.<name>.spike_count`                   | int\|null   | B                                    |
| `probes.<name>.drift_slope`                   | float\|null | B                                    |
| `correlations.pairs`                          | list        | C                                    |
| `correlations.pairs[i].a`                     | str         | C                                    |
| `correlations.pairs[i].b`                     | str         | C                                    |
| `correlations.pairs[i].rho`                   | float       | C                                    |
| `correlations.pairs[i].strength`              | str         | C — lu, non recalculé                |
| `correlations.pairs[i].n_samples`             | int         | C                                    |
| `correlations.summary.truncated_by_max_pairs` | bool        | C — avertissement                    |
| `gauges.<name>`                               | float       | D                                    |

### 3.3 Données temporelles — niveau `target` (pas par bucket)

`temporal_events` est présent **uniquement** à la racine de `target`, pas dans chaque bucket individuel. Il est lu une seule fois dans `main()` et transmis en paramètre à `run_pass_d`.

| Clé JSON                                           | Type        | Passe                       |
| -------------------------------------------------- | ----------- | --------------------------- |
| `target.temporal_events.<canal>.median_interval_s` | float\|null | D.3                         |
| `target.temporal_events.<canal>.gaps_stat`         | int\|null   | D.3                         |
| `target.temporal_events.<canal>.gaps_fixed`        | int\|null   | D.3 — null si canal `frame` |

> `gaps_fixed` du canal `frame` est structurellement `null` (event-driven) — affiché `N/A`.

### 3.4 Sondes ciblées — Passe D (robustesse tracking)

Préfixes lus dynamiquement (présence non garantie) :

```text
tracker_confirmed, tracker_pending, tracker_lost   → gauges (par bucket)
motion_*                                           → probes (par bucket)
associator_*                                       → probes (par bucket)
main_match_ms                                      → probe  (par bucket, conditionnel)
```

### 3.5 Comparaison — Passe E

| Clé JSON                                                          | Utilisation               |
| ----------------------------------------------------------------- | ------------------------- |
| `comparisons.<type>.reference_session`                            | Libellé référence         |
| `comparisons.<type>.deltas.probes.<name>.avg_delta_pct`           | Delta moyen sonde         |
| `comparisons.<type>.deltas.frame_budget.groups.<g>.pct_delta_pct` | Delta poids groupe budget |
| `comparisons.<type>.deltas.gauges.<name>.delta_pct`               | Delta gauge               |
| `comparisons.<type>.appeared_probes`                              | Nouvelles sondes          |
| `comparisons.<type>.disappeared_probes`                           | Sondes disparues          |

> **Ajout** : `deltas.gauges` est désormais exploité en passe E. Les gauges dont `|delta_pct| > DELTA_GAUGE_WARN_PCT` sont affichées (même seuil et logique que les probes).

---

## 4. Passes détaillées

### Passe A — Budget frame

**Source** : `frame_budget` de chaque bucket
**Logique** :

1. Si `frame_budget is null` ou absent → section vide + note `"Non disponible"`, continue
2. Trier les groupes par `pct` décroissant (None en dernier)
3. Afficher tableau : `groupe | sonde | pct% | sum_ms | presence_rate | flags`
4. Flags affichés :
   - `[LOW_PRESENCE]` si `low_presence == true`
   - `[CONDITIONNEL]` si `conditional == true`
5. Ligne `unaccounted_pct` + `[WARN]` si `unaccounted_warn == true`
6. Identifier **top `GOULOT_TOP_N`** groupes (pct non null, non conditionnel, non low_presence) → set de noms de **sondes** transmis à la passe C

**Aucun recalcul de seuil** : `unaccounted_warn` est lu tel quel depuis le JSON.

---

### Passe B — Instabilité des sondes

**Source** : `probes` de chaque bucket
**Logique par sonde** :

```python
flags = []
if p99_exact is not None and avg > 0:
    if p99_exact > avg * WARN_P99_OVER_AVG_FACTOR:
        flags.append("P99_HIGH")
if iqr_exact is not None and avg > 0:
    if iqr_exact > avg * WARN_IQR_OVER_AVG_FACTOR:
        flags.append("IQR_HIGH")
if spike_count is not None:
    if spike_count > WARN_SPIKE_COUNT:
        flags.append("SPIKES")
if drift_slope is not None:
    if abs(drift_slope) > WARN_DRIFT_SLOPE_MS_S:
        flags.append("DRIFT")
```

Affichage : **uniquement les sondes avec au moins un flag** (les sondes propres ne sont pas listées, sauf en mode verbose futur).
Tri : nombre de flags décroissant, puis nom alphabétique.
Résumé final : `N sondes analysées, K instables`.

---

### Passe C — Déclencheurs

**Source** : `correlations.pairs` + goulots identifiés en passe A
**Logique** :

1. Si `truncated_by_max_pairs == true` → avertissement affiché
2. Pour chaque paire : si `abs(rho) >= TRIGGER_MIN_ABS_RHO` **et** (`a` ou `b` est la sonde d'un goulot identifié en A) → déclencheur candidat
3. Affichage : `goulot ← sonde_corrélée | rho=X.XXX | strength | n_samples`
4. Tri : `|rho|` décroissant
5. Si aucun déclencheur → `"Aucun déclencheur identifié pour les goulots top-N"`

**Lien A→C** : les goulots sont passés comme `set[str]` de noms de **sondes** (clé `probe` du groupe).

---

### Passe D — Robustesse tracking

**Signature** : `run_pass_d(bucket, temporal_events)` — `temporal_events` transmis depuis `main()`.
**Trois sous-sections** :

#### D.1 — État tracker

```text
tracker_confirmed | tracker_pending | tracker_lost
```

Lus depuis `gauges` si présents. Ratio `lost / (confirmed + pending + lost)` calculé localement si les trois sont présents.

#### D.2 — Stabilité motion / associator

Sondes lues : tout nom commençant par `motion_` ou `associator_` dans `probes`.
Flags B réutilisés (même logique, mêmes seuils) — pas de duplication de code : appel à la même fonction helper.

#### D.3 — Régularité temporelle

Source : paramètre `temporal_events` (lu au niveau `target` dans `main()`).
Pour chaque canal présent :

```text
canal | median_interval_s | gaps_stat | gaps_fixed
```

`gaps_fixed` du canal `frame` affiché `N/A` (null structurel, event-driven).
Si `temporal_events` est absent ou vide → `"Non disponible"`.

---

### Passe E — Comparaison _(absente si session unique)_

**Condition d'entrée** : `comparisons` présent dans le JSON
**Pour chaque type de comparaison** (`absolute`, `relative`) :

1. Afficher `reference_session`
2. Lister `appeared_probes` / `disappeared_probes` (changements structurels)
3. **Deltas probes** : afficher uniquement les sondes dont `|avg_delta_pct| > 10%`, tri décroissant
4. **Deltas budget** : afficher uniquement les groupes dont `|pct_delta_pct| > 5%`
5. **Deltas gauges** : gauges dont `|delta_pct| > DELTA_GAUGE_WARN_PCT` (10%), même logique que probes
6. Pas de verdict automatique pass/fail — affichage factuel uniquement

---

### Passe F — Résumé consolidé

Agrégation cross-buckets :

```text
── RÉSUMÉ ──────────────────────────────────────
Buckets analysés       : N
Sondes instables       : K  (liste des noms)
Goulots identifiés     : top-3 par bucket
Déclencheurs corrélés  : M paires
Gaps temporels         : X gaps_stat / Y gaps_fixed (tous canaux, source target)
Comparaison disponible : oui/non
```

> Les gaps temporels sont ceux de `target.temporal_events` (niveau racine), agrégés une seule fois — pas par bucket.

Aucun verdict global pass/fail — le script **rapporte**, l'humain décide.

---

## 5. Architecture du code

```text
bench_analyse.py
│
├── # Constantes (section 1)
│
├── def load_json(path)            → dict
├── def check_schema(data)         → abort si schema_version != 1
│
├── def fmt_flag(flags)            → str  "[FLAG1][FLAG2]"
├── def fmt_float(v, digits=2)     → str  arrondi + "—" si None
├── def fmt_pct(v)                 → str  "12.3%" ou "—"
│
├── def probe_flags(probe_stats)   → list[str]
│       # logique B — réutilisée en D.2
│
├── def run_pass_a(bucket)         → dict  {goulots: set[str], warnings: list}
├── def run_pass_b(bucket)         → dict  {instables: list, total: int}
├── def run_pass_c(bucket, goulots)→ dict  {declencheurs: list}
├── def run_pass_d(bucket, temporal_events)    → dict  {gaps: dict, tracker: dict}
│       # temporal_events : dict lu depuis target, pas depuis bucket
├── def run_pass_e(data)           → dict  {deltas: list}  # niveau racine
├── def run_pass_f(results, has_comparison, temporal_events) → None  # affiche résumé
│
└── def main()
        data = load_json(sys.argv[1])
        check_schema(data)
        has_comparison = bool(data.get("comparisons"))
        print_header(data, mode)

        target = data.get("target") or {}
        temporal_events = target.get("temporal_events") or {}   # ← lu une fois ici

        buckets = target.get("buckets") or {}
        results = []

        for label, bucket_or_list in buckets.items():
            # hot → liste ; cold, tail → dict direct
            items = bucket_or_list if isinstance(bucket_or_list, list) else [bucket_or_list]
            for i, bucket in enumerate(items):
                sub_label = f"{label}[{i}]" if isinstance(bucket_or_list, list) else label
                print_bucket_header(sub_label)
                a = run_pass_a(bucket)
                b = run_pass_b(bucket)
                c = run_pass_c(bucket, a["goulots"])
                d = run_pass_d(bucket, temporal_events)   # ← transmis ici
                results.append({"label": sub_label, "a": a, "b": b, "c": c, "d": d})

        if has_comparison:
            run_pass_e(data)

        run_pass_f(results, has_comparison, temporal_events)   # ← transmis pour F
```

---

## 6. Comportements défensifs

| Situation                             | Comportement                                                |
| ------------------------------------- | ----------------------------------------------------------- |
| `frame_budget` absent ou null         | Passe A affiche `"Non disponible"`, continue                |
| Sonde sans `p99_exact` (samples < 20) | Champ ignoré silencieusement, pas de flag                   |
| `correlations` absent                 | Passe C affiche `"Non disponible"`                          |
| `comparisons` absent                  | Passe E sautée, F note `"session unique"`                   |
| `avg == 0` pour ratio P99/IQR         | Division protégée, flag non posé                            |
| `target.temporal_events` absent       | D.3 affiche `"Non disponible"`, F gaps = 0/0                |
| `hot` est une liste                   | Itération indexée `hot[0]`, `hot[1]`… — aucun bucket ignoré |
| `hot` contient un élément non-dict    | Ignoré avec note, pas d'abort                               |
| Bucket vide (aucune sonde)            | Affiché avec note, pas d'abort                              |
| `deltas.gauges` absent en comparaison | Section gauges E sautée silencieusement                     |
| JSON invalide / fichier manquant      | `sys.exit(1)` avec message clair                            |

---

## 7. Ce que le script ne fait pas

- Il **n'importe aucun module du projet** (`_config.py`, `bench/`, etc.)
- Il **ne recalcule aucune corrélation, aucun percentile, aucune anomalie** — tout est lu dans le JSON
- Il **ne produit pas de fichier de sortie** — stdout uniquement (redirection possible par l'appelant)
- Il **ne pose aucun verdict pass/fail global** — il rapporte, l'humain décide
- Il **ne modifie pas** le JSON d'entrée
