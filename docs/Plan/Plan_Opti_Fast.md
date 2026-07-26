# Plan_Opti_Fast.md — Guide des features d'optimisation (worker fast)

> **Statut du document** : v1.0 — réorganisation suite audit Code ↔ Plan
> **Légende statut** : ✅ réalisé · 🟡 proposé/prêt · 🔴 à faire

---

## 1. API Bench — Référence vérifiée

> Inspéction de `bench.py` (classe `BenchRegistry`, singleton importé via `from bench import bench` en `fast_track_thread.py` L16)

### Méthodes confirmées

| Méthode                                                  | Signature                                  | Comportement                                                                                                 |
| -------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| `note`                                                   | `note(name: str, value)`                   | Ajoute au buffer `_frame_notes` (vidangé par `snapshot_frame()`) — pas de série temporelle                   |
| `count`                                                  | `count(name: str, n: int = 1)`             | Incrémente compteur cumulé `_counters` + historique horodaté `_count_history` + buffer frame `_frame_counts` |
| `probe`                                                  | `probe(name: str, duration_ms: float)`     | Latence : `_last` + historique `_probe_history` + buffer frame `_frame_probes`                               |
| `gauge`                                                  | `gauge(name: str, value: float)`           | **Écrase** la valeur + historique glissant `_gauge_history` — **pas de buffer frame**                        |
| `timer`                                                  | `timer(name: str)` → context manager       | Appelle `probe(name, ms)` en sortie de bloc                                                                  |
| `rate`                                                   | `rate(name: str, window_s: float) → float` | Débit (counts/s) sur fenêtre glissante — **lecture seule**, n'écrit rien                                     |
| `last`                                                   | `last(name: str)`                          | Dernière mesure brute                                                                                        |
| `read_count`                                             | `read_count(name: str) → int`              | Compteur cumulé                                                                                              |
| `read_gauge`                                             | `read_gauge(name: str)`                    | Dernière valeur gauge                                                                                        |
| `emit_lifecycle`                                         | `emit_lifecycle(event, mask, reason)`      | Émet un `LifecycleRecord` dans `_events`                                                                     |
| `push_frame / push_events / push_detections`             | `push_frame()` etc.                        | Écrit le snapshot courant dans le writer JSONL对应 canal                                                     |
| `snapshot_frame / snapshot_events / snapshot_detections` | `snapshot_*(...)`                          | Construit le dict Python du snapshot pour le writer                                                          |
| `snapshot_all`                                           | `snapshot_all(window_s)`                   | Cumulatif depuis start (canal agg)                                                                           |
| `snapshot_fast`                                          | `snapshot_fast()`                          | Fenêtre glissante, filtre préfixe `fastC_` (canal fast)                                                      |

### ⚠️ Attention — `prob()` n'existe pas

Pour une probabilité empirique : utiliser `count()` + lire via `rate()`.
Pour une valeur brute par frame : utiliser `note()`.
Pour une latence : utiliser `probe()` ou `timer()`.

### Canaux JSONL existants

| Canal        | Méthode d'écriture  | Contenu principal                                          |
| ------------ | ------------------- | ---------------------------------------------------------- |
| `frame`      | `push_frame()`      | `{probes, counts, gauges, notes}` — différentiel par frame |
| `events`     | `push_events()`     | `{events: {mask_id: LifecycleRecord}}`                     |
| `detections` | `push_detections()` | Détections                                                 |

Schéma commun : `{schema_version: 1, ts, mono, session_id, mode, ...}`

`LifecycleRecord` contient : `event`, `mask_id`, `state`, `rx, ry, rw, rh`, `confidence`, `created_ts`, `event_ts`, `total_matches_cumul`, `frames_matched`, `source`, `lost_since_ts`, `reason`, `revived`, `frame_id`, `scores`, `hash_history`.

---

## 2. Bloc 1 — Bench-frame de référence (SOCLE) 🟢

> **Statut** : 🟢 Clôturé — Run 1 (598 skips NCC-fail) + Run 2 (2 824 skips NCC-fail / 3 430 frames).
> Toutes les sondes sont cohérentes et l'invariant est validé.

---

### 2a — Compteur cumulé de skip NCC-stale ✅

**Famille 1** — Clôturé.

```python
bench.count("F_ncc_stale_skipped_total")   # dans la branche else finale de la phase 4b
```

**Runs validées** : 598 skips (Run 1) + 2 824 skips (Run 2). Invariant tenu sur les 2 runs :
`count(F_ncc_stale_skipped_total) = appends(F_ncc_stale_skip_uid) = appends(F_ncc_score_at_skip) = 2 824`. Zéro None/NaN.

---

### 2b — UID du dernier masque skipé (par run) ✅

**Famille 1** — Clôturé.

```python
bench.note("F_ncc_stale_skip_uid", v.uid)  # même branche else, co-localisé avec 2a
```

**Runs validées** : 598 UID-appends (Run 1) + 2 824 UID-appends (Run 2). Cohérence 100 % avec 2a sur les 2 runs.

**Distribution UID (Run 1 — n=598 skips)** :

| Métrique              | Valeur          | Lecture                          |
| --------------------- | --------------- | -------------------------------- |
| UID distincts touchés | 20              | tous les masques actifs impactés |
| Gini                  | **0,249**       | modéré, loin d'un Pareto (0,80+) |
| Top-1 (UID 13)        | 9,4 %           | aucun UID dominant               |
| Top-5                 | 39,1 %          | queue non lourde                 |
| Bottom-50 % des UID   | 32,8 %          | aucun masque épargné             |
| Skips moyens/UID      | ~30 (CV ≈ 0,54) | plage 1→56, homogène             |

**Verdict** : distribution **structurelle uniforme**. Le stale NCC n'est pas une pathologie par masque — c'est un phénomène systémique du pipeline NCC qui touche tous les masques à des rythmes comparables.

---

### 2c — Score NCC au moment du skip ✅

**Famille 2** — Clôturé.

```python
bench.note("F_ncc_score_at_skip", float(score))          # même branche else, co-localisé
bench.note("F_ncc_template_age_at_skip", template_age_ms)  # timestamp template en ms
```

**Run validée** : 3 430 frames, n=2 824 skips, 100 % des skips soniqués.

**Population split (n=2 824 skips)** :

| Population                      | Effectif | %          | Signature NCC       | Cause identifiée                                             |
| ------------------------------- | -------- | ---------- | ------------------- | ------------------------------------------------------------ |
| **A** — ROI/template absent     | 97       | **3,4 %**  | `score == 0.0`      | échec structurel amont (marginal)                            |
| **B** — NCC calculé, sous seuil | 2 727    | **96,6 %** | score médian ≈ 0,23 | matching NCC en scène (occlusion / appearance / motion blur) |

**Résultat décisif — Hypothèse template-age RÉFUTÉE** :

- Pearson r = **−0,05** entre `score_at_skip` et `template_age_at_skip` → **aucune corrélation**.
- **61 % des skips ont un template age ≈ 0 ms** (frais) et échouent quand même.
- Le template refresh fonctionne ; la cause des 96,6 % d'échecs est **le matching NCC en scène**, pas la dégradation du template.

> ⚠️ **Vigilance** : ce diagnostic repose sur une seule session dense. Reconfirmer r≈0 et 61 % template-frais sur une 2ᵉ session avant d'en faire un invariant.

---

### 2d — Template age au moment du skip ✅

Co-localisé avec 2c (même bloc `bench.note`). Déjà couvert ci-dessus. Aucune action complémentaire.

---

### Canal JSONL vérifié

```text
- counts.F_ncc_stale_skipped_total      ≥ 0
- notes.F_ncc_stale_skip_uid            (par run, zéro None/NaN)
- notes.F_ncc_score_at_skip            (par run, zéro None/NaN)
- notes.F_ncc_template_age_at_skip     (par run)
```

Vérifiable via un script d'analyse JSONL post-run.

---

### Compteurs dynamiques par UID

```python
bench.count(f"F_ncc_bypass_uid_{v.uid}")   # non implémenté dans les runs validées ;
                                             # réservé pour F3 si nécessaire
```

---

### 2e — Source OF du candidate_rect au moment du skip 🔶

> **Statut** : 🔶 **NIÉE par la mesure** — reconfirmée sur 2 runs (post-patch `F_roi_source_at_skip` + run 2f).

**Sonde implémentée** (branche else finale phase 4b, delta minimal — contrat 2a/2b/2c respecté) :

\```python
bench.note("F_roi_source_at_skip", int(of_succeeded and last_state.get("of_fail_streak", 0) == 0)) # → 1 = OF sain, 0 = fallback/extrapolé
\```

**Résultat décisif — levier OF-conditionnel RÉFUTÉ** :

- Point-biserial r = **0,067** entre `F_roi_source_at_skip` et `F_ncc_score_at_skip` → **aucun signal**.
- **91,4 %** des échecs Population B surviennent avec `source=1` (**OF sain**) — reconfirmé à **89,2 %** en run 2f.
- La qualité du rect OF d'entrée **n'explique pas** l'échec NCC. Le levier **Bloc 3 (OF-conditionnel)** est donc **inerte** : son fondement mesuré manque.

> ⚠️ **Caveat socle** : la sonde ne capture que le **dernier skip de chaque frame** (voir Bloc 1 reformulé). Le r=0,067 porte sur ~2 500 skip-frames appariées, pas sur les ~1 500 events surnuméraires. Signal franchement nul (pas marginal) → suffisant pour nier le levier, à ne pas ériger en invariant absolu.

**Conséquence** : élimination convergente template-age (2c) → source OF (2e). La cause reste à chercher côté **appearance/scène** → Famille 2f.

---

### 2f — Texture & motion du patch au moment du skip 🔶

> **Statut** : 🔶 **texture RÉFUTÉE / motion réinterprétée** — run dédiée 2f (n=2 532 skips appariés / 2 673 skip-frames).

**Sonde implémentée** (même branche else finale phase 4b, delta minimal — lecture pure, aucune écriture d'état) :

\```python
bench.note("F_roi_texture_std_at_skip", float(curr_gray[ry0:ry1, rx0:rx1].std()))
bench.note("F_roi_motion_px_s_at_skip", float(math.hypot(last_state.get("vx_of", 0.0), last_state.get("vy_of", 0.0))))
texture_std : texturabilité du patch candidate_rect (proxy fragilité NCC)
motion_px_s : magnitude vitesse OF déjà maintenue en phase 4a (aucun recalcul)
\```

**Résultats** :

| Hypothèse testée   | Mesure                                                   | Verdict                       |
| ------------------ | -------------------------------------------------------- | ----------------------------- |
| **Texture faible** | Pearson r = **+0,073** (n=2 532)                         | 🔶 **RÉFUTÉE**                |
| **Motion blur**    | r global = **+0,051** ; Δquartile = **+0,282** _inverse_ | 🔶 **RÉFUTÉE, réinterprétée** |
| Template age (2c)  | r = **−0,016**                                           | ✅ toujours réfutée           |
| Source OF (2e)     | **89,2 %** skips OF sains                                | ✅ toujours réfutée           |

**Signal décisif — score bas ↔ motion OF nulle (proxy drift géométrique)** :

- Motion **Q1 (vitesse OF ≈ 0)** → score médian **0,065**.
- Motion **Q3 (motion > 0)** → score médian **0,347**.
- Δ = **+0,282**, soit **5,4×** l'effet texture (Δ=0,052), et dans la direction **inverse** du blur attendu.

**Interprétation** : `motion=0` n'est pas un flou, c'est un proxy de **drift positionnel** — `candidate_rect` reste stationnaire pendant que le template a dérivé, le NCC compare une signature géométriquement désalignée et échoue. Quand `motion>0`, l'OF a suivi le déplacement réel, l'alignement template↔position tient, le score remonte.

**Conclusion de la chaîne d'élimination (4 causes testées, 4 réfutées)** : template-age (2c) → source OF (2e) → texture (2f) → motion blur (2f). La convergence désigne un **drift géométrique template↔position**, seul signal positif exploitable.

> ⚠️ **Caveats** :
> r global motion quasi nul **car distribution bimodale** (75 % des skip-frames à motion=0) → lire le **split quartile**, pas le Pearson.
> **141 skip-frames** exclues (candidate_rect hors-écran, clip vide) — garde de bornage OK, aucun crash, 2 532/2 673 appariées.
> Caveat socle inchangé (dernier skip/frame). Invariants session sains : Gini 0,392, Top-1 UID 3,2 %, 59 % templates frais.

### 2h — Bypass NCC : réarmement inconditionnel de la staleness 🟡

> **Statut** : 🟡 **constat code confirmé par audit** (`fast_track_thread.py` L278-287, L334-343, L356-357). Racine des « plaques fantômes ».

**Constat** : le chemin bypass NCC publie un masque avec `score = 1.0` **forcé** et exécute `last_state["stale"] = 0` (L339) **sans aucune validation de ressemblance**. Tant que l'OF paraît sain (`of_fail_streak == 0`, template non expiré), le bypass réarme la staleness à chaque tick → le compteur n'atteint jamais `max_stale`, le masque **ne meurt jamais par staleness** même si la plaque réelle a disparu.

**Corollaire — pourquoi un NCC bas n'interrompt pas** : la branche else (L344-378) ne déclenche un LOST **que** sur plafond `stale > max_stale` (L357). Un score bas ≠ perte : il ne fait qu'incrémenter `stale` de +1 (L356), sans jamais retirer le masque de `_last_known`. Seul garde-fou résiduel : le filtre drift temporel `tracker.py` L126-129, mais **silencieux** (`continue` sans sonde `skipped`) → invisible au monitoring.

**Lien mesure** : converge avec 2f/2g — le drift géométrique franc se produit **précisément dans le régime bypass**, là où aucun NCC ne peut détecter le décrochage position↔plaque. Le bypass **transforme un drift en plaque fantôme persistante**.

**Sondes à créer** (lecture pure, delta minimal, co-localisées L339) :

\```python
bench.count("F_bypass_consecutive_total") # incréments de bypass consécutifs
bench.note("F_bypass_streak_at_publish", int(streak)) # longueur de chaîne bypass sans NCC
\```

> ⚠️ **Non actionnable en code avant Bloc 2 / 3-2c** — ces sondes instrumentent le remède (budget bypass / re-sync forcée), pas une hypothèse à réfuter.

## 3. Bloc 2 — Coût unitaire OF/NCC 🟢

**Dépend de** : Bloc 1 ✅ **+ étape 0 (reconfirmation 2g sur run motion distribuée)** — voir §7.

> **Réordonnancement acté** : le point d'entrée du Bloc 2 n'est plus 3-2a mais **3-2c**, seul chantier adossé à un signal mesuré positif (drift franc, 2g). 3-2a et 3-2b restent des hypothèses 🟠 non désignées dominantes → **différés, conditionnels**.

### 3-2c — Gate re-sync template sur drift franc + budget bypass `À CRÉER` 🟡

**Problème traité** : RC-NCC-drift 🟡 **+ RC-bypass-immortel** (2h) — deux faces d'une même faille : `candidate_rect` dérive pendant que le bypass réarme `stale=0` (L339), rendant le masque immortel et désaligné.

**Action** (delta minimal, branche else finale phase 4b + point bypass L339) :

1. **Drift** : quand `F_roi_drift_px_at_skip > seuil` (~50 px, **À CALER**), forcer une re-sync du template sur la position OF courante au lieu de skiper.
2. **Budget bypass (2h)** : ne plus réarmer `stale=0` inconditionnellement — soit ne pas remettre à zéro, soit plafonner à N bypass consécutifs, au-delà desquels une re-sync NCC est **forcée**. Cible la ligne L339.

**Métriques de contrôle** (Bloc 1 requis) :

- `counts.F_ncc_stale_skipped_total` — doit **diminuer**
- `counts.F_bypass_consecutive_total` — la queue longue doit s'assécher (2h)
- `notes.F_roi_drift_px_at_skip` — queue >50 px asséchée
- `notes.F_bypass_last_uid` — stable (le bypass doit rester fonctionnel, pas supprimé)

### 3-2a — ROI NCC sous-pixel `À CRÉER` — 🟠 DIFFÉRÉ (conditionnel)

**Problème traité** : RC-NCC-of-alignment — la position OF est arrondie avant l'extraction du template NCC.

**Action** : extraire le crop NCC à partir des coordonnées OF **float32 non arrondies** (précision sous-pixel), en utilisant l'interpolation `INTER_LINEAR` de `cv2.getRectSubPix`.

**Localisation** : `fast_track_thread.py` ( `_ncc_on_roi` ou directement dans la boucle worker) et `optical_flow.py`.

**Métriques de contrôle** (Bloc 1 requis) :

- `probes.fast_ncc_call_ms` — doit diminuer vs baseline
- `notes.F_bypass_last_uid` — stable (le bypass doit rester fonctionnel)

### 3-2b — LK Shi-Tomasi corner detector `À CRÉER` — 🟢 DIFFÉRÉ (conditionnel)

**Problème traité** : RC-OF-1 — Sparse OF Lucas-Kanade basé sur des coins détectés par Shi-Tomasi.

**Action** : remplacer ou compléter le tracking OF existant par un détecteur `goodFeaturesToTrack` avec paramètres calibrés :

\```python
cv2.goodFeaturesToTrack(
gray,
maxCorners=50, # À CALER
qualityLevel=0.01, # À CALER
minDistance=5, # À CALER
blockSize=3
)
\```

- tracking des 4 coins du masque (approche hybride corners + edges).

**Localisation** : `optical_flow.py` ou directement dans la boucle worker de `fast_track_thread.py`.

**Métriques de contrôle** :

- `probes.fast_of_total_ms` — à mesurer avant/après
- `counts.F_stale_skipped_total` — ne doit pas augmenter

---

## 4. Bloc 3 — Conditionnalité 🔴

**Dépend de** : Bloc 1 ✅

### 3 — OF conditionnel (symétrique du bypass NCC) 🔶 HORS-SCOPE / INERTE

> **Statut** : 🔶 **fondement mesuré manquant** — nié par 2e (r=0,067) et reconfirmé par 2f (89,2 % des skips sont OF sains). 3ᵉ réfutation convergente.

**Décision tranchée par la mesure** : la source OF du candidate n'explique pas l'échec NCC → aucune gate OF-conditionnelle ne réduira les skips. Ce chantier est **gelé** tant qu'aucune mesure ne le rouvre. Le levier exploitable identifié est le **drift géométrique** (voir 2f → Famille candidate 2g).

### 3-6 — Wakeup lag

**Action** : utiliser la sonde existante `F_wakeup_lag_ms` (L197) comme baseline. Si le p95 dépasse un seuil défini, investiguer la cause (I/O, verrou, taille de la frame).

---

## 5. Bloc 4 — Staleness 🔴

**Dépend de** : arbitrage Bloc 2 / 3-2c — un cap de staleness et une gate de re-sync agissent sur la **même dynamique de désalignement** ; à ordonnancer après 3-2c pour éviter deux leviers concurrents sur le drift.

### 3-5b — `motion_staleness_capped` et `motion_predict_source_slow` `À CRÉER`

**Précaution** : ces deux sondes sont **absentes aujourd'hui** — elles sont citées dans l'ancien plan mais **n'existent pas dans le code**. Les créer uniquement dans ce bloc, pas avant.

**`motion_staleness_capped` — À CRÉER** :
Sonde émise quand un masque est marqué stale maistronqué par un plafond (stale > max_stale mais mask pas encore LOST).

```python
# Dans fast_track_thread.py, bloc stale (autour de L350)
if last_state["stale"] > max_stale:
    bench.count("F_stale_capped_total")
```

**`motion_predict_source_slow` — À CRÉER** :
Sonde émise quand la prédiction utilise les données du tracker slow (vitesse OF extrapolée depuis `_last_known`) vs données NCC.

```python
# Dans fast_track_thread.py, bloc extrapolation (autour de L260)
bench.note("F_predict_source", "slow")  # ou "of" / "ncc"
```

**Localisation** : `fast_track_thread.py`, `motion.py` (si séparé), `tracker.py`.

---

## 6. Bloc 5 — Robustesse (parallélisable) 🟢

**Aucune dépendance inter-bloc**.

### 3-4 — Lock MaskRegistry + frame.copy()

**Problème** : data race suspectée sur l'accès concurrent au `MaskRegistry`.

**Actions** (à confirmer après analyse de thread-safety) :

1. Ajouter un `threading.Lock` sur les accès lecture/écriture au registre des masques si non déjà protégé.
2. Trancher `frame.copy()` (L111, commenté) : si la frame est partagée sans copie, un thread pourrait modifier le buffer pendant que le worker lit —风险的 réel vs coût mémoire à mesurer.

**Sondes de contrôle** :

```python
bench.gauge("F_registry_lock_ms", lock_wait_us)  # si lock existe
bench.gauge("F_frame_copy_enabled", 0.0 ou 1.0)  # flag de décision
```

---

## 7. Séquencement recommandé

```text
0. Bloc 1 (socle bench-frame) → ✅ CLÔTURÉ (2a→2f). Aucune feature évaluable sans lui.

1. ÉTAPE 0 — Prérequis mesure BLOQUANT 🔴
   a. Run de reconfirmation 2g (drift) sur run à motion SAINE (non dégénérée)
   b. Instrumentation 2h — sondes bypass (F_bypass_consecutive_total,
      F_bypass_streak_at_publish), lecture pure, co-localisées L339
   → Débloque le calage du seuil drift (~50 px) ET du budget bypass.
   → Tant que 0 n'est pas levée, 3-2c reste GELÉ.

2. Bloc 2 — Coût OF/NCC 🔴 (dépend de étape 0)
   → 3-2c UNIQUE point d'entrée : gate re-sync drift + budget bypass (remède 2g + 2h).
     3-2a (sous-pixel) et 3-2b (Shi-Tomasi) DIFFÉRÉS CONDITIONNELS :
     ouverts seulement si 3-2c s'avère insuffisant / si une mesure rouvre RC-OF-1.

3. Bloc 4 — Staleness 🔴 (APRÈS 3-2c)
   → 3-5b (motion_staleness_capped, motion_predict_source_slow).
     Ordonnancé après 3-2c : cap staleness et gate re-sync agissent sur la
     MÊME dynamique de désalignement → éviter deux leviers concurrents sur le drift.

4. Bloc 3 — Conditionnalité 🔶 GELÉ / INERTE
   → OF-conditionnel nié par 2e+2f (fondement mesuré manquant). Ne pas ouvrir
     tant qu'aucune mesure ne le rouvre.
   → 3-6 (wakeup lag) : rapport p50/p95/p99, indépendant, planifiable à tout moment.

5. Bloc 5 — Robustesse 🔴 PARALLÉLISABLE
   → 3-4 (lock MaskRegistry + frame.copy()). Aucune dépendance inter-bloc,
     lançable en parallèle des étapes 1→4.
```

**Règle commits** : un chantier = un commit minimal. Chaque commit comprend :

- Le code (delta ≤ ~10 lignes si possible)
- Le test / la vérification (script JSONL ou assertion)
- La mise à jour de ce document (nouvelle ligne de statut)

---

## 8. Invariants anti-régression

Les valeurs suivantes doivent rester inchangées sauf explicitement décidé dans un commit dédié :

| Paramètre                         | Valeur                              | Localisation                    |
| --------------------------------- | ----------------------------------- | ------------------------------- |
| `template_refresh_ms`             | 350 ms                              | `FastTrackConfig`               |
| `ncc_refresh_gate` (défaut)       | 0.30                                | `FastTrackConfig`               |
| `ncc_v_gate` (défaut)             | 0.55                                | `FastTrackConfig`               |
| Filtre CONFIRMED en entrée worker | via snapshot slow                   | `main.py` (hors périmètre fast) |
| Conversion couleur                | `cv2.COLOR_RGB2GRAY`                | `fast_track_thread.py` L204     |
| Gate bypass décidée par masque    | `_of_healthy and not _needs_resync` | `fast_track_thread.py` L278     |
| Budget mesuré par frame           | `snapshot_frame()`                  | `main.py`                       |

---

## 9. Table de traçabilité Root Causes

| Root Cause                                                           | Statut                        | Bloc / Chantier              |
| -------------------------------------------------------------------- | ----------------------------- | ---------------------------- |
| **RC-NCC-age** (scores < 0,55 / négatifs)                            | ✅ élucidée — F3-1 réalisé    | Baseline                     |
| **RC-NCC-of-alignment** (arrondi OF avant crop)                      | 🟠 ouverte                    | Bloc 2 / F3-2a               |
| **RC-OF-1** (dérive corner tracker)                                  | 🟠 ouverte                    | Bloc 2 / F3-2b               |
| **RC-NCC-couleur** (contraste WGC)                                   | 🟢 close                      | —                            |
| **RC-NCC-méthode** (méthode NCC inadaptée)                           | 🟢 close                      | —                            |
| **RC-NCC-drift** (désalignement template↔position sur skip motion=0) | 🟠 ouverte — révélée par 2f   | Famille 2g (candidate)       |
| **RC-NCC-drift** (désalignement template↔position)                   | 🟡 partiellement validée — 2g | Bloc 2 / 3-2c (gate re-sync) |

---

## 10. Statut des sondes citées dans ce document

| Sonde                        | Statut                                                           | Commentaire                 |
| ---------------------------- | ---------------------------------------------------------------- | --------------------------- |
| `F_ncc_bypass_total`         | 🔴 À CRÉER                                                       | Bloc 1 / 2a                 |
| `F_bypass_last_uid`          | 🔴 À CRÉER                                                       | Bloc 1 / 2b                 |
| `F_mask_processed_total`     | 🔴 À CRÉER                                                       | Bloc 1 / 2c                 |
| `F_ncc_v_gate`               | 🔴 À CRÉER                                                       | Bloc 1 / 2d                 |
| `F_ncc_refresh_gate`         | 🔴 À CRÉER                                                       | Bloc 1 / 2d                 |
| `F_wakeup_lag_ms`            | ✅ Existante                                                     | L197                        |
| `fast_tick_ms`               | ✅ Existante                                                     | L199                        |
| `fast_cvt_ms`                | ✅ Existante                                                     | L203                        |
| `fast_of_total_ms`           | ✅ Existante                                                     | L220                        |
| `fast_margin_ms`             | ✅ Existante                                                     | L282                        |
| `fast_ncc_call_ms`           | ✅ Existante                                                     | L284                        |
| `fast_ncc_total_ms`          | ✅ Existante                                                     | L270                        |
| `fast_publish_ms`            | ✅ Existante                                                     | L350                        |
| `F_template_age_ms`          | ✅ Existante                                                     | L286                        |
| `F_extrapolation_px`         | ✅ Existante                                                     | L254                        |
| `F_stale_skipped_total`      | ✅ Existante                                                     | L266 / L353                 |
| `F_stale_capped_total`       | 🔴 À CRÉER                                                       | Bloc 4                      |
| `F_predict_source`           | 🔴 À CRÉER                                                       | Bloc 4                      |
| `motion_staleness_capped`    | 🔴 Variables/méthodes fantômes — NE PAS créer hors Bloc 4        | ???                         |
| `motion_predict_source_slow` | 🔴 Variables/méthodes fantômes — NE PAS créer hors Bloc 4        | ???                         |
| `prob()`                     | ❌ N'existe pas — remplacer par `rate()` ou `note()`             | ???                         |
| `F_roi_source_at_skip`       | ✅ Existante — 2e (levier OF-conditionnel NIÉ)                   | Bloc 1 / 2e                 |
| `F_roi_texture_std_at_skip`  | ✅ Existante — 2f (texture RÉFUTÉE)                              | Bloc 1 / 2f                 |
| `F_roi_motion_px_s_at_skip`  | ✅ Existante — 2f (motion → proxy drift)                         | Bloc 1 / 2f                 |
| `F_roi_drift_px_at_skip`     | ✅ Existante — 2g (RC-drift partiellement validée) ; pilote 3-2c | Bloc 1 / 2g → Bloc 2 / 3-2c |
