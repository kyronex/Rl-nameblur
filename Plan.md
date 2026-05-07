# 📘 Plan v4.1 — Tracker / Plan consolidé

> **Document autoportant.** Aucune référence implicite aux plans antérieurs. En cas de doute sur le périmètre exact d'un item, un **audit de code base** est explicitement requis avant démarrage.

---

## 📐 Conventions

| Préfixe | Catégorie                                        |
| ------- | ------------------------------------------------ |
| `B-XX`  | Bug (correction d'un comportement incorrect)     |
| `A-XX`  | Architecture / hygiène structurelle              |
| `P-XX`  | Performance (conditionnée à profiling ou signal) |
| `F-XX`  | Feature (Phase 2, conditionnelle)                |

**Statuts** : 🔴 P0 bloquant · 🟢 actif · 🟡 conditionnel · 🧊 gelé

**Numérotation** : indépendante par catégorie, démarre à 01.

---

## 🗺️ Articulation globale

```text
🚨 P0 — Bugs bloquants (séquentiel strict)
   ├── B-01  Quick-fix zombies via plafonnement fast_max_drift_s
   ├── B-02  Correction association fast tracker (cause racine)
   ├── B-03  TTL temporel (cycle de vie en secondes côté Tracker)
   └── B-04  Investigation dérive dt motion
         │
         ▼
🟦 Phase 0 — Clôture stabilisation noyau
   ├── A-01  Audit cohérence noyau tracker
   └── A-02  Hygiène config & invariants
         │
         ▼
🟢 Lot 1 — Stabilisation post-prod
         │
         ▼
🟡 Lot 2 — Bande passante
         │
         ▼
🟠 Lot 3 — Performance (menu à la carte, sur signal)
   ├── P-01  Batch LK pyramide partagée (FastTrackThread)
   ├── P-02  Shi-Tomasi features (FastTrackThread)
   └── P-03  Cache config lookup compatible hot-reload
         │
         ▼
🔮 Phase 2 — Évolutions conditionnelles
   ├── F-01..F-07  LifecycleManager
   ├── F-08        Horizon dynamique motion
   ├── F-09        Thread tick dédié
   └── F-10        Affinements prédiction motion

🧊 Backlog gelé
   └── Prédiction par historique (régression pondérée)
```

**Règle d'or séquentielle** : chaque bloc se termine avant que le suivant démarre. Toute feature peut être supprimée du V4 et reportée dans un plan ultérieur sans casser la chaîne.

---

## 🚨 P0 — Bugs bloquants (versions révisées)

### 🟢 B-01 — Quick-fix zombies via plafonnement du drift fast `[LIVRÉ — partiel acté]`

- **Symptôme initial** (mesuré via instrumentation `[B01]`) :
  - Masks confirmés vivent **5 à 9 s** au lieu de ~0.4 s post-disparition.
  - Cas type initial : `uid=4 age=8.63s state=CONFIRMED ttl=24 matches=177 (slow=22 fast=155) last_match=fast 0.01s_ago`.
  - Mécanisme de purge **fonctionnel** (masks jamais matchés expirent en 0.2-0.6 s comme prévu).

- **Cause racine identifiée** : le **fast tracker ré-associe à tort** ses détections sur des UIDs qui ne correspondent plus à un objet réel. Chaque match fast réinitialise `ttl=25` via `mark_matched`, ce qui empêche la transition vers `LOST` et donc la purge.
  → **Bug d'association amont, pas bug de TTL.** Traité en profondeur par B-02.

- **Décision de scope B-01** : on **ne touche pas** au fast tracker. On applique un **garde-fou côté tracker** déjà existant mais mal calibré : `fast_max_drift_s`.

- **Approche appliquée** :
  1. `masks.fast_max_drift_s` ramené de `1.5` → **`0.5`** dans `config.yaml`, commenté `# QUICK-FIX B-01, à recalibrer après B-02`.
  2. **Aucun changement de code** — garde-fou existant dans `apply_fast_direct` :

     ```python
     if drift > self.cfg.fast_max_drift_s:
         drift_skipped += 1
         continue
     ```

  3. Instrumentation `[B01]` de `registry.py` **maintenue active jusqu'à validation B-02**.

- 🔍 **Audit code base** : ✅ effectué pré-livraison.

- **Effort réel** : 5 min (édition config) + session de validation.

- **Résultats observés sur session de validation** :
  - ✅ Garde-fou opérationnel : compteur `drift_skipped` non nul corrélé aux disparitions réelles (ex. `received=4 applied=0 drift_skipped=4 max_drift=1.0s`).
  - ✅ Disparition du pic catastrophique (plus de cas `age=8s+` **sans aucun filtrage**, comme `uid=4` initial).
  - ✅ Purge des masks jamais matchés intacte (`lifetime ≈ 0.24s`).
  - ✅ Pas de régression visuelle observée.
  - ⚠️ **Critère initial "< 1.5 s" non atteint** : zombies résiduels observés à `lifetime` 3.79 s → 8.38 s, alimentés par des matches `last_match=fast 0.00s_ago` qui passent le seuil drift.
  - ⚠️ Observation collatérale : le **slow associator** ré-attribue lui aussi des détections sur des UIDs morts (ex. `uid=23 last_match=slow 0.11s_ago` sur zombie 3.59 s) → **B-02 doit auditer slow + fast**, pas uniquement fast.

- **Critères de succès — version actée** :
  - ✅ Compteur `drift_skipped` opérationnel et corrélé aux disparitions.
  - ✅ Suppression du régime catastrophique (zombies sans aucun filtrage).
  - ✅ Pas de régression visuelle.
  - ⚠️ **Critère "lifetime < 1.5 s" reporté à B-02** — atteignable uniquement en traitant la cause racine (sur-association slow + fast).

- **Dette créée — confirmée** :
  - Cause racine intacte : sur-association sur UIDs morts continue de produire des matches `0.00s_ago`. → résorbée par **B-02 (périmètre élargi)**.
  - `fast_max_drift_s = 0.5` reste en place comme **filet permanent jusqu'à B-02**, à recalibrer ensuite.

- **Statut** : ✅ **livré dans son scope**. Cause racine renvoyée à B-02 avec périmètre élargi (slow + fast).

- **Bloque** : rien.

---

### 🟢 B-02 — Correction d'association (slow + fast) `[LIVRÉ]`

- **Symptôme résiduel après B-01** :
  - Zombies `lifetime` 3-8 s avec `last_match=fast 0.00s_ago` et/ou `last_match=slow 0.1s_ago`.

- **Cause racine** : deux producteurs ré-attribuaient des détections à des UIDs morts :
  1. **Associator (slow)** — gating trop laxiste, continuité insuffisante.
  2. **FastTrackThread (fast)** — NCC trop permissif, gate géographique trop large, stale republié.

- **Corrections appliquées** :

  _Config (`config.yaml`)_ :
  - `ncc_threshold` : `0.40` → `0.65`
  - `geo_gate_base_radius_px` : `450` → `150`
  - `geo_gate_velocity_k` : `3.0` → `1.5`
  - `fast_max_drift_s` : `0.5` — **valeur définitive validée session 18s**
    (max_drift observé ≤ 0.9s, aucun zombie pathologique, drift_skipped actif et utile).

  _Code — Fast_ :
  - **S1** : gate géographique activé avant NCC.
  - **S2** : seuil NCC renforcé (`0.65`).
  - **F1** : stale republish supprimé — branche stale ne pousse plus dans `results`.

  _Code — Slow_ :
  - **S3** : champ `last_slow_ts` ajouté au registry.
  - **S4** : gate de continuité dans l'associator slow — refus d'associer un UID absent depuis > N frames slow.

- **Résultats** :
  - ✅ `fast_mask_lost` augmente sur scène statique.
  - ✅ `fast_stale_skipped` remplace `fast_stale_used`.
  - ✅ Nombre de masks affichés baisse sur fond statique.
  - ✅ Hot-reload propre.
  - ✅ **Validation empirique session 18s** (`drift=0.5`) :
    - `unknown=0` sur 100% des EXPIRE (aucune régression d'attribution match).
    - Lifetime médian ≈ 1.5 s, p95 ≈ 2.1 s.
    - Outlier unique `lifetime=7.23s` jugé légitime (132 matches, ratio slow/fast 1:3 sain, `last_match=slow 0.83s_ago`).
    - `max_drift` plafonne à 0.9 s → **monter à 1.0+ s n'apporterait aucun gain mesurable**

- **Dette résorbée** : cause racine B-01 traitée côté fast ET slow.

- **Dette créée** :
  - Cap adaptatif stale sur scène statique → **B-03**.
  - Instrumentation `[B01]` retirable immédiatement (validation faite) → débloque **B-04**.
  - Dégradation FPS observée 84→44 sur session 18s (probablement overhead logs INFO + accumulation masks)
    → à investiguer hors-scope B-02, **suspect : à scoper en B-03b ou ticket dédié**.
- **Statut** : ✅ **livré dans son scope**.
- **Bloque** : retrait `[B01]`, **B-03**, **B-04**.

---

### 🟢 B-03 — TTL temporel (cycle de vie en secondes) `[LIVRÉ]`

- **Périmètre strict** : cycle de vie côté `Tracker` / `MaskRegistry` uniquement.
  ⚠️ Ne pas toucher `detect.fast.max_stale_frames` (frames par design, FastTrack event-driven).

- **Cause racine** : vieillissement des `MaskState` en ticks → dépendant de la cadence main loop.

- **Symptômes confirmés (logs post-B-02)** :
  - Cycle CREATE/EXPIRE perpétuel d'UIDs orphelins (lifetime ≈ 0.13 s, matches=0) sur masks non re-matchés par le slow suivant.
  - Mask uid=372 : EXPIRE 0.19 s après dernier match légitime → trop court pour couvrir une pause d'immobilité.

- **Changements config** :

  ```yaml
  tracker:
    lifecycle:
      lost_after_s: 0.3 # CONFIRMED → LOST si pas de match depuis N secondes
      expire_after_lost_s: 1.0 # LOST → purge si LOST depuis N secondes
  # supprimé : équivalents en frames côté tracker
  # NON TOUCHÉ : detect.fast.max_stale_frames
  # NON TOUCHÉ : masks.fast_max_drift_s = 0.5 (valeur définitive fixée par B-02)
  ```

  **Calibration finale validée empiriquement** :
  - Valeurs initiales proposées (`1.0` / `1.5`) jugées trop laxistes après tests → resserrées à `0.3` / `1.0`.
  - Compromis retenu : purge orphelins en ~1 s (vs. cycle perpétuel avant), tout en couvrant les pauses d'immobilité courtes (< 0.3 s) sans EXPIRE prématuré.

- **Changements code appliqués** :
  - `MaskState` : ajout `last_seen_ts`, `lost_since_ts`, suppression TTL en ticks.
  - `MaskRegistry` : transitions par delta temporel.
  - Marquage de match : `last_seen_ts = ts`.

- **Résultats — validation empirique session post-fix** :
  - ✅ EXPIRE orphelins (`matches=0`) : lifetime 1.02–1.13 s (vs. cycle perpétuel avant).
  - ✅ EXPIRE tracked normaux : lifetime 1.4–2.5 s, cohérent avec (`lost_after_s` + `expire_after_lost_s` + dernier match).
  - ✅ Long-runners stables : uid=0 (8 s), uid=9 (10.93 s, 175 matches, ratio slow/fast 40:135 sain).
  - ✅ Transitions CONFIRMED → LOST observées correctement quand `since_last_seen > 0.3 s` (uid=10 à 0.77 s, uid=4 à 1.13 s).
  - ✅ Purge cohérente indépendante de la cadence main loop (testée sur FPS variable 45–88).
  - ✅ Tous les `EXPIRE` affichent `lost_for=1.00–1.02 s` → `expire_after_lost_s` respecté avec précision.

- **Dette résorbée** :
  - Cycle CREATE/EXPIRE perpétuel : éliminé.
  - Dépendance cadence main loop : supprimée.

- **Dette créée / héritée** :
  - **Burst de faux positifs slow** : rafales de CREATE avec `matches=0` (uid=20→37 sur ~3 s) → pollution registry, hors-scope B-03 → **à scoper côté slow detector** (seuil confiance / NMS).
  - **`motion.dt avg=575–982 ms / capped=99–100%`** : pathologie constante, indépendante du tuning lifecycle, handicape fast tracking en permanence (`drift_skipped` fréquents) → **B-04b prioritaire**.
  - Logs `ZOMBIE-SUSPECT` (B-01) toujours actifs en INFO → saturation logs, à passer DEBUG ou retirer.

- **Statut** : ✅ **livré dans son scope**.
- **Bloque** : F-08, **B-04b** (motion.dt saturé).

---

### 🔴 B-04 — Investigation dérive `dt` motion + nettoyage post-B-03 `[ex-B-02 + ex-B-03 fusionnés, recadré post-B-03]`

- **Trois périmètres** :

  **B-04a — Cap adaptatif stale (dette B-02)** :
  - Sur fond statique prolongé, `max_stale_frames` permet encore N cycles avant `fast_mask_lost`.
  - Piste : NCC threshold relevé à `0.70+` quand `velocity == 0` depuis > 1 s.
  - **Statut révisé post-B-03** : ✅ **fermé par observation**. Logs session post-fix : aucun zombie fast pathologique, long-runners (uid=0/9) stables avec ratio slow/fast sain (40:135). Rouvrir uniquement si régression observée.

  **B-04b — Dérive `dt` motion** `[PRIORITAIRE]` :
  - Symptôme historique : `motion.dt` dérive (750 ms → 2 300 ms), `capped_pct` saturé.
  - **Symptôme reconfirmé logs post-B-03** : `motion.dt avg=575–982 ms / max=1431–1802 ms / capped=99–100 %` sur **toute** la session (FPS 45–88). Attendu ≈ 11–22 ms.
  - **Impact opérationnel mesuré** : `drift_skipped` fréquents avec `max_drift=0.5–0.8 s` → fast tracker handicapé en permanence, applied/received ratio dégradé (parfois 0/1).
  - **Hypothèse principale** : `predict_position` calcule `dt = now - last_detected_ts` sur masks en attente de re-match → cappé systématiquement. **Indépendant de B-03**, confirmé par persistance post-fix.
  - **Hypothèse secondaire** : `last_detected_ts` non harmonisé avec `last_seen_ts` introduit par B-03 → possible double source de vérité.
  - **Actions** :
    1. Audit `predict_position` : tracer site exact de `dt`, identifier ref temporelle utilisée.
    2. Harmoniser `last_detected_ts` ↔ `last_seen_ts` (B-03).
    3. Vérifier que `dt` motion utilise un delta **inter-frames motion** et non `now - last_match`.

  **B-04c — Dégradation FPS session longue** :
  - Session 18s historique : 84 FPS → 44 FPS, dégradation continue et monotone.
  - **Observation post-B-03** : FPS oscille 45–88 sur ~14 s sans tendance monotone claire (88.3 → 50.2 → 60.7 → 49.9 → 58.0). **Dégradation atténuée mais non éliminée.**
  - **Hypothèses restantes** :
    - Overhead logs INFO `[FAST-APPLY]` (haute fréquence) + `ZOMBIE-SUSPECT` toujours actifs.
    - Couplage avec B-04b : `drift_skipped` à chaque frame motion = surcharge CPU.
  - **Actions préalables** (avant profilage) :
    1. Passer `[FAST-APPLY]` et `ZOMBIE-SUSPECT` en DEBUG (gain attendu : significatif).
    2. Retirer commentaire `# QUICK-FIX B-01`.
    3. Re-mesurer session > 30 s.
  - **Si FPS toujours instable post-cleanup** : profilage requis (cProfile / py-spy).

  **B-04d — Burst faux positifs slow detector** `[NOUVEAU, hérité B-03]` :
  - Symptôme : rafales de CREATE avec `matches=0` (ex. uid=20→37 sur ~3 s) → pollution registry, EXPIRE en cascade.
  - **Hors-scope motion/tracker** : pathologie côté slow detector (seuil confiance trop bas / NMS insuffisant / faux positifs YOLO sur transitions de scène).
  - **Statut** : ⚠️ **à scoper en ticket dédié B-05** (slow detector tuning), pas de fix dans B-04.

- 🔍 **Audit requis** : sondes `motion.dt` / `capped_pct`, `predict_position`, sites de calcul `dt`, harmonisation `last_detected_ts` vs `last_seen_ts`.

- **Effort estimé** :
  - B-04a : 0 (fermé).
  - B-04b : 2–4 h (audit + fix + validation).
  - B-04c : 30 min cleanup + 15 min re-mesure ; +2 h si profilage requis.
  - B-04d : déplacé en B-05.
  - **Total B-04 : 3–6 h.**

- **Critères de succès** :
  - `motion.dt.avg < 2 × (1/FPS)` (ex. < 33 ms à 60 FPS).
  - `capped_pct < 10 %`.
  - `drift_skipped` rate < 5 % sur scène normale (vs. ~25 % actuel).
  - FPS stable sur session > 30 s (variance < 15 %).
  - Logs `[FAST-APPLY]` / `ZOMBIE-SUSPECT` en DEBUG.

- **Ordre d'exécution recommandé** :
  1. B-04c cleanup logs (15 min, gain immédiat lisibilité + FPS).
  2. B-04b audit + fix `predict_position` (cœur du problème).
  3. Validation conjointe sur session > 30 s.
  4. Si OK → fermer B-04, ouvrir B-05 (slow detector).

- **Bloque** : F-08.

---

### 🔴 B-05 — Slow detector : faux positifs en burst `[NOUVEAU, hérité B-03]`

- **Symptôme** : rafales de CREATE slow avec `matches=0` (uid orphelins purgés en ~1 s par B-03).
- **Exemple logs B-03** : uid=18, 19, 24, 25, 26, 34, 35, 36, 37 — tous EXPIRE avec `matches=0 last_match=create`.
- **Cause probable** : seuil confiance YOLO trop bas, NMS insuffisant, ou détections une-frame sur transitions visuelles (loading, transitions UI Rocket League).
- **Impact** :
  - Pollution registry (UIDs gaspillés).
  - Cycle CREATE→EXPIRE inutile (CPU + logs).
  - Risque masquage transitoire indésirable côté rendu.
- **Actions à scoper** :
  - Audit `detect.slow.confidence_threshold`.
  - Vérifier NMS (IoU threshold, agnostic).
  - Ajouter gate "confirmé après N détections consécutives" avant CREATE (équivalent slow de la confirmation fast).
- **Effort** : 1–3 h.
- **Bloque** : F-08 (qualité), pas critique B-04.

---

## 🟦 Phase 0 — Clôture stabilisation noyau

**Trigger** : ✅ B-01, B-02, B-03, B-04, B-05 livrés.

### 🟢 A-01 — Audit cohérence noyau tracker

- **Périmètre** : `Tracker`, `Associator`, `MaskRegistry`, `MaskState`, propagation `ts` (`perf_counter`), interactions avec `DetectThread` / `FastTrackThread`.
- 🔍 **Audit requis** : modules touchés sur les 3 derniers mois, vérification des invariants (horloge unique, contrat `tick()` public, étanchéité frames vs secondes, intégrité du flux `MaskState`).
- **Livrable** : rapport d'audit + correctifs mineurs si nécessaire.
- **Effort** : 2-3 h.
- **Critère de succès** : invariants README documentés et vérifiés ; aucune divergence runtime non justifiée.

### 🟢 A-02 — Hygiène config & invariants

- **Objectif** : valider au chargement (et à chaque hot-reload) tous les invariants critiques de `config.yaml`.
- **Invariants à vérifier (liste minimale, à compléter par audit)** :
  - `tracker.lifecycle.expire_after_lost_s >= tracker.lifecycle.lost_after_s` (post B-04).
  - **Étanchéité unités** : aucune clé en secondes dans `detect.fast.*` (qui doit rester en frames), aucune clé en frames dans `tracker.lifecycle.*`.
  - Plages de valeurs cohérentes pour `detect.fast.ncc_threshold ∈ [0,1]`.
  - Cohérence `screen.capture_fps` ≥ `screen.vcam_fps` (à confirmer par produit).
- 🔍 **Audit requis** : recenser exhaustivement les clés consommées par le pipeline tracker et leurs types attendus.
- **Effort** : 1-2 h.
- **Critère de succès** : config invalide → erreur explicite au démarrage **et au hot-reload** (pas de dégradation silencieuse).

---

## 🟢 Lot 1 — Stabilisation post-prod

- **Statut** : volontairement non détaillé.
- **Trigger** : Phase 0 livrée + observations prod sur session de référence.
- 🔍 **Audit requis au démarrage** : recenser les régressions / signaux faibles observés depuis la livraison de Phase 0 avant de figer le contenu.

---

## 🟡 Lot 2 — Bande passante

- **Statut** : volontairement non détaillé.
- **Trigger** : Lot 1 livré + signaux explicites sur l'utilisation des ressources réseau / capture / send.
- 🔍 **Audit requis au démarrage** : caractériser les goulots d'étranglement effectifs (sondes `send`, `fast_wakeup_lag`, `frames`) avant de figer le contenu.

---

## 🟠 Lot 3 — Performance (menu à la carte)

> **Principe** : aucun item P-XX ne démarre sans signal mesurable. Tous les critères de succès s'adossent aux sondes bench officielles du README.

### 🟡 P-01 — Batch LK sur pyramide partagée dans `FastTrackThread`

- **Trigger** : `N MaskState simultanés ≥ 3` observé en prod.
- **Description** : un seul appel LK sur pyramide partagée, au lieu de N appels indépendants, dans la phase OF de `FastTrackThread`.
- 🔍 **Audit code base requis** : confirmer le point d'appel LK actuel dans `FastTrackThread` et la structure de la pyramide.
- **Effort** : ~3 h.
- **Mesure de succès** :
  - Réduction de la sonde **`fast_of_total`** proportionnelle à N.
  - Pas d'impact négatif sur **`fast_ncc_confirmed`**.
- **Risque** : faible.

### 🟡 P-02 — Shi-Tomasi features

- **Trigger** : fallbacks stale fréquents — sonde **`fast_stale_used`** élevée et/ou **`fast_mask_lost`** élevée.
- **Description** : remplacer la grille fixe de features LK par une détection Shi-Tomasi (`cv2.goodFeaturesToTrack`) dans `FastTrackThread`.
- 🔍 **Audit code base requis** : confirmer la stratégie de génération de features actuelle.
- **Effort** : ~2 h.
- **Mesure de succès** : baisse mesurable de `fast_stale_used` et/ou `fast_mask_lost`.
- **Risque** : faible.

### 🟡 P-03 — Cache config lookup compatible hot-reload

- **Trigger** : profiling confirme un overhead non négligeable sur `config.get()` en hot path (ex. `_adaptive_margin` ou équivalent).
- **Description** : memoization du lookup config, **avec invalidation explicite sur événement de hot-reload** (le README documente le watcher).
  - Implémentation possible : epoch / version interne au singleton config, incrémentée à chaque reload ; cache invalidé si epoch change.
- 🔍 **Audit code base requis** :
  - Identifier les hot paths qui appellent la config par frame.
  - Identifier l'API d'event/notification du watcher de hot-reload.
- **Effort** : 30 min – 1 h (selon API du watcher).
- **Mesure de succès** :
  - Réduction mesurable au profileur sur la fonction ciblée.
  - Test de régression hot-reload (modification YAML à chaud → cache rafraîchi).
- **Risque** : **faible** (interaction avec hot-reload à ne pas casser).

---

## 🔮 Phase 2 — Évolutions conditionnelles

**Principe** : aucune feature n'est démarrée sans son trigger. Phase 2 peut rester totalement dormante si aucun signal ne se déclenche.

### 🟡 F-01 à F-07 — LifecycleManager

- **Objectif** : centraliser la gestion du cycle de vie des `MaskState` dans un composant dédié.
- **Trigger global** : signal métier / test / bug indiquant que la dispersion actuelle (entre `Tracker`, `MaskRegistry`, `Associator`) devient un blocage de maintenance.
- 🔍 **Audit code base requis** : recenser tous les sites de logique de cycle de vie avant de figer le découpage F-01..F-07 (sept sous-tâches à figer au démarrage de la feature).
- **Contrainte de compatibilité (invariant README)** : préserver l'API publique `Tracker.tick(detections, ts)` / `get_confirmed_masks()`. Le LifecycleManager est un détail d'implémentation interne.
- **Effort** : ~1 jour, +0,5 j si interaction avec F-09.
- **Critères de succès** :
  - Transition cycle de vie unifiée.
  - Couverture test unitaire complète sur le LifecycleManager.
  - Aucune régression sur session de référence.

### 🟡 F-08 — Horizon dynamique motion (EMA latence détection)

- **Triggers cumulatifs (les deux requis)** :
  1. Signal métier / mesure prod sur variance d'horizon.
  2. **B-03 résolu** (sinon anti-pattern documenté).
- 🔍 **Audit code base requis** :
  - Confirmer existence et nom canonique de la clé config `dt_cap` (non listée dans le tableau « paramètres clés » du README — tableau explicitement non exhaustif).
  - Confirmer le module motion et son point d'injection.
- **Garde-fous obligatoires** :
  - Clamp anti-spike sur `dt`.
  - Cold-start : si `< 5 samples`, fallback sur `dt_cap` statique.
  - EMA globale (pas par masque).
  - Damping pour absorber les transitoires.
  - Log HUD du `dt_cap` courant.
- **Effort** : 1 h dev + 30 min calibration.
- **Critères de succès** :
  - `motion.dt` reste stable post-déploiement.
  - `capped_pct` stable < 10 %.
  - Pas de régression overlap rect moyen.
- **Anti-pattern documenté** : déployer F-08 sur un système dont les `dt` sont aberrants → l'EMA absorbe le bug et masque le diagnostic.

### 🟡 F-09 — Thread dédié pour la phase TTL/éviction du `MaskRegistry`

> ⚠️ **Recadrage important** : F-09 ne peut **pas** externaliser `Tracker.tick()` complet, car `tick()` orchestre aussi `Associator.match()` (gating + Hungarian) sur les détections fraîches consolidées par le main loop. Externaliser `tick()` casserait le contrat README.

- **Périmètre exact** : externaliser **uniquement la phase TTL / éviction** du `MaskRegistry` dans un thread dédié cadencé à fréquence fixe (ex. 30 Hz), pour garantir la régularité de la purge indépendamment de la cadence main loop.
- **Trigger** : jitters de purge mesurés au-delà d'un seuil acceptable **après B-04 livré** (la temporalisation seule ne suffit pas, il faut aussi régulariser l'appel).
- 🔍 **Audit code base requis** :
  - Identifier le point d'appel actuel de la phase éviction dans `Tracker.tick()`.
  - Caractériser la thread-safety du `MaskRegistry` (lock interne ? structures partagées avec main loop ?).
  - Définir le contrat de synchronisation entre `Associator.match()` (main loop) et la phase éviction (thread dédié).
- **Effort** : 2-3 h, +complexité thread-safety.
- **Risque** : interaction avec LifecycleManager (F-01..F-07). **Ordre recommandé** : F-01..F-07 puis F-09 si les deux sont déclenchés.
- **Anti-patterns documentés** :
  - Déployer F-09 sans B-04 (revient à cadencer un compteur en ticks → aucun bénéfice).
  - Externaliser `Tracker.tick()` complet au lieu de la seule éviction (casse le contrat tracker).

### 🟡 F-10 — Affinements prédiction motion

- **Statut** : slot réservé, **à détailler post-F-08**
- **Trigger** : F-08 livré et stable + mesure prod montre que la vélocité EMA seule ne suffit pas sur certains cas d'usage à caractériser
- **Description prévue** : intégrer les briques réutilisables issues de la R&D prédiction par historique (cf. backlog gelé) qui apportent un gain net sans dupliquer F-08
- **Action de cadrage** : lors de la livraison de F-08, ouvrir une revue dédiée → décider quelles briques migrent en F-10 et lesquelles restent gelées
- **Effort** : non chiffré (slot)
- **Note** : ce slot existe pour **garantir que le travail R&D antérieur ne soit pas perdu** sans s'engager sur un périmètre prématurément

---

## 🧊 Backlog gelé

### Prédiction par historique (régression pondérée)

- **Doc source** : `docs/backlog/prediction-historique.md` (à maintenir indépendamment du V4)
- **Statut** : gelé — ne pas démarrer
- **Conditions de réveil** (toutes requises) :
  1. F-08 livré et stable en prod
  2. Mesure prod confirme une limite identifiée de la vélocité EMA sur des cas d'usage caractérisés
  3. Revue F-10 conclut que cette R&D apporte un gain net sur des briques précises
- **Action en cas de réveil** : décomposition fine par brique, intégration sélective en F-10 ou nouveau lot dans un plan ultérieur

---

## 📁 Hygiène documentaire

- `docs/closed-tickets.md` : tickets clos au fil de l'eau, avec date et lien commit.
- `docs/backlog/prediction-historique.md` : R&D gelée, marquée explicitement « GELÉ — voir Plan v4.1 §Backlog ».
- `docs/session-de-reference.md` : protocole de bench canonique réutilisable pour B-02, F-08, Lot 3, F-10.
  - 🔍 **Audit requis** : la session de référence doit explicitement consommer :
    - **Sondes officielles README** : `capture_wait`, `slow_poll`, `match`, `fast_poll`, `predict`, `blur`, `frames`, `masks_total`, `fast_wakeup_lag`, `fast_tick`, `fast_of_total`, `fast_ncc_total`, `fast_ncc_confirmed`, `fast_stale_used`, `fast_mask_lost`, `send`.
    - **Sondes motion à confirmer** : `motion.dt`, `capped_pct` (existence à valider lors de l'audit B-03).
  - Définir le footage tagué et les scénarios standards avant le premier usage en B-03.

---

## ✅ Critères globaux V4.1

- **Conformité README** : ✅ types (`MaskState`), composants (`MaskRegistry`, `Associator`, `Tracker`, `FastTrackThread`), API publique (`tick`, `get_confirmed_masks`), horloge (`perf_counter`), hiérarchie YAML, hot-reload config.
- **Étanchéité unités** : ✅ frames côté FastTrack (`detect.fast.max_stale_frames`), secondes côté Tracker (`tracker.lifecycle.*`), invariant verrouillé par A-02.
- **Sondes bench** : ✅ critères de succès Lot 3 et tests de régression adossés aux sondes officielles README.
- **Séquentialité simple** : ✅ chaque bloc indépendant du suivant.
- **Reportabilité** : ✅ toute feature peut être retirée vers un plan ultérieur sans casser la chaîne.
- **Couverture bugs** : ✅ B-01, B-02, B-03, B-04 explicites en P0..
- **Préservation R&D antérieure** : ✅ via F-10 + backlog gelé documenté.
- **Triggers mesurables** : ✅ chaque item conditionnel a un signal explicite.
- **Audits explicites** : ✅ chaque zone d'incertitude marquée 🔍, prérequis bloquant à la livraison.

---

## 📌 Points d'attention pour la suite

1. **Sections Lot 1 et Lot 2** : volontairement non détaillées car leur contenu dépend d'observations futures. Audit requis au démarrage de chaque lot.
2. **F-01..F-07** : sept sous-tâches à figer au démarrage de la feature, après audit du code base. Contrainte non négociable : préserver l'API publique `Tracker.tick()` / `get_confirmed_masks()`.
3. **F-10** : slot délibérément vide, à remplir post-F-08.
4. **F-09** : recadré sur la phase TTL/éviction uniquement — ne jamais externaliser `tick()` complet.
5. **Tous les 🔍 audits code base** sont des prérequis bloquants à la livraison de l'item correspondant — ne pas les sauter.

---
