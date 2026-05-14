# 📘 Plan v4.3 — Tracker / Plan consolidé

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

**Numérotation** : indépendante par catégorie, démarre à 01, **strictement séquentielle**.

**Règle de séquentialité dure** : un ticket ne peut être démarré que si **toutes ses préconditions sont livrées et validées**. Aucun parallélisme implicite.

---

## 🗺️ Articulation globale

```text
🚨 P0 — Bugs bloquants (séquentiel strict)
   ├── B-01  Quick-fix zombies via plafonnement fast_max_drift_s            ✅
   ├── B-02  Correction association fast tracker (cause racine)             ✅
   ├── B-03  TTL temporel (cycle de vie en secondes côté Tracker)           ✅
   ├── B-00  Anomalies config #2, #3, #4                                    ✅
   ├── B-04  Investigation dérive dt + nettoyage post-B-03                  🔴
   │    ├── livrable interne : correction compute_predicted_rect (#60)
   │    └── déclencheur humain : footage tagué teleport (pendant B-04)
   ├── B-00b Anomalie config #1 (teleport_thresh)                           🔴
   ├── B-05  Slow detector : faux positifs en burst                         🔴
   └── B-06  Auto-keepalive masks stationnaires (patch tactique)            🔴
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
         │
         ▼
🔬 Audits différés (long terme)
   └── B-07  Refonte sémantique last_seen_ts (options F vs G)

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

- **Effet de bord acté** : `lost_after_s = 0.3 s` calibré agressivement. À haut FPS sur scènes statiques, ce choix génère un cycle CONFIRMED/LOST/EXPIRE/re-CREATE qui sera **traité par B-06**. Ne pas remonter `lost_after_s` sans avoir livré et observé B-06.

---

### 🟢 B-00 — Anomalies config préalables `[LIVRÉ]`

> **Justification d'insertion** : 4 anomalies détectées dans `config.yaml` lors de la revue v4.3, antérieures au plan. Doivent être traitées avant A-02 (pour ne pas polluer l'audit invariants) et avant B-05 (l'anomalie #1 peut générer des symptômes cohérents avec les bursts CREATE observés).
>
> **Réorganisation** : l'anomalie #1 est extraite en **B-00b** (préconditions non satisfaites à ce stade). B-00 traite uniquement #2, #3, #4.

- **Préconditions dures** : aucune.

- **Anomalies traitées** :
  1. ~~**🔴 `motion.teleport_thresh < vx_max × dt_cap`**~~ → **déplacé en B-00b** (dépend de stats motion fiables post-B-04 et de footage tagué).

  2. **🟢 `motion.dt_slow_max` vs `tracker.lifecycle.lost_after_s` — RÉSOLU** (faux positif)
     - Audit code (motion.py, registry.py, mask.py) : les deux clés sont strictement orthogonales.
     - `lost_after_s` : TTL registry, basé sur last_seen_ts (rafraîchi par tout match slow+fast).
     - `dt_slow_max` : garde anti-bruit sur calcul vélocité, basé sur last_slow_ts (slow uniquement).
     - Aucun couplage code. La comparaison numérique 0.5 vs 0.3 n'a pas de sens sémantique.
     - Action appliquée : commentaires YAML reformulés pour lever l'ambiguïté. Aucune valeur modifiée.

  3. **🟢 `detect.fast.roi_margin` — clé morte** SUPPRIMÉE
     - Audit fast_track_thread.py : seul appelant `_ncc_on_roi` passe toujours
       `margin=_adaptive_margin(...)`, le fallback `snap.roi_margin` n'était jamais atteint.
     - Action appliquée :
       - Suppression de la clé dans `config.yaml`.
       - Suppression du champ `roi_margin` et de sa validation dans `FastTrackConfig`.
       - Resserrement de la signature `_ncc_on_roi(..., margin: int)` (paramètre obligatoire,plus d'`Optional`).
     - Effet : zéro code mort, contrat explicite, impossibilité de retomber dans le piège.

  4. **🟢 `masks.adaptive_margin.base == .min` — redondance volontaire ASSUMÉE**
     - Audit fast_track_thread.py : formule `_adaptive_margin` monotone croissante
       (speed≥0, dt≥0, factor≥0), donc le plancher `max(am_min, ...)` est inerte
       sur le hot path tant que `am_min == am_base`.
     - Décision produit retenue : **conserver am_min == am_base** comme garde-fou structurel
       (filet contre une régression future de la formule, ex. introduction d'un terme décroissant).
     - Action appliquée :
       - Commentaires YAML explicites sur les trois clés `base` / `min` / `max`.
       - Warning logging au chargement dans `FastTrackConfig.__post_init__` si `am_min == am_base`,
         traçant la dette pour tout futur lecteur.

- **Critères de succès** :
  - ✅ Anomalie #2 résolue (sémantique documentée, faux positif tracé).
  - ✅ Anomalies #3 et #4 résolues (clé supprimée + warning au chargement).
  - ✅ Aucune régression observée sur session de référence post-patch.
  - ✅ Hot-reload propre (warning ré-émis si condition `am_min == am_base` reste vraie).

- **Dette résorbée** : 3 anomalies config (#2, #3, #4) sur les 4 identifiées en revue v4.3.
- **Bloque** : ❌ plus rien .

- **Statut** : ✅ **livré dans son scope**.

---

### 🔴 B-04 — Investigation dérive `dt` motion + nettoyage post-B-03 `[MIS À JOUR post-session JSONL]`

- **Préconditions dures** :
  - B-03 livré et validé ✅
  - Sondes `motion.dt` / `capped_pct` opérationnelles ✅ (héritées B-03, renommage `staleness_slow` livré dans B-04b ci-dessous)
  - Session de référence disponible ✅
  - **[AJOUT]** Footage de référence avec au moins un vrai teleport tagué — ⚠️ précondition humaine pour B-00b, à planifier.

---

- **Trois périmètres** :

  **B-04a — Cap adaptatif stale (dette B-02)** :
  ✅ **FERMÉ par observation.** Aucun zombie fast pathologique, long-runners stables. Rouvrir uniquement si régression observée.

  **B-04b — Dérive `dt` motion** :
  ✅ **FERMÉ.**
  - **Filiation** : traite la dette héritée B-03 (`motion.dt avg=575–982 ms / capped=99–100%`).
  - **Cause racine traitée** : `predict_position` calculait `dt = now - last_detected_ts` (référence temporelle incorrecte). Migré vers `last_slow_ts` — sémantique correcte, site unique.
  - **Renommage sonde** : `motion.dt` → `staleness_slow` (reflète la sémantique réelle : délai depuis dernière détection slow, pas latence prédiction).
  - **Correction #60** : `compute_predicted_rect` rendue pure — n'alimente plus les sondes. Sonde alimentée uniquement depuis `predict_position` (1×/mask/tick).
  - **Bug `get_and_reset_stats()`** : clé dupliquée `staleness_slow_max_ms` / `staleness_slow_sum_ms` absente → corrigé, crash `KeyError` résolu.
  - **Calibration `dt_cap`** : valeur initiale `0.10 s` systématiquement cappée (`capped=100%`). Recalibrée à **`0.35 s`** sur la base des métriques réelles.
  - **Validation `dt_cap=0.30`** (session intermédiaire) :
    - Régime nominal FPS ≥ 120 : `capped=7–11%` ✅ sous le seuil cible.
    - Régime moyen FPS 125–141 : `capped=16–19%` ⚠️ légèrement au-dessus.
    - Régime bas FPS < 60 : `capped=12–44%` — corrélé à la montée de `staleness_slow max` (385–428 ms). **Jugé acceptable** : dégradation mécanique liée au ralentissement slow, non bloquante.
  - **Valeur définitive retenue : `dt_cap: 0.35`**. Critère de succès réévalué :
    - `capped_pct < 10%` en régime nominal **main loop ≥ 120 FPS** — critère dur, conditionné à un main loop nominal (cf. B-04c).
    - `capped_pct < 50%` à main loop < 60 FPS — critère souple, non bloquant pour fermeture.
    - **Note de calibration** : seuils initiaux établis sur boucle capture 60–80 FPS. Post-optimisations capture, le facteur déterminant est devenu le FPS **main loop**, pas le FPS capture. Tant que B-04c n'a pas stabilisé le main loop à régime nominal, `capped_pct` observé reste structurellement supérieur — comportement attendu, non bloquant.
  - **Métriques de référence** : `staleness_slow avg=233–289 ms / max=300–445 ms` en régime nominal.

  **B-04c — Dégradation FPS session longue** :
  - **Statut** : ⚠️ **non clôturé.**

- **Symptôme** :
  - **Symptôme initial (pré-instrumentation B-04b)** : FPS oscille 44–204 (forte variance). Corrélation supposée avec présence/absence de masks actifs (C=4 → FPS bas, C=0 → FPS haut).
  - **Symptôme confirmé (session JSONL post-B-04b)** : FPS main loop 30–42 fps (régime nominal), pics 67–72 fps (régime allégé). La fourchette `44–204` du symptôme initial est **caduque** — résultait d'une mesure agrégée non fiable avant instrumentation bench dédiée.

- **Hypothèse principale (plan original)** : ~~surcoût floutage + fast tracker sur masks actifs~~ → **INVALIDÉE par session JSONL.**
  - Les corrélations observées (cf. tableau d'audit ci-dessous) ne mettent pas en évidence de surcoût dominant sur le couple floutage / fast tracker à charge masks active.
  - **Nouvelle hypothèse à formuler** lors de l'audit fichier-par-fichier (étape 4 protocole) — pas de pré-engagement avant données.

- **Instrumentation bench posée (B-04b)** :
  - `bench.gauge("masks_confirmed/pending/lost")` — `Tracker.tick()`, chaque frame.
  - `bench.probe("staleness_slow_ms")`, `bench.count("staleness_capped")` — `motion.predict_position()`.
  - `bench.count("frames")`, `bench.count("masks_total")` — `main.py`, chaque frame.
  - **[ajout post-B-04b]** `bench.timer("fast_wakeup_lag")` — `FastTrackThread`, mesure latence wakeup.
  - **[ajout post-B-04b]** `bench.count("fast_stale_skipped")` — `FastTrackThread`, branche stale skip.
  - **[ajout post-B-04b]** `bench.count("fast_ncc_confirmed")` — `FastTrackThread`, validation NCC réussie.
  - **[ajout post-B-04b]** `bench.timer("capture_wait")` — boucle capture, attente frame source.

- **Export JSONL — format normalisé (étape 3 validée)** :
  - **Deux fichiers** :
    - `bench_frame.jsonl` — 1 objet/frame (granularité maximale).
    - `bench_agg.jsonl` — 1 objet/intervalle N secondes, agrégats min/max/p50/p95.
  - **Contenu** : défini fichier par fichier après audit des fonctions principales du workflow (étape 4 protocole, en cours).
  - **Format** : JSON-Lines (robustesse crash).

- **Protocole de validation** :
  1. ✅ Audit `bench.py` ↔ README terminé.
  2. ✅ Instrumentation B-04b posée.
  3. ✅ Format normalisé JSONL défini.
  4. ⏳ Audit contenu fichiers par fichier (fonctions principales workflow → sondes retenues) — **repoussé**, à reprendre après exploitation session JSONL.
     4bis. ✅ Session JSONL réelle collectée et analysée (session post-B-04b) — invalidation hypothèse floutage/fast, requalification fourchette FPS.
  5. ⏳ Session > 30 s — variance FPS mesurée **séparément** scène avec/sans masks actifs.
  6. ⏳ Si variance > 15% à charge constante → profilage cProfile / py-spy.

- **Dettes à solder avant clôture** :
  - Reprendre étape 4 protocole : localiser les sondes restantes par fichier, compléter cartographie.
  - `README.md` : documenter la feature bench (cf. section dédiée ci-dessous).

- **Bloque** : B-04d (scopé B-05), B-05, B-06.

  **B-04d — Burst faux positifs slow detector** :
  ⚠️ **Hors-scope B-04 — scopé en B-05.** Aucun changement.

---

- **Critères de succès — état** :

| Critère                                               | Statut                                       |
| ----------------------------------------------------- | -------------------------------------------- |
| Sonde `staleness_slow` opérationnelle                 | ✅                                           |
| `compute_predicted_rect` pure (#60)                   | ✅                                           |
| Crash `KeyError get_and_reset_stats`                  | ✅ résolu                                    |
| `capped_pct < 10%` régime nominal main loop ≥ 120 FPS | ✅ validé à `dt_cap=0.30`, confirmé à `0.35` |
| `dt_cap` valeur définitive `0.35` appliquée           | ✅ appliquée dans `config.yaml`              |
| FPS stable > 30 s (variance < 15% à charge constante) | ⚠️ pending validation B-04c                  |
| Logs `[FAST-APPLY]` / `ZOMBIE-SUSPECT` en DEBUG       | ✅                                           |
| Footage teleport tagué                                | ❌ précondition humaine non déclenchée       |
| README.md alligner avec le code                       | ⏳ En cours                                  |

---

- **Actions restantes avant fermeture B-04** :
  1. Appliquer `dt_cap: 0.35` dans `config.yaml`. ✅ fait.
  2. Session > 30 s — confirmer FPS stable à charge constante (B-04c, via protocole Option C : audit bench → format normalisé → exécution).
  3. Déclencher footage teleport (précondition B-00b).
  4. Si les 3 critères ci-dessus sont verts → **fermer B-04, déverrouiller B-00b + B-05**.

---

- **Bloque** : B-00b (footage + stats motion fiables), B-05, B-06, F-08.

- **Anti-patterns à éviter** :
  - Fermer B-04 sans `dt_cap: 0.35` appliqué et validé en session live.
  - Fermer B-04 sans footage teleport — B-00b bloqué indéfiniment.
  - Interpréter `capped_pct` élevé à bas FPS comme un bug — c'est une dégradation mécanique attendue.
  - Réouvrir B-04a sans régression observée explicite.
  - Embarquer B-04d dans B-04 — scopé B-05.
  - Extraire percentiles `dist` pour B-00b avant `dt_cap: 0.35` validé en session.
  - **[AJOUT]** Lancer la session de validation B-04c avant que l'audit bench ↔ README ne soit clos et le format de restitution figé.

---

### 🔴 B-00b — Anomalie config #1 : `teleport_thresh` vs `vx_max × dt_cap`

> **Justification d'extraction depuis B-00** : cette anomalie nécessite des stats motion fiables (post-B-04 / correction #60) et un footage de référence avec vrai teleport tagué (action humaine à déclencher pendant B-04). Ces deux préconditions n'étant pas satisfaites au moment de B-00, elle est traitée en ticket dédié après B-04.

- **Préconditions dures** :
  - B-04 livré et validé (`compute_predicted_rect` #60 corrigé, `motion.dt` fiabilisé).
  - Percentiles `dist` inter-frames extraits sur session de référence post-B-04.
  - Footage de référence avec au moins un vrai teleport tagué disponible (déclenché pendant B-04 — action humaine).

- **Anomalie** :
  - `motion.teleport_thresh = 300 < vx_max × dt_cap = 4000 × 0.10 = 400`
  - Un déplacement nominal à vitesse max sur `dt_cap` déclenche un faux teleport → reset vélocité → CREATE parasite cohérent avec les bursts observés en B-05.

- **Actions** :
  1. Extraire percentiles `dist` p95/p99 inter-frames sur session de référence post-B-04.
  2. Trancher : `vx_max` sur-dimensionné (valeur fictive non atteignable) ou `teleport_thresh` sous-dimensionné ?
  3. Réajuster la valeur incohérente. Cible : `teleport_thresh > vx_max × dt_cap × 1.2` (marge de sécurité 20 %).
  4. Valider sur footage tagué que les vrais teleports restent détectés après réajustement.
  5. Documenter la décision produit (seuil retenu + justification percentiles).

- **Effort estimé** : 1 h.

- **Critères de succès** :
  - `teleport_thresh > vx_max × dt_cap` (incohérence éliminée).
  - Marge de sécurité documentée (`× 1.2` ou valeur justifiée).
  - Vrais teleports détectés sur footage tagué.
  - Percentiles `dist` p95/p99 consignés dans `docs/session-de-reference.md`.
  - Aucune régression sur session de référence.

- **Bloque** : B-05 (anomalie #1 peut générer des bursts CREATE parasites cohérents avec le symptôme B-05 — diagnostic faussé si non résolu avant).

- **Effets de bord à anticiper** :
  - Si `teleport_thresh` relevé significativement : vérifier que les cas-limites de teleport (ex. transition de scène brutale) restent couverts.
  - Si `vx_max` abaissé : impact sur les seuils de drift du fast tracker — auditer les consommateurs de `vx_max` dans le codebase.

- **Anti-patterns à éviter** :
  - Réajuster sans extraction préalable des percentiles → décision aveugle non reproductible.
  - Démarrer avant B-04 livré → stats motion structurellement fausses (correction #60 non appliquée).
  - Valider uniquement sur session sans teleport → absence de preuve que les vrais teleports restent détectés.
  - Embarquer dans B-00 → préconditions non satisfaites à ce stade.
  - Embarquer dans B-05 → B-05 partirait sur une config potentiellement incohérente.

---

### 🔴 B-05 — Slow detector : faux positifs en burst `[RÉÉCRIT, découpé en B-05a + B-05b]`

> **Note d'honnêteté préservée** : la cause racine exacte n'est pas connue. Le découpage audit/implémentation reflète cette incertitude. B-05a peut révéler que le problème est ailleurs (effet de bord B-04 résiduel, anomalie B-00 #1, etc.), auquel cas B-05b devient sans objet.

---

### 🔴 B-05a — Audit slow detector

- **Préconditions dures** :
  - B-00 livré (anomalie #1 sur `teleport_thresh` traitée — peut générer des symptômes confondus avec B-05).
  - B-04 livré et validé (`motion.dt` fiabilisé, FPS stable, logs nettoyés).
  - Session de référence à jour intégrant les corrections B-00 et B-04.
  - Sondes registry disponibles : taux CREATE/s, ratio masks éphémères, taux EXPIRE.

- **Symptôme** :
  - Rafales de CREATE observées en logs post-B-03 (ex. uid=20→37 sur ~3 s) → pollution registry, EXPIRE en cascade peu après.
  - Hors-scope motion/tracker : pathologie identifiée côté **slow detector** (pipeline CV : HSV + morpho + refine + geometry).

- **Mesure baseline obligatoire** _(pré-requis à toute action)_ :
  - **Taux CREATE/s** sur fenêtre glissante 1 s (moyenne, p95, max).
  - **Ratio masks éphémères** : `count(masks where total_matches == 1 at EXPIRE) / count(created_masks)` sur fenêtre 10 s.
  - **Taux EXPIRE en cascade** : EXPIRE survenant < 2 s après CREATE / total EXPIRE.
  - **Localisation des bursts** : timestamps + frames source des rafales > 3 CREATE/s.

- **Hypothèses à instruire** _(ordre par coût croissant, recommandé)_ :
  1. **Tuning `tracker.lifecycle.confirm_after`**
     - Actuellement = 1 (CREATE immédiat, aucune consolidation temporelle).
     - Test : passer à 2 puis 3, mesurer impact sur taux masks éphémères.
     - Coût : 1 ligne config, effet immédiat.

  2. **Resserrement validation refine (texte)**
     - Clés : `detect.refine.min_text_fill` (0.08), `min_transition` (0.10), `min_proj_score` (0.10).
     - Test : incréments de +0.02 sur chaque clé, mesure ROC TP/FP.
     - Coût : config seule.

  3. **Resserrement filtres geometry**
     - Clés : `detect.geometry.min_fill`, `min_area`, `min_ratio`, `max_ratio`.
     - Test : audit des blobs créés en burst, identifier la dimension exploitée par les FP.
     - Coût : config seule.

  4. **Resserrement seuils HSV / morpho**
     - Clés : `detect.hsv.white_core.lower/upper`, `detect.morpho.white_dilate.*`.
     - Test : analyse pixel-niveau sur frames déclenchant des bursts.
     - Coût : config + risque plus diffus sur TP.

  5. **Cooldown post-burst** _(dernier recours)_
     - Limitation taux CREATE/s côté tracker (pas detector).
     - Coût : code, à éviter si options 1-4 suffisent.

- 🔍 **Audit requis — bloquant à la livraison de B-05a** :
  - Inventaire pipeline slow detector : étapes, seuils actuels, points de filtrage.
  - Mesure baseline (cf. ci-dessus) sur session de référence.
  - Identification des frames déclencheuses des bursts (corrélation avec événements scène : transitions, apparitions multiples, mouvements rapides).
  - Confirmation que la pathologie n'est pas un effet de bord résiduel B-00, B-02, B-03 ou B-04.
  - Test rapide hypothèse #1 (`confirm_after`) car coût marginal.

- **Livrables B-05a** :
  - Rapport d'audit : baseline mesurée, hypothèses validées/invalidées, recommandation chiffrée.
  - Décision go/no-go pour B-05b et périmètre exact (quelle(s) hypothèse(s) implémenter).
  - Si l'audit révèle que le problème est ailleurs : B-05b annulé, ticket dédié ouvert.

- **Effort estimé B-05a** : 2-4 h.

- **Bloque** : B-05b (et donc B-06).

---

### 🔴 B-05b — Correctifs slow detector

- **Préconditions dures** :
  - B-05a livré avec recommandation validée.

- **Périmètre** : implémentation des hypothèses retenues par B-05a (1 à 5, non exclusives).

- **Critères de succès** _(ancrés sur baseline B-05a)_ :
  - **Réduction relative ≥ 70 %** du ratio masks éphémères vs baseline B-05a.
    _(Ancrage relatif plutôt qu'absolu : un seuil < 5 % serait arbitraire si la baseline est déjà à 8 % ; un seuil < 5 % serait inatteignable si la baseline est à 50 %.)_
  - **Sur fenêtre glissante 1 s** : taux CREATE/s < 3 hors événement de scène taggé "apparition multiple" dans la session de référence.
  - **Taux EXPIRE en cascade** (EXPIRE < 2 s après CREATE) < 10 %.
  - **Non-régression TP** : taux de détection vrai positif préservé à ±2 % vs baseline, mesuré sur footage tagué incluant transitions de scène.

- **Instrumentation permanente** _(à conserver jusqu'à B-07 tranché)_ :
  - `slow_create_rate_1s` : CREATE/s glissant 1 s.
  - `slow_ephemeral_ratio_10s` : ratio masks éphémères glissant 10 s.
  - `slow_burst_events` : compteur cumulé bursts détectés (> 3 CREATE/s sur 1 s).
  - Exposition : HUD + CSV (cohérent avec `detect.csv.*` existant).
  - Ces métriques alimentent B-07 au même titre que `keepalive_*` de B-06. **Ne jamais retirer avant B-07 tranché.**

- **Effort estimé B-05b** : 2-6 h selon hypothèse(s) retenue(s) par B-05a.

- **Bloque** : B-06.

- **Effets de bord à anticiper** :
  - **Vers B-06** : slow detector pollué fausse le keepalive (un mask stationnaire entouré de faux positifs reçoit des "matches" parasites). B-06 doit s'exécuter sur un slow propre.
  - **Vers B-07** : la sémantique de `last_seen_ts` dépend de la qualité du flux slow. Bursts de FP polluent les statistiques alimentant l'audit B-07.
  - **Vers `lost_after_s`** : si `confirm_after` ≥ 2 adopté, latence CREATE augmente. Invariant à vérifier (cf. A-02 catégorie 8) :
    `confirm_after / capture_fps < 0.3 × lost_after_s`
    À 120 FPS, `confirm_after = 3` ajoute ~25 ms ; `lost_after_s = 0.3 s` reste très large. ✅ OK pour valeurs nominales.
  - **Vers session de référence** : footage tagué doit inclure transitions de scène + apparitions multiples + régime statique long + mouvements rapides. Réutilisable pour B-06.

- **Anti-patterns à éviter** :
  - Implémenter sans audit préalable (B-05a est obligatoire).
  - Relever les seuils sans mesurer l'impact sur le taux de vrais positifs.
  - Démarrer B-05b avant que B-04 soit validé : un `dt` saturé peut masquer des comportements registry et fausser le diagnostic.
  - Embarquer un fix B-05 dans une refonte sémantique (B-07) : périmètres distincts.
  - Retirer l'instrumentation `slow_*` avant B-07 tranché.
  - Confondre tuning detector (B-05) et tuning tracker (`confirm_after`, qui est côté tracker mais agit comme filtre temporel sur le flux detector — légitime dans le périmètre B-05).

---

### 🔴 B-06 — Auto-keepalive masks stationnaires

- **Préconditions dures** :
  1. **B-04 livré** : sinon `estimated_velocity_pps` est biaisé par dérive `dt`.
  2. **B-05 livré** : sinon `last_slow_ts` est pollué par faux positifs burst, garde-fou inopérant.
  3. **Sondes B-03 actives** : `since_last_seen`, lifetime médian/p95.

  ⚠️ **Démarrage interdit avant validation des trois préconditions.** Si B-04 ou B-05 traînent, B-06 reste en attente — pas de contournement.

- **Périmètre strict** : transition CONFIRMED → LOST côté `Tracker.tick()` uniquement.
  Ne touche **ni** au fast tracker, **ni** au slow associator, **ni** à la sémantique de `last_seen_ts` côté événements externes (ce dernier sujet relève de **B-07**).

- **Cause racine** :
  - `last_seen_ts` est mis à jour exclusivement sur match slow ou fast.
  - Le **fast tracker n'émet pas d'événement** quand le mouvement inter-frame est sous le seuil de détection (typique à 150–200 FPS sur plaques quasi-statiques).
  - Conséquence : un mask **réellement présent et immobile** voit `since_last_seen` croître linéairement → franchit `lost_after_s = 0.3 s` → transition LOST → purge à 1.3 s, alors que rien n'a disparu.
  - Cycle observé : CREATE → CONFIRMED → LOST → EXPIRE → re-CREATE par slow suivant, en boucle.

- **Symptôme attendu (à mesurer pré/post fix)** :
  - À haut FPS (≥ 150) sur scène statique : oscillation `C/L` non nulle alors que la scène est figée.
  - Logs `[FAST-APPLY] drift dégradé` corrélés aux re-CREATE.
  - Compteur `fast_received` constant mais `fast_applied` ≈ 0 sur fenêtres immobiles.

- **Approche** : patch tactique côté `Tracker.tick()`. Avant `tick_and_expire`, marquer comme matché tout mask CONFIRMED dont la **vitesse estimée est sous seuil**, sous condition d'un **garde-fou slow** :

  ```python
  # Pseudo-code, à intégrer dans Tracker.tick() avant tick_and_expire
  for mask in registry.iter_confirmed():
      if mask.estimated_velocity_pps < STATIONARY_VEL_EPS:
          if (ts - mask.last_slow_ts) <= MAX_KEEPALIVE_WITHOUT_SLOW_S:
              registry.mark_keepalive(uid, ts)  # met à jour last_seen_ts
              keepalive_count += 1
  ```

  - Le garde-fou `MAX_KEEPALIVE_WITHOUT_SLOW_S` empêche un mask de survivre indéfiniment si **toutes** les sources se taisent.
  - Utilise `last_slow_ts` introduit par B-02 (S3).
  - **Distinct de `mark_matched`** : `mark_keepalive` trace une raison différente pour debug (`last_match_kind = "keepalive"`).

- **Paramètres config** :

  ```yaml
  tracker:
    lifecycle:
      stationary_keepalive:
        enabled: true
        velocity_eps_pps: 0.5 # px/s, sous ce seuil = stationnaire
        max_without_slow_s: 2.0 # garde-fou : ≥ 2× période slow attendue
  ```

- 🔍 **Audit requis avant livraison** :
  - Confirmer que `last_slow_ts` est renseigné par B-02 (S3) sur tous les chemins de match slow.
  - Identifier la source exacte de `estimated_velocity_pps` (motion EMA ? dérivée de positions ?). Recalibrer `velocity_eps_pps` avec données post-B-04.
  - Vérifier qu'aucun consommateur aval ne distingue `match` vs `keepalive` de manière régressive.

- **Effort** : 30 min code + 30 min validation + instrumentation métriques.

- **Critères de succès** :
  - À FPS ≥ 150 sur scène statique > 30 s : aucun cycle CREATE/LOST/EXPIRE/re-CREATE sur masks réellement immobiles.
  - À l'arrêt forcé du flux slow (test contrôlé) : transition LOST en `≤ MAX_KEEPALIVE_WITHOUT_SLOW_S + lost_after_s` (≈ 2.3 s).
  - Métrique `keepalive_ratio = keepalive_count / (keepalive_count + match_count)` < 80 % en régime mobile, ~100 % en régime statique (attendu).
  - Aucune régression sur lifetime médian/p95 mesurés à B-03.

- **Instrumentation à ajouter (prérequis dur de B-07)** :
  - Compteur par mask : `keepalive_count`, `match_count`, `slow_match_count`.
  - Compteur global : `keepalive_total`, ratio glissant 1 s / 10 s.
  - Log INFO toutes les 10 s : `keepalive_ratio`, masks actifs, masks en garde-fou.
  - ⚠️ **Ces métriques alimentent B-07. Ne jamais les retirer tant que B-07 n'a pas tranché.**

- **Risque régression** : faible. ~10 lignes, isolé, désactivable via `enabled: false`.

- **Effets de bord à anticiper** :
  - **Vers B-03** : B-06 ré-élargit la durée de vie effective sur scènes statiques. Ne **pas** remonter `lost_after_s` après livraison de B-06 sans concertation : cela invaliderait l'observation de la couverture du keepalive et fausserait les métriques alimentant B-07.
  - **Vers B-07** : la sémantique de `last_seen_ts` devient "preuve **ou** présomption d'existence". Cette ambiguïté est **dette acceptée** à court terme et est l'objet exact de l'audit B-07.

- **Bloque** : Phase 0 (par discipline d'ordonnancement P0)
- **Débloque** : démarrage observation longue durée → B-07.

---

## 🟦 Phase 0 — Clôture stabilisation noyau

**Trigger** : ✅ B-01, B-02, B-03, B-04, B-05, B-06 livrés.

### 🟢 A-01 — Audit cohérence noyau tracker

- **Périmètre** : `Tracker`, `Associator`, `MaskRegistry`, `MaskState`, propagation `ts` (`perf_counter`), interactions avec `DetectThread` / `FastTrackThread`.
- 🔍 **Audit requis** : modules touchés sur les 3 derniers mois, vérification des invariants (horloge unique, contrat `tick()` public, étanchéité frames vs secondes, intégrité du flux `MaskState`).
- **Livrable** : rapport d'audit + correctifs mineurs si nécessaire.
- **Effort** : 2-3 h.
- **Critère de succès** : invariants README documentés et vérifiés ; aucune divergence runtime non justifiée.

### 🟢 A-02 — Hygiène config & invariants

- **Préconditions dures** :
  - B-00 livré (anomalies préalables traitées).
  - A-01 livré (audit cohérence noyau tracker).

- **Objectif** : valider au chargement (et à chaque hot-reload) tous les invariants critiques de `config.yaml`. Toute violation → erreur explicite, jamais de dégradation silencieuse.

- **Invariants à vérifier** _(liste consolidée post-audit YAML, à compléter si l'audit code base révèle d'autres dépendances)_ :

  #### Catégorie 1 — Cohérence sémantique cycle de vie

  ```text
  tracker.lifecycle.expire_after_lost_s >= tracker.lifecycle.lost_after_s
  detect.fast.max_stale_frames / screen.capture_fps < masks.fast_max_drift_s
  ```

  _Note_ : le 3ème invariant croise une clé en frames et une clé en secondes via `capture_fps` — ce croisement est légitime à condition d'être explicite.

  #### Catégorie 2 — Cohérence motion / teleport

  ```text
  motion.teleport_thresh > max(motion.vx_max, motion.vy_max) × motion.dt_cap
  motion.dt_cap < motion.dt_slow_max
  ```

  #### Catégorie 3 — Étanchéité unités (frames vs secondes)

  Toute clé suffixée `_s` doit être en secondes ; toute clé suffixée `_frames` doit être en frames. Convention non négociable.
  - **Aucune clé en secondes** dans `detect.fast.*` _sauf_ `event_timeout_s` (explicitement temporel).
  - **Aucune clé en frames** dans `tracker.lifecycle.*`, `masks.motion.*`, `masks.fast_max_drift_s`.

  #### Catégorie 4 — Plages de valeurs [0, 1]

  ```text
  detect.fast.ncc_threshold ∈ [0, 1]
  masks.associator.match_score_min_slow ∈ [0, 1]
  masks.associator.match_score_min_fast ∈ [0, 1]
  masks.associator.source_confidence_slow ∈ ]0, 1]
  masks.associator.source_confidence_fast ∈ ]0, 1]
  detect.refine.background.min_bg_score ∈ [0, 1]
  detect.refine.background.min_hue_score ∈ [0, 1]
  detect.refine.min_text_fill ∈ [0, 1]
  detect.refine.min_transition ∈ [0, 1]
  detect.refine.min_proj_score ∈ [0, 1]
  detect.geometry.min_fill ∈ [0, 1]
  detect.geometry.max_fill ∈ [0, 1]
  ```

  #### Catégorie 5 — Cohérence min/max

  ```text
  detect.geometry.min_fill < detect.geometry.max_fill
  detect.geometry.min_area < detect.geometry.max_area
  detect.geometry.min_ratio < detect.geometry.max_ratio
  masks.adaptive_margin.min <= masks.adaptive_margin.base <= masks.adaptive_margin.max
  detect.horizontal_bands.gap_fill < detect.horizontal_bands.min_fill   # hystérésis
  ```

  #### Catégorie 6 — Sommes et invariants associator

  ```text
  sum(masks.associator.weights_source_slow) ≈ 1.0  (tolérance ±0.01)
  sum(masks.associator.weights_source_fast) ≈ 1.0  (tolérance ±0.01)
  masks.associator.source_confidence_fast <= masks.associator.source_confidence_slow
  ```

  #### Catégorie 7 — Capture / vcam

  ```text
  screen.capture_fps >= screen.vcam_fps
  screen.width > 0 && screen.height > 0
  ```

  #### Catégorie 8 — Cohérence consolidation temporelle (post B-05)

  ```text
  tracker.lifecycle.confirm_after / screen.capture_fps < 0.3 × tracker.lifecycle.lost_after_s
  ```

  _Note_ : invariant introduit par B-05b si filtrage temporel adopté. À ajouter conditionnellement.

- 🔍 **Audit requis — bloquant à la livraison** :
  - Recensement exhaustif des clés consommées par le pipeline (`Tracker`, `Associator`, `MaskRegistry`, `MaskState`, `DetectThread`, `FastTrackThread`, `detect.py`, `motion.*`).
  - Pour chaque clé : type attendu, plage valide, unité (frames/secondes/ratio/pixels).
  - Identification des clés non consommées (mortes) — extension du cas `detect.fast.roi_margin`.
  - Identification des clés consommées avec valeur par défaut hardcodée hors YAML (dette config).

- **Effort estimé** : 2-3 h (audit + implémentation validateur + tests).

- **Critères de succès** :
  - Config invalide → erreur explicite au démarrage, message identifiant la clé fautive et la règle violée.
  - Hot-reload d'une config invalide → rejet avec conservation de la config précédente, message d'erreur clair.
  - Aucune dégradation silencieuse possible.
  - Documentation README mise à jour avec la liste complète des invariants.

- **Bloque** : Lot 1 (signaux config doivent être propres avant observations prod).

- **Effets de bord à anticiper** :
  - Validateur strict peut révéler des configs prod historiques invalides → prévoir un mode `--strict` vs warning pour migration.
  - Hot-reload validation peut introduire une latence — mesurer.

- **Anti-patterns à éviter** :
  - Valider uniquement au démarrage et pas au hot-reload (régression silencieuse possible).
  - Lister les invariants en code sans les documenter dans le README.
  - Embarquer le nettoyage des clés mortes (relève de B-00).

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

**Principe** : aucune feature n'est démarrée sans son trigger **et** sans la totalité de ses préconditions livrées. Phase 2 peut rester totalement dormante si aucun signal ne se déclenche.

---

### 🟡 F-01 à F-07 — LifecycleManager

- **Statut** : ⬜ dormant, déclenchement conditionnel.
- **Objectif** : centraliser la gestion du cycle de vie des `MaskState` dans un composant dédié.

- **Trigger global** : signal métier / test / bug indiquant que la dispersion actuelle de la logique cycle de vie (entre `Tracker`, `MaskRegistry`, `Associator`) devient un blocage de maintenance.

- **Préconditions dures (toutes requises)** :
  - ✅ Phase 0 close (B-00, B-01, B-02, B-03, B-04, B-05a, B-05b ou annulation tracée, B-06 livrés).
  - ✅ Trigger global déclenché et tracé formellement.
  - ✅ Audit code base réalisé (cf. ci-dessous) et découpage F-01..F-07 figé.

- 🔍 **Audit code base requis au démarrage** :
  - Recenser tous les sites de logique cycle de vie dans `Tracker`, `MaskRegistry`, `Associator`.
  - Sur cette base, figer le découpage en 7 sous-tâches F-01..F-07.
  - Tant que cet audit n'est pas mené, le découpage F-01..F-07 reste un placeholder et **F-01 ne peut pas démarrer**.

- **Contrainte de compatibilité (invariant README)** :
  - Préserver l'API publique `Tracker.tick(detections, ts)` et `Tracker.get_confirmed_masks()`.
  - Le `LifecycleManager` est un **détail d'implémentation interne** — aucune fuite dans l'API publique.

- **Effort estimé** : ~1 jour, +0,5 j si interaction avec F-09.

- **Critères de succès** :
  - Transitions cycle de vie unifiées dans `LifecycleManager`.
  - Couverture test unitaire complète sur le `LifecycleManager`.
  - Aucune régression sur session de référence.

- **Anti-patterns à éviter** :
  - Démarrer le découpage F-01..F-07 sans audit préalable (découpage arbitraire).
  - Laisser fuiter `LifecycleManager` dans l'API publique du `Tracker`.
  - Démarrer une sous-tâche F-0x avant que la précédente F-0(x-1) soit livrée et validée (séquentialité interne stricte).

- **Débloque** : F-09 (ordre dur, voir précondition F-09).

---

### 🟡 F-08 — Horizon dynamique motion (EMA latence détection)

- **Statut** : ⬜ dormant, déclenchement conditionnel.
- **Objectif** : remplacer le `dt_cap` statique par une EMA globale sur la latence de détection mesurée, pour absorber les variations de cadence sans bloquer la prédiction motion.

- **Triggers cumulatifs (les deux requis)** :
  1. Signal métier / mesure prod sur variance d'horizon `dt`.
  2. ✅ B-04 livré et stable.

- **Préconditions dures (toutes requises)** :
  - ✅ B-00 livré (anomalie #1 `teleport_thresh < vx_max × dt_cap` résolue, sinon l'EMA s'appliquerait sur des bornes motion incohérentes).
  - ✅ B-04 livré et stable (sinon anti-pattern : l'EMA masque le bug `dt`).
  - ✅ Audit code base réalisé (cf. ci-dessous).
  - ✅ Triggers cumulatifs déclenchés et tracés.

- 🔍 **Audit code base requis au démarrage** :
  - Confirmer existence et nom canonique de la clé config `dt_cap` (non listée dans le tableau « paramètres clés » du README — tableau explicitement non exhaustif).
  - Confirmer le module motion et son point d'injection.

- **Garde-fous obligatoires** (non négociables) :
  - Clamp anti-spike sur `dt` brut avant EMA.
  - Cold-start : si `< 5 samples`, fallback sur `dt_cap` statique.
  - EMA **globale** (pas par masque).
  - Damping pour absorber les transitoires.
  - Log HUD du `dt_cap` courant.

- **Effort estimé** : 1 h dev + 30 min calibration.

- **Critères de succès** :
  - `motion.dt` stable post-déploiement.
  - `capped_pct` stable < 10 %.
  - Aucune régression sur overlap rect moyen.

- **Anti-patterns à éviter** :
  - Déployer F-08 sur un système dont les `dt` sont aberrants → l'EMA absorbe le bug et masque le diagnostic.
  - EMA par masque (explose en complexité, aucun gain mesuré).
  - Démarrer F-08 sans B-00 livré (bornes motion incohérentes).

- **Débloque** : F-10 (précondition dure de F-10) + déclenche la condition de réveil n°1 du backlog gelé R&D prédiction historique.

---

### 🟡 F-09 — Thread dédié pour la phase TTL/éviction du `MaskRegistry`

> ⚠️ **Recadrage important** : F-09 ne peut **pas** externaliser `Tracker.tick()` complet, car `tick()` orchestre aussi `Associator.match()` (gating + Hungarian) sur les détections fraîches consolidées par le main loop. Externaliser `tick()` casserait le contrat README.

- **Statut** : ⬜ dormant, déclenchement conditionnel.
- **Périmètre strict** : externaliser **uniquement la phase TTL / éviction** du `MaskRegistry` dans un thread dédié cadencé à fréquence fixe (ex. 30 Hz).

- **Trigger** : jitters de purge mesurés au-delà d'un seuil acceptable.

- **Préconditions dures (toutes requises)** :
  - ✅ B-04 livré (sinon temporalisation insuffisante, voir anti-pattern).
  - ✅ **F-01..F-07 livré** si déclenché — règle de séquentialité dure : afin d'éliminer toute interaction non maîtrisée avec `LifecycleManager`, F-09 ne peut pas démarrer tant que F-01..F-07 est en cours ou non livré, **dès lors que son trigger est lui-même déclenché**. Si F-01..F-07 n'est jamais déclenché, F-09 peut démarrer seul (pas de blocage par un ticket dormant).
  - ✅ Audit code base réalisé (cf. ci-dessous).
  - ✅ Trigger jitter purge déclenché et tracé.

> 📌 **Clarification règle séquentielle** : un ticket dormant non déclenché ne bloque pas un ticket aval. C'est uniquement un ticket **déclenché et non livré** qui bloque. Cette règle s'applique à toute paire `(amont conditionnel, aval)` du plan.

- 🔍 **Audit code base requis au démarrage** :
  - Identifier le point d'appel actuel de la phase éviction dans `Tracker.tick()`.
  - Caractériser la thread-safety du `MaskRegistry` (lock interne ? structures partagées avec main loop ?).
  - Définir le contrat de synchronisation entre `Associator.match()` (main loop) et la phase éviction (thread dédié).

- **Effort estimé** : 2-3 h + complexité thread-safety.

- **Critères de succès** :
  - Jitter de purge < seuil cible (à fixer au démarrage).
  - Aucune race condition observée sur session de référence.
  - Aucune régression fonctionnelle tracker.

- **Anti-patterns à éviter** :
  - Déployer F-09 sans B-04 → revient à cadencer un compteur en ticks, aucun bénéfice.
  - Externaliser `Tracker.tick()` complet au lieu de la seule éviction → casse le contrat tracker (README).
  - Démarrer F-09 alors que F-01..F-07 est déclenché mais non livré → interaction non maîtrisée avec `LifecycleManager`.

---

### 🟡 F-10 — Affinements prédiction motion

- **Statut** : ⬜ slot réservé, à détailler post-F-08.
- **Objectif** : intégrer les briques réutilisables issues de la R&D prédiction par historique (cf. backlog gelé `docs/backlog/prediction-historique.md`) qui apportent un gain net sans dupliquer F-08.

- **Trigger** : F-08 livré et stable + mesure prod montrant que la vélocité EMA seule ne suffit pas sur certains cas d'usage à caractériser.

- **Préconditions dures (toutes requises)** :
  - ✅ F-08 livré et stable en prod.
  - ✅ Mesure prod ayant caractérisé les cas d'usage où l'EMA seule est insuffisante (tracée formellement).
  - ✅ Revue de cadrage F-10 menée et conclusions formalisées (cf. ci-dessous).
  - ✅ Conditions de réveil du backlog gelé R&D prédiction historique **toutes satisfaites** (cohérence avec `docs/backlog/prediction-historique.md`).

- **Action de cadrage obligatoire** : lors de la livraison de F-08, ouvrir une **revue dédiée** → décider quelles briques R&D migrent en F-10 et lesquelles restent gelées. Sortie attendue : périmètre F-10 figé, briques retenues listées, briques rejetées tracées.

- **Effort estimé** : non chiffré (slot — sera chiffré à la sortie de la revue de cadrage).

- **Critères de succès** : à définir à la sortie de la revue de cadrage (dépendent des briques retenues).

- **Justification du slot** : garantir que le travail R&D antérieur ne soit pas perdu, sans s'engager sur un périmètre prématurément.

- **Anti-patterns à éviter** :
  - Démarrer F-10 sans la revue de cadrage post-F-08 (re-création des erreurs R&D antérieures).
  - Dupliquer la logique EMA de F-08 dans F-10.
  - Réveiller le backlog gelé sans satisfaire les **3** conditions de réveil documentées.

---

## 🔬 Audits différés (long terme)

> **Principe** : un audit différé n'est **pas** un ticket d'implémentation. Il s'agit d'une décision documentée d'**investiguer plus tard**, sur la base de métriques collectées entre-temps, avant d'engager un effort d'implémentation potentiellement structurel.

### 🔬 B-07 — Refonte sémantique `last_seen_ts` (options F vs G)

- **Préconditions dures** (toutes requises, non négociables) :
  1. **B-04 livré et stable** ≥ 2 semaines en prod.
  2. **B-05 livré et stable** ≥ 2 semaines en prod.
  3. **B-06 livré et stable** ≥ 2 semaines en prod.
  4. **Métriques B-06 collectées** sur cette période :
     - distribution `keepalive_ratio` par mask et glissante,
     - taux de garde-fou `MAX_KEEPALIVE_WITHOUT_SLOW_S` déclenché,
     - durée moyenne/p95 en CONFIRMED par mask,
     - faux LOST (mask LOST puis ré-apparition immédiate ≤ 0.5 s),
     - faux survivants (mask CONFIRMED alors que la plaque a réellement disparu).

  ⚠️ **Démarrage interdit avant validation des quatre préconditions.** Toute tentative d'implémentation F/G/F+G sans données B-06 = anti-pattern explicite.

- **Origine** : B-06 introduit un `keepalive` qui élargit la sémantique de `last_seen_ts` ("preuve **ou** présomption d'existence"). Cette ambiguïté est tolérable à court terme mais doit être tranchée structurellement avant que d'autres consommateurs (F-08, futurs detectors, multi-source) en dépendent.

- **Objectif de l'audit** : trancher entre trois architectures candidates pour remplacer le timeout binaire actuel par une sémantique de vie/mort des masks plus riche et plus robuste :
  - **Option F** — Missing event-driven
  - **Option G** — Confidence decay
  - **Option F+G** — Hybride

- **Statut** : 🔬 **différé**.

---

#### 📘 Option F — Missing event-driven

- **Principe** : remplacer le timeout passif (`since_last_seen > lost_after_s`) par des **événements explicites** émis par les sources de détection. Chaque source (slow, fast, futur detector) signale activement :
  - `match(uid, ts, position)` — preuve de présence (existant)
  - `missing(uid, ts, reason)` — **preuve d'absence** (nouveau)
  - `silence(uid, ts)` — incapacité de statuer (nouveau, optionnel)

  La transition CONFIRMED → LOST n'est plus déclenchée par un timeout, mais par **accumulation d'événements `missing`** au-delà d'un seuil (quorum, fenêtre temporelle, ou logique dépendante de la source).

- **Modèle de données** :

  ```python
  class MaskState:
      ...
      evidence_log: deque[Evidence]  # ring buffer borné
      # Evidence = (ts, source, kind, payload)
      # source ∈ {slow, fast, motion, ...}
      # kind   ∈ {match, missing, silence}
  ```

  Le calcul d'état devient une **fonction pure** sur `evidence_log` plutôt qu'un side-effect du timeout.

- **Flux d'événements type** :

  ```text
  t=0.00  slow.match     → CONFIRMED
  t=0.10  fast.match     → CONFIRMED (refresh)
  t=0.50  slow.missing   → CONFIRMED (1 missing isolé, sous seuil)
  t=0.60  fast.silence   → CONFIRMED (silence non-incriminant)
  t=1.00  slow.missing   → LOST (2e missing slow consécutif → quorum atteint)
  ```

- **Avantages** :
  - **Sémantique explicite** : on sait _pourquoi_ un mask est LOST.
  - **Debuggabilité maximale** : `evidence_log` sérialisable, rejouable.
  - **Robuste aux sources hétérogènes** : ajouter un nouveau detector = ajouter une source d'events.
  - **Pas d'arbitraire temporel** : plus de `lost_after_s = 0.3` magique.

- **Inconvénients** :
  - Demande que **chaque source soit capable d'émettre des `missing`** — non trivial pour le fast tracker actuel.
  - Ring buffer + logique de quorum = **structure à thread-safer** (interaction avec F-09).
  - Coût mémoire par mask (proportionnel à la taille du ring buffer).

- **Cas où elle brille** : multi-source asymétrique, besoin de traçabilité forte (audit, replay, post-mortem).

- **Cas où elle échoue** : sources incapables d'émettre des `missing` proprement → on retombe sur du timeout déguisé.

- **Coût d'implémentation estimé** :
  - Refactor `MaskRegistry` : 1 j.
  - Adaptation slow associator (émission `missing` post-Hungarian non concluant) : 0,5 j.
  - Adaptation fast tracker (émission `missing` post-NCC échoué) : 0,5–1 j.
  - Logique de quorum + tests : 0,5 j.
  - **Total : 2,5–3 j.**

---

#### 📘 Option G — Confidence decay

- **Principe** : remplacer l'état discret `{CONFIRMED, LOST}` par un **score de confiance continu** `c ∈ [0, 1]` qui :
  - **augmente** sur match (boost dépendant de la source),
  - **décroît** dans le temps (decay exponentiel ou linéaire),
  - **transitionne** d'état quand des seuils sont franchis (`c > θ_confirm` → CONFIRMED, `c < θ_lost` → LOST).

  Le timeout devient implicite : un mask non rafraîchi voit sa confiance décroître naturellement jusqu'au seuil LOST.

- **Modèle de données** :

  ```python
  class MaskState:
      ...
      confidence: float           # ∈ [0, 1]
      last_confidence_update: float
      # state dérivé de confidence + hystérésis
  ```

- **Dynamique type** :

  ```python
  # À chaque tick :
  dt = ts - mask.last_confidence_update
  mask.confidence *= exp(-dt / TAU_DECAY)
  mask.last_confidence_update = ts

  # Sur match :
  mask.confidence = min(1.0, mask.confidence + BOOST[source])
  # BOOST_slow = 0.6, BOOST_fast = 0.2

  # Sur keepalive (héritage B-06) :
  mask.confidence = max(mask.confidence, 0.5)  # plancher, pas boost
  ```

- **Avantages** :
  - **Graduation naturelle** : un mask "à moitié sûr" est représentable.
  - **Pas d'événement `missing` requis** : le decay fait le travail. Compatible avec sources existantes sans modification.
  - **Calibration mono-paramètre** par source intuitive.
  - **Faible coût mémoire** : un float par mask.

- **Inconvénients** :
  - **Sémantique floue** : `confidence = 0.43` ne dit pas _pourquoi_.
  - **Couplage temporel** : `TAU_DECAY` global peut mal s'adapter à des régimes hétérogènes.
  - **Hystérésis nécessaire** : sinon flapping autour des seuils.
  - **Pas de traçabilité native** : impossible de répondre à "qui a tué ce mask ?" sans logs additionnels.

- **Cas où elle brille** : rendu/consommateur aval qui exploite la graduation, sources homogènes en fiabilité.

- **Cas où elle échoue** : besoin de raisonnement explicite sur les sources (ex. "ignorer le slow s'il a rapporté missing 3 fois").

- **Coût d'implémentation estimé** :
  - Ajout champ + calcul decay : 0,5 j.
  - Hystérésis + transitions : 0,5 j.
  - Calibration `BOOST[source]` + `TAU` + seuils sur métriques B-06 : 0,5 j.
  - Tests régression : 0,5 j.
  - **Total : 2 j.**

---

#### 📘 Option F+G — Hybride

- **Principe** : utiliser **G comme moteur d'état** (decay + seuils) et **F comme couche d'enrichissement** (events `missing` qui décrémentent la confiance plus agressivement qu'un simple decay temporel).

- **Synergie** :
  - G fournit la graduation et la robustesse aux sources silencieuses.
  - F fournit la traçabilité et la réactivité aux preuves d'absence explicites.
  - `missing(uid)` → `confidence -= PENALTY[source]`.

- **Avantages** :
  - Couvre les deux régimes : sources qui savent dire `missing` **et** sources silencieuses.
  - Migration progressive possible : démarrer G seul, ajouter émissions `missing` source par source.

- **Inconvénients** :
  - **Surface de paramétrage doublée** : `TAU`, seuils, `BOOST`, `PENALTY`.
  - Risque d'effets contre-intuitifs si decay et `missing` se combinent mal.
  - **Coût cumulé : 3,5–4,5 j**.

- **Cas où l'hybride se justifie** : système multi-source mature avec besoins de traçabilité **et** de graduation. À ne retenir que si l'audit montre que ni F ni G seuls ne suffisent.

---

#### 🎯 Critères de décision (à appliquer pendant l'audit)

| Critère                                              | Option F  | Option G  | Option F+G |
| ---------------------------------------------------- | --------- | --------- | ---------- |
| Richesse sémantique (debuggabilité)                  | ✅✅      | ⚠️        | ✅✅       |
| Coût d'implémentation                                | Moyen     | Faible    | Élevé      |
| Compatibilité sources existantes (sans modification) | ❌        | ✅✅      | ⚠️         |
| Calibration intuitive                                | ⚠️        | ✅        | ❌         |
| Extensibilité multi-detector futur                   | ✅✅      | ⚠️        | ✅✅       |
| Adéquation aux métriques B-06 observées              | À mesurer | À mesurer | À mesurer  |

**Questions à trancher pendant l'audit** :

1. Le fast tracker peut-il émettre des `missing` proprement, ou son silence est-il intrinsèquement ambigu ?
2. Le rendu / consommateurs aval bénéficieraient-ils d'une `confidence` graduée, ou un état binaire suffit ?
3. Les régimes observés en prod sont-ils homogènes (G seul OK) ou contrastés (F ou F+G requis) ?
4. Quel est le coût opérationnel d'une mauvaise calibration de chaque option (faux LOST vs faux survivants) sur métriques B-06 ?

---

#### 📦 Livrables de l'audit

1. **Inventaire** : tous les call sites de `last_seen_ts`, `lost_after_s`, `expire_after_lost_s`, `mark_matched`, `mark_keepalive`. Cartographie des consommateurs aval.
2. **Cartographie des sources de vérité** : slow, fast, motion, futurs detectors. Pour chacun : capacité à émettre `match` / `missing` / `silence`, fiabilité mesurée.
3. **Analyse des métriques B-06** : distribution `keepalive_ratio`, faux LOST, faux survivants, garde-fou déclenché.
4. **Comparaison F vs G vs F+G** sur la matrice de décision, **chiffrée sur données réelles**.
5. **Décision argumentée** + plan de migration **incrémental** :
   - Phase 1 : ajout du nouveau modèle en parallèle, double-écriture, comparaison runtime.
   - Phase 2 : bascule consommateurs un par un.
   - Phase 3 : retrait ancien modèle.
6. **Plan de rollback** si régression observée en phase 2.

- **Effort estimé** :
  - Audit pur : 1 j.
  - Implémentation post-décision : 2–4,5 j selon option.
  - **Total : 3–5,5 j.**

- **Bloque** : rien à court terme.
- **Débloque** : architectures multi-detector futures, F-08 raffiné si confidence consommée par horizon dynamique.

- **Effets de bord à anticiper** :
  - **Vers B-06** : si B-07 retient G ou F+G, le `keepalive` de B-06 doit être migré vers `confidence = max(confidence, plancher)`. Code de B-06 explicitement marqué comme **point de migration** dans son commit.
  - **Vers `lost_after_s`** : selon option retenue, ce paramètre disparaît (F) ou est remplacé par seuils + hystérésis (G). Mise à jour `config.yaml` + doc requise.

- **Anti-patterns à éviter** :
  - Trancher F vs G **sans** données de prod B-06 → choix à l'aveugle.
  - Engager F+G par défaut "pour avoir le meilleur des deux" → complexité non justifiée.
  - Démarrer l'implémentation avant que B-04 et B-05 soient livrés → métriques biaisées.

---

## 🧊 Articulation avec le backlog gelé

Le backlog gelé `Prédiction par historique (régression pondérée)` est **l'amont R&D de F-10**. Ses 3 conditions de réveil sont **strictement équivalentes** aux préconditions de F-10 listées ci-dessus. La règle séquentielle dure s'applique aussi à cette frontière :

- Tant que F-08 n'est pas livré et stable → backlog gelé reste gelé.
- Tant que la mesure prod n'a pas caractérisé les limites EMA → backlog gelé reste gelé.
- Tant que la revue F-10 n'a pas conclu → backlog gelé reste gelé.

**Aucun travail R&D ne peut être démarré tant que ces 3 verrous ne sont pas levés**, conformément au statut « gelé — ne pas démarrer ».

---

## 📁 Hygiène documentaire

- `docs/closed-tickets.md` : tickets clos au fil de l'eau, avec date et lien commit.
- `docs/backlog/prediction-historique.md` : R&D gelée, marquée explicitement « GELÉ — voir Plan v4.1 §Backlog ».
- `docs/session-de-reference.md` : protocole de bench canonique réutilisable pour B-02, F-08, Lot 3, F-10.
  - 🔍 **Audit requis** : la session de référence doit explicitement consommer :
    - **Sondes officielles README** : `capture_wait`, `slow_poll`, `match`, `fast_poll`, `predict`, `blur`, `frames`, `masks_total`, `fast_wakeup_lag`, `fast_tick`, `fast_of_total`, `fast_ncc_total`, `fast_ncc_confirmed`, `fast_stale_used`, `fast_mask_lost`, `send`.
    - **Sondes motion à confirmer** : `motion.dt`, `capped_pct` (existence à valider lors de l'audit B-03).
  - Définir le footage tagué et les scénarios standards avant le premier usage en B-03.

- `docs/audits/B-07-last-seen-ts.md` : à créer **au démarrage de l'audit B-07**, contiendra l'inventaire, les métriques consommées, la matrice de décision et la décision finale.

---

## ✅ Critères globaux V4.3

- **Conformité README** : ✅ types (`MaskState`), composants (`MaskRegistry`, `Associator`, `Tracker`, `FastTrackThread`), API publique (`tick`, `get_confirmed_masks`), horloge (`perf_counter`), hiérarchie YAML, hot-reload config.
- **Étanchéité unités** : ✅ frames côté FastTrack (`detect.fast.max_stale_frames`), secondes côté Tracker (`tracker.lifecycle.*`), invariant verrouillé par A-02.
- **Sondes bench** : ✅ critères de succès Lot 3 et tests de régression adossés aux sondes officielles README.
- **Séquentialité simple** : ✅ chaque bloc indépendant du suivant.
- **Reportabilité** : ✅ toute feature peut être retirée vers un plan ultérieur sans casser la chaîne.
- **Préservation R&D antérieure** : ✅ via F-10 + backlog gelé documenté.
- **Triggers mesurables** : ✅ chaque item conditionnel a un signal explicite.
- **Audits explicites** : ✅ chaque zone d'incertitude marquée 🔍, prérequis bloquant à la livraison.
- **Couverture bugs** : ✅ B-01..B-06 explicites en P0, séquentiels stricts.
- **Différenciation tactique / structurel** : ✅ B-06 (patch) et B-07 (refonte) explicitement séparés, préconditions chaînées.
- **Continuité métriques** : ✅ instrumentation B-06 dimensionnée pour alimenter B-07.
- **Pas de suffixes `bis`** : ✅ numérotation strictement séquentielle.

---

## 📌 Points d'attention pour la suite

1. **Sections Lot 1 et Lot 2** : volontairement non détaillées car leur contenu dépend d'observations futures. Audit requis au démarrage de chaque lot.
2. **F-01..F-07** : sept sous-tâches à figer au démarrage de la feature, après audit du code base. Contrainte non négociable : préserver l'API publique `Tracker.tick()` / `get_confirmed_masks()`.
3. **F-10** : slot délibérément vide, à remplir post-F-08.
4. **F-09** : recadré sur la phase TTL/éviction uniquement — ne jamais externaliser `tick()` complet.
5. **Tous les 🔍 audits code base** sont des prérequis bloquants à la livraison de l'item correspondant — ne pas les sauter.
6. **B-06 vs B-07** : ne **jamais** confondre. B-06 est un patch tactique (30 min). B-07 est un audit structurel (plusieurs jours). Sauter de B-06 directement à F/G/F+G sans audit = anti-pattern.
7. **Métriques B-06 = matière première de B-07** : ne jamais retirer les compteurs `keepalive_*` tant que B-07 n'a pas tranché.
8. **Séquentialité B-04 → B-05 → B-06** : non négociable. B-06 dépend des deux pour des raisons documentées (vitesse fiable, slow non pollué). Aucun parallélisme toléré.
9. **Effet de bord B-06 sur B-03** : ne pas remonter `lost_after_s` après livraison B-06 sans concertation — fausserait les métriques alimentant B-07.

---
