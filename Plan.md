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
   ├── B-03  Investigation dérive dt motion
   └── B-04  TTL temporel (cycle de vie en secondes côté Tracker)
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

### 🔴 B-02 — Correction d'association (slow + fast) `[PÉRIMÈTRE ÉLARGI]`

- **Symptôme résiduel après B-01** :
  - Zombies persistent à `lifetime` 3-8 s avec `last_match=fast 0.00s_ago` et/ou `last_match=slow 0.1s_ago`.
  - Ratio `match_count_fast / match_count_slow` reste anormalement haut sur masks vivants.
  - Cas types observés en validation B-01 :
    - `uid=19 age=7.56s state=CONFIRMED ttl=25 matches=122 (slow=22 fast=100) last_match=fast 0.00s_ago` ← fast match instantané sur zombie 7,5 s
    - `uid=23 age=3.59s state=CONFIRMED ttl=20 matches=59 (slow=17 fast=42) last_match=slow 0.11s_ago` ← **slow** ré-associe sur zombie

- **Cause racine** : **deux** producteurs d'association ré-attribuent des détections à des UIDs morts au lieu d'en créer de nouveaux :
  1. **`Associator` (slow)** — gating IoU/distance/hash trop laxiste : un nouvel objet entrant dans la zone d'un zombie hérite de son UID.
  2. **`FastTrackThread` (fast)** — pipeline LK+NCC produit des `uid_to_rect` sur des cibles fantômes que `fast_max_drift_s` filtre **partiellement** seulement.

- **Objectif** : supprimer la sur-association à la **source**, des deux côtés. Un mask sans objet réel ne doit plus jamais être ré-associé, même dans la fenêtre de drift.

- 🔍 **Audit code base requis (préalable bloquant — double)** :
  - **Slow** : module `Associator`, caractériser les seuils de gating (IoU min, distance max, hash) et la logique de création vs ré-association.
  - **Fast** : module `FastTrackThread`, identifier l'algorithme produisant `uid_to_rect`, caractériser les seuils, vérifier si NCC/phash participe à la décision d'association.

- **Pistes** (à arbitrer après audit, applicable aux deux côtés) :
  1. **Resserrer seuils IoU/distance** d'association.
  2. **Ajouter/renforcer un score d'apparence** (NCC, phash) en condition d'association.
  3. **Cap dur `match_count_fast_since_last_slow`** (refuser le match au-delà de N).
  4. **Préférer la création** de nouveaux UIDs en cas d'ambiguïté (paramètre de "création-favorisée").
  5. **Combinaison** des précédents.

- **Effort estimé** : 4-8 h (audit double + correction sur deux modules + recalibration).

- **Critères de succès** :
  - Aucun mask `EXPIRE` avec `lifetime > 1.5 s` après disparition réelle (critère B-01 reporté, atteint ici).
  - Plus aucun `ZOMBIE-SUSPECT` log avec `last_match=fast 0.00s_ago` ou `last_match=slow < 0.5s_ago` sur masks `age > 2 s`.
  - Ratio `match_count_fast / match_count_slow` aligné sur `FPS_fast / FPS_slow ± 20 %`.
  - `fast_max_drift_s` peut être remonté à **`1.0-1.5 s`** sans réapparition de zombies (test de validation : remettre `1.5` après B-02 livré).
  - Sondes `fast_ncc_confirmed` et `fast_mask_lost` non dégradées vs baseline.

- **Livrable** :
  - Diagnostic écrit (algorithmes d'association slow ET fast caractérisés).
  - Corrections ciblées dans `Associator` et/ou `FastTrackThread`.
  - Recalibration `fast_max_drift_s` documentée.
  - Retrait du commentaire `# QUICK-FIX B-01` sur `fast_max_drift_s`.
  - Retrait de l'instrumentation `[B01]` dans `registry.py` (libère B-04).

- **Dette résorbée** : B-01 (cause racine traitée des deux côtés).

- **Bloque** : retrait définitif de l'instrumentation `[B01]` dans `registry.py` (à conserver jusqu'à validation B-02).

---

### 🔴 B-03 — Investigation dérive `dt` motion `[ex-B-02]`

- **Symptôme** : `motion.dt` dérive (mesures historiques : 750 ms → 2300 ms), `capped_pct` saturé.

- 🔍 **Audit code base requis (préalable)** :
  - Confirmer existence et noms canoniques des sondes `motion.dt` et `capped_pct` (non listées dans la table bench officielle du README, qui n'est pas exhaustive).
  - Localiser le module motion (non explicité dans la liste de fichiers du README).

- **Hypothèses prioritaires** (révisées post-B-01/B-02) :
  1. **Masques zombies non purgés** : ⚠️ hypothèse à **réévaluer** post-B-01 et post-B-02. Si la dérive `dt` disparaît après livraison de B-02, B-03 devient sans objet (à clore par observation, pas par développement).
  2. `last_seen_ts` (ou équivalent dans `MaskState`) non mis à jour sur certains chemins (slow vs fast).

- **Action préalable obligatoire avant démarrage B-03** : observer `motion.dt` et `capped_pct` sur session de référence **après B-02 livré**. Si stables → fermer B-03 sans développement. Sinon → poursuivre selon le périmètre original.

- **Livrable** (si B-03 reste pertinent) :
  - Diagnostic écrit (cause racine identifiée).
  - Correction ciblée.
  - Test de régression sur session de référence : `dt` stable avec variance < 5 %.

- **Fichiers concernés** :
  - 🔍 **Audit code base requis** : recenser tous les sites de calcul de `dt`, tous les sites de mise à jour du timestamp de match (ex. `last_seen_ts`), et tous les appels d'horloge dans le pipeline tracker.

- **Effort** : 0 (si fermé par observation) à 2-4 h selon cause racine.

- **Critère de succès** : `avg dt` stable dans le temps, `capped_pct < 10 %`.

- **Bloque** : F-08 (anti-pattern absolu = ne pas déployer EMA sur `dt` aberrants).

---

### 🔴 B-04 — TTL temporel (cycle de vie en secondes) `[ex-B-03]`

- **Périmètre strict** : **uniquement le cycle de vie côté `Tracker` / `MaskRegistry`**.
  ⚠️ **Ne pas toucher** `detect.fast.max_stale_frames` : cette clé est en frames **par design** (FastTrack event-driven, cadencé par dépôt de frame, pas par horloge wall).

- **Cause racine** : le cycle de vie des `MaskState` est exprimé en nombre de ticks, donc dépendant de la cadence main loop.

- **Objectif** : rendre le vieillissement indépendant de la fréquence d'appel de `Tracker.tick()`.

- **Changements config (proposition, à valider par audit du namespace `TrackerConfig`)** :

  ```yaml
  tracker:
    lifecycle:
      lost_after_s: 1.0 # secondes avant transition LOST
      expire_after_lost_s: 1.5 # secondes en LOST avant éviction
  # à supprimer : équivalents en frames côté tracker (ex. ttl_default, lost_after)
  # NE PAS TOUCHER : detect.fast.max_stale_frames (frames, FastTrack)
  # À RECALIBRER (et non plus supprimer) : masks.fast_max_drift_s, valeur définitive fixée par B-02
  ```

- **Changements code (sous réserve d'audit)** :
  - Sur `MaskState` : ajout `last_seen_ts: float`, `lost_since_ts: float | None`, suppression du compteur TTL en ticks.
  - Sur `MaskRegistry` : la phase TTL/éviction interne (invoquée depuis `Tracker.tick()`) calcule les transitions par delta temporel (`now - last_seen_ts`).
  - Sur le marquage de match (Associator → registry) : mise à jour de `last_seen_ts = ts` reçu par `tick()`.

- 🔍 **Audit code base requis** :
  - Confirmer que la classe runtime correspond bien à `MaskState` (alignement README).
  - Confirmer le namespace YAML actuel consommé par `TrackerConfig`.
  - Confirmer le nom interne de la méthode TTL+éviction du `MaskRegistry`.
  - Confirmer le nom de la méthode de marquage de match (interne à `Associator` ou `MaskRegistry`).

- **Effort** : 1 h dev + 30 min calibration.

- **Critères de succès** :
  - Comportement de purge identique à 3 Hz et à 30 Hz (test sous charge variable).
  - `motion.dt` reste stable post-déploiement.
  - `capped_pct` stable < 10 %.
  - Pas de régression overlap rect moyen.
  - Instrumentation `[B01]` retirée (déjà fait à la livraison de B-02, à confirmer ici).
  - Valeurs config en `_s` propres côté tracker.

- **Bénéfice annexe** : si B-03 était causé par le cycle de vie en ticks, B-04 le résout en cascade.

- **Anti-pattern documenté** : déployer F-08 sur un système dont les `dt` sont aberrants → l'EMA absorbe le bug et masque le diagnostic.

---

## 🟦 Phase 0 — Clôture stabilisation noyau

**Trigger** : ✅ B-01, B-02, B-03, B-04 livrés.

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
