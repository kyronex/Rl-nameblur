# 📘 Plan v4 — Tracker / Plan consolidé

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
   ├── B-01  Quick-fix purge masques zombies
   ├── B-02  Investigation dérive dt motion
   └── B-03  TTL temporel (cycle de vie en secondes côté Tracker)
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

## 🚨 P0 — Bugs bloquants

### 🔴 B-01 — Quick-fix purge masques zombies

- **Symptôme** : masques zombies persistent dans le workflow, polluent logs, affichage et stats motion
- **Objectif** : mitigation immédiate pour débloquer le développement courant — **pas une correction de fond**
- **Approche** :
  - Réduction agressive temporaire des seuils de purge dans `config.yaml` (TTL `MaskRegistry`).
  - Logging explicite à chaque purge avec raison
  - Marquage `# QUICK-FIX B-01, à retirer après B-03` sur les valeurs touchées
- 🔍 **Audit code base requis** :
  - Confirmer le namespace YAML du TTL `MaskRegistry` consommé par `TrackerConfig`.
  - Confirmer le nom interne de la méthode de purge invoquée par `Tracker.tick()` (le contrat public reste `tick()`).
- **Effort** : 30 min
- **Critère de succès** : zombies disparaissent en < 3 s en conditions nominales de dev
- **Dette créée** : valeurs config fragiles si la cadence de la main loop change → résorbée par B-03

### 🔴 B-02 — Investigation dérive `dt` motion

- **Symptôme** : `motion.dt` dérive (mesures historiques : 750 ms → 2300 ms), `capped_pct` saturé
- 🔍 **Audit code base requis (préalable)** :
  - Confirmer existence et noms canoniques des sondes `motion.dt` et `capped_pct` (non listées dans la table bench officielle du README, qui n'est pas exhaustive).
  - Localiser le module motion (non explicité dans la liste de fichiers du README).
- **Hypothèses prioritaires** :
  1. Masques zombies non purgés (lien direct avec B-01 — peut résoudre B-02 en cascade)
  2. `last_seen_ts` (ou équivalent dans `MaskState`) non mis à jour sur certains chemins (slow vs fast).
- **Livrable** :
  - Diagnostic écrit (cause racine identifiée)
  - Correction ciblée
  - Test de régression sur session de référence : `dt` stable avec variance < 5 %
- **Fichiers concernés** :
  - 🔍 **Audit code base requis** : recenser tous les sites de calcul de `dt`, tous les sites de mise à jour du timestamp de match (ex. `last_seen_ts`), et tous les appels d'horloge dans le pipeline tracker.
- **Effort** : 2-4 h selon cause racine
- **Critère de succès** : `avg dt` stable dans le temps, `capped_pct < 10 %`
- **Bloque** : F-08 (anti-pattern absolu = ne pas déployer EMA sur `dt` aberrants)

### 🔴 B-03 — TTL temporel (cycle de vie en secondes)

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
  - Quick-fix B-01 retiré, valeurs config en `_s` propres.
- **Bénéfice annexe** : si B-02 était causé par le cycle de vie en ticks, B-03 le résout en cascade.
- **Anti-pattern documenté** : déployer F-08 sur un système dont les `dt` sont aberrants → l'EMA absorbe le bug et masque le diagnostic.

---

## 🟦 Phase 0 — Clôture stabilisation noyau

**Trigger** : ✅ B-01, B-02, B-03 livrés.

### 🟢 A-01 — Audit cohérence noyau tracker

- **Périmètre** : `Tracker`, `Associator`, `MaskRegistry`, `MaskState`, propagation `ts` (`perf_counter`), interactions avec `DetectThread` / `FastTrackThread`.
- 🔍 **Audit requis** : modules touchés sur les 3 derniers mois, vérification des invariants (horloge unique, contrat `tick()` public, étanchéité frames vs secondes, intégrité du flux `MaskState`).
- **Livrable** : rapport d'audit + correctifs mineurs si nécessaire.
- **Effort** : 2-3 h.
- **Critère de succès** : invariants README documentés et vérifiés ; aucune divergence runtime non justifiée.

### 🟢 A-02 — Hygiène config & invariants

- **Objectif** : valider au chargement (et à chaque hot-reload) tous les invariants critiques de `config.yaml`.
- **Invariants à vérifier (liste minimale, à compléter par audit)** :
  - `tracker.lifecycle.expire_after_lost_s >= tracker.lifecycle.lost_after_s` (post B-03).
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
  2. **B-02 résolu** (sinon anti-pattern documenté).
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
- **Trigger** : jitters de purge mesurés au-delà d'un seuil acceptable **après B-03 livré** (la temporalisation seule ne suffit pas, il faut aussi régulariser l'appel).
- 🔍 **Audit code base requis** :
  - Identifier le point d'appel actuel de la phase éviction dans `Tracker.tick()`.
  - Caractériser la thread-safety du `MaskRegistry` (lock interne ? structures partagées avec main loop ?).
  - Définir le contrat de synchronisation entre `Associator.match()` (main loop) et la phase éviction (thread dédié).
- **Effort** : 2-3 h, +complexité thread-safety.
- **Risque** : interaction avec LifecycleManager (F-01..F-07). **Ordre recommandé** : F-01..F-07 puis F-09 si les deux sont déclenchés.
- **Anti-patterns documentés** :
  - Déployer F-09 sans B-03 (revient à cadencer un compteur en ticks → aucun bénéfice).
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
    - **Sondes motion à confirmer** : `motion.dt`, `capped_pct` (existence à valider lors de l'audit B-02).
  - Définir le footage tagué et les scénarios standards avant le premier usage en B-02.

---

## ✅ Critères globaux V4.1

- **Conformité README** : ✅ types (`MaskState`), composants (`MaskRegistry`, `Associator`, `Tracker`, `FastTrackThread`), API publique (`tick`, `get_confirmed_masks`), horloge (`perf_counter`), hiérarchie YAML, hot-reload config.
- **Étanchéité unités** : ✅ frames côté FastTrack (`detect.fast.max_stale_frames`), secondes côté Tracker (`tracker.lifecycle.*`), invariant verrouillé par A-02.
- **Sondes bench** : ✅ critères de succès Lot 3 et tests de régression adossés aux sondes officielles README.
- **Séquentialité simple** : ✅ chaque bloc indépendant du suivant.
- **Reportabilité** : ✅ toute feature peut être retirée vers un plan ultérieur sans casser la chaîne.
- **Couverture bugs** : ✅ B-01, B-02, B-03 explicites en P0.
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
