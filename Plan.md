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
   └── B-03  TTL temporel (cycle de vie en secondes)
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
   ├── P-01  Batch LK pyramide partagée
   ├── P-02  Shi-Tomasi features
   └── P-03  Cache config lookup
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
  - Réduction agressive temporaire des seuils de purge dans la config
  - Logging explicite à chaque purge avec raison
  - Marquage `# QUICK-FIX B-01, à retirer après B-03` sur les valeurs touchées
- **Fichiers concernés** : configuration runtime + module gérant la purge des masques
  - 🔍 **Audit code base requis** : identifier le module exact gérant `tick_and_expire` ou équivalent, et les clés config associées (`lost_after`, `ttl_default` ou autres noms en vigueur)
- **Effort** : 30 min
- **Critère de succès** : zombies disparaissent en < 3 s en conditions nominales de dev
- **Dette créée** : valeurs config fragiles si la cadence de la main loop change → résorbée par B-03

### 🔴 B-02 — Investigation dérive `dt` motion

- **Symptôme** : `motion.dt` dérive (mesures historiques : 750 ms → 2300 ms), `capped_pct` saturé
- **Hypothèses prioritaires** :
  1. Masques zombies non purgés (lien direct avec B-01 — peut résoudre B-02 en cascade)
  2. `last_detected_ts` (ou équivalent) non mis à jour sur certains chemins (fast vs slow)
  3. Résidus d'horloges hétérogènes (`time.time()` vs `time.perf_counter()`)
- **Livrable** :
  - Diagnostic écrit (cause racine identifiée)
  - Correction ciblée
  - Test de régression sur session de référence : `dt` stable avec variance < 5 %
- **Fichiers concernés** :
  - 🔍 **Audit code base requis** : recenser tous les sites de calcul de `dt`, tous les sites de mise à jour de `last_detected_ts` (ou nom équivalent), et tous les appels d'horloge dans le pipeline tracker
- **Effort** : 2-4 h selon cause racine
- **Critère de succès** : `avg dt` stable dans le temps, `capped_pct < 10 %`
- **Bloque** : F-08 (anti-pattern absolu = ne pas déployer EMA sur `dt` aberrants)

### 🔴 B-03 — TTL temporel (cycle de vie en secondes)

- **Cause racine** : le cycle de vie des masques est exprimé en **nombre de ticks**, dépendant de la cadence de la main loop. Toute variation de charge CPU, résolution ou complexité scène modifie le comportement de purge.
- **Objectif** : rendre le vieillissement **indépendant de la fréquence** d'appel de `tick()`
- **Changements config** :

  ```yaml
  masks:
    lost_after_s: 1.0 # secondes avant transition LOST
    expire_after_lost_s: 1.5 # secondes en LOST avant purge
  # à supprimer : ttl_default, lost_after (en frames)
  ```

- **Changements code** :
  - Sur `Mask` : ajout `last_seen_ts: float`, ajout `lost_since_ts: float | None`, suppression `ttl`
  - Sur le registry : `tick_and_expire(ts, updated_uids)` calcule les transitions par delta temporel
  - Sur le marquage de match : mise à jour de `last_seen_ts`
- **Fichiers concernés** :
  - 🔍 **Audit code base requis** : confirmer les noms exacts des classes `Mask`, du registry et de la méthode `mark_matched` (ou équivalent)
- **Effort** : 1-2 h
- **Critères de succès** :
  - Comportement de purge **identique** à 3 Hz et à 30 Hz (test sous charge variable)
  - Quick-fix B-01 retiré, valeurs config en `_s` propres
- **Bénéfice annexe** : si B-02 était causé par le cycle de vie, B-03 le résout en cascade

---

## 🟦 Phase 0 — Clôture stabilisation noyau

**Trigger** : ✅ tous les P0 livrés (B-01, B-02, B-03)

### 🟢 A-01 — Audit cohérence noyau tracker

- **Objectif** : passer en revue les zones du noyau tracker récemment modifiées par les correctifs P0 et les patchs de stabilisation antérieurs, valider qu'aucune incohérence ne subsiste
- **Périmètre** :
  - 🔍 **Audit code base requis** : lister les modules touchés au cours des 3 derniers mois sur le tracker et son cycle de vie ; vérifier la cohérence des invariants
- **Livrable** : rapport d'audit + liste de correctifs mineurs si nécessaire
- **Effort** : 2-3 h

### 🟢 A-02 — Hygiène config & invariants

- **Objectif** : valider au chargement de la config tous les invariants critiques (ex. cohérence des durées, plages de valeurs, dépendances entre clés)
- **Exemples d'invariants à vérifier** :
  - `expire_after_lost_s >= lost_after_s` (post B-03)
  - Cohérence des seuils inter-modules
  - 🔍 **Audit code base requis** : recenser toutes les clés config consommées par le pipeline tracker pour produire la liste exhaustive des invariants
- **Effort** : 1-2 h
- **Critère de succès** : config invalide → erreur explicite au démarrage, pas un comportement silencieux dégradé

---

## 🟢 Lot 1 — Stabilisation post-prod

**Trigger** : Phase 0 livrée + retours premiers usages prod stabilisés.

> ⚠️ **Contenu à préciser par audit.** Ce lot regroupe les correctifs de stabilisation identifiés par les retours prod après la clôture Phase 0. Sa composition exacte dépend des observations terrain.

- 🔍 **Audit requis au démarrage du lot** :
  - Logs prod sur la fenêtre post-Phase 0
  - Tickets ouverts sur le tracker en condition réelle
  - Items de backlog antérieurs marqués "stabilisation"
- **Sortie de l'audit** : liste numérotée d'items `B-XX` ou `A-XX` à intégrer dans le Lot 1, avec effort et critère de succès par item
- **Effort total estimé** : à déterminer après audit

---

## 🟡 Lot 2 — Bande passante

**Trigger** : Lot 1 livré + bande passante équipe disponible.

> ⚠️ **Contenu à préciser par audit.** Ce lot regroupe les améliorations qui ne sont ni des bugs ni de la performance critique : refactos opportunistes, dette technique légère, améliorations de lisibilité.

- 🔍 **Audit requis au démarrage du lot** :
  - Backlog de dette technique
  - TODO / FIXME en code base
  - Demandes de refacto issues des revues
- **Sortie de l'audit** : liste numérotée d'items `A-XX` à intégrer
- **Effort total estimé** : à déterminer après audit

---

## 🟠 Lot 3 — Performance (menu à la carte)

**Principe** : tous les items sont **conditionnels et indépendants**. On livre uniquement ceux dont le trigger est satisfait. Les autres restent latents sans bloquer le plan.

### 🟡 P-01 — Batch LK sur pyramide partagée

- **Trigger** : `N masks simultanés ≥ 3` observé en prod
- **Description** : un seul appel LK sur pyramide partagée entre tous les masques, au lieu de N appels indépendants
- **Fichiers concernés** :
  - 🔍 **Audit code base requis** : confirmer le module gérant le fast tracking et le point d'appel LK actuel
- **Effort** : ~3 h
- **Mesure de succès** : réduction du temps de fast tick proportionnelle à N
- **Risque** : faible — modification localisée

### 🟡 P-02 — Shi-Tomasi features

- **Trigger** : fallbacks stale fréquents observés dans logs
- **Description** : remplacer la grille fixe de features LK par une détection Shi-Tomasi
- **Fichiers concernés** :
  - 🔍 **Audit code base requis** : confirmer le module fast tracking et la stratégie de génération de features actuelle
- **Effort** : ~2 h
- **Mesure de succès** : réduction du taux de fallback stale
- **Risque** : faible — algorithme standard OpenCV

### 🟡 P-03 — Cache config lookup

- **Trigger** : profiling confirme un overhead non négligeable sur des appels `cfg.get()` en hot path (ex. dans `_adaptive_margin` ou équivalent)
- **Description** : memoization simple du résultat de lookup
- **Fichiers concernés** :
  - 🔍 **Audit code base requis** : identifier les hot paths qui appellent la config par frame
- **Effort** : 30 min
- **Mesure de succès** : réduction mesurable au profileur sur la fonction ciblée
- **Risque** : nul

---

## 🔮 Phase 2 — Évolutions conditionnelles

**Principe** : aucune feature n'est démarrée sans son trigger. Phase 2 peut rester totalement dormante si aucun signal ne se déclenche.

### 🟡 F-01 à F-07 — LifecycleManager

- **Objectif** : centraliser la gestion du cycle de vie des masques dans un composant dédié, avec règles de transition explicites et testables
- **Trigger global** : signal métier, test ou bug indiquant que la dispersion actuelle de la logique de cycle de vie devient un blocage de maintenance ou d'évolution
- **Périmètre** : 7 sous-tâches à détailler au démarrage
  - 🔍 **Audit code base requis** : recenser tous les sites où la logique de cycle de vie est actuellement dispersée (registry, mask, tracker, associator…) avant de figer le découpage F-01..F-07
- **Effort total estimé** : ~1 jour, +0,5 j si interaction avec F-09
- **Critères de succès** :
  - Toute transition de cycle de vie passe par un point unique
  - Couverture test unitaire complète des transitions
  - Aucune régression sur session de référence

### 🟡 F-08 — Horizon dynamique motion (EMA latence détection)

- **Triggers cumulatifs** :
  - Signal métier : variance de l'horizon de prédiction motion devient un facteur de dégradation observable
  - **ET** B-02 résolu (anti-pattern absolu = ne pas déployer EMA sur `dt` aberrants)
- **Description** : remplacer un horizon `dt_cap` statique par un horizon adaptatif basé sur l'EMA de la latence de détection (`k_horizon * ema_latency`)
- **Garde-fous obligatoires** :
  - Clamp anti-spike sur la mesure brute
  - Cold-start : moins de 5 samples → fallback statique
  - EMA **globale** (pas per-mask)
  - Damping adapté pour éviter oscillations
  - Log HUD du `dt_cap` courant pour diagnostic
- **Fichiers concernés** :
  - 🔍 **Audit code base requis** : confirmer les modules motion et config actuels, et le point d'injection de `dt_cap`
- **Effort** : 1 h dev + 30 min calibration
- **Critères de succès** :
  - `motion.dt` reste stable post-déploiement
  - `capped_pct` stable < 10 %
  - Pas de régression overlap rect moyen
- **Anti-pattern documenté** : déployer F-08 sur un système dont les `dt` sont aberrants → l'EMA absorbe le bug et masque le diagnostic

### 🟡 F-09 — Thread `tick()` dédié

- **Trigger** : jitters de purge mesurés au-delà d'un seuil acceptable après B-03 livré (la purge en secondes ne suffit pas, il faut aussi garantir la régularité d'appel)
- **Description** : externaliser `tick_and_expire` dans un thread dédié cadencé à fréquence fixe (ex. 30 Hz), garantissant la régularité indépendamment de la main loop
- **Fichiers concernés** :
  - 🔍 **Audit code base requis** : identifier le point d'appel actuel de `tick_and_expire` et les contraintes de thread-safety du registry
- **Effort** : 2-3 h
- **Risque** : interaction avec LifecycleManager (F-01..F-07) si livré avant — privilégier l'ordre F-01..F-07 puis F-09 si les deux sont déclenchés
- **Anti-pattern** : déployer F-09 sans B-03 (revient à cadencer un compteur en ticks → aucun bénéfice)

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

- `docs/closed-tickets.md` : tickets clos au fil de l'eau, avec date et lien commit
- `docs/backlog/prediction-historique.md` : R&D gelée, marquée explicitement "GELÉ — voir Plan v4 §Backlog"
- `docs/session-de-reference.md` : protocole de bench canonique réutilisable pour B-02, F-08, Lot 3, F-10
  - 🔍 **Audit requis** : définir le contenu de la session de référence (footage tagué, scénarios standards, métriques mesurées) avant le premier usage en B-02

---

## ✅ Critères globaux V4

- **Séquentialité simple** : ✅ chaque bloc indépendant du suivant
- **Reportabilité** : ✅ toute feature peut être retirée vers un plan ultérieur sans casser la chaîne
- **Couverture bugs** : ✅ B-01, B-02, B-03 explicites en P0
- **Préservation R&D antérieure** : ✅ via F-10 + backlog gelé documenté
- **Triggers mesurables** : ✅ chaque item conditionnel a un signal explicite
- **Audits explicites** : ✅ chaque zone d'incertitude marquée 🔍

---

## 📌 Points d'attention pour la suite

1. **Sections Lot 1 et Lot 2** : volontairement non détaillées car leur contenu dépend d'observations futures. Audit requis au démarrage de chaque lot.
2. **F-01..F-07** : sept sous-tâches à figer au démarrage de la feature, après audit du code base.
3. **F-10** : slot délibérément vide, à remplir post-F-08.
4. **Tous les 🔍 audits code base** sont des prérequis bloquants à la livraison de l'item correspondant — ne pas les sauter.
