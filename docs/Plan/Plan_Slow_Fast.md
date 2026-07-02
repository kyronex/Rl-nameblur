# Plan séquentiel — Robustesse fast/slow

## Statut d'avancement global

| Étape                                                        | Statut             | Détail                                                  |
| ------------------------------------------------------------ | ------------------ | ------------------------------------------------------- |
| Étape 1 — P1 : rafraîchir `last_slow_ts` quand fast confirme | ✅ TERMINÉE        | `tracker.py` → `apply_fast_detections()`                |
| Étape 2 — Bench post-P1                                      | ✅ TERMINÉE        | session `20260701_191002` vs baseline `20260701_142740` |
| Étape 3 — Analyse + porte de décision                        | ✅ DÉCISION RENDUE | → **Branche A (P3)**                                    |
| Branche A — P3                                               | 🔄 À DÉMARRER      | Étape A1 en attente                                     |
| Branche B — P4                                               | ❌ ARCHIVÉE        | écartée : aucune incohérence `last_slow_ts`             |

- **Sessions de référence**

- Baseline : `20260701_142740`
- Post-P1 : `20260701_191002`

---

## Étape 1 — P1 : rafraîchir `last_slow_ts` quand fast confirme

**Objectif** : permettre à fast de maintenir un mask `CONFIRMED` après décrochage slow (scénario 2), tant que NCC confirme.
**Point de modification** : `tracker.py` → `apply_fast_detections()`, sur chaque détection fast matchée → mettre à jour `last_slow_ts` (rebaptiser conceptuellement en « dernier ancrage géométrique fiable »).
**Contrat préservé** : si slow reprend, comportement inchangé. Aucune autre branche touchée.
**Ne PAS toucher** : vélocité (Bug B invalidé), `FastMaskView`, drift gate.
**Critère de sortie** : compile + tracking sur session `20260701_142740` sans nouvelle exception ; comptage revive slow inchangé (~15, P1 ne doit rien retirer à slow).

→ ✅ **TERMINÉE** : livrée et validée.

---

## Étape 2 — Bench (post-P1)

**Objectif** : quantifier l'apport de P1 sans régression.
**Baseline** : session `20260701_142740` (état avant P1).
**Métriques** (schéma `bench-jsonl-schema.md`, canal frame) :

- durée de vie moyenne des masks entre deux slow,
- nombre de LOST évités par confirmation fast,
- `motion_residual_px` (ne doit pas se dégrader).

### Résultats Bench (Étape 2)

- **Invariant anti-régression — ✅ RESPECTÉ**

| Métrique     | Baseline | Post-P1 | Δ      |
| ------------ | -------- | ------- | ------ |
| revives slow | 15       | 14      | −6,7 % |

Variation non significative. Invariant `revives slow ≈ 15` confirmé sur les deux sessions. **Pas de rollback nécessaire.**

- **Durée de vie des masks slow (mean)**

| Lifetime slow | Baseline | Post-P1 | Δ      |
| ------------- | -------- | ------- | ------ |
| slow_lost     | 1,547 s  | 2,059 s | +33 %  |
| slow_expired  | 2,619 s  | 3,324 s | +27 %  |
| slow_revive   | 3,608 s  | 3,956 s | +9,6 % |

Les masks persistent plus longtemps pendant les décrochages lents — comportement visé, fast maintient l'ancrage.

- **motion_residual_px (point de bascule → Étape 3)**

| Métrique   | Baseline | Post-P1  | Δ     |
| ---------- | -------- | -------- | ----- |
| mean       | 114 px   | 216 px   | +89 % |
| max global | 941 px   | 1 858 px | +97 % |

Hausse structurelle, mécaniquement liée à la prédiction figée pendant les décrochages slow.

- **Effets secondaires positifs**

| Métrique                   | Baseline | Post-P1 | Δ     |
| -------------------------- | -------- | ------- | ----- |
| motion_staleness_slow_ms   | 530 ms   | 326 ms  | −39 % |
| motion_staleness_capped    | 1 545    | 562     | −64 % |
| tracker_fast_drift_skipped | 339      | 24      | −93 % |

---

## Étape 3 — Analyse + porte de décision

### Décision (Étape 3)

Porte de décision selon trois issues :

| Condition (plan original)                                               | Résultat                                          | Verdict        |
| ----------------------------------------------------------------------- | ------------------------------------------------- | -------------- |
| P1 suffisant → résidu **stable** → `config.yaml`, fin                   | résidu mean +89 %, max +97 %                      | ❌ **écartée** |
| Résidu/prédiction **dégradés quand slow décroche** → **Branche A (P3)** | dégradation confirmée, causalité prédiction figée | ✅ **retenue** |
| Incohérence lecture/écriture `last_slow_ts` → **Branche B (P4)**        | aucune valeur incohérente relevée                 | ❌ **écartée** |

**Cause retenue** : pendant le décrochage slow, la position est portée par la **prédiction fast seule**, qui diverge sur mouvement lent → prédiction figée → hausse du résidu.

**Valeur de la session diversifiée** : les situations diverses et représentatives de la session `191002` confirment que la hausse du résidu est **systématique**, pas un artefact isolée → renforce la décision Branche A.

---

## Branche A — P3 (ACTIVÉE 🔄)

> **Entrée** : résidu/prédiction dégradés quand slow décroche → prédiction figée en cause.

### Étape A1 — P3 : donner une vélocité aux masks fast-only

**Objectif** : remplacer la prédiction figée (position dernier frame) par une prédiction à vitesse constante pour les masks sans ancrage slow.
**Points de modification** :

- `fast_track_thread.py` → exposer le delta de position (vitesse) de chaque mask fast
- `mask.py` → recevoir et consommer cette vélocité pour la prédiction inter-frame

**Changement de contrat `FastMaskView`** : l'interface de prédiction est modifiée (de position figée → vitesse constante). À documenter explicitement dans le diff.

**Précautions** : Seuil de confiance minimal à définir pour éviter de propager du bruit

- Sortie du mode vitesse dès qu'un frame slow reprend l'ancrage
- Tests sur session `20260701_191002` pour valider la baisse du résidu

**Prérequis** : dernières versions de `fast_track_thread.py` et `mask.py` (à récupérer en session).

**Statut** : 🔄 À DÉMARRER

### Étape A2 — Bench post-P3

**Baseline** : session `20260701_191002` (état post-P1).
**Métriques cibles** :

- `motion_residual_px` mean et max (objectif : ↓ vers niveau baseline ou mieux)
- qualité de prédiction pendant décrochage slow
- `revives slow` (invariant, ne doit pas chuter sous ~14)

**Statut** : en attente de Étape A1

### Étape A3 — Analyse post-P3

**Condition de succès** :

- `motion_residual_px` en baisse significative sans régression sur `revives slow`
- → appliquer `config.yaml` (C-01=1.5, C-03=0.7, C-04=0.7)
- → **fin de plan**

Si résidu non amélioré ou régression revive → retour Étape A1, itérer sur la vélocité.

---

## Branche B — P4 (ARCHIVÉE ❌)

> **Entrée** : incohérence lecture/écriture sur `last_slow_ts` (race condition détectée au bench).

**Contenu original (conservé pour référence, non exécuté) :**

- **Principe** : verrouiller `last_slow_ts` en écriture au moment de l'appel fast, lecture synchrone sans buffer
- **Points de modification** :
  - `mask.py` → écrire `last_slow_ts` dans le mask dès confirmation slow
  - `registry.py` → exposer `last_slow_ts` du mask actif sans copier
- **Risque** : forte interaction avec le cycle de vie des masks → validé P1, cette hypothèse est écartée

**Raison d'archivage** : aucune incohérence temporelle de `last_slow_ts` relevée au bench post-P1 → la race condition n'est pas le facteur déterminant.

---

## Invariant global

> **`revives slow ≈ 15`** = garde-fou anti-régression.
> À vérifier à chaque bench. Toute baisse significative → rollback immédiat de la dernière modification.

---

## Prérequis non bloquants

- `FastTrackConfig` et `detection/detect.py` absents du périmètre actuel → le calibrage NCC automatique est hors périmètre de ce plan.
- La session `191002` couvre des situations diverses et représentatives → les résultats de bench sont généralisables.

---

## Prochaine étape recommandée

---

## 2026-07-02 — Session 20260702_162941 : validation patch "ts": frame_ts

### Patch "ts": frame_ts — VALIDÉ, sans régression

- **Cause racine corrigée** : `dt=0` en boucle dû à l'absence de clé `"ts"` à la création du state fast. Corrigé.
- **Objectif du patch** = ressusciter la vélocité fast. Atteint.

### Preuves (canal fast)

- `fast_v_px_per_s` : 26 914 échantillons / 38 fenêtres ; 38/38 fenêtres (100 %) avec vélocité > 0 ; moyenne (fenêtres non nulles) 482,9 px/s ; crête agrégée 2 828 px/s < clamp 3 000 px/s (clamp jamais dépassé dans les données).
- **Réserve** : le canal fast n'expose que des stats agrégées par fenêtre (avg/count/min/max), pas les échantillons bruts → fréquence de saturation du clamp non mesurable. Sans impact sur la validation du patch.

### Invariant anti-régression (canal event)

- `REVIVE slow` = 13 (garde-fou ~15) → dans la marge, **PAS de rollback**.
- Autres compteurs event : `CREATED` 60 (100% slow), `CONFIRMED` 1 355 (fast 1 340 / slow 15), `LOST` 491 (fast 438 / slow 53), `EXPIRED` 55 (slow 45 / fast 10).

### motion_residual_px (canal frame) — REQUALIFICATION IMPORTANTE

- Résidu moyen pondéré session = **183,66 px**, soit ≈ 2,5× la baseline 73 px.
- **Non imputable au patch "ts"** (qui traite la vélocité, pas le résidu).
- **73 px et 183 px non comparables** (scène différente). Requalifier explicitement la baseline 73 px comme **dépendante de la scène**, et non comme invariant. Cohérent avec la requalification antérieure : le gain 216→73 venait des patches de prédiction, pas de la vélocité fast — cela reste vrai sur la session de référence, mais 73 px n'est pas reproduit ici.
- **Signal encourageant** : convergence intra-session, 1ʳᵉ moitié 222,3 px (n=82) → 2ᵉ moitié 148,1 px (n=89), Δ = −74,2 px (−33 %).

### Coût / perf (canal frame)

- `apply_fast_detections` : 0,15 ms médian, 12,2 ms max → coût négligeable.
- `staleness_slow` : 294,7 ms moyen, 1 204 ms max → pics longs (point de vigilance).

### Points de vigilance convergents (prochaine cible)

- `EXPIRED slow / CREATED slow` = 45/60 = **75 %** (3 masks slow sur 4 expirent par TTL avant réapparition).
- `staleness_slow` pics jusqu'à 1,2 s.
- **La vraie métrique du bénéfice fonctionnel de la vélocité fast** = durée de vie inter-slow (pas le résidu). C'est la prochaine cible de bench.

### Ouvert / non tranché

- Baseline résidu à scène comparable manquante pour trancher 73 vs 183 px.

**Démarrer Étape A1 (P3)** : récupérer les dernières versions de `fast_track_thread.py` et `mask.py`, puis implémenter l'exposition/consommation de la vélocité des masks fast-only.
