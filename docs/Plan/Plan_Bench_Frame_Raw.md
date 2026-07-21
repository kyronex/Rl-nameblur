# 📘 Plan Feature — Migration `bench_frame.jsonl` → Format Raw Hybrid

> **Document autoportant.** Aucune référence externe. Le format Raw Hybrid **remplace** l'ancien format agrégé. Aucune rétro-compatibilité requise.

---

## 📐 Conventions

| Élément                    | Définition                                                                    |
| -------------------------- | ----------------------------------------------------------------------------- |
| **Raw Hybrid**             | Scalaire si `count=1`, array si `count>1` → une seule clé `probes_raw`        |
| **Historique brut**        | Deque interne préservant les N dernières mesures par probe                    |
| **Count**                  | Nombre de fois qu'une probe a été appelée dans une même frame                 |
| **Polymorphe**             | Une valeur dans `probes_raw[name]` est toujours `list[float]` (homogène)      |
| **Validation comparative** | Vérifier cohérence entre JSON old (agrégé) et new (raw) sur session identique |

**Statut feature** : 🟢 Actif
**Priorité** : 🟡 Moyenne (optimisation stockage + analytique)
**Périmètre** : `bench/bench.py`, `bench/jsonl_writer.py`, documentation
**Rétro-compatibilité** : ❌ AUCUNE — le format raw remplace l'agrégé

## 📋 Référence des arbitrages (validée — 9/9)

| #   | Sujet                                | Décision                                                                                        |
| --- | ------------------------------------ | ----------------------------------------------------------------------------------------------- |
| 1   | Schéma exact de `probes_raw`         | liste brute `list[float]`, pas de pré-agrégation                                                |
| 2   | Stratégie de migration               | Suppression sèche — pas de rétrocompat, phase dev                                               |
| 3   | Périmètre                            | counts + gauges conservés inchangés ; seules les probes passent en raw                          |
| 4   | Reconstruction `count`               | `len(list)` comme source de vérité, pas de champ `count` redondant                              |
| 5   | `format_probe_value()` sur `count=0` | retourner `[]` et émettre la clé `"key":[]`                                                     |
| 6   | Reset inter-frames                   | réutilise `_frame_probes.clear()` dans `snapshot_frame()`, delta limité au bloc d'agrégation    |
| 7   | Validation                           | Diff JSON avant/après manuel — simple constat de mise en place                                  |
| 8   | `frame_dumper.py`                    | Hors scope — aucun impact, débat annexe écarté de ce lot                                        |
| 9   | Structure de fichiers                | Le code actuel fait foi comme structure fonctionnelle — `bench/` existe déjà, aucun déplacement |

---

## 🎯 Objectifs

| Objectif                        | Métrique                                 | Critique  |
| ------------------------------- | ---------------------------------------- | --------- |
| Réduire taille fichier frame    | -60 à -70% vs format agrégé              | ✅ Dur    |
| Préserver 100% de l'information | Aucune mesure perdue, historique complet | ✅ Dur    |
| Cohérence old↔new validée       | Toute valeur agrégée dérivable du raw    | ✅ Dur    |
| Analytique simplifiée           | Parser frame en < 5ms pour 1000 lignes   | 🟡 Souple |
| Détecter anomalies (count>1)    | Identifier probes "bruyantes" facilement | 🟡 Souple |

---

## 🗺️ Architecture décisionnelle

```text
┌─────────────────────────────────────────────────────────────┐
│ A0 — AUDIT CODE BASE (obligatoire)                          │
│  ├─ Identifier tous les appels à bench.probe() par fichier  │
│  ├─ Vérifier la structure actuelle de BenchRegistry         │
│  ├─ Confirmer l'absence de breaking changes récents         │
│  └─ Valider applicabilité du plan                           │
└──────────────────┬──────────────────────────────────────────┘
                   │ ✅ Plan applicable ?
         ┌─────────▼──────────┐
         │ OUI → Continuer    │ NON → Retour audit
         └─────────┬──────────┘ + reformulation plan
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ A1 — DESIGN : Structure historique brut                     │
│  ├─ Déf. `deque(maxlen=N)` par probe                        │
│  ├─ Arbitrage : N=10 (capture bursts typ.)                  │
│  └─ Impact mémoire : ~4 KB total                            │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ A2 — DESIGN : Logique de sérialisation                      │
│  ├─ Fonction `format_probe_value(stats) → float|list`       │
│  │   ├─ count == 1 → scalaire                               │
│  │   ├─ count > 1  → array                                  │
│  │   └─ count == 0 → skip                                   │
│  └─ Clé JSON unifiée : `probes_raw`                         │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ A3 — SUPPRESSION format agrégé (décision actée)             │
│  ├─ Format `probes` {avg,max,min,count} → SUPPRIMÉ          │
│  ├─ Aucun dual-mode, aucun deprecation window               │
│  ├─ Justification : format raw contient toute l'info         │
│  │   └─ Agrégats dérivables du raw à la volée si besoin      │
│  └─ Cleanup direct du code d'agrégation frame                │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ A4 — DESIGN : Polymorphe float|list (acté)                  │
│  ├─ Une clé, deux types possibles                           │
│  ├─ Parser : 1 isinstance check                             │
│  └─ Schéma : "scalaire par défaut, array si multi-mesure"   │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ B1 — IMPLÉMENTATION : BenchRegistry historique brut         │
│  ├─ Ajouter `_probe_values: dict[str, deque]`              │
│  ├─ Modifier `probe()` : append à deque                     │
│  ├─ Ajouter getter `get_probe_values(name)`                 │
│  └─ Ajouter `reset_probe_values()`                          │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ B2 — IMPLÉMENTATION : Fonction format_probe_value()         │
│  ├─ Signature : (name, stats) → float | list | None         │
│  ├─ Localisation : inline dans `bench/bench.py` (recommandé)  │
│  └─ Corps : dispatch selon count                            │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ B3 — IMPLÉMENTATION : Refonte snapshot_frame()              │
│  ├─ Supprimer boucle agrégée (ancien format)               │
│  ├─ Générer clé `probes_raw` (scalaires + arrays)          │
│  ├─ Conserver filtres fast_* / writer_*                     │
│  └─ Conserver counts + gauges (inchangés)                   │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ B4 — IMPLÉMENTATION : config.yaml                           │
│  ├─ Path inchangé (`bench_frame.jsonl`)                     │
│  ├─ NOUVEAU optionnel : `preserve_probe_history` (default true)
│  └─ Commentaires YAML explicites                            │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ B5 — IMPLÉMENTATION : Cycle reset par frame                 │
│  ├─ Appel `reset_probe_values()` post-snapshot              │
│  └─ Isolation stricte inter-frames                          │
└──────────────────┬──────────────────────────────────────────┘
                   ▼
              ✅ LIVRAISON
```

---

## 🔍 Audit code base (Phase A0) — Checklist

**Objectif** : valider que le plan est applicable avec le codebase actuel.

### Fichiers à auditer

```text
bench/
  ├─ bench.py                    (BenchRegistry, probe, _probes structure)
  ├─ jsonl_writer.py             (snapshot_frame, filtres fast/writer)
  └─ [lifecycle.py / ...]        (autres modules bench)

config/
  └─ config.yaml                 (debug.bench.frame.*)

```

## 🎬 Étapes directrices (Phases A-E)

### **Phase B — Implémentation** (3 jours)

#### B1 — BenchRegistry historique (1 j)

- **Résumé** : ajouter `_probe_values: dict[str, deque]` + getter + reset
- **Tâches** :
  1. Ajouter `_probe_values` init dans `__init__`
  2. Modifier `probe(name, value)` : append à deque (+ stats cumulatives internes si conservées pour agg/fast)
  3. Ajouter `get_probe_values(name) → list[float]`
  4. Ajouter `reset_probe_values()`
- **Note** : les canaux `agg` et `fast` peuvent conserver leur agrégation propre — cette feature ne touche **que le canal frame**
- **Critère d'acceptation** : deque FIFO fonctionnel, canaux agg/fast intacts

#### B2 — Fonction format_probe_value() (0.5 j)

- **Résumé** : implémenter dispatch polymorphe
- **Signature** : `format_probe_value(probe_name, stats) → float | list[float] | None`
- **Localisation** : inline dans `bench/bench.py` (recommandé ; fichier `bench/formatters.py` non requis)
- **Logique** :
  - count=1 → scalaire
  - count>1 → list
  - count=0 → None
- **Critère d'acceptation** : dispatch correct, edge cases gérés

#### B3 — Refonte snapshot_frame() (1 j)

- **Résumé** : remplacer format agrégé par `probes_raw` hybrid
- **Tâches** :
  1. Supprimer boucle d'agrégation frame `{avg,max,min,count}`
  2. Générer `probes_raw` via `format_probe_value()`
  3. Conserver filtres fast*\* / writer*\* (inchangé)
  4. Conserver counts + gauges (inchangé)
- **Critère d'acceptation** : JSON valide, JSONL parseable, format agrégé absent

#### B4 — config.yaml (0.5 j)

- **Résumé** : ajouter option `preserve_probe_history`
- **Tâches** :
  1. Ajouter `debug.bench.frame.preserve_probe_history: true`
  2. Docstring : "If false, deque(maxlen=1) → memory savings"
  3. Path frame inchangé
- **Critère d'acceptation** : config loads, no validation error

#### B5 — Cycle reset par frame (facultatif si couvert par B1)

- **Résumé** : garantir isolation inter-frames
- **Tâches** :
  1. Appel `reset_probe_values()` post-snapshot
  2. Documenter cycle de vie
- **Critère d'acceptation** : isolation vérifiée en C2

---

## ✅ Définition de fait (DoD) — Feature complète

Une feature est **livrée** si et seulement si :

1. **A0** : Audit code base ✅ + go/no-go tracé + session dual-branch confirmée faisable
2. **A1-A4** : Tous les arbitrages documentés et approuvés
3. **B1-B5** : Implémentation fonctionnelle (canal frame migré, agg/fast intacts)
4. **C1-C3** : Validation comparative ✅ — **0 divergence old↔new** (critère dur)
5. **D1-D3** : Documentation complète (schéma, README, parsers, script validation)
6. **E1** : Tous les critères (mesurables + qualitatifs + déploiement) ✅
7. **Format agrégé frame supprimé** : plus de `{avg,max,min,count}` dans le canal frame
8. **Rollback plan** : documenté

---

## 🎯 Anti-patterns à éviter

| Anti-pattern                         | Raison                          | Conséquence                         |
| ------------------------------------ | ------------------------------- | ----------------------------------- |
| Sauter A0 audit                      | Codebase change oublié          | B1 bloqué, itérations inutiles      |
| Supprimer format agrégé **avant** C1 | JSON OLD perdu pour comparaison | Validation comparative impossible   |
| Session old↔new non iso              | Comparaison invalide            | Faux divergences, faux verts        |
| Tolérance flottante trop laxiste     | Masque vraies divergences       | Cohérence non prouvée               |
| Sauter C2 cohérence dérivée          | "Le raw contient tout, évident" | Régression silencieuse non détectée |
| Toucher canaux agg/fast              | Hors périmètre                  | Régression sur canaux non ciblés    |
| Documentation après code             | "Update later"                  | Documentation incomplète            |
| Merging sans 0 divergence            | "Presque bon"                   | Perte d'info non détectée en prod   |

---
