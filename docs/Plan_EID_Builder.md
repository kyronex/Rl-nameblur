# Plan EID Builder

> Document normatif du plan EID. Chaque règle porte son « Pourquoi ». Aucun renvoi ne compare une option retenue à une option écartée ; aucune formulation de type « la formulation reste autonome ». Aucun renvoi vers un document antérieur ou une version précédente du plan.

## Statut du document — North Star

### Rôle

Ce document est la **référence d'intention** du EID Builder : principes,doctrine d'association, contrats de sortie, bornes. Il décrit la cible,pas l'état du code.

Pour tout ce qui concerne **l'état réel du code, l'ordre d'implémentation et les décisions tranchées**, `Patch_Plan.md` fait foi. En cas de contradiction entre les deux documents, `Patch_Plan.md` l'emporte ; le présent document n'est pas corrigé lot par lot, il sera réaligné une seule fois en finalisation (L25/L27).

Règle pour les lots L14 à L27 : aucun symbole, nom de clé ou formule de ce document ne doit être réintroduit dans le code sans vérification préalable dans `Patch_Plan.md` (registre des décisions §5 et §6).

### Delta assumé avec `Patch_Plan.md`

Les points ci-dessous sont connus et acceptés. Ils ne sont pas des erreurs à corriger dans le code.

| Sujet                                                                                    | Ce document (North Star)                              | Décision en vigueur (`Patch_Plan.md`)                                                                                                         |
| ---------------------------------------------------------------------------------------- | ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `SourceStats` (§3, l.161-229, l.296, l.1974)                                             | Façade typée sur `Counters`, requise par `from_entry` | **Supprimée** (L03). Il ne reste qu'un commentaire dans `stream_eid_builder.py`, retiré en L25.                                               |
| `dedup_window_s` (l.271, l.1976)                                                         | Mentionné comme encore présent dans le code           | Renommage `dedup_ttl_s` **appliqué** (L02). `dedup_window_s` ne doit pas être réintroduit.                                                    |
| `_CALIB_PROBE`, `set_calib_probe`, `get_calib_probe` (l.815-817, l.1058, l.2002, l.2013) | Listés comme présents ou à activer                    | **Supprimés** (L10). Les tables l.2002 et l.2013 sont historiques.                                                                            |
| Géométrie slow sur `cluster.anchor.anchor` (l.319, l.471)                                | Décrit comme violation connue                         | Migration vers `last_obs` prévue en **L14**. Jusque-là le code calcule `d_pos`/`d_dim` sur l'ancre.                                           |
| `geo_shortcircuit`                                                                       | Non mentionné                                         | Motif encore émis par `builder.py` ; reformulé en L14.                                                                                        |
| Compteur `admit_reject_cost` (l.409, l.1986)                                             | Nom cible                                             | Le code porte encore `COST_REJECT_SLOW` ; renommage en **L15**.                                                                               |
| `admit_reject_families`, `min_cost_families`                                             | Non mentionnés                                        | `min_cost_families = 2` en place ; compteur ajouté en **L18** ; décision 2/1/hybride en **L27**.                                              |
| `f_hs`, `hs_px_ref` (l.508-518, l.1507-1531, l.1707)                                     | Formule complète de `QualityScore`                    | Contrat fixé en L12 (`f_hs` depuis `sig_hs_px[1]`), **calcul en L21**, `hs_px_ref` calibré en **L27**. Le code renvoie `f_hs=None` jusque-là. |
| `promo_outcome`                                                                          | Non mentionné                                         | Introduit en L13 : état vivant `{early, window, locked_no_promo, None}`, état finalisé sans `None`, assertion en **L23**.                     |
| `_dispersion_for_family` (l.1865)                                                        | Fonction proposée dans `signature_ref.py`             | Le code livre `_dispersion_for_family_e` (L01), **conservée** pour la calibration L18/L27. Le nom sans suffixe n'existe pas.                  |
| `AnchorState`                                                                            | Machine à états décrite sans ce nom                   | Enum `AnchorState {PROBATION, LOCKED}` introduite en L12, distincte de `ClusterState`.                                                        |
| `LiveSource` (l.161, l.178, l.305, l.1927)                                               | Hors périmètre, Bloc F                                | Reste hors code ; planifié en **L24**. Cohérent, pas de delta de fond.                                                                        |
| `QualityScore` « absent de la codebase » (l.2041)                                        | Table historique                                      | Présent depuis L12 ; composantes encore optionnelles.                                                                                         |

Les sections concernées ne sont **pas** réécrites : elles restent lisibles comme intention d'origine, avec ce tableau comme grille de lecture.

## Encadré — codebase à trois niveaux

Le plan distingue explicitement trois niveaux de rappel à la codebase :

1. **Contrats et invariants** — ce document.
2. **Périmètre de propagation** — douze fichiers listés ci-dessous ; chaque modification est rattachée à un bloc.
3. **Rappels en table et checklist de pré-vol** — chaque table de propagation rappelle le fichier concerné ; l’annexe fournit une checklist de pré-vol par fichier.

Le périmètre est limité à : `signature.py`, `stream_eid_builder.py`, `batching.py`, `builder.py`, `calibration.py`, `cluster.py`, `config.py`, `cost.py`, `counters.py`, `eid_distance.py`, `observation.py`, `signature_ref.py`.

## Préconditions à valider avant implémentation

- [ ] Vérifier dans la codebase les signatures exactes des classes, méthodes, paramètres et compteurs cités.
- [ ] Vérifier la présence des champs de `ClusterSlow`, `ClusterFast`, `Observation`, et des objets de signature dans les fichiers concernés.
- [ ] Vérifier que toutes les clés de compteurs consommées par le rapport final figurent dans la liste fermée `_KEYS`.
- [ ] Vérifier les chemins d’écriture des empreintes et de la calibration dans le point d’entrée.
- [ ] Vérifier les seuils de configuration et leur validation dans `StreamConfig`.

## Inventaire du périmètre

| Fichier               | Responsabilité à vérifier                    |
| --------------------- | -------------------------------------------- |
| signature.py          | construction et sérialisation des signatures |
| stream_eid_builder.py | point d’entrée, flux et rapport final        |
| batching.py           | lecture, lots et déduplication               |
| builder.py            | association, lifecycle et compteurs          |
| calibration.py        | écriture des paires de calibration           |
| cluster.py            | état, observation, snapshot et purge         |
| config.py             | paramètres et bornes                         |
| cost.py               | matrice de coûts et temps                    |
| counters.py           | clés fermées et statistiques                 |
| eid_distance.py       | distance et gates de signature               |
| observation.py        | contrat de parsing et admission              |
| signature_ref.py      | référence et dispersion                      |

## Matrice de rapprochement bloc × fichier

| Bloc                        | Fichiers concernés                                                                  |
| --------------------------- | ----------------------------------------------------------------------------------- |
| A — Régime de lot unique    | batching.py, config.py, counters.py, builder.py                                     |
| A-bis — Contrat d’entrée    | observation.py, batching.py, config.py, counters.py                                 |
| B — Doctrine d’association  | builder.py, cost.py, cluster.py, counters.py                                        |
| C1 — Référence de signature | signature_ref.py, cluster.py, builder.py, counters.py                               |
| C2 — Dispersion             | signature_ref.py, eid_distance.py, cluster.py, config.py, counters.py               |
| C3 — Distance de signature  | eid_distance.py, cost.py, calibration.py, config.py, signature.py, signature_ref.py |
| C4 — Gating géométrique     | builder.py, cluster.py, config.py, counters.py                                      |
| C5 — Signatures partielles  | observation.py, eid_distance.py, config.py, counters.py                             |
| D — Contrat de sortie       | cluster.py, signature_ref.py, counters.py, stream_eid_builder.py                    |
| E — Bornes & performance    | cluster.py, config.py, signature_ref.py, counters.py                                |
| F — Mode flux               | stream_eid_builder.py, batching.py, builder.py, counters.py                         |

## 1. Principes directeurs

## 2. Régime de lot unique (Bloc A)

**Pourquoi** : A1 = régime unique · `batch_zero_slow` sans préfixe · `batch_size_max` propriété de `BatchReader`.

> **Fichiers touchés** : `batching.py`, `config.py`, `counters.py`, `stream_eid_builder.py`, `builder.py`.

### A.0 — Principe fondateur

Un lot est défini par la **frame** (`lot = frame`). Les observations sont regroupées par `frame_id` ; l'absence de `frame_id` est rejetée strictement avant le découpage en lots et comptée dans `src_obs_missing_frame_id`. L'ordre des frames est strict ; tout retour en arrière incrémente `batch_frame_out_of_order`.

**Invariants :**

| #          | Invariant                                                                                                                                          |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **INV-A1** | Un lot est fermé à l’arrivée d’une observation slow; il contient les observations accumulées jusqu’à cette fermeture selon le contrat BatchReader. |
| **INV-A2** | `frame_id` absent : rejet strict avant le découpage en lots, compteur `src_obs_missing_frame_id`.                                                  |
| **INV-A3** | L'ordre des frames est strict ; un retour en arrière incrémente `batch_frame_out_of_order`.                                                        |

INV-A1 rend **structurellement impossible** la présence de deux observations concurrentes du même mask dans une matrice hongroise.

### A.0.b — Implémentation BatchReader et fermeture sur slow

`BatchReader` (`batching.py:124-162`) doit porter la sémantique INV-A1: fermeture nominale d’un lot à l’arrivée d’une observation slow, transmission via `record_batch`, et compteurs de lots/observations cohérents. Le code constaté ferme actuellement sur `epsilon_close_s` (`batching.py:157-162`); ce paramètre est conservé comme timeout de sécurité dégradé lorsque le slow manque ou arrive en retard. Le test doit couvrir slow/fast, `record_batch`, `batches_read`, `batch_obs_cumul`, `n_batch_closed_timeout` et l’ordre des frames.

### A.1 — Ce qui disparaît

| Élément                                                                    | Fichier       | Justification                                                                                                             |
| -------------------------------------------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `BatchReader.sliding` (paramètre + 2 branches)                             | `batching.py` | Régime unique. Le chemin `sliding=True` n'a **jamais été exécuté** (`sliding=False` en dur dans `stream_eid_builder.py`). |
| `BatchReader._max_span_s` (méthode)                                        | `batching.py` | La borne devient lecture directe de config.                                                                               |
| `StreamConfig.epsilon_close_s`                                             | `config.py`   | Pilote uniquement le timeout de sécurité; la fermeture nominale se fait sur slow.                                         |
| `StreamConfig.batch_max_span_factor`                                       | `config.py`   | Multiple d'un paramètre disparu.                                                                                          |
| `assert temporal_radius_slow_s > temporal_radius_fast_s > epsilon_close_s` | `config.py`   | Assertion à revoir avec la nouvelle sémantique de timeout (paramètre conservé).                                           |

### A.2 — Ce qui apparaît

#### A.2.a — `StreamConfig.batch_max_span_s`

```txt
batch_max_span_s: float = 0.10
```

Borne absolue de secours. Ferme un lot si aucun slow n'arrive dans l'intervalle — cas du flux fast orphelin.

#### A.2.b — Assertions `__post_init__` — forme finale

```txt
assert temporal_radius_evict_s > temporal_radius_slow_s
assert temporal_radius_slow_s  > temporal_radius_fast_s
assert 0.0 < batch_max_span_s  < temporal_radius_fast_s
```

1. L'éviction survient **après** que le rapprochement d'identité soit devenu impossible.
2. Le rayon d'identité est **plus large** que celui de continuité — une identité se retrouve après coupure, une piste géométrique non.
3. Un lot ne dépasse **jamais** le rayon de continuité.

#### A.2.c — `BatchReader.batch_size_max`

```txt
@property
def batch_size_max(self) -> int
```

Maximum du cardinal de lot observé depuis l'instanciation. Mis à jour à chaque émission.

**Rationale** : `Counters` est un registre à sémantique **additive** (`bump`). Un maximum n'y a pas sa place sans introduire `set_max()` — complexification injustifiée pour une métrique qui appartient naturellement au producteur de lots.

### A.3 — Interface cible

```txt
BatchReader
  ├─ __init__(source: ObservationSource, config: StreamConfig, counters: Counters)
  ├─ __iter__() -> Iterator[List[Observation]]
  ├─ batch_size_max -> int   (property)
  └─ état : _anchor_ts: float · _pending: Optional[Observation] · _current: List[Observation]
```

Le paramètre `builder` disparaît de la signature — `BatchReader` n'a aucune raison de connaître son consommateur (voir aussi INV-B1).

### A.3-bis — INV-A1 : fermeture nominale sur slow, timeout de sécurité

Le lot se ferme selon deux motifs explicitement distincts :

1. **Slow (nominal)** — à l’arrivée d’une observation `slow`, `BatchReader` ferme le lot courant conformément à INV-A1. Cette fermeture est le chemin normal.
2. **Timeout (dégradé)** — si l’observation `slow` manque ou arrive en retard, `epsilon_close_s` ferme le lot resté ouvert. Ce garde-fou est conservé pour borner la mémoire et la latence; il doit rester visible dans les métriques et ne doit pas être présenté comme une fermeture nominale.

Le compteur dédié proposé est `n_batch_closed_timeout`, incrémenté uniquement pour le second motif. Il est **à créer** dans `counters.py::_KEYS` (aucun compteur équivalent n’est présent dans la version de référence). `batches_read` et `batch_obs_cumul` restent les compteurs généraux d’émission.

Le sens de `epsilon_close_s` change donc de « fenêtre de regroupement » à « timeout de sécurité ». La valeur actuelle est `0.15` s (`config.py:43`); elle a été réglée pour l’ancienne sémantique de fenêtre et doit être revue avant implémentation du nouveau contrat.

### A.4 — Compteurs

Toutes clés déclarées dans `counters.py::_KEYS`.

| Clé                      | Nature | Incrémenté quand                                               |
| ------------------------ | ------ | -------------------------------------------------------------- |
| `batches_emitted`        | stat   | Un lot est émis                                                |
| `batch_obs_cumul`        | stat   | +`len(batch)` à chaque émission                                |
| `batch_closed_by_slow`   | stat   | Fermeture nominale                                             |
| `n_batch_closed_timeout` | stat   | Fermeture dégradée par timeout de sécurité (`epsilon_close_s`) |
| `batch_zero_slow`        | stat   | Lot émis sans aucune observation slow                          |

`batch_zero_slow` est **sans préfixe `anomaly_`** : un lot fast-only est un régime légitime (perte temporaire du détecteur slow), pas une anomalie. Son ratio par rapport à `batches_emitted` est un indicateur de santé du pipeline amont, pas un défaut EID.

### A.5 — Table de propagation

| Fichier                 | Modification                                                                                                                                                                                      | Nature        |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| `batching.py`           | Suppression `sliding` + `_max_span_s`. Régime unique.                                                                                                                                             | Structurel    |
| `config.py`             | Conservation de `epsilon_close_s` comme timeout de sécurité; revue de sa valeur, car `0.15` a été réglé pour l’ancienne sémantique de fenêtre; `batch_max_span_factor` et assertions à réaligner. | Structurel    |
| `counters.py`           | +5 clés                                                                                                                                                                                           | Additif       |
| `stream_eid_builder.py` | Construction sans `sliding`. Rapport lit `reader.batch_size_max`.                                                                                                                                 | 🟠 Adaptation |
| `builder.py`            | Aucune                                                                                                                                                                                            | Néant         |

---

## 3. Contrat d'entrée (Bloc A-bis)

**Pourquoi** : + (1) décorateur `DedupSource` + `SourceStats` dédié · façade typée sur `Counters` · renommage `dedup_ttl_s` · `LiveSource` hors périmètre, Bloc F nommé.

> **Fichiers touchés** : `observation.py`, `batching.py`, `config.py`, `stream_eid_builder.py`.

### A-bis.0 — Principe

```txt
Une source produit des Observations. Elle ne connaît ni le builder,
ni le batching, ni la politique de dédup.
La dédup est un DÉCORATEUR, pas une responsabilité de la source.
```

**Invariants :**

| #          | Invariant                                                                                                       |
| ---------- | --------------------------------------------------------------------------------------------------------------- |
| **INV-B1** | Aucun module de la chaîne d'entrée ne référence `builder`. Zéro occurrence de `hasattr(builder, ...)`.          |
| **INV-B2** | Toute `ObservationSource` est interchangeable. Le `BatchReader` ne distingue pas `JsonlSource` de `LiveSource`. |
| **INV-B3** | La dédup est optionnelle et composable — `DedupSource(JsonlSource(...))`.                                       |

### A-bis.1 — Chaîne cible

```txt
SourceStats(counters)
    ↓
JsonlSource(path, stats)          ← lecture pure, zéro dédup
    ↓
DedupSource(inner, config, stats) ← décorateur optionnel
    ↓
BatchReader(source, config, counters)
    ↓
StreamEIDBuilder.process_batch(batch)
```

### A-bis.2 — `SourceStats` — façade typée sur `Counters`

```txt
class SourceStats:
    __slots__ = ("_c",)
    def __init__(self, counters: Counters) -> None: ...

    # Méthodes de comptage — une par anomalie de parsing
    def built(self) -> None:              # → src_obs_built
    def source_fallback(self) -> None:    # → src_obs_source_fallback
    def ts_fallback(self) -> None:        # → src_obs_ts_fallback
    def degenerate_box(self) -> None:     # → src_obs_degenerate_box
    def missing_signature(self) -> None:  # → src_obs_missing_signature
    def malformed(self) -> None:          # → src_obs_malformed
    def read_total(self) -> None:         # → src_obs_read_total

    # Dédup (écrites par DedupSource)
    def dedup_skipped(self) -> None:      # → src_dedup_skipped
    def dedup_purge(self) -> None:        # → src_dedup_purges
```

**Rationale** — `Counters` est déjà le puits unique du module, générique, à clés fermées, consommé par `stream_eid_builder.py` via `stats()` / `anomalies()`. Introduire une seconde structure de comptage fragmenterait le rapport et dupliquerait le mécanisme. La façade donne le **typage explicite** voulu par INV-B2 (une source reçoit un objet dont le contrat est visible) sans créer de second puits.

**Préfixe `src_`** : toutes les clés écrites via `SourceStats` le portent. Le rapport peut ainsi isoler la section « Qualité de l'entrée » par simple filtrage de préfixe, sans table de correspondance.

**Note** : les clés `src_*` ne portent **pas** `ANOMALY_PREFIX`. Ce sont des statistiques de lecture, pas des anomalies de traitement. Un `src_obs_degenerate_box` élevé signale un problème **amont** (détecteur), pas un dysfonctionnement EID.

### A-bis.3 — `Observation.from_entry` — signature cible

```txt
@classmethod
def from_entry(cls, entry: dict, stats: SourceStats) -> Optional[Observation]
```

`stats` est **requis**, non optionnel. Une source sans `SourceStats` n'est pas une source valide (INV-B2).

**Table de correspondance des anomalies de parsing :**

| Condition                                 | Méthode appelée       | Retour                       |
| ----------------------------------------- | --------------------- | ---------------------------- |
| Toute entry lue                           | `read_total()`        | —                            |
| `source` absent ou hors `{"slow","fast"}` | `source_fallback()`   | Observation avec fallback    |
| `ts` absent → repli sur `frame_ts`        | `ts_fallback()`       | Observation                  |
| `w <= 0` ou `h <= 0`                      | `degenerate_box()`    | **`None`** (rejet)           |
| Signature absente ou vide                 | `missing_signature()` | Observation (signature vide) |
| Exception de parsing                      | `malformed()`         | **`None`** (rejet)           |
| Construction réussie                      | `built()`             | Observation                  |

> **Changement de comportement assumé** : la politique sur boîte dégénérée est fixée à **rejet**. Une boîte de surface nulle ne peut alimenter aucun coût géométrique. Sur le corpus existant, cela fera baisser `src_obs_built` et monter `src_obs_degenerate_box`. À noter lors de la comparaison avant/après.

Le tuple de retour actuel `(obs, reject_reason, warn)` disparaît : les compteurs remplacent les canaux de retour secondaires.

### A-bis.4 — `DedupSource` — décorateur

```txt
class DedupSource(ObservationSource):
    def __init__(self, inner: ObservationSource,
                 config: StreamConfig,
                 stats: SourceStats) -> None
```

**Clé de dédup** — conservée à l'identique :

```txt
("slow", round(ts, 4), round(ldr_x, 1), round(ldr_y, 1), round(ldr_w, 1), round(ldr_h, 1))
```

Le préfixe `"slow"` est conservé : **seules les observations slow sont candidates**. Les fast, produites une fois par frame par nature, passent sans test — évite un coût de hachage inutile sur le flux majoritaire.

**Horloge de purge** : `time.monotonic()`, base locale au filtre. Ne mesure aucune latence métier, donc **hors du périmètre de D7**. À documenter en commentaire pour éviter qu'un lecteur futur y voie une violation.

**Purge amortie** : inchangée (`PURGE_EVERY = 256`), avec `src_dedup_purges` désormais compté.

### A-bis.5 — `dedup_ttl_s` — renommage

```txt
# AVANT : StreamConfig.dedup_window_s = 2.0
# APRÈS : StreamConfig.dedup_ttl_s    = 2.0

assert dedup_ttl_s > 0.0
```

> ⚠️ **C'est un RENOMMAGE, pas un ajout.** Le paramètre existe dans `config.py` et est lu par `batching.py::iter_observations`. La table de propagation doit le refléter.

**Rationale** — `window` est surchargé dans le module (`temporal_radius_*`, fenêtre de probation C1, fenêtre de lot). `ttl` décrit sans ambiguïté une durée de rétention de clé.

### A-bis.6 — Points de test

| #        | Test                                                               | Attendu                                                   |
| -------- | ------------------------------------------------------------------ | --------------------------------------------------------- |
| **T-B1** | Instanciation d'une sous-classe d'`ObservationSource` sans `stats` | `TypeError` (ABC)                                         |
| **T-B2** | Grep `batching.py` + `observation.py`                              | Zéro occurrence de `builder`                              |
| **T-B3** | Non-régression corpus                                              | `src_dedup_skipped` identique à l'implémentation actuelle |
| **T-B4** | `JsonlSource` seule (sans `DedupSource`)                           | Aucune observation filtrée                                |
| **T-B5** | Entry avec `w = 0`                                                 | Retour `None`, `src_obs_degenerate_box == 1`              |
| **T-B6** | Somme de contrôle                                                  | `src_obs_read_total == src_obs_built + rejets`            |

### A-bis.7 — Table de propagation

| Fichier                 | Modification                                                                                               | Nature        |
| ----------------------- | ---------------------------------------------------------------------------------------------------------- | ------------- |
| `observation.py`        | Ajout `SourceStats`. `from_entry(entry, stats)`. Suppression du tuple de retour.                           | Structurel    |
| `batching.py`           | `ObservationSource` → ABC. `JsonlSource` sans dédup. Ajout `DedupSource`. Suppression `iter_observations`. | Structurel    |
| `config.py`             | **Renommage** `dedup_window_s` → `dedup_ttl_s` + assertion.                                                | 🟠 Renommage  |
| `counters.py`           | +9 clés `src_*`                                                                                            | Additif       |
| `stream_eid_builder.py` | Construction de la chaîne. Section « Qualité de l'entrée ».                                                | 🟠 Adaptation |
| `builder.py`            | Aucune (INV-B1 déjà satisfait)                                                                             | Néant         |

### A-bis.8 — Hors périmètre

- **`LiveSource`** — sa spécification (drain de queue, non-bloquant, backpressure, politique de perte) relève du **Bloc F**. Le contrat `ObservationSource` fixé ici lui suffit. Le Bloc F est nommé au sommaire pour que la référence croisée soit valide.
- **Suppression de la dédup à la source** — écartée : `bank.py` hors périmètre, corpus JSONL à préserver pour la calibration.
- **Contenu du rapport final** — **Bloc G**.

---

## 4. Doctrine d'association (Bloc B)

**Pourquoi** : A3-a = conjonctif absolu + relatif · A3-b = (2) `mask_uid` = signal de doute exposé · A3-c = borne d'empan **et** borne de dispersion.

> **Fichiers touchés** : `cluster.py`, `cost.py`, `config.py`, `counters.py`, `builder.py`.

### B.0 — Principe fondateur

**Règle structurelle.** Le lot d'association est la frame. Le chemin slow calcule dans B `geom_ok`, `cost_sig = clip(sd.cost, 0, 1)`, la marge ligne puis colonne, et `admit_costs`; la géométrie est un filtre et n'est pas intégrée au coût de signature. `w_sig` est supprimé du dataclass et l'assertion `config.py:58` est retirée en B ; `w_geo` reste présent mais n'est pas lu par B jusqu'au lot C4. Le seuil géométrique est `seuil_geo = w_pos`, avec rejet lorsque `d_pos = 1`.

```txt
Un cluster est un échantillon COURT et HOMOGÈNE, pas une trajectoire longue.
Face au doute, il refuse et se fragmente. Jamais il n'absorbe.
```

Application directe de **D1**.

**Invariants :**

| #          | Invariant                                                                                    |
| ---------- | -------------------------------------------------------------------------------------------- |
| **INV-C1** | Aucune association n'est validée sans franchir les **trois** critères de B.2.                |
| **INV-C2** | `mask_uid` n'a **aucun** effet sur la décision d'association.                                |
| **INV-C3** | Un cluster refusé produit systématiquement un nouveau cluster, jamais une absorption forcée. |

### B.1 — Ordre de priorité des signaux

```txt
signature  ≫  géométrie  >  uid (nul)
```

| Signal         | Rôle                                                                          | Poids                             |
| -------------- | ----------------------------------------------------------------------------- | --------------------------------- |
| **Signature**  | Critère primaire d'identité. Porte la décision.                               | Dominant                          |
| **Géométrie**  | Gating de plausibilité. Ne départage pas deux candidats proches en signature. | Secondaire, **gating uniquement** |
| **`mask_uid`** | Aucun rôle décisionnel. Observable exposée.                                   | **Nul**                           |

> ⚠️ **Conséquence non tranchée** : `builder.py::_eval_slow_pair` calcule aujourd'hui `cost = w_sig·d_sig + w_geo·d_geo`. Le Bloc B pose « géométrie = gating uniquement », ce qui **condamne implicitement `w_geo`**. → dette \*\*\*\*, question préalable de C4.

### B.2 — Critère d'admission conjonctif

Une observation `o` est admise dans le cluster `c*` **si et seulement si** les trois conditions sont vraies :

```python
# 1. GATING GÉOMÉTRIQUE — plausibilité spatio-temporelle (binaire, non scoré)
geom_ok = geometric_gate(o, c*)

# 2. SEUIL ABSOLU — qualité intrinsèque du match signature
absolute_ok = cost_sig(o, c*) <= cfg.cluster_admit_cost_max

# 3. SEUIL RELATIF — marge sur le second meilleur candidat
relative_ok = (cost_second - cost_best) >= cfg.cluster_admit_margin_min
```

**Rationale du conjonctif** :

- Le **seuil absolu** seul laisse passer un match médiocre en l'absence de concurrent.
- Le **seuil relatif** seul laisse passer deux mauvais candidats éloignés l'un de l'autre.
- Leur conjonction exige un match _bon_ **et** _non ambigu_ — la seule configuration où D1 autorise l'absorption.

**Cas dégénéré** : s'il n'existe qu'un seul candidat, `cost_second = +∞` et `relative_ok` est vrai par construction. Le seuil absolu reste seul juge — c'est cohérent : sans concurrence, il n'y a pas d'ambiguïté à trancher.

### B.3 — `mask_uid` : signal de doute exposé

`mask_uid` ne participe pas à l'association, mais un **désaccord** est une information légitime pour AuthenticsMask.

| Compteur / champ       | Sémantique                                                               |
| ---------------------- | ------------------------------------------------------------------------ |
| `uid_agree`            | L'observation admise porte le même `mask_uid` que la majorité du cluster |
| `uid_disagree`         | Désaccord — flag posé sur l'observation, compteur incrémenté             |
| `cluster.n_uid_switch` | Nombre de transitions de `mask_uid` majoritaire dans le cluster          |

Un `n_uid_switch` élevé sur un cluster par ailleurs homogène en signature est un **signal fort** : soit le tracker amont a swappé, soit deux entités sont réellement identiques visuellement. AuthenticsMask arbitre.

> **Note sur Q3** — cette position résout la circularité : `mask_uid` n'est jamais vérité terrain **dans EID**. Il devient une observable parmi d'autres, dont la fiabilité est jugée en aval.

### B.4 — Homogénéité interne : double borne

Un cluster cesse d'accepter des observations quand **l'une** des deux bornes est atteinte :

| Borne | Paramètre | Rationale
|
| ------------------ | ------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| **Empan temporel** | `cluster_max_span_s` | Au-delà, l'apparence a pu dériver légitimement (angle, éclairage). Un cluster long n'est plus un échantillon homogène. |
| **Dispersion** | `cluster_max_dispersion` | Mesurée par C2. Un cluster dont les observations s'écartent trop de sa référence est déjà en train de se contaminer. |

**Ordre d'évaluation** : la borne d'empan est vérifiée **avant** le calcul de coût (court-circuit gratuit). La borne de dispersion l'est **après** admission provisoire — on calcule la dispersion incluant la candidate, et on refuse si elle dépasse.

**Effet** : un cluster atteignant une borne est **finalisé**, pas figé. Les observations suivantes ouvrent un nouveau cluster. C'est exactement la fragmentation voulue par D1.

### B.5 — Compteurs

| Clé                            | Incrémenté quand                           |
| ------------------------------ | ------------------------------------------ |
| `admit_ok`                     | Les trois critères passent                 |
| `geo_reject_prefilter`         | Filtre géométrique refuse par paire        |
| `admit_reject_same_frame`      | Même frame slow par paire                  |
| `admit_reject_same_frame_fast` | Même frame fast par paire                  |
| `admit_reject_cost`            | Coût d'admission refuse par paire          |
| `admit_reject_margin`          | Marge ligne insuffisante par observation   |
| `admit_reject_margin_col`      | Marge colonne insuffisante par observation |
| `cluster_closed_span`          | Borne d'empan atteinte                     |
| `cluster_closed_dispersion`    | Borne de dispersion atteinte               |
| `uid_agree` / `uid_disagree`   | Cf. B.3                                    |

**Usage attendu** : un ratio `admit_reject_margin / admit_ok` élevé signale que les clusters actifs sont trop proches en signature — soit le seuil de marge est trop strict, soit la scène contient des entités réellement indiscernables (cas légitime en Rocket League : deux joueurs de même équipe).

### B.6 — Ce que le Bloc B ne tranche pas

| Question                                      | Renvoyée à                          |
| --------------------------------------------- | ----------------------------------- |
| Définition de `geometric_gate`                | **C4**                              |
| Définition de `cost_sig`                      | **C3**                              |
| Définition de `dispersion`                    | **C2**                              |
| Sort de `w_geo` dans `_eval_slow_pair`        | **C4**                              |
| Forme de `prime_footprint`                    | **Bloc D**                          |
| Fusion _a posteriori_ des clusters fragmentés | **Hors périmètre** (AuthenticsMask) |

---

## 5. BLOC C — Métrique et référence

### Référence de signature (C1)

L'ancre (`AnchorManager`) est définie et consommée dans `signature_ref.py`. Elle est un étalon de signature uniquement ; les lectures de position, dimensions ou temps pour le gating sont portées par `last_obs`/`last_seen_ts` et non par `cluster.anchor.anchor`.

**Pourquoi** : C1-a = (δ1+δ2) ancre figée + promotion unique + observables de dérive · C1-b = (q3) qualité intrinsèque avec garde-fou d'identité · C1-c = (f4) promotion anticipée sur ancre parfaite, sinon premier des deux seuils · C1-d = (e3) outliers conservés et signalés · INV-R6 posé ici.

> **Fichiers touchés** : `signature_ref.py`, `cluster.py`, `config.py`, `counters.py`.

#### C1.0 — Principe

```txt
La référence d'un cluster est TOUJOURS une observation réelle, jamais une moyenne.
Elle peut être remplacée UNE SEULE FOIS, dans une fenêtre de probation bornée,
par une observation MIEUX OBSERVÉE — jamais par une observation plus centrale.
Après verrouillage, elle est immuable et la dérive résiduelle est MESURÉE, pas absorbée.
```

**Justification structurelle** — trois faits de `core/signature.py` interdisent toute référence moyennée :

| Fait                                                                  | Source                   | Conséquence                                                        |
| --------------------------------------------------------------------- | ------------------------ | ------------------------------------------------------------------ |
| `sig_map` a une largeur `L` variable et un alignement par glissement  | `_fill_map`, `MAP_L_MAX` | Aucune moyenne point-à-point n'a de sens entre deux cartes.        |
| `sig_shape` encode les bandes vides par `SHAPE_EMPTY_SENTINEL = -1.0` | `SHAPE_EMPTY_SENTINEL`   | Moyenner produit des valeurs hors domaine, ni « vide » ni densité. |
| `sig_map_clip` encode la troncature comme bitmask                     | `MAP_CLIP_*`             | Moyenner deux cartes de `clip` différents crée des bords fantômes. |

Une référence-observation préserve nativement ces trois sémantiques.

#### C1.1 — Invariants

| #          | Invariant                                                                                                                                                                                               |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **INV-R1** | La référence est toujours une `Observation` réelle du cluster.                                                                                                                                          |
| **INV-R2** | Au plus **une** promotion par cluster.                                                                                                                                                                  |
| **INV-R3** | Après verrouillage, la référence est immuable.                                                                                                                                                          |
| **INV-R4** | La promotion se fait sur **qualité intrinsèque**, jamais sur centralité.                                                                                                                                |
| **INV-R5** | Aucune observation n'est retirée du cluster (application de **D4**).                                                                                                                                    |
| **INV-R6** | **L'ancre est un étalon de SIGNATURE. Elle ne porte aucune sémantique spatiale ni temporelle. Tout usage de `anchor.rect`, `anchor.cx`, `anchor.cy`, `anchor.ts` hors calcul de signature est un bug.** |

> **INV-R6 — violation connue** : `builder.py::_eval_slow_pair` lit `cluster.anchor.anchor` pour calculer `d_pos` et `d_dim`. Sur un objet mobile, la position de la meilleure vue n'a aucune valeur prédictive — à 500 px/s, 30 frames à 120 fps représentent 125 px de déplacement réel, et `d_pos` sature à 1.0 alors que l'association est correcte. **Fragmentation garantie dès que l'objet bouge.** → dette \*\*\*\*, correction en C4.
>
> Note : le gating temporel (`cost.py::_get_cluster_ts`) a **déjà** été corrigé vers `last_seen_ts` (cf. §0.2). Seule la géométrie reste à purger.

#### C1.2 — Machine à états

```txt
          création cluster
                 │
                 ▼
         ┌───────────────┐
         │  PROBATION    │  ← ancre initiale = 1ʳᵉ observation
         └───────┬───────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
  ancre      fin fenêtre   fin fenêtre
 parfaite    + candidate   sans candidate
 (f4)        meilleure         │
    │            │             │
    ▼            ▼             ▼
 PROMOTION   PROMOTION    PAS DE PROMOTION
    │            │             │
    └────────────┴─────────────┘
                 │
                 ▼
         ┌───────────────┐
         │   LOCKED      │  ← référence immuable
         └───────────────┘
```

#### C1.3 — Critère de qualité intrinsèque

```txt
quality(obs) = w_clip   · f_clip(sig_map_clip)
             + w_h      · f_h(sig_map_h_txt)
             + w_fill   · f_fill(sig_shape_fill)
             + w_hs     · f_hs(sig_hs_px)
```

| Composante | Source           | Forme                                                                                                                    |
| ---------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `f_clip`   | `sig_map_clip`   | `1.0` si `clip == 0`, sinon `0.5^popcount(clip & (LEFT\|RIGHT))` — les bords latéraux sont décisifs (fragment de plaque) |
| `f_h`      | `sig_map_h_txt`  | `min(1.0, h_txt / MAP_REF_H)` — au-delà de `MAP_REF_H`, pas de bonus                                                     |
| `f_fill`   | `sig_shape_fill` | Gaussienne centrée sur `fill_ref`, largeur `fill_tol` — ni trop vide, ni saturé                                          |
| `f_hs`     | `sig_hs_px[1]`   | `min(1.0, n_px_eroded / hs_px_ref)` — support du plan érodé                                                              |

> **Note de calibration** — `fill_ref`, `fill_tol` et `hs_px_ref` ne peuvent être fixés _a priori_ : ils dépendent de la distribution réelle des plaques Rocket League. Calibration sur corpus JSONL en observant `sig_shape_fill` et `sig_hs_px[1]` sur des crops manuellement validés. La somme `w_*` doit valoir `1.0` — validation `__post_init__`.

#### C1.4 — Garde-fou d'identité

Une candidate à la promotion, même de qualité supérieure, est **refusée** si elle est trop éloignée de l'ancre courante :

```python
if sig_distance(anchor, candidate).cost > cfg.promo_max_drift:
    counters.bump("promo_blocked_identity")
    return  # candidate rejetée
```

**Rationale** : sans ce garde-fou, une observation de haute qualité **mais d'une autre entité** (swap de tracker en début de cluster) deviendrait référence et contaminerait tout le cluster. C'est le cas d'échec le plus dommageable, et il est précisément celui que D1 interdit.

#### C1.5 — Fenêtre de probation

```txt
PROMOTION IMMÉDIATE si :
    sig_map_clip == 0  ET  sig_map_h_txt >= MAP_REF_H
    → ancre "parfaite", inutile d'attendre

SINON, fin de fenêtre au PREMIER des deux seuils :
    n_obs >= promo_window_n_obs      (ex. 8)
    OU
    span_s >= promo_window_span_s    (ex. 0.5)
```

**Rationale du « premier des deux »** : sur un flux dense, `n_obs` se remplit vite et la promotion est rapide. Sur un flux clairsemé (objet partiellement occlus), la borne temporelle évite qu'un cluster reste indéfiniment en probation — et donc que sa référence reste provisoire au moment où le Bloc B doit l'utiliser pour admettre.

#### C1.6 — Outliers post-promotion

Après verrouillage, les observations déjà admises sont réévaluées contre la référence finale. Celles qui ne passeraient pas `cluster_admit_cost_max` sont **flaguées**, jamais retirées.

```txt
obs.post_promo_outlier: bool
cluster.n_post_promo_outliers: int
```

**Application de D4 et INV-R5.** Le compteur et les flags partent dans `prime_footprint` (Bloc D). AuthenticsMask dispose alors de l'information : _« ce cluster contient N observations qui ne seraient pas passées avec la référence finale »_ — signal de doute exploitable, pas donnée effacée.

> **Arbitrage écarté** : (e2) éjection vers un nouveau cluster. Exigerait une réinjection dans le flux d'admission, donc une boucle de rétroaction dans `cluster.py`, pour un cas rare. Rapport complexité/bénéfice défavorable.

#### C1.7 — Observables de dérive

Maintenus **en permanence**, dans les deux états, **sans aucun effet sur l'admission**.

| Observable              | Type  | Sémantique                                                            |
| ----------------------- | ----- | --------------------------------------------------------------------- |
| `drift_last`            | float | `cost_to_ref` de la dernière observation admise                       |
| `drift_max`             | float | Maximum observé sur la vie du cluster                                 |
| `drift_trend`           | float | Pente linéaire sur les `k` derniers `cost_to_ref` (`deque(maxlen=k)`) |
| `anchor_promoted`       | bool  | Trace de la promotion                                                 |
| `n_post_promo_outliers` | int   | Incohérence résiduelle                                                |

Une `drift_trend` positive et soutenue sur un cluster par ailleurs valide signale une **dérive d'apparence légitime** (rotation, éloignement) — AuthenticsMask peut décider de le scinder ou de le conserver selon son horizon.

#### C1.8 — Compteurs

| Compteur                 | Incrémenté quand                         |
| ------------------------ | ---------------------------------------- |
| `ref_promotions`         | Une promotion aboutit                    |
| `ref_promo_early`        | Promotion anticipée (ancre parfaite, f4) |
| `ref_promo_window`       | Promotion en fin de fenêtre              |
| `promo_blocked_identity` | Garde-fou d'identité refuse              |
| `promo_blocked_quality`  | Gain de qualité insuffisant              |
| `ref_locked_no_promo`    | Verrouillage sans promotion              |
| `post_promo_outliers`    | Cumul des observations flaguées          |

**Usage attendu** — un ratio `ref_promotions / n_clusters_created` très élevé (> 0.7) signalerait que les ancres initiales sont systématiquement mauvaises : il faudrait alors interroger le **moment de création** du cluster en amont plutôt que le mécanisme de promotion. Un `promo_blocked_identity` non négligeable est au contraire un **bon signe** — le garde-fou travaille, donc le pipeline amont produit bien des switches UID que le Bloc B est censé absorber par fragmentation.

#### C1.9 — Ce que C1 ne tranche pas

| Question                    | Renvoyée à |
| --------------------------- | ---------- |
| Définition de `cost_sig`    | **C3**     |
| Définition de `dispersion`  | **C2**     |
| Gating géométrique          | **C4**     |
| Signatures partielles       | **C5**     |
| Champs de `prime_footprint` | **Bloc D** |

---

### Dispersion (C2)

**Pourquoi** : C2-a = quantile · C2-b = (o3) les deux valeurs exposées · C2-c = (t3) repli sur max · C2-d = (u2) par descripteur + fusionnée · **= (c2)** deque parallèle borné · **= (d3)** dispersion d'admission **et** finale.

> **Fichiers touchés** : `cluster.py`, `config.py`, `counters.py`.
> **Statut** : figé. Consomme `SigDistance` (C3). Alimente B.4 (borne d'homogénéité) et le Bloc D (contrat de sortie).

---

#### C2.0 — Définition

```python
cost_to_ref[i] = sig_distance(ref, obs_i).cost      # pour chaque obs NON gatée
```

Les observations **gatées** ne contribuent pas à la dispersion — leur `cost` vaut `None` par invariant C3 (INV-D3). Elles sont comptabilisées séparément (§ C2.5).

---

#### C2.1 — Stockage des coûts

```txt
ClusterSlow:
    observations: Deque[Observation] = deque(maxlen=CLUSTER_OBS_MAXLEN)
    admit_costs:  Deque[float]       = deque(maxlen=CLUSTER_OBS_MAXLEN)   # ← même maxlen
```

**Invariant testé** : `len(admit_costs) == len(observations)` après tout `add_observation`.

**Rationale** — `maxlen` identique garantit l'alignement index-à-index et évite qu'une dispersion soit calculée sur une population décalée de sa population d'observations. Le surcoût est d'un `float` par observation : négligeable.

**Application de D5** — la borne est structurelle, pas une politique de purge.

> **Arbitrage écarté — (c3) stocker le coût dans `Observation`.** Rendrait `Observation` mutable après construction, ce qui casse sa nature de valeur immuable traversant source → batching → builder → cluster. Une mutation en fin de chaîne serait invisible aux étages amont et interdirait de rejouer une observation contre plusieurs clusters candidats sans effet de bord.
>
> **Arbitrage écarté — (c1) deque indépendant non aligné.** L'absence de garantie d'alignement rendrait impossible le recalcul de `dispersion_excl_outliers` (§ C2.4), qui exige de croiser coût et flag d'observation.

---

#### C2.2 — Deux populations

| Valeur             | Population                                                                                  | Moment du calcul                | Rôle                                               |
| ------------------ | ------------------------------------------------------------------------------------------- | ------------------------------- | -------------------------------------------------- |
| `dispersion_admit` | `admit_costs` — coûts mesurés **au moment de l'admission**, contre la référence de l'époque | Incrémental, à chaque admission | **Opérationnel** — arme la borne B.4 en temps réel |
| `dispersion_final` | `cost_to_ref` **recalculés** contre la référence finale après verrouillage                  | À la finalisation du cluster    | **Normatif** — alimente `prime_footprint`          |

**Rationale de la dualité** :

1. `dispersion_admit` est la seule valeur **disponible pendant la vie du cluster**. B l’utilise pour armer la borne `cluster_max_dispersion` de B.4; C2 en assure la métrologie et l’exposition — on ne peut pas attendre la finalisation pour décider de refuser une observation.

2. `dispersion_final` porte sur une population **homogène** (un seul référentiel). C'est la seule valeur statistiquement propre, donc la seule à exposer à AuthenticsMask.

3. **Leur écart mesure exactement l'impact de la promotion C1.** Un écart systématiquement grand signale que la promotion déplace trop la référence, donc que `promo_max_drift` (C1.4) est trop permissif. C'est un signal de calibration qu'aucune des deux valeurs seule ne porte.

**Coût** — le thread EID est dédié et sans contrainte de performance (contexte ). Le recalcul de `n ≤ CLUSTER_OBS_MAXLEN` distances à la finalisation est gratuit.

```txt
delta_dispersion = dispersion_final - dispersion_admit
```

Exposé dans `prime_footprint`. **Non interprété par EID** (application de D2).

---

#### C2.3 — Agrégat : quantile

```python
# config.py
dispersion_quantile: float = 0.90

dispersion = float(np.quantile(costs, cfg.dispersion_quantile))
```

**Rationale** :

| Agrégat           | Défaut                                                                                                                                                                       |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Moyenne           | Masque les outliers. Un cluster contaminé par 2 observations sur 20 garde une moyenne acceptable.                                                                            |
| Max               | Dominé par une seule aberration. Un unique crop flou referme le cluster à tort.                                                                                              |
| **Quantile 0.90** | Capture « la queue haute sans l'extrême ». Répond précisément à _« à quel point les 10 % pires observations s'écartent-elles ? »_ — la question d'homogénéité posée par B.4. |

**Validation** : `assert 0.5 < dispersion_quantile < 1.0` dans `__post_init__`.

---

#### C2.4 — Deux vues : brute et hors outliers

| Valeur                     | Population                                                                   |
| -------------------------- | ---------------------------------------------------------------------------- |
| `dispersion`               | Quantile sur **toutes** les observations non gatées                          |
| `dispersion_excl_outliers` | Quantile après retrait des observations flaguées `post_promo_outlier` (C1.6) |

**Rationale** — application directe de **D2** : EID mesure, AuthenticsMask juge. Exposer uniquement la valeur nettoyée reviendrait à décider à la place du consommateur que les outliers sont du bruit ; exposer uniquement la brute masquerait la structure du cluster.

L'écart entre les deux répond à : _« ce cluster est-il globalement dispersé, ou homogène avec quelques intrus identifiés ? »_ — deux situations qui appellent des traitements aval opposés.

**Contrainte d'ordre** : `dispersion_excl_outliers` n'est calculable qu'**après** verrouillage de la référence (le
s flags `post_promo_outlier` n'existent pas avant). En état `PROBATION`, la valeur est `None`.

| État du cluster | `dispersion`                 | `dispersion_excl_outliers` |
| --------------- | ---------------------------- | -------------------------- |
| `PROBATION`     | Calculée (sur `admit_costs`) | `None`                     |
| `LOCKED`        | Calculée                     | Calculée                   |
| `FINALIZED`     | `dispersion_final`           | Calculée sur `cost_to_ref` |

---

#### C2.5 — Repli sur petit effectif

```python
if len(costs) < cfg.dispersion_n_min:      # ex. 4
    dispersion = float(max(costs)) if costs else 0.0
    counters.bump("dispersion_fallback_max")
```

**Rationale** — sous `n_min`, le quantile empirique n'a aucun sens statistique : `np.quantile([0.1, 0.5], 0.90)` interpole entre deux points et produit une valeur arbitraire.

Le repli sur `max` est **conservateur** (application de D1) : il surestime la dispersion, referme le cluster plus tôt, donc fragmente. C'est exactement l'erreur qu'on accepte de commettre.

**Cas `costs` vide** — cluster d'une seule observation, ou toutes gatées. `dispersion = 0.0` : un singleton est homogène par construction. La borne B.4 ne peut pas le refermer, ce qui est correct — c'est le critère d'admission (B.2) qui décide de sa croissance.

> **Arbitrages écartés** :
> — **(t1) quantile quel que soit `n`** : produit des valeurs non comparables entre clusters de tailles différentes, ce qui ruine la calibration du seuil `cluster_max_dispersion`.
> — **(t2) dispersion `None` sous `n_min`** : forcerait B.4 à gérer un cas nul, et laisserait un cluster de 2-3 observations croître sans contrôle d'homogénéité — précisément la phase où une contamination est la plus probable (peu de masse pour la diluer).

---

#### C2.6 — Dispersion par descripteur

En plus de la dispersion fusionnée, trois dispersions élémentaires sont maintenues :

```txt
dispersion_hs     = quantile([d.d_hs    for d in dists if d.d_hs    is not None])
dispersion_shape  = quantile([d.d_shape for d in dists if d.d_shape is not None])
dispersion_map    = quantile([d.d_map   for d in dists if d.d_map   is not None])
```

**Rationale** — c'est un **diagnostic de nature de dérive**, pas un critère de décision :

| Signature observée                              | Interprétation probable                                                       |
| ----------------------------------------------- | ----------------------------------------------------------------------------- |
| `dispersion_hs` élevée, `shape` et `map` basses | Variation d'éclairage ou de fond. Même entité, conditions changeantes.        |
| `dispersion_shape` élevée, `hs` basse           | Changement d'échelle ou d'angle. Même entité, géométrie apparente différente. |
| `dispersion_map` élevée seule                   | Texte différent → **très probablement deux entités**.                         |
| Les trois élevées                               | Cluster franchement contaminé.                                                |

Le dernier cas est le seul où AuthenticsMask devrait rejeter en bloc. Les trois autres appellent des traitements distincts. **Sans la décomposition, ces quatre situations produisent la même valeur fusionnée.**

**Règle** : ces trois valeurs sont **exposées, jamais lues par EID**. La borne B.4 porte exclusivement sur la dispersion fusionnée. Application stricte de D2.

**Populations disjointes** — chaque descripteur a sa propre population (les `None` diffèrent selon les paires). Les effectifs sont donc exposés à côté :

```txt
n_hs, n_shape, n_map    # effectifs par descripteur
```

Sans eux, une `dispersion_map` calculée sur 2 observations serait indistinguable d'une calculée sur 40.

---

#### C2.7 — Pondération par `confidence` — **écartée**

La `confidence` de `SigDistance` (C3.j) **ne pondère pas** la dispersion.

**Rationale** — pondérer reviendrait à fusionner _ce qu'on mesure_ et _à quel point on y croit_, en violation directe de **D3**. Une observation peu fiable mais très divergente doit compter pleinement dans la dispersion : c'est précisément le cas où D1 impose de refermer le cluster.

À la place, la `confidence` moyenne du cluster est exposée séparément :

```txt
confidence_mean = mean([d.confidence for d in dists])
confidence_min  = min([d.confidence for d in dists])
```

AuthenticsMask dispose alors de : _« ce cluster a une dispersion de 0.22, mesurée avec une confiance moyenne de 0.61 »_ — deux informations séparées, librement combinables en aval.

---

#### C2.8 — Paramètres

| Clé                      | Type    | Défaut | Description                                                       |
| ------------------------ | ------- | ------ | ----------------------------------------------------------------- |
| `dispersion_quantile`    | `float` | `0.90` | Quantile d'agrégation. `0.5 < q < 1.0`.                           |
| `dispersion_n_min`       | `int`   | `4`    | Effectif minimal pour le quantile. Sous ce seuil : repli `max`.   |
| `cluster_max_dispersion` | `float` | `0.35` | Borne d'homogénéité B.4. Lue sur `dispersion` (fusionnée, brute). |

> **Note de tuning** — `cluster_max_dispersion` interagit directement avec `cluster_admit_cost_max` (B.2). Si `cluster_max_dispersion < cluster_admit_cost_max`, la borne de dispersion se déclenche **avant** que le critère d'admission ne refuse — les clusters se ferment sur dispersion plutôt que sur refus d'admission. Les compteurs `cluster_closed_dispersion` vs `admit_reject_absolute` révèlent lequel domine. Un déséquilibre massif indique une incohérence de calibration entre les deux.

---

#### C2.9 — Compteurs

| Compteur                    | Incrémenté quand                        |
| --------------------------- | --------------------------------------- |
| `dispersion_fallback_max`   | Repli sur `max` faute d'effectif        |
| `dispersion_recomputed`     | Recalcul complet à la finalisation (d3) |
| `cluster_closed_dispersion` | _(déclaré en B.5)_ — borne atteinte     |

---

#### C2.10 — Ce que C2 ne tranche pas

| Question                                           | Renvoyée à                          |
| -------------------------------------------------- | ----------------------------------- |
| Définition de `sig_distance` et de ses composantes | **C3** _(clos)_                     |
| Champs exacts de `prime_footprint`                 | **Bloc D**                          |
| Politique de fusion des clusters fragmentés        | **Hors périmètre** (AuthenticsMask) |

---

### Distance de signature (C3)

La distance (`SigDistance`, `sig_distance` et les champs de distance) est définie dans `eid_distance.py`. Le schéma envoyé à AuthenticsMask expose la dispersion finale réelle dans l'empreinte ; le canal `CostMatrix.payload` reste typé `(sd, d_geo)`.

> **Encart — symboles vérifiés contre la codebase** :
>
> - Présents et conformes : `eid_distance.compute_signature_distance(sig1, sig2, config)` ; `DEFAULT_WEIGHTS = {"hs": 0.4, "shape": 0.6}` ; `_effective_weights`, `_select_hs_plan`, `_compute_hs_distance`, `_compute_shape_distance`, `_compute_map_sliding_distance` ; `set_calib_probe` / `get_calib_probe` ; constantes `_HS_PLAN_RAW=0`, `_HS_PLAN_ERODED=1`, `_SHAPE_BANDS=16`, `_MAP_MIN_H_TXT=8`, `_SHAPE_FAMILY_W=(0.40, 0.40, 0.20)`.
> - Présents dans `signature.py` (`core/signature.py`) : `compute_signature`, `empty_signature`, `sig_get`, `serialize_signature`, `deserialize_signature` ; tuple `SIG_KEYS` de 15 clés ; constantes `HS_H_BINS=24`, `HS_S_BINS=8`, `SHAPE_DIM=48`, `MAP_REF_H=32`, `MAP_N=16`, `MAP_L_MAX=512`, `MAP_MIN_W=8`, `MAP_CLIP_LEFT/RIGHT/TOP/BOTTOM=1/2/4/8`.
> - **`_CALIB_PROBE`** : nom interne du module `eid_distance.py` ; la sonde de calibration est activée par `set_calib_probe(probe)` — le plan utilise le terme « probe » en C3.6, qui correspond bien à cette variable.
> - **`_MAP_MIN_H_TXT`** (≠ `map_h_txt_gate` évoqué dans C3.5) : la constante existante s’appelle `_MAP_MIN_H_TXT = 8` ; toute référence à `map_h_txt_gate` ou `map_h_txt_full` dans le plan est **NOM INTROUVABLE** dans `eid_distance.py` — à renommer ou à créer.

**Pourquoi** : C3-a = (h1) Bhattacharyya · C3-b = (p3) plan érodé conditionnel + pénalité · C3-c = (s1) masquage explicite + `d_occupancy` · C3-d = (w4) complémentarité · C3-d-bis = (k2) centrage · C3-d-ter = pondération `sy_k` par effectif · C3-d-quater = (z3) pénaliser + signaler · C3-e = (m4) glissement + rejet dur · C3-f = (c3) NCC · C3-g = (v2) tolérance ±1 ligne · C3-h = (iv-a) gate + coût, `d_hs` en gate seul · C3-j = (n4) coût + incertitude exposée · **= (g3)+(r1)** gate dur `h_txt` + pénalité de confiance · **= (p1)+(p3)** suppression `_CALIB_PROBE`, calibration via `CostMatrix` · **= (n2)** note A4 + histogramme · **= (s1)** symétrie à `rtol=1e-6`.

> **Fichiers touchés** : `eid_distance.py` (réécriture complète), `config.py`, `counters.py`, `cost.py`, `calibration.py`.
> **Statut** : figé. Lecture seule de `core/signature.py`. Consommé par C1 (`cost_to_ref`), C2 (dispersion), C4 (gating), Bloc D.

---

#### C3.0 — Contrat de la fonction

```python
@dataclass(frozen=True, slots=True)
class SigDistance:
    """Résultat d'une comparaison signature ↔ signature.

    INVARIANT : cost est None SI ET SEULEMENT SI gated is True.
    """
    cost:        Optional[float]   # [0,1] ou None si gaté
    gated:       bool
    gate_reason: Optional[str]     # "h_txt" | "l_ratio" | "no_descriptor" | "hs_divergence"
    confidence:  float             # [0,1] — fiabilité de la MESURE, jamais fusionnée au cost

    # Composantes élémentaires — None si le descripteur est indisponible
    d_hs:        Optional[float]
    d_shape:     Optional[float]
    d_map:       Optional[float]
    d_occupancy: Optional[float]

    # Diagnostic
    hs_plan:     Optional[str]     # "raw" | "eroded"
    map_offset:  Optional[int]     # décalage retenu par le glissement
    l_ratio:     Optional[float]
    h_txt_min:   Optional[int]
    n_bands_valid: Optional[int]
```

```python
def sig_distance(sig_ref: dict, sig_obs: dict, cfg: StreamConfig) -> SigDistance
```

**Invariants :**

| #          | Invariant                                                                                              |
| ---------- | ------------------------------------------------------------------------------------------------------ |
| **INV-D1** | Pureté : aucun effet de bord, aucun état global, aucune écriture de compteur depuis `eid_distance.py`. |
| **INV-D2** | Symétrie : `sig_distance(a, b).cost ≈ sig_distance(b, a).cost` à `rtol=1e-6`.                          |
| **INV-D3** | `cost is None` ⟺ `gated is True`.                                                                      |
| **INV-D4** | `cost` et `confidence` ne fusionnent jamais (**D3**).                                                  |

> **INV-D1 — conséquence** : les compteurs sont incrémentés par l'**appelant** (`cost.py`, `cluster.py`) sur lecture des champs de `SigDistance`. C'est ce qui rend la fonction testable en isolation et rejouable sur corpus.

---

#### C3.1 — Les trois distances élémentaires

##### `d_hs` — histogramme Hue×Saturation

```python
d_hs = sqrt(1.0 - sum(sqrt(h1 * h2)))      # Bhattacharyya
```

**Rationale (h1)** — les deux plans de `sig_hs` sont L1-normalisés par construction (`_hs_hist_from_mask`). Bhattacharyya est la métrique naturelle entre distributions de probabilité : bornée `[0,1]`, symétrique, sans paramètre. Chi-2 aurait exigé un epsilon anti-division-par-zéro sur un histogramme creux (24×8 = 192 cases, la plupart nulles).

**Sélection du plan (p3)** — plan érodé conditionnel :

```txt
si sig_hs_px[1] >= hs_px_min POUR LES DEUX signatures :
    plan = "eroded"     # fond pur, moins pollué par les bords
sinon :
    plan = "raw"
    confidence *= hs_raw_penalty      # ex. 0.85
```

Le plan érodé isole mieux le fond du cartouche, mais peut être vidé sur un crop serré. Le repli sur brut est légitime, la pénalité de confiance signale la dégradation sans la masquer.

##### `d_shape` — vecteur (48,) à sentinelle

**Masquage explicite** — `SHAPE_EMPTY_SENTINEL = -1.0` n'est **jamais** traité comme une valeur :

```txt
valid = (v1 != SENTINEL) & (v2 != SENTINEL)      # bandes valides des DEUX côtés
```

**Trois familles** :

| Famille | Contenu                        | Distance                                    |
| ------- | ------------------------------ | ------------------------------------------- |
| `d_k`   | Densité par bande              | Moyenne des écarts absolus sur `valid`      |
| `c_k`   | Centre de masse par bande      | Moyenne des écarts absolus **centrés** (k2) |
| `sy_k`  | Dispersion verticale par bande | Moyenne pondérée **par effectif** de bande  |

**Centrage de `c_k`** — le centre de masse absolu encode la position dans le crop, qui dépend du cadrage. On soustrait la moyenne des `c_k` valides de chaque côté avant comparaison : on compare la **forme du profil**, pas sa position.

**Pondération de `sy_k`** — une bande quasi vide a une dispersion verticale bruitée. Le poids de chaque bande dans la moyenne est proportionnel à `min(d_k1, d_k2)`.

**Pondération inter-familles** — poids issus de `StreamConfig`, **plus de magic number** :

```txt
shape_w_d: float = 0.45      # densité — le signal principal
shape_w_c: float = 0.35      # profil centré — complémentaire
shape_w_y: float = 0.20      # dispersion — le plus bruité
assert abs(shape_w_d + shape_w_c + shape_w_y - 1.0) < 1e-6
```

**Rationale (w4)** — la complémentarité prime : deux plaques de même densité mais de profils différents doivent diverger. Les trois familles mesurent des aspects orthogonaux, d'où trois poids non nuls plutôt qu'une famille dominante.

##### `d_occupancy` — désaccord de support

```txt
d_occupancy = |bandes_valides(sig1) XOR bandes_valides(sig2)| / 48
```

> **Note A4** — La « pénalité de support » proposée par v3 (P2-3/A4) **est implémentée**, mais sous une forme différente : elle n'est **pas** un terme additif dans `d_shape`. Elle devient `d_occupancy`, un descripteur **autonome et exposé**, promu en **gate** au-delà d'un seuil.
>
> **Rationale du changement de forme** : injecter une pénalité additive dans `d_shape` la rendait indistinguable d'une divergence de densité, en violation de D2 (agrégat opinioné). Un désaccord de support massif — « l'une des plaques a du texte là où l'autre est vide » — n'est pas une nuance de distance : c'est une **incompatibilité structurelle**. Le traiter comme gate est plus honnête et plus discriminant.
>
> **Instrumentation obligatoire** : la distribution de `d_occupancy` sur corpus est exportée sous forme d'**histogramme** (32 bins sur `[0,1]`) dans le rapport final. Sans elle, `occupancy_gate_max` ne peut pas être calibré. Cet histogramme est produit par `calibration.py`, pas par `eid_distance.py` (INV-D1).

##### `d_map` — carte (16, L) à glissement

**Rejet dur préalable (m4)** :

```txt
l_ratio = min(L1, L2) / max(L1, L2)
si l_ratio < cfg.map_l_ratio_min :      # ex. 0.60
    → gate, reason = "l_ratio"
```

Deux cartes de largeurs très différentes ne représentent pas le même texte. Le glissement les ferait « matcher » sur un sous-ensemble, produisant un faux positif structurel.

**NCC sur glissement (c3)** :

```python
res = cv2.matchTemplate(map_large, map_small, cv2.TM_CCOEFF_NORMED)
d_map = (1.0 - float(res.max())) * 0.5      # NCC ∈ [-1,1] → d ∈ [0,1]
map_offset = int(res.argmax())
```

**Rationale (c3)** — NCC normalisée est invariante à un facteur d'échelle de densité (contraste global du crop), là où SQDIFF ne l'est pas. `cv2.matchTemplate` vectorise la boucle Python (point O6 de v3, désormais gratuit).

**Tolérance verticale ±1 ligne (v2)** — le template glisse aussi de -1, 0, +1 ligne ; le meilleur des trois est retenu. Justification : `_fill_map` binarise à `MAP_TAU_H`, et une ligne de texte peut basculer d'un bin à l'autre entre deux crops. Sans tolérance, un décalage d'un pixel produit une divergence artificielle.

---

#### C3.2 — Gate dur sur `h_txt`

```python
h_txt_min = min(sig_ref["sig_map_h_txt"], sig_obs["sig_map_h_txt"])      # (r1)

if h_txt_min < cfg.map_h_txt_gate:          # 6
    → gate, reason = "h_txt"                # d_map non calculé
elif h_txt_min < cfg.map_h_txt_full:        # 12
    confidence *= _h_txt_penalty(h_txt_min) # rampe linéaire [gate, full] → [0.5, 1.0]
```

**Rationale (g3)** — le seuil `_MAP_MIN_H_TXT = 8` de la codebase actuelle est un **gate déguisé sans zone de transition** : à 7 lignes la carte est refusée, à 8 elle est acceptée à pleine confiance. Or la qualité d'une carte de 8 lignes est nettement inférieure à celle d'une carte de 32.

Le dispositif retenu sépare les deux questions :

| Zone             | Comportement                          | Question répondue                                                             |
| ---------------- | ------------------------------------- | ----------------------------------------------------------------------------- |
| `h_txt < 6`      | **Gate dur**                          | « Cette carte a-t-elle assez de lignes pour porter une information ? » — Non. |
| `6 ≤ h_txt < 12` | Coût calculé, **confiance pénalisée** | « J'ai une mesure, mais je n'y crois qu'à moitié. »                           |
| `h_txt ≥ 12`     | Coût calculé, confiance pleine        | Carte exploitable.                                                            |

**Rationale du seuil abaissé à 6** — `MAP_N = 16` et `MAP_REF_H = 32`. Une carte de 6 lignes sur 16 porte encore une structure verticale discriminante (le rapport 6/16 ≈ 0.38 laisse de la matière). Supprimer le gate entièrement (option g2) créerait un **faux positif structurel** : une carte de 2 lignes matche presque n'importe quoi par NCC sur une fenêtre courte.

**Rationale de (r1) `min(ref, obs)`** — la comparaison est aussi fiable que sa signature la plus dégradée. Utiliser `max` laisserait une carte excellente « tirer » une carte médiocre vers une fausse confiance.

---

#### C3.3 — Fusion

**Architecture à deux étages** : certaines composantes ne contribuent qu'au **gate**, d'autres au **coût**.

| Composante         | Gate                                            | Coût       |
| ------------------ | ----------------------------------------------- | ---------- |
| `d_hs`             | `d_hs > hs_gate_max` → gate `"hs_divergence"`   | ❌ **Non** |
| `d_occupancy`      | `> occupancy_gate_max` → gate `"no_descriptor"` | ❌ Non     |
| `h_txt`, `l_ratio` |                                                 | —          |
| `d_shape`          | ❌                                              |            |
| `d_map`            | ❌                                              |            |

```python
cost = (w_shape * d_shape + w_map * d_map) / (w_shape + w_map)   # sur composantes présentes
```

**Rationale de `d_hs` en gate seul** — la couleur de fond d'un cartouche Rocket League est **quasi constante** (fond sombre semi-transparent). Elle discrimine mal entre joueurs, mais un écart massif signale de façon fiable que l'un des deux crops n'est pas un cartouche (faux positif détecteur, reflet, particule). C'est donc un **excellent gate et un mauvais critère de coût** — l'inclure au coût diluerait le signal discriminant de `d_shape`/`d_map` avec du bruit.

**Renormalisation** — si `d_map` est absent (gaté ou `sig_map is None`), `cost = d_shape`. Si `d_shape` est absent, `cost = d_map`. Si les deux manquent → gate `"no_descriptor"`.

```txt
w_shape: float = 0.40
w_map:   float = 0.60
```

**Rationale des poids** — `sig_map` porte la structure du texte (le discriminant le plus fort entre pseudonymes), `sig_shape` porte la silhouette globale (plus robuste mais moins spécifique). Le déséquilibre reflète cette hiérarchie tout en gardant `d_shape` significatif quand `d_map` est dégradé.

---

#### C3.4 — Normalisation et `confidence`

```python
confidence = (
    conf_hs_plan            # 1.0 si "eroded", hs_raw_penalty si "raw"
  * conf_h_txt              # rampe C3.2
  * conf_coverage           # fraction de descripteurs présents
  * conf_bands              # n_bands_valid / 48
)
```

**Règle stricte (D3 / INV-D4)** : `confidence` ne modifie **jamais** `cost`.

**Usage aval** :

| Consommateur | Usage                                                                                                         |
| ------------ | ------------------------------------------------------------------------------------------------------------- |
| **C2**       | Exposée en `confidence_mean` / `confidence_min`, jamais en pondération (C2.7)                                 |
| **Bloc B**   | Un cluster dont la `confidence` moyenne est sous `conf_min_cluster` peut être refusé au titre du double seuil |
| **Bloc D**   | Exposée dans `prime_footprint`                                                                                |

Séparation nette entre _ce qu'on mesure_ et _à quel point on y croit_. Toute tentative de les fusionner en un score unique détruit l'information et rend le module inauditable.

---

#### C3.5 — Table de suppression

| Élément | Fichier | Justification
|
| ---------------------------------------------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `map_ambiguity_min` / `map_ambiguity_max` | `config.py` | Le Tier 2 conditionnel disparaît — `d_map` est calculé systématiquement quand la carte passe les gates. La bande d'ambiguïté `[0.20, 0.40]` chevauchait `c_max = 0.35`, ce qui encodait structurellement un **mécanisme de sauvetage** : une paire au-dessus de `c_max` pouvait redescendre sous le seuil grâce à la map. Exactement le risque anti-divergence identifié par I8. |
| `min(base_cost, dist_map)` | `eid_distance.py` | « La map ne peut que sauver » — asymétrie qui interdit à la map de **contredire**. Remplacé par la fusion pondérée C3.3. |
| `_is_valid_signature` | `eid_distance.py` | Garde binaire remplacée par le système de gates typés (`gate_reason`). |

| `_effective_weights` | `eid_distance.py` | Renormalisation ad hoc, absorbée par C3.3. |
| `DEFAULT_WEIGHTS` | `eid_distance.py` | Poids en dur. → `StreamConfig`. |
| `_SHAPE_FAMILY_W` | `eid_distance.py` | Magic number. → `shape_w_d/c/y`. |
| `_SHAPE_SY_SCALE` | `eid_distance.py` | Magic number de normalisation, rendu inutile par la pondération par effectif . |
| `_MAP_MIN_H_TXT = 8` | `eid_distance.py` | Remplacé par `map_h_txt_gate` / `map_h_txt_full` (C3.2). |
| `_CALIB_PROBE`, `set_calib_probe`, `get_calib_probe` | `eid_distance.py` | état global mutable, viole INV-D1. Jamais branché dans la codebase. |
| `sig_weights` (champ) | `config.py` | Remplacé par `w_shape` / `w_map` typés. |

---

#### C3.6 — Calibration

`_CALIB_PROBE` est supprimée. La collecte passe par le chemin existant :

```txt
sig_distance() → SigDistance (objet riche, pur)
      ↓
cost.py : construction de CostMatrix
      ↓
CostMatrix.calib_pairs : List[CalibRecord]
      ↓
CalibrationWriter.add_record()
```

**`CalibRecord` porte désormais le `SigDistance` complet** plutôt qu'un `dbg: dict` ad hoc :

```txt
CalibRecord:
    obs_uid, cluster_id, eid
    seq, frame_id, ts
    gap_s
    dist: SigDistance        # ← toutes les composantes, typées
    accepted: bool
```

**Rationale (p3)** — `CalibRecord` existe, est déjà sérialisé, porte déjà `seq`/`frame_id`/`ts`/`eid`. Le seul travail est de remplacer `dbg: dict` + `mode: str` par le `SigDistance` typé. `asdict()` sur une dataclass `frozen+slots` produit directement le JSON.

**Bénéfice** : la calibration n'a plus de chemin d'activation séparé. Elle est armée par `--dump-pairs`, comme aujourd'hui, et capture **toutes** les composantes sans code dédié dans `eid_distance.py`.

**Histogramme `d_occupancy`** — produit par `calibration.py` en agrégeant les `CalibRecord`, 32 bins sur `[0,1]`, exporté dans le manifeste de fin de run.

---

#### C3.7 — Paramètres

| Clé                                     | Type    | Défaut           | Description                                                       |
| --------------------------------------- | ------- | ---------------- | ----------------------------------------------------------------- |
| `hs_px_min`                             | `int`   | `64`             | Pixels minimum du plan érodé pour l'utiliser                      |
| `hs_raw_penalty`                        | `float` | `0.85`           | Multiplicateur de confiance sur repli plan brut                   |
| `hs_gate_max`                           | `float` | `0.55`           | Au-delà : gate `"hs_divergence"`                                  |
| `shape_w_d` / `shape_w_c` / `shape_w_y` | `float` | `0.45/0.35/0.20` | Poids inter-familles, somme = 1.0                                 |
| `occupancy_gate_max`                    | `float` | `0.35`           | Au-delà : gate `"no_descriptor"`. **À calibrer sur histogramme.** |
| `map_l_ratio_min`                       | `float` | `0.60`           | Sous ce ratio : gate `"l_ratio"`                                  |
| `map_h_txt_gate`                        | `int`   | `6`              | Gate dur                                                          |
| `map_h_txt_full`                        | `int`   | `12`             | Confiance pleine au-delà                                          |
| `map_row_tolerance`                     | `int`   | `1`              | Tolérance verticale ±1 ligne                                      |
| `w_shape` / `w_map`                     | `float` | `0.40/0.60`      | Poids de fusion, renormalisés sur composantes présentes           |

> **Note de tuning** — `occupancy_gate_max` et `hs_gate_max` sont les deux paramètres non calibrables _a priori_. Ils dépendent de la distribution réelle sur corpus. Valeurs de départ **délibérément permissives** (D1 s'applique à l'admission, pas au gate de mesure : un gate trop strict prive C2 de population et fausse la dispersion).

---

#### C3.8 — Compteurs (incrémentés par l'appelant, INV-D1)

| Clé                      | Incrémenté quand                                  |
| ------------------------ | ------------------------------------------------- |
| `sig_gate_h_txt`         | `gate_reason == "h_txt"`                          |
| `sig_gate_l_ratio`       | `gate_reason == "l_ratio"`                        |
| `sig_gate_occupancy`     | `gate_reason == "no_descriptor"` via occupancy    |
| `sig_gate_no_descriptor` | Les deux descripteurs absents                     |
| `sig_gate_hs_divergence` | `gate_reason == "hs_divergence"`                  |
| `sig_hs_plan_raw`        | Repli sur plan brut                               |
| `sig_map_absent`         | `d_map is None` sans gate (carte `None` en amont) |
| `sig_h_txt_penalized`    | Zone `[gate, full]`                               |

---

#### C3.9 — Invariants testables

| #        | Test                                         | Attendu                                   |
| -------- | -------------------------------------------- | ----------------------------------------- |
| **T-D1** | `sig_distance(a, a)`                         | `cost ≈ 0.0`, `gated is False`            |
| **T-D2** | `sig_distance(a, b)` vs `sig_distance(b, a)` | Égalité à `rtol=1e-6`                     |
| **T-D3** | Signature vide vs quelconque                 | `gated is True`, `cost is None`           |
| **T-D4** | Balayage corpus                              | `cost is None` ⟺ `gated is True` (INV-D3) |
| **T-D5** | `h_txt = 5`                                  | `gate_reason == "h_txt"`                  |
| **T-D6** | `h_txt = 9`                                  | `cost` calculé, `confidence < 1.0`        |
| **T-D7** | Grep `eid_distance.py`                       | Zéro `global`, zéro import de `Counters`  |

> **T-D2 — tolérance `rtol=1e-6` .** La symétrie stricte n'est pas atteignable : `cv2.matchTemplate` glisse la **petite** carte sur la **grande**, ce qui rend `d_map` structurellement asymétrique au niveau du float. De même, `deserialize_signature` renormalise `sig_hs` par plan après arrondi à 8 décimales, introduisant une dérive L1 de l'ordre de `1e-7`.
>
> **Arbitrage écarté — (s2) renormaliser systématiquement côté natif.** Modifierait `core/signature.py`, hors périmètre, et masquerait la dérive plutôt que de la borner.
> **Arbitrage écarté — (s3) symétrie stricte interdite.** Renoncer à l'invariant priverait le module de son test de non-régression le plus simple.
>
> **Dette liée** : la parité natif ↔ JSONL (point A9/N3-bis de v3) reste ouverte. `rtol=1e-6` la rend testable mais ne la démontre pas. → Bloc F.

---

#### C3.10 — Ce que C3 ne tranche pas

| Question                                                      | Renvoyée à |
| ------------------------------------------------------------- | ---------- |
| Gating géométrique, sort de `w_geo`                           | **C4**     |
| Traitement des signatures structurellement invalides en amont | **C5**     |
| Format d'export de `SigDistance` dans `prime_footprint`       | **Bloc D** |
| Calibration empirique des seuils (Q2)                         | **Ouvert** |

---

### Gating géométrique (C4)

#### Contexte

Le BLOC C4 formalise deux mécanismes distincts mais couplés :

1. Comment le gate géométrique évalue une candidate par rapport à un cluster
   slow, en tenant compte du temps écoulé depuis la dernière observation
   admise (C4-1).
2. Comment et quand un cluster (ou sa référence `last_obs`) doit être purgé
   ou voir son gate se désactiver lorsque ce temps devient trop grand (C4-2).

Ce bloc s'inscrit dans la hiérarchie e `signature > (géométrie ⊕
continuité fast) > uid` (cf. bloc sur le rôle distinctif slow/fast) : le
gate géométrique est un **filtre binaire exposé**, jamais fondu dans un
coût continu (D2), cohérent avec le principe D1 : ne jamais inventer un
signal (pas d'extrapolation directionnelle, pas de couplage non prouvé
entre grandeurs).

---

#### C4.1 — Référence spatiale et forme du gate géométrique

**Référence** : `last_obs` brut porté par le cluster slow — dernière
observation admise (position + taille), figée, sans extrapolation
directionnelle. Structure **distincte** de l'ancre de signature (`anchor`)
— INV-R6 inchangé : `anchor.rect/cx/cy/ts` reste réservé au calcul de
signature (C1), `last_obs` est une structure séparée dédiée au gate
géométrique.

**Tolérance** : croissante avec `Δt = t_candidate − last_obs.ts`, bornée.
Aucune dépendance à `vx`/`vy` du tracker amont — on élargit l'incertitude
admise, on n'invente jamais une position prédite (D1).

**Forme retenue (C4-1-a)** : linéaire bornée
tol(Δt) = min(tol_base + k · Δt, tol_max)

Nouveaux paramètres à calibrer (C5, hors périmètre ici) : `tol_base`, `k`,
`tol_max`.

**Grandeur tolérée (C4-1-b)** : gate scindé en deux tests indépendants,
non couplés à la même fonction de Δt :
Rayon sur le centre `(cx, cy)` : seuil = `tol(Δt)` — croît avec Δt
(la dérive de position croît avec le temps écoulé).
Tolérance de taille `(w, h)` : seuil fixe, indépendant de Δt (la
dérive de taille apparente dépend de la perspective/distance caméra,
pas du temps écoulé).

**Définition du gate** :
geometric_gate = pass ⟺ (dist_centre ≤ tol(Δt)) ET (dist_taille ≤ tol_taille_fixe)

Binaire, exposé — jamais fondu dans un coût (D2).

---

#### C4.2 — Purge des clusters et dégénérescence de `last_obs`

Deux objets distincts à considérer :

1. **`last_obs`** — sa fraîcheur conditionne `tol(Δt)` ; au-delà d'un
   certain Δt, le signal géométrique n'est plus informatif.
2. **Le cluster complet** (slow ou fast) — sa purge retire une entité
   entière de l'espace de recherche EID.

Axe A — Purge du cluster complet : (A2) + (A4)

TTL modulé par l'état du cluster, avec accélération en cas de doute exposé :
`ttl_locked` : ancre verrouillée — TTL
long, protège l'investissement de calibration.
`ttl_probation` : ancre non verrouillée — TTL court, élimination
rapide du bruit non confirmé.
Purge anticipée si flag de doute actif (`contaminated`/`ambiguous`,
cf. C1-d e3, A3-a γ) : `ttl_effectif = min(ttl_nominal, ttl_contaminated)`
— un cluster verrouillé mais flaggé est purgé plus vite qu'un cluster
verrouillé sain, jamais aussi long.

Règle : `now − last_obs.ts > ttl_effectif(état, flags)` ⟹ éviction
complète du cluster (slow ou fast).

**Axe B — Dégénérescence de `last_obs` seul (sans purger le cluster) : (B3)**

Au-delà de `Δt_max` (distinct de `tol_max` — `Δt_max` borne la _validité_
du signal, `tol_max` borne son _imprécision_ tant qu'il est valide) — le
gate géométrique sort du calcul : état **`n/a`**, explicitement typé et
exposé (ni pass, ni fail).

**Impact sur C4-0 (rouverture ciblée, non structurelle)** : le domaine de
`geometric_gate` s'élargit de `{pass, fail}` à **`{pass, fail, n/a}`**.
Le statut de gate binaire non fondu dans un coût (D2) est inchangé — seul
son domaine de valeurs gagne un troisième état. Le consommateur
(`_eval_slow_pair`) traite `n/a` comme "signal non-votant" : la décision
retombe sur `signature > continuité fast > uid` (ou signature seule si le
fast est indisponible).

**Invariant de cohérence (à vérifier en C5, non bloquant ici)** :
Δt_max (validité du gate géométrique) ≤ ttl_probation (purge du cluster non verrouillé)

Sinon un cluster encore vivant porterait un gate géométrique déjà dégénéré
en routine — signe de mauvais calibrage plutôt qu'un cas limite légitime.

---

#### Paramètres introduits par C4 (nommage définitif à confirmer en C5)

| Paramètre          | Rôle                                                |
| ------------------ | --------------------------------------------------- |
| `tol_base`         | Tolérance géométrique plancher (Δt = 0)             |
| `k`                | Pente de croissance de la tolérance avec Δt         |
| `tol_max`          | Plafond de tolérance géométrique                    |
| `tol_taille_fixe`  | Tolérance sur `(w, h)`, indépendante de Δt          |
| `ttl_locked`       | TTL du cluster à ancre verrouillée                  |
| `ttl_probation`    | TTL du cluster à ancre non verrouillée              |
| `ttl_contaminated` | TTL réduit en cas de flag de doute actif            |
| `Δt_max`           | Seuil de dégénérescence du gate géométrique (`n/a`) |

#### Décisions es

- Séparation stricte `last_obs` (gate géométrique) / `anchor` (signature) — INV-R6 préservé.
- Gate géométrique toujours binaire exposé (D2), désormais `{pass, fail, n/a}`.
- Aucune extrapolation directionnelle (D1) — tolérance croissante, jamais prédiction de position.
- TTL différencié par état + flags, cohérent avec C1-a/c/d et A3-a/c.
- `n/a` est un signal explicite d'absence, pas un pass ou fail déguisé .

---

### Signatures partielles (C5)

**Pourquoi** : C5-1 = admission différenciée par source · C5-2 =

> aucun nouveau champ.
> **Fichiers touchés** : `observation.py`, `builder.py`, `config.py`,
> `counters.py`.
> **Statut** : figé. Ne rouvre aucun mécanisme de C3 (gates par paire
> inchangés) — traite exclusivement le cas de l'observation _isolée_, en
> amont de toute comparaison.

---

#### C5.0 — Périmètre exact

C3 répond déjà, **par paire**, à toute dégradation partielle d'une
signature (`no_descriptor`, `h_txt`, `l_ratio`, `d_occupancy` — cf. C3.3).
C5 ne rouvre rien de cela. C5 traite uniquement ce que C3 ne peut pas voir :
une `Observation` dont la signature est **entièrement vide** (les 15 clés
de `SIG_KEYS` à `None`, cf. `EMPTY_SIG_DICT`), _avant_ toute mise en
présence d'un second terme de comparaison.

---

#### C5.1 — Admission différenciée par source

```txt
Une observation source="slow" à signature entièrement vide est REJETÉE
à la construction — jamais transmise à process_batch.

Une observation source="fast" à signature entièrement vide est ADMISE
sans restriction — elle ne participe qu'au gate géométrique (C4), rôle
pour lequel elle ne requiert structurellement aucune signature.
```

**Rationale** — cette asymétrie n'est pas un compromis, elle découle
directement de deux faits déjà s séparément dans le plan :

| Fait                                                                                                                                                                                    | Bloc | Conséquence pour C5-1                                                                                                                                                                                                                                                                                                         |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Le slow est seul porteur de l'identité — c'est l'unique flux qui alimente `_solve_slow_identity` (Hungarian sur `cost_sig`)                                                             | B.1  | Sans signature, un slow ne peut satisfaire **aucun** gate de C3 (`no_descriptor` systématique) → il ne peut jamais être admis dans un cluster existant, et ne peut jamais valider une candidature d'ancre (C1). Un tel slow ne peut que créer un cluster orphelin stérile, qui consommera un slot jusqu'à éviction TTL (C4-2) |
| sans avoir jamais rien apporté.                                                                                                                                                         |
| Le fast valide la **continuité de tracking**, gate binaire sur la trajectoire (`continuity_ok`), au même rang que la géométrie — rôle reformulé en comme indépendant de toute signature | C4 / | Un fast sans signature reste pleinement fonctionnel pour ce rôle : le gate géométrique de C4 (rayon sur centre + tolérance de taille) ne lit jamais `signature`. Le rejeter priverait `ClusterFast` de sa fonction native.                                                                                                    |

**Point d'implémentation** — le rejet slow a lieu dans `observation.py`,
au même niveau que les rejets déjà existants (`_REJECT_SOURCE`,
`_REJECT_DEGENERATE_BOX`) :

```txt
_parse_entry(entry):
    ...
    si source == "slow" ET signature entièrement vide (has_shape/has_hs/has_map tous False) :
        retourner None, _REJECT_EMPTY_SIG_SLOW, warn
    # aucune branche équivalente pour source == "fast"
```

Nouveau motif de rejet, nouveau compteur dédié (§ C5.4) — pas de
réutilisation détournée d'un compteur existant (D6, clés fermées).

**Ce que C5-1 ne fait PAS** :
Ne modifie aucun gate de C3 (paires toujours évaluées à l'identique).
Ne modifie pas `ClusterFast` ni `_process_fast_obs` (builder.py) — le
fast à signature vide traverse le pipeline fast exactement comme un
fast à signature complète, puisque ce pipeline ne consulte jamais
`signature` (confirmé par lecture de `builder.py` § continuité fast).
Ne crée aucun troisième état — c'est un rejet binaire à la lecture,
au même niveau que les rejets déjà s en A-bis.

---

```txt

```

**Rationale** — identifié mort dès §0.1 (S4 : garde binaire jamais
branchée, résidu partiel en C3). L'examen de C5-1 confirme qu'aucun
usage légitime ne subsiste :
Un mécanisme de "taux de couverture minimal à l'admission" ferait
doublon avec les gates par paire de C3 (`no_descriptor`, `h_txt`,
`l_ratio`), qui produisent déjà le même effet — une observation sans
descripteur exploitable ne passera jamais aucun gate.
L'ajouter comme second mécanisme, à un niveau différent (par
observation plutôt que par paire), créerait un risque de divergence
silencieuse entre deux seuils censés capturer la même réalité —
violation implicite de D2 (mesure unique, pas de doublon opinioné).

**Table de suppression (extension de C3.5)** :

| Élément | Fichier | Justification |
| ------- | ------- | ------------- |

**Vérification requise à l'implémentation** — §0.3 notait l'absence de

`config.py` fourni initialement ; le fragment ultérieur (`config.py`,

— la suppression est donc actionnable sans vérification supplémentaire.

---

#### C5.3 — Confidence d'une observation isolée : clarification documentaire

**Aucun nouveau champ, aucun nouveau mécanisme.** C5-3 clôt une ambiguïté
terminologique potentielle, rien de plus :

```txt
Il existe DEUX notions de confiance dans le système, non substituables,
jamais fusionnées (D3) :

  1. quality(obs)          — qualité INTRINSÈQUE d'une observation isolée.
                              Définie en C1.3, seul usage : arbitrer la
                              promotion d'ancre .
                              N'existe QUE pour une observation seule,
                              sans second terme de comparaison.

  2. SigDistance.confidence — confiance d'une COMPARAISON entre deux
                              signatures. Définie en C3.4. N'existe QUE
                              pour une paire, jamais pour une observation
                              seule.
```

**Rationale** — ces deux notions répondent à des questions différentes
("cette observation est-elle assez bonne pour devenir référence ?" vs
"à quel point je crois à cette mesure de distance ?") et ne doivent
jamais être confondues ni fusionnées en un troisième concept générique
de "confiance". Documenter explicitement leur périmètre respectif suffit
à prévenir toute tentative future de les unifier — ce qui violerait D3.

**Aucun impact fichier** — clarification pure du plan, aucune ligne de
code n'en découle directement.

---

#### C5.4 — Table de propagation

| Fichier          | Modification                                                                                            | Nature     |
| ---------------- | ------------------------------------------------------------------------------------------------------- | ---------- |
| `observation.py` | Ajout branche de rejet slow-signature-vide dans `_parse_entry`. Nouveau motif `_REJECT_EMPTY_SIG_SLOW`. | Structurel |
| `counters.py`    | +1 clé `obs_reject_empty_sig_slow` (déclaration fermée, D6).                                            | Additif    |

| `builder.py` | Aucune — le pipeline fast ne consulte jamais `signature` (confirmé). | Néant |
| `stream_eid_builder.py` | Section « Lecture » du rapport : ajout ligne rejet dédiée, cohérente avec le format existant (`obs_reject_source`, `obs_reject_degenerate_box`, ...). | Additif |

**Invariant testable** :

| #        | Test                                                      | Attendu                                             |
| -------- | --------------------------------------------------------- | --------------------------------------------------- |
| **T-E1** | `from_entry` sur un entry `source="slow"`, signature vide | `None`, motif `_REJECT_EMPTY_SIG_SLOW`              |
| **T-E2** | `from_entry` sur un entry `source="fast"`, signature vide | `Observation` valide, `signature == EMPTY_SIG_DICT` |

## 6. Contrat de sortie (Bloc D)

**Pourquoi** : D-1 = imbrication par bloc producteur · D-2 =

> `anchor_quality` remplace `anchor_confidence` · D-3 =
> suppression pure de `density_fps` · D-4 = dispersion
> exposée en 4 valeurs (admission + finale, ×2) · D-5 = compteurs bruts
> du gate géométrique, aucun ratio calculé · D-6 = `mask_uid` comme
> liste bornée + compteur de switches.
> **Fichiers touchés** : `cluster.py`, `signature_ref.py`, `config.py`,
> `counters.py`.

---

### D.0 — Principe fondateur

```txt
Le footprint est une PROJECTION, pas une INTERPRÉTATION.
Sa structure matérialise la provenance : chaque sous-dict correspond
à EXACTEMENT un bloc du plan. Un champ qui ne peut pas être rattaché
à un bloc producteur est un champ à ne pas exposer.
```

Application directe de D2/D3 : la séparation des responsabilités n'est
plus seulement une règle rédactionnelle du plan, elle devient une
propriété **structurelle et vérifiable** du JSON produit.

**Invariants :**

| #          | Invariant                                                                                                                  |
| ---------- | -------------------------------------------------------------------------------------------------------------------------- |
| **INV-E1** | Aucun champ du footprint n'est un agrégat opinioné (ratio, moyenne pondérée par confiance, score composite non documenté). |
| **INV-E2** | Tout score composite exposé l'est accompagné de ses composantes (D2).                                                      |
| **INV-E3** | `cost`/`quality` et `confidence` n'apparaissent jamais dans le même champ (D3, INV-D4).                                    |
| **INV-E4** | Chaque sous-dict de premier niveau porte le nom du bloc qui l'a spécifié — traçabilité plan → code.                        |

---

### D.1 — Structure générale

```python
prime_footprint = {
    "eid": int,
    "identity": { ... },     # ← C1 : ancre, verrouillage, dérive
    "dispersion": { ... },   # ← C2 : homogénéité du cluster
    "signal": { ... },       # ← C3 : confidence agrégée de paire
    "geometry": { ... },     # ← C4 : gate géométrique, compteurs bruts
    "tracking": { ... },     # ← A3-b/B.3 : mask_uid comme observable
    "lifecycle": { ... },    # ← commun : span, cardinalité, frames
}
```

**Pourquoi** — pour expliquer plus tard pourquoi `mask_uid_switches` existe
dans le footprint ?") remonte en une lecture au sous-dict `tracking`, donc
au bloc B du plan, sans recherche. C'est la forme la plus stricte parmi
les trois proposées, retenue précisément parce qu'elle rend visible dans
le JSON produit la discipline déjà imposée par D2/D3 au niveau du code.

---

### D.2 — `identity`

```python
"identity": {
    "anchor_signature": dict,        # serialize_signature(anchor.signature)
    "anchor_locked": bool,           # état LOCKED vs PROBATION (C1.2)
    "anchor_quality": float,         # quality(anchor) — C1.3, recalculé
    "anchor_quality_components": {   # INV-E2 — jamais de score nu
        "f_clip": float,
        "f_h": float,
        "f_fill": float,
        "f_hs": float,
    },
    "n_post_promo_outliers": int,    # C1.6 (e3) — flag exposé, jamais purgé
}
```

**Purge** — `anchor_confidence` disparaît intégralement. Le bug
identifié (`.get("sig_map_l", 1.0)`, fallback inatteignable, confiance
maximale sur signature dégradée) n'a plus de support : `anchor_quality`
est calculé par la fonction déjà spécifiée et validée en C1.3
(`_calculate_quality_score`, désormais exposée publiquement plutôt que
lue directement dans `signature_ref.py`).

**Conséquence sur `signature_ref.py`** — `_calculate_quality_score` doit
retourner ses composantes en plus du score agrégé, pour satisfaire
INV-E2 :

```python
@dataclass(frozen=True, slots=True)
class QualityScore:
    total: float
    f_clip: float
    f_h: float
    f_fill: float
    f_hs: float
```

Signature actuelle (`-> float`) devient `-> QualityScore`. Impact
propagé : tout appelant de `_calculate_quality_score` (uniquement
`AnchorManager.try_update` en interne) lit désormais `.total` pour la
comparaison, sans changement de logique de sélection.

**`anchor_locked`** — dérive directement de la machine à états C1.2
(`PROBATION` → `LOCKED`). Champ brut, booléen, aucune interprétation.

---

### D.3 — `dispersion`

```python
"dispersion": {
    "at_admission": float,               # dernière valeur connue en flux
    "at_admission_excl_outliers": float,
    "final": float,                      # recalculée sur tout l'historique
    "final_excl_outliers": float,
    "n_hs": int, "n_shape": int, "n_map": int,   # C2.4 — effectifs par descripteur
}
```

**Rationale des 4 valeurs** — (d3) a explicitement deux
populations de mesure distinctes :

| Valeur         | Origine                                                                                                                                         | Ce qu'elle capture                                                                   |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `at_admission` | Dernier `dispersion` calculé par B sur les observations du cluster au moment de la dernière admission; C2 le mesure et l’écrit dans l’empreinte | L'état du cluster **tel qu'il a fonctionné en flux**, avant tout traitement post-hoc |
| `final`        | Recalculée à `compile_prime_footprint()` sur l'historique complet, y compris post-promotion (C1.6)                                              | L'état du cluster **après verrouillage définitif de l'ancre**                        |

Un écart important entre `at_admission` et `final` est lui-même un
signal exploitable par AuthenticsMask (le cluster a dérivé après coup
suite au changement de référence en C1) — signal brut, non interprété
ici, cohérent avec D2.

**Suppression `density_fps`** — cf. D.6 (lifecycle), pas ici.

---

### D.4 — `signal`

```python
"signal": {
    "confidence_mean": float,   # C2.7 — moyenne des SigDistance.confidence
    "confidence_min": float,
    "n_gate_hs": int,           # C3.8 — comptage des gates déclenchés
    "n_gate_occupancy": int,
    "n_gate_h_txt": int,
    "n_gate_l_ratio": int,
}
```

**Rationale** — reprend exactement C2.7 (confidence jamais fusionnée à
`cost`, exposée séparément) et ajoute les compteurs de gate déjà
spécifiés en C3.8, jusqu'ici incrémentés dans `Counters` global mais
jamais consolidés **par cluster**. Cette consolidation par cluster est
nécessaire : un compteur global ne dit rien de la qualité d'un cluster
individuel, seul le sous-total du cluster est actionnable par
AuthenticsMask.

**Application d'INV-E1** — pas de ratio `n_gate_*/n_obs` calculé ici,
seuls les bruts. Cohérent avec D-5 (même discipline appliquée à C4).

---

### D.5 — `geometry`

```python
"geometry": {
    "n_geo_pass": int,
    "n_geo_fail": int,
    "n_geo_na": int,           # cf. C4 — Δt au-delà du seuil, gate n/a
    "contaminated": bool,      # C1-d/A3-c — flag de dégénérescence
}
```

**Rationale du choix strict** — aucun ratio calculé. Cohérence
directe avec la discipline déjà appliquée en C3 (`cost`/`confidence`
jamais fusionnés) et en C2.7 (`confidence` jamais utilisée en
pondération). Introduire `geo_fail_ratio` ici casserait cette
cohérence pour un gain marginal — le calcul est trivial côté
consommateur.

**`contaminated`** — flag déjà nommé en A3-c/C1-d (purge anticipée si
ce flag est exposé — cf. C4.2 Axe A4, TTL + condition de contamination).
Le champ existe donc déjà comme mécanisme interne ; D.5 le rend
visible dans le contrat de sortie.

---

### D.6 — `tracking`

```python
"tracking": {
    "mask_uids_seen": Deque[int],     # maxlen aligné sur D5 (ex. 32)
    "mask_uid_switches": int,          # cf. B.3 — n_uid_switch
    "uid_agree": int,
    "uid_disagree": int,
}
```

**Rationale** — matérialisation directe de l'arbitrage A3-b (« `mask_uid`
= signal de doute exposé, aucun rôle décisionnel »). `mask_uids_seen`
et `mask_uid_switches` étaient déjà spécifiés conceptuellement en B.3
(`cluster.n_uid_switch`) mais jamais formellement intégrés au contrat de
sortie — D.6 corrige cette omission.

**Borne (D5)** — `mask_uids_seen` est une `deque(maxlen=N)`, jamais une
`list`. `N` = paramètre `tracking_uid_history_max` (config), valeur de
départ 32 — largement suffisant : un cluster homogène en signature ne
devrait raisonnablement croiser qu'une poignée d'UID distincts avant
d'être finalisé par les bornes B.4 (empan ou dispersion).

**Conséquence sur `ClusterSlow`** — nouvel état à maintenir en cours de
vie (pas seulement à la finalisation), alimenté à chaque admission
(cf. B.3) :

```python
if obs.mask_uid not in self._uid_seen_set:
    self.mask_uids_seen.append(obs.mask_uid)
    self._uid_seen_set.add(obs.mask_uid)
if obs.mask_uid != self._last_majority_uid:
    self.mask_uid_switches += 1
    self._last_majority_uid = obs.mask_uid
```

---

### D.7 — `lifecycle`

```python
"lifecycle": {
    "n_obs": int,
    "n_slow": int,
    "n_fast": int,
    "span_frames": int,
    "span_time_s": float,
    "first_frame": int,
    "last_frame": int,
}
```

**Purge** — `density_fps` disparaît sans remplacement. `n_obs` et
`span_time_s` restent bruts et séparés ; toute division est de la
responsabilité du consommateur (D2).

**Ajout `n_slow`/`n_fast`** — déjà maintenus sur `ClusterSlow`
(`add_observation`, cf. fragment `cluster.py`), jamais exposés
jusqu'ici. Signal brut utile : un cluster à dominante fast avec peu de
`n_slow` a une identité moins assise qu'un cluster densément confirmé
par le slow.

---

### D.8 — Forme finale complète

```python
def compile_prime_footprint(self) -> dict:
    anchor = self.anchor.anchor
    quality = _calculate_quality_score(anchor) if anchor else QualityScore(0.0, 0.0, 0.0, 0.0, 0.0)
    disp_final = self._compute_dispersion(exclude_outliers=False)
    disp_final_excl = self._compute_dispersion(exclude_outliers=True)

    self.prime_footprint = {
        "eid": self.eid,
        "identity": {
            "anchor_signature": serialize_signature(anchor.signature if anchor else None),
            "anchor_locked": self.state_ref == RefState.LOCKED,
            "anchor_quality": quality.total,
            "anchor_quality_components": {
                "f_clip": quality.f_clip,
                "f_h": quality.f_h,
                "f_fill": quality.f_fill,
                "f_hs": quality.f_hs,
            },
            "n_post_promo_outliers": self.n_post_promo_outliers,
        },
        "dispersion": {
            "at_admission": self.dispersion_at_admission,
            "at_admission_excl_outliers": self.dispersion_at_admission_excl,
            "final": disp_final,
            "final_excl_outliers": disp_final_excl,
            "n_hs": self.n_hs, "n_shape": self.n_shape, "n_map": self.n_map,
        },
        "signal": {
            "confidence_mean": self.confidence_mean,
            "confidence_min": self.confidence_min,
            "n_gate_hs": self.n_gate_hs,
            "n_gate_occupancy": self.n_gate_occupancy,
            "n_gate_h_txt": self.n_gate_h_txt,
            "n_gate_l_ratio": self.n_gate_l_ratio,
        },
        "geometry": {
            "n_geo_pass": self.n_geo_pass,
            "n_geo_fail": self.n_geo_fail,
            "n_geo_na": self.n_geo_na,
            "contaminated": self.contaminated,
        },
        "tracking": {
            "mask_uids_seen": list(self.mask_uids_seen),
            "mask_uid_switches": self.mask_uid_switches,
            "uid_agree": self.uid_agree,
            "uid_disagree": self.uid_disagree,
        },
        "lifecycle": {
            "n_obs": self.n_obs_total,
            "n_slow": self.n_slow,
            "n_fast": self.n_fast,
            "span_frames": max(1, self.last_seen_frame - self.created_at_frame),
            "span_time_s": round(max(0.001, self.last_seen_ts - self.created_ts), 3),
            "first_frame": self.created_at_frame,
            "last_frame": self.last_seen_frame,
        },
    }
    self.state = ClusterState.FINALIZED
    return self.prime_footprint
```

---

### D.9 — Table de propagation

| Fichier            | Modification                                                                                                                                                                                                         | Nature     |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `signature_ref.py` | `_calculate_quality_score` retourne `QualityScore` (dataclass) au lieu de `float`. `AnchorManager.try_update` lit `.total`.                                                                                          | Structurel |
| `cluster.py`       | `compile_prime_footprint` réécrite intégralement (D.8). Nouveaux champs d'état maintenus en cours de vie : `mask_uids_seen`, `mask_uid_switches`, `n_gate_*`, `n_geo_*`, `contaminated`, `dispersion_at_admission*`. | Structurel |
| `config.py`        | Ajout `tracking_uid_history_max: int = 32`.                                                                                                                                                                          | Additif    |
| `counters.py`      | Aucun changement direct — les compteurs par cluster sont des attributs d'instance, pas des clés `Counters` globales.                                                                                                 | Néant      |

**Invariants testables** :

| #                                        | Test                                                                                                                    | Attendu                                                                                                                 |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **T-F1**                                 | Grep `prime_footprint`                                                                                                  | Aucune clé `density_fps`, aucune clé `anchor_confidence`                                                                |
| **T-F2**                                 | `compile_prime_footprint()` sur cluster à ancre None (jamais promue)                                                    | `anchor_quality == 0.0`, `anchor_signature == serialize_signature(None)`, aucune exception                              |
| **T-F3**                                 | `len(mask_uids_seen) <= tracking_uid_history_max` toujours                                                              | Structurel (deque)                                                                                                      |
| **T-F4**                                 | Somme de contrôle                                                                                                       | `n_obs == n_slow + n_fast`                                                                                              |
| **T-F5**                                 | Cluster jamais évalué en gate géométrique (100% fast orphelin sans slow)                                                | `n_geo_pass + n_geo_fail + n_geo_na == n_fast` (ou `n_obs - 1` selon convention d'ancrage à trancher en implémentation) |

---

## 7. Bornes & performance (Bloc E)

**Pourquoi** : E-1 = + TTL fait foi, compteur de vigilance seul

> ajout · E-2 = `n_min` par famille de descripteur · E-3 =
> `compile_prime_footprint()` est un snapshot à la demande, non finalisant
> · E-4 = aucune finalisation par empan — seul le TTL finalise.
> **Fichiers touchés** : `config.py`, `counters.py`, `cluster.py`.
> **Pourquoi** : le snapshot (`compile_prime_footprint`) ne doit jamais modifier l’état du cluster ; le point de finalisation unique est l’éviction TTL, déclenchée par `StreamEIDBuilder._evict_stale_slow(now_ts)` dans `builder.py` (la fonction `purge_if_expired()` évoquée par d’anciennes rédactions est **NOM INTROUVABLE** dans la codebase — c’est `_evict_stale_slow` qui joue ce rôle).

---

### E.0 — Principe fondateur

```txt
EID ne s'auto-limite jamais par une politique de capacité arbitraire.
La seule limite légitime est celle DÉJÀ décidée fonctionnellement
ailleurs (TTL de C4/A4). Toute borne supplémentaire est un THERMOMÈTRE
(observabilité), jamais un DISJONCTEUR (décision).
```

Corollaire direct de D2 étendu à la couche opérationnelle : si EID
commençait à évincer des clusters pour des raisons de capacité mémoire,
il **jugerait** (quel cluster sacrifier) — rôle explicitement réservé
à AuthenticsMask / au TTL fonctionnel, pas à une politique de LRU.

---

### E.1 — Cardinalité des clusters actifs

**Décision** : aucune borne dure sur `len(clusters_slow)` /
`len(clusters_fast)`. Le TTL (C4.2/A4) reste l'unique mécanisme de purge.
Un compteur de vigilance est ajouté, purement observationnel.

```python
# config.yaml
eid:
  perf:
    active_clusters_warn_threshold: 200   # seuil de vigilance, pas de plafond dur
```

```python
# counters.py — ajout
class Counters:
    ...
    def sample_active_clusters(self, n_slow: int, n_fast: int) -> None:
        """Appelé périodiquement (cf. E.3) — pure observabilité, aucune purge."""
        self.gauge("eid_active_clusters_slow", n_slow)
        self.gauge("eid_active_clusters_fast", n_fast)
        threshold = cfg.get("eid.perf.active_clusters_warn_threshold", 200)
        if n_slow + n_fast > threshold:
            self.incr("eid_active_clusters_over_threshold")
```

**Rationale contre** — une éviction LRU dupliquerait la
responsabilité du TTL déjà spécifié en C4/A4 avec un risque concret :
deux mécanismes de purge concurrents peuvent produire un ordre
d'éviction non déterministe (quel cluster meurt en premier si le TTL
et la LRU se déclenchent la même frame ?). Absence de tout incident
mémoire documenté dans la codebase actuelle → pas de justification à
introduire cette complexité maintenant. Si un incident réel survient
en production, redevient recevable en amendement futur — **pas
aujourd'hui, faute de preuve**.

**Invariant** :

| #          | Invariant                                                                                  |
| ---------- | ------------------------------------------------------------------------------------------ |
| **INV-E5** | `sample_active_clusters()` ne modifie jamais l'état des dicts de clusters — lecture seule. |

---

### E.2 — `n_min` par famille de descripteur

Consolidation de C2.2(u2) : trois seuils indépendants, cohérents avec
la séparation déjà e en C3.0/C2.4 (`n_hs`/`n_shape`/`n_map` exposés
séparément en D.3).

```python
# config.yaml
eid:
  dispersion:
    n_min_hs:    5
    n_min_shape: 5
    n_min_map:   8   # descripteur map plus coûteux à stabiliser → seuil plus haut par défaut
```

```python
# signature_ref.py — lecture au calcul de dispersion (rappel C2.2)
def _dispersion_for_family(costs: list[float], family: Literal["hs", "shape", "map"]) -> float:
    n_min = cfg.get(f"eid.dispersion.n_min_{family}", 5)
    if len(costs) < n_min:
        return max(costs) if costs else 0.0   # repli (u2), déjà clos en C2.2
    return _quantile(costs, cfg.get("eid.dispersion.quantile", 0.90))  #  quantile, C2.1
```

**Invariant** :

| #          | Invariant                                                                                                                                                              |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **INV-E6** | `n_min_map >= n_min_shape` et `n_min_map >= n_min_hs` par défaut (map = coût le plus élevé à stabiliser) — vérifié en config, pas en dur dans le code (reste tunable). |

---

### E.3 — `compile_prime_footprint` : snapshot à la demande

**Décision** : méthode publique, appelable à tout moment sur un cluster
vivant ou finalisé. Aucun appel automatique périodique côté EID.
Le consommateur (AuthenticsMask) décide seul de sa fréquence d'appel.

```python
# cluster.py
def compile_prime_footprint(self) -> dict:
    """
    Snapshot à la demande — PAS un événement de cycle de vie.
    Appelable sur un cluster ACTIVE (anchor_locked possiblement False,
    dispersion possiblement en repli n_min) ou ACTIVE : le snapshot est une lecture, il ne finalise pas.
    N'altère JAMAIS self.state — la transition vers `FINALIZED` est l’apanage exclusif de `StreamEIDBuilder._evict_stale_slow(now_ts)` (TTL écoulé).
    """
    return {
        "eid": self.eid,
        "identity": self._compile_identity_block(),     # C1
        "dispersion": self._compile_dispersion_block(),  # C2
        "signal": self._compile_signal_block(),          # C3
        "geometry": self._compile_geometry_block(),       # C4
        "tracking": self._compile_tracking_block(),        # A3-b/D.6
        "lifecycle": self._compile_lifecycle_block(),        # commun
    }
```

**Invariant** :

| #          | Invariant                                                                                                                                                                                               |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **INV-E7** | `compile_prime_footprint()` est **idempotent** et **sans effet de bord** sur `self.state` — appelable N fois de suite sans changer le résultat (hors évolution naturelle du cluster entre deux appels). |

---

### E.5 — Table de rapprochement des dettes soldées par ce bloc

| Dette / renvoi                                         | Origine         | Statut après E                                                                                                                                                                                                                           |
| ------------------------------------------------------ | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Cardinalité `clusters_slow`/`clusters_fast` non bornée | Constat initial | Assumé — TTL fait foi, compteur de vigilance ajouté (E.1)                                                                                                                                                                                |
| `n_min` de C2.2 non chiffré                            | C2.2            | Soldé — trois seuils distincts par famille (E.2)                                                                                                                                                                                         |
| Fréquence d'appel de `compile_prime_footprint` (D.10)  | D.10            | Soldé — snapshot libre, non finalisant (E.3)                                                                                                                                                                                             |
| Confusion snapshot/finalisation dans D.8               | D.8             | 🟠 **Amendement appliqué** — suppression du flip `self.state = ClusterState.FINALIZED` dans `compile_prime_footprint()` ; la finalisation est désormais exclusivement déclenchée par `StreamEIDBuilder._evict_stale_slow(now_ts)` (TTL). |

---

### 8. Mode flux (Bloc F)

`LiveSource` , parité natif ↔ JSONL , documentation des horloges , espaces d'identifiants (I1).

---

## Contrat du rapport final de `stream_eid_builder.py`

> **Encart — clés `Counters` vérifiées contre `counters.py`** :
>
> - Présentes dans `_KEYS` et consommées par `stream_eid_builder.main()` : `slow_obs`, `fast_obs`, `clusters_slow_created`, `clusters_fast_created`, `clusters_slow_finalized`, `evicted_fast`, `hungarian_batches`, `batches_read`, `batch_obs_cumul`, `gate_temporal_reject_slow`, `admit_reject_cost`, `shortcircuit_slow`, `cols_gate_only`, `cols_cost_rejected`, `fast_matched`, `gate_temporal_reject_fast`, `cost_reject_fast`, `track_adopted`, `obs_read_total`, `obs_reject_source`, `obs_reject_degenerate_box`, `obs_reject_malformed`, `obs_slow_ts_fallback`, `obs_sig_empty`.
> - **`obs_dedup_skipped`** : déclarée dans `_KEYS` mais **non imprimée** par la section « Lecture » du rapport ; si elle doit figurer, ajouter la ligne `log.info("  └─ Dédup sautées         : %d", stats["obs_dedup_skipped"])` dans `main()`.
> - **`batch_size_max`** : valeur suivie par `builder._batch_size_max` (attribut d’instance) mais **ABSENTE de `_KEYS`** — à ajouter dans `counters.py::_KEYS` si l’on veut l’exposer via `counters.stats()`.
> - **`obs_slow_ts_fallback_zero`** : déclarée dans `_KEYS` mais non distinguée de `obs_slow_ts_fallback` dans le rapport (la première semble être un doublon de la seconde — à élaguer dans `counters.py`).

Le rapport final de `stream_eid_builder.py` lit `Counters` via une liste de clés fermée. Toute clé absente produit une erreur explicite — il ne doit y avoir aucun échec silencieux. Le format d’affichage du rapport n’est pas spécifié.

Les clés effectivement consommées par le rapport dérivent des préfixes mentionnés ailleurs dans le plan : `src_*`, `geo_*`, `promo_*` et `live_*`. La clé `mode` est supprimée.

## Points ouverts

- **PP-D7 (2026-09-27)** : valider le coût B côte à côte avec le coût historique incluant `w_sig`/`w_geo`; sortie par taux d’assignations identiques ou divergences expliquées. La décision est actée : fermeture nominale sur slow (INV-A1), avec `epsilon_close_s` conservé comme timeout de sécurité dégradé; revoir sa valeur, actuellement réglée pour une sémantique de fenêtre.

- **Correspondance finale des préfixes `src_`, `geo_*`, `promo_*`, `live_*` avec `_KEYS`.**
  - Critère de déblocage : confronter chaque préfixe aux clés déclarées dans `counters.py` et lever toute clé non documentée.
- **Vérification du format final de `prime_footprint` après suppression du champ `mode`.**
  - Critère de déblocage : valider la structure attendue dans `cluster.py` et la consommation dans `stream_eid_builder.py`.

## Contrat de sortie et finalisation

**Règle normative** : `compile_prime_footprint()` est un snapshot à la demande, sans effet de bord sur `self.state`.

La transition vers `ClusterState.FINALIZED` est une opération distincte, réalisée exclusivement par `StreamEIDBuilder._evict_stale_slow(now_ts)` (`builder.py`) lorsque `now_ts - cluster.last_seen_ts > StreamConfig.temporal_radius_evict_s`. Deux opérations, deux responsabilités : la photographie est une lecture ; la finalisation est une purge. **À VÉRIFIER DANS LA CODEBASE** : `ClusterSlow.compile_prime_footprint()` modifie encore l’état en queue de méthode — la règle normative ci-dessus n’est pas encore appliquée dans le code (`cluster.py` v. actuelle) ; le retrait de cette ligne est un pré-requis du contrat de sortie.

## Tables de propagation

Chaque table ci-dessous est normative : elle rappelle les fichiers touchés et la nature de la modification.

### Régime de lot unique (Bloc A) — propagation

| Fichier       | Symbole(s) concerné(s)                                                                                                                                                                                                              | Nature     | Dépendance |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ---------- |
| `batching.py` | `BatchReader.__init__(source, config, builder, sliding)` ; `BatchReader.next_batch()` ; `BatchReader._max_span_s()` ; suppression du paramètre `sliding` ; suppression de `iter_observations` (remplacé par `JsonlSource.__iter__`) | Réécriture | Aucune     |
| `config.py`   | `StreamConfig.epsilon_close_s: float = 0.15` (`config.py:43`, timeout de sécurité; valeur à revoir car calibrée comme fenêtre), conservation du garde-fou; `batch_max_span_factor` et assertions à réaligner                        | Réécriture | A.2        |
| `counters.py` | Clés fermées `_KEYS` : `"batches_read"`, `"batch_obs_cumul"` (déjà déclarées) ; `"shortcircuit_slow"` (déjà déclarée)                                                                                                               | Additif    | A.4        |
| `builder.py`  | `StreamEIDBuilder.record_batch(batch)` ; `StreamEIDBuilder._batch_size_max` ; suppression du couplage à `BatchReader.batch_size_max`                                                                                                | Réécriture | B          |

### Contrat d’entrée (Bloc A-bis) — propagation

| Fichier          | Symbole(s) concerné(s)                                                                                                                                                                                                                                                                                                                                       | Nature                                                            | Dépendance |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------- | ---------- |
| `observation.py` | `Observation.from_entry(entry, counters=None)` (**NOM INTROUVABLE** dans la codebase — la signature réelle est `Observation.from_entry(entry: dict, counters: Optional[Counters] = None)`, le plan évoque `SourceStats` qui n'existe pas dans `observation.py`) ; `Observation.dedup_key` ; `Observation.seq` (assigné par `batching.py::iter_observations`) | Réécriture                                                        | A          |
| `batching.py`    | `ObservationSource` (classe abstraite) ; `JsonlSource.__init__(path, counters, config)` ; `JsonlSource.__iter__()` ; décorateur de dédup interne `last_keys: Dict[str, Dict[Tuple, float]]` ; `PURGE_EVERY = 256`                                                                                                                                            | Réécriture                                                        | A          |
| `config.py`      | `StreamConfig.dedup_window_s: float = 2.0` (existe, conservé sous ce nom — le renommage `dedup_ttl_s` proposé en A-bis.5 **n’a pas été appliqué dans la codebase**)                                                                                                                                                                                          | Aucun changement effectif (renommage déclaré mais non implémenté) | A          |
| `counters.py`    | Clés existantes : `src_obs_read_total`, `src_obs_reject_source`, `src_obs_reject_degenerate_box`, `src_obs_reject_malformed`, `src_obs_ts_fallback`, `src_obs_sig_empty`, `src_dedup_skipped` ; `"obs_reject_empty_sig_slow"` mentionné par le plan — **ABSENT DE LA CODEBASE — à créer** dans `_KEYS`                                                       | Additif (avec une clé à créer)                                    | A-bis.3    |

### Doctrine d’association (Bloc B) — propagation

| Fichier       | Symbole(s) concerné(s)                                                                                                                                                                                                                                                                                                  | Nature     | Dépendance |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ---------- |
| `builder.py`  | `StreamEIDBuilder._eval_slow_pair(obs, cluster)` ; `_solve_slow_identity(slow_obs)` ; `_process_slow_obs(obs, match)` ; `_process_fast_track(obs, identity_eid)` ; `_create_slow(obs)` ; `_update_slow(cid, obs)`                                                                                                       | Réécriture | A          |
| `cost.py`     | `build_slow_cost_matrix(config, batch, clusters_slow, cost_fn, want_calib=False)` ; `CostMatrix` (dataclass : `matrix`, `rows`, `col_ids`, `payload`, `n_gate_rejects`, `n_cost_rejects`, `n_shortcircuit`, `n_cols_gate_only`, `n_cols_cost_rejected`, `calib_pairs`) ; `BIG_COST_COST = 10.0`, `BIG_COST_GATE = 20.0` | Réécriture | C3         |
| `cluster.py`  | `ClusterSlow.add_observation(obs)` ; `ClusterSlow.anchor: AnchorManager` (champ dataclass) ; `ClusterFast.add_observation(obs)` (avec incréments `n_slow`/`n_fast`)                                                                                                                                                     | Réécriture | C1         |
| `counters.py` | `"hungarian_batches"`, `"gate_temporal_reject_slow"`, `"admit_reject_cost"`, `"shortcircuit_slow"`, `"gate_temporal_reject_fast"`, `"cost_reject_fast"`, `"fast_matched"`, `"track_adopted"`, `"anomaly_identity_track_mismatch"`, `"anomaly_fast_ambiguous"` (toutes clés déjà déclarées)                              | Additif    | A.4        |

### Référence de signature (C1) — propagation

| Fichier            | Symbole(s) concerné(s)                                                                                                                                                                                                                                                                                                    | Nature                              | Dépendance |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- | ---------- |
| `signature_ref.py` | `AnchorManager.__init__()` ; `AnchorManager.try_update(obs) -> bool` ; `AnchorManager.anchor` (property) ; `AnchorManager.best_obs`, `best_score`, `update_count` ; `_calculate_quality_score(obs) -> float` ; `dim_distance(obs_w, obs_h, ref_w, ref_h) -> float`                                                        | Réécriture                          | C3         |
| `cluster.py`       | `ClusterSlow.anchor: AnchorManager` ; `ClusterSlow.add_observation(obs)` qui appelle `self.anchor.try_update(obs)` ; `ClusterSlow.contaminated`, `mask_uids_seen`, `mask_uid_switches`, `n_gate_*`, `n_geo_*`, `dispersion_at_admission*` mentionnés par le plan — **ABSENTS DE LA CODEBASE — à créer** sur `ClusterSlow` | Réécriture (avec attributs à créer) | D          |
| `builder.py`       | `StreamEIDBuilder._create_slow(obs)` qui instancie `ClusterSlow(cluster_id, eid)` ; `_update_slow(cid, obs)`                                                                                                                                                                                                              | Réécriture                          | B          |
| `counters.py`      | Clés existantes : `"clusters_slow_created"`, `"clusters_fast_created"` ; clés `anchor_promoted`, `anchor_replaced`, `outlier_post_promotion` mentionnées par le plan — **ABSENTES DE LA CODEBASE — à créer** dans `_KEYS`                                                                                                 | Additif (avec clés à créer)         | C1         |

### Dispersion (C2) — propagation

| Fichier            | Symbole(s) concerné(s)                                                                                                                                                                                                                                    | Nature                             | Dépendance |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- | ---------- |
| `signature_ref.py` | `dim_distance(obs_w, obs_h, ref_w, ref_h)` (utilisée par `builder.py::_eval_slow_pair` pour `d_dim`) ; fonction d’agrégat quantile mentionnée par le plan — **ABSENTE DE LA CODEBASE — à créer** (signature `quantile_dispersion(costs, q=0.9) -> float`) | Réécriture (avec fonction à créer) | C3         |
| `eid_distance.py`  | `compute_signature_distance(sig1, sig2, config)` (consommée par `builder.py::_eval_slow_pair` pour `d_sig`) ; `set_calib_probe(probe)`, `get_calib_probe()`                                                                                               | Aucun changement                   | C3         |
| `cluster.py`       | attribut `cost_to_ref` listé dans `prime_footprint["dispersion"]` — **ABSENT DE LA CODEBASE — à créer** sur `ClusterSlow` ; `ClusterSlow.observations` (Deque source)                                                                                     | Additif (avec attribut à créer)    | D          |
| `config.py`        | paramètres `dispersion_quantile: float = 0.90`, `n_min_per_family: int = 3` mentionnés par le plan — **ABSENTS DE LA CODEBASE — à créer** dans `StreamConfig`                                                                                             | Additif (avec paramètres à créer)  | E.2        |
| `counters.py`      | clés `dispersion_quantile_*`, `n_min_fallback_*` — **ABSENTES DE LA CODEBASE — à créer** dans `_KEYS`                                                                                                                                                     | Additif                            | C2         |

### Distance de signature (C3) — propagation

| Fichier            | Symbole(s) concerné(s)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Nature                                 | Dépendance |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------- | ---------- |
| `signature.py`     | `compute_signature(template, rect, source, crop_color) -> dict` ; `empty_signature(source) -> dict` ; `sig_get(sig, key, default)` ; `serialize_signature(sig) -> dict` ; `deserialize_signature -> dict` ; `SIG_KEYS` (tuple de 15 clés : `sig_source`, `sig_hs`, `sig_hs_mode`, `sig_hs_px`, `sig_shape`, `sig_shape_mode`, `sig_shape_fill`, `sig_shape_cy`, `sig_map`, `sig_map_n`, `sig_map_l`, `sig_map_mode`, `sig_map_clip`, `sig_map_tau_h`, `sig_map_h_txt`) ; constantes `HS_H_BINS=24`, `HS_S_BINS=8`, `SHAPE_BANDS=16`, `SHAPE_DIM=48`, `MAP_REF_H=32`, `MAP_N=16`, `MAP_L_MAX=512`, `MAP_MIN_W=8`, `MIN_STD=8.0`, `MAP_CLIP_LEFT=1`, `MAP_CLIP_RIGHT=2`, `MAP_CLIP_TOP=4`, `MAP_CLIP_BOTTOM=8` | Lecture seule (consommé par C3)        | Aucune     |
| `signature_ref.py` | `sig_get` (ré-importé depuis `core.signature`) ; `AnchorManager.anchor` consomme `sig_map_clip`, `sig_map_l` via `_calculate_quality_score`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Lecture seule                          | C1         |
| `eid_distance.py`  | `compute_signature_distance(sig1, sig2, config) -> float` ; `_is_valid_signature(sig)` ; `_effective_weights(sig1, sig2, weights)` ; `_select_hs_plan(sig1, sig2)` ; `_compute_hs_distance(hs1, hs2, plan)` ; `_compute_shape_distance(shape1, shape2)` ; `_compute_map_sliding_distance(sig1, sig2)` ; `DEFAULT_WEIGHTS = {"hs": 0.4, "shape": 0.6}` ; `_HS_PLAN_RAW=0`, `_HS_PLAN_ERODED=1`, `_SHAPE_BANDS=16`, `_MAP_MIN_H_TXT=8`, `_SHAPE_FAMILY_W = (0.40, 0.40, 0.20)` ; `set_calib_probe(probe)`, `get_calib_probe()`                                                                                                                                                                                 | Aucun changement (module de référence) | Aucune     |
| `cost.py`          | `CalibRecord` (dataclass : `obs_uid`, `cluster_id`, `cost`, `dbg`, `mode`, `gap_s`, `accepted`, `seq`, `frame_id`, `ts`, `eid`) ; `CostMatrix.calib_pairs: List[CalibRecord]` ; `build_slow_cost_matrix(..., want_calib=False)`                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Aucun changement                       | B          |
| `calibration.py`   | `CalibrationWriter.__init__(path, session_id=None)` ; `CalibrationWriter.session_id` ; `CalibrationWriter._ensure_open()` ; `CalibrationWriter._write(row)` ; `CalibrationWriter.write_manifest(config, input_path, extra=None)` ; `CalibrationWriter.add_record(rec)` ; `CalibrationWriter.close()` ; `SCHEMA_VERSION = 1`                                                                                                                                                                                                                                                                                                                                                                                  | Aucun changement                       | Aucune     |
| `config.py`        | `StreamConfig.sig_weights: Optional[dict]` (consommé via `config.sig_weights or DEFAULT_WEIGHTS`) ; `StreamConfig.min_sig_coverage: float = 0.5` ; `StreamConfig.map_ambiguity_min: float = 0.20`, `map_ambiguity_max: float = 0.40` ; `StreamConfig.disable_shortcircuit: bool = False`                                                                                                                                                                                                                                                                                                                                                                                                                     | Aucun changement                       | Aucune     |

### Gating géométrique (C4) — propagation

| Fichier       | Symbole(s) concerné(s)                                                                                                                                                                                                                                                                               | Nature           | Dépendance |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- | ---------- |
| `builder.py`  | `StreamEIDBuilder._process_fast_track(obs, identity_eid)` (gating cinématique : `radius_px = cfg.max_travel_px_per_s * gap_s + cfg.fast_radius_floor_px`, puis gate temporel `gap_s > cfg.temporal_radius_fast_s`) ; `StreamEIDBuilder._evict_stale_fast(now_ts)` (purge temporelle des pistes fast) | Réécriture       | B          |
| `cluster.py`  | `ClusterFast.last_seen_ts` ; `ClusterFast.state` ; `ClusterState` enum (`ACTIVE`, `FINALIZED`)                                                                                                                                                                                                       | Aucun changement | D          |
| `config.py`   | `StreamConfig.max_travel_px_per_s: float = 25.0 * 120.0` ; `fast_radius_floor_px: float = 24.0` ; `temporal_radius_fast_s: float = 0.35` ; `fast_ambiguity_ratio: float = 0.75`                                                                                                                      | Aucun changement | A          |
| `counters.py` | `"gate_temporal_reject_fast"`, `"cost_reject_fast"`, `"anomaly_fast_ambiguous"`, `"evicted_fast"` (toutes déjà déclarées)                                                                                                                                                                            | Aucun changement | A.4        |

### Signatures partielles (C5) — propagation

| Fichier           | Symbole(s) concerné(s)                                                                                                                                                                                                                                                     | Nature                         | Dépendance |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ | ---------- |
| `observation.py`  | `Observation.signature: Dict[str, Any]` ; `Observation.has_shape()`, `Observation.has_map()` ; `_parse_entry(entry)` qui produit `EMPTY_SIG_DICT` (référencé via `core.signature.EMPTY_SIG_DICT`)                                                                          | Aucun changement               | A-bis      |
| `eid_distance.py` | `_is_valid_signature(sig)` (teste `sig_hs is not None or sig_shape is not None`) ; `_effective_weights(sig1, sig2, weights)` (renormalisation sur les descripteurs présents des deux côtés) ; `_compute_map_sliding_distance(sig1, sig2)` (rejet dur `_MAP_MIN_H_TXT = 8`) | Aucun changement               | C3         |
| `config.py`       | `StreamConfig.min_sig_coverage: float = 0.5` (**non consommé** par `eid_distance.py` dans la codebase actuelle — clé exposée mais seuillage effectif délégué à `_is_valid_signature`)                                                                                      | Aucun changement               | Aucune     |
| `counters.py`     | `"obs_sig_empty"` (déjà déclarée) ; `"obs_reject_empty_sig_slow"` **ABSENT DE LA CODEBASE — à créer** dans `_KEYS`                                                                                                                                                         | Additif (avec une clé à créer) | A-bis      |

### Contrat de sortie (Bloc D) — propagation

| Fichier                 | Symbole(s) concerné(s)                                                                                                                                                                                                                                                                                                                                                                                                                                        | Nature                                             | Dépendance |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- | ---------- |
| `cluster.py`            | `ClusterSlow.compile_prime_footprint() -> dict` (**régression** : la version actuelle de la méthode **flippe encore** `self.state = ClusterState.FINALIZED` en fin de fonction — la règle normative « snapshot non finalisant » n’est pas respectée dans la codebase) ; `ClusterSlow.last_obs`, `ClusterSlow.created_at_frame`, `ClusterSlow.last_seen_frame`, `ClusterSlow.n_obs_total` ; `ClusterFast.add_observation(obs)` (mise à jour `n_slow`/`n_fast`) | Réécriture (avec retrait impératif du flip d’état) | C1, E.3    |
| `signature_ref.py`      | `AnchorManager.anchor` (property consommé par `compile_prime_footprint`) ; `_calculate_quality_score(obs)` ; `QualityScore` (dataclass mentionnée D.9) — **ABSENT DE LA CODEBASE — à créer** ou renoncer et conserver `float`                                                                                                                                                                                                                                 | Additif (avec refactor à confirmer)                | C1         |
| `counters.py`           | `"clusters_slow_finalized"` (incrémenté par `builder.py::_evict_stale_slow`) ; `"clusters_slow_created"`, `"clusters_fast_created"`, `"evicted_fast"` (toutes déjà déclarées) ; aucune clé dédiée aux compteurs par cluster mentionnés dans D.9 (`n_gate_*`, `n_geo_*`) car ils sont attributs d’instance — cohérence conservée                                                                                                                               | Aucun changement                                   | A.4        |
| `stream_eid_builder.py` | `StreamEIDBuilder.__init__(..., on_cluster_finalized=None, ...)` ; `on_finalized(footprint)` (closure dans `main()`) ; `_safe_dumps(obj, label) -> str` ; `_find_ndarray(obj, path)` ; section « Rapport d’exécution » du `main()` (utilise `builder.counters.stats()` et `anomalies()`)                                                                                                                                                                      | Réécriture                                         | B          |

### Bornes & performance (Bloc E) — propagation

| Fichier            | Symbole(s) concerné(s)                                                                                                                                                                                                                                                                                                   | Nature                                             | Dépendance |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------- | ---------- |
| `cluster.py`       | `ClusterSlow.observations: Deque[Observation]` (`deque(maxlen=50)`) ; `ClusterFast.observations` (`deque(maxlen=100)`) ; `ClusterSlow.compile_prime_footprint()` (snapshot sans flip d’état — **à corriger**) ; `ClusterSlow.last_seen_ts`, `created_at_frame`                                                           | Réécriture (suppression impérative du flip d’état) | D          |
| `config.py`        | `StreamConfig.temporal_radius_slow_s: float = 1.0` ; `temporal_radius_evict_s: float = 2.0` ; `epsilon_close_s: float = 0.15` (timeout, valeur à revoir) ; `batch_max_span_factor: int = 4` ; `k_pos_norm: float = 3.0` ; `cluster_max_dispersion: float = ...` mentionné dans B.4 — **ABSENT DE LA CODEBASE — à créer** | Additif (avec un seuil à créer)                    | B.4        |
| `signature_ref.py` | `dim_distance(obs_w, obs_h, ref_w, ref_h)` ; `_calculate_quality_score(obs)` ; `n_min_per_family` mentionné en E.2 — exposé comme paramètre de `signature_ref.py` mais **n’est pas un symbole du module** (à matérialiser soit comme constante de module, soit comme paramètre de `compute_signature_distance`)          | Additif (avec constante à créer)                   | C2         |
| `counters.py`      | `"batches_read"`, `"batch_obs_cumul"`, `"clusters_slow_finalized"`, `"evicted_fast"` (toutes déjà déclarées) ; `"batch_size_max"` mentionné par le rapport — **ABSENT DE LA CODEBASE — à créer** (l’info existe comme attribut d’instance `builder._batch_size_max` mais n’est pas dans `_KEYS`)                         | Additif (avec une clé à créer)                     | A.4        |

### Mode flux (Bloc F) — propagation

| Fichier                 | Symbole(s) concerné(s)                                                                                                                                                                                                                                                                                                | Nature           | Dépendance |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- | ---------- |
| `stream_eid_builder.py` | `main() -> int` ; argparse (`--input`, `--output`, `--verbose`, `--dump-pairs`, `--no-shortcircuit`) ; `on_finalized(footprint)` (callback d’éviction) ; `out_footprints` (chemin `*_eid_footprints.jsonl`) ; `out_assign` (chemin `*_eid_assignments.jsonl`) ; rapport final (logs INFO sur `stats()`/`anomalies()`) | Réécriture       | B, D, E    |
| `batching.py`           | `JsonlSource(path, counters, config)` ; `BatchReader(source, config, builder=builder, sliding=False)` ; `BatchReader.next_batch() -> List[Observation]` ; `BatchReader._pending` (lookahead 1)                                                                                                                        | Aucun changement | A          |
| `builder.py`            | `StreamEIDBuilder.process_batch(batch) -> List[dict]` ; `_evict_stale_slow(now_ts)` (déclenche la finalisation TTL — c’est le point de purge effectif, **voir note de décorrélation snapshot et état terminal plus bas**) ; `_evict_stale_fast(now_ts)`                                                               | Aucun changement | B, C4, D   |
| `counters.py`           | `ANOMALY_PREFIX = "anomaly_"` ; `Counters.as_dict()` ; `Counters.anomalies()` ; `Counters.stats()` (méthodes consommées par `main()`)                                                                                                                                                                                 | Aucun changement | Aucune     |

## Annexe — checklist de pré-vol par fichier

- [ ] `signature.py` — vérifier construction et sérialisation.
- [ ] `stream_eid_builder.py` — vérifier rapport à clés fermées et sorties.
- [ ] `batching.py` — vérifier lots et déduplication.
- [ ] `builder.py` — vérifier association, purge et callbacks.
- [ ] `calibration.py` — vérifier écriture et fermeture.
- [ ] `cluster.py` — **touché par six blocs** (B, C1, C2, C4, D, E) ; relire contre les six simultanément.
- [ ] `config.py` — vérifier tous les paramètres et bornes.
- [ ] `cost.py` — vérifier matrice, timestamps et calibration.
- [ ] `counters.py` — vérifier `_KEYS` et erreurs de clé absente.
- [ ] `eid_distance.py` — vérifier distances, gates et sentinelles.
- [ ] `observation.py` — vérifier parsing et rejet d’entrée.
- [ ] `signature_ref.py` — vérifier référence, dispersion et minima.
