# Patch Plan EID Builder — référentiel consolidé

> **Source de vérité.** Ce document réorganise le plan antérieur et l’aligne sur les 13 fichiers Python effectivement présents. Les décisions déjà validées restent normatives, les points calibrables demeurent des paramètres à fixer en L24 et les statuts de lots décrivent l’état de travail (non l’implémentation réelle). Les termes entre crochets signalent une transition prévue, pas une affirmation que le code l’implémente déjà. Le contrat cible est défini une seule fois en §3.2 ; toute autre mention du contrat renvoie à cette section.

## 1. Contexte, Mission (@S-0014 AuthenticsMask) et Principes directeurs

### 1.1 Mission et périmètre

<!-- @S-0014 -->

**Mission exacte EID → AuthenticsMask (AM) :** l’EID produit un `Sighting` à la finalisation de chaque `ClusterSlow` pour qu’AM réalise, au niveau de la session, le regroupement par similarité complète ou partielle, évalue la fréquence de réapparition et en extraie les vrais positifs. Le flux est unidirectionnel EID → AM en v1 : aucun aller-retour entre les deux systèmes.

L’EID mesure, associe à court terme, finalise et émet ; AM juge et consolide à long terme. Le vrai positif est fondé sur la récurrence (nombre d’apparitions distinctes dans un regroupement et leur étalement temporel). Les Empreintes AM vivent pendant la session en v1. Les seuils propres à AM sont dans une `MaskConfig` distincte ; aucun seuil consommateur n’est ajouté au producteur partagé de signature. `sig_distance` est partagée ; le module producteur de signatures fournit des mesures et ne porte pas une politique de consommateur.

**Lien exact de mission :** `@S-0014 AuthenticsMask` — le passage ci-dessus est la définition normative de ce lien.

**Inclus :** les 13 modules du code EID inventoriés en §2.1, leur flux de données, leurs contrats de sortie, les compteurs, la configuration, les lots L01–L24 et les tests de non-régression spécifiés ici.

**Non-objectifs :** floutage ou création du masque dans l’image ; algorithme AM de regroupement et ses modalités internes ; structure et traitements aval des Empreintes ; traitement des empreintes après AM. AM ne peut pas séparer une apparition contaminée : l’EID privilégie donc la précision à court terme ; AM peut fusionner des apparitions fragmentées à long terme.

### 1.2 Principes directeurs et méthode

- Code effectivement présent faisant foi pour l’inventaire et les noms : les décisions validées prévalent sur le code lorsque le lot correspondant reste à faire ; l’écart est explicite, jamais silencieusement présenté comme livré.
- Un `ClusterSlow` finalisé produit un unique `Sighting` ; un `ClusterFast` n’en produit pas. Le callback est un canal de remise, non une condition d’émission.
- Le fast est une piste géométrique, sans identité slow/EID héritée. Chaque observation reçoit toutefois un `obs_fast_id` en Passe 1; ce lien technique temporaire sert uniquement à la continuité cinématique slow, sans transfert d’identité. Les traitements fast précèdent l’association slow.
- Un filtrage absent, négatif, insuffisant ou ambigu ne peut provoquer une association ; en cas de doute, on fracture. Les composantes inconnues restent `None`, jamais un zéro ou un un favorable.
- Une fermeture de lot/frame n’est pas une finalisation de cluster. Horloge métier = `Observation.ts`. `epsilon_close_s` est un délai de sécurité de lots, pas le déclencheur métier de fermeture d’un cluster.
- Pas de poids géométrique dans le score d’identité ; gates géométriques éliminatoires séparés du coût de signature.
- Un `Sighting` est immuable en profondeur, sérialisable en JSON et stable comme contrat AM.
- Une lecture d’état/échantillonnage de clusters est en lecture seule. Les jauges observent ; elles ne déclenchent pas une purge/capacité arbitraire.
- Une mise à jour de code et de plan se fait dans la même passe de lot. Avant chaque lot, contrôler signatures des méthodes, champs réels, clés lues et écrites, chemins d’export, validation `StreamConfig`, et intersections de modifications `cluster.py`.
- Décisions structurelles fermées, sauf les arbitrages explicitement marqués `[Différé — Attendre l'évolution du code (L14–L20) avant de trancher]` ; les formules et valeurs réservées à la calibration L24 restent des tâches de calibration, non des arbitrages ouverts.

### 1.3 Conventions strictes et vocabulaire

- Chaîne : `Observation` → `ClusterFast` (piste) et `ClusterSlow` (identité) → `Sighting` → regroupement AM → Empreinte AM. « Empreinte » ne désigne jamais le `Sighting` EID.
- **Finaliser** : retirer définitivement du registre. Slow : `pop`, geler `promo_outcome`, compiler/cache le `Sighting`, compter puis émettre une fois (callback s’il existe). Fast : retrait et comptage sans sortie `Sighting`. Eviction et `flush()` appellent les mêmes routines de finalisation.
- **Compiler** : `ClusterSlow.compile_prime_sighting()` fabrique/cache l’instantané ; cache non nul n’est pas un critère de finalisation. Répéter retourne le même objet. En v1, compilation à finalisation, sans effet de bord sur l’état.
- **Fermer un lot/frame** : opération `BatchReader`/tranche, ne finalise aucun cluster. Une tranche est un `frame_id`; un lot est une unité de lecture.
- Horloge métier `obs.ts`; `epsilon_close_s` est timeout de sécurité ; `slow_assoc_window_s`/`fast_assoc_window_s` sont fenêtres d’association ; `slow_idle_close_s`/`fast_idle_close_s` sont inactivité ; `slow_max_lifetime_s`/`fast_max_lifetime_s` sont durées de vie maximales. Au maximum, fermer l’ancien et ouvrir un nouveau cluster sans chaînage. `cluster_idle_ttl_s` est garde-fou global.
- `dt = obs.ts - ClusterSlow.last_obs.ts`; la garde d’association slow réutilise `slow_assoc_window_s` (1.5 s) et respecte `0.0 <= dt <= self.config.slow_assoc_window_s`; `slow_idle_close_s` reste distinct pour l’éviction idle.
- Gate spatial slow évalué contre `last_obs`, jamais contre l’ancre : centre-à-centre avec tolérance temporelle et contrôle dimensionnel par axe. Gates désactivés si paramètres `None`, comptés `pass`.
- `cost_sig = clip(sd.cost, 0, 1)` demeure le score de signature v1 ; le gate géométrique n’est pas un poids. `continuity_ok ∈ {pass, fail, n/a}` ; `fail` est éliminatoire, `n/a` est le repli défensif qui poursuit avec Gate D2 + signature.
- Ancre réelle `Observation`; un cluster sans ancre est valide et finalisé. `AnchorState ∈ {PROBATION, LOCKED}` ; `anchor_locked` est dérivé. `QualityScore(total, f_clip, f_h, f_fill, f_hs)` est immuable et seul `.total` compare les candidats. `blocked_identity` et `blocked_quality` sont motifs de tentative, jamais résultats finaux.
- Probation se termine au premier seuil `n_obs ≥ promo_window_n_obs` ou `span_s ≥ promo_window_span_s`. Promotion anticipée si `sig_map_clip == 0` et `sig_map_h_txt ≥ MAP_REF_H`. Résultats finaux : `early`, `window`, `locked_no_promo`.
- Dispersion `quantile_dispersion(costs,q,n_min)`: `n=0 → (None, True)` ; `1≤n<n_min → (max(costs), True)` ; `n≥n_min → (quantile q, False)`. Repli insuffisant n’est jamais favorable. `f_hs` utilise `sig_hs_px[1]`; interface, domaine et repli sont normatifs, formule et `hs_px_ref` relèvent de L24.
- `confidence_*`, `cost` et `quality` sont distincts. `n_promo_blocked_identity` (état par cluster) ≠ `promo_blocked_identity` (compteur global). `blocked_cost` est diagnostic/compteur, non sérialisé dans `Sighting`.
- `CounterKey` s’incrémente avec `bump`; `GaugeKey` se règle à valeur instantanée ; ensembles disjoints, ne jamais sommer des jauges. Une clé absente au rapport doit échouer explicitement. Valeur logique `n/a` est sérialisée `na` si elle figure dans une clé. Les clés d’enums sont majuscules, valeurs de sortie minuscules.
- Identifiants complets : lot `Lnn`, invariant `INV-Lnn-nn`, test `T-Lnn-nn`, décision de lot `ARB-Lnn-nn`, décision transverse `ARB-G-nn`, décision fondatrice `ARB-P0`…`ARB-P8`; `ARB-A*` est une famille historique distincte. Une référence abrégée n’est pas valide. **NOUVEAU** signale un symbole cible absent du dépôt ; **non résolu** n’est utilisé qu’en audit de référence, non pour rouvrir une décision verrouillée.
- Statuts de lot : **Terminé**, **À faire**, **En cours**. Arbitrages : **Acté/verrouillé**, **Remplacé**, **Retiré**, **Fusionné → cible**, ou **Différé** avec marqueur explicite. Les choix numériques explicitement délégués à L24 restent des tâches de calibration.

## 2. Architecture modulaire (Codebase réelle : 13 fichiers) et Invariants globaux

### 2.1 Inventaire vérifié des 13 fichiers Python

La vérification a porté sur les 13 fichiers présents sous les fichiers fournis. Le code fourni contient les chemins suivants ; des noms parfois utilisés dans l’ancien plan (`utils.py`, `distance.py`, `spatial.py`, `io_jsonl.py`, `run_summary.py`) ne sont pas des fichiers actuels et ne sont pas présentés comme tels. Les équivalences de rôle indiquées sont explicatives, non de nouveaux modules.

- **Fichier réel :** `config.py`
  - **Classes / fonctions / constantes effectivement déclarées (surface publique et helpers inclus) :** `StreamConfig`, `StreamConfig.__post_init__`. Champs réels : `temporal_radius_slow_s`, `temporal_radius_evict_s`, `k_pos_norm`, `w_geo`, `w_pos`, `w_dim`, `c_max`, `temporal_radius_fast_s`, `max_travel_px_per_s`, `max_scale_ratio`, `fast_radius_floor_px`, `fast_ambiguity_ratio`, `hs_px_min`, `hs_raw_penalty`, `hs_gate_max`, `shape_w_d`, `shape_w_c`, `shape_w_y`, `occupancy_min`, `map_l_ratio_min`, `map_h_txt_gate`, `map_h_txt_full`, `min_cost_families`, `w_shape`, `w_map`, `epsilon_close_s`, `batch_max_span_factor`, `dedup_ttl_s`, `cluster_max_dispersion`, `n_min_hs`, `n_min_shape`, `n_min_map`, `promo_window_n_obs`, `promo_window_span_s`, `promo_max_drift`, `active_clusters_warn_threshold`. L14/L23 remplacent/suppriment les champs obsolètes ; L23 retire/déprécie `batch_max_span_factor` car `BatchReader` coupe déjà si `obs.ts - batch[0].ts > window_s` (ARB-L23-14).

- **Fichier réel :** `counters.py`
  - **Classes / fonctions / constantes effectivement déclarées (surface publique et helpers inclus) :** `ANOMALY_PREFIX`; `CounterKey`; `GaugeKey`; `Counters.__init__`, `bump`, `_set_gauge`, `sample_active_clusters`, `get`, `as_dict`, `gauges`, `stats`, `items`, `anomalies`. Registre de clés actuel (toutes) : `SLOW_OBS`, `FAST_OBS`, `CLUSTERS_SLOW_CREATED`, `CLUSTERS_FAST_CREATED`, `CLUSTERS_SLOW_FINALIZED`, `EVICTED_FAST`, `HUNGARIAN_BATCHES`, `BATCHES_READ`, `BATCH_OBS_CUMUL`, `BATCH_FRAME_OUT_OF_ORDER`, `GATE_TEMPORAL_REJECT_SLOW`, `COST_REJECT_SLOW`, `COLS_GATE_ONLY`, `COLS_COST_REJECTED`, `FAST_MATCHED`, `GATE_TEMPORAL_REJECT_FAST`, `COST_REJECT_FAST`, `TRACK_ADOPTED`, `EID_ACTIVE_CLUSTERS_OVER_THRESHOLD`, `ANOMALY_IDENTITY_TRACK_MISMATCH`, `ANOMALY_FAST_AMBIGUOUS`, `SRC_OBS_READ_TOTAL`, `SRC_OBS_BUILT`, `SRC_OBS_REJECT_SOURCE`, `SRC_OBS_REJECT_DEGENERATE_BOX`, `SRC_OBS_REJECT_MALFORMED`, `SRC_OBS_SLOW_TS_FALLBACK`, `SRC_OBS_SLOW_TS_FALLBACK_ZERO` (à supprimer L13b), `SRC_OBS_SIG_EMPTY`, `SRC_OBS_DEDUP_SKIPPED`, `SRC_OBS_MISSING_FRAME_ID`, `SRC_DEDUP_KEPT`, `SRC_DEDUP_PURGES`, `SIG_GATE_H_TXT`, `SIG_GATE_L_RATIO`, `SIG_GATE_OCCUPANCY`, `SIG_GATE_NO_DESCRIPTOR`, `SIG_GATE_HS_DIVERGENCE`, `SIG_HS_PLAN_RAW`, `SIG_H_TXT_PENALIZED`, `REF_PROMOTIONS`, `REF_PROMO_EARLY`, `REF_PROMO_WINDOW`, `PROMO_BLOCKED_IDENTITY`, `PROMO_BLOCKED_QUALITY`, `REF_LOCKED_NO_PROMO`, `REF_INITIAL_SIGNATURE_EMPTY`; jauges `EID_ACTIVE_CLUSTERS_SLOW`, `EID_ACTIVE_CLUSTERS_FAST`. Manque réel à corriger L13b : `SRC_OBS_REJECT_MISSING_GEOMETRY`. Compteurs planifiés L14–24 sont énumérés dans leurs fiches, distincts des clés existantes.

- **Fichier réel :** `observation.py`
  - **Classes / fonctions / constantes effectivement déclarées (surface publique et helpers inclus) :** `DEDUP_GEOM_NDIGITS`; `Observation` (`cx`, `cy`, `rw`, `rh`, `has_shape`, `has_map`, `from_entry`). `_parse_row` désigne le parsing cible de `from_entry` (aucune fonction de ce nom actuellement). Champs construits : `uid`, `frame_id`, `ts`, `rect`, `source`, `signature`, `confidence`, `vx`, `vy`, `dedup_key`; validations et échecs `CounterKey.SRC_*`.

- **Fichier réel :** `eid_distance.py` (rôle de distance, ancien `distance.py`)
  - **Classes / fonctions / constantes effectivement déclarées (surface publique et helpers inclus) :** `SigDistance` (`gated_only`), `_shape_array`, `_map_array`, `_h_txt_penalty`, `_eroded_usable`, `_select_hs_plan`, `_compute_hs_distance`, `_occupancy`, `_compute_shape_distance`, `_compute_map_sliding_distance`, `sig_distance`. Sortie diagnostic/familles et gates signature. Pur ; compteurs gérés par orchestration (`builder`).

- **Fichier réel :** `signature.py` (dans le dépôt sous le nom `signature.py`, importé comme `core.signature`)
  - **Classes / fonctions / constantes effectivement déclarées (surface publique et helpers inclus) :** `empty_signature`, `sig_get`, `compute_signature`, `_compute_from_template`, `_erode_kernel`, `_fill_hs`, `_hs_hist_from_mask`, `_fill_shape`, `_fill_map`, `_binarize_text`, `_resolve_shape`, `serialize_signature`, `deserialize_signature`; constante `MAP_REF_H` et formes/imports internes. Producteur, sérialiseur et désérialiseur de signatures.

- **Fichier réel :** `signature_ref.py`
  - **Classes / fonctions / constantes effectivement déclarées (surface publique et helpers inclus) :** `MAP_CLIP_LEFT`, `MAP_CLIP_RIGHT`, `dim_distance`, `QualityScore`, `AnchorState`, `PromoOutcome`, `PromotionResult`, `_calculate_quality_score`, `_dispersion_for_family_e` (fonction morte à supprimer L17), `AnchorManager` (`try_update`, `freeze_promotion`, `anchor`, `anchor_locked`). `try_update` actuel peut renvoyer `blocked_cost=None`; L16 doit l’exposer pour diagnostique de blocage.

- **Fichier réel :** `cluster.py` (rôle géométrique, ancien `spatial.py`)
  - **Classes / fonctions / constantes effectivement déclarées (surface publique et helpers inclus) :** `ClusterState`; `ClusterSlow` (`last_obs`, `add_observation`, `finalize_promotion`, `compile_prime_sighting`); `ClusterFast` (`last_obs`, `add_observation`). Tampon observé `deque(maxlen=50)` slow et `deque(maxlen=100)` fast ; compteurs slow/fast déjà incrémentés dans `ClusterFast.add_observation`. `compile_prime_sighting` émet actuellement des placeholders, dont `geometry.n_geo_*`, champs à restructurer selon §3.2.

- **Fichier réel :** `batching.py`
  - **Classes / fonctions / constantes effectivement déclarées (surface publique et helpers inclus) :** `ObservationSource` (`__init__`, `__iter__`), `JsonlSource` (`__init__`, `__iter__`), `DedupSource` (`__init__`, `__iter__`), `BatchReader` (`__init__`, `batch_size_max`, `_pull`, `_record`, `next_batch`). La purge TTL de `_seen` avant comparaison est le correctif L13b.

- **Fichier réel :** `builder.py`
  - **Classes / fonctions / constantes effectivement déclarées (surface publique et helpers inclus) :** `EID_START`; `StreamEIDBuilder` (`__init__`, `_finalize_slow`, `_finalize_fast`, `_evict_stale_slow`, `_evict_stale_fast`, `flush`, `process_batch`, `_eval_slow_pair`, `_count_sig`, `_solve_slow_identity`, `_process_slow_obs`, `_process_fast_obs`, `_process_fast_track`, `_create_slow`, `_update_slow`, `_count_promotion`, `_create_fast`). `flush` et monotonicité d’horloge sont L13b.

- **Fichier réel :** `cost.py` (construction des coûts, ancien rôle parfois nommé `utils.py`)
  - **Classes / fonctions / constantes effectivement déclarées (surface publique et helpers inclus) :** `BIG_COST_COST`, `BIG_COST_GATE`, `CalibRecord`, `CostMatrix`, `_get_cluster_ts`, `_get_eligible_slow_clusters`, `build_slow_cost_matrix`. `CostMatrix.payload` ne doit pas transporter de distance géométrique (ARB-L10-02).

- **Fichier réel :** `calibration.py` (écrivain; n’existe pas de `run_summary.py` autonome)
  - **Classes / fonctions / constantes effectivement déclarées (surface publique et helpers inclus) :** `SCHEMA_VERSION`, `CalibrationWriter` (`__init__`, `session_id`, `_ensure_open`, `_write`, `write_manifest`, `add_record`, `close`). Le rapport d’exécution est intégré au point d’entrée, non module séparé.

- **Fichier réel :** `sighting.py`
  - **Classes / fonctions / constantes effectivement déclarées (surface publique et helpers inclus) :** `_freeze`, `_thaw`, `Sighting` (`__post_init__`, `span_frames`, `span_time_s`, `to_json`). À l’état actuel, objet frozen/slots et cinq blocs figés profondément, mais `schema_version` et `lifecycle.n_slow/n_fast` n’y figurent pas encore : cibles prévues en §3.2/L14/L19.

- **Fichier réel :** `stream_eid_builder.py`
  - **Classes / fonctions / constantes effectivement déclarées (surface publique et helpers inclus) :** `_find_ndarray`, `_safe_dumps`, `main` (parse CLI, JSONL, `StreamEIDBuilder`, `BatchReader`, callback d’écriture `_eid_sightings.jsonl`, exports/rapport). Le code de `main` intègre les rôles d’orchestration d’entrée/sortie ; pas de `io_jsonl.py` ni `run_summary.py` séparé.

> Modules relatifs : les fichiers fournis sont les modules internes `eid`/`core` importés par ces noms. Le mapping respecte leurs **noms de fichier réels** ; ne créer ni prétendre existants `utils.py`, `distance.py`, `spatial.py`, `io_jsonl.py`, `run_summary.py` sans décision de périmètre.

### 2.2 Invariants actifs, consolidés

- **ID :** `INV-L02-01`
  - **Invariant normatif :** Pas de nom actif `dedup_window_s`; seul `dedup_ttl_s` est utilisé.

- **ID :** `INV-L08-01`
  - **Invariant normatif :** AM ne dépend que de `Sighting` et `sig_distance`, jamais des modules internes EID (`builder`, `cluster`, `cost`, `batching`).

- **ID :** `INV-L08-02`
  - **Invariant normatif :** Composantes par famille ; indisponible = `None`, jamais `1.0`/`0.0` par défaut.

- **ID :** `INV-L08-03`
  - **Invariant normatif :** Sighting porte les temps début/fin et nombre d’observations dans le contrat cible (§3.2).

- **ID :** `INV-L08-04`
  - **Invariant normatif :** Callback est seul canal externe ; émission/finalisation ne dépend pas de sa présence.

- **ID :** `INV-L08-05`
  - **Invariant normatif :** Sighting émis et Empreinte publiée sont immuables.

- **ID :** `INV-L08-06`
  - **Invariant normatif :** Aucun état modifiable partagé avec le thread vidéo ; lecture sur instantané.

- **ID :** `INV-L08-07`
  - **Invariant normatif :** Aucun retour AM → EID en v1.

- **ID :** `INV-L10-01`
  - **Invariant normatif :** `eid_distance.sig_distance` reste pur ; comptage SIG par orchestrateur, une fois par paire évaluée.

- **ID :** `INV-L10-DIST-SYM`
  - **Invariant normatif :** Pour signatures `a`, `b`, `sig_distance(a,b) == sig_distance(b,a)` ; contrôle et test de symétrie portés par L15 (`INV-L15-SYM`).

- **ID :** `INV-L12-02`
  - **Invariant normatif :** `promo_outcome=None` en probation ; à finalisation asserté parmi `early/window/locked_no_promo`, jamais `None`/`blocked_*` dans sortie.

- **ID :** `INV-L12-03`
  - **Invariant normatif :** Les quatre composantes `QualityScore` sont soit toutes renseignées soit toutes `None`, jamais partielles.

- **ID :** `INV-L13-01`
  - **Invariant normatif :** Ancre issue d’observation réelle ; signature vide jamais ancre/remplacement ; promotion unique comparée sur `QualityScore.total`; aucune observation supprimée de l’historique de décision.

- **ID :** `INV-L13-02`
  - **Invariant normatif :** Ancre immuable après `LOCKED`; aucune géométrie calculée à partir de l’ancre.

- **ID :** `INV-L14-01`
  - **Invariant normatif :** À chaque émission : `geometry.reject_total = geometry.spatial.reject + geometry.temporal.reject_dt`.

- **ID :** `INV-L14-02`
  - **Invariant normatif :** En Passe 2, filtre temporel slow précède filtre spatial; hors `0.0 ≤ dt ≤ self.config.slow_assoc_window_s` (1.5 s), rejet temporel, spatial non évalué. En Passe 1 fast, `abs(dt) <= fast_assoc_window_s`; un léger `dt` négatif est admis pour latence slow.

- **ID :** `INV-L14-03`
  - **Invariant normatif :** Valider en `StreamConfig`, sans anciens alias : `slow_max_lifetime_s > slow_idle_close_s >= slow_assoc_window_s`; `fast_max_lifetime_s > fast_idle_close_s ≥ fast_assoc_window_s`; `slow_assoc_window_s > fast_assoc_window_s > epsilon_close_s`; `fast_max_lifetime_s ≥ 2·epsilon_close_s`; `cluster_idle_ttl_s ≥ max(slow_idle_close_s,fast_idle_close_s)`. En L14, `slow_assoc_window_s=1.5 s` est la garde d’association; `slow_idle_close_s=1.5 s` est un paramètre distinct d’éviction idle.

- **ID :** `INV-L14-04`
  - **Invariant normatif :** Tranche tardive n’évince pas : traitement normal et `SLICE_LATE`; affectation fast d’une tranche avant association slow.

- **ID :** `INV-L15-01`
  - **Invariant normatif :** Références `INV-*`, `T-*`, `ARB-*` toujours complètes.

- **ID :** `INV-L15-02`
  - **Invariant normatif :** Rapport lit une liste fermée de `CounterKey`; clé absente = erreur explicite.

- **ID :** `INV-L15-03`
  - **Invariant normatif :** Association acceptée après empan temporel et coût admissible ; dispersion vérifiée a posteriori.

- **ID :** `INV-L15-04`
  - **Invariant normatif :** `mask_uid` informatif, ne décide jamais ; refus ouvre un nouveau cluster ; `continuity_ok=fail` ne provoque jamais d’association; `n/a` poursuit les filtres spatial + signature selon L14.

- **ID :** `INV-L15-05`
  - **Invariant normatif :** Score composite expose ses composantes.

- **ID :** `INV-L15-06`
  - **Invariant normatif :** `cost`, `quality`, `confidence` ne partagent pas un même champ.

- **ID :** `INV-L15-07`
  - **Invariant normatif :** `compile_prime_sighting()` idempotente : même objet, aucune nouvelle émission.

- **ID :** `INV-L15-08`
  - **Invariant normatif :** Rejet porte exactement le motif du premier filtre échoué dans l’ordre d’association.

- **ID :** `INV-L15-SYM`
  - **Invariant normatif :** `NO_SIG_EQUIVALENT` suit le même chemin de rejet quelle que soit la source, y compris `SRC_OBS_REJECT_MISSING_GEOMETRY`; distance symétrique.

- **ID :** `INV-L16-01`
  - **Invariant normatif :** Fermeture nominale de lot à l’arrivée d’un slow ; `epsilon_close_s` seulement timeout de sécurité.

- **ID :** `INV-L19-01`
  - **Invariant normatif :** Données insuffisantes ou `n_min_*` non atteint : jamais de valeur favorable.

- **ID :** `INV-L20-01`
  - **Invariant normatif :** Interface `f_hs`: données insuffisantes ou minimum non atteint → `(None, fallback=True)`, jamais favorable ; formule calibrable L24.

- **ID :** `INV-L21-01`
  - **Invariant normatif :** `sample_active_clusters()` ne modifie pas les registres/dicts ; lecture seule.

- **ID :** `INV-L21-02`
  - **Invariant normatif :** Par défaut `n_min_map ≥ n_min_shape` et `n_min_map ≥ n_min_hs` (map réputé plus coûteux à stabiliser) ; contrôle en config, non codé en dur, valeurs tunables.

- **ID :** `INV-L21-03`
  - **Invariant normatif :** `compile_prime_sighting()` idempotente et sans effet de bord sur `self.state`; répétitions identiques hors évolution naturelle entre appels.

- **ID :** `INV-L23-01`
  - **Invariant normatif :** `blocked_cost` jamais sérialisé dans `Sighting`; limité compteurs et diagnostics.

- **ID :** `INV-L06-03`
  - **Invariant normatif :** **Strictement positif** : toutes les distances/mesures de distance ciblées sont dans `[0.0, +inf[` ; pas de clip supérieur à `1.0`. Les distances gardent leur domaine mathématique naturel ; ne pas confondre avec `cost_sig`, dont l’écrêtage v1 à `[0,1]` est une décision distincte. Selon `ARB-G-13` / D4.

## 3. Spécification unifiée du contrat cible Sighting (Source unique de vérité)

Aucune autre section ne redéfinit ni ne propose un schéma alternatif : se référer à §3.2. Les structures ci-après sont cibles L14–L19, et non description erronée du code actuel. Le `Sighting` demeure frozen/slots et profondément immuable ; `to_json()` restitue des conteneurs JSON frais.

### §3.1 Format d’entrée JSONL

- JSONL d’observations avec `entries`; chaque entrée comprend une source `last_source` (`slow` ou `fast`), `uid`, `frame_id`, un horodatage adapté à la source (`last_slow_ts` avec fallback actuel à `last_detected_ts`, ou `last_detected_ts` pour fast), rectangle et signature.
- `rx` et `ry` sont obligatoires à la source, présents et non-`None`; ne pas substituer zéro. Les dimensions `rw`,`rh` doivent être positives et bbox non dégénérée. L’absence de `frame_id` (incluant `None`) est rejet strict avant lots ; valeur négative ou non entière exclue par construction prévue.
- `Observation.from_entry` accepte toute observation valide sauf si `entry.get("state") == "LOST"`; `CONFIRMED`, `PENDING` et état absent sont acceptés (compatibilité arrière, notamment des tests synthétiques). L’observation `LOST` est rejetée avant les passes. La structure `signature` fait foi quant à la disponibilité et aux valeurs des canaux (`hs`, `shape`, `map`) ; ne pas inférer la présence de canal par un champ séparé. Entrée `signal.valid` est vérifiée : `False` → `SRC_NOT_VALID` (clé à intégrer au registre de compteurs); signature sans aucun canal → `SRC_NO_SIGNAL`. Ces clés de compteur et la validation explicite sont à implémenter dans le contrat d’entrée L22/L23 si elles ne sont pas disponibles dans le registre courant.
- La géométrie obligatoire se vérifie avant de constituer une `Observation`; manquante → `SRC_OBS_REJECT_MISSING_GEOMETRY`, sans dégradation vers une observation faussement valide. Entrée mal formée/conversion impossible → rejet `SRC_OBS_REJECT_MALFORMED`; source invalide, boîte dégénérée, signature vide et horodatage fallback gardent des compteurs séparés.
- `confidence`, `vx`, `vy` restent optionnels `Optional[float]`, absents/`None` → `None`; valeurs présentes non convertibles → rejet malformed. `vx`,`vy` ne sont pas utilisés pour le contrôle cinématique actuel, qui utilise centres `cx`,`cy` et déplacement.
- JSONL lu par `JsonlSource`, déduplication `DedupSource`, lots `BatchReader`; contrat des sources `ObservationSource` avec `Counters` injecté. `dedup_ttl_s=0.020` cible L23 ; identité de dédup et âge contrôlés au moment de la recherche ; purge amortie actuelle toutes 256 observations corrigée par L13b avant le contrôle du doublon.

### §3.2 Schéma JSON cible unique de `Sighting`

L’objet racine de sortie est exactement organisé ainsi (les blocs listés existent même si les valeurs facultatives sont `null`; les types indiqués sont contractuels) :

```json
{
  "schema_version": 1,
  "eid": 123,
  "lifecycle": {
    "first_frame": 1,
    "last_frame": 9,
    "created_ts": 0.0,
    "last_seen_ts": 1.0,
    "n_obs": 9,
    "n_slow": 5,
    "n_fast": 4,
    "span_frames": 8,
    "span_time_s": 1.0
  },
  "identity": {
    "anchor_signature": null,
    "anchor_locked": false,
    "anchor_quality": null,
    "anchor_quality_components": {
      "f_clip": null,
      "f_h": null,
      "f_fill": null,
      "f_hs": null
    },
    "promo_outcome": "locked_no_promo",
    "promo_frame": null,
    "n_post_promo_outliers": null,
    "n_promo_blocked_identity": null,
    "promo_blocked_max_cost": null
  },
  "dispersion": {
    "at_admission": null,
    "at_admission_excl_outliers": null,
    "final": null,
    "final_excl_outliers": null,
    "delta_dispersion": null,
    "hs": null,
    "shape": null,
    "map": null,
    "n_hs": null,
    "n_shape": null,
    "n_map": null,
    "confidence_mean": null,
    "confidence_min": null,
    "at_admission_sighting": null,
    "final_sighting": null
  },
  "signal": {
    "confidence_mean": null,
    "confidence_min": null,
    "n_gate_hs": null,
    "n_gate_occupancy": null,
    "n_gate_h_txt": null,
    "n_gate_l_ratio": null,
    "f_hs": null
  },
  "geometry": {
    "first": null,
    "last": null,
    "anchor": null,
    "txt_scale": {
      "h_txt_med": null,
      "h_txt_iqr": null,
      "w_med": null,
      "w_iqr": null,
      "ar_med": null,
      "ar_iqr": null
    },
    "spatial": { "pass": 0, "reject": 0 },
    "temporal": { "reject_dt": 0 },
    "continuity": { "pass": 0, "fail": 0, "na": 0 },
    "reject_total": 0
  },
  "tracking": {
    "mask_uids_seen": null,
    "mask_uid_switches": null,
    "uid_agree": null,
    "uid_disagree": null
  }
}
```

Schéma sous forme de bloc structurel compact : racine `schema_version`, `eid`, `lifecycle`, `identity`, `dispersion`, `signal`, `geometry`, `tracking`. `lifecycle` comporte obligatoirement `first_frame`, `last_frame`, `created_ts`, `last_seen_ts`, `n_obs`, `n_slow`, `n_fast`, `span_frames`, `span_time_s`. `identity` comporte la signature d’ancre toujours présente (null si aucune), verrouillage, qualité + quatre composantes, promotion finale, champs réservés/outliers et diagnostics par cluster. `mask_uid_switches` apparaît **uniquement** sous `tracking`, jamais sous `identity`. `geometry` est imbriqué exactement selon les branches ci-dessus, dont `txt_scale.w_med` (jamais `w_Med`). `reject_total = spatial.reject + temporal.reject_dt`; la continuité est mesurée séparément et exclue de ce total. L’enveloppe technique `Sighting` continue de porter `first_frame`, `last_frame`, `created_ts`, `last_seen_ts`, `n_obs` puis `to_json()` les sérialise sous `lifecycle`. `schema_version` est ajouté à la racine du contrat stable (à ne pas confondre avec le `mode` de session de l’export JSONL). Le détail numérique des mesures géométriques suit §4; `None` signifie inconnue/non disponible, zéro est réservé à un compte réellement nul.

`GeometrySnapshot` est une valeur immuable `@dataclass(frozen=True)` avec champs `x: float`, `y: float`, `w: float`, `h: float`, `ts: float`, `frame: int`; `to_dict()` retourne exactement ces six clés/types (`x`, `y`, `w`, `h`, `ts`, `frame`). `GeometrySnapshot.from_obs(obs: Observation)` construit l’instantané et `cx=x+w*0.5`, `cy=y+h*0.5`. Les blocs `geometry.first`, `last` et `anchor` sérialisent ces instantanés ou `null`. Pour chaque observation slow conservée par `ClusterSlow`, les valeurs physiques sont `w=float(obs.rect[2])`, `h=float(obs.rect[3])`, `h_txt=float(obs.signature.get("gates", {}).get("h_txt"))` si valeur numérique valide `>0`, sinon `h`, et `ar=w/h`. `txt_scale` expose les médianes exactes et dispersions interquartiles exactes (`q75-q25`) `h_txt_med`, `h_txt_iqr`, `w_med`, `w_iqr`, `ar_med`, `ar_iqr`; aucun quantile interpolé. Le calcul couvre toutes les observations slow de la durée de vie du cluster, sans troncature `deque(maxlen=50)` (conserver au besoin la liste légère complète des triplets `(h_txt,w,ar)`). Aucun `n_samples` n’est ajouté au bloc du contrat.

`anchor_signature` est toujours présent, éventuellement `null`; sans ancre `anchor_quality=null` et les quatre clés de composantes existent avec `null`. Les composantes de qualité ne sont remplies que lorsqu’elles sont toutes calculables. Aucun `SignalFlags` : les booléens inventés `hs_ok`, `shape_ok`, `map_ok` ne sont pas ajoutés. `mask_uid_switches` n’existe pas dans `identity`. `blocked_cost` n’est pas dans le schéma. Les compteurs et diagnostics résiduels dans le JSONL de calibration/rapport ne modifient pas le contrat du `Sighting`.

## 4. Registre unifié et dédoublonné des décisions (ARB-\* et D1–D7)

### §4.1 Décisions transverses `ARB-G-01` à `ARB-G-17` (toutes verrouillées)

Les règles ci-dessous sont les entrées normatives ; les invariants opérationnels sont regroupés exclusivement en §2.2, le contrat exclusivement en §3.2. Toutes sont verrouillées (Acté). Aucune entrée ne réouvre une question.

- **Décision :** `ARB-G-01`
  - **Décision verrouillée :** Architecture de responsabilité : l’EID mesure et émet l’apparition finalisée ; AM groupe et juge. Voir §1.1 et `ARB-P8`.

- **Décision :** `ARB-G-02`
  - **Décision verrouillée :** Fast et slow séparés/séquentiels ; fast géométrique sans identité slow héritée. Voir D1, D3.

- **Décision :** `ARB-G-03`
  - **Décision verrouillée :** Enveloppe d’échange AM stable et unidirectionnelle en v1 ; le contrat canonique est exclusivement §3.2.

- **Décision :** `ARB-G-04`
  - **Décision verrouillée :** `ARB-P0…P8` désignent les décisions fondatrices, pas les champs ; utiliser les noms de code des champs exportés.

- **Décision :** `ARB-G-05`
  - **Décision verrouillée :** Compteurs globaux et conventions de nommage/type selon D7 ; une seule écriture par compteur.

- **Décision :** `ARB-G-06`
  - **Décision verrouillée :** Champ par cluster distinct du compteur global, notamment `n_promo_blocked_identity` ≠ `promo_blocked_identity`.

- **Décision :** `ARB-G-07`
  - **Décision verrouillée :** `quantile_dispersion` est le noyau pur appliqué par famille ; supprimer la variante morte `_dispersion_for_family_e` lors de L17.

- **Décision :** `ARB-G-08`
  - **Décision verrouillée :** `dispersion_n_min` global sert au filtre décisionnel ; `n_min_*` sert aux diagnostics/métriques L19.

- **Décision :** `ARB-G-09`
  - **Décision verrouillée :** Supprimer le commentaire historique `SourceStats`; vérifier grep et sous-classe `ObservationSource` sans initialisation de compteurs en L23.

- **Décision :** `ARB-G-10`
  - **Décision verrouillée :** Supprimer le repli `anchor_confidence=1.0`; utiliser `anchor_quality`, avec null si inconnue.

- **Décision :** `ARB-G-11`
  - **Décision verrouillée :** Géométrie source requise; `rx/ry` manquants/null rejetés; `confidence/vx/vy` facultatifs; aucun défaut de frame; `frame_id` absent/null rejeté.

- **Décision :** `ARB-G-12`
  - **Décision verrouillée :** Déduplication vérifie l’âge au moment de la recherche et conserve une purge amortie, sans laisser un doublon périmé bloquer son nouvel enregistrement.

- **Décision :** `ARB-G-13`
  - **Décision verrouillée :** Décision verrouillée : `DedupSource` exige `StreamConfig` et lit le TTL configuré sans fallback ; et distances métriques respectent positivité stricte `[0.0, +inf[` sans clip supérieur à 1 (clarification L06-03 / D4 demandée). L’écrêtage propre à `cost_sig` demeure la seule règle distincte.

- **Décision :** `ARB-G-14`
  - **Décision verrouillée :** Les lectures des jauges/compteurs typés utilisent `CounterKey`/`GaugeKey` et des clés déclarées, non des chaînes ad hoc.

- **Décision :** `ARB-G-15`
  - **Décision verrouillée :** Test symétrie inclut `NO_SIG_EQUIVALENT` et au moins la source `SRC_OBS_REJECT_MISSING_GEOMETRY`.

- **Décision :** `ARB-G-16`
  - **Décision verrouillée :** État du rapport aligné sur clés typées réelles; test de symétrie relève de L15.

- **Décision :** `ARB-G-17`
  - **Décision verrouillée :** Tout champ hors schéma final ne peut être ajouté qu’au registre/version de contrat ; `SignalFlags` rejeté ; `mask_uid_switches` dans `tracking` seulement.

Autres transverses historiquement identifiés, tous verrouillés : `ARB-G-18` les quatre composantes de qualité sont remplies en L24 quand calculables ; `ARB-G-19` une apparition sans ancre est finalisée/émise, callback facultatif ; `ARB-G-20` sans ancre, signature et qualité nulles avec quatre composantes nulles ; `ARB-G-21` compteur de finalisation sans ancre appliqué une fois même sans callback (nom final décidé D7); `ARB-G-24` `confidence`, `vx`, `vy` conservés `Optional[float]`, non consommés actuellement.

Décisions fondatrices : `ARB-P0` sortie EID = `Sighting`; `ARB-P1` AM ne reçoit que les apparitions finalisées; `ARB-P2` pas de retour en v1; `ARB-P3` thread dédié, EID puis AM en séquence; `ARB-P4` durée Empreinte = session; `ARB-P5` vrai positif = récurrence; `ARB-P6` seuils AM dans `MaskConfig`, aucun dans le producteur commun; `ARB-P7` renommage historique `footprint`→`sighting`; `ARB-P8` EID mesure, AM juge.

Famille `ARB-A` conservée et clôturée : `ARB-A4d` champ cluster/global distinct ; `ARB-A5c` précision avant rappel ; `ARB-A5d` délais spécifiques + TTL global ; `ARB-A5e` géométrie et sérialisation selon D2/D7 ; `ARB-A6` contrat de `f_hs` fixé en L20, formule et calibration L24 ; `ARB-A7-1` nom `dedup_ttl_s`; `ARB-A7-2` délais propres et TTL global (ancienne valeur 2.0 provisoire, calibration L24) ; `ARB-A10` bornes thermométriques, lecture seule, coordination n_min, mode flux interchangeable avec deux horloges/parité.

### §4.2 Décisions de domaine D1 à D7, sans contradiction

#### D1 — Identité et association slow

Architecture à deux passes : chaque observation valide (fast ou slow) passe d’abord par Passe 1 (`ClusterFast`), puis seules les observations `source == "slow"` passent par Passe 2 (`ClusterSlow`). L’association slow applique dans cet ordre les exclusions : (a) `0.0 <= dt <= slow_assoc_window_s`; (b) Gate spatial D2; (c) continuité cinématique fast (`CONTINUITY_FAIL` éliminatoire; `CONTINUITY_NA` défensif non éliminatoire); (d) disponibilité et seuil de signature; (e) Hongrois 1-à-1 sur le coût admissible. Il n’y a pas dans ce lot de marge D1 ou de post-filtre de dispersion qui modifie cette règle. Rejet porte le premier motif échoué. Une signature insuffisante, une géométrie refusée ou une continuité cassée ne peut provoquer une association : créer un nouveau `ClusterSlow`. `mask_uid` ne participe pas à la décision. Détails géométriques D2, fast D3, temporalité D4, promotion D5, signatures D6, sorties/counters D7.

#### D2 — Géométrie, continuité, distances

Passe 1 (`ClusterFast`) apparie toutes les observations valides de la tranche aux pistes actives par Hungarian 1-à-1 (`scipy.optimize.linear_sum_assignment`). Une paire fast est éligible si `abs(dt) <= fast_assoc_window_s`, où `dt=obs.ts-cluster_fast.last_obs.ts` (le `dt` légèrement négatif autorise le retard de 1–2 frames du thread slow), et si `_geo_compatible(cluster_fast.last_obs, obs, abs(dt), cfg)` passe. Ce prédicat applique le gate anisotrope `vx <= geo_vmax_px_s * (1 + abs(dt) * geo_time_relax_factor)` et `vy <= vx * geo_y_axis_ratio`, OU `iou >= geo_iou_min`. Les coûts compatibles sont `dist_center / d_max`; les incompatibles valent `+inf`/sont interdits. `obs_fast_id` prend l’ID du `ClusterFast` apparié. Si `obs.ts >= last_obs.ts`, ajouter l’observation et avancer la tête; si elle est plus ancienne, rattacher l’ID sans réécrire la tête monotone. Toute observation non appariée ouvre une piste fast neuve et obtient son ID dès sa première frame.

Passe 2 ne traite que les observations slow et applique les quatre filtres séquentiels : (1) `0.0 <= dt=obs_slow.ts-cluster_slow.last_obs.ts <= slow_assoc_window_s`; (2) `_geo_compatible(cluster_slow.last_obs, obs_slow, dt, cfg)`, rejet `SRC_SLOW_GEO_REJECT` sans évaluer signature; (3) comparaison `last_obs_fast_id`/`obs_fast_id`, égalité non-None = `CONTINUITY_PASS`, différence non-None = `CONTINUITY_FAIL` et rejet immédiat, un ID None = `CONTINUITY_NA` et poursuite (repli défensif); (4) `compute_pair_similarity(...)`, rejet `SRC_SLOW_LOW_SIM` si `sim < cfg.sim_threshold`, sinon coût Hongrois `1.0-sim`. Les candidats refusés valent `+inf`; seuls les couples finis sont retenus par résolution Hongroise 1-à-1. Un match appelle `cluster_slow.add(obs_slow, fast_cluster_id=obs_fast_id)`; toute slow non appariée ouvre un nouveau `ClusterSlow` avec `first_obs=last_obs=obs_slow` et `last_obs_fast_id=obs_fast_id`.

Les filtres géométriques sont éliminatoires et n’ajoutent aucun poids au score d’identité. Le contrôle slow est contre `last_obs`, jamais l’ancre. `geometry.reject_total = spatial.reject + temporal.reject_dt`; la continuité est comptée séparément. Paramètres spatiaux configurables indiqués en L14, `None` désactive le gate concerné et compte pass. Distances générales dans `[0.0,+inf[` sans clip supérieur à 1; ne pas confondre avec le coût de signature borné séparément. Une fois une piste fast terminée, le slow ne peut plus s’y associer; finalisation D4 et AM peuvent refusionner.

#### D3 — ClusterFast

Piste géométrique courte durée, contenant toutes les observations (fast et slow), Hungarian simultané un-à-un par tranche; fenêtres temporelle et spatiale configurées. Le coût est distance euclidienne normalisée `dist_center/d_max` pour les paires compatibles, incompatible = interdit. Tout observation appariée porte `obs_fast_id`; l’ancrage est monotone (une observation retardée s’attache sans recul de `last_obs`). Non affectée → nouvelle piste fast. `last_obs_fast_id` est seulement la référence de continuité qui traverse Passe 1→Passe 2, jamais un `eid` hérité ni une identité transmise au slow. Délais, durées max, horloges, fermeture : D4.

#### D4 — Cycle de vie, batch et configuration temporelle

Horloge métier `obs.ts`; en mode batch elle gouverne aussi `epsilon_close_s` afin de rejouer à l’identique; en mode flux L22, horloge murale de réception sert au timeout de sécurité et surcharge. Éviction à `ts_tranche=min(obs.ts)`; frame ordonné strictement au lecteur. Lot fermé ne finalise rien; arrivée slow ferme normalement le batch, `epsilon_close_s` timeout sécurité. Délais par registre et durées sont les inégalités `INV-L14-03`; anciens noms rejetés nativement par dataclass, sans alias/from_dict. `max_lifetime` ferme avant affectation et l’observation repart dans une piste neuve sans chaînage. Fin de cluster par finalisation seulement (inactivité, max lifetime, flush, critère métier); flush idempotent. Distances `[0.0,+inf[` dans le domaine strict ; aucune coupure supérieure à 1; tout filtre coûteux garde sa définition particulière (coût signature étant plafonné séparément). La valeur `dedup_ttl_s` et expurgation de `batch_max_span_factor` sont en D7/L23.

#### D5 — Ancre, qualité, promotion

Ancre = observation réelle; signature vide exclue avant score, sans rejet slow supplémentaire, `ref_initial_signature_empty` conservé. État `PROBATION`/`LOCKED`; score qualité historique est aire×`sig_map_l`, plus 10 000 000 si `sig_map_clip==0`, signature vide/aire≤0 historique -1 (à remplacer avec la formule L24); seul `.total` compare. La promotion anticipée dépend strictement de `sig_map_clip==0` et `sig_map_h_txt≥MAP_REF_H`; sinon seuil premier atteint. Sans ancre, état transitoire permis puis résultat final `locked_no_promo`; meilleures observations conservées si existantes. `blocked_identity/blocked_quality` ne sont pas statuts finaux. `promo_max_drift=None` désactivé jusqu’à calibration L24. `QualityScore` immuable avec quatre composantes toutes disponibles ou toutes nulles; L24 calcule et remplit ensemble. `anchor_locked` dérivé. `promo_frame` correspond à la frame de promotion s’il y en a eu une.

#### D6 — Signature, famille, dispersion

`sig_distance` commune et pure; compteurs au builder une seule fois par paire. Ordre de gates : `no_descriptor`, `occupancy`, `hs_divergence`, `l_ratio`, `h_txt`; distinguer absence de canal d’insuffisance de `min_cost_families` (v1 valeur indicatrice 2; champ actuellement config 1–2, calibration/hybride L24). `cost_sig=clip(sd.cost,0,1)` comparé au `cluster_admit_cost_max`, sans poids géométrique. Signature vide non rejet slow additionnel; jamais ancre. Après admission provisoire, recalcul de dispersion; au-delà de `cluster_max_dispersion`, fermer/finaliser et compter `cluster_closed_dispersion`; `None` ne ferme jamais. `quantile_dispersion` par famille selon règles §1.3 ; `dispersion_fallback_max` quand booléen repli True. `dispersion_n_min` global décisionnel; `n_min_hs/shape/map` diagnostics uniquement et condition par défaut `n_min_map ≥ n_min_shape`, `n_min_map ≥ n_min_hs`. Métriques détaillées `dispersion_admit`, `dispersion_final`, excluant outliers, `delta_dispersion`, ventilations, `n_post_promo_outliers`, `drift_*` sont L19. `f_hs` et borne/repli en L20; formule `hs_px_ref` L24. Le gate occupancy historique qui lit `sig_shape_n_valid` absent est dette L17.

#### D7 — Contrat, compteurs, export et performance

Contrat unique en §3.2 ; blocs toujours présents, unavailable `None`, Sighting gelé profondément. Finalisé signifie compilé et émis, callback s’il existe. AM dépend uniquement du type stable et de `sig_distance`; aucun retour AM en v1. Les compteurs globaux font foi, la projection per-cluster `geometry` suit §3.2. Compteur sans ancre = appliquer une fois à chaque finalisation slow sans ancre, callback indifférent; confirmer son identifiant définitif dans `CounterKey` L13b/L23. `_safe_dumps` produit le JSONL `*_eid_sightings.jsonl`; `schema_version` et `mode` sont à la racine du document d’export; `mode` fixé par session dans `CalibrationWriter`, pas dans un champ ajouté à `Sighting` à chaque ligne. `CounterKey` additionnel, `GaugeKey` valeur instantanée. `EID_ACTIVE_CLUSTERS_SLOW/FAST` sont jauges; `EID_ACTIVE_CLUSTERS_OVER_THRESHOLD` compteur, hausse au front montant seulement; aucune limite dure sur les registres. `sample_active_clusters` ne modifie pas registre, les jauges étant les seules écritures d’observabilité autorisées. `maxlen` ne peut influer sur décision; les stats `geometry.txt_scale` sont exactes sur toutes les slow (L14, sans troncature). `n_fast` est suivi de `ClusterFast` à `ClusterSlow` puis lifecycle Sighting (L14/L19/L21). Aucun budget perf chiffré convenu ; cible qualitative « sans overhead mesurable ». Mode flux, back-pressure, politique perte et parité L22.

### §4.3 Registre unique des décisions par lot `ARB-L01-*` à `ARB-L24-*`

Les entrées ci-dessous fusionnent les anciens registres de lot en une seule occurrence de chaque identifiant. Les arbitrages sont verrouillés sauf les décisions expressément marquées comme différées ; entrées absentes dans l’ancienne numérotation ne sont pas inventées. `ARB-L13b-01…04` sont les correctifs complémentaires obligatoires. Les fiches de lots n’en recopient pas le contenu normatif, elles y renvoient.

- **ID :** `ARB-L02-01`
  - **Statut :** Acté
  - **Décision consolidée :** `dedup_ttl_s` seul nom de TTL dédup; ancien nom proscrit.

- **ID :** `ARB-L03-01`
  - **Statut :** Acté
  - **Décision consolidée :** Sources portent `Counters`/`CounterKey.SRC_*`; pas de `SourceStats`, pas de valeur fictive; repli qualité `1.0` supprimé.

- **ID :** `ARB-L10-01`
  - **Statut :** Acté
  - **Décision consolidée :** Supprimer court-circuit géométrique dans coût, remplacer par gate géométrique séparé L14; retirer `w_pos`,`k_pos_norm`,`d_pos`; composants non applicables coût max/confiance 0.

- **ID :** `ARB-L10-02`
  - **Statut :** Acté
  - **Décision consolidée :** `CostMatrix.payload` sans distance géométrique.

- **ID :** `ARB-L10-03`
  - **Statut :** Acté
  - **Décision consolidée :** Supprimer `_CALIB_PROBE`, `set_calib_probe`, `get_calib_probe`.

- **ID :** `ARB-L11-01`
  - **Statut :** Acté
  - **Décision consolidée :** Reclassement `anchor_signature` dans `identity`, `tracking.stats` vers `lifecycle`, supprimer `density_fps`, sans nouveau calcul.

- **ID :** `ARB-L11-02`
  - **Statut :** Acté
  - **Décision consolidée :** Historique L11 : ancienne qualité reste en `signal` jusqu’au renommage `anchor_quality` L12; pas de déplacement prématuré.

- **ID :** `ARB-L11-03`
  - **Statut :** Acté
  - **Décision consolidée :** `n_slow/n_fast` absents au stade L11; exposés L19 dans lifecycle (anciennement dette L18).

- **ID :** `ARB-L11-04`
  - **Statut :** Acté
  - **Décision consolidée :** lifecycle depuis enveloppe; ajoute propriétés `span_frames`/`span_time_s`, garde fenêtre et nombre d’observations.

- **ID :** `ARB-L11-05`
  - **Statut :** Acté
  - **Décision consolidée :** Clés identity pas encore calculées restent null jusqu’au lot remplisseur.

- **ID :** `ARB-L11-06`
  - **Statut :** Acté
  - **Décision consolidée :** `tracking` maintenu, valeurs préservées, complété L15.

- **ID :** `ARB-L11-07`
  - **Statut :** Acté
  - **Décision consolidée :** Livraison historique limitée à `sighting.py`, puis `cluster.py`; diff de ce lot limité à ces fichiers.

- **ID :** `ARB-L11-08`
  - **Statut :** Acté
  - **Décision consolidée :** L13 garde `QualityScore`, retire FINALIZED, remplit identity sans rejouer reclassements L11.

- **ID :** `ARB-L12-01`
  - **Statut :** Acté
  - **Décision consolidée :** Retirer affectation de `FINALIZED`; état cluster seulement `ACTIVE`, finalisation = retrait.

- **ID :** `ARB-L12-02`
  - **Statut :** Acté
  - **Décision consolidée :** Signature vide ni ancre ni remplacement, sans rejet slow supplémentaire; garder `ref_initial_signature_empty`.

- **ID :** `ARB-L12-03`
  - **Statut :** Acté
  - **Décision consolidée :** `ClusterState.FINALIZED` absent; `prime_sighting` n’est pas critère de fin.

- **ID :** `ARB-L12-04`
  - **Statut :** Acté
  - **Décision consolidée :** Gel final `promo_outcome` parmi trois états valides, ne pas remplacer un résultat déjà fixé.

- **ID :** `ARB-L13-01`
  - **Statut :** Remplacé → `ARB-G-07`
  - **Décision consolidée :** Variante historique dispersion par famille abandonnée; utiliser noyau L15 puis supprimer helper.

- **ID :** `ARB-L13-05`
  - **Statut :** Acté
  - **Décision consolidée :** Préfixe interne préservé pour distinguer cette série de ses homonymes historiques.

- **ID :** `ARB-L13-06`
  - **Statut :** Remplacé → `ARB-G-07`
  - **Décision consolidée :** Pas de fonction libre parallèle de dispersion.

- **ID :** `ARB-L13-07`
  - **Statut :** Acté
  - **Décision consolidée :** `anchor_signature`/ancienne confiance migrent vers identity/`anchor_quality`.

- **ID :** `ARB-L13-08`
  - **Statut :** Acté
  - **Décision consolidée :** `stats` migre vers lifecycle, supprimer `density_fps`; compteurs slow/fast exposés dans lifecycle.

- **ID :** `ARB-L13-09`
  - **Statut :** Acté
  - **Décision consolidée :** Finalisation slow une émission unique; fast retire/compte sans Sighting; compile idempotent; `promo_outcome/frame` dans identity, résultats gelés au retrait.

- **ID :** `ARB-L13b-01`
  - **Statut :** Acté
  - **Décision consolidée :** Ajouter `SRC_OBS_REJECT_MISSING_GEOMETRY` dans `counters.py`; supprimer clé morte `SRC_OBS_SLOW_TS_FALLBACK_ZERO` et toute lecture correspondante dans `stream_eid_builder.py`.

- **ID :** `ARB-L13b-02`
  - **Statut :** Acté
  - **Décision consolidée :** `flush()` affecte le résultat de `_fast_pass(batch)` à `self.counters`; horloge d’éviction monotone mise à jour par `self._clock_ts = max(self._clock_ts, batch[-1].ts)`, sans jamais reculer.

- **ID :** `ARB-L13b-03`
  - **Statut :** Acté
  - **Décision consolidée :** `AnchorManager.try_update()` traite `dist_res.total is None` comme `SRC_NO_COMMON_CHANNEL` et ne lève pas `TypeError`.

- **ID :** `ARB-L13b-04`
  - **Statut :** Acté
  - **Décision consolidée :** Purger TTL de `DedupIterator._seen` avant le test de doublon anticipé, pour qu’une clé expirée puisse repasser. Implémentation vérifiée dans le module `batching.py` (classe actuelle `DedupSource`); si `DedupIterator` n’existe pas, le nom dans l’arbitrage est sémantique et le symbole réel est à conserver tant qu’aucun renommage décidé. L’horloge monotone est spécifiée séparément par `ARB-L13b-02` : `self._clock_ts = max(self._clock_ts, batch[-1].ts)`.

- **ID :** `ARB-L14-01`
  - **Statut :** Acté
  - **Décision consolidée :** Toute observation valide passe en Passe 1 (`ClusterFast`) avant Passe 2 slow; association Hungarian 1-à-1 par tranche, fast puis slow.

- **ID :** `ARB-L14-02`
  - **Statut :** Acté
  - **Décision consolidée :** En Passe 1, filtre `abs(dt) <= fast_assoc_window_s`, puis Gate D2 `_geo_compatible` avec `abs(dt)`; coût géométrique normalisé `dist_center/d_max`, incompatible interdit. En Passe 2, filtres temporel slow, spatial D2, continuité, puis signature dans cet ordre.

- **ID :** `ARB-L14-03`
  - **Statut :** Acté
  - **Décision consolidée :** Aucune identité/EID slow n’est héritée du fast. Chaque observation obtient `obs_fast_id` en Passe 1; ce lien alimente seulement la continuité via `ClusterSlow.last_obs_fast_id` en Passe 2.

- **ID :** `ARB-L14-04`
  - **Statut :** Acté — **garde slow dédiée verrouillée (ancienne étiquette Option 2A)**
  - **Décision consolidée :** En Passe 2, égalité de deux ID fast non-None = `CONTINUITY_PASS`; différence = `CONTINUITY_FAIL` éliminatoire avant signature; ID manquant = `CONTINUITY_NA`, repli défensif qui poursuit avec Gate D2 + signature. `dt` slow doit satisfaire `0.0 <= dt <= slow_assoc_window_s`; les doublons grace-period ne sont pas réingérés (dédup ou timestamp déjà ingéré).

- **ID :** `ARB-L14-05`
  - **Statut :** Acté
  - **Décision consolidée :** Garder distincts `slow_assoc_window_s=1.5`, `slow_idle_close_s=1.5`, `slow_max_lifetime_s=10.0`, `fast_assoc_window_s=0.25`, `fast_idle_close_s=0.5`, `fast_max_lifetime_s=5.0`, `epsilon_close_s=1e-3`, `geo_vmax_px_s=2000.0`, `geo_y_axis_ratio=0.6`, `geo_time_relax_factor=0.5`, `geo_iou_min=0.05`; vieux champs retirés et dataclass échoue sans alias.

- **ID :** `ARB-L14-06`
  - **Statut :** Acté
  - **Décision consolidée :** Max lifetime ferme avant affectation, puis observation ouvre nouveau cluster sans chaînage.

- **ID :** `ARB-L14-07`
  - **Statut :** Acté
  - **Décision consolidée :** Continuité différente (`CONTINUITY_FAIL`) élimine; ID manquant (`CONTINUITY_NA`) est le repli défensif autorisé à poursuivre Gate D2 + signature, jamais une preuve positive par lui-même.

- **ID :** `ARB-L14-08`
  - **Statut :** Acté
  - **Décision consolidée :** Une piste fast éteinte ne peut plus être référencée comme continuité positive; si l’ID est absent, traiter `CONTINUITY_NA` selon le repli strict D2 + signature; priorité précision, fragmentation rattrapable par AM.

- **ID :** `ARB-L14-09`
  - **Statut :** Acté — affectation fast; **Option 1A verrouillée** pour le snapshot géométrique slow
  - **Décision consolidée :** Passe 1 universelle et ancrage monotone fast : toutes observations valides sont associées/ouvertes en `ClusterFast`; timestamp non ancien avance `last_obs`, timestamp ancien s’attache sans recul. Gate anisotrope D2 ou seuil IoU et coût `dist_center/d_max`. `GeometrySnapshot` et son ancrage dans `ClusterSlow` suivent §3.2; `AnchorManager` reste découplé de la géométrie.

- **ID :** `ARB-L14-12`
  - **Statut :** Acté
  - **Décision consolidée :** Gate spatial `_geo_compatible`: anisotropie `vx <= geo_vmax_px_s*(1+abs(dt)*geo_time_relax_factor)`, `vy <= vx*geo_y_axis_ratio`, OU `iou >= geo_iou_min`; paramètres spatiaux désactivables selon leur contrat, prédicat distinct du calcul de signature.

- **ID :** `ARB-L14-13`
  - **Statut :** Acté, numérique en L24
  - **Décision consolidée :** Garde d’association slow `slow_assoc_window_s=1.5 s`, séparée de `slow_idle_close_s=1.5 s`; refus temps avant espace, puis gate spatial.

- **ID :** `ARB-L14-14`
  - **Statut :** Acté, valeurs L24
  - **Décision consolidée :** Fermetures idle/max-life distinctes fast et slow, TTL global garde-fou; paramètres et valeurs par défaut explicites en L14.

- **ID :** `ARB-L14-17`
  - **Statut :** Acté
  - **Décision consolidée :** Geometry exporte les comptes par cluster selon §3.2; les distributions `txt_scale` sont des statistiques exactes sur toutes les observations slow du cluster, sans troncature ni interpolation.

- **ID :** `ARB-L14-18`
  - **Statut :** Acté
  - **Décision consolidée :** Fenêtre slow `slow_assoc_window_s=1.5 s` et inactivité `slow_idle_close_s=1.5 s` sont deux paramètres distincts; valeurs fixées par L14.

- **ID :** `ARB-L15-01`
  - **Statut :** Acté
  - **Décision consolidée :** Dépendance historique L15 à L06/L07.

- **ID :** `ARB-L15-02`
  - **Statut :** Acté
  - **Décision consolidée :** Noyau quantile pur `quantile_dispersion`; métriques riches uniques portées par L19.

- **ID :** `ARB-L15-03`
  - **Statut :** Acté
  - **Décision consolidée :** Repli quantile vide/insuffisant tel que §1.3; compteur `dispersion_fallback_max` seulement si fallback.

- **ID :** `ARB-L15-04`
  - **Statut :** Acté
  - **Décision consolidée :** Chemin source ne rend pas la géométrie de canal signature obligatoire; entrée bbox est exigée selon §3.1.

- **ID :** `ARB-L15-05`
  - **Statut :** Acté
  - **Décision consolidée :** `geom_ok` séparé de `cost_sig`; aucun poids géométrique; retirer `w_sig`.

- **ID :** `ARB-L15-06`
  - **Statut :** Acté
  - **Décision consolidée :** `cost_sig=clip(sd.cost,0,1)`; exception locale, ne modifie pas positivité des distances brutes.

- **ID :** `ARB-L15-07`
  - **Statut :** Acté
  - **Décision consolidée :** `w_sig` retiré; `w_geo` et autres poids obsolètes retirés L14.

- **ID :** `ARB-L15-08`
  - **Statut :** Remplacé
  - **Décision consolidée :** Rejet gate géométrique est `GEO_GATE_FAIL`/motif normalisé, pas ancien marqueur.

- **ID :** `ARB-L15-09`
  - **Statut :** Acté
  - **Décision consolidée :** Dispersion finale exposée; détails et populations restent définis L19.

- **ID :** `ARB-L15-10`
  - **Statut :** Acté
  - **Décision consolidée :** Gate spatial sur dernière observation, pas ancre.

- **ID :** `ARB-L15-11`
  - **Statut :** Remplacé
  - **Décision consolidée :** Fast un-à-un par construction; ne pas créer compteur doublon same-frame.

- **ID :** `ARB-L15-12`
  - **Statut :** Remplacé → D1.b
  - **Décision consolidée :** Marges selon D1.b.

- **ID :** `ARB-L15-13`
  - **Statut :** Acté
  - **Décision consolidée :** Tranche=frame, lot=unité de lecture; garde frame_id.

- **ID :** `ARB-L15-14`
  - **Statut :** Acté
  - **Décision consolidée :** Ordre frames strict au BatchReader, `batch_frame_out_of_order`.

- **ID :** `ARB-L15-15`
  - **Statut :** Acté
  - **Décision consolidée :** Marges d’ambiguïté selon D1.b, compteurs disjoints.

- **ID :** `ARB-L15-16`
  - **Statut :** Acté
  - **Décision consolidée :** Marges déterministes avant Hungarian, ligne avant colonne.

- **ID :** `ARB-L15-17`
  - **Statut :** Acté
  - **Décision consolidée :** Ordre unique d’association défini D1.

- **ID :** `ARB-L15-18`
  - **Statut :** Acté
  - **Décision consolidée :** `frame_id` absent/None rejeté avant lots et compteur source exact à aligner; pas de valeur négative/non entière par construction.

- **ID :** `ARB-L15-19`
  - **Statut :** Acté
  - **Décision consolidée :** Frame négatif/non entier impossible par construction.

- **ID :** `ARB-L15-20`
  - **Statut :** Acté
  - **Décision consolidée :** Renommer `COST_REJECT_SLOW` en `admit_reject_cost` sans alias; appelants/rapports/tests migrés.

- **ID :** `ARB-L15-21`
  - **Statut :** Acté
  - **Décision consolidée :** Rejets de marge ouverts aux seules paires admises; échec ouvre cluster.

- **ID :** `ARB-L16-01`
  - **Statut :** Acté
  - **Décision consolidée :** Fermeture nominale au slow, timeout `epsilon_close_s` sécurité, valeur L24.

- **ID :** `ARB-L16-02`
  - **Statut :** Acté
  - **Décision consolidée :** `BatchReader(source,config,counters)`, sources bump direct `SRC_*`.

- **ID :** `ARB-L16-03`
  - **Statut :** Acté
  - **Décision consolidée :** Une fast par frame s’entend par joueur; horloge / timeout et lot ne finalisent pas.

- **ID :** `ARB-L16-04`
  - **Statut :** Acté
  - **Décision consolidée :** `BatchReader` seul voit ordre des frames et porte les comptes de lecture.

- **ID :** `ARB-L19-01`
  - **Statut :** Acté
  - **Décision consolidée :** L19 seul lot des métriques détaillées; n_min diagnostic séparé du seuil global.

- **ID :** `ARB-L20-01`
  - **Statut :** Acté
  - **Décision consolidée :** L20 porte signal, qualité intrinsèque `f_hs`, mesures finales de canal.

- **ID :** `ARB-L20-02`
  - **Statut :** Acté
  - **Décision consolidée :** Contrat `f_hs` avec domaine/bornes/fallback défavorable; formule et `hs_px_ref` L24.

- **ID :** `ARB-L21-01`
  - **Statut :** Acté; volet tie-break remplacé par L14
  - **Décision consolidée :** Thermomètres sans purge; sampling lecture seule; pas de règle dure de capacité. Le rattachement Fast→Slow est défini en deux passes par L14; aucun tie-break de cascade additionnel.

- **ID :** `ARB-L21-02`
  - **Statut :** Remplacé par L14
  - **Décision consolidée :** `slow_ambiguity_margin` n’est pas utilisé dans l’architecture arrêtée en L14; filtre de continuité puis signature et Hungarian 1-à-1 définissent l’association.

- **ID :** `ARB-L22-01`
  - **Statut :** Acté
  - **Décision consolidée :** `LiveSource`/`JsonlSource`, surcharge et perte, deux horloges, espaces d’identifiants, parité native↔JSONL; preuve test L22.

- **ID :** `ARB-L23-01`
  - **Statut :** Acté
  - **Décision consolidée :** `ObservationSource` ABC, propriété `counters` obligatoire.

- **ID :** `ARB-L23-02`
  - **Statut :** Acté
  - **Décision consolidée :** Registre typed `CounterKey`/`GaugeKey`, noms source exacts, `SCHEMA_VERSION` figée et tests sur les clés.

- **ID :** `ARB-L23-03`
  - **Statut :** Acté
  - **Décision consolidée :** Renommer réellement clés source en `src_obs_*`/`src_*`, pas d’alias.

- **ID :** `ARB-L23-04`
  - **Statut :** Acté
  - **Décision consolidée :** `schema_version`/`mode` racine de l’enveloppe JSONL; `mode` paramètre/session `CalibrationWriter`. Pour la sortie Sighting, se référer à §3.2.

- **ID :** `ARB-L23-05`
  - **Statut :** Acté
  - **Décision consolidée :** Grep des motifs périmés (`sample_active_clusters` quand ancien usage documentaire, helper dispersion, anciens TTL, probe), sans supprimer le symbole fonctionnel actif `Counters.sample_active_clusters`.

- **ID :** `ARB-L23-08`
  - **Statut :** Acté
  - **Décision consolidée :** `mode` n’est pas un champ `Sighting`; paramètre d’export/session.

- **ID :** `ARB-L23-09`
  - **Statut :** Acté
  - **Décision consolidée :** Grep et suppression des références périmées `_dispersion_for_family_e`, ancien TTL, `_CALIB_PROBE`.

- **ID :** `ARB-L23-11`
  - **Statut :** Acté
  - **Décision consolidée :** Ordre L03/L06; si interaction de renommage, reporter la finition en lot qui consomme ces clés, sans incohérence.

- **ID :** `ARB-L23-12`
  - **Statut :** Acté
  - **Décision consolidée :** Producteur signatures `core.signature`; `AnchorManager` `eid.signature_ref`; distance `eid.eid_distance`.

- **ID :** `ARB-L23-13`
  - **Statut :** Acté
  - **Décision consolidée :** Clés typées dans lot dédié, réalisé par registre `CounterKey`/`GaugeKey` ; contrôle déclaré/écrit/lu.

- **ID :** `ARB-L23-14`
  - **Statut :** Acté
  - **Décision consolidée :** Optimisations au nettoyage, calibration conserve périmètre distinct; retire/déprécie `batch_max_span_factor` inutile puisque BatchReader coupe à `obs.ts-batch[0].ts > window_s`.

- **ID :** `ARB-L23-15`
  - **Statut :** Acté
  - **Décision consolidée :** Capturer coût ayant bloqué une promotion depuis `signature_ref.py`, diagnostics/compteurs seulement; champs réservés null en v1, jamais dans Sighting (`INV-L23-01`).

## 5. Feuille de route et graphe d’exécution (L01–L13 -> L13b -> L14–L24)

### 5.1 Tableau synoptique des lots

| Lot      | Statut      | Objet/critère synthétique                                                           |
| -------- | ----------- | ----------------------------------------------------------------------------------- |
| L01      | Terminé     | Socle stats/dispersion, compteurs/jauges consolidés; variante morte supprimée L17.  |
| L02      | Terminé     | `dedup_ttl_s`, invariant de nommage.                                                |
| L03      | Terminé     | Compteurs sources directs, validation bbox, optionnels.                             |
| L04      | Terminé     | Flush final idempotent, slow puis fast, jauges à zéro, aucun timestamp factice.     |
| L05      | Terminé     | Enums clés typées et compteurs lecture.                                             |
| L06      | Terminé     | Sources JSONL/dedup, séquence dense, TTL config, purge amortie.                     |
| L07      | Terminé     | Rapport/jauges et clé CounterKey; symétrie vérifiée L15.                            |
| L08      | Terminé     | Renommage de concepts `footprint` → `sighting`.                                     |
| L09      | Terminé     | Sighting frozen, slots, `to_json`, immuabilité.                                     |
| L10      | Terminé     | `SigDistance`, gates, retrait calibration probe; dette gate géométrique L14.        |
| L11      | Terminé     | Reclassements structurels historiques.                                              |
| L12      | Terminé     | `AnchorState`, `QualityScore`, promotion, signature vide.                           |
| L13      | Terminé     | `ClusterState.ACTIVE` seul, finalisation centralisée, identity partielle.           |
| **L13b** | **À faire** | 4 correctifs post-audit précisément listés §7.1; prérequis immédiat à L14.          |
| L14      | À faire     | Deux passes `ClusterFast`→`ClusterSlow`, géométrie exacte, `obs_fast_id`, `n_fast`. |
| L15      | À faire     | Admission, ambiguïté, dispersion de base, symétrie/signature.                       |
| L16      | À faire     | Batch fermant sur slow et timeout de sécurité.                                      |
| L17      | À faire     | Signatures partielles, ancre recalculée, dispersion slow exacte sans troncature.    |
| L18      | À faire     | Signal agrégé et complétion de counters lifecycle/tracking.                         |
| L19      | À faire     | Dispersions d’admission/finales riches et `n_slow/n_fast`.                          |
| L20      | À faire     | Signal `f_hs`, interface et champs signal.                                          |
| L21      | À faire     | Observabilité/performance sans purge; comptage fast rattaché aux slow.              |
| L22      | À faire     | Contrat strict (`state != LOST`), flux live via mêmes deux passes, parité JSONL.    |
| L23      | À faire     | Alignement config/counters, suppression dette, optimisations et tests globaux.      |
| L24      | À faire     | Calibration finale, données réelles, valeurs/paramètres prévus.                     |

### 5.2 Ordre d’exécution / dépendances

Ordre impératif : **L13b → L14 → L15 → L17 → L19 → L18 → L20 → L21 → L22 → L23 → L24** ; L16 est exécuté après L14 et avant la recette des lots dépendants. Dépendances croisées : L17 exige L10/L15 ; L18 exige complétion `n_slow/n_fast` L19 ; L20 exige L17/L19 ; L21 exige L14/L15/L16 ; L22 traite le flux et doit précéder l’alignement global ; L23 après toutes les consolidations ; L24 après L23. Tous L01–L13 sont statut « Terminé » du plan historique, même si les audits de conformité identifient encore les corrections L13b et les écarts visibles dans les 13 fichiers fournis.

## 6. Référentiel du socle implémenté (L01 à L13 — Aligné code réel)

Chaque lot ci-dessous est compact; décisions et invariants sont les identifiants du §2.2/§4. Le contenu décrit le socle historique réellement présent, pas une supposition que le contrat cible entier soit déjà livré.

- **L01 — Dispersion et compteurs de session [Terminé]** — `Counters` expose `as_dict()`, `gauges()`, `stats()`, `items()`, `anomalies()` et `sample_active_clusters`; compteurs/jauges distincts, ne pas additionner les jauges. `quantile_dispersion` est la règle de calcul de référence apportée plus tard en L15; supprimer le helper mort de `signature_ref.py` en L17.
- **L02 — TTL de dédup [Terminé]** — `StreamConfig.dedup_ttl_s` est le seul nom actif; `DedupSource` filtre les clés, tient son état `_seen`, incrémente les compteurs de maintien/rejet/purge. `[Corrigé par L13b]` expiration avant test de doublon.
- **L03 — Stats de source [Terminé]** — `Observation.from_entry(entry,counters)` valide `last_source`, frame, géométrie et rectangle, désérialise signature, établit horodatage et dedup*key, convertit `confidence/vx/vy` facultatifs; compteurs `SRC*\*`. `[Corrigé par L13b]` registre inclut la géométrie manquante et aucune clé fallback-zero morte.
- **L04 — Flush/idempotence [Terminé]** — `StreamEIDBuilder.flush()` finalise slow puis fast, sans timestamp fictif, ne double-purge pas au travers d’un `finally`; flush répété sans clusters ne réémet rien. `[Corrigé par L13b]` flush doit conserver le retour/état de `_fast_pass(batch)` selon §7.1.
- **L05 — Clés typées [Terminé]** — `CounterKey`, `GaugeKey` StrEnum, valeurs disjointes; `Counters.bump()`/`_set_gauge()` vérifient les types, vues retournent chaînes simples; `BatchReader` comptabilise lectures et maximum de taille.
- **L06 — Sources et batching [Terminé]** — `ObservationSource`, `JsonlSource`, `DedupSource`, `BatchReader`; `Counters` injecté; source JSONL itère entries; âge dédup testé à la recherche et purge amortie. `[Corrigé par L13b]` purge TTL doit précéder early duplicate check. La stricte obligation `StreamConfig` de dédup prévaut sur signature historique encore `Optional` dans le code fourni. L’horloge d’éviction demeure monotone : `self._clock_ts = max(self._clock_ts, batch[-1].ts)` (`ARB-L13b-02`).
- **L07 — Rapport [Terminé]** — rapport d’exécution est dans `stream_eid_builder.main`, pas dans un fichier autonome; `CalibrationWriter` dans `calibration.py`; lecture clés via types enregistrés; liste close de compteurs, pas de clé manquante silencieuse.
- **L08 — Nommage [Terminé]** — `Sighting`, `prime_sighting`, `compile_prime_sighting`, `_eid_sightings.jsonl` remplacent l’ancien vocabulaire identifiant `footprint`; le mot « Empreinte » demeure pour AM.
- **L09 — Objet immuable [Terminé]** — `sighting.Sighting` dataclass frozen/slots, enveloppe EID/frame/temps/n_obs, dictionnaires gelés récursivement via `MappingProxyType` et tuples, `to_json()` restitue des dictionnaires/listes JSON; propriétés `span_frames`, `span_time_s`. Cible enrichie §3.2.
- **L10 — Distance/signature [Terminé]** — `eid_distance.SigDistance` et `sig_distance`; `builder._eval_slow_pair` et `cost.build_slow_cost_matrix`; compte SIG par orchestrateur `_count_sig`; fonctions signature sous `signature.py`; `CalibrationWriter` porte schema/mode; `CostMatrix.payload` sans géométrie. `[Supplanté par L14 (§3.2)]` court-circuit `geo_shortcircuit`, anciennes distances/poids géométriques remplacés par gates séparés. `[Corrigé par L13b]` distance sans canal partagé doit être reconnue sans crash.
- **L11 — Reclassement [Terminé]** — `identity.anchor_signature`, lifecycle reconstruit par `Sighting.to_json()`, plus `density_fps` ni `tracking.stats`; conserve placeholders null jusqu’aux lots de remplissage. `n_slow/n_fast` ne font pas partie du stade L11.
- **L12 — Ancre et QualityScore [Terminé]** — `signature_ref.QualityScore` frozen/slots; `AnchorState`, `PromoOutcome`, `PromotionResult`, `AnchorManager`; seul total compare; signature vide exclue; compteurs promotion/blocage; `promo_outcome` final valide obligatoire. `[Supplanté par L16]` expose le coût de blocage; `[Supplanté par L17]` le score de référence final recalculé; `[Supplanté par L24]` formule/valeurs calibrées et quatre composantes calculées ensemble.
- **L13 — Finalisation [Terminé]** — `ClusterState` contient `ACTIVE`; `ClusterSlow.compile_prime_sighting()` construit/cache et retourne même objet; `StreamEIDBuilder._finalize_slow()` retire/finalise/compte/émet une fois, `_finalize_fast()` retire sans Sighting; `identity` rempli en partie. `[Supplanté par L14 (§3.2)]` géométrie placeholders et contrat enrichi; `[Supplanté par L15 (§3.2)]` disp/tracking; `[Corrigé par L13b]` quatre erreurs post-audit inscrites §4.3.

## 7. Lots de consolidation du socle et complétion Sighting (L13b, L14 à L19)

### 7.1 L13b — Correctifs post-audit du socle L01–L13

**Statut : À faire — lot de correction avant toute intervention L14.** Périmètre strict : `counters.py`, `builder.py`, `signature_ref.py`, `batching.py`, et le point d’entrée pour enlever une référence de compteur supprimée. Ne change aucun invariant architectural, aucune politique de matching ni aucune forme de contrat finale. Arbitrages `ARB-L13b-01` à `ARB-L13b-04` en §4.3.

1. **`ARB-L13b-01` — Registre source :** déclarer `CounterKey.SRC_OBS_REJECT_MISSING_GEOMETRY` utilisé par `Observation.from_entry`; supprimer `CounterKey.SRC_OBS_SLOW_TS_FALLBACK_ZERO` mort de `counters.py`, son entrée de rapport/lecture dans `stream_eid_builder.py` et son commentaire. Contrôler tous les lecteurs/rapport. Le compteur de géométrie manquante n’est pas le compteur de box dégénérée.
2. **`ARB-L13b-02` — flush et horloge :** `EIDBuilder.flush()` (classe effective `StreamEIDBuilder`) doit mettre à jour `self.counters = self._fast_pass(batch)` dans `builder.py`; protéger la base d’horloge d’éviction contre les tranches hors ordre avec l’expression exacte `self._clock_ts = max(self._clock_ts, batch[-1].ts)`. Ne jamais reculer cette horloge. Vérifier flush lent puis rapide, un seul retrait et jauges à zéro.
3. **`ARB-L13b-03` — distance sans canal :** `AnchorManager.try_update()` dans `signature_ref.py` doit traiter `dist_res.total is None` comme absence de canal commun et classer `SRC_NO_COMMON_CHANNEL` (ou état diagnostique équivalent prévu dans l’API), sans comparaison numérique `None` vs float ni `TypeError`; conservé dans `PromotionResult`/diagnostics, aucun coût fictif favorable.
4. **`ARB-L13b-04` — expiration dédup :** purger TTL de `DedupIterator._seen` avant le early duplicate check dans `batching.py`, de sorte qu’une clé expirée n’est pas rejetée comme doublon et puisse se réinsérer avec timestamp frais. Le dépôt fourni nomme la classe `DedupSource` (pas `DedupIterator`); modifier l’implémentation réelle sans inventer une seconde source.

**Tests/sortie :** test compteur de géométrie déclaré, écrit et reporté; grep de fallback-zero = 0 hors historique des décisions; flush met à jour état/compteurs; temps d’éviction monotone en arrivée retardée; total nul sans TypeError; clé dédup périmée puis répétée conservée; tests existants L04/L06 non régressés. Préserver T-L08-02 (Sighting frozen), T-L08-05 (grep renommage sur périmètre), T-L09-01 (mutation de source après publication sans effet), T-L09-02 (mutation du Sighting publié rejetée), T-L13-01 (émission/compilation/promo), T-L14-01 à T-L14-08 (conditions détaillées en §7.2), T-L15-01…05, T-L16-01…03, T-L21-01…03, T-L22-01, T-L24-01. L13b est le prérequis immédiat et bloquant de L14.

### 7.2 L14 — Sighting.geometry complet et architecture deux passes Fast→Slow

**Statut : À faire.** Prérequis L13b et L13. Périmètre réel : `observation.py`, `builder.py`, `cost.py`, `config.py`, `cluster.py`, `counters.py`, `eid_distance.py`, `sighting.py`; `signature_ref.py` reste découplé de la géométrie. Règle de sortie unique §3.2. `Observation.from_entry` accepte `CONFIRMED`, `PENDING` ou `state` absent; rejette uniquement `entry.get("state") == "LOST"`. Garder la protection `dedup_key`. Une observation slow redéversée en grace period sans avancement de `last_detected_ts` est naturellement non nouvelle : dédup la filtre; si déjà intégrée, même `last_detected_ts` (et rect) ou `dt==0.0` n’ajoute pas un second échantillon. En traitement live/batch, toute slow dont l’horodatage est antérieur ou égal à celui de la détection déjà ingérée n’est jamais ajoutée deux fois.

**EIDConfig** garde distincts les paramètres et valeurs de référence : `slow_assoc_window_s: float = 1.5`, `slow_idle_close_s: float = 1.5`, `slow_max_lifetime_s: float = 10.0`, `fast_assoc_window_s: float = 0.25`, `fast_idle_close_s: float = 0.5`, `fast_max_lifetime_s: float = 5.0`, `epsilon_close_s: float = 1e-3`, `geo_vmax_px_s: float = 2000.0`, `geo_y_axis_ratio: float = 0.6`, `geo_time_relax_factor: float = 0.5`, `geo_iou_min: float = 0.05`. Elles restent distinctes (ne pas fusionner `slow_idle_close_s` et `slow_assoc_window_s`) et validées avec les autres inégalités `INV-L14-03`.

**Passe 1 — `ClusterFast`, piste universelle, ancrage monotone (Approche 1 + Option 1 hybride).** Chaque observation valide, fast ou slow, est candidate à l’association aux `ClusterFast` actifs dans sa tranche; Hungarian `scipy.optimize.linear_sum_assignment`, 1-à-1. Les filtres sont `abs(obs.ts-cluster_fast.last_obs.ts) <= fast_assoc_window_s` (dt peut être légèrement négatif pour la latence slow de 1–2 frames), puis `_geo_compatible(cluster_fast.last_obs, obs, abs(dt), cfg)`. Gate anisotrope D2 : `vx <= geo_vmax_px_s*(1+abs(dt)*geo_time_relax_factor)` et `vy <= vx*geo_y_axis_ratio`, OU `iou >= geo_iou_min`; coût géométrique centre normalisé `dist_center/d_max`; paire incompatible disallow/`+inf`. Match : `obs_fast_id=cluster_fast.cluster_id`. Si `obs.ts >= cluster_fast.last_obs.ts`, appeler `cluster_fast.add(obs)` et avancer la tête; si `obs.ts < last_obs.ts`, attacher l’ID sans écraser la tête par une géométrie/horodatage ancien. Toute observation non appariée ouvre un `ClusterFast` et reçoit immédiatement `obs_fast_id`; ainsi toute observation valide a un `track_id` dès sa première frame. Ne pas confondre cette référence de continuité avec une identité/EID transmise au slow.

**Passe 2 — création/association `ClusterSlow` (source slow seulement).** Avant l’association, évincer les `ClusterSlow` si `ts-last_obs.ts > slow_idle_close_s` ou `ts-first_obs.ts > slow_max_lifetime_s`, et `ClusterFast` selon `fast_idle_close_s`/`fast_max_lifetime_s`; max-life ferme avant affectation, sans chaînage. Construire la matrice $M-imes N$ entre $M$ slow actifs et $N$ observations slow de la tranche, selon ces quatre filtres séquentiels :

1. **Temporel :** `0.0 <= dt <= slow_assoc_window_s`, `dt=obs_slow.ts-cluster_slow.last_obs.ts`. Une même détection slow déjà ingérée n’est pas réajoutée : conserver `dedup_key`; au redump grace-period, même `last_detected_ts` et rect, ou horodatage déjà atteint avec `dt==0.0`, évite l’entrée slow dupliquée.
2. **Spatial — Gate D2 :** `_geo_compatible(cluster_slow.last_obs, obs_slow, dt, cfg)`. Faux → rejet `SRC_SLOW_GEO_REJECT`, coût `+inf`, sans évaluer la signature.
3. **Continuité cinématique fast (`ARB-L14-04`) :** comparer `cluster_slow.last_obs_fast_id` et `obs_fast_id` attribué en Passe 1. Deux IDs non-None égaux = `CONTINUITY_PASS`; IDs non-None différents = `CONTINUITY_FAIL`, rejet avant signature; l’un des IDs None = `CONTINUITY_NA`, repli défensif (Passe 1 désactivée ou test synthétique appelant Passe 2 seule), ne rejette pas et passe au filtre signature.
4. **Signature et coût Hongrois (`ARB-L14-02`) :** pour spatial pass + continuité pass/na, calculer `sim=compute_pair_similarity(cluster_slow.last_obs, obs_slow, cfg)`. `sim < cfg.sim_threshold` → `SRC_SLOW_LOW_SIM`, coût `+inf`; sinon candidat valide, coût `1.0-sim`.

Résolution Hungarian 1-à-1; ne garder que paires attribuées de coût fini (`sim >= cfg.sim_threshold`). Slow appariée : `cluster_slow.add(obs_slow, fast_cluster_id=obs_fast_id)`, mise à jour de `last_obs` et `last_obs_fast_id`. Slow non appariée : créer `ClusterSlow(first_obs=last_obs=obs_slow, last_obs_fast_id=obs_fast_id)`. `n_fast` est compté sur les observations fast de la piste et attribué/agrégé au `ClusterSlow` correspondant pour le lifecycle; ce transfert de compteurs n’est pas un transfert d’identité.

**GeometrySnapshot / `geometry.txt_scale` (`ARB-L14-03`).** `GeometrySnapshot` `@dataclass(frozen=True)` comporte `x: float,y: float,w: float,h: float,ts: float,frame: int`, `to_dict()` retourne exactement `{"x": float, "y": float, "w": float, "h": float, "ts": float, "frame": int}`, `from_obs(obs)` construit le snapshot, propriétés `cx=x+w*0.5`, `cy=y+h*0.5`. Valeurs `first/last/anchor` selon §3.2; Option 1A : chaque `ClusterSlow` maintient son snapshot d’ancre, mis à jour uniquement si l’ancre est réellement acceptée. Pour chaque slow : `w=float(obs.rect[2])`, `h=float(obs.rect[3])`, `h_txt=float(gates["h_txt"])` seulement si valeur numérique valide et positive, sinon `h`; `ar=w/h`. Calcul exact sans interpolation de médiane et IQR `q75-q25` pour toutes les slow observées : `h_txt_med`, `h_txt_iqr`, `w_med`, `w_iqr`, `ar_med`, `ar_iqr`. Garder l’historique léger complet `(h_txt,w,ar)` de la durée de vie du `ClusterSlow`; ne pas tronquer avec `deque(maxlen=50)`. Clé exacte `w_med` (corriger `w_Med`); aucun `n_geo_*` ni `SignalFlags`.

Maintenir `geometry={first,last,anchor,txt_scale,spatial,temporal,continuity,reject_total}` et l’invariant `reject_total=spatial.reject+temporal.reject_dt`. Gates géométriques avant signature; tranche tardive ne déclenche pas d’éviction; les compteurs géométriques/continuité et fermetures sont déclarés, écrits et reportés. Tests `T-L14-01` à `T-L14-08` : (01) Hungarian fast 1-à-1 et ID dès première observation; (02) tranche retardée sans éviction, `SLICE_LATE`, dt négatif permis fast et ancrage monotone; (03) max-life avant affectation, ancien fermé/nouveau ID sans chaînage; (04) `CONTINUITY_PASS/FAIL/NA` et premier motif; (05) filtres temporels fast/slow et slow `dt` hors `[0,slow_assoc_window_s]` refusé avant spatial; (06) Gate D2 anisotrope/IoU et refus spatial avant signature; (07) `reject_total` à chaque émission et stats exactes géométriques sur plus de 50 slow; (08) paramètres configurés distincts, valeurs/inégalités valides et clés anciennes rejetées.

### 7.3 L15 — Admission, tracking complet, bornes et dispersion de base

**Statut : À faire.** Prérequis L06/L07/L14. Implémenter ordre strict D1 et première cause unique; gates signature, coût admissible, ambiguïté ligne/colonne, Hungarian puis mise à jour provisoire et contrôle dispersion. Compléter le bloc `tracking` selon §3.2, avec `mask_uids_seen`, `mask_uid_switches`, `uid_agree`, `uid_disagree`; `mask_uid_switches` existe uniquement sous `tracking`; `mask_uid` n’influe sur aucune décision (`INV-L15-04`). Noyau `quantile_dispersion(costs,q,n_min)` pure par famille avec retours définis §1.3; `dispersion_fallback_max` incrémenté au point unique si fallback. `None` ne ferme ni n’est favorable. Exposer `dispersion.final` dans §3.2; seuils numériques L24. Retirer `COST_REJECT_SLOW` et remplacer par `admit_reject_cost` (pas d’alias); compteurs `admit_reject_margin_row/col`; pas de compteur `admit_reject_same_frame_fast` (fast un-à-un). Frame absent → rejet avant lots et `src_obs_missing_frame_id`; ordre strict lecteur `batch_frame_out_of_order`, tranche tardive séparée `SLICE_LATE`. Sources des distances dans le domaine `[0.0, +inf[` et aucune coupure à 1.0; seul le coût transformé `cost_sig` est clampé `[0,1]`. Test exact symétrie `INV-L15-SYM`, y compris `NO_SIG_EQUIVALENT` depuis géométrie absente. Tests T-L15-01 premier rejet; T-L15-02 marges; T-L15-03 quantiles/fallback; T-L15-04 mask_uid non décisionnel; T-L15-05 frame absent.

### 7.4 L16 — Sighting.identity complet et coût de blocage de promotion

**Statut : À faire.** Compléter `identity` exclusivement selon §3.2 : `anchor_signature`, `anchor_locked`, `anchor_quality`, quatre `anchor_quality_components`, `promo_outcome`, `promo_frame`, `n_post_promo_outliers`, `n_promo_blocked_identity`, `promo_blocked_max_cost`. Aucun `mask_uid_switches` dans `identity` : sa seule localisation est `tracking` (§3.2). Mettre à jour `AnchorManager.try_update()` dans `signature_ref.py` pour exposer `last_blocked_cost` (ou retourner le coût bloqué dans `PromotionResult`) afin que `ClusterSlow` mette à jour `self.promo_blocked_max_cost = max(self.promo_blocked_max_cost, blocked_cost)`. Un blocked cost nul/inconnu reste absent, jamais inventé; coût diagnostique uniquement et pas sérialisé (`INV-L23-01`). Tests : blocages identité successifs gardent maximum, `try_update` reflète le coût exact du blocage, promotion réussie n’est pas comptée blocked, pas de clé tracking en identity, champs quality null/all four coherent, sortie finalisée `promo_outcome` valide.

### 7.5 L17 — Signatures partielles et dispersion finale de la promotion

**Statut : À faire.** Distinguer aucun descripteur de nombre insuffisant de familles (`min_cost_families=2` par défaut). Ordre d’admission du D6 et compteurs `admit_reject_families` depuis builder; une signature vide ne devient jamais une ancre et ne crée pas de rejet slow additionnel. Appliquer noyau `quantile_dispersion`; ne réintroduire aucune ancienne variante.

**Sous-option validée 2A-1 :** recalcul exact à la finalisation `ClusterSlow` dans `compile_prime_sighting()` par rapport à l’ancre définitive `self.anchor_mgr.sig_ref` (nom à adapter au symbole effectif `self.anchor` / `AnchorManager.anchor` du code fourni). Aucun échantillon de mesure utilisé pour ce calcul exact ne peut être perdu par un buffer borné; préserver l’historique complet requis (en particulier tous les triplets slow `(h_txt,w,ar)` de L14). Aucune interaction retour avec AM en v1. Recalcul déterministe à finalisation, pas de correspondance de back-and-forth. Détruire helper mort `_dispersion_for_family_e` de `signature_ref.py`. Protéger cohérence L15 admission/finale. Conserver clairement `n_fast` du `ClusterFast` rattaché au `ClusterSlow` pour lifecycle L19/L21, sans doublon. Tests de canal absent/insuffisance, ancre qui change et dispersion recalculée par rapport à final anchor, exactitude au-delà de 50 observations slow, aucun rejet favorable sur données insuffisantes, symétrie héritée.

### 7.6 L18 — Sighting.signal complet

**Statut : À faire.** Remplir `signal` selon §3.2 avec la consolidation des observations/familles, `confidence_mean`, `confidence_min`, compteurs `n_gate_hs`, `n_gate_occupancy`, `n_gate_h_txt`, `n_gate_l_ratio` et `f_hs` quand disponible. `None` pour une mesure indisponible; `cost`, `quality` et `confidence` distincts. Aucun faux bloc `SignalFlags` ni booléens `hs_ok`, `shape_ok`, `map_ok`. Tests de cumul par cluster, aucune double incrémentation, null si sans données, bloc toujours présent et JSON stable. Interface/domaine `f_hs` est précisé L20; formule/calibration L24.

### 7.7 L19 — Sighting.identity/dispersions et lifecycle dual `n_slow`/`n_fast`

**Statut : À faire.** Produire `lifecycle.n_slow` et `lifecycle.n_fast` du contrat §3.2, distincts de `n_obs` et cohérents avec chaque observation slow/fast du `ClusterSlow`; maintenir également la spécification des métriques riches uniques de dispersion : populations admission et finale contre ancre définitive; variantes excluant outliers; `delta_dispersion`; agrégats hs/shape/map et `confidence_mean/min` en champs propres; compteurs `n_hs/n_shape/n_map`; `n_post_promo_outliers`, `drift_*` diagnostiques. L15 calcule une dispersion finale exposée; L19 réutilise sans recalcul ni double incrément, et ventile les composantes sans être consommées par l’EID lorsqu’indiqué. `dispersion_n_min` seul décisionnel; `n_min_*` diagnostique, données insuffisantes conservatrices (max), n=0 null + fallback.

Compléter `lifecycle.n_slow`/`n_fast` depuis compteurs réels d’observations slow/fast : Passe 1 rattache chaque observation à `ClusterFast`, Passe 2 attribue/agrège les compteurs fast réels au `ClusterSlow` correspondant sans perdre l’attribution ni compter deux fois; la valeur `n_fast` reste identifiable entre `ClusterFast`, `ClusterSlow` et Sighting. `mask_uid_switches` reste dans tracking uniquement. Tester deux populations, delta, ventilations et confidence, représentation `at_admission`/`final`, données insuffisantes, aucune double lecture/calcul/compte, compte exact n_slow/n_fast et invariants. Les statistiques géométriques `txt_scale` L14 couvrent 100% des observations slow, sans `maxlen=50` ni biais de quantile interpolé. Le schéma d’ensemble demeure §3.2.

## 8. Lots d’architecture d’ancrage et de rattachement (L20, L21)

### 8.1 L20 — Ancrages doubles `sig_work` / `sig_prime`

**Statut : À faire.** Définir l’ancre de travail `sig_work` (utilisée pendant accumulation/admission/promotion) et l’ancre définitive `sig_prime` (snapshot immuable choisi au `ClusterSlow` final). Garder références/snapshots distincts et provenance claire; `sig_prime` ne change plus après compilation/finalisation. Comparaison candidate ancre, mise à jour/lock, coût ayant bloqué promotion et sélection finale d’ancre doivent être traçables. La final dispersion finale est recalculée par rapport à définitive `sig_prime` suivant 2A-1 (§7.5), sans retour AM. Exposer `f_hs` avec domaine convenu `[0,1]` si disponible et `(None, fallback=True)` sur insuffisance; jamais un score favorable synthétique. Formule/`hs_px_ref` L24. Essais: promotion non lockée mise à jour sig_work, LOCKED bloque, final sig_prime immutable, dispersion recalc vs sig_prime, données HS absentes fallback true, scores bornés.

### 8.2 L21 — Observabilité/performance et cohérence de suivi Fast→Slow

**Statut : À faire.** L’architecture de rattachement est arrêtée en L14 : Passe 1 universelle attribue `obs_fast_id` à chaque observation et Passe 2 compare les identifiants fast des observations slow successives; ne pas réintroduire une cascade alternative à trois niveaux ni un tie-break d’association slow. `CONTINUITY_FAIL` rejette le candidat; `CONTINUITY_NA` est un repli défensif qui poursuit Gate D2 + signature, selon §7.2/`ARB-L14-04`. Pas d’adoption EID ni de transfert d’identité.

Coordonner et exposer sans perte la provenance du suivi et les nombres `n_fast` entre `ClusterFast` et `ClusterSlow`, puis dans `lifecycle.n_fast` du Sighting; `n_slow` demeure le compte lent propre à `ClusterSlow` (§3.2/L19). Chaque observation est comptée une seule fois, malgré l’ancrage monotone et les tranches live retardées. Les diagnostics fast/slow et compteurs/jauges sont typés; aucune règle de purge/capacité. `sample_active_clusters()` est en lecture seule (`INV-L21-01`); `EID_ACTIVE_CLUSTERS_SLOW/FAST` jauges, `EID_ACTIVE_CLUSTERS_OVER_THRESHOLD` compteur sur front montant; seuil de warning à choisir/calibrer L24. Optimisations après comportement correct, cible qualitative « sans overhead mesurable ». Tests T-L21-01 overhead/reporting, T-L21-02 échantillonnage sans mutation, T-L21-03 aucun retrait hors finalisation/idempotence cache, plus correspondance exacte `n_fast` Fast→Slow→Sighting sans perte ni double compte.

## 9. Lots de contrat d’entrée, configuration et observabilité (L22, L23)

### 9.1 L22 — Contrat strict d’entrée et mode flux/parité

**Statut : À faire.** `_parse_row` cible = la routine réelle `Observation.from_entry` de `observation.py`; auditer le format exact et modifier la méthode réelle sans ajouter inutilement une seconde source/parser. Accepter tous les états sauf `entry.get("state") == "LOST"`; `CONFIRMED`, `PENDING` et état absent (tests synthétiques rétrocompatibles) passent. Garder `dedup_key` pour les slow redéversées en grace period sans nouvel `last_detected_ts`. `rx`,`ry` obligatoires à la source, non None; le schéma `signature` détermine les canaux; `signal.valid is False` est rejeté comme `SRC_NOT_VALID`; absence de tout canal dans signature est `SRC_NO_SIGNAL`; compteurs typés dédiés et tests déclarés/écrits/lus. Valeurs booléennes ne valent pas métriques de distance.

Les distances métriques ont le domaine `[0.0,+inf[` : valeurs négatives interdites, zéro admis, borne supérieure infinie; aucun clip supérieur à `1.0` (D4/`INV-L06-03`). Preserve l’exception distincte de `cost_sig` définie en §1.3/D6. Do not conflate `SRC_NOT_VALID`, `SRC_NO_SIGNAL`, `SRC_OBS_REJECT_MISSING_GEOMETRY`, `SRC_OBS_REJECT_MALFORMED`.

Source `LiveSource` ↔ `JsonlSource` interchangeable; queue draining nonblocking, explicit overload/loss policy, ID namespaces documented; live injection suit exactement le même filtrage `state != "LOST"` et le même ordre Passe 1 `ClusterFast` → Passe 2 `ClusterSlow` que batch/JSONL; test parity native↔JSONL (T-L22-01). Two horloges: business `obs.ts` for dedup and nominal closure; wall receipt horloge for `epsilon_close_s` and overload detection only in live mode. Parity question already decided: must be demonstrated by T-L22-01, not remain open. Input list source mentioned; any new module must be accounted for within this task’s codebase constraints. La fermeture du lot sur l’arrivée slow, timeout de sécurité `epsilon_close_s` et `batch_closed_timeout` appartient à ce lot de batching/flux (ARB-L16-01…04) : arrivée slow ferme sans attendre; sans slow expiration sécurité; la fermeture n’émet/finalise aucun cluster. Garder `BatchReader(source,config,counters)` avec compteurs ordre/lecture injectés. Tests T-L16-01 slow; T-L16-02 timeout; T-L16-03 aucun Sighting.

### 9.2 L23 — Alignement `StreamConfig` & `Counters`

**Statut : À faire.** Synchroniser configuration, clés enum, écrivains/lecteurs, exports, version de schéma, compteurs source and tests. `dedup_ttl_s = 0.020` (20 ms) target, mandatory in `DedupSource`, no default fallback. Remove or deprecate dead `batch_max_span_factor` in `config.py`: `BatchReader.next_batch()` already enforces break if `obs.ts - batch[0].ts > self.window_s`; do not keep two controls for the same max span. Ensure typed names declared/written/read match and no old aliases. Add `SRC_OBS_REJECT_MISSING_GEOMETRY`, `SRC_NOT_VALID`, `SRC_NO_SIGNAL` to CounterKey as applicable; remove `SRC_OBS_SLOW_TS_FALLBACK_ZERO` and every read. Remove stale `SourceStats` doc comments; test an `ObservationSource` subclass without constructor `counters` init fails explicitly. Resolve actuel `batch_closed_timeout` key in counters and report. `_get_cluster_ts` docstring corrected; cleanup dead code only with reachability proof. `CalibrationWriter` session `mode` root export; `schema_version` fixed; output per §3.2, no duplicated schema source. `blocked_cost` visible only diagnostic, never output Sighting. Global grep old identifiers, symmetrical declared/written/read; tests exacts de `CounterKey`/`GaugeKey`. Optimize only after correct behavior and verify no measurable overhead; L24 remains calibration-only.

### 9.3 Transition source ↔ configuration

Configuration actuelle réelle listée intégralement §2.1. Intended renames/removals/additions in D2/D4/ARB-L23-14 are not implemented in the provided snapshot: explicitly migrate old temporal fields, geometrical fields/weights, TTL, max-life and geometry gate fields; validate all inequalities in `StreamConfig.__post_init__`. `batch_max_span_factor` is presently a live dataclass field but is not read by actuel `BatchReader`; its requested removal/deprecation is still a lot change, not accomplished. Counter registry present in code lacks entries promised by open lots.

## 10. Lot de recette globale et non-régression (L24)

### 10.1 L24 — Calibration finale

**Statut : À faire.** Prérequis toutes consolidations L13b/L14–L23 terminées, rapport exact, tests et sources réelles. Ne redéfinit pas les interfaces arrêtées en L20/L22. Aucune valeur d’étalonnage n’est prétendue validée sans données réelles.

Calibration documentée, sur corpus de données réel, de : formule `f_hs` et `hs_px_ref`; nouvelle formule `QualityScore.total` et remplissage groupé des composantes; `slow_assoc_window_s=1.5 s` et `slow_idle_close_s=1.5 s` (deux paramètres distincts fixés par décision L14); `min_cost_families` (2 par défaut, 1/hybride seulement si autorisé après revue de l’assertion); `cluster_admit_cost_max`; `cluster_max_dispersion`; `dispersion_quantile`; `dispersion_n_min`; `promo_max_drift`; `hs_divergence`, `l_ratio`, `h_txt` gates; `epsilon_close_s`; `slow/fast_idle_close_s`, `slow/fast_max_lifetime_s`, `cluster_idle_ttl_s`; `geo_tol_*`, `geo_dim_tol`; seuil spatial fast px; `promo_window_n_obs`, `promo_window_span_s`, `MAP_REF_H`, `active_clusters_warn_threshold`; `dedup_ttl_s=0.020` imposé par décision; autres champs provisoires du tableau config. Choix explicités et valeur/contextes reportés, sans réouvrir les décisions d’architecture.

**Tests finaux obligatoires :** T-L24-01 toutes les valeurs calibrables sont documentées et non-None, sauf désactivation explicitement documentée; tests de limites strictes du domaine distance et absence de clip supérieur à 1; test `cost_sig` clip seulement dans son propre domaine; qualité composantes toutes ensemble ou toutes nulles; lots/frames/évictions/max-lifetime, flush idempotent; tous CounterKey/GaugeKey déclaré-écrit-lu, rapports sur liste fermée; JSONL et Sighting immuables, version/schema, callback absent/présent; parité T-L22-01; tests transverses référencés aux lots. Recette n’est close que si L13b + chaque lot L14–L24 est validé, tous tests/greps passent, aucun mismatch document/code ouvert.

## 11. Pilotage, Risques et Critères de sortie globaux

### 11.1 Arbitrages

**Les arbitrages sont fermés et verrouillés.** Les anciennes questions de tie-break et `slow_ambiguity_margin` en L21 sont retirées au profit de l’architecture deux passes arrêtée L14; les seuils/formules répertoriés en L24 restent des paramètres de calibration déléguée, non des questions décisionnelles ouvertes. L’ancienne référence à `ARB-L14-18` « Ouvert » est supprimée; la fenêtre d’association et l’inactivité slow sont des paramètres distincts tels que fixés en §7.2. Toute formulation « parité à trancher » est supplantée par le registre §4.3 et le test obligatoire T-L22-01. Les tests et décisions d’interface restent arrêtés même si un lot n’est pas implémenté.

### 11.2 Matrice des risques

| Risque                                                                         | Réponse et lot propriétaire                                                                |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| Plan/code divergents (ici, socle fourni a encore l’ancien contrat/champs)      | Audit module/références/tests avant chaque lot; vérification livrable §2.1; L13b d’abord.  |
| Consommateur AM couplé aux internals EID ou retours de contrôle                | Interface §3.2 unidirectionnelle, `INV-L08-01/07`, tests de frontière.                     |
| Association contaminante en cas de signal partiel, géométrie ou fast incertain | ordre quatre filtres D1/D2, `CONTINUITY_FAIL` éliminatoire, Gate D2 avant signature.       |
| Compteur absent, compte double ou clé de rapport inconnue                      | enums typés, point unique, déclaration/écriture/lecture, rapport strict; L13b/L23.         |
| Biais de dispersion du buffer borné                                            | recompute final vs ancre, audit `maxlen`, tests L17/L19.                                   |
| Sighting inconsistant/dupliqué                                                 | schéma unique §3.2, immutabilité profonde, émission unique, tests de callback/no callback. |
| Dérive de horloge sur tranche retardée ou dédup périmée                        | horloge monotone + purge TTL avant contrôle doublon, L13b.                                 |
| Perte/parité de données flux                                                   | deux horloges documentées, stratégie de surcharge, test L22 T-L22-01.                      |
| Valeurs de seuil arbitraires ou données insuffisantes favorables               | fallback conservateur, calibration corpus réel L24.                                        |
| Faux invariant borné `[0,1]` sur distance métrique                             | `INV-L06-03`, domaine positif `[0,+inf[`, aucun upper clip sauf `cost_sig` explicite.      |

### 11.3 Critères de sortie globaux

Clôture globale quand L01–L13 sont audités, L13b puis L14–L24 exécutés, tests par lot passent et greps d’obsolescence sont nuls, toutes clés typées annoncées sont réellement déclarées/écrites/lues, chaque invariant §2.2 est vérifié, Sighting correspond à §3.2 dans JSON exporté et code, symétrie/parité/native-JSONL prouvées, chaque paramètre calibrable est documenté sur données réelles, et aucune contradiction ni décision ouverte ne demeure. Les références de sections sont les numéros stables du présent document : contrat uniquement §3.2; invariants §2.2; décision domaines §4.2; registre lot §4.3; roadmap §5; détails lots §7–10.
