# 🌊 Flux de la donnée et Séquencement du Plan

Le pipeline EID naturel suit ce chemin :
`Configuration` ➔ `Entrée/Lotissement` ➔ `Mesure (Distance & Réf)` ➔ `État (Cluster)` ➔ `Orchestration & Sortie (Builder/Cost)`.

---

## ÉTAPE 1 : Les Fondations & Règles (Configuration)

**Fichier : `eid/config.py`**
**Objectif :** Nettoyer la dette et préparer les seuils pour la "Smart Signature" et le Cycle de Vie.
**Modifications :**
Suppression des paramètres morts (`grid_method`).
Ajout des seuils d'ambiguïté pour le court-circuit du glissement (`map_ambiguity_min`, `map_ambiguity_max`).
Ajout de la règle d'éviction (`temporal_radius_evict_s`).

## ÉTAPE 2 : L'Ingestion & Le Lotissement (L'Entrée)

**Fichiers : `eid/observation.py` puis `eid/batching.py`**
**Objectif :** Garantir que ce qui entre dans le système est robuste, performant (O(1)) et prêt pour le temps réel.
**Modifications :**
_observation.py_ : Purge des propriétés mortes (`has_signature`, `is_degenerate_box`), ajout du garde fou sur la lecture JSONL (`JSONDecodeError`).
_batching.py_ : Remplacement de la déduplication `list.pop(0)` (O(n)) par un `dict` + purge temporelle (O(1)). Découplage strict entre la lecture brute et le métier (métier 100% agnostique de l'I/O).

## ÉTAPE 3 : Le Moteur de Distance (La Mesure)

**Fichier : `eid/eid_distance.py`**
**Objectif :** Optimiser le calcul et corriger les biais mathématiques sans perdre la précision du glissement.
**Modifications :**
Correction du biais `dc` (utilisation de `SHAPE_EMPTY_SENTINEL`).
**Smart Signature :** Modification de `_map_distance`. Le glissement (`map`) ne sera calculé par l'appelant QUE si l'identité (`shape` + `hs`) n'est pas évidente.
Suppression du code mort (`n_mismatch`).

## ÉTAPE 4 : L'Ancre d'Identité (Le Référentiel)

**Fichier : `eid/signature_ref.py`**
**Objectif :** Remplacer le médoïde O(n²) par un système de "Prototype" stable pour le consommateur aval.
**Modifications :**
Suppression de `select_medoid`.
Création d'une logique de "Best Anchor" : on met à jour l'empreinte de référence du cluster uniquement si la nouvelle observation a une couverture (`n_valid / len`) strictement supérieure. L'historique ne sert plus qu'à la validation, pas au calcul O(n²).

## ÉTAPE 5 : La Machine à États (Le Stockage)

**Fichier : `eid/cluster.py`**
**Objectif :** Préparer la structure pour qu'elle puisse être consommée proprement à la fin de sa vie.
**Modifications :**
Ajout d'un statut de cycle de vie (`ACTIVE`, `FINALIZED`).
Remplacement de `observations: List` par un stockage borné pour éviter la fuite RAM.
Ajout d'un champ `prime_footprint` (l'empreinte compilée prête à être exportée).

## ÉTAPE 6 : Le Solveur Pur (Le Calcul)

**Fichier : `eid/cost.py`**
**Objectif :** Rendre la matrice de coût mathématiquement pure.
**Modifications :**
Extraction de toutes les mutations d'états (`n_gate_temporal_reject_slow`, `calib_writer`).
La fonction prend les clusters, les observations, et retourne uniquement la matrice de coûts et les masques de rejets explicites.

## ÉTAPE 7 : L'Orchestrateur & L'Émetteur (La Sortie)

**Fichier : `eid/builder.py`**
**Objectif :** Piloter le cycle de vie et émettre les empreintes propres pour le consommateur.
**Modifications :**
Consommation de `cost.py` purifié. Mise à jour de `EIDCounters` en externe.
Implémentation de `_evict_stale_slow` : lorsqu'un cluster slow sort du rayon temporel, on le passe en `FINALIZED`, on compile son empreinte (fusion de l'ancre identité et de la trajectoire fast), on émet l'événement au consommateur aval, puis on le purge de la RAM.
