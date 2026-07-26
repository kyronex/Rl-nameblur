# Plan autoporté — Robustification du suivi optique (O-c + S-b)

## 0. Objet et principe directeur

Ce plan décrit une évolution du composant de **suivi par flux optique** (le module qui, entre deux détections lourdes, prédit où se déplace une région suivie sur l'image). Il combine deux décisions déjà arbitrées :

- **O-c — robustesse à l'occlusion partielle** : suivi multi-points + validation aller-retour.
- **S-b — gestion de l'échelle** : estimation d'une transformation de similarité robuste (translation + facteur d'échelle) par consensus.

**Principe de séparation fondamental :** l'occlusion est un problème de _qualité de mesure_, l'échelle est un problème de _modèle de sortie_. Les deux ne se traitent pas dans le même incrément. **O-c se livre d'abord, seul et complet. S-b vient ensuite, dans un incrément distinct**, car il modifie la nature de ce que le suivi produit.

---

## PARTIE A — Incrément O-c (occlusion) — à livrer en premier

### A.1 Intention

Rendre la prédiction de déplacement fiable même quand une partie de la région suivie est masquée (par un autre élément mobile, un effet visuel, un chevauchement). Aujourd'hui, le suivi s'appuie sur un très petit nombre de points d'ancrage : si la majorité disparaît, la prédiction s'effondre silencieusement.

### A.2 Mécanisme 1 — Densification des points suivis

- Remplacer le petit ensemble de points d'ancrage par une **grille régulière** répartie sur toute la surface de la région suivie.
- La densité de la grille est un **paramètre calibrable** (point de calage A-1).
- **Bénéfice :** si une zone est occludée, les points situés hors de la zone masquée survivent et suffisent à estimer le déplacement.

### A.3 Mécanisme 2 — Validation aller-retour (forward-backward)

- Pour chaque point, mesurer le déplacement dans un sens (image précédente → image courante), puis **remesurer en sens inverse** (image courante → image précédente).
- Un point fiable revient approximativement à sa position d'origine. Un point occludé ou aberrant **ne revient pas** : l'écart de boucle dépasse un seuil.
- Ne conserver que les points dont l'écart aller-retour reste sous un **seuil calibrable** (point de calage A-2).
- **Bénéfice :** les points parasites (occlusion, reflet, bord instable) sont éliminés _proprement_, avant tout calcul de déplacement global.

### A.4 Agrégation

- Le déplacement final est la **valeur médiane** des déplacements des points validés (robuste aux outliers résiduels).
- **Garde-fou de survie :** si le nombre de points validés tombe sous un minimum, le suivi déclare un **échec explicite** pour cette itération, plutôt que de renvoyer une prédiction douteuse.

### A.5 Contrat de sortie — INCHANGÉ (exigence stricte)

- L'incrément O-c produit exactement la même information qu'avant : un **déplacement (translation)** et un **indicateur de succès/échec**.
- La taille de la région suivie **reste fixe**. Aucun composant en aval n'est modifié.
- **Aucune** logique de repli, d'extrapolation ou de confirmation par corrélation en aval n'est touchée. Cet incrément est **entièrement contenu** dans le module de flux.

### A.6 Points de calage O-c

| Réf | Paramètre                        | Effet si trop haut                 | Effet si trop bas                      |
| --- | -------------------------------- | ---------------------------------- | -------------------------------------- |
| A-1 | Densité de la grille             | Coût de calcul en hausse           | Moins de survivants en cas d'occlusion |
| A-2 | Seuil d'écart aller-retour       | Laisse passer des points aberrants | Rejette trop, réduit la robustesse     |
| A-3 | Nombre minimal de points validés | Trop d'échecs déclarés             | Prédictions sur base trop maigre       |

### A.7 Critères de validation O-c

- Sous occlusion partielle simulée, la prédiction reste stable (pas d'effondrement).
- Le taux de « faux succès » (prédiction acceptée alors que le flux est faux) diminue.
- Le coût par itération reste dans le budget de temps de la phase de flux.

---

## PARTIE B — Incrément S-b (échelle) — à livrer séparément, après O-c

### B.1 Intention

Permettre au suivi d'exprimer un **changement de taille** de la région (rapprochement/éloignement du sujet). Aujourd'hui, le suivi ne peut produire qu'une translation : même une mesure d'échelle parfaite serait ignorée, car la sortie fige la taille. **S-b n'a de sens que parce qu'il change le modèle de sortie.**

### B.2 Pré-requis

S-b **s'appuie sur les points validés produits par O-c**. Il ne se conçoit pas sans O-c livré et stabilisé au préalable.

### B.3 Mécanisme

- À partir de l'ensemble des points validés (positions avant / après), estimer une **transformation de similarité** : translation + facteur d'échelle (la rotation est traitée au point de décision B-D1).
- L'estimation utilise une méthode à **consensus robuste** (élimination automatique des points minoritaires incohérents), pour rester fiable même si quelques points résiduels sont mauvais.
- Le facteur d'échelle obtenu met à jour la **taille** de la région suivie, en plus de sa position.

### B.4 Contrat de sortie — MODIFIÉ (impact assumé)

C'est le cœur de la séparation : S-b fait passer la sortie de « translation seule » à « translation + taille variable ». **Tous les composants qui consomment la région suivie doivent être revus de façon synchronisée**, notamment :

1. la logique d'**extrapolation** du déplacement (qui suppose aujourd'hui une taille constante) ;
2. la logique de **correction de dérive** ;
3. le mécanisme de **rafraîchissement du gabarit de référence** (qui prélève une zone à la taille courante) ;
4. la **confirmation par corrélation en aval** (dont la marge de recherche dépend de la taille).

### B.5 Décisions à acter avant écriture

| Réf  | Décision                       | Options                                                                     |
| ---- | ------------------------------ | --------------------------------------------------------------------------- |
| B-D1 | Traitement de la rotation      | La figer à zéro (hypothèse d'orientation stable) **ou** la propager         |
| B-D2 | Bornes du facteur d'échelle    | Plafonner la variation par itération pour éviter l'emballement              |
| B-D3 | Repli si estimation non fiable | Retomber sur le comportement O-c (translation seule) si le consensus échoue |

### B.6 Points de calage S-b

| Réf | Paramètre                               | Rôle                                    |
| --- | --------------------------------------- | --------------------------------------- |
| B-1 | Tolérance du consensus                  | Séparation points cohérents / aberrants |
| B-2 | Bornes min/max d'échelle par itération  | Stabilité de la taille                  |
| B-3 | Seuil de bascule vers repli translation | Sécurité en cas d'estimation dégénérée  |

### B.7 Critères de validation S-b

- Sur une séquence de rapprochement/éloignement, la taille suivie évolue de manière cohérente et lissée.
- En cas d'occlusion, l'estimation ne diverge pas (le consensus absorbe les points masqués ; sinon, repli B-D3).
- Aucun composant aval ne présente de régression après mise à jour synchronisée.

---

## PARTIE C — Séquencement et règles de non-régression

1. **Ordre imposé :** O-c d'abord (complet, validé, mesuré), **puis** S-b. Ne jamais mélanger les deux dans un même lot de modifications.
2. **Atomicité :** au sein de O-c, la densification des points et la validation aller-retour sont deux modifications **logiquement indépendantes** et peuvent être livrées en deux étapes distinctes pour un suivi de changement plus lisible.
3. **Isolation du contrat :** tant que S-b n'est pas engagé, le contrat de sortie reste strictement inchangé — c'est la garantie que O-c n'introduit aucun effet de bord.
4. **Fiche orthogonale rappelée :** la question de la précision numérique de la sortie (arrondi de position) est un sujet distinct des deux axes ci-dessus ; elle ne doit être ni fusionnée à O-c ni à S-b.

---

Ce plan est autoporté : il peut être compris et appliqué sans consulter le code ni aucune référence externe. Dis-moi si tu veux que je le formate en fichier livrable (Markdown) ou que je détaille les critères de validation en check-list opérationnelle.
