# capture/base.py
"""
Toute source (DXCam, cv2, WGC, NDI, ...) doit implémenter cette interface sans extension.
La surface minimale (4 membres) garantit qu'un seul comportement est à tester par implémentation.

Format de sortie obligatoire (grab) : np.ndarray, shape (H, W, 3), dtype uint8, ordre canaux RGB.
Toute conversion couleur (BGR→RGB notamment) est de la responsabilité de la source, jamais de l'appelant.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class CaptureSourceNotFound(Exception):
    """
    Levée par SourceSelector.resolve() quand aucune source de la source_priority n'a pu être validée par la probe.

    Catchée dans main.py pour exit propre avec message utilisateur actionnable (cf. Phase 6).
    """
    pass


class CaptureSource(ABC):
    """
    Contrat A1 strict — 4 membres, non négociable.

    Cycle de vie :
        s = ConcreteSource(...)
        s.start(target_fps)   # peut lever si source indisponible
        while running:
            frame = s.grab()  # None autorisé (pas de frame dispo)
        s.stop()              # idempotent

    Idempotence requise : start() et stop() peuvent être appelés plusieurs fois (probe + run réel via SourceSelector option β).
    """

    @abstractmethod
    def start(self, target_fps: int) -> None:
        """
            target_fps: fréquence de capture cible (indicative pour certaines sources comme cv2/OBS où le débit est imposé par le producteur).
        Initialise et démarre la source.
        Args:

        Raises:
            Toute exception en cas d'indisponibilité de la source.
            Le SourceSelector catche et tente la source suivante.

        Idempotence : appel répété sur une source déjà démarrée doit être un no-op (ne pas relancer le stream, ne pas lever).
        """

    @abstractmethod
    def grab(self) -> Optional[np.ndarray]:
        """
        Récupère la dernière frame disponible.

        Returns:
            np.ndarray RGB (H, W, 3) uint8 si frame disponible,
            None si aucune frame n'est encore prête.

        Contraintes :
            - Retour rapide. Peut bloquer brièvement sur I/O système, jamais d'attente active longue. (retour immédiat).
            - Ne lève pas en régime nominal.
            - Format strict : RGB uint8 (H, W, 3). Toute conversion couleur est faite ici, pas chez l'appelant.
            - L'appelant peut conserver/copier la frame retournée selon ses besoins (pas de garantie de validité au-delà du prochain grab()).
        """

    @abstractmethod
    def stop(self) -> None:
        """
        Arrête la source et libère les ressources.

        Idempotence : appel répété (notamment après un start raté ou un stop déjà effectué) doit être un no-op silencieux, ne jamais lever.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Identifiant court de la source pour les logs.
        Exemples : "dxcam", "cv2", "wgc".

        Doit être stable (même valeur pendant tout le cycle de vie).
        """