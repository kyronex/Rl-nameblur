# models.py

"""
Structures d'entrée/sortie du tracker.
Aucune logique métier — uniquement des conteneurs typés.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict
import numpy as np


@dataclass
class FrameInput:
    """Snapshot d'une frame entrante."""
    image: np.ndarray
    frame_id: int
    timestamp: float


@dataclass
class Detection:
    """Une détection brute issue de YOLO ou d'un autre détecteur."""
    rect: tuple                          # (x, y, w, h)
    confidence: float
    template: Optional[np.ndarray] = None
    scores: Optional[Dict] = None
    crop: Optional[np.ndarray] = None    # sous-image plaque pour le hash


@dataclass
class TrackResult:
    """Résultat de tracking renvoyé au pipeline pour une mask active."""
    global_id: int
    rect: tuple                          # (x, y, w, h)
    state: str                           # PENDING | CONFIRMED | LOST
    confidence: float
    source: str                          # slow | fast | predicted
    bbox_clipped: bool = False
    ambiguous: bool = False
    age_frames: int = 0
    missing: int = 0
