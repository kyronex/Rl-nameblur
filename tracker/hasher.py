# hasher.py — Optimisé

import cv2
import numpy as np

# Pré-calcul des puissances de 2 (une seule fois au chargement du module)
_HASH_SIZE = 8
_N_BITS = _HASH_SIZE * _HASH_SIZE  # 64
_POWERS = 1 << np.arange(_N_BITS - 1, -1, -1, dtype=np.uint64)


def compute_phash(crop: np.ndarray) -> int:
    """
    pHash d'un crop.
    Retourne 0 si crop invalide.
    """
    if crop is None or crop.size == 0:
        return 0

    # Grayscale si nécessaire
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

    # Resize + DCT
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))

    # Basses fréquences → bits → entier
    low = dct[:_HASH_SIZE, :_HASH_SIZE]
    bits = (low > np.median(low)).flatten().astype(np.uint64)

    return int(np.dot(bits, _POWERS))


def hamming_distance(h1: int, h2: int) -> int:
    """Nombre de bits différents."""
    return bin(h1 ^ h2).count("1")


def hash_similarity(h1: int, h2: int) -> float:
    """Similarité [0, 1]. 0 si un hash est nul."""
    if h1 == 0 or h2 == 0:
        return 0.0
    return 1.0 - hamming_distance(h1, h2) / _N_BITS


def best_hash_similarity(h: int, history: list[int]) -> float:
    """Meilleure similarité entre h et un historique."""
    if not history or h == 0:
        return 0.0
    return max(hash_similarity(h, old) for old in history if old != 0)
