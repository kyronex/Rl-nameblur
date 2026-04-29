# tracker/hasher.py
from __future__ import annotations
from typing import List
import cv2
import numpy as np
from typing import List, Optional


def compute_phash(crop: np.ndarray, hash_size: int = 8) -> int | None:
    if crop is None or crop.size == 0:
        return None
    if crop.shape[0] < 2 or crop.shape[1] < 2:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean_val = resized.mean()
    bits = (resized > mean_val).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


def hash_similarity(h1: int, h2: int, hash_size: int = 8) -> float:
    total_bits = hash_size * hash_size
    return 1.0 - hamming_distance(h1, h2) / total_bits


def best_hash_similarity(h: int, history: List[int], hash_size: int = 8, top_k: Optional[int] = None) -> float:
    if not history:
        return 0.0
    sims = [hash_similarity(h, prev, hash_size) for prev in history]
    if top_k is None or top_k <= 0:
        return max(sims)  # comportement legacy
    k = min(top_k, len(sims))
    sims.sort(reverse=True)
    return sum(sims[:k]) / k
