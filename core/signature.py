# core/signature.py

from __future__ import annotations

import logging
from typing import Any, Optional, Union

import cv2
import numpy as np

log = logging.getLogger("core.signature")

# ── Constantes générales ──────────────────────────────────────────────────────
MIN_STD: float = 8.0                 # garde-fou : sous ce std, Otsu n'a pas de sens
# ── Point 1 — histogramme couleur ─────────────────────────────────────────────
# Source de vérité, à importer depuis eid_distance.py (jamais de redéfinition locale).
HS_H_BINS: int = 24
HS_S_BINS: int = 8
HS_SMOOTH_KSIZE: int = 3              # noyau de lissage 3×3
BG_MIN_PIXEL_RATIO: float = 0.05      # fond min 5% des pixels pour histogramme fiable
# Érosion adaptative du masque de fond (plan 1) : rayon proportionnel à la
# taille du crop, borné. Objectif : écarter les pixels de bordure / halo
# d'anti-aliasing entre texte et fond, qui polluent la teinte dominante.
HS_ERODE_RATIO: float = 0.06          # rayon = ratio × min(h, w)
HS_ERODE_R_MIN: int = 1
HS_ERODE_R_MAX: int = 4
# Si le fond érodé descend sous ce ratio des pixels de fond BRUT, l'érosion est
# jugée trop agressive → plan 1 = recopie du plan 0, sig_hs_mode[1] = "copy".
HS_ERODE_MIN_KEEP: float = 0.25
# ── Point 2 — descripteur de forme ────────────────────────────────────────────
SHAPE_BANDS: int = 16                 # K bandes verticales, fixe
SHAPE_DIM: int = 3 * SHAPE_BANDS      # 48 : [d_k | c_k | sy_k]
SHAPE_EMPTY_SENTINEL: float = -1.0    # c_k / sy_k quand n_k == 0
SHAPE_MIN_W: int = SHAPE_BANDS        # crop plus étroit que K → sig_shape None
SHAPE_MIN_H: int = 4
# ── Binarisation (partagée sig_shape + sig_map) ───────────────────────────────
SHAPE_HIST_SMOOTH: int = 5            # lissage de l'histogramme gris (nb de bins)
SHAPE_MODE_OFFSET: int = 18           # seuil = mode_fond + offset (niveaux gris)
SHAPE_INK_RATIO_MIN: float = 0.02     # validation du seuil : encre ∈ [2%, 60%]
SHAPE_INK_RATIO_MAX: float = 0.60
# ── Point 3 — carte glissante ─────────────────────────────────────────────────
MAP_REF_H: int = 32                   # hauteur de référence du texte, en px
MAP_N: int = 16                       # N FIXE — cartes homogènes, plus de switch
# MAP_REF_H doit rester multiple de MAP_N (32 = 2 × 16) → reshape exact.
MAP_TAU_H: float = 0.10               # densité min d'une ligne pour être "texte"
MAP_L_MAX: int = 512                  # garde-fou volumétrie (colonnes)
MAP_MIN_W: int = 8                    # crop plus étroit → sig_map None
MAP_CLIP_LEFT: int = 1
MAP_CLIP_RIGHT: int = 2
MAP_CLIP_TOP: int = 4
MAP_CLIP_BOTTOM: int = 8
# ── Clés du schéma — ordre de référence (utilisé par bank.py et le probe) ─────
SIG_KEYS: tuple[str, ...] = (
    "sig_source",
    "sig_hs", "sig_hs_mode", "sig_hs_px",
    "sig_shape", "sig_shape_mode", "sig_shape_fill", "sig_shape_cy",
    "sig_map", "sig_map_n", "sig_map_l", "sig_map_mode", "sig_map_clip",
    "sig_map_tau_h", "sig_map_h_txt",
)
# Dict de référence "tout à None" — centralisé ici (source unique, importé
# par core/mask.py et core/bank.py).
EMPTY_SIG_DICT: dict[str, Any] = {k: None for k in SIG_KEYS}

def empty_signature(source: str = "slow") -> dict[str, Any]:
    """Dict dégradé : les 15 clés présentes, tout à None sauf sig_source."""
    sig = dict(EMPTY_SIG_DICT)
    sig["sig_source"] = source
    return sig

def sig_get(sig: Optional[dict[str, Any]], key: str, default: Any) -> Any:
    """
    Lecture sûre d'un champ de signature.
    ATTENTION — `sig.get(key, default)` est INSUFFISANT : une signature porte
    TOUJOURS les 15 clés SIG_KEYS (cf. empty_signature / serialize_signature),
    dont certaines valent None quand le descripteur n'a pas pu être calculé.
    Le default de dict.get() n'est donc jamais déclenché.
    Retourne `default` si `sig` est None, si la clé est absente, ou si la
    valeur est None.
    """
    if not sig:
        return default
    v = sig.get(key)
    return default if v is None else v

def compute_signature(template: Optional[np.ndarray],rect: Optional[tuple[int, int, int, int]] = None,source: str = "slow",crop_color: Optional[np.ndarray] = None,) -> dict[str, Any]:
    """
    Descripteur complet d'un template gris.
    Args:
        template   : ndarray 2D uint8 (gris), taille variable. Peut être None.
        rect       : (x, y, w, h) du Box. NON UTILISÉ — conservé pour
                     compatibilité d'appel positionnel (detection/detect.py).
                     L'ancienne clé sig_ar qui en dépendait a été supprimée.
        source     : "slow" | "fast" | "inherited".
        crop_color : ndarray 3D uint8 RGB, mêmes bornes (x0,y0,x1,y1) que
                     template. Peut être None (sig_hs reste None ; sig_shape
                     et sig_map restent calculés).
    Returns:
        dict des 15 clés SIG_KEYS. Jamais d'exception.
    """
    sig = empty_signature(source)
    try:
        return _compute_from_template(template, sig, crop_color)
    except Exception as exc:      # filet de sécurité absolu
        log.debug("[signature] échec calcul : %s", exc)
        return sig

# ---------------------------------------------------------------------------
# Coeur du calcul — isolé pour que compute_signature reste un pur try/except
# ---------------------------------------------------------------------------
def _compute_from_template(template: Optional[np.ndarray],sig: dict[str, Any],crop_color: Optional[np.ndarray],) -> dict[str, Any]:
    if template is None or template.size == 0:
        return sig
    if template.ndim != 2:
        log.debug("[signature] template ndim=%d (attendu 2)", template.ndim)
        return sig
    h, w = template.shape
    if float(template.std()) < MIN_STD:
        return sig
    # ── Binarisation UNIQUE, partagée par sig_shape et sig_map ────────────
    # Garantit que les deux descripteurs décrivent exactement le même ensemble
    # de pixels d'encre (pas de divergence de seuil entre les deux vues).
    ink, ink_mode = _binarize_text(template)
    if ink is not None:
        _fill_shape(ink, ink_mode, h, w, sig)
        _fill_map(ink, h, w, sig)
    # ── Point 1 : histogrammes couleur du fond sur crop_color ─────────────
    if crop_color is None or crop_color.ndim != 3 or crop_color.shape[:2] != (h, w):
        return sig
    _fill_hs(template, crop_color, h, w, sig)
    return sig

# ---------------------------------------------------------------------------
# Point 1 — histogrammes couleur du fond (double plan : brut + érodé)
# ---------------------------------------------------------------------------
# Cache des kernels d'érosion : radius ∈ [HS_ERODE_R_MIN, HS_ERODE_R_MAX],
# soit un espace discret et borné → précalcul paresseux, kernels immuables.
_ERODE_KERNEL_CACHE: dict[int, np.ndarray] = {}

def _erode_kernel(radius: int) -> np.ndarray:
    """Retourne le kernel elliptique pour `radius`, mis en cache (module-level)."""
    kernel = _ERODE_KERNEL_CACHE.get(radius)
    if kernel is None:
        ksize = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        _ERODE_KERNEL_CACHE[radius] = kernel
    return kernel

def _fill_hs(template: np.ndarray,crop_color: np.ndarray,h: int,w: int,sig: dict[str, Any]) -> None:
    """
    Renseigne sig_hs / sig_hs_mode / sig_hs_px. Mute `sig` en place.
    Deux plans sont produits :
      plan 0 — masque de fond BRUT (complémentaire du seuil Otsu)
      plan 1 — masque de fond ÉRODÉ (rayon adaptatif), qui écarte le halo
               d'anti-aliasing en bordure de glyphe
    Si l'érosion fait tomber le nombre de pixels de fond sous
    HS_ERODE_MIN_KEEP × n_bg_brut, elle est jugée trop agressive : le plan 1
    devient une recopie du plan 0 et sig_hs_mode[1] vaut "copy". Ce compromis
    laisse eid_distance.py choisir le plan à exploiter en connaissance de cause
    (le plan érodé est plus pur mais peut être statistiquement plus pauvre).
    """
    # Texte clair sur fond sombre attendu (cartouche RL) → THRESH_BINARY.
    _, binary = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bg_raw = cv2.bitwise_not(binary)  # 255 = fond, 0 = texte
    n_bg_raw = int(np.count_nonzero(bg_raw))
    if n_bg_raw < BG_MIN_PIXEL_RATIO * bg_raw.size:
        return
    hsv = cv2.cvtColor(crop_color, cv2.COLOR_RGB2HSV)
    hist_raw = _hs_hist_from_mask(hsv, bg_raw)
    if hist_raw is None:
        return
    # ── Érosion adaptative ────────────────────────────────────────────────
    radius = int(round(HS_ERODE_RATIO * min(h, w)))
    radius = max(HS_ERODE_R_MIN, min(HS_ERODE_R_MAX, radius))
    kernel = _erode_kernel(radius)
    bg_ero = cv2.erode(bg_raw, kernel)
    n_bg_ero = int(np.count_nonzero(bg_ero))
    hist_ero = None
    if n_bg_ero >= HS_ERODE_MIN_KEEP * n_bg_raw:
        hist_ero = _hs_hist_from_mask(hsv, bg_ero)
    if hist_ero is None:
        # Érosion trop agressive (ou histogramme dégénéré) → recopie du brut.
        hist_ero = hist_raw
        mode = ("otsu", "copy")
        px = (n_bg_raw, n_bg_raw)
    else:
        mode = ("otsu", "eroded")
        px = (n_bg_raw, n_bg_ero)
    stacked = np.ascontiguousarray(np.stack((hist_raw, hist_ero), axis=0).astype(np.float32))
    stacked.setflags(write=False)
    sig["sig_hs"] = stacked
    sig["sig_hs_mode"] = mode
    sig["sig_hs_px"] = px

def _hs_hist_from_mask(hsv: np.ndarray,mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Histogramme 2D Hue×Saturation (HS_H_BINS × HS_S_BINS) sur les pixels
    marqués par `mask`, lissé 3×3 puis L1-normalisé.
    Retourne un ndarray float32 (H, S) ou None si dégénéré.
    """
    hist = cv2.calcHist(
        [hsv], [0, 1], mask,
        [HS_H_BINS, HS_S_BINS], [0, 180, 0, 256],
    )
    hist = cv2.blur(hist, (HS_SMOOTH_KSIZE, HS_SMOOTH_KSIZE))
    total = float(hist.sum())
    if total <= 0.0:
        return None
    return (hist / total).astype(np.float32)

# ---------------------------------------------------------------------------
# Point 2 — descripteur de forme (projection verticale + stats par bande)
# ---------------------------------------------------------------------------
def _fill_shape(ink: np.ndarray,ink_mode: str,h: int,w: int,sig: dict[str, Any]) -> None:
    """
    Renseigne sig_shape / sig_shape_mode / sig_shape_fill / sig_shape_cy.
    Mute `sig` en place. Consomme la binarisation déjà calculée par l'appelant.
    """
    if w < SHAPE_MIN_W or h < SHAPE_MIN_H:
        return
    ys, xs = np.nonzero(ink)          # coordonnées des pixels texte
    n_total = ys.size
    if n_total == 0:
        return
    # Affectation bande : xs ∈ [0, w) → k ∈ [0, SHAPE_BANDS)
    band = (xs.astype(np.int64) * SHAPE_BANDS) // w
    ysf = ys.astype(np.float64)
    n_k    = np.bincount(band, minlength=SHAPE_BANDS).astype(np.float64)
    sum_y  = np.bincount(band, weights=ysf,        minlength=SHAPE_BANDS)
    sum_y2 = np.bincount(band, weights=ysf * ysf,  minlength=SHAPE_BANDS)
    nonempty = n_k > 0
    safe_n = np.where(nonempty, n_k, 1.0)
    # d_k : densité locale — dénominateur = surface exacte de la bande.
    # Les bornes entières de la bande k sont [ceil(k*w/K), ceil((k+1)*w/K)),
    # ce qui donne des largeurs pouvant différer de 1 px quand w % K != 0.
    edges = np.ceil(np.arange(SHAPE_BANDS + 1, dtype=np.float64) * w / SHAPE_BANDS)
    band_w = np.maximum(edges[1:] - edges[:-1], 1.0)
    d_k = n_k / (band_w * float(h))
    np.clip(d_k, 0.0, 1.0, out=d_k)
    # c_k : centre de masse vertical normalisé, -1.0 si bande vide
    cy_k = sum_y / safe_n
    c_k = np.where(nonempty, cy_k / float(h), SHAPE_EMPTY_SENTINEL)
    # sy_k : dispersion verticale normalisée, -1.0 si bande vide
    var_k = np.maximum(sum_y2 / safe_n - cy_k * cy_k, 0.0)
    sy_k = np.where(nonempty, np.sqrt(var_k) / float(h), SHAPE_EMPTY_SENTINEL)
    # TODO(2.6) : skewness par bande (3e moment) — écarté après arbitrage
    # (faible pouvoir discriminant sur glyphes 8-40 px, instabilité numérique
    # quand sy_k → 0, coût de calibration). Point d'insertion : accumuler
    # sum_y3 = np.bincount(band, weights=ysf**3, ...) ici, puis étendre
    # SHAPE_DIM à 4*SHAPE_BANDS et concaténer la famille en fin de vecteur
    # (le layout par famille garantit que [0:48] reste inchangé).
    shape = np.concatenate((d_k, c_k, sy_k)).astype(np.float32)
    shape = np.ascontiguousarray(shape)
    shape.setflags(write=False)
    sig["sig_shape"] = shape
    sig["sig_shape_mode"] = ink_mode
    sig["sig_shape_fill"] = round(float(n_total) / float(w * h), 5)
    sig["sig_shape_cy"] = round(float(ysf.mean()) / float(h), 5)

# ---------------------------------------------------------------------------
# Point 3 — carte glissante à hauteur normalisée
# ---------------------------------------------------------------------------
def _fill_map(ink: np.ndarray,h: int,w: int,sig: dict[str, Any]) -> None:
    """
    Renseigne sig_map / sig_map_n / sig_map_l / sig_map_mode / sig_map_clip /
    sig_map_tau_h.
    Principe — invariance au zoom par normalisation de la HAUTEUR DU TEXTE :
      1. On mesure h_txt = hauteur de la bbox verticale de l'encre, définie par
         les lignes dont la densité dépasse MAP_TAU_H. Cette hauteur est la
         grandeur physiquement stable entre deux observations d'une même plaque
         à des profondeurs différentes (la largeur, elle, varie avec la
         troncature ; la hauteur du crop varie avec la marge du détecteur).
      2. On rééchantillonne le crop pour que h_txt devienne MAP_REF_H. Le
         facteur d'échelle s = MAP_REF_H / h_txt s'applique AUX DEUX AXES,
         ce qui préserve l'aspect des glyphes. La largeur devient L = round(w · s).
      3. On agrège verticalement en N lignes (N adaptatif). Chaque cellule (i, j)
         porte la densité d'encre de la bande i sur la colonne-pixel j.
    Conséquence directe pour eid_distance.py : deux observations d'une même
    plaque, à des échelles différentes et avec des troncatures différentes,
    produisent des cartes dont les colonnes sont MÉTRIQUEMENT COMPARABLES.
    Un fragment "PLAIRE" est alors une sous-séquence de colonnes de
    "EXEMPLAIRE", détectable par glissement sur l'axe 1.
    """
    if w < MAP_MIN_W or h < SHAPE_MIN_H:
        return

    ink_u8 = ink.astype(np.uint8)
    # ── 1. Bande verticale du texte (h_txt) ───────────────────────────────
    row_density = ink_u8.sum(axis=1, dtype=np.float64) / float(w)
    rows = np.nonzero(row_density >= MAP_TAU_H)[0]
    if rows.size >= 2:
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        h_txt = y1 - y0
        map_mode = "txt"
    else:
        # Bbox d'encre inexploitable (texte trop diffus ou trop fin) :
        # repli documenté sur la hauteur du crop entier.
        y0, y1 = 0, h
        h_txt = h
        map_mode = "crop"
    if h_txt < 2:
        return
    # ── 2. Bitmask de troncature (calculé sur le crop ENTIER, pas la bbox) ──
    # Un bit armé = de l'encre touche ce bord → la plaque est probablement
    # coupée de ce côté. Les bits gauche/droite sont les plus décisifs :
    # ils autorisent eid_distance.py à traiter la carte comme un FRAGMENT.
    clip = 0
    if bool(ink[:, 0].any()):
        clip |= MAP_CLIP_LEFT
    if bool(ink[:, -1].any()):
        clip |= MAP_CLIP_RIGHT
    if bool(ink[0, :].any()):
        clip |= MAP_CLIP_TOP
    if bool(ink[-1, :].any()):
        clip |= MAP_CLIP_BOTTOM
    # ── 3. Rééchantillonnage isotrope pour ramener h_txt à MAP_REF_H ──────
    scale = float(MAP_REF_H) / float(h_txt)
    band = ink_u8[y0:y1, :]                      # on ne garde que la bande texte
    l_cols = int(round(w * scale))
    l_cols = max(1, min(MAP_L_MAX, l_cols))
    # INTER_AREA en réduction (moyenne exacte des pixels sources → densité),
    # INTER_LINEAR en agrandissement. Le résultat float32 ∈ [0,1] EST déjà une
    # densité locale d'encre : pas de re-binarisation, on veut le niveau continu.
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(band.astype(np.float32), (l_cols, MAP_REF_H), interpolation=interp)
    # ── 4. Agrégation verticale en N lignes (N FIXE) ───────────────────────
    # Plus de switch conditionnel : toute carte a MAP_N lignes, donc toute paire
    # est comparable sans cas n_mismatch. Le sur-échantillonnage sur petit h_txt
    # est borné et SIGNALÉ par sig_map_h_txt : le consommateur pondère sa
    # confiance, le producteur ne jette plus d'information.
    step = MAP_REF_H // MAP_N
    grid = resized[: step * MAP_N, :].reshape(MAP_N, step, l_cols).mean(axis=1)
    np.clip(grid, 0.0, 1.0, out=grid)
    grid = np.ascontiguousarray(grid.astype(np.float32))
    grid.setflags(write=False)
    sig["sig_map"] = grid
    sig["sig_map_n"] = MAP_N
    sig["sig_map_l"] = int(l_cols)
    sig["sig_map_mode"] = map_mode
    sig["sig_map_clip"] = int(clip)
    sig["sig_map_tau_h"] = MAP_TAU_H
    # Résolution verticale NATIVE avant rééchantillonnage : permet au
    # consommateur de pondérer la confiance (h_txt=6 → 2.7× sur-échantillonné).
    sig["sig_map_h_txt"] = int(h_txt)

# ---------------------------------------------------------------------------
# Binarisation du texte — appelée UNE FOIS, partagée sig_shape / sig_map
# ---------------------------------------------------------------------------
def _binarize_text(template: np.ndarray) -> tuple[Optional[np.ndarray], Optional[str]]:
    """
    Binarise le crop gris en masque d'encre (texte clair sur fond sombre).
    Stratégie « mode + offset » :
      1. histogramme 256 bins lissé sur SHAPE_HIST_SMOOTH bins
      2. mode = bin dominant = niveau de gris du fond (le fond majoritaire
         dans un cartouche RL)
      3. seuil = mode + SHAPE_MODE_OFFSET
      4. validation : le taux d'encre doit tomber dans
         [SHAPE_INK_RATIO_MIN, SHAPE_INK_RATIO_MAX]
      5. sinon repli Otsu, revalidé par le même critère
    Returns:
        (ink_bool ndarray HxW, "mode"|"otsu") ou (None, None) si aucun
        seuillage ne produit un taux d'encre plausible.
    """
    size = float(template.size)
    hist = cv2.calcHist([template], [0], None, [256], [0, 256]).ravel()
    if SHAPE_HIST_SMOOTH > 1:
        k = np.ones(SHAPE_HIST_SMOOTH, dtype=np.float32) / SHAPE_HIST_SMOOTH
        hist = np.convolve(hist, k, mode="same")
    mode_level = int(np.argmax(hist))
    thr = mode_level + SHAPE_MODE_OFFSET
    if thr < 255:
        ink = template > thr
        ratio = float(np.count_nonzero(ink)) / size
        if SHAPE_INK_RATIO_MIN <= ratio <= SHAPE_INK_RATIO_MAX:
            return ink, "mode"
    # Repli Otsu
    otsu_thr, _ = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = template > otsu_thr
    ratio = float(np.count_nonzero(ink)) / size
    if SHAPE_INK_RATIO_MIN <= ratio <= SHAPE_INK_RATIO_MAX:
        return ink, "otsu"
    return None, None

# ---------------------------------------------------------------------------
# Sérialisation JSONL — projection en sortie.
#
# eid_distance.py sera développé et validé D'ABORD sur ces JSONL avant d'être
# branché sur le flux natif : la projection doit donc être aussi fidèle que
# possible, et l'écart au natif explicitement documenté par clé.
# Table déclarative : {clé: (shape_spec, precision)}
#   shape_spec — tuple dont chaque élément est
#                  · un int  → dimension littérale
#                  · un str  → nom d'une clé SCALAIRE du dict, lue pour obtenir
#                              la dimension (shape dynamique)
#   precision  — nb de décimales de l'arrondi JSON, calibré par clé :
#                  sig_hs    = 8 → histogramme L1 dont les cases valent ~5e-3 ;
#                                  un arrondi grossier tronquerait la queue de
#                                  distribution et ferait dériver la somme L1
#                  sig_shape = 5 → densités / positions bruitées, 1e-5 est très
#                                  en deçà du bruit de mesure
#                  sig_map   = 4 → clé la plus volumineuse (N×L, jusqu'à 16×512) ;
#                                  4 décimales divisent le volume par ~2 sans
#                                  impact mesurable sur une densité de cellule
# ---------------------------------------------------------------------------

_ARRAY_KEYS: dict[str, tuple[tuple[Union[int, str], ...], int]] = {
    "sig_hs":    ((2, HS_H_BINS, HS_S_BINS),   8),
    "sig_shape": ((SHAPE_DIM,),                5),
    "sig_map":   (("sig_map_n", "sig_map_l"),  4),
}

def _resolve_shape(spec: tuple[Union[int, str], ...],src: dict[str, Any]) -> Optional[tuple[int, ...]]:
    """
    Résout un shape_spec en tuple d'entiers.
    Les dimensions symboliques sont lues dans `src` (le dict SOURCE, complet dès
    l'entrée) et non dans le dict en construction : l'ordre de parcours de
    SIG_KEYS est donc indifférent.
    Retourne None si une dimension symbolique est absente, None, ou invalide.
    """
    out: list[int] = []
    for dim in spec:
        if isinstance(dim, int):
            out.append(dim)
            continue
        val = src.get(dim)
        if not isinstance(val, int) or val <= 0:
            return None
        out.append(val)
    return tuple(out)

def serialize_signature(sig: Optional[dict[str, Any]]) -> dict[str, Any]:
    """
    Projette une signature (forme native) vers un dict JSON-safe, exhaustif et
    réversible sur SIG_KEYS.

    Les clés ndarray sont aplaties en liste de floats, arrondis à la précision
    déclarée dans _ARRAY_KEYS. Les tuples (sig_hs_mode, sig_hs_px) sont convertis
    en listes (JSON n'a pas de tuple ; deserialize les restaure en tuples).
    Toutes les autres clés sont des passthrough.
    """
    if sig is None:
        return dict(EMPTY_SIG_DICT)

    out: dict[str, Any] = {}
    for k in SIG_KEYS:
        v = sig.get(k)
        entry = _ARRAY_KEYS.get(k)
        if entry is not None:
            if v is None:
                out[k] = None
            else:
                _, precision = entry
                out[k] = np.round(np.asarray(v).ravel(), precision).tolist()
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out

def deserialize_signature(d: Optional[dict[str, Any]]) -> dict[str, Any]:
    """
    Inverse de serialize_signature(). Les clés ndarray sont reconstruites dans
    leur shape native (float32, C-contigu, read-only) :
      sig_hs    → (2, HS_H_BINS, HS_S_BINS), chaque plan RENORMALISÉ L1
      sig_shape → (SHAPE_DIM,)
      sig_map   → (sig_map_n, sig_map_l)

    sig_hs est renormalisé plan par plan après reconstruction : l'arrondi JSON
    fait dériver la somme de ~1e-5, cette renormalisation garantit l'invariant
    L1 quelle que soit la précision d'écriture — eid_distance.py peut donc
    supposer sum == 1.0 sans dépendre du réglage de sérialisation.

    Robustesse (option i) : si une shape ne peut être résolue ou si le nombre
    d'éléments ne correspond pas, la clé est dégradée à None avec un
    log.warning. Aucune exception n'est levée, un batch d'analyse offline sur
    un fichier partiellement corrompu reste exploitable.
    """
    if d is None:
        return empty_signature()
    out: dict[str, Any] = {}
    for k in SIG_KEYS:
        v = d.get(k)
        entry = _ARRAY_KEYS.get(k)

        if entry is None:
            # Clés scalaires — restauration des tuples aplatis par JSON.
            if k in ("sig_hs_mode", "sig_hs_px") and isinstance(v, list):
                out[k] = tuple(v)
            else:
                out[k] = v
            continue

        if v is None:
            out[k] = None
            continue

        spec, _ = entry
        shape = _resolve_shape(spec, d)
        if shape is None:
            log.warning(
                "[signature] %s : shape non résoluble (spec=%s) → dégradé à None",
                k, spec,
            )
            out[k] = None
            continue

        arr = np.asarray(v, dtype=np.float32).ravel()
        expected = int(np.prod(shape))
        if arr.size != expected:
            log.warning(
                "[signature] %s : %d éléments pour une shape %s (%d attendus) "
                "→ dégradé à None",
                k, arr.size, shape, expected,
            )
            out[k] = None
            continue

        arr = arr.reshape(shape)

        if k == "sig_hs":
            # Renormalisation L1 par plan (axe 0 = brut / érodé).
            arr = arr.copy()
            sums = arr.sum(axis=(1, 2), keepdims=True)
            np.divide(arr, sums, out=arr, where=sums > 0.0)

        arr = np.ascontiguousarray(arr, dtype=np.float32)
        arr.setflags(write=False)
        out[k] = arr

    return out
