# capture/selector.py
"""
SourceSelector — sélection automatique de la source de capture.

Stratégie : tente chaque source dans l'ordre capture_config.source_priority.
Probe = start() + grab() jusqu'à probe_min_frames valides en probe_timeout_s max + stop() systématique.

Retour : première source validée, ARRÊTÉE (décision β).
CaptureThread la redémarrera via start().

Lève CaptureSourceNotFound si aucune source ne passe la probe.

Extensibilité : pour ajouter une nouvelle source (NDI, WGC, Spout, ...),
1) implémenter CaptureSource dans capture/<nom>_source.py
2) ajouter une entrée dans _SOURCE_REGISTRY ci-dessous
3) ajouter le nom dans config.capture.source_priority
4) ajouter le nom dans la whitelist CaptureConfig.__post_init__


Injection config :
    B1 — explicite (recommandé)  : SourceSelector.resolve(CaptureConfig(...))
    B2 — fallback ergonomique    : SourceSelector.resolve()  → CaptureConfig() auto

Sonde L3.7 :
    selector_source_<name> (count) — incrément unique à la sélection.
    Écart documenté vs Plan_Bench.md (qui spécifiait gauge) — à acter en README.
"""
from __future__ import annotations

import logging
from time import perf_counter, sleep
from typing import Dict, Optional, Type

from bench import bench
from capture.base import CaptureSource, CaptureSourceNotFound
from capture.config import CaptureConfig
from capture.dxcam_source import DXCamSource
from capture.cv2_source import Cv2Source
from capture.mss_source import MSSSource
from capture.wgc_source import WgcSource

log = logging.getLogger("capture.selector")

# Backoff entre deux grab() ratés pendant la probe — évite spin CPU
# si la source retourne None instantanément en boucle.
_PROBE_GRAB_BACKOFF_S = 0.005

# Registry des sources disponibles — point d'extension unique.
_SOURCE_REGISTRY: Dict[str, Type[CaptureSource]] = {
    "dxcam": DXCamSource,
    "cv2": Cv2Source,
    "mss": MSSSource,
    "wgc":   WgcSource,
}


class SourceSelector:
    """
    Sélecteur de source de capture. Méthode statique unique : resolve().

    Usage (main.py) :
        try:
            source = SourceSelector.resolve()              # B2 — fallback auto
            # ou
            source = SourceSelector.resolve(CaptureConfig())  # B1 — explicite
        except CaptureSourceNotFound as e:
            print(str(e))
            sys.exit(1)
        capture_thread = CaptureThread(source=source, ...)
        capture_thread.start()
    """

    @staticmethod
    def resolve(config: Optional[CaptureConfig] = None) -> CaptureSource:
        """
        Itère sur capture_config.source_priority, retourne la première source
        validée par probe (arrêtée, prête à être start()-ée par le consommateur).

        Args:
            config: snapshot CaptureConfig figé. Si None, instancie CaptureConfig()
                    (lit cfg). B1 = explicite, B2 = fallback auto.

        Returns:
            CaptureSource validée, en état ARRÊTÉ (décision β).

        Raises:
            CaptureSourceNotFound: aucune source ne passe la probe.
        """
        # B2 — fallback : instanciation depuis cfg si non fourni
        config = config or CaptureConfig()

        # Note : source_priority non vide garanti par CaptureConfig.__post_init__.
        # Pas de re-vérification ici — invariant amont.

        log.info("[selector] Résolution — priorité = %s", config.source_priority)

        for source_name in config.source_priority:
            if source_name not in _SOURCE_REGISTRY:
                # Défense en profondeur : CaptureConfig valide déjà la whitelist,
                # mais si le registry diverge de la whitelist, on log et on continue.
                log.warning(
                    "[selector] Source '%s' absente du registry — ignorée. "
                    "Sources enregistrées : %s",
                    source_name, list(_SOURCE_REGISTRY.keys()),
                )
                continue

            log.info("[selector] Tentative source : %s", source_name)

            try:
                source = _SOURCE_REGISTRY[source_name](config=config)
            except Exception as e:
                log.warning("[%s] instanciation a levé : %s", source_name, e)
                continue

            if SourceSelector._probe(
                source, source_name,
                capture_fps=config.capture_fps,
                timeout_s=config.probe_timeout_s,
                min_frames=config.probe_min_frames,
            ):
                log.info("[selector] Source retenue : %s", source_name)
                bench.count(f"selector_source_{source_name}")
                return source  # arrêtée (β), prête pour start() par CaptureThread

        # Aucune source validée
        raise CaptureSourceNotFound(
            "Aucune source de capture disponible.\n\n"
            "Solutions :\n"
            "  1. Lancez Rocket League en mode Fullscreen ou Borderless Window\n"
            "  2. OU lancez OBS avec une source 'Game Capture' puis démarrez\n"
            "     la caméra virtuelle (Outils → Démarrer la caméra virtuelle)\n\n"
            "Puis relancez le script."
        )

    @staticmethod
    def _probe(source: CaptureSource, name: str, capture_fps: int,
               timeout_s: float, min_frames: int) -> bool:
        """
        Probe lifecycle complet : start → grab × N → stop.

        Retourne True si min_frames frames valides obtenues en ≤ timeout_s.
        stop() systématique (même en cas d'échec), exceptions absorbées et loggées.
        """
        # ── start ──
        try:
            source.start(target_fps=capture_fps)
        except Exception as e:
            log.warning("[%s] start() a levé : %s", name, e)
            return False

        # ── grab × N ──
        valid_frames = 0
        deadline = perf_counter() + timeout_s
        try:
            while perf_counter() < deadline:
                frame = source.grab()
                if frame is not None:
                    valid_frames += 1
                    if valid_frames >= min_frames:
                        break
                else:
                    sleep(_PROBE_GRAB_BACKOFF_S)
        except Exception as e:
            log.warning("[%s] grab() a levé pendant la probe : %s", name, e)
            # On continue vers stop() — pas de return prématuré

        # ── stop systématique (β) ──
        try:
            source.stop()
        except Exception as e:
            log.warning("[%s] stop() après probe a levé : %s", name, e)

        # ── verdict ──
        if valid_frames >= min_frames:
            log.info("[%s] Probe OK (%d/%d frames)",
                     name, valid_frames, min_frames)
            return True

        log.warning("[%s] Probe échouée (%d/%d frames en %.2fs)",
                    name, valid_frames, min_frames, timeout_s)
        return False
