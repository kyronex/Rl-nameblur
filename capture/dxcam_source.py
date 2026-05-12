# capture/dxcam_source.py
"""
Implémentation CaptureSource pour DXCam (capture DirectX, Windows).

Spécificités DXCam :
    - Sortie native RGB si output_color="RGB" (pas de conversion ici).
    - get_latest_frame() retourne None tant qu'aucune frame n'est prête (comportement nominal, propagé tel quel par grab()).
    - Capture FSE (Fullscreen Exclusive) instable : c'est précisément pourquoi le SourceSelector existe (Phase 4).
"""
from __future__ import annotations

import logging
from typing import Optional

import dxcam
import numpy as np

from capture.base import CaptureSource
from capture.config import CaptureConfig

log = logging.getLogger("capture.dxcam_source")


class DXCamSource(CaptureSource):
    """
    Source DXCam — capture DirectX bas niveau sous Windows.

    Idempotence :
        - start() : no-op si déjà démarrée.
        - stop()  : no-op si déjà arrêtée ou jamais démarrée.

    Le log dxcam.device_info() n'est émis qu'une seule fois sur toute la durée de vie du process (flag de classe), pour éviter le doublonprobe + run réel imposé par l'option β du SourceSelector.
    """

    _device_info_logged: bool = False  # flag de classe, partagé

    def __init__(self, config: CaptureConfig) -> None:
        self._config: CaptureConfig = config
        self._camera = None
        self._started: bool = False

    @property
    def name(self) -> str:
        return "dxcam"

    def start(self, target_fps: int) -> None:
        if self._started:
            return  # idempotence

        if not DXCamSource._device_info_logged:
            try:
                devices = dxcam.device_info()
                log.info(f"[DXCam] Devices disponibles : {devices}")
            except Exception as e:
                log.warning(f"[DXCam] device_info() a échoué : {e}")
            DXCamSource._device_info_logged = True

        self._camera = dxcam.create(output_color="RGB")
        if self._camera is None:
            raise RuntimeError("dxcam.create() a retourné None")

        self._camera.start(target_fps=target_fps)
        self._started = True
        log.info(f"[DXCam] Démarré @ {target_fps} fps cible")

    def grab(self) -> Optional[np.ndarray]:
        if not self._started or self._camera is None:
            return None
        # DXCam délivre déjà du RGB (output_color="RGB"), pas de conversion.
        # None retourné nominalement quand aucune nouvelle frame n'est prête.
        return self._camera.get_latest_frame()

    def stop(self) -> None:
        if not self._started:
            return  # idempotence

        try:
            if self._camera is not None:
                self._camera.stop()
        except Exception as e:
            log.warning(f"[DXCam] stop() a levé : {e}")
        finally:
            self._camera = None
            self._started = False
            log.info("[DXCam] Arrêté")
