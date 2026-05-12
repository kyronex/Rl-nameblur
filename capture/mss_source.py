# capture/mss_source.py
"""
Implémentation CaptureSource pour MSS (capture multi-plateforme via API système).

Spécificités MSS :
    - Capture synchrone à la demande (pas de stream interne, pas de target_fps natif).
    - Sortie native BGRA → conversion BGRA→RGB obligatoire dans grab().
    - Sélection du moniteur via config.capture.mss.monitor_index (défaut 1 = écran principal).
    - Validation stricte de la résolution au start() : raise si mismatch avec config.screen.* (Option A — fail-fast).
      Sémantique : MSS = capture directe écran, mismatch = config projet incohérente avec setup matériel.
      (Diffère volontairement de Cv2Source qui émet un WARNING — sémantique config OBS externe.)
"""
from __future__ import annotations

import logging
from typing import Optional

import mss
import numpy as np

from capture.base import CaptureSource
from capture.config import CaptureConfig

log = logging.getLogger("capture.mss_source")


class MSSSource(CaptureSource):
    """
    Source MSS — capture écran via API système (DXGI sous Windows, Xlib sous Linux, Quartz sous macOS).

    Idempotence :
        - start() : no-op si déjà démarrée.
        - stop()  : no-op si déjà arrêtée ou jamais démarrée.

    Le log des monitors disponibles n'est émis qu'une seule fois sur toute la durée
    de vie du process (flag de classe), pour éviter le doublon probe + run réel imposé
    par l'option β du SourceSelector.

    target_fps :
        Reçu pour conformité au contrat A1, stocké en attribut, non utilisé activement.
        MSS capture à la cadence d'appel de grab() (synchrone), pas de stream interne.
    """

    _monitors_logged: bool = False  # flag de classe, partagé

    def __init__(self, config: CaptureConfig) -> None:
        self._config: CaptureConfig = config
        self._sct: Optional[mss.base.MSSBase] = None
        self._monitor: Optional[dict] = None
        self._target_fps: int = 0
        self._started: bool = False

    @property
    def name(self) -> str:
        return "mss"

    def start(self, target_fps: int) -> None:
        if self._started:
            return  # idempotence

        self._target_fps = target_fps
        self._sct = mss.mss()

        if not MSSSource._monitors_logged:
            try:
                log.info(f"[MSS] Monitors disponibles : {self._sct.monitors}")
            except Exception as e:
                log.warning(f"[MSS] Listing monitors a échoué : {e}")
            MSSSource._monitors_logged = True

        monitor_index = self._config.mss_monitor_index
        try:
            self._monitor = self._sct.monitors[monitor_index]
        except IndexError:
            self._sct.close()
            self._sct = None
            raise RuntimeError(
                f"[MSS] monitor_index={monitor_index} hors plage "
                f"(disponibles: 0..{len(self._sct.monitors) - 1 if self._sct else '?'})"
            )

        # Validation stricte résolution (Option A — fail-fast)
        expected_h = self._config.screen_h
        expected_w = self._config.screen_w
        actual_w = self._monitor["width"]
        actual_h = self._monitor["height"]

        if actual_w != expected_w or actual_h != expected_h:
            self._sct.close()
            self._sct = None
            self._monitor = None
            raise RuntimeError(
                f"[MSS] monitor[{monitor_index}] = {actual_w}x{actual_h}, "
                f"config.screen attend {expected_w}x{expected_h}. "
                f"Ajustez config.screen.* ou capture.mss.monitor_index."
            )

        self._started = True
        log.info(
            f"[MSS] Démarré sur monitor[{monitor_index}] "
            f"({actual_w}x{actual_h}) — target_fps={target_fps} (indicatif, non utilisé)"
        )

    def grab(self) -> Optional[np.ndarray]:
        if not self._started or self._sct is None or self._monitor is None:
            return None

        # MSS retourne un objet ScreenShot avec buffer BGRA.
        # Conversion BGRA → RGB via slicing numpy (zéro dépendance cv2 ici).
        shot = self._sct.grab(self._monitor)
        frame_bgra = np.asarray(shot, dtype=np.uint8)  # (H, W, 4) BGRA
        frame_rgb = frame_bgra[:, :, [2, 1, 0]]         # BGRA → RGB (drop alpha)
        return np.ascontiguousarray(frame_rgb)          # garantit contiguïté pour downstream

    def stop(self) -> None:
        if not self._started:
            return  # idempotence

        try:
            if self._sct is not None:
                self._sct.close()
        except Exception as e:
            log.warning(f"[MSS] stop() a levé : {e}")
        finally:
            self._sct = None
            self._monitor = None
            self._started = False
            log.info("[MSS] Arrêté")
