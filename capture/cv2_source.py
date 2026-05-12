# capture/cv2_source.py
"""
Implémentation CaptureSource pour cv2.VideoCapture ciblant OBS Virtual Camera.

Spécificités cv2 :
    - read() retourne (ret: bool, frame: ndarray BGR) — conversion BGR→RGB obligatoire dans grab() avant publication (contrat A1 = RGB).
    - Pas de timeout natif sur read() — on s'appuie sur ret==False et BUFFERSIZE=1 pour limiter la latence accumulée.
    - Découverte device via DirectShow (pygrabber, dépendance obligatoire).

Cas d'usage : fallback quand DXCam échoue (FSE indisponible, jeu en Borderless Window capturé par OBS Game Capture + Virtual Camera active).
"""
from __future__ import annotations

import logging
from typing import Optional, List, Tuple

import cv2
import numpy as np
from pygrabber.dshow_graph import FilterGraph

from capture.base import CaptureSource
from capture.config import CaptureConfig

log = logging.getLogger("capture.cv2_source")


class Cv2Source(CaptureSource):
    """
    Source cv2.VideoCapture — fallback OBS Virtual Camera.

    Idempotence :
        - start() : no-op si déjà démarrée.
        - stop()  : no-op si déjà arrêtée ou jamais démarrée.

    Découverte device :
        Énumération DirectShow via pygrabber, filtre par substring (config.cv2_device_name, case-insensitive).
        Lève RuntimeError au start() si aucun device ne matche.

    Configuration :
        Toutes les valeurs proviennent du snapshot CaptureConfig injecté.
        Aucun accès direct à `cfg` — testabilité + cohérence avec SourceSelector qui pilote l'instanciation via _SOURCE_REGISTRY.
    """

    def __init__(self, config: CaptureConfig) -> None:
        self._config: CaptureConfig = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._device_index: Optional[int] = None
        self._started: bool = False
        self._device_logged: bool = False

    @property
    def name(self) -> str:
        return "cv2"

    # ──────────────────────────────────────────────────────────────
    # Découverte device
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _enumerate_directshow_devices() -> List[Tuple[int, str]]:
        """
        Retourne [(index, name), ...] via DirectShow (pygrabber).
        Lève RuntimeError si l'énumération échoue.
        """
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            return list(enumerate(devices))
        except Exception as e:
            raise RuntimeError(
                f"[cv2] Énumération DirectShow échouée : {e}"
            ) from e

    def _find_device_index(self, name_substring: str) -> Optional[int]:
        """
        Cherche un device dont le nom contient `name_substring` (case-insensitive).
        Retourne l'index ou None si non trouvé.
        Les logs de découverte ne sont émis qu'une seule fois (probe + run partagent la même instance)
        """
        devices = self._enumerate_directshow_devices()

        if not self._device_logged:
            log.info("[cv2] Devices DirectShow détectés : %s", devices)

        if not devices:
            if not self._device_logged:
                log.warning("[cv2] Aucun device DirectShow disponible")
            return None

        needle = name_substring.lower()
        for idx, dev_name in devices:
            if needle in dev_name.lower():
                if not self._device_logged:
                    log.info("[cv2] Match : index=%d name='%s'", idx, dev_name)
                self._device_logged = True
                return idx

        if not self._device_logged:
            log.warning(
                "[cv2] Aucun device DirectShow ne contient '%s'. "
                "Devices disponibles : %s",
                name_substring,
                [d[1] for d in devices],
            )
        self._device_logged = True
        return None

    # ──────────────────────────────────────────────────────────────
    # Contrat CaptureSource
    # ──────────────────────────────────────────────────────────────

    def start(self, target_fps: int) -> None:
        if self._started:
            return  # idempotence

        cam_name = self._config.cv2_device_name
        idx = self._find_device_index(cam_name)
        if idx is None:
            raise RuntimeError(
                f"[cv2] Aucun device DirectShow ne contient '{cam_name}'. "
                "Vérifiez qu'OBS Virtual Camera est démarrée "
                "(OBS → Outils → Démarrer la caméra virtuelle)."
            )

        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError(f"[cv2] Échec ouverture device index={idx}")

        # Configuration : buffer minimal pour limiter la latence
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Tentative de réglage résolution / FPS (best effort, non bloquant)
        target_w = self._config.screen_w
        target_h = self._config.screen_h
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
        cap.set(cv2.CAP_PROP_FPS, target_fps)

        # Lecture effective des paramètres appliqués
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        log.info(
            "[cv2] Device ouvert : index=%d résolution=%dx%d fps_reporté=%.1f",
            idx, actual_w, actual_h, actual_fps,
        )

        if (actual_w, actual_h) != (target_w, target_h):
            log.warning(
                "[cv2] Résolution mismatch : attendu %dx%d, obtenu %dx%d. "
                "Vérifiez la configuration OBS (Paramètres → Vidéo → "
                "Résolution de sortie).",
                target_w, target_h, actual_w, actual_h,
            )

        self._cap = cap
        self._device_index = idx
        self._started = True

    def grab(self) -> Optional[np.ndarray]:
        if not self._started or self._cap is None:
            return None

        ret, frame_bgr = self._cap.read()
        if not ret or frame_bgr is None:
            return None

        # Conversion BGR → RGB obligatoire (contrat A1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def stop(self) -> None:
        if not self._started:
            return  # idempotence

        try:
            if self._cap is not None:
                self._cap.release()
        except Exception as e:
            log.warning("[cv2] Erreur lors du release : %s", e)
        finally:
            self._cap = None
            self._started = False
