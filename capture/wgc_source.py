# capture/wgc_source.py
"""
Implémentation CaptureSource pour Windows Graphics Capture (WGC).

Spécificités WGC :
    - API push/callback (windows-capture==2.0.0) : les frames arrivent via on_frame_arrived().
    - start_free_threaded() utilisé (non bloquant) — retourne un CaptureControl.
    - stop() appelé sur le CaptureControl retourné par start_free_threaded().
      Si on_closed a déjà été appelé (CaptureControl.is_finished == True), stop() est skip silencieusement.
    - Buffer interne (_latest_frame + _lock) pour exposer grab() non-bloquant conforme au contrat A1.
    - Sortie native BGRA → conversion BGRA→RGB via cv2.cvtColor dans le callback.
    - Deux modes : "monitor" (monitor_index) ou "window" (window_name substring match).
      Défaut : "monitor" + monitor_index=0 (primaire) — opt-in window via config.
    - Validation résolution : WARNING si mismatch (sémantique identique à Cv2Source — source externe).

target_fps :
    Reçu pour conformité au contrat A1, non utilisé activement.
    WGC délivre les frames à la cadence native de la source (écran ou fenêtre).

Thread-safety :
    on_frame_arrived() s'exécute dans le thread interne WGC.
    grab() s'exécute dans le thread appelant (main loop ou SourceSelector).
    _lock protège _latest_frame entre ces deux threads.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

import cv2
import numpy as np
from windows_capture import CaptureControl, Frame, WindowsCapture

from capture.base import CaptureSource
from capture.config import CaptureConfig

log = logging.getLogger("capture.wgc_source")


class WgcSource(CaptureSource):
    """
    Source WGC — Windows Graphics Capture via windows-capture==2.0.0.

    Idempotence :
        - start() : no-op si déjà démarrée.
        - stop()  : no-op si déjà arrêtée ou jamais démarrée.
    """

    def __init__(self, config: CaptureConfig) -> None:
        self._config: CaptureConfig = config
        self._capture: Optional[WindowsCapture] = None
        self._capture_control: Optional[CaptureControl] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._lock: threading.Lock = threading.Lock()
        self._started: bool = False

    @property
    def name(self) -> str:
        return "wgc"

    def start(self, target_fps: int) -> None:
        if self._started:
            return  # idempotence

        # --- CP1 : Validation et résolution des paramètres WGC ---
        target = self._config.wgc_target

        if target == "monitor":
            monitor_index: Optional[int] = self._config.wgc_monitor_index
            window_name: Optional[str] = None
            log.info(f"[WGC] Mode monitor — index={monitor_index}")

        elif target == "window":
            window_name_cfg = self._config.wgc_window_name
            if not window_name_cfg:
                raise RuntimeError(
                    "[WGC] target='window' mais wgc_window_name est vide. "
                    "Renseignez capture.wgc.window_name dans config.yaml."
                )
            monitor_index = None
            window_name = window_name_cfg
            log.info(f"[WGC] Mode window — name substring='{window_name}'")

        else:
            raise RuntimeError(
                f"[WGC] wgc_target invalide : '{target}'. "
                f"Valeurs attendues : 'monitor' | 'window'."
            )

        # --- Instanciation WindowsCapture ---
        try:
            self._capture = WindowsCapture(
                cursor_capture=None,
                draw_border=None,
                monitor_index=monitor_index,
                window_name=window_name,
            )
        except Exception as e:
            raise RuntimeError(f"[WGC] WindowsCapture() a échoué : {e}") from e

        # --- Callbacks ---

        @self._capture.event
        def on_frame_arrived(frame: Frame, capture_control) -> None:  # type: ignore[type-arg]
            # capture_control ici = InternalCaptureControl (interne lib, non importable).
            # L'arrêt se fait exclusivement via self._capture_control (CaptureControl)
            # retourné par start_free_threaded() — on n'utilise pas ce capture_control local.

            try:
                frame_rgb: np.ndarray = cv2.cvtColor(
                    frame.frame_buffer, cv2.COLOR_BGRA2RGB
                )
            except Exception as e:
                log.warning(f"[WGC] Conversion frame échouée : {e}")
                return

            # Validation résolution — WARNING uniquement (source externe, cohérent Cv2Source)
            h, w = frame_rgb.shape[:2]
            if w != self._config.screen_w or h != self._config.screen_h:
                log.warning(
                    f"[WGC] Résolution reçue {w}x{h} ≠ "
                    f"config {self._config.screen_w}x{self._config.screen_h}. "
                    f"Vérifiez la résolution de la source WGC."
                )

            with self._lock:
                self._latest_frame = frame_rgb

        @self._capture.event
        def on_closed() -> None:
            log.info("[WGC] Source fermée (on_closed)")
            with self._lock:
                self._latest_frame = None

        # --- Démarrage ---
        try:
            # start_free_threaded() démarre le thread WGC et retourne un CaptureControl
            self._capture_control = self._capture.start_free_threaded()
        except Exception as e:
            self._capture = None
            self._capture_control = None
            raise RuntimeError(f"[WGC] start_free_threaded() a échoué : {e}") from e

        self._started = True
        log.info(
            f"[WGC] Démarré — target='{target}' "
            f"(target_fps={target_fps} indicatif, non utilisé)"
        )

    def grab(self) -> Optional[np.ndarray]:
        if not self._started:
            return None
        with self._lock:
            return self._latest_frame

    def stop(self) -> None:
        if not self._started:
            return  # idempotence

        # CP2 — is_finished vérifié avant stop() pour éviter double-stop
        # Pas de wait() — fire-and-forget (robustesse > élégance)
        try:
            if self._capture_control is not None:
                if not self._capture_control.is_finished:
                    self._capture_control.stop()
                else:
                    log.debug("[WGC] stop() skippé — CaptureControl déjà terminé (on_closed antérieur)")
        except Exception as e:
            log.warning(f"[WGC] capture_control.stop() a levé : {e}")
        finally:
            self._capture = None
            self._capture_control = None
            self._started = False
            with self._lock:
                self._latest_frame = None
            log.info("[WGC] Arrêté")

