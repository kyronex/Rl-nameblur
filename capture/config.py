# capture/config.py
"""
Configuration figée pour le sous-système de capture.

Snapshot immuable instancié au démarrage (ou injecté pour test).
Pas de hot-reload : toutes les clés capture.* sont consommées une seule fois au bootstrap (résolution de la source).
Modifier ces clés à chaud n'a aucun sens fonctionnel — relance requise.

Périmètre : CaptureConfig agrège **toutes** les valeurs dont les sources et le selector ont besoin.
Pattern cohérent avec TrackerConfig (tracker/models.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from config import cfg


@dataclass(frozen=True)
class CaptureConfig:
    """
    Snapshot immuable des paramètres capture.*.

    Instancié par main.py au bootstrap, passé à SourceSelector.resolve(config).
    Pour tests : instanciable directement avec valeurs custom (B1).
    Si non fourni à resolve(), un défaut est créé à partir de cfg (B2).
    """

    screen_w: int = field(default_factory=lambda: cfg.get("screen.width"))
    screen_h: int = field(default_factory=lambda: cfg.get("screen.height"))
    capture_fps: int = field(default_factory=lambda: cfg.get("screen.capture_fps"))

    source_priority: List[str] = field(default_factory=lambda: list(cfg.get("capture.source_priority", ["dxcam", "cv2"])))
    probe_timeout_s: float = field(default_factory=lambda: cfg.get("capture.probe_timeout_s"))
    probe_min_frames: int = field(default_factory=lambda: cfg.get("capture.probe_min_frames"))

    cv2_device_name: str = field(default_factory=lambda: cfg.get("capture.cv2.device_name"))

    mss_monitor_index: int = field(default_factory=lambda: cfg.get("capture.mss.monitor_index"))

    wgc_target: str = field(default_factory=lambda: cfg.get("capture.wgc.target"))
    wgc_monitor_index: int = field(default_factory=lambda: cfg.get("capture.wgc.monitor_index"))
    wgc_window_name: str = field(default_factory=lambda: cfg.get("capture.wgc.window_name"))

    def __post_init__(self) -> None:
        # ── Commun ──
        if not isinstance(self.screen_w, int) or self.screen_w <= 0:
            raise ValueError(
                f"screen.width doit être un int > 0, reçu {self.screen_w!r}"
            )
        if not isinstance(self.screen_h, int) or self.screen_h <= 0:
            raise ValueError(
                f"screen.height doit être un int > 0, reçu {self.screen_h!r}"
            )
        if not isinstance(self.capture_fps, int) or self.capture_fps <= 0:
            raise ValueError(
                f"screen.capture_fps doit être un int > 0, reçu {self.capture_fps!r}"
            )
        if not isinstance(self.source_priority, list) or not self.source_priority:
            raise ValueError(
                f"capture.source_priority doit être une liste non vide, "
                f"reçu {self.source_priority!r}"
            )
        allowed = {"dxcam", "cv2", "mss", "wgc"}
        invalid = [s for s in self.source_priority if s not in allowed]
        if invalid:
            raise ValueError(
                f"capture.source_priority contient des valeurs inconnues {invalid}, "
                f"autorisées : {sorted(allowed)}"
            )
        if not isinstance(self.probe_timeout_s, (int, float)) or self.probe_timeout_s <= 0:
            raise ValueError(
                f"capture.probe_timeout_s doit être > 0, reçu {self.probe_timeout_s!r}"
            )
        if not isinstance(self.probe_min_frames, int) or self.probe_min_frames < 1:
            raise ValueError(
                f"capture.probe_min_frames doit être un int >= 1, "
                f"reçu {self.probe_min_frames!r}"
            )

        # ── cv2 ──
        if not isinstance(self.cv2_device_name, str) or not self.cv2_device_name:
            raise ValueError(
                f"capture.cv2_device_name doit être une str non vide, "
                f"reçu {self.cv2_device_name!r}"
            )

        # ── mss ──
        if not isinstance(self.mss_monitor_index, int) or self.mss_monitor_index <= 0:
            raise ValueError(
                f"capture.mss_monitor_index doit être un int > 0, reçu {self.mss_monitor_index!r}"
            )

        # ── wgc ──
        if not isinstance(self.wgc_target, str) or self.wgc_target not in {"monitor", "window"}:
            raise ValueError(
                f"capture.wgc.target doit être 'monitor' ou 'window', "
                f"reçu {self.wgc_target!r}"
            )
        if self.wgc_target == "monitor":
            if not isinstance(self.wgc_monitor_index, int) or self.wgc_monitor_index < 0:
                raise ValueError(
                    f"capture.wgc.monitor_index doit être un int >= 0 "
                    f"(target='monitor'), reçu {self.wgc_monitor_index!r}"
                )
        if self.wgc_target == "window":
            if not isinstance(self.wgc_window_name, str) or not self.wgc_window_name:
                raise ValueError(
                    f"capture.wgc.window_name doit être une str non vide "
                    f"(target='window'), reçu {self.wgc_window_name!r}"
                )
