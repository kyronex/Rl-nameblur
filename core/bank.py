# core/bank.py
# ── Objectif ──────────────────────────────────────────────────────────────────
# Historiser les Mask (slow) et FastMaskView (fast) en JSONL append-only.
# Source de vérité pour prédiction + optimisation d'association.
#
# ── Schéma BankSnapshot v1.0 ──────────────────────────────────────────────────
# bank_version         : str  ("1.0")
# snap_ts              : float  (time.time())
# flush_reason         : str  ("periodic" | "stop")
# flush_frame_id       : int
# flush_frame_ts       : float
# flush_source         : str  ("slow" | "fast")  ← appelé par la branche slow ou fast
# entries              : list[dict]  — un dict par Mask ou FastMaskView observée
#
# Chaque entry slow (Mask) contient les champs gratuits sérialisables de Mask :
#   identité/géométrie, cinématique slow, cycle de vie scalaires, timestamps capture.

from __future__ import annotations
from bench import bench

from datetime import datetime
import json
import logging
import os
import time
from config import cfg
from typing import Literal

from core.mask import Mask, MaskState, FastMaskView, serialize_scores
from core.signature import serialize_signature, empty_signature

log = logging.getLogger("core.bank")

# ---------------------------------------------------------------------------
# Helpers de sérialisation
# ---------------------------------------------------------------------------

def _get(obj, attr: str, default=None):
    """Accès défensif : retourne default si l'attribut est absent."""
    return getattr(obj, attr, default)

def _mask_to_entry(mask: Mask) -> dict:
    """
    Sérialise UN Mask en dict JSON-safe.
    """
    return {
        # — Identité / Géométrie —
        "uid":           mask.uid,
        "rx":            mask.rect[0],
        "ry":            mask.rect[1],
        "rw":            mask.rect[2],
        "rh":            mask.rect[3],
        "created_frame_id": _get(mask, "created_frame_id", -1),
        "frame_id": _get(mask, "frame_id", -1),
        "ldr_x":         _get(mask, "last_detected_rect", (0, 0, 0, 0))[0],
        "ldr_y":         _get(mask, "last_detected_rect", (0, 0, 0, 0))[1],
        "ldr_w":         _get(mask, "last_detected_rect", (0, 0, 0, 0))[2],
        "ldr_h":         _get(mask, "last_detected_rect", (0, 0, 0, 0))[3],
        "last_detected_ts": round(_get(mask, "last_detected_ts", 0.0), 4),
        # — last_fast_rect (slow) —
        "lfr_x":         _get(mask, "last_fast_rect", (0, 0, 0, 0))[0] if _get(mask, "last_fast_rect") is not None else None,
        "lfr_y":         _get(mask, "last_fast_rect", (0, 0, 0, 0))[1] if _get(mask, "last_fast_rect") is not None else None,
        "lfr_w":         _get(mask, "last_fast_rect", (0, 0, 0, 0))[2] if _get(mask, "last_fast_rect") is not None else None,
        "lfr_h":         _get(mask, "last_fast_rect", (0, 0, 0, 0))[3] if _get(mask, "last_fast_rect") is not None else None,
        # — last_slow_ts —
        "last_slow_ts":  round(_get(mask, "last_slow_ts", 0.0), 4),
        # — Source —
        "last_source":   _get(mask, "last_source", "unknown"),
        # — Cinématique slow —
        "vx":            round(_get(mask, "vx", 0.0), 2),
        "vy":            round(_get(mask, "vy", 0.0), 2),
        "vw":            round(_get(mask, "vw", 0.0), 2),
        "vh":            round(_get(mask, "vh", 0.0), 2),
        # — Confidence —
        "confidence":    round(_get(mask, "confidence", 0.0), 4),
        # — fast_miss_count —
        "fast_miss_count": _get(mask, "fast_miss_count", 0),
        # — Cycle de vie : état —
        "state":         _get(mask, "state", MaskState.PENDING).name,
        # — Cycle de vie : compteurs —
        "frames_matched":  _get(mask, "frames_matched", 0),
        "total_matches_cumul": _get(mask, "total_matches_cumul", 0),
        # — Timestamps perf_counter —
        "last_seen_ts":  round(_get(mask, "last_seen_ts", 0.0), 4),
        "lost_since_ts": round(_get(mask, "lost_since_ts", None), 4) if _get(mask, "lost_since_ts") is not None else None,
        "created_ts":    round(_get(mask, "created_ts", 0.0), 4),
        # — Timestamps capture —
        "last_seen_frame_ts":    round(_get(mask, "last_seen_frame_ts", 0.0), 4),
        "lost_since_frame_ts":   round(_get(mask, "lost_since_frame_ts", None), 4) if _get(mask, "lost_since_frame_ts") is not None else None,
        "scores":        serialize_scores(_get(mask, "scores", {})),
        "hash_history":  list(_get(mask, "hash_history", [])),
        **serialize_signature(_get(mask, "signature", None)),
    }


def _fast_view_to_entry(view: FastMaskView) -> dict:
    """
    Sérialise UN FastMaskView en dict JSON-safe.
    """
    return {
        # — Identité / Géométrie —
        "uid":           view.uid,
        "rx":            view.rect[0],
        "ry":            view.rect[1],
        "rw":            view.rect[2],
        "rh":            view.rect[3],
        "created_frame_id": _get(view, "created_frame_id", -1),
        "frame_id": _get(view, "frame_id", -1),
        "last_detected_ts": round(_get(view, "last_detected_ts", 0.0), 4),
        # — current_rect (fast — similaire à last_fast_rect pour Mask) —
        "lfr_x":         view.current_rect[0] if view.current_rect is not None else None,
        "lfr_y":         view.current_rect[1] if view.current_rect is not None else None,
        "lfr_w":         view.current_rect[2] if view.current_rect is not None else None,
        "lfr_h":         view.current_rect[3] if view.current_rect is not None else None,
        # — last_slow_ts —
        "last_slow_ts":  round(_get(view, "last_slow_ts", 0.0), 4),
        # — Source —
        "last_source":   _get(view, "last_source", "unknown"),
        # — total_matches_cumul (fast a le champ, ligne 48) —
        "total_matches_cumul": _get(view, "total_matches_cumul", 0),
        # — frames_matched (fast a le champ, ligne 49) —
        "frames_matched":  _get(view, "frames_matched", 0),
        # — fast_miss_count (fast a le champ, ligne 50) —
        "fast_miss_count": _get(view, "fast_miss_count", 0),
        # — created_ts —
        "created_ts":    round(_get(view, "created_ts", 0.0), 4),
        # — lost_since_ts —
        "lost_since_ts": round(_get(view, "lost_since_ts", None), 4) if _get(view, "lost_since_ts") is not None else None,
        # — Cinématique slow —
        "vx":            round(_get(view, "vx", 0.0), 2),
        "vy":            round(_get(view, "vy", 0.0), 2),
        "vw":            round(_get(view, "vw", 0.0), 2),
        "vh":            round(_get(view, "vh", 0.0), 2),
        # — Confidence —
        "confidence":    round(_get(view, "confidence", 0.0), 4),
        # — Cycle de vie : état —
        "state":         _get(view, "state", MaskState.PENDING).name,
        # — Cinématique fast (properties déléguées à fast_kin) —
        "vx_f":          round(view.vx_f, 2),          # propriété ligne 54
        "vy_f":          round(view.vy_f, 2),          # propriété ligne 57
        "last_fast_ts":  round(view.last_fast_ts, 4),  # propriété ligne 60

        "scores":        serialize_scores(_get(view, "scores", {})),
        "hash_history":  list(_get(view, "hash_history", [])),
        **serialize_signature(_get(view, "signature", None)),
    }


# ---------------------------------------------------------------------------
# Banque
# ---------------------------------------------------------------------------

class Bank:
    """
    Historisation append-only en JSONL.
    Appel typique (slow) ::
        bank.add_observation(masks=tracker.all_masks(), source="slow",frame_id=frame_id, frame_ts=frame_ts)
        bank.flush_if_needed(frame_id)
    Appel typique (fast) ::
        bank.add_observation(masks=fast_tracker.get_results(), source="fast",frame_id=frame_id, frame_ts=frame_ts)
        bank.flush_if_needed(frame_id)
    stop() appelé dans le finally de main.py.
    """

    def __init__(self):
        self._base, self._ext = os.path.splitext(cfg.get("tracker.bank.path", "logs/json/bank.jsonl"))
        self._path = self._build_path(bench._session_id)
        self._flush_every: int = cfg.get("tracker.bank.push_every")
        self._pending: list[dict] = []
        self._last_flush_frame_id: int = 0
        self._check_signature_roundtrip()

    def _check_signature_roundtrip(self) -> None:
        """
        Invariant de démarrage : vérifie que serialize_signature() produit
        toujours un JSON valide sur une signature vide. Échec bruyant au
        lancement plutôt qu'au premier flush réel (bug de sérialisation
        détecté avant production).
        """
        try:
            json.dumps(serialize_signature(empty_signature()))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"[Bank] Invariant signature échoué au démarrage : {exc}"
            ) from exc

    def _build_path(self, index: int) -> str:
            suffix = f"_{index}"
            return f"{self._base}{suffix}{self._ext}" if self._ext else f"{self._base}{suffix}"

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def add_observation(self,source: Literal["slow", "fast"],frame_id: int,frame_ts: float,masks: list[Mask],views : list[FastMaskView]) -> None:
        """Accumule une observation pour le prochain flush."""
        if source == "slow":
            entries = [_mask_to_entry(m) for m in masks]
        else:
            entries = [_fast_view_to_entry(v) for v in views]
        if not entries:
            return
        self._pending.append({
            "bank_version":        "1.0",
            "snap_ts":             time.time(),
            "flush_reason":        "periodic",
            "flush_frame_id":            frame_id,
            "flush_frame_ts":            frame_ts,
            "flush_source":              source,
            "entries":             entries,
        })

    def flush_if_needed(self, frame_id: int) -> None:
        """
        Déclenche un flush quand flush_every frames se sont écoulées
        depuis le dernier flush.
        Appelé depuis la boucle principale uniquement (pas de lock).
        """
        if self._last_flush_frame_id > 0:
            if frame_id - self._last_flush_frame_id < self._flush_every:
                return
        if not self._pending:
            return
        self._do_flush("periodic", frame_id)

    def stop(self) -> None:
        """
        Force le flush final (reason='stop') avant fermeture propre.
        Appelé dans le finally de main.py.
        """
        if self._pending:
            last = self._pending[-1]
            self._do_flush("stop", last["flush_frame_id"])

    # ------------------------------------------------------------------
    # Interne
    # ------------------------------------------------------------------

    def _do_flush(self, reason: str, frame_id: int) -> None:
            """
            Écrit toutes les lignes pending en une opération append atomique.
            - TypeError/ValueError (bug de sérialisation, non récupérable côtéI/O) → log.error + compteur bench, PAS de clear() : les données restent en mémoire pour un futur flush après correctif (au prix d'une croissance mémoire assumée en dev — à surveiller si le bug persiste plusieurs cycles).
            - OSError (disque plein, permission, transitoire) → log.warning, comportement identique à avant : clear() pour ne pas boucler indéfiniment sur la même écriture.
            """
            try:
                # Met à jour flush_reason sur chaque entrée avant écriture
                for snap in self._pending:
                    snap["flush_reason"] = reason
                lines = [json.dumps(snap, ensure_ascii=False) for snap in self._pending]
            except (TypeError, ValueError) as exc:
                bench.count("bank_flush_error_total")
                log.error("[Bank] Échec sérialisation (données conservées) : %s", exc)
                return
            try:
                with open(self._path, "a", encoding="utf-8") as fh:
                    for line in lines:
                        fh.write(line + "\n")
                #log.info("[Bank] Flush %s — %d lignes — frame_id=%d",reason, len(self._pending), frame_id)
            except OSError as exc:
                log.warning("[Bank] Échec écriture %s : %s", self._path, exc)
            self._pending.clear()
            self._last_flush_frame_id = frame_id
