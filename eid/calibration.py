# eid/calibration.py
"""Dump JSONL des paires évaluées pour calibration."""

import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Optional

SCHEMA_VERSION = 1


class CalibrationWriter:
    def __init__(self, path: str, session_id: Optional[str] = None, mode: str = "full"):
        self._path = path
        self._fh = None
        self._session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.n_written = 0
        self._mode = mode

    @property
    def session_id(self) -> str:
        return self._session_id

    def _ensure_open(self):
        if self._fh is None:
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
            self._fh = open(self._path, "a", encoding="utf-8")

    def _write(self, row: dict):
        row["schema_version"] = SCHEMA_VERSION
        row["session_id"] = self._session_id
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def write_manifest(self, config: dict, input_path: str, extra: Optional[dict] = None):
        self._ensure_open()
        row = {
            "record": "manifest",
            "created": datetime.now().isoformat(timespec="seconds"),
            "input": input_path,
            "config": config,
        }
        if extra:
            row.update(extra)
        self._write(row)

    def add_record(self, rec):
        if rec.seq is None:
            raise ValueError("CalibrationRecord.seq ne peut pas être None")
        self._ensure_open()
        row = {
            "seq": rec.seq,
            "frame_id": rec.frame_id,
            "ts": round(rec.ts, 4),
            "obs_uid": rec.obs_uid,
            "cluster_id": rec.cluster_id,
            "eid": rec.eid,
            "gap_s": round(rec.gap_s, 4),
            "cost": round(float(rec.cost), 6),
            "cost_sig": round(float(rec.cost_sig), 6),
            "mode": self._mode,
            "accepted": bool(rec.accepted),
            "gated": bool(rec.dist.gated),
            "gate_reason": rec.dist.gate_reason,
        }
        # Exporter les diagnostics de SigDistance, mais pas dist.cost :
        # le coût final est déjà à la racine et cost_sig est explicite.
        dist_row = asdict(rec.dist)
        dist_row.pop("cost", None)
        row.update(dist_row)
        self._write(row)
        self.n_written += 1

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
