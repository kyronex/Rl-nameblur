# bench/compare/bench_compare.py

from __future__ import annotations

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
import logging
from datetime import datetime

from bench.compare._config import (LOG_FORMAT,SCHEMA_VERSION,DIR_JSON,DIR_RESULTS,_r)
from bench.compare._io import (find_sessions,load_session,write_report,move_session_to_results)
from bench.compare._builder import (build_session_block,build_comparison)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    sessions = find_sessions()
    if len(sessions) == 0:
        log.error("Aucune session disponible.")
        sys.exit(1)
    sorted_ids = sorted(sessions.keys())
    target_id = sorted_ids[-1]
    absolute_id = sorted_ids[0] if len(sorted_ids) >= 2 else None
    relative_id = sorted_ids[-2] if len(sorted_ids) >= 3 else None
    log.info("Cible       : %s", target_id)
    log.info("Abs. réf.   : %s", absolute_id if absolute_id else "N/A (N==1)")
    log.info("Rel. réf.   : %s", relative_id if relative_id else "N/A (N<3)")
    target_agg, target_frame, target_fast = load_session(sessions[target_id])
    target_block = build_session_block(target_agg, target_frame, target_fast)
    abs_block = None
    if absolute_id:
        abs_agg, abs_frame, abs_fast = load_session(sessions[absolute_id])
        abs_block = build_session_block(abs_agg, abs_frame, abs_fast)
    rel_block = None
    if relative_id:
        rel_agg, rel_frame, rel_fast = load_session(sessions[relative_id])
        rel_block = build_session_block(rel_agg, rel_frame, rel_fast)
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(),
        "target_session": target_id,
        "target": target_block,
        "comparisons": {
            "absolute": (
                build_comparison(absolute_id, abs_block, target_block)
                if abs_block is not None
                else None
            ),
            "relative": (
                build_comparison(relative_id, rel_block, target_block)
                if rel_block is not None
                else None
            ),
        },
    }
    target_in_json = sessions[target_id]["agg"].is_relative_to(DIR_JSON)
    report_dir = DIR_RESULTS / target_id
    report_path = report_dir / f"{target_id}.json"
    if target_in_json:
        try:
            move_session_to_results(target_id, sessions[target_id])
        except OSError as exc:
            log.error("Échec déplacement session — rapport non écrit. %s", exc)
            sys.exit(1)
    else:
        report_dir.mkdir(parents=True, exist_ok=True)  # garantie explicite spec point 5
    try:
        write_report(report, report_path)
    except OSError as exc:
        log.error("Échec écriture rapport. %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
