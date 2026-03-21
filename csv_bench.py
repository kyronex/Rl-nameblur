# csv_bench.py — CSV benchmark extraction
import os
import time
import csv

CSV_HEADERS = [
    "timestamp",
    "frame_id",
    "loop_ms",
    "capture_wait_ms",
    "slow_poll_ms",
    "fast_poll_ms",
    "predict_ms",
    "blur_ms",
    "send_ms",
    "detect_age_ms",
    "fast_age_ms",
    "mask_age_avg_ms",
    "slow_updated",
    "fast_updated",
    "predicted",
    "mask_count",
    "jitter_center_px",
    "jitter_corners_px",
    "masks_created",
    "masks_killed",
]

_csv_file = None
_csv_writer = None


def csv_open():
    global _csv_file, _csv_writer
    os.makedirs("logs", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = f"logs/bench_{ts}.csv"
    _csv_file = open(path, "w", newline="")
    _csv_writer = csv.DictWriter(_csv_file, fieldnames=CSV_HEADERS)
    _csv_writer.writeheader()
    print(f"📊 CSV benchmark → {path}")


def csv_write(row: dict):
    if _csv_writer is not None:
        _csv_writer.writerow(row)


def csv_flush():
    if _csv_file is not None:
        _csv_file.flush()


def csv_close():
    global _csv_file, _csv_writer
    if _csv_file is not None:
        _csv_file.flush()
        _csv_file.close()
        _csv_file = None
        _csv_writer = None
