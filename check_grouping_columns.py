"""Where does `collector_id` actually live in this dataset?

Run this before the scan. It takes seconds and answers the one question the
calibration plan depends on: are the grouping columns in the per-frame data
parquet, or in the v3.0 episode metadata under `meta/episodes/`?

    python check_grouping_columns.py ~/lerobot-data/droid
"""

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

WANTED = (
    "collector_id", "building", "task_category", "date",
    "is_episode_successful", "operator_id", "session_id", "task_index",
)

root = Path(sys.argv[1] if len(sys.argv) > 1 else "~/lerobot-data/droid").expanduser()
print(f"dataset: {root}\n")

# --- what info.json declares -----------------------------------------------
info = json.loads((root / "meta" / "info.json").read_text())
features = info.get("features") or {}
print(f"codebase_version : {info.get('codebase_version')}")
print(f"total_episodes   : {info.get('total_episodes')}")
print(f"features declared: {len(features)}")
declared = [k for k in WANTED if k in features]
print(f"  wanted keys declared in features: {declared or 'none'}\n")

# --- what the per-frame data parquet actually contains ----------------------
data_files = sorted(root.glob("data/**/*.parquet"))
print(f"data parquet files: {len(data_files)}")
if data_files:
    cols = pq.read_schema(data_files[0]).names
    print(f"  columns in {data_files[0].name}: {len(cols)}")
    found = [c for c in WANTED if c in cols]
    print(f"  >>> wanted keys present in DATA: {found or 'NONE'}")
    if not found:
        print(f"      (first 12 columns: {cols[:12]})")

# --- what the v3.0 episode metadata contains --------------------------------
meta_files = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
print(f"\nmeta/episodes parquet files: {len(meta_files)}")
if meta_files:
    cols = pq.read_schema(meta_files[0]).names
    print(f"  columns in {meta_files[0].name}: {len(cols)}")
    found = [c for c in WANTED if c in cols]
    print(f"  >>> wanted keys present in META: {found or 'NONE'}")
    if not found:
        print(f"      (all columns: {cols})")
    else:
        table = pq.read_table(meta_files[0], columns=found)
        for key in found:
            values = table.column(key).to_pylist()
            uniq = {v for v in values if v is not None}
            print(f"      {key}: {len(uniq)} distinct over {len(values)} episodes")

print(
    "\nIf the wanted keys are in META and not in DATA, raftaar needs the v3.0\n"
    "episode-metadata reader (raftaar >= 0.4.2). If they are in DATA, 0.4.1\n"
    "already picks them up."
)
