#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="/home/gexinggexing/miniconda3/bin/conda"
PY=("$CONDA_BIN" run --no-capture-output -n timesfm2 python -u)

echo "started_at=$(date -Iseconds)"
echo "cwd=$(pwd)"
echo "python=$("${PY[@]}" --version 2>&1)"

"${PY[@]}" run_sota_bcic2a.py
"${PY[@]}" run_sota_seed.py
"${PY[@]}" run_sota_chinese.py
"${PY[@]}" run_sota_mdd.py
"${PY[@]}" run_sota_sleep.py
"${PY[@]}" run_sota_finalize.py --update-submission

"${PY[@]}" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

summary_path = Path("artifacts/results/sota_final_selection_summary.json")
payload = {
    "summary_path": str(summary_path),
    "summary_exists": summary_path.exists(),
}
if summary_path.exists():
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["rows"] = rows
    payload["selected_for_submission"] = [
        row["dataset"] for row in rows if row.get("selected_for_submission")
    ]
    payload["submission_updated"] = [
        row["dataset"] for row in rows if row.get("submission_updated")
    ]

summary_file = os.environ.get("CODEX_SUMMARY_FILE")
if summary_file:
    Path(summary_file).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

print(json.dumps(payload, indent=2, ensure_ascii=False))
PY

echo "finished_at=$(date -Iseconds)"
