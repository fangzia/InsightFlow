"""Extract the real dbt error for every failed instance (re-run dbt, read stdout)."""

import json
import subprocess
from pathlib import Path

ar = json.load(open("runs/agent-m1/run_records.json", encoding="utf-8"))
fails = [r["instance_id"] for r in ar if not r["success"] and r["attempts"] > 0]

for iid in fails:
    ws = Path("runs/agent-m1/workspaces") / iid
    proc = subprocess.run(
        ["dbt", "run", "--project-dir", str(ws.resolve()), "--profiles-dir", str(ws.resolve()), "--no-use-colors"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    out = proc.stdout
    # last non-empty lines after the final "[ERROR]"
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    tail = " | ".join(lines[-4:])
    print(f"{iid}: {tail[:250]}")
