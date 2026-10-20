"""Compare zeroshot vs agent full-run results."""

import json
import re

ZLOG = r"C:\Users\11360\AppData\Local\Temp\trae-agent-toolhost\jobs\job-0959db392ba4415fbfb1073ad0cd1763\output.log"
ALOG = r"C:\Users\11360\AppData\Local\Temp\trae-agent-toolhost\jobs\job-7ba371397105449fac018d1fc1d00bed\output.log"
ZRECORDS = "runs/zeroshot-m1v2/run_records.json"
ARECORDS = "runs/agent-m1v2/run_records.json"


def passed(text: str, score: str) -> list[str]:
    end = text.rfind(score)
    start = text.rfind("official harness")
    seg = text[start:end].splitlines()
    return [s.strip() for s in seg if re.fullmatch(r"[a-z][a-z0-9_]*", s.strip()) and s.strip() != "gold"]


zlog = open(ZLOG, encoding="utf-8", errors="replace").read()
alog = open(ALOG, encoding="utf-8", errors="replace").read()
zp, ap = passed(zlog, "0.21875"), passed(alog, "0.234375")
print(f"zeroshot passes ({len(zp)}):", sorted(zp))
print(f"agent passes    ({len(ap)}):", sorted(ap))
print("both:", sorted(set(zp) & set(ap)))
print("zeroshot only:", sorted(set(zp) - set(ap)))
print("agent only:", sorted(set(ap) - set(zp)))

zr = {r["instance_id"]: r for r in json.load(open(ZRECORDS, encoding="utf-8"))}
ar = {r["instance_id"]: r for r in json.load(open(ARECORDS, encoding="utf-8"))}
zok = sum(1 for r in zr.values() if r["success"])
aok = sum(1 for r in ar.values() if r["success"])
print(f"\ndbt success: zeroshot={zok}/64  agent={aok}/64")
fixed = sorted(k for k in ar if not zr[k]["success"] and ar[k]["success"])
regressed = sorted(k for k in ar if zr[k]["success"] and not ar[k]["success"])
print("agent fixed (dbt) vs zeroshot:", fixed)
print("agent regressed (dbt):", regressed)
att = [r["attempts"] for r in ar.values()]
print("agent attempts histogram:", {n: att.count(n) for n in sorted(set(att))})
