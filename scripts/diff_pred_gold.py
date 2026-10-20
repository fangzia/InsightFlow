"""通用诊断：对比某个实例 pred vs gold 各条件表的行数与逐列值差异。"""

import json
import sys
from pathlib import Path

import duckdb

GOLD_DIR = Path("third_party/Spider2/spider2-dbt/evaluation_suite/gold")
EVAL = json.loads(
    Path("third_party/Spider2/spider2-dbt/evaluation_suite/gold/spider2_eval.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()[0]
)  # 占位，不用


def load_specs(instance_id: str):
    specs = {}
    for line in (GOLD_DIR / "spider2_eval.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e["instance_id"] == instance_id:
            specs = e
    params = specs["evaluation"]["parameters"]
    return params["gold"], params.get("condition_tabs", []), params.get("condition_cols", [])


def table_diff(gold_path: str, pred_path: str, table: str, cond_idx: list[int] | None):
    con_g = duckdb.connect(gold_path, read_only=True)
    con_p = duckdb.connect(pred_path, read_only=True)
    try:
        gcols = [r[0] for r in con_g.execute(
            "select column_name from information_schema.columns where table_name=? order by ordinal_position", [table]
        ).fetchall()]
        pcols = [r[0] for r in con_p.execute(
            "select column_name from information_schema.columns where table_name=? order by ordinal_position", [table]
        ).fetchall()]
        grows = con_g.execute(f'select * from "{table}"').fetchall()  # noqa: S608
        prows = con_p.execute(f'select * from "{table}"').fetchall()  # noqa: S608
        print(f"== {table}: gold {len(grows)}x{len(gcols)} | pred {len(prows)}x{len(pcols)}")
        if gcols != pcols:
            print(f"   列不一致!\n   gold: {gcols}\n   pred: {pcols}")
            return
        # 官方按 cond_idx 只检查 gold 的这些列；这里全列对比以定位问题
        for i, col in enumerate(gcols):
            gvals = sorted(str(r[i]) for r in grows)
            pvals = sorted(str(r[i]) for r in prows)
            tag = "CHECKED" if cond_idx is None or i in cond_idx else "ignored"
            if gvals == pvals:
                print(f"   [{col}] 一致 ({tag})")
            else:
                only_g = [v for v in gvals if v not in pvals][:4]
                only_p = [v for v in pvals if v not in gvals][:4]
                print(f"   [{col}] 不同 ({tag}): 仅gold={only_g} 仅pred={only_p}")
    finally:
        con_g.close()
        con_p.close()


def main(instance_id: str, pred_dir: str):
    gold_file, tabs, cols_idx = load_specs(instance_id)
    gold_path = str(GOLD_DIR / instance_id / gold_file)
    pred_path = str(Path(pred_dir) / instance_id / gold_file)
    for j, t in enumerate(tabs):
        table_diff(gold_path, pred_path, t, cols_idx[j] if cols_idx else None)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
