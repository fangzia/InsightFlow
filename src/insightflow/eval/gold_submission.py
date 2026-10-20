"""构造 gold 标准提交，用于对官方 Spider 2.0-DBT harness 做冒烟验证。

将每个任务的 gold 答案文件复制为标准提交目录结构
（``<instance_id>/<文件>`` + ``results_metadata.jsonl``），
从而可以端到端验证官方 ``evaluate.py``。
gold 提交必须得到 1.0 分；任何低于 1.0 的结果都意味着
harness 或数据本身存在问题。

用法::

    uv run python -m insightflow.eval.gold_submission [--limit N]
"""

import argparse
import json
import shutil
from pathlib import Path

from insightflow.core.config import get_settings


def build_gold_submission(limit: int | None = None) -> Path:
    settings = get_settings()
    gold_dir = settings.spider2_dbt_dir / "evaluation_suite" / "gold"
    eval_jsonl = gold_dir / "spider2_eval.jsonl"
    out_dir = settings.spider2_dbt_dir / "evaluation_suite" / "gold_submission_smoke"

    if not eval_jsonl.exists():
        raise FileNotFoundError(f"Gold definitions not found: {eval_jsonl}. Run data setup first.")

    entries = [json.loads(line) for line in eval_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        entries = entries[:limit]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    metadata = []
    skipped = []
    for entry in entries:
        instance_id = entry["instance_id"]
        eval_spec = entry["evaluation"]
        # 所有 spider2-dbt 任务都使用 duckdb_match，且只有一个 gold 文件。
        if eval_spec["func"] != "duckdb_match":
            raise ValueError(f"Unexpected eval func for {instance_id}: {eval_spec['func']}")
        gold_file = eval_spec["parameters"]["gold"]

        instance_gold_dir = gold_dir / instance_id
        src = instance_gold_dir / gold_file
        if not src.exists():
            # 上游数据缺陷容错：少数实例没有发布 gold 答案（跳过）；
            # 另一些实例的 gold 文件名与 eval 定义不一致
            # （回退到该目录下唯一的 .duckdb 文件）。
            if not instance_gold_dir.exists():
                skipped.append(f"{instance_id} (no gold folder)")
                continue
            candidates = list(instance_gold_dir.glob("*.duckdb"))
            if len(candidates) != 1:
                skipped.append(f"{instance_id} (gold file mismatch: expected {gold_file})")
                continue
            # 规范化 gold 目录，使官方 evaluate.py
            # （按 gold_dir/<instance_id>/<eval spec 中的文件名> 拼路径）能找到文件。
            shutil.copy2(candidates[0], instance_gold_dir / gold_file)
            src = instance_gold_dir / gold_file
            print(f"WARNING: {instance_id}: normalized gold '{candidates[0].name}' -> '{gold_file}' per eval spec")

        instance_out = out_dir / instance_id
        instance_out.mkdir()
        shutil.copy2(src, instance_out / gold_file)

        metadata.append(
            {
                "instance_id": instance_id,
                "answer_type": "file",
                "answer_or_path": gold_file,
            }
        )

    if skipped:
        print(f"Skipped {len(skipped)} instances without usable gold answers:")
        for item in skipped:
            print(f"  - {item}")

    with (out_dir / "results_metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in metadata:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote gold submission for {len(metadata)} instances -> {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="只构建前 N 个实例")
    args = parser.parse_args()
    build_gold_submission(limit=args.limit)


if __name__ == "__main__":
    main()
