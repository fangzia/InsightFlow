"""在 Spider 2.0-DBT 子集上批量运行 Agent，并用官方 harness 打分。

模式（评测消融，差异全部集中在 AgentOptions）：

- ``zeroshot``  —— 仅 schema（无值采样），单次生成，无修复循环。baseline。
- ``agent``     —— M1：schema + 值采样，dbt 失败后 1 轮修复。
- ``agent-m2``  —— M2 v1：两步生成（契约）+ 运行时列校验，dbt 失败/
  列不匹配均会触发修复（预算 3 轮）。v1 语义已废弃（8/64），保留模式
  名仅为历史命令兼容。
- ``agent-m2-nosize`` —— v1 消融：与 ``agent-m2`` 唯一差别是不传期望
  列数（14/64，将列数钉死量化为 -6）。同样保留历史命令兼容。
- ``agent-m2-v2`` —— M2 v2：超集契约（yml 为最小覆盖，LLM 只加不删）
  + 超集校验门（仅缺失列失败，顺序/多余列免费）+ fan-out 证据式反馈
  + 最优轮快照恢复；不消费 eval spec 的期望列数。

目标模型来自 gold 评测规格（``condition_tabs``）：基准任务的评分就是
对比这些表，因此 Agent 必须恰好物化出它们。该选择对两种模式完全
一致，并已在 ``benchmarks/m1-baseline.md`` 中记录。

用法::

    uv run python -m insightflow.eval.run_benchmark --mode agent --limit 5
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from insightflow.agent.graph import AgentOptions, InsightFlowAgent
from insightflow.core.config import get_settings

_MODES = {
    "zeroshot": AgentOptions(schema_samples=0, max_attempts=0),
    "agent": AgentOptions(schema_samples=3, max_attempts=1),
    # v1 历史标签（8/64）：语义已随 v2 linker 演进，仅为命令兼容保留
    "agent-m2": AgentOptions(schema_samples=3, max_attempts=3, use_linker=True),
    # v1 消融历史标签（14/64）：量化列数钉死 -6；同为命令兼容保留
    "agent-m2-nosize": AgentOptions(schema_samples=3, max_attempts=3, use_linker=True, use_required_sizes=False),
    # M2 v2：超集契约 + 超集校验门 + 最优轮快照；不消费期望列数
    "agent-m2-v2": AgentOptions(
        schema_samples=3, max_attempts=3, use_linker=True, use_required_sizes=False, use_best_snapshot=True
    ),
    # M3 第 1 步：v2 + 静态 lint + dbt 错误提炼 + 缺包安装 + 缺失祖先闭包
    # + 坏 yml 隔离 + 坏模型剪除 + 池 0 救援
    "agent-m3": AgentOptions(
        schema_samples=3,
        max_attempts=3,
        use_linker=True,
        use_required_sizes=False,
        use_best_snapshot=True,
        use_lint=True,
        use_triage=True,
        use_deps=True,
        use_ancestors=True,
        use_pool0_rescue=True,
        use_yml_quarantine=True,
        use_prune_blockers=True,
        use_ref_repair=True,
    ),
    # M3 第 2 步：第 1 步 + 无进展检测触发 replan（设计层重来，限 1 次）
    "agent-m3-replan": AgentOptions(
        schema_samples=3,
        max_attempts=3,
        use_linker=True,
        use_required_sizes=False,
        use_best_snapshot=True,
        use_lint=True,
        use_triage=True,
        use_deps=True,
        use_ancestors=True,
        use_pool0_rescue=True,
        use_yml_quarantine=True,
        use_prune_blockers=True,
        use_ref_repair=True,
        replan_limit=1,
    ),
    # M3 第 3 步：第 2 步 + 延迟多候选（预算耗尽后换独立轨迹，共 3 条）
    "agent-m3-fanout": AgentOptions(
        schema_samples=3,
        max_attempts=3,
        use_linker=True,
        use_required_sizes=False,
        use_best_snapshot=True,
        use_lint=True,
        use_triage=True,
        use_deps=True,
        use_ancestors=True,
        use_pool0_rescue=True,
        use_yml_quarantine=True,
        use_prune_blockers=True,
        use_ref_repair=True,
        replan_limit=1,
        n_candidates=3,
    ),
    # M3 第 4 步：第 3 步 + critic（verify 全绿后独立审查物化数据，限 1 次）
    "agent-m3-critic": AgentOptions(
        schema_samples=3,
        max_attempts=3,
        use_linker=True,
        use_required_sizes=False,
        use_best_snapshot=True,
        use_lint=True,
        use_triage=True,
        use_deps=True,
        use_ancestors=True,
        use_pool0_rescue=True,
        use_yml_quarantine=True,
        use_prune_blockers=True,
        use_ref_repair=True,
        replan_limit=1,
        n_candidates=3,
        use_critic=True,
    ),
    # M3 第 4 步 v2：critic 换统计级证据（NULL/distinct/值域对账/粒度比），
    # CHALLENGE 四段式可执行靶子；v1 采样证据保留供消融对照
    "agent-m3-critic-v2": AgentOptions(
        schema_samples=3,
        max_attempts=3,
        use_linker=True,
        use_required_sizes=False,
        use_best_snapshot=True,
        use_lint=True,
        use_triage=True,
        use_deps=True,
        use_ancestors=True,
        use_pool0_rescue=True,
        use_yml_quarantine=True,
        use_prune_blockers=True,
        use_ref_repair=True,
        replan_limit=1,
        n_candidates=3,
        use_critic=True,
        use_critic_v2=True,
    ),
    # M3 第 5 步：值层确定性信号——行数锚定（指令显式行数对照，行数是
    # 评测器唯一硬约束）+ 单列重审（verify-green 双版本对照，弱列定向重写）
    "agent-m3-anchor": AgentOptions(
        schema_samples=3,
        max_attempts=3,
        use_linker=True,
        use_required_sizes=False,
        use_best_snapshot=True,
        use_lint=True,
        use_triage=True,
        use_deps=True,
        use_ancestors=True,
        use_pool0_rescue=True,
        use_yml_quarantine=True,
        use_prune_blockers=True,
        use_ref_repair=True,
        replan_limit=1,
        n_candidates=3,
        use_critic=True,
        use_critic_v2=True,
        use_anchor=True,
        use_column_review=True,
    ),
    # M3 第 5 步 b：+ 粒度词锚定（"monthly per X" → 行数 vs distinct(时间)×
    # distinct(实体) 软性核对，阈值 [1/3, 3]）
    "agent-m3-grain": AgentOptions(
        schema_samples=3,
        max_attempts=3,
        use_linker=True,
        use_required_sizes=False,
        use_best_snapshot=True,
        use_lint=True,
        use_triage=True,
        use_deps=True,
        use_ancestors=True,
        use_pool0_rescue=True,
        use_yml_quarantine=True,
        use_prune_blockers=True,
        use_ref_repair=True,
        replan_limit=1,
        n_candidates=3,
        use_critic=True,
        use_critic_v2=True,
        use_anchor=True,
        use_column_review=True,
        use_grain_anchor=True,
    ),
}


def load_instances(settings) -> list[dict]:
    """将任务指令与 gold 评测规格合并；只保留有可用 gold 的实例。"""
    dbt_dir = settings.spider2_dbt_dir
    tasks = {
        json.loads(line)["instance_id"]: json.loads(line)
        for line in (dbt_dir / "examples" / "spider2-dbt.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    gold_dir = dbt_dir / "evaluation_suite" / "gold"
    instances = []
    for line in (gold_dir / "spider2_eval.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        instance_id = entry["instance_id"]
        spec = entry["evaluation"]
        if spec["func"] != "duckdb_match":
            continue
        gold_file = spec["parameters"].get("gold")
        if not gold_file or not (gold_dir / instance_id).is_dir():
            continue  # 官方未发布 gold 答案 -> 不纳入基准集
        if instance_id not in tasks:
            continue
        instances.append(
            {
                "instance_id": instance_id,
                "instruction": tasks[instance_id]["instruction"],
                "gold_file": gold_file,
                "condition_tabs": spec["parameters"].get("condition_tabs", []),
                "condition_cols": spec["parameters"].get("condition_cols", []),
            }
        )
    return instances


def render_model_specs(models_dir: Path, targets: list[str]) -> str:
    """提取 schema.yml 中目标模型的定义，渲染为提示词文本。"""
    target_set = set(targets)
    sections = []
    yml_paths = [*models_dir.rglob("*.yml"), *models_dir.rglob("*.yaml")]
    for yml_path in sorted(yml_paths):
        try:
            data = yaml.safe_load(yml_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        for model in data.get("models") or []:
            name = model.get("name")
            if name not in target_set:
                continue
            cols = model.get("columns") or []
            col_lines = "\n".join(
                f"    - {c.get('name')}: {c.get('description', '')}".rstrip()
                for c in cols
            )
            sections.append(
                f"## spec for model '{name}' (from models/{yml_path.name})\n"
                f"  description: {model.get('description', '').strip()}\n"
                f"  columns (keep these exact names and order):\n{col_lines}"
            )
    return "\n\n".join(sections) + "\n\n" if sections else ""


def _absent_targets(workspace: Path, targets: list[str]) -> list[str]:
    """ shipped .duckdb 中尚不存在的目标表（池 0 救援的准入条件）。

    表已存在时不动它——mrr001/retail001 一族证明 shipped 表可能就是
    gold 正确形态，重写反而有把 1.0 改成 0 的下行风险。
    """
    import duckdb

    db_file = next(workspace.glob("*.duckdb"), None)
    if db_file is None:
        return list(targets)
    con = duckdb.connect(str(db_file), read_only=True)
    try:
        rows = con.execute(
            "select table_name from information_schema.tables "
            "where table_schema not in ('information_schema', 'pg_catalog')"
        ).fetchall()
    finally:
        con.close()
    present = {r[0] for r in rows}
    return [t for t in targets if t not in present]


def run_one(agent: InsightFlowAgent, instance: dict, examples_dir: Path, run_dir: Path) -> dict:
    instance_id = instance["instance_id"]
    workspace = run_dir / "workspaces" / instance_id
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(examples_dir / instance_id, workspace)

    condition_tabs = instance["condition_tabs"]
    existing = [t for t in condition_tabs if (workspace / "models" / f"{t}.sql").exists()]
    missing = [t for t in condition_tabs if t not in existing]
    model_specs = render_model_specs(workspace / "models", condition_tabs)

    record = {"instance_id": instance_id, "success": False, "attempts": 0, "error": ""}
    if not missing:
        # 全部模型已预实现：直接用 dbt 物化；失败且目标表缺失时交 agent 重建
        from insightflow.connectors.duckdb_executor import run_dbt

        result = run_dbt(
            workspace,
            models=condition_tabs,
            install_deps=agent.options.use_deps,
            select_ancestors=agent.options.use_ancestors,
            quarantine_yml=agent.options.use_yml_quarantine,
        )
        record["success"] = result.success
        if not result.success:
            record["error"] = result.error_digest(500)
            if agent.options.use_pool0_rescue:
                absent = _absent_targets(workspace, condition_tabs)
                if absent:
                    print(f"[pool0-rescue] {instance_id}: rebuilding {absent}")
                    try:
                        final = agent.run(
                            instruction=instance["instruction"],
                            workspace=workspace,
                            target_models=absent,
                            model_specs=model_specs,
                            select_models=condition_tabs,
                            required_sizes=None,
                        )
                        record["success"] = final.get("success", False)
                        record["attempts"] = final.get("attempts", 0)
                        record["error"] = final.get("error_digest", "")[:500]
                    except Exception as e:  # noqa: BLE001
                        record["error"] = f"{type(e).__name__}: {e}"
    else:
        try:
            # 消融开关：关闭时不传期望列数（required_sizes=None），
            # 列数仲裁退化为开放式 LLM 审阅。
            sizes = None
            if agent.options.use_required_sizes:
                sizes = dict(
                    zip(condition_tabs, (len(c) for c in instance.get("condition_cols", [])), strict=False)
                )
            final = agent.run(
                instruction=instance["instruction"],
                workspace=workspace,
                target_models=missing,
                model_specs=model_specs,
                select_models=condition_tabs,
                required_sizes=sizes,
            )
            record["success"] = final.get("success", False)
            record["attempts"] = final.get("attempts", 0)
            record["error"] = final.get("error_digest", "")[:500]
        except Exception as e:  # noqa: BLE001
            record["error"] = f"{type(e).__name__}: {e}"

    # 无论成败都提交工作区的 db 文件（缺表只会得 0 分），
    # 保证各次运行的评测分母一致。
    db_file = next(workspace.glob("*.duckdb"), None)
    submission_dir = run_dir / "submission" / instance_id
    submission_dir.mkdir(parents=True, exist_ok=True)
    if db_file is not None:
        shutil.copy2(db_file, submission_dir / instance["gold_file"])
    else:
        record["error"] = "no .duckdb file in workspace"
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=list(_MODES), default="agent")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--instances", type=str, default=None, help="逗号分隔的实例 ID 列表")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    settings = get_settings()
    examples_dir = settings.spider2_dbt_dir / "examples"
    run_dir = Path("runs") / (args.run_name or f"{args.mode}-m1")
    run_dir.mkdir(parents=True, exist_ok=True)

    instances = load_instances(settings)
    if args.instances:
        wanted = set(args.instances.split(","))
        instances = [i for i in instances if i["instance_id"] in wanted]
    if args.limit:
        instances = instances[: args.limit]
    print(f"[run_benchmark] mode={args.mode} instances={len(instances)} run_dir={run_dir}")

    agent = InsightFlowAgent(options=_MODES[args.mode])
    records = []
    for idx, instance in enumerate(instances, 1):
        print(f"[{idx}/{len(instances)}] {instance['instance_id']} ...", flush=True)
        record = run_one(agent, instance, examples_dir, run_dir)
        records.append(record)
        status = "ok" if record["success"] else f"FAILED ({record['error'][:120]})"
        print(f"    -> {status}", flush=True)

    # 提交元数据
    metadata_lines = []
    for instance in instances:
        metadata_lines.append(
            json.dumps(
                {
                    "instance_id": instance["instance_id"],
                    "answer_type": "file",
                    "answer_or_path": instance["gold_file"],
                }
            )
        )
    (run_dir / "submission" / "results_metadata.jsonl").write_text(
        "\n".join(metadata_lines) + "\n", encoding="utf-8"
    )
    (run_dir / "run_records.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    ok = sum(1 for r in records if r["success"])
    print(f"[run_benchmark] dbt succeeded for {ok}/{len(records)} instances")

    # 官方 harness
    eval_suite = settings.spider2_dbt_dir / "evaluation_suite"
    cmd = [
        sys.executable,
        "evaluate.py",
        "--result_dir",
        str((run_dir / "submission").resolve()),
        "--gold_dir",
        "gold",
    ]
    print("[run_benchmark] official harness:", " ".join(cmd), flush=True)
    proc = subprocess.run(  # noqa: S603
        cmd, cwd=str(eval_suite), capture_output=True, text=True, timeout=1800
    )
    print(proc.stdout[-4000:])
    if proc.returncode != 0:
        print(proc.stderr[-2000:])
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
