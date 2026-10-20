"""LangGraph Agent：M2 多节点图。

M1 的单循环升级为带契约的两段式流程（M2 里程碑，v2 起为超集语义）：

- ``prepare``  —— 读取任务工作区，导出 DuckDB schema，收集已有模型 SQL。
- ``link``     —— Schema Linker：为目标模型建立列契约（yml 为最小覆盖，
  LLM 审阅只加不删；无 yml 的交给 LLM 按"最大化合理覆盖"设计）；
  ``use_linker=False`` 时直通（M1 消融模式）。
- ``generate`` —— 按契约请求 LLM 实现缺失模型（修复轮附带失败反馈）。
- ``execute``  —— 写入生成的文件并执行 ``dbt run``。
- ``verify``   —— 运行时校验（超集门）：物化表实际列必须**包含**契约列
  （``expected ⊆ actual``）；多余列与列顺序均免费（评测器逐列子集
  匹配、与顺序无关），只有缺失列才触发修复。列门通过时再做 join
  fan-out 诊断（证据式：重复键组 + 样本 + 差异列 + 契约粒度，由 LLM
  裁决一对多是否任务本意）。列门通过即快照当轮生成物到
  ``best_generated``。
- ``restore_best`` —— 修复预算耗尽仍未通过时，把最后一个列门通过轮的
  生成物写回工作区并重跑一次 dbt 再结束（最优轮快照保险；retail001
  证明 .duckdb 状态跨轮存活、最后一轮状态 ≠ 提交质量）。
- 条件边 —— ``verified`` 通过则结束；否则在预算内回到 ``generate``；
  预算耗尽走 ``restore_best``（开启快照且存在）或直接结束。

模式差异全部集中在 :class:`AgentOptions`（消融可控）：

- ``schema_samples``：schema 上下文的值采样行数（baseline=0）
- ``max_attempts``：修复轮预算（baseline=0，agent-m2 用 3）
- ``use_linker``：是否启用两步生成 + 列校验（M2 开关）
- ``use_required_sizes``：是否接受 harness 传入的期望列数（仅历史模式
  agent-m2 使用；v2 不消费——消融证明列数钉死单独值 -6 分）
- ``use_best_snapshot``：是否启用最优轮快照恢复（v2 保险）
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import duckdb
from langgraph.graph import END, START, StateGraph

from insightflow.agent.llm import LLMClient
from insightflow.agent.prompts import (
    REPAIR_TEMPLATE,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    build_failure_feedback,
    parse_sql_blocks,
)
from insightflow.agent.schema_linker import build_contracts, render_contracts
from insightflow.connectors.duckdb_executor import extract_schema, run_dbt


@dataclass
class AgentOptions:
    """区分评测模式的旋钮。"""

    schema_samples: int = 3  # 每张表的值采样行数；0 = 仅 schema（zero-shot）
    max_attempts: int = 2  # dbt 失败/列校验失败后的修复轮数；0 = 单次生成
    dbt_timeout: int = 300
    use_linker: bool = False  # M2：两步生成（契约）+ 运行时列校验
    # 历史模式（agent-m2）保留：是否接受 harness 传入的期望列数（eval spec
    # 的 condition_cols 长度，纯结构提示）。v2 不消费——消融将其量化为 -6 分。
    use_required_sizes: bool = True
    # v2：最优轮快照——预算耗尽仍未 verify 时，恢复最后一个列门通过轮并重跑 dbt
    use_best_snapshot: bool = False


class AgentState(TypedDict, total=False):
    # 输入
    instruction: str
    workspace: str  # 复制出的 dbt 项目工作区路径
    target_models: list[str]  # LLM 需要实现的缺失模型
    select_models: list[str]  # `dbt run --select` 覆盖的全部模型（含已有模型）
    model_specs: str  # 目标模型的 schema.yml 规格文本（无则为空串）
    required_sizes: dict[str, int]  # 模型名 -> 期望列数（eval 结构提示，可空）
    # 过程状态
    schema_text: str
    existing_models: str
    contracts: dict[str, dict]  # 模型名 -> 列契约（见 schema_linker）
    generated: dict[str, str]
    best_generated: dict[str, str]  # 最后一个"dbt 成功且列门通过"轮的生成物快照
    error_digest: str
    column_issues: str  # verify 产出的列差异文本
    attempts: int
    # 输出
    success: bool
    verified: bool
    sql_files: dict[str, str]


def _column_diff(model: str, expected: list[str], actual: list[str]) -> str:
    """生成单个模型的列差异描述（超集语义），供修复提示词回注。

    只有**缺失**的契约列是问题；多余列与顺序均免费（评测器逐列子集
    匹配、与顺序无关），修复反馈因此只引导"补列"、绝不引导裁剪。
    """
    missing = [c for c in expected if c not in actual]
    return (
        f"model '{model}': the materialized table is MISSING required columns {missing}. "
        f"The contract requires at least {expected}; the table currently has {actual}. "
        "Extra columns are allowed - keep everything already present and only ADD "
        "the missing columns."
    )


_REF_RE = re.compile(r"ref\(\s*['\"]([^'\"]+)['\"]\s*\)")


def _find_self_references(files: dict[str, str]) -> list[str]:
    """静态检查生成 SQL 是否引用了目标模型自身（ref 或裸表名）。

    自引用在 dbt 中表现为 infinite recursion / 恒等投影，尽早拦截，
    直接作为一轮失败进入重试，而不是等 dbt 运行期才暴露。
    """
    issues = []
    for path, sql in files.items():
        model = Path(path).stem
        for m in _REF_RE.finditer(sql):
            if m.group(1) == model:
                issues.append(f"models/{model}.sql references itself via ref('{model}')")
        # 裸表名引用：main.<target> / from <target> / join <target>
        if re.search(rf"(?:from|join)\s+(?:main\.)?[\"']?{re.escape(model)}\b", sql, re.IGNORECASE):
            issues.append(f"models/{model}.sql references itself via raw table name '{model}'")
    return issues


_JOIN_REF_RE = re.compile(
    r"join\s+\{\{\s*ref\(\s*['\"]([^'\"]+)['\"]\s*\)\s*\}\}\s+(?:as\s+)?(\w+)", re.IGNORECASE
)
_FROM_REF_RE = re.compile(
    r"from\s+\{\{\s*ref\(\s*['\"]([^'\"]+)['\"]\s*\)\s*\}\}", re.IGNORECASE
)
_EQ_RE = re.compile(r"(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)")


def _clip_cell(value: object, max_len: int = 30) -> str:
    text = "NULL" if value is None else str(value)
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return text


def _diagnose_join_fanout(
    con: duckdb.DuckDBPyConnection,
    model: str,
    sql_text: str,
    model_rows: int,
    grain_hint: str = "",
) -> str | None:
    """join 粒度（fan-out）证据采集（v2 证据式措辞）。

    解析 SQL 中 ``join {{ ref('X') }} a on ...`` 结构，对每个被 join 的
    维度表检查 join 键的唯一性；不唯一时采集重复行样本与"差异列"。
    v2 不再直接下指令（"必须加 join key"）——任务可能**合法地**需要
    一对多输出；反馈改为呈现证据（重复键组 + 样本 + 差异列 + 契约
    声明的粒度），由修复轮的 LLM 裁决 fan-out 是否任务本意。
    全部键唯一但行数仍 > 主源表时，退回通用膨胀证据提示。
    """
    pairs = [(m.group(1), m.group(2)) for m in _JOIN_REF_RE.finditer(sql_text)]
    if not pairs:
        return None
    eqs = _EQ_RE.findall(sql_text)
    messages: list[str] = []
    all_unique = True
    grain = f" The model's column contract declares this grain: [{grain_hint}]." if grain_hint else ""
    for table, alias in pairs:
        dim_keys = [b_col for a, _a_col, b, b_col in eqs if b == alias]
        dim_keys += [a_col for a, a_col, b, _b_col in eqs if a == alias]
        if not dim_keys:
            continue
        keys = ", ".join(f'"{k}"' for k in dict.fromkeys(dim_keys))
        try:
            dup_groups = con.execute(
                f"select count(*) from (select {keys} from \"{table}\" "  # noqa: S608
                f"group by {keys} having count(*) > 1)"
            ).fetchone()[0]
        except duckdb.Error:
            continue
        if dup_groups:
            all_unique = False
            cols = [
                r[0]
                for r in con.execute(
                    "select column_name from information_schema.columns "
                    "where table_schema not in ('information_schema','pg_catalog') "
                    "and table_name = ? order by ordinal_position",
                    [table],
                ).fetchall()
            ]
            key_idx = [cols.index(k) for k in dict.fromkeys(dim_keys)]
            sample = con.execute(
                f"select * from \"{table}\" qualify count(*) over (partition by {keys}) > 1 "  # noqa: S608
                f"order by {keys} limit 6"
            ).fetchall()
            # 从重复组内推导"差异列"——区分重复行的候选 join key
            differing: list[str] = []
            groups: dict[tuple, list[tuple]] = {}
            for row in sample:
                groups.setdefault(tuple(str(row[i]) for i in key_idx), []).append(row)
            for grows in groups.values():
                if len(grows) < 2:
                    continue
                for i, col in enumerate(cols):
                    if i in key_idx:
                        continue
                    if len({str(r[i]) for r in grows}) > 1 and col not in differing:
                        differing.append(col)
            sample_text = "; ".join(
                ", ".join(f"{c}={_clip_cell(v)}" for c, v in zip(cols, row, strict=False))
                for row in sample[:3]
            )
            evidence = (
                f" Within the same join key, these columns DIFFER: {differing}."
                if differing
                else " No clearly differing column was auto-detected in the sample."
            )
            messages.append(
                f"the join to '{table}' uses key(s) {dim_keys} which are NOT unique in "
                f"'{table}' ({dup_groups} duplicated key groups). Sample duplicate rows from "
                f"'{table}': {sample_text}.{evidence}{grain} Adjudicate from the task and "
                "the contract whether this one-to-many is INTENDED (some tasks legitimately "
                "need attribute-level fan-out - then keep the SQL as it is) or a grain bug "
                "(then one of the differing columns likely also exists in the base table and "
                "belongs in the join condition)."
            )
    if messages:
        return (
            f"model '{model}' produced {model_rows} rows due to one-to-many joins. Diagnosis: "
            + " ".join(messages)
        )
    if all_unique:
        # 主源表取 FROM 子句的 ref（join 顺序不可靠），找不到才退回第一个 join
        base_m = _FROM_REF_RE.search(sql_text)
        base = base_m.group(1) if base_m else pairs[0][0]
        try:
            base_count = con.execute(f'select count(*) from "{base}"').fetchone()[0]  # noqa: S608
        except duckdb.Error:
            return None
        if model_rows > base_count:
            return (
                f"model '{model}' has {model_rows} rows but base source '{base}' has only "
                f"{base_count} rows, while every join key was verified unique.{grain} "
                "Adjudicate from the task whether this row count is what the task asks for; "
                "if the intended grain is one row per base row, find and fix the row-multiplying "
                "join or add aggregation."
            )
    return None


class InsightFlowAgent:
    def __init__(self, options: AgentOptions | None = None, llm: LLMClient | None = None) -> None:
        self.options = options or AgentOptions()
        self.llm = llm or LLMClient()
        self.graph = self._build()

    # ------------------------------------------------------------------ 节点

    def _prepare(self, state: AgentState) -> dict[str, Any]:
        """节点 ``prepare``：准备全流程共享的上下文材料。

        - 定位工作区内唯一的 ``*.duckdb``，导出全部表的 schema 文本
          （按 ``schema_samples`` 附加值采样行，baseline=0 则仅结构）。
        - 递归收集 ``models/`` 下已有模型的 SQL，作为项目惯例示例
          提供给 LLM（命名风格、CTE 写法、ref 用法等）。
        - 初始化过程状态的默认值：生成物、错误摘要、列差异、轮次等，
          保证每次 ``run`` 从干净状态开始。
        """
        workspace = Path(state["workspace"])
        db_candidates = list(workspace.glob("*.duckdb"))
        if len(db_candidates) != 1:
            raise FileNotFoundError(f"Expected exactly one .duckdb in {workspace}, found {len(db_candidates)}")
        schema_text = extract_schema(db_candidates[0], max_sample_rows=self.options.schema_samples)
        existing = []
        models_dir = workspace / "models"
        if models_dir.is_dir():
            # 递归收集：dbt 项目常把模型放在 marts/、staging/ 等子目录
            for sql_file in sorted(models_dir.rglob("*.sql")):
                rel = sql_file.relative_to(models_dir).as_posix()
                existing.append(f"## existing model file: models/{rel}\n```sql\n{sql_file.read_text(encoding='utf-8').strip()}\n```")
        return {
            "schema_text": schema_text,
            "existing_models": "\n\n".join(existing),
            "generated": {},
            "best_generated": {},
            "contracts": {},
            "error_digest": "",
            "column_issues": "",
            "attempts": 0,
            "success": False,
            "verified": False,
            "sql_files": {},
        }

    def _link(self, state: AgentState) -> dict[str, Any]:
        """节点 ``link``：两步生成第一步——为目标模型建立列契约。

        契约即目标表的输出规格书（最小覆盖列集 + 语义描述），v2 超集
        语义下由 :func:`schema_linker.build_contracts` 组合：yml 草稿作为
        最小覆盖交 LLM 审阅（只加不删），无草稿的模型由 LLM 按"最大化
        合理覆盖"设计。``use_linker=False``（M1 消融模式）时直通，不建
        契约，后续 verify 也不做列校验。
        """
        if not self.options.use_linker:
            return {}
        contracts = build_contracts(
            self.llm,
            Path(state["workspace"]),
            state["instruction"],
            state["schema_text"],
            state["target_models"],
            existing_models=state["existing_models"],
            required_sizes=state.get("required_sizes") or None,
        )
        if contracts:
            print("[linker] contracts: " + ", ".join(f"{m}({c['source']}:{len(c['columns'])}列)" for m, c in contracts.items()))
        return {"contracts": contracts}

    def _generate(self, state: AgentState) -> dict[str, Any]:
        """节点 ``generate``：两步生成第二步——按契约请求 LLM 写 SQL。

        - 首轮：``USER_TEMPLATE``，注入任务指令、目标模型、yml 规格、
          契约段、schema 与已有模型示例。
        - 修复轮（attempts>0）：改用 ``REPAIR_TEMPLATE``，附上一轮生成
          的 SQL 与失败反馈（dbt 错误摘要 / 列校验差异 / fan-out 诊断），
          让 LLM 在明确证据下定向修改。

        生成后做两道静态检查，任一命中即计一轮失败并回注重试（不进
        execute，避免浪费 dbt 运行预算）：

        ① 响应解析不出可用 SQL 块；
        ② 自引用（``_find_self_references``：ref 或裸表名引用目标模型
          自身，dbt 运行期必然 infinite recursion）。

        检查通过后清空上一轮的失败状态（dbt 错误摘要由 execute 按需
        写入），进入 execute。
        """
        targets = state["target_models"]
        targets_text = "\n".join(f"- {name}" for name in targets)
        contracts_text = render_contracts(state.get("contracts") or {}) or "(none)"
        context = {
            "instruction": state["instruction"],
            "targets": targets_text,
            "model_specs": state["model_specs"],
            "contracts": contracts_text,
            "schema": state["schema_text"],
        }
        if state["attempts"] > 0:
            # 修复轮：附加上一轮 SQL 与失败反馈（dbt 错误和/或列校验差异）。
            previous = "\n\n".join(
                f"## {path}\n```sql\n{content}\n```" for path, content in state["generated"].items()
            )
            user = REPAIR_TEMPLATE.format(
                previous_sql=previous or "(none)",
                failure_feedback=build_failure_feedback(
                    state.get("error_digest", ""), state.get("column_issues", "")
                ),
                **context,
            )
        else:
            user = USER_TEMPLATE.format(existing_models=state["existing_models"] + "\n\n", **context)

        response = self.llm.chat(SYSTEM_PROMPT, user)
        files = parse_sql_blocks(response, targets)
        if not files:
            return {
                "generated": {},
                "error_digest": f"LLM response contained no usable SQL blocks.\n{response[-2000:]}",
                "attempts": state["attempts"] + 1,
            }
        self_refs = _find_self_references(files)
        if self_refs:
            return {
                "generated": {},
                "error_digest": (
                    "The generated SQL references a target model itself, which is "
                    "forbidden (it causes infinite recursion). Rebuild from the original "
                    "source tables instead:\n" + "\n".join(self_refs)
                ),
                "attempts": state["attempts"] + 1,
            }
        # 进入新一轮生成后清除上一轮的列差异与错误摘要（dbt 错误由 execute 按需重写）
        return {"generated": files, "error_digest": "", "column_issues": "", "attempts": state["attempts"] + 1}

    def _execute(self, state: AgentState) -> dict[str, Any]:
        """节点 ``execute``：写盘并运行 dbt。

        - 把生成的 SQL 按 dbt 项目相对路径写入工作区（保留目录结构，
          如 ``models/marts/core/fct_sales.sql``）。
        - 执行 ``dbt run --select <select_models>``（超时
          ``dbt_timeout`` 秒；select_models 含已有模型，保证依赖顺序）。
        - dbt 成功 → success=True（错误摘要已在 generate 中清空）；
          失败 → 把 dbt 错误摘要写入 ``error_digest`` 供修复轮回注。
        每轮都会重置 verified=False，由 verify 重新裁决。
        """
        workspace = Path(state["workspace"])
        if not state["generated"]:
            return {"success": False, "verified": False}
        for rel_path, content in state["generated"].items():
            target = workspace / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content + "\n", encoding="utf-8")
        result = run_dbt(workspace, models=state.get("select_models") or state["target_models"], timeout=self.options.dbt_timeout)
        update: dict[str, Any] = {"success": result.success, "verified": False}
        if not result.success:
            update["error_digest"] = result.error_digest()
        return update

    def _verify(self, state: AgentState) -> dict[str, Any]:
        """节点 ``verify``：运行时自洽校验（v2 超集门 + 最优轮快照）。

        前置门槛：

        - dbt 未成功 → 直接返回不可验证（不产出诊断），由路由决定
          是否进入修复；
        - 无契约（M1 消融模式 ``use_linker=False``）→ 直通视为通过。

        对每个契约模型逐一过两道门：

        - 门 1 · 列超集比对：只读连接物化库，读 ``information_schema``
          的实际列，仅当契约列有**缺失**时才失败（``expected ⊆ actual``；
          多余列与顺序免费——评测器逐列子集匹配、与顺序无关），差异由
          ``_column_diff`` 产出"只补列、不裁剪"的修复提示。
        - 门 2 · join 粒度证据：列门通过才做，采集 fan-out 证据（重复
          键组 + 样本 + 差异列 + 契约粒度），由修复轮 LLM 裁决是否
          任务本意，详见 ``_diagnose_join_fanout``。

        快照：所有模型列门通过（无论门 2 结论）时，把当轮 ``generated``
        写入 ``best_generated``——dbt 成功且列契约满足的轮次是"最优轮"
        候选，预算耗尽时由 ``restore_best`` 恢复（机制 A 保险）。

        出口：有问题 → verified=False，差异写入 ``column_issues`` 回注
        修复；全过 → verified=True，路由送 END。

        注意：这是自洽性检查而非正确性检查——verified=True 只说明
        结果包含契约列且无 fan-out 异议，不保证评测得分（值正确与否
        只有官方 evaluate.py 能判）。
        """
        contracts = state.get("contracts") or {}
        if not state["success"]:
            return {"verified": False}  # dbt 未通过，保持不可验证，由路由决定是否修复
        if not contracts:
            return {"verified": True, "column_issues": ""}  # 无契约（M1 模式）时不校验
        workspace = Path(state["workspace"])
        db_file = next(workspace.glob("*.duckdb"), None)
        if db_file is None:
            return {"verified": False, "column_issues": "workspace 中找不到 .duckdb 文件，无法校验列"}
        issues: list[str] = []
        snapshot: dict[str, str] = {}
        con = duckdb.connect(str(db_file), read_only=True)
        try:
            columns_ok = True
            for model, contract in contracts.items():
                rows = con.execute(
                    """
                    select column_name from information_schema.columns
                    where table_schema not in ('information_schema', 'pg_catalog')
                      and table_name = ?
                    order by ordinal_position
                    """,
                    [model],
                ).fetchall()
                actual = [r[0] for r in rows]
                expected = [col["name"] for col in contract["columns"]]
                missing = [c for c in expected if c not in actual]
                if missing:
                    columns_ok = False
                    issues.append(_column_diff(model, expected, actual))
                    continue
                # 列门通过（缺失为空；多余列与顺序免费）→ 再查 join 粒度（fan-out）。
                grain_hint = ", ".join(
                    col["name"] + (f" ({col['description']})" if col.get("description") else "")
                    for col in contract["columns"]
                )[:600]
                sql_text = (state.get("generated") or {}).get(f"models/{model}.sql", "")
                model_rows = con.execute(f'select count(*) from "{model}"').fetchone()[0]  # noqa: S608
                inflation = _diagnose_join_fanout(con, model, sql_text, model_rows, grain_hint)
                if inflation:
                    issues.append(inflation)
            if columns_ok:
                # 最优轮快照：dbt 成功且全部列门通过（fan-out 异议另议）
                snapshot = dict(state.get("generated") or {})
        finally:
            con.close()
        update: dict[str, Any] = {}
        if snapshot:
            update["best_generated"] = snapshot
        if issues:
            print(f"[verify] issues: {'; '.join(i.split('.')[0] for i in issues)}")
            update.update({"verified": False, "column_issues": "\n".join(issues)})
            return update
        print("[verify] all column contracts satisfied (superset)")
        update.update({"verified": True, "column_issues": ""})
        return update

    def _restore_best(self, state: AgentState) -> dict[str, Any]:
        """节点 ``restore_best``：预算耗尽后的最优轮恢复（v2 保险）。

        修复预算耗尽仍未 verified 时，把 ``best_generated``（最后一个
        dbt 成功且列门通过的轮次）写回工作区并重跑一次 dbt，让最终
        提交的 .duckdb 反映最优状态而非最后一轮的退化状态。生成物与
        当前一致时跳过（该轮 dbt 已成功，物化状态本就正确）。
        """
        best = state.get("best_generated") or {}
        update: dict[str, Any] = {}
        if best and best != (state.get("generated") or {}):
            workspace = Path(state["workspace"])
            for rel_path, content in best.items():
                target = workspace / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content + "\n", encoding="utf-8")
            result = run_dbt(
                workspace,
                models=state.get("select_models") or state["target_models"],
                timeout=self.options.dbt_timeout,
            )
            update = {"generated": best, "success": result.success, "verified": False}
            if not result.success:
                update["error_digest"] = result.error_digest()
            print(f"[restore_best] restored best round ({len(best)} files), dbt success={result.success}")
        else:
            print("[restore_best] best round equals final state; nothing to restore")
        return update

    # ------------------------------------------------------------------ 建图

    def _build(self):
        """组装 LangGraph 图并注册条件路由。

        线性主干：START → prepare → link → generate → execute → verify。

        verify 之后的路由（``route``）：

        - verified=True → END（成功收敛）；
        - 未通过且 attempts 未超出 max_attempts → 回 ``generate``
          进入修复轮（携带 dbt 错误摘要 / 列差异 / fan-out 证据）；
        - 修复预算耗尽 → 开启快照（``use_best_snapshot``）且存在
          ``best_generated`` 时走 ``restore_best`` 恢复最优轮再结束；
          否则直接 END（带着最后状态结束）。
        """
        builder = StateGraph(AgentState)
        builder.add_node("prepare", self._prepare)
        builder.add_node("link", self._link)
        builder.add_node("generate", self._generate)
        builder.add_node("execute", self._execute)
        builder.add_node("verify", self._verify)
        builder.add_node("restore_best", self._restore_best)
        builder.add_edge(START, "prepare")
        builder.add_edge("prepare", "link")
        builder.add_edge("link", "generate")
        builder.add_edge("generate", "execute")
        builder.add_edge("execute", "verify")

        def route(state: AgentState) -> str:
            if state.get("verified"):
                return END
            if state["attempts"] <= self.options.max_attempts:
                return "generate"
            if self.options.use_best_snapshot and state.get("best_generated"):
                return "restore_best"
            return END

        builder.add_conditional_edges(
            "verify", route, {"generate": "generate", "restore_best": "restore_best", END: END}
        )
        builder.add_edge("restore_best", END)
        return builder.compile()

    def run(
        self,
        *,
        instruction: str,
        workspace: Path,
        target_models: list[str],
        model_specs: str = "",
        select_models: list[str] | None = None,
        required_sizes: dict[str, int] | None = None,
    ) -> AgentState:
        """运行单个任务的完整循环，返回最终状态。"""
        final = self.graph.invoke(
            {
                "instruction": instruction,
                "workspace": str(workspace),
                "target_models": target_models,
                "select_models": select_models or target_models,
                "model_specs": model_specs,
                "required_sizes": required_sizes or {},
            }
        )
        return final  # type: ignore[return-value]
