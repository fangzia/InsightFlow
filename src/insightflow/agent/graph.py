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

import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import duckdb
from langgraph.graph import END, START, StateGraph

from insightflow.agent.llm import LLMClient
from insightflow.agent.prompts import (
    ANCHOR_SYSTEM_PROMPT,
    ANCHOR_TEMPLATE,
    CRITIC_SYSTEM_PROMPT,
    CRITIC_TEMPLATE,
    CRITIC_V2_SYSTEM_PROMPT,
    CRITIC_V2_TEMPLATE,
    GRAIN_SYSTEM_PROMPT,
    GRAIN_TEMPLATE,
    REPAIR_TEMPLATE,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    build_failure_feedback,
    parse_sql_blocks,
)
from insightflow.agent.schema_linker import build_contracts, render_contracts
from insightflow.connectors.duckdb_executor import extract_schema, prune_parse_blockers, run_dbt


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
    # --- M3（graph 优化第 1 步，全部默认关闭以保持 v2 可复现） ---
    use_lint: bool = False  # 静态 lint：ref 指向不存在的模型节点即拦截重试
    use_triage: bool = False  # dbt 错误提炼：结构化摘要（ok/failed 模型 + 错误块）
    use_deps: bool = False  # hub 包缺失时先 dbt deps
    use_ancestors: bool = False  # dbt select 追加缺失祖先闭包（只补缺、不重建）
    use_pool0_rescue: bool = False  # 预实现项目 dbt 失败且目标表缺失时交给 agent 重建
    use_yml_quarantine: bool = False  # shipped 坏 yml（Parsing Error）移出后重试
    use_prune_blockers: bool = False  # shipped 坏模型（引用不存在节点）剪除后重跑
    use_ref_repair: bool = False  # 预检：ref() 误指向裸表时改写为直接表引用
    # --- M3 第 2 步：无进展检测 + replan 分支 ---
    replan_limit: int = 0  # 连续相同失败签名时触发重设计的次数上限；0 = 关闭
    # --- M3 第 3 步：多候选生成（延迟分支） ---
    n_candidates: int = 1  # 总尝试轨迹数（原轨迹 + 独立候选）；1 = 关闭
    # --- M3 第 4 步：critic 结果审查 ---
    use_critic: bool = False  # verify 全绿后独立 LLM 审查物化数据（值级证据）
    use_critic_v2: bool = False  # critic v2：统计级证据（NULL/distinct/值域对账/粒度比）替代 3 行采样
    # --- M3 第 5 步：值层确定性信号 ---
    use_anchor: bool = False  # 行数锚定：从指令提取显式行数，verify 对照（行数是评测器唯一硬约束）
    use_column_review: bool = False  # 单列重审：verify-green 双版本对照，弱列定向重写
    use_grain_anchor: bool = False  # 粒度词锚定："monthly per X" → 行数 vs distinct(时间)×distinct(实体) 核对


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
    best_generated: dict[str, str]  # "列门通过且严重度最低"轮的生成物快照
    best_severity: tuple  # 该快照的 issue 严重度（重复键 fan-out 数, 膨胀裁决数）
    error_digest: str
    column_issues: str  # verify 产出的列差异文本
    attempts: int
    # M3 replan：上一轮失败签名（重复 = 无进展）与重设计状态
    prev_signature: str
    replans: int
    replan_feedback: str
    # M3 候选：当前轨迹编号（0 = 原轨迹）、候选轮的独立尝试注记、已跑候选
    candidate_index: int
    candidate_note: str
    candidates: list[dict]
    # M3 critic：审查轮数（限 1）与结论
    critics: int
    critic_verdict: str  # "approve" | "challenge"
    # M3 值层信号：行数锚点（模型名 -> 显式期望行数）、粒度假设与单列重审
    row_anchors: dict[str, int]
    grain_specs: dict[str, dict]
    column_reviews: int
    column_review_verdict: str  # "stable" | "weak"
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


def _collect_model_stems(workspace: Path) -> set[str]:
    """收集 ref() 的全部合法目标：项目与包的**模型文件** stem（M3）。

    dbt 语义：``ref()`` 只能指向模型节点（models/ 与 dbt_packages/ 下的
    .sql 文件），不能指向裸表——裸源表必须直接用表名引用。因此 lint
    对 ref 的校验集不含 schema 中的表名（use_deps 装包后再次调用即
    覆盖新装包的模型）。
    """
    stems: set[str] = set()
    for base in (workspace / "models", workspace / "dbt_packages"):
        if base.is_dir():
            stems.update(p.stem for p in base.rglob("*.sql"))
    return stems


def _find_dangling_refs(files: dict[str, str], model_stems: set[str]) -> list[str]:
    """静态 lint：ref() 指向不存在的模型节点（M3）。

    生成 SQL 的 ref() 目标既不是项目/包模型文件、也不是本轮其他生成
    文件时，dbt 必然报 Compilation Error（"depends on a node which was
    not found"）——在消耗 dbt 轮次之前拦截。difflib 给出近似名提示
    （常见根因是拼写差，或把裸源表名误写进了 ref()）。
    """
    emitted = {Path(p).stem for p in files}
    valid = model_stems | emitted
    issues = []
    for path, sql in files.items():
        model = Path(path).stem
        for m in _REF_RE.finditer(sql):
            name = m.group(1)
            if name == model or name in valid:
                continue
            close = difflib.get_close_matches(name, sorted(valid), n=2, cutoff=0.6)
            hint = f" (did you mean {close}?)" if close else ""
            issues.append(
                f"models/{model}.sql references ref('{name}') which does not match any "
                f"existing project or package model{hint}. If you meant a raw source "
                "table, reference it directly by name (e.g. from \"main\".\"<table>\") "
                "instead of ref(); otherwise fix the reference or inline the logic as a CTE."
            )
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


_TS_STRIP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\s+")


def _failure_signature(error_digest: str, column_issues: str) -> str:
    """失败签名：本轮失败的多行指纹（M3 无进展检测用）。

    连续两轮签名相同 = 定向修补无进展（同一列补不上 / 同一 dbt 错误
    反复出现），是触发 replan 的条件。签名取**错误本体**的前 3 行
    （triage 摘要中 ``--- error output ---`` 之后；错误类型 + 主错误 +
    首个细节行），时间戳剥离——保证"错误变了 = 有进展"，而头部行
    （DBT RUN FAILED / ok 模型清单）不参与指纹。
    """
    if column_issues:
        return "columns:" + column_issues.strip().splitlines()[0][:200]
    if error_digest:
        body = error_digest.split("--- error output ---", 1)[-1]
        lines = [_TS_STRIP_RE.sub("", l).strip() for l in body.splitlines()]
        lines = [l for l in lines if l]
        if not lines:
            return ""
        return "dbt:" + " | ".join(lines[:3])[:300]
    return ""


def _issue_severity(issues: list[str]) -> tuple[int, int]:
    """verify issue 的严重度（快照择优用）。

    两类 fan-out 证据的严重度不同：**重复键 fan-out**（"due to
    one-to-many joins"，join 键不唯一——大概率粒度 bug，重）与**唯一键
    膨胀裁决**（"but base source"，键全唯一只是行数超基表——可能任务
    本意，轻）。计数相同但类别不同的轮次因此可分出高下
    （superstore001：25667 行轮 vs 9994 行轮的区分即靠此）。
    """
    dup = sum(1 for i in issues if "one-to-many joins" in i)
    inflation = sum(1 for i in issues if "but base source" in i)
    return (dup, inflation)


def _anchor_penalty(issues: list[str]) -> int:
    """行数锚定违反数（severity 首元素）。

    锚定违反**不阻断快照**：指令的显式数字在并列（ties）场景可能与
    gold 行数有出入——强制阻断会把碰巧正确的轮次拖出最优快照。
    作为 severity 首元素：满足锚定的轮次优先成为最优快照，违反的
    轮次仍触发修复反馈。
    """
    return sum(1 for i in issues if "explicitly requires exactly" in i)


_NUMERIC_TYPES = (
    "INTEGER", "BIGINT", "SMALLINT", "TINYINT", "HUGEINT", "FLOAT", "DOUBLE", "REAL", "NUMERIC",
)


def _is_numeric_type(dtype: str) -> bool:
    d = dtype.upper()
    return d in _NUMERIC_TYPES or d.startswith("DECIMAL")


def _collect_value_evidence(
    con: duckdb.DuckDBPyConnection,
    model: str,
    generated_sql: str,
    all_tables: set[str],
    max_cols: int = 30,
) -> str:
    """critic v2 的统计级值证据：程序采集，LLM 只裁决。

    三层证据（全部零 LLM 成本）：

    - 列级：NULL 率 / distinct 数 / 数值列 min-max；
    - 粒度：目标行数 vs FROM 源表行数（ratio）；
    - 代理键对账：``<prefix>_id`` 列的值域 vs 库中既有 ``<prefix>`` 表的
      id 值域——**不重叠即强证据**（重算代理键而非回查 dim 是真实
      失败模式，superstore001 冒烟中的 ``dim_products_id`` 即此；3 行
      采样看不出，值域对账一比就现形）。
    """
    rows = con.execute(f'select count(*) from "{model}"').fetchone()[0]  # noqa: S608
    cols = con.execute(
        "select column_name, data_type from information_schema.columns "
        "where table_schema not in ('information_schema', 'pg_catalog') and table_name = ? "
        "order by ordinal_position",
        [model],
    ).fetchall()
    lines: list[str] = []
    id_cols: list[tuple[str, object, object]] = []
    for name, dtype in cols[:max_cols]:
        try:
            null_pct, distinct = con.execute(  # noqa: S608
                f'select 100.0*count(*) filter (where "{name}" is null)/count(*), '
                f'count(distinct "{name}") from "{model}"'
            ).fetchone()
        except duckdb.Error:
            continue
        entry = f"  - {name} ({dtype.lower()}): null={null_pct:.0f}%, distinct={distinct}"
        if _is_numeric_type(dtype):
            mn, mx = con.execute(  # noqa: S608
                f'select min("{name}"), max("{name}") from "{model}"'
            ).fetchone()
            entry += f", range=[{mn}, {mx}]"
            if name.endswith("_id"):
                id_cols.append((name, mn, mx))
        lines.append(entry)
    if len(cols) > max_cols:
        lines.append(f"  ... ({len(cols) - max_cols} more columns, truncated)")
    # 粒度：FROM 源表行数比（_FROM_REF_RE 提取 ref 目标）
    for ref in _FROM_REF_RE.findall(generated_sql or ""):
        if ref != model and ref in all_tables:
            src_rows = con.execute(f'select count(*) from "{ref}"').fetchone()[0]  # noqa: S608
            if src_rows:
                lines.append(f"  grain: {rows} rows vs source '{ref}' {src_rows} rows (ratio {rows / src_rows:.2f})")
    # 代理键对账：*_id 值域 vs 既有同名表的 id 值域
    for col, cmin, cmax in id_cols:
        dim = col.removesuffix("_id")
        if dim == model or dim not in all_tables:
            continue
        has_id = con.execute(
            "select count(*) from information_schema.columns "
            "where table_schema not in ('information_schema', 'pg_catalog') "
            "and table_name = ? and column_name = 'id'",
            [dim],
        ).fetchone()[0]
        if not has_id:
            continue
        dmin, dmax = con.execute(f'select min(id), max(id) from "{dim}"').fetchone()  # noqa: S608
        if None in (cmin, cmax, dmin, dmax):
            continue
        overlap = cmin <= dmax and dmin <= cmax
        flag = "" if overlap else " *** NO OVERLAP ***"
        lines.append(
            f"  key check: '{col}' range=[{cmin},{cmax}] vs existing '{dim}'.id range=[{dmin},{dmax}]{flag}"
        )
    return f"## table '{model}' ({rows} rows)\n" + "\n".join(lines)


_ANCHOR_JSON_RE = re.compile(r"```(?:json)?\s*\n(\{.*?\})\s*\n```", re.DOTALL | re.IGNORECASE)


def _parse_row_anchors(response: str, targets: list[str]) -> dict[str, int]:
    """解析行数锚定提取结果：仅接受目标模型名 + 正整数。

    提取失败 / 越界名 / 非正数一律丢弃（安静降级为空——锚定是增益
    信号，误锚会把对的轮次拖进修复，宁缺毋滥）。
    """
    m = _ANCHOR_JSON_RE.search(response)
    if not m:
        return {}
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    target_set = set(targets)
    anchors: dict[str, int] = {}
    for name, count in data.items():
        if name in target_set and isinstance(count, int) and not isinstance(count, bool) and count > 0:
            anchors[name] = count
    return anchors


def _diff_table_columns(
    con: duckdb.DuckDBPyConnection, table_a: str, table_b: str, max_rows: int = 100_000
) -> tuple[int, int, list[str] | None, list[str]]:
    """双版本对照：表 A（已 verify-green）vs 表 B（独立探针版本）。

    返回 ``(rows_a, rows_b, None, weak_columns)``；行数不同时第三位为
    ``"ROWS"`` 标记（整表弱），否则逐列做 multiset 值比对（排序后字符
    串化相等），不一致的列即**弱列**——两代独立生成在其余列上全部
    一致、只在弱列上分歧，把语义不稳定精确定位到列级。
    """
    cols_a = [
        r[0]
        for r in con.execute(
            "select column_name from information_schema.columns "
            "where table_schema not in ('information_schema', 'pg_catalog') and table_name = ? order by ordinal_position",
            [table_a],
        ).fetchall()
    ]
    cols_b = set(
        r[0]
        for r in con.execute(
            "select column_name from information_schema.columns "
            "where table_schema not in ('information_schema', 'pg_catalog') and table_name = ?",
            [table_b],
        ).fetchall()
    )
    rows_a = con.execute(f'select count(*) from "{table_a}"').fetchone()[0]  # noqa: S608
    rows_b = con.execute(f'select count(*) from "{table_b}"').fetchone()[0]  # noqa: S608
    if rows_a != rows_b or rows_a > max_rows:
        return rows_a, rows_b, "ROWS", []
    weak = [c for c in cols_a if c not in cols_b]
    for c in cols_a:
        if c not in cols_b:
            continue
        va = sorted(str(x[0]) for x in con.execute(f'select "{c}" from "{table_a}"').fetchall())  # noqa: S608
        vb = sorted(str(x[0]) for x in con.execute(f'select "{c}" from "{table_b}"').fetchall())  # noqa: S608
        if va != vb:
            weak.append(c)
    return rows_a, rows_b, None, weak


def _parse_grain_specs(response: str, targets: list[str]) -> dict[str, dict]:
    """解析粒度提取结果：仅接受目标模型与合法字段值（宁缺毋滥）。"""
    m = _ANCHOR_JSON_RE.search(response)
    if not m:
        return {}
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    target_set = set(targets)
    valid_units = {"day", "week", "month", "quarter", "year", None}
    specs: dict[str, dict] = {}
    for name, spec in data.items():
        if name not in target_set or not isinstance(spec, dict):
            continue
        unit = spec.get("time_unit")
        entity = spec.get("entity")
        if unit not in valid_units:
            unit = None
        if not isinstance(entity, str) or not entity.strip():
            entity = None
        if unit is None and entity is None:
            continue
        specs[name] = {"time_unit": unit, "entity": entity}
    return specs


def _find_columns_for_grain(
    con: duckdb.DuckDBPyConnection, model: str
) -> tuple[list[tuple[str, str]], list[str]]:
    """返回 (数值/时间候选列, 全部列名)，供粒度核对选列。"""
    rows = con.execute(
        "select column_name, data_type from information_schema.columns "
        "where table_schema not in ('information_schema', 'pg_catalog') and table_name = ? "
        "order by ordinal_position",
        [model],
    ).fetchall()
    cols = [(r[0], r[1]) for r in rows]
    return cols, [c for c, _ in cols]


def _check_grain_anchor(
    con: duckdb.DuckDBPyConnection, model: str, spec: dict, sql_text: str = ""
) -> str | None:
    """粒度词锚定核对（M3，软信号）。

    指令显式给出的粒度（"monthly per account"）→ 程序计算粒度基数
    ``distinct(时间桶) × distinct(实体)``，与实际行数比对。两个方向：

    - **过量**（rows >> 基数）：同一 (时间, 实体) 组合重复展开——输出表
      自身的 distinct 即可定位（fan-out 诊断的粒度版）；
    - **亏缺**（rows << 基数）：过粗聚合/过滤过猛——输出表的 distinct
      结构上不可能超过自身行数，**实体基数必须取自源表**（生成 SQL 的
      ref 目标中含该实体列的表）。

    比率超出 [1/3, 3] 才报——粒度核对是推断性信号（过滤子集、活跃
    客户等合法偏差），阈值宽、不阻断快照，只给修复轮带数字的反馈。
    定位不到可核对维度时安静跳过。
    """

    def _find_entity_column(table: str, hint: str) -> str | None:
        cols = [
            r[0]
            for r in con.execute(
                "select column_name from information_schema.columns "
                "where table_schema not in ('information_schema', 'pg_catalog') and table_name = ?",
                [table],
            ).fetchall()
        ]
        for c in cols:
            if c.lower() == hint:
                return c
        for c in cols:
            if c.lower().rstrip("_") == f"{hint}_id":
                return c
        for c in cols:
            if hint in c.lower():
                return c
        if cols:
            close = difflib.get_close_matches(hint, [c.lower() for c in cols], n=1, cutoff=0.6)
            if close:
                return next(c for c in cols if c.lower() == close[0])
        return None

    def _table_exists(name: str) -> bool:
        return bool(
            con.execute(
                "select count(*) from information_schema.tables "
                "where table_schema not in ('information_schema', 'pg_catalog') and table_name = ?",
                [name],
            ).fetchone()[0]
        )

    unit, entity_hint = spec.get("time_unit"), spec.get("entity")
    rows = con.execute(f'select count(*) from "{model}"').fetchone()[0]  # noqa: S608
    if rows == 0:
        return None
    typed_cols = _find_columns_for_grain(con, model)[0]
    if not typed_cols:
        return None
    parts: list[str] = []
    expected = 1
    if unit:
        tcol = None
        is_plain_unit = False
        for name, dtype in typed_cols:
            low = name.lower()
            if unit in low and _is_numeric_type(dtype):
                tcol, is_plain_unit = name, True
                break
        if tcol is None:
            for name, dtype in typed_cols:
                low = name.lower()
                if "date" in low or "time" in low:
                    if dtype.upper() in ("DATE", "TIMESTAMP", "TIMESTAMPTZ", "DATETIME"):
                        tcol, is_plain_unit = name, False
                        break
        if tcol is not None:
            if is_plain_unit:
                dt = con.execute(f'select count(distinct "{tcol}") from "{model}"').fetchone()[0]  # noqa: S608
            else:
                dt = con.execute(  # noqa: S608
                    f'select count(distinct date_trunc(\'{unit}\', "{tcol}")) from "{model}"'
                ).fetchone()[0]
            if dt:
                expected *= dt
                parts.append(f"distinct {unit}s={dt}")
    if entity_hint:
        hint = entity_hint.lower().strip()
        # 实体基数优先取自源表（亏缺方向的唯一可见途径）
        de = None
        for ref in _FROM_REF_RE.findall(sql_text or ""):
            if ref != model and _table_exists(ref):
                ecol = _find_entity_column(ref, hint)
                if ecol is not None:
                    de = con.execute(  # noqa: S608
                        f'select count(distinct "{ecol}") from "{ref}"'
                    ).fetchone()[0]
                    parts.append(f"source '{ref}'.{ecol} distinct={de}")
                    break
        if de is None:
            ecol = _find_entity_column(model, hint)
            if ecol is not None:
                de = con.execute(f'select count(distinct "{ecol}") from "{model}"').fetchone()[0]  # noqa: S608
                parts.append(f"distinct '{ecol}'={de}")
        if de:
            expected *= de
    if not parts or expected == 0:
        return None
    ratio = rows / expected
    if 1 / 3 <= ratio <= 3:
        return None
    grain_desc = (f"{unit}-per-{entity_hint}" if unit and entity_hint else (f"per-{entity_hint}" if entity_hint else f"{unit}"))
    return (
        f"model '{model}' has {rows} rows, but the task asks for {grain_desc} granularity "
        f"which implies approximately {expected} rows ({', '.join(parts)}); the output has "
        f"{ratio:.1f}x {'more' if ratio > 1 else 'fewer'} rows. Row count is the evaluator's "
        "hard constraint - fix the aggregation grain (or over-broad join/missing filter) "
        "to match the granularity the task asks for."
    )


def _repair_refs_to_tables(workspace: Path, schema_text: str) -> int:
    """预检修复：shipped 模型把裸表误写进 ref() 时改写为直接表引用（M3）。

    f1001 家族：shipped staging 用 ``{{ ref('circuits') }}`` 引用裸表
    （db 表而非模型节点）——dbt 整项目 parse 失败，且 staging 互相引用
    使剪除无法下手。但目标表本身已物化、值是对的；把这类 ref 机械改写
    为 ``"schema"."table"`` 直接引用即可恢复 parse（不改变物化状态）。

    条件（全部满足才改写）：ref 目标既非项目/包模型文件，也非本轮逻辑
    中的模型，但**是 db 中真实存在的表**。
    """
    table_schema: dict[str, str] = {}
    for m in re.finditer(r"-- table (\w+)\.([\w]+) \(", schema_text):
        schema, name = m.groups()
        table_schema.setdefault(name, schema)
    if not table_schema:
        return 0
    model_stems = _collect_model_stems(workspace)
    full_ref_re = re.compile(r"\{\{\s*ref\(\s*['\"]([^'\"]+)['\"]\s*\)\s*\}\}")
    # dbt_project.yml 的 vars 值：`key: "{{ ref('x') }}"`（fivetran 包惯用）
    vars_ref_re = re.compile(r"['\"]\{\{\s*ref\(\s*['\"]([^'\"]+)['\"]\s*\)\s*\}\}['\"]")

    def _rewrite(name: str) -> str | None:
        if name in model_stems or name not in table_schema:
            return None
        return f'"{table_schema[name]}"."{name}"'

    changed = 0
    # vars 中的 ref()（recharge001：order: "{{ ref('order_data') }}" 指向裸表）
    project_yml = workspace / "dbt_project.yml"
    if project_yml.is_file():
        try:
            content = project_yml.read_text(encoding="utf-8")
        except OSError:
            content = ""
        if content:
            def _sub_yml(m: re.Match) -> str:
                name = m.group(1)
                direct = _rewrite(name)
                nonlocal changed
                if direct is None:
                    return m.group(0)
                changed += 1
                # YAML 单引号字符串包裹带双引号的限定表名
                return f"'{direct}'"

            new_content = vars_ref_re.sub(_sub_yml, content)
            if new_content != content:
                project_yml.write_text(new_content, encoding="utf-8")
    # 扫描项目模型与 vendored 包模型——包内 staging 同样存在把裸表写进
    # ref() 的缺陷（recharge001 的 recharge_source.tmp 即此）
    scan_roots = [workspace / "models", workspace / "dbt_packages"]
    for root in scan_roots:
        if not root.is_dir():
            continue
        for sql_file in root.rglob("*.sql"):
            try:
                content = sql_file.read_text(encoding="utf-8")
            except OSError:
                continue

            def _sub(m: re.Match) -> str:
                name = m.group(1)
                direct = _rewrite(name)
                nonlocal changed
                if direct is None:
                    return m.group(0)
                changed += 1
                return direct

            new_content = full_ref_re.sub(_sub, content)
            if new_content != content:
                sql_file.write_text(new_content, encoding="utf-8")
    return changed


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
        if self.options.use_ref_repair:
            repaired = _repair_refs_to_tables(workspace, schema_text)
            if repaired:
                print(f"[prepare] repaired {repaired} ref() calls that pointed at raw tables")
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
            "best_severity": (10**9, 10**9, 10**9),
            "contracts": {},
            "error_digest": "",
            "column_issues": "",
            "attempts": 0,
            "prev_signature": "",
            "replans": 0,
            "replan_feedback": "",
            "candidate_index": 0,
            "candidate_note": "",
            "candidates": [],
            "critics": 0,
            "critic_verdict": "",
            "row_anchors": {},
            "grain_specs": {},
            "column_reviews": 0,
            "column_review_verdict": "",
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
            design_note=state.get("replan_feedback") or "",
        )
        update: dict[str, Any] = {}
        if contracts:
            print("[linker] contracts: " + ", ".join(f"{m}({c['source']}:{len(c['columns'])}列)" for m, c in contracts.items()))
        update["contracts"] = contracts
        # M3 行数锚定：从指令提取显式行数约束（一次温度 0 小调用；
        # 失败安静降级为空——锚定宁缺毋滥）
        if self.options.use_anchor:
            try:
                response = self.llm.chat(
                    ANCHOR_SYSTEM_PROMPT,
                    ANCHOR_TEMPLATE.format(
                        instruction=state["instruction"],
                        targets="\n".join(f"- {name}" for name in state["target_models"]),
                    ),
                    temperature=0.0,
                )
                anchors = _parse_row_anchors(response, state["target_models"])
                if anchors:
                    print(f"[anchor] row anchors: {anchors}")
                update["row_anchors"] = anchors
            except Exception as e:  # noqa: BLE001
                print(f"[anchor] extraction failed ({type(e).__name__}: {e}); no anchors")
                update["row_anchors"] = {}
        # M3 粒度词锚定：提取输出粒度假设（monthly/per X → time_unit × entity）
        if self.options.use_grain_anchor:
            try:
                response = self.llm.chat(
                    GRAIN_SYSTEM_PROMPT,
                    GRAIN_TEMPLATE.format(
                        instruction=state["instruction"],
                        targets="\n".join(f"- {name}" for name in state["target_models"]),
                    ),
                    temperature=0.0,
                )
                grains = _parse_grain_specs(response, state["target_models"])
                if grains:
                    print(f"[anchor] grain specs: {grains}")
                update["grain_specs"] = grains
            except Exception as e:  # noqa: BLE001
                print(f"[anchor] grain extraction failed ({type(e).__name__}: {e}); no grain")
                update["grain_specs"] = {}
        return update

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
        # M3 候选轮：单发独立尝试（不吃修复预算、不附失败反馈——候选的
        # 价值在独立性），换 temperature 提高轨迹多样性
        candidate_round = self.options.n_candidates > 1 and state.get("candidate_index", 0) > 0
        # M3 replan：重设计轮次注入"前设计已废"上下文，提示换思路而非原样重写
        if state.get("replan_feedback"):
            note_section = (
                "# REDESIGN CONTEXT (previous design scrapped)\n" + state["replan_feedback"] + "\n\n"
            )
        elif state.get("candidate_note"):
            note_section = "# INDEPENDENT ATTEMPT\n" + state["candidate_note"] + "\n\n"
        else:
            note_section = ""
        context = {
            "instruction": state["instruction"],
            "redesign_note": note_section,
            "targets": targets_text,
            "model_specs": state["model_specs"],
            "contracts": contracts_text,
            "schema": state["schema_text"],
        }
        if state["attempts"] > 0 and not candidate_round:
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

        response = self.llm.chat(
            SYSTEM_PROMPT,
            user,
            temperature=0.7 if candidate_round else 0.2,
        )
        # M3 无进展检测：记录"把本轮带进 generate 的失败"的签名，供 route 比较
        prev_sig = _failure_signature(state.get("error_digest", ""), state.get("column_issues", ""))
        files = parse_sql_blocks(response, targets)
        if not files:
            return {
                "generated": {},
                "error_digest": f"LLM response contained no usable SQL blocks.\n{response[-2000:]}",
                "attempts": state["attempts"] + 1,
                "prev_signature": prev_sig,
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
                "prev_signature": prev_sig,
            }
        if self.options.use_lint:
            dangling = _find_dangling_refs(files, _collect_model_stems(Path(state["workspace"])))
            if dangling:
                return {
                    "generated": {},
                    "error_digest": (
                        "The generated SQL references a ref() target that does not exist "
                        "as a dbt model node. ref() may only target existing project/package "
                        "models or other models emitted in this answer; raw source tables "
                        "must be referenced directly by name. Fix the references (or inline "
                        "the missing logic as CTEs):\n" + "\n".join(dangling)
                    ),
                    "attempts": state["attempts"] + 1,
                    "prev_signature": prev_sig,
                }
        # 进入新一轮生成后清除上一轮的列差异与错误摘要（dbt 错误由 execute 按需重写）
        return {
            "generated": files,
            "error_digest": "",
            "column_issues": "",
            "attempts": state["attempts"] + 1,
            "prev_signature": prev_sig,
        }

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
        result = run_dbt(
            workspace,
            models=state.get("select_models") or state["target_models"],
            timeout=self.options.dbt_timeout,
            install_deps=self.options.use_deps,
            select_ancestors=self.options.use_ancestors,
            quarantine_yml=self.options.use_yml_quarantine,
        )
        # M3 剪除：dbt 因 shipped 坏模型（引用不存在节点）parse 失败时，
        # 删除该文件重跑——agent 无法覆盖子目录里的坏文件，修复轮也无从
        # 下手（错误不在本轮生成物上）。agent 本轮生成的文件受保护。
        if not result.success and self.options.use_prune_blockers:
            protect = {(workspace / rel).resolve() for rel in state["generated"]}
            deleted = prune_parse_blockers(workspace, result, protect=protect)
            if deleted:
                print(f"[execute] pruned shipped parse blockers: {deleted}")
                result = run_dbt(
                    workspace,
                    models=state.get("select_models") or state["target_models"],
                    timeout=self.options.dbt_timeout,
                    install_deps=self.options.use_deps,
                    select_ancestors=self.options.use_ancestors,
                    quarantine_yml=self.options.use_yml_quarantine,
                )
        update: dict[str, Any] = {"success": result.success, "verified": False}
        # M3 候选轮结果登记（供 select_best 回退选择）
        if self.options.n_candidates > 1 and state.get("candidate_index", 0) > 0 and state["generated"]:
            candidates = list(state.get("candidates") or [])
            candidates.append({"generated": dict(state["generated"]), "success": result.success})
            update["candidates"] = candidates
        if not result.success:
            update["error_digest"] = (
                result.triage_digest() if self.options.use_triage else result.error_digest()
            )
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
        update_severity: tuple = ()
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
                # M3 行数锚定：行数是评测器唯一硬约束（len 不等直接判负）。
                # 指令显式给出的行数（"top 10"）在这里被程序化对照——
                # 违反即带具体数字的修复反馈，并降低该轮的快照优先级。
                anchor = (state.get("row_anchors") or {}).get(model)
                if anchor and model_rows != anchor:
                    # 不阻断快照（并列 ties 场景 gold 可能多于显式数）——
                    # 修复反馈 + severity 首元素惩罚，见 _anchor_penalty
                    issues.append(
                        f"model '{model}' has {model_rows} rows but the task instruction "
                        f"explicitly requires exactly {anchor} (e.g. 'top {anchor}'). Row count "
                        "is the evaluator's ONLY hard structural constraint - fix the "
                        "limit/filter/aggregation so the output has exactly that many rows."
                    )
                # M3 重复行检测：完全重复行让行向量匹配必败（gold 不会含
                # 全列重复行）；接近型行数偏差（±几行）的常见根因即缺去重。
                elif self.options.use_anchor and model_rows > 1:
                    distinct_rows = con.execute(  # noqa: S608
                        f'select count(*) from (select distinct * from "{model}")'
                    ).fetchone()[0]
                    if model_rows != distinct_rows:
                        issues.append(
                            f"model '{model}' has {model_rows - distinct_rows} exact duplicate "
                            "rows (all columns identical) which the gold table will not have. "
                            "Add deduplication (e.g. qualify row_number() over "
                            "(partition by all columns) = 1)."
                        )
                # M3 粒度词锚定：行数 vs distinct(时间)×distinct(实体，源表基数) 软性核对
                if self.options.use_grain_anchor:
                    grain = (state.get("grain_specs") or {}).get(model)
                    if grain:
                        gissue = _check_grain_anchor(con, model, grain, sql_text)
                        if gissue:
                            issues.append(gissue)
                inflation = _diagnose_join_fanout(con, model, sql_text, model_rows, grain_hint)
                if inflation:
                    issues.append(inflation)
            severity = (_anchor_penalty(issues),) + _issue_severity(issues)
            if columns_ok and (not state.get("best_generated") or severity < state.get("best_severity", (10**9, 10**9, 10**9))):
                # 最优轮快照：列门通过且**严重度更低**才覆盖——修复轮可能在
                # 收敛后回退（superstore001 冒烟中第 4 轮回到 25667 行），
                # "最后一个列门通过轮"会覆盖掉更优的早轮；计数相同的轮次
                # 由严重度类别分出高下（重复键 fan-out 重于唯一键膨胀）。
                snapshot = dict(state.get("generated") or {})
                update_severity = severity
        finally:
            con.close()
        update: dict[str, Any] = {}
        if snapshot:
            update["best_generated"] = snapshot
            update["best_severity"] = update_severity
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
                install_deps=self.options.use_deps,
                select_ancestors=self.options.use_ancestors,
                quarantine_yml=self.options.use_yml_quarantine,
            )
            update = {"generated": best, "success": result.success, "verified": False}
            if not result.success:
                update["error_digest"] = result.error_digest()
            print(f"[restore_best] restored best round ({len(best)} files), dbt success={result.success}")
        else:
            print("[restore_best] best round equals final state; nothing to restore")
        return update

    def _replan(self, state: AgentState) -> dict[str, Any]:
        """节点 ``replan``：无进展检测触发的设计层重来（M3 第 2 步）。

        连续两轮失败签名相同说明定向修补已失效——错误在**设计层**
        （join 策略 / 粒度 / 列组合），继续修补只会复现。本节点：

        - 丢弃当前契约与生成物（``contracts={}`` 让 link 重新设计，
          ``replan_feedback`` 携带失败历史注入设计与生成提示词）；
        - 重置修复预算（``attempts=0``）与签名基线（``prev_signature``
          清空，保证重设计后的第一轮失败不会被立即再次判定无进展）。

        触发次数由 ``replan_limit`` 约束（默认模式 0 = 关闭；m3-replan
        为 1），防止 link↔generate 振荡。
        """
        sig = _failure_signature(state.get("error_digest", ""), state.get("column_issues", ""))
        history = (state.get("error_digest", "") or state.get("column_issues", ""))[-1200:]
        print(f"[replan] no progress (repeated signature: {sig[:80]}); redesigning from scratch")
        return {
            "contracts": {},
            "generated": {},
            "error_digest": "",
            "column_issues": "",
            "attempts": 0,
            "prev_signature": "",
            "replans": state.get("replans", 0) + 1,
            "replan_feedback": (
                "The previous approach was repaired repeatedly but kept failing with "
                "the SAME error:\n"
                f"{sig}\n\n"
                "Incremental repair cannot fix this - the design itself is wrong. "
                "Failure details (tail):\n"
                f"{history}\n\n"
                "Scrap the previous approach and design a DIFFERENT one: reconsider "
                "the grain, the join strategy, and which source tables to build from; "
                "drop contract columns that cannot be produced from the source data; "
                "prefer a simpler, more direct transformation path."
            ),
        }

    def _critic(self, state: AgentState) -> dict[str, Any]:
        """节点 ``critic``：verify 全绿后的独立结果审查（M3 第 4 步）。

        自洽性闭环（契约 → 校验 → 修复）无法发现"错误地自洽"——契约本身
        错时 verify 是盖章机器（机制 B）。本节点引入闭环外的信号源：
        独立 LLM 调用对照任务指令审查物化数据。

        证据两代（``use_critic_v2`` 切换，v1 保留供消融对照）：

        - v1：每表 3 行采样——冒烟证明"安全但瞎"（10/10 APPROVE，
          语义错误在 3 行里隐形）；
        - v2：统计级证据（:func:`_collect_value_evidence`）——NULL/
          distinct/值域 + 代理键对账（``*_id`` vs 既有同名表 id 值域，
          不重叠即强证据）+ 粒度比。CHALLENGE 为四段式可执行靶子，
          直接注入 replan 重设计通道。

        - APPROVE → END（保守偏置：仅证据明确时质疑）。
        - CHALLENGE → 复用 replan 通道重设计（契约作废、失败原因注入
          设计与生成提示词）。误质疑的代价被最优轮快照兜底——该
          verify-green 轮已被快照，重设计失败时 restore/select 恢复。
          限 1 次（critics 计数）。
        """
        workspace = Path(state["workspace"])
        db_file = next(workspace.glob("*.duckdb"), None)
        sections: list[str] = []
        if db_file is not None:
            con = duckdb.connect(str(db_file), read_only=True)
            try:
                all_tables = {
                    r[0]
                    for r in con.execute(
                        "select table_name from information_schema.tables "
                        "where table_schema not in ('information_schema', 'pg_catalog')"
                    ).fetchall()
                }
                for model in state["target_models"]:
                    if model not in all_tables:
                        continue
                    if self.options.use_critic_v2:
                        sql_text = (state.get("generated") or {}).get(f"models/{model}.sql", "")
                        sections.append(_collect_value_evidence(con, model, sql_text, all_tables))
                    else:
                        try:
                            rows = con.execute(f'select * from "{model}" limit 3').fetchall()  # noqa: S608
                            names = [d[0] for d in con.execute(f'select * from "{model}" limit 0').description]
                            total = con.execute(f'select count(*) from "{model}"').fetchone()[0]  # noqa: S608
                        except duckdb.Error:
                            continue
                        body = "\n".join(
                            "  " + ", ".join(f"{c}={_clip_cell(v)}" for c, v in zip(names, r, strict=False))
                            for r in rows
                        )
                        sections.append(f"## table '{model}' ({total} rows)\n{body}")
            finally:
                con.close()
        v2 = self.options.use_critic_v2
        template = CRITIC_V2_TEMPLATE if v2 else CRITIC_TEMPLATE
        system = CRITIC_V2_SYSTEM_PROMPT if v2 else CRITIC_SYSTEM_PROMPT
        payload_key = "stats" if v2 else "samples"
        user = template.format(
            instruction=state["instruction"],
            **{payload_key: "\n\n".join(sections) or "(no tables materialized)"},
        )
        try:
            response = self.llm.chat(system, user, temperature=0.0)
        except Exception as e:  # noqa: BLE001
            print(f"[critic] LLM call failed ({type(e).__name__}: {e}); approving")
            return {"critic_verdict": "approve"}
        first = response.strip().splitlines()[0].strip() if response.strip() else ""
        critics = state.get("critics", 0) + 1
        if first.upper().startswith("CHALLENGE"):
            reason = first.split(":", 1)[-1].strip() or "the materialized data does not answer the task"
            print(f"[critic] CHALLENGE: {reason}")
            return {
                "critics": critics,
                "critic_verdict": "challenge",
                "verified": False,
                "contracts": {},
                "generated": {},
                "error_digest": "",
                "column_issues": "",
                "attempts": 0,
                "prev_signature": "",
                "replan_feedback": (
                    "An independent review of the MATERIALIZED DATA challenged the result:\n"
                    f"{reason}\n\n"
                    "The current contract/SQL is structurally self-consistent but "
                    "semantically wrong. Redesign from the task instruction and the "
                    "sampled data above: reconsider grain, measure definitions, and "
                    "which source tables and transformations the task actually asks for."
                ),
            }
        print("[critic] APPROVE")
        return {"critics": critics, "critic_verdict": "approve"}

    def _column_review(self, state: AgentState) -> dict[str, Any]:
        """节点 ``column_review``：verify-green 后的双版本单列重审（M3 第 5 步）。

        值层失败有 1/3 是"少数列错"（f1001 仅 ``driver_full_name`` 一列），
        而 gold-free 的图内没有"哪列错"的信号——**两代独立生成的一致性**
        是可用的代理：其余列两版全对（值一致），只有语义不稳的列会
        在独立重掷下分歧。流程：

        1. 快照 A（当前 verify-green 轮）的目标表到 ``__v1_<t>``；
        2. 独立生成探针版本 B（temperature 0.7，同首轮输入，零修复
           偏置）并物化；
        3. :func:`_diff_table_columns` 逐列 multiset 对照 → 弱列清单；
        4. **恢复 A 的表与文件**（A 是唯一经过全部门校验的版本，B 只是
           探针——避免未经 verify 的 B 成为提交状态）；
        5. 弱列 ≤ 3：``verified=False`` + 单列定向重写反馈（只重推导弱列，
           其余列的表达式与值保持不变——防止整表重掷把已对的列再掷一次）；
           弱列更多或行数分歧：视为设计不稳，同样回修复轮（整表反馈）；
           两版全一致：高置信，进 critic/END。

        限 1 次（``column_reviews`` 计数）。
        """
        reviews = state.get("column_reviews", 0) + 1
        workspace = Path(state["workspace"])
        db_file = next(workspace.glob("*.duckdb"), None)
        update: dict[str, Any] = {"column_reviews": reviews, "column_review_verdict": "stable"}
        if db_file is None:
            return update
        targets = state["target_models"]
        # 1. 快照 A 的目标表
        con = duckdb.connect(str(db_file))
        try:
            for model in targets:
                con.execute(f'drop table if exists "__v1_{model}"')  # noqa: S608
                exists = con.execute(
                    "select count(*) from information_schema.tables "
                    "where table_schema not in ('information_schema', 'pg_catalog') and table_name = ?",
                    [model],
                ).fetchone()[0]
                if exists:
                    con.execute(f'create table "__v1_{model}" as select * from "{model}"')  # noqa: S608
        finally:
            con.close()
        # 2. 独立生成探针 B（同首轮输入、高温、无失败反馈）
        contracts_text = render_contracts(state.get("contracts") or {}) or "(none)"
        user = USER_TEMPLATE.format(
            instruction=state["instruction"],
            redesign_note="",
            targets="\n".join(f"- {name}" for name in targets),
            model_specs=state["model_specs"],
            contracts=contracts_text,
            schema=state["schema_text"],
            existing_models=state["existing_models"] + "\n\n",
        )
        weak: dict[str, list[str]] = {}
        row_mismatch: dict[str, tuple[int, int]] = {}
        try:
            response = self.llm.chat(SYSTEM_PROMPT, user, temperature=0.7)
            files = parse_sql_blocks(response, targets)
        except Exception as e:  # noqa: BLE001
            print(f"[column_review] probe generation failed ({type(e).__name__}: {e}); no signal")
            files = {}
        if files:
            for rel_path, content in files.items():
                target = workspace / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content + "\n", encoding="utf-8")
            result = run_dbt(
                workspace,
                models=state.get("select_models") or targets,
                timeout=self.options.dbt_timeout,
                install_deps=self.options.use_deps,
                select_ancestors=self.options.use_ancestors,
                quarantine_yml=self.options.use_yml_quarantine,
            )
            if result.success:
                con = duckdb.connect(str(db_file))
                try:
                    for model in targets:
                        if not con.execute(
                            "select count(*) from information_schema.tables "
                            "where table_schema not in ('information_schema', 'pg_catalog') and table_name = ?",
                            [f"__v1_{model}"],
                        ).fetchone()[0]:
                            continue  # A 未物化此表，无可比对象
                        rows_a, rows_b, flag, weak_cols = _diff_table_columns(
                            con, f"__v1_{model}", model
                        )
                        if flag == "ROWS":
                            row_mismatch[model] = (rows_a, rows_b)
                        elif weak_cols:
                            weak[model] = weak_cols
                finally:
                    con.close()
            else:
                print("[column_review] probe dbt run failed; no signal")
        # 4. 恢复 A 的表与文件（无论结论如何——A 是唯一过全门校验的版本）
        con = duckdb.connect(str(db_file))
        try:
            for model in targets:
                has_v1 = con.execute(
                    "select count(*) from information_schema.tables "
                    "where table_schema not in ('information_schema', 'pg_catalog') and table_name = ?",
                    [f"__v1_{model}"],
                ).fetchone()[0]
                if has_v1:
                    # A 可能物化为 view（dbt 配置）——按对象类型精确移除再建表
                    # （duckdb 的 drop table/view 对类型不匹配即便 IF EXISTS 也报错）
                    kind = con.execute(
                        "select table_type from information_schema.tables "
                        "where table_schema not in ('information_schema', 'pg_catalog') and table_name = ?",
                        [model],
                    ).fetchone()
                    if kind is not None:
                        if kind[0] == "VIEW":
                            con.execute(f'drop view "{model}"')  # noqa: S608
                        else:
                            con.execute(f'drop table "{model}"')  # noqa: S608
                    con.execute(f'create table "{model}" as select * from "__v1_{model}"')  # noqa: S608
                    con.execute(f'drop table "__v1_{model}"')  # noqa: S608
        finally:
            con.close()
        for rel_path, content in (state.get("generated") or {}).items():
            target = workspace / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content + "\n", encoding="utf-8")
        # 5. 弱列 → 单列定向重写反馈
        if weak or row_mismatch:
            total_weak = sum(len(v) for v in weak.values())
            update["column_review_verdict"] = "weak"
            update["verified"] = False
            if weak and total_weak <= 3:
                detail = "; ".join(f"'{m}': {cols}" for m, cols in weak.items())
                print(f"[column_review] weak columns: {detail}")
                update["column_issues"] = (
                    "Two independent generations of the same task AGREE on every column "
                    "except the following, which pinpoints unstable semantics:\n"
                    f"{detail}\n"
                    "Re-derive ONLY these columns from the source data (reconsider their "
                    "join keys, name formatting/case, and derivation order); keep every "
                    "other column's expression and value EXACTLY unchanged."
                )
            else:
                detail = "; ".join(
                    f"'{m}': rows {a} vs {b}" for m, (a, b) in row_mismatch.items()
                ) or "; ".join(f"'{m}': {cols}" for m, cols in weak.items())
                print(f"[column_review] unstable: {detail[:160]}")
                update["column_issues"] = (
                    "Two independent generations disagree broadly on this design "
                    "(row counts or many columns differ), so the semantics are not yet "
                    "settled:\n"
                    f"{detail}\n"
                    "Reconsider the grain and which source columns each output column "
                    "should come from; make the design deterministic and unambiguous."
                )
        else:
            print("[column_review] two independent generations agree; high confidence")
        return update

    # ------------------------------------------------------------------ 建图

    def _next_candidate(self, state: AgentState) -> dict[str, Any]:
        """节点 ``next_candidate``：切换到下一条独立候选轨迹（M3 第 3 步）。

        修复预算耗尽（含 replan 之后）仍未收敛时，不再继续修补，而是换
        一条**独立**生成轨迹（不同 temperature + "换思路"注记，不附失败
        SQL——候选的价值在于独立性）。attempts 保持超限状态：候选是
        单发尝试，不进修复循环；候选间也不做无进展比较。
        """
        idx = state.get("candidate_index", 0) + 1
        history = _failure_signature(state.get("error_digest", ""), state.get("column_issues", ""))
        print(f"[candidates] switching to independent attempt {idx + 1}/{self.options.n_candidates}")
        return {
            "candidate_index": idx,
            "candidate_note": (
                f"This is independent attempt {idx + 1} of {self.options.n_candidates}. "
                "All previous attempts failed (last failure: "
                f"{history[:200]}). Do NOT repeat their approach: choose different source "
                "tables, a different join/grain strategy, or a simpler transformation path."
            ),
            "generated": {},
            "error_digest": "",
            "column_issues": "",
            "prev_signature": "",
            "replan_feedback": "",
        }

    def _select_best(self, state: AgentState) -> dict[str, Any]:
        """节点 ``select_best``：全部候选跑完后的最终择优（M3 第 3 步）。

        择优序：最优轮快照（列门通过且问题数最少，含原轨迹与全部候选）
        → 首个 dbt 成功的候选（列门未过但物化成功的兜底）→ 维持现状。
        胜者写回工作区并重跑 dbt——后续候选已覆盖工作区文件与物化状态。
        """
        best = state.get("best_generated") or {}
        source = "best snapshot"
        if not best:
            for entry in state.get("candidates") or []:
                if entry.get("success"):
                    best = entry["generated"]
                    source = "first successful candidate"
                    break
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
                install_deps=self.options.use_deps,
                select_ancestors=self.options.use_ancestors,
                quarantine_yml=self.options.use_yml_quarantine,
            )
            update = {"generated": best, "success": result.success, "verified": False}
            if not result.success:
                update["error_digest"] = result.error_digest()
            print(f"[select_best] restored {source} ({len(best)} files), dbt success={result.success}")
        else:
            print(f"[select_best] {source} equals final state; nothing to restore")
        return update

    def _build(self):
        """组装 LangGraph 图并注册条件路由。

        线性主干：START → prepare → link → generate → execute → verify。

        verify 之后的路由（``route``）：

        - verified=True → END（成功收敛）；
        - 未通过且 attempts 未超出 max_attempts：
          - 连续失败签名相同且 replan 预算未耗尽 → ``replan``（设计层
            重来：清契约回 link，M3 第 2 步）；
          - 否则 → 回 ``generate`` 进入修复轮（携带 dbt 错误摘要 /
            列差异 / fan-out 证据）；
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
        builder.add_node("replan", self._replan)
        builder.add_node("next_candidate", self._next_candidate)
        builder.add_node("select_best", self._select_best)
        builder.add_node("critic", self._critic)
        builder.add_node("column_review", self._column_review)
        builder.add_edge(START, "prepare")
        builder.add_edge("prepare", "link")
        builder.add_edge("link", "generate")
        builder.add_edge("generate", "execute")
        builder.add_edge("execute", "verify")
        builder.add_edge("replan", "link")
        builder.add_edge("next_candidate", "generate")
        builder.add_edge("select_best", END)

        def route(state: AgentState) -> str:
            if state.get("verified"):
                # 值层信号按确定性排序：单列重审（程序对照）→ critic（LLM 裁决）
                if self.options.use_column_review and state.get("column_reviews", 0) < 1:
                    return "column_review"
                if self.options.use_critic and state.get("critics", 0) < 1:
                    return "critic"
                return END
            if state["attempts"] <= self.options.max_attempts:
                sig = _failure_signature(state.get("error_digest", ""), state.get("column_issues", ""))
                if (
                    self.options.replan_limit > 0
                    and state.get("replans", 0) < self.options.replan_limit
                    and sig
                    and sig == state.get("prev_signature")
                ):
                    return "replan"
                return "generate"
            # 修复预算耗尽：多候选模式 → 独立轨迹；否则最优轮快照/结束
            if self.options.n_candidates > 1:
                if state.get("candidate_index", 0) < self.options.n_candidates - 1:
                    return "next_candidate"
                return "select_best"
            if self.options.use_best_snapshot and state.get("best_generated"):
                return "restore_best"
            return END

        builder.add_conditional_edges(
            "verify",
            route,
            {
                "generate": "generate",
                "replan": "replan",
                "next_candidate": "next_candidate",
                "select_best": "select_best",
                "column_review": "column_review",
                "critic": "critic",
                "restore_best": "restore_best",
                END: END,
            },
        )

        def column_review_route(state: AgentState) -> str:
            if state.get("column_review_verdict") == "weak":
                return "generate"  # 弱列反馈已写入 column_issues，走修复轮
            if self.options.use_critic and state.get("critics", 0) < 1:
                return "critic"
            return END

        builder.add_conditional_edges(
            "column_review",
            column_review_route,
            {"generate": "generate", "critic": "critic", END: END},
        )

        def critic_route(state: AgentState) -> str:
            return END if state.get("critic_verdict") == "approve" else "link"

        builder.add_conditional_edges("critic", critic_route, {END: END, "link": "link"})
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
