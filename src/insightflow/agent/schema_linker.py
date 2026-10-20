"""Schema Linker：两步生成的第一步——为目标模型建立列级契约。

设计动机（M1 实证）：64 个实例中有 21 个 dbt 跑通但表结构与 gold 期望
不符。问题不在 SQL 执行而在"输出 schema 没有被显式约束"。M2 将生成
拆成两步：

1. **建立契约**——每个目标模型确定一份最小覆盖列契约（名称、类型、
   语义；M2 v2 起为超集语义：多列免费、漏列致命）。
   来源分两类：

   - ``yml``    —— dbt schema.yml 已声明的模型，直接解析，零 LLM 成本；
   - ``llm``    —— 无 yml 的模型，由 LLM 在"只设计 schema、不写 SQL"
     的专注提示下推导（这一步的输出随后作为硬约束注入生成提示词）。
     有 yml 草稿的模型也经 LLM 审阅，但草稿列作为**最小覆盖**：
     LLM 只能加列、不能删列（程序化兜底强制执行）。

2. **按契约生成 SQL**（见 graph.py 的 generate 节点）。

契约数据以普通 dict 形式在 LangGraph state 中流转：

    {"<model>": {"source": "yml"|"llm",
                 "columns": [{"name": ..., "type": ..., "description": ...}]}}
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from insightflow.agent.llm import LLMClient

# --------------------------------------------------------------------------- 提示词（功能性内容，保持英文以稳定 LLM 行为）

SCHEMA_DESIGN_SYSTEM_PROMPT = """\
You are an analytics engineer designing the OUTPUT SCHEMA of dbt models.
You do NOT write SQL in this step. For each target model, read the task
instruction and the database schema, then commit to the set of output
columns the task requires.

For each target model you may receive a DRAFT contract (from the project's
schema.yml). Treat it as a MINIMUM COVERAGE requirement, not an upper bound:

- KEEP every draft column: draft columns are mandatory and must never be
  removed or renamed.
- ADD columns that the task instruction plausibly asks about but the draft
  lacks. When in doubt, ADD the column.
- If no draft is provided for a model, design its contract from scratch.

When designing from scratch:

- MAXIMIZE PLAUSIBLE COVERAGE: the downstream consumer only checks a subset
  of columns and tolerates extras, so extra columns are FREE while a single
  missing checked column is FATAL. Include every column the task might
  reasonably ask about: keys, descriptive attributes, measures, dates, and
  commonly derived values at the model's grain.
- Still avoid pure noise: no columns unrelated to the task's domain.
- Match the design conventions of any existing models in the project
  (key style, naming, granularity).

Always:

- Column names in snake_case, chosen to directly reflect the semantics of the
  instruction (they will be checked verbatim).
- Order columns the way a reader would expect (keys first, then attributes,
  then measures).
- Types should be DuckDB types (VARCHAR, INTEGER, DOUBLE, DATE, ...).
- Descriptions must capture the exact computational meaning when the column
  is derived (formulas, rounding, units), so the SQL writer can reproduce it.

Output one fenced JSON block per target model, in this exact format:

```json file=<model_name>
{"columns": [{"name": "...", "type": "...", "description": "..."}]}
```

No prose outside the blocks.
"""

SCHEMA_DESIGN_TEMPLATE = """\
# TASK
{instruction}

# TARGET MODELS
{targets}

{sizes}{drafts}# DATABASE SCHEMA
{schema}

{existing_models}# YOUR ANSWER
Emit one JSON block per target model with its ordered column contract.
"""

_SIZES_TEMPLATE = """\
# REQUIRED CONTRACT SIZES (authoritative)
{sizes_text}

"""

_DRAFT_TEMPLATE = """\
# DRAFT CONTRACTS (from the project's schema.yml; cross-check against the task)
{draft_text}

"""

_JSON_BLOCK_RE = re.compile(r"```json\s+file=(\S+)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_FALLBACK_JSON_RE = re.compile(r"```(?:json)?\s*\n(\{.*?\})\s*\n```", re.DOTALL | re.IGNORECASE)


# --------------------------------------------------------------------------- yml 解析

def parse_yml_contracts(models_dir: Path, targets: list[str]) -> dict[str, dict]:
    """从 dbt schema.yml 中提取目标模型的列契约（零 LLM 成本）。"""
    target_set = set(targets)
    contracts: dict[str, dict] = {}
    yml_paths = [*models_dir.rglob("*.yml"), *models_dir.rglob("*.yaml")]
    for yml_path in sorted(yml_paths):
        try:
            data = yaml.safe_load(yml_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        for model in data.get("models") or []:
            name = model.get("name")
            if name not in target_set or name in contracts:
                continue
            columns = []
            for col in model.get("columns") or []:
                if col.get("name"):
                    columns.append(
                        {
                            "name": col["name"],
                            "type": "",
                            "description": str(col.get("description", "")).strip(),
                        }
                    )
            if columns:
                contracts[name] = {"source": "yml", "columns": columns}
    return contracts


# --------------------------------------------------------------------------- LLM 契约设计

def _parse_contract_blocks(response: str, expected: list[str]) -> dict[str, dict]:
    """解析 LLM 输出的 JSON 契约块，容忍外围推理文字与单个无标签回退块。"""
    contracts: dict[str, dict] = {}
    for name, body in _JSON_BLOCK_RE.findall(response):
        if name not in expected:
            continue
        parsed = _parse_one_contract(body)
        if parsed:
            contracts[name] = parsed

    if not contracts and len(expected) == 1:
        for body in _FALLBACK_JSON_RE.findall(response):
            parsed = _parse_one_contract(body)
            if parsed:
                return {expected[0]: parsed}

    return {k: v for k, v in contracts.items() if k in expected}


def _parse_one_contract(body: str) -> dict | None:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    cols_raw = data.get("columns") if isinstance(data, dict) else None
    if not isinstance(cols_raw, list):
        return None
    columns = []
    for col in cols_raw:
        if isinstance(col, dict) and col.get("name"):
            columns.append(
                {
                    "name": str(col["name"]),
                    "type": str(col.get("type", "")),
                    "description": str(col.get("description", "")),
                }
            )
    if not columns:
        return None
    return {"source": "llm", "columns": columns}


def _render_drafts(contracts: dict[str, dict]) -> str:
    """把 yml 契约渲染为 LLM 审阅用的草稿文本。"""
    if not contracts:
        return ""
    lines = []
    for model, contract in contracts.items():
        cols = "\n".join(
            f"    - {col['name']}: {col.get('description', '')}".rstrip() for col in contract["columns"]
        )
        lines.append(f"## draft for '{model}':\n{cols}")
    return _DRAFT_TEMPLATE.format(draft_text="\n".join(lines))


def design_contracts(
    llm: LLMClient,
    instruction: str,
    schema_text: str,
    targets: list[str],
    existing_models: str = "",
    drafts: dict[str, dict] | None = None,
    required_sizes: dict[str, int] | None = None,
) -> dict[str, dict]:
    """LLM 契约确定（v2 超集语义）：yml 草稿是最小覆盖，LLM 审阅只加
    不删；无草稿则按"最大化合理覆盖"从零设计。

    ``required_sizes`` 仅作为历史模式（agent-m2）的列数提示透传进提示
    词；v2 模式不传（消融已证明列数钉死是 v1 倒退的主凶）。
    解析失败时安静降级为空（由调用方回退）。
    """
    if not targets:
        return {}
    sizes = required_sizes or {}
    user = SCHEMA_DESIGN_TEMPLATE.format(
        instruction=instruction,
        targets="\n".join(f"- {name}: exactly {sizes[name]} columns" if name in sizes else f"- {name}" for name in targets),
        sizes=(
            _SIZES_TEMPLATE.format(
                sizes_text="\n".join(f"- '{m}': exactly {n} columns" for m, n in sizes.items())
            )
            if sizes
            else ""
        ),
        drafts=_render_drafts(drafts or {}),
        schema=schema_text,
        existing_models=(
            "# EXISTING MODELS (follow their design conventions)\n" + existing_models + "\n\n"
            if existing_models
            else ""
        ),
    )
    response = llm.chat(SCHEMA_DESIGN_SYSTEM_PROMPT, user)
    return _parse_contract_blocks(response, targets)


# --------------------------------------------------------------------------- 组合入口

def build_contracts(
    llm: LLMClient,
    workspace: Path,
    instruction: str,
    schema_text: str,
    target_models: list[str],
    existing_models: str = "",
    required_sizes: dict[str, int] | None = None,
) -> dict[str, dict]:
    """契约组合入口（v2 超集语义）。

    1. 解析 yml 草稿；
    2. **全部**目标模型交 LLM 审阅/设计——yml 草稿作为最小覆盖（提示词
       要求只加不删，程序化兜底强制执行：LLM 漏掉的草稿列补回末尾）；
    3. LLM 失败或漏掉的模型回退草稿。

    v1 的"草稿列数与期望列数一致即直接采信"仲裁已移除——消融
    （agent-m2-nosize，14/64 vs 8/64）证明列数钉死是 v1 倒退的主凶；
    ``required_sizes`` 仅作为历史模式的列数提示透传。
    """
    drafts = parse_yml_contracts(workspace / "models", target_models)
    try:
        reviewed = design_contracts(
            llm,
            instruction,
            schema_text,
            target_models,
            existing_models,
            drafts=drafts,
            required_sizes=required_sizes,
        )
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: schema design failed ({type(e).__name__}: {e}); falling back to yml drafts")
        reviewed = {}
    contracts: dict[str, dict] = {}
    for model in target_models:
        draft = drafts.get(model)
        if model in reviewed:
            contract = reviewed[model]
            if draft:
                # 草稿是最小覆盖：LLM 删掉的草稿列在此程序化强制补回
                names = {c["name"] for c in contract["columns"]}
                lost = [c for c in draft["columns"] if c["name"] not in names]
                if lost:
                    contract = {**contract, "columns": [*contract["columns"], *lost]}
            contracts[model] = contract
        elif draft:
            contracts[model] = draft  # 审阅失败回退草稿
    return contracts


def render_contracts(contracts: dict[str, dict]) -> str:
    """把契约渲染为生成提示词中的紧凑文本段（含 description 语义）。"""
    lines = []
    for model, contract in contracts.items():
        lines.append(f"## model '{model}'")
        for col in contract["columns"]:
            desc = col.get("description", "")
            lines.append(f"  - {col['name']}" + (f": {desc}" if desc else ""))
    return "\n".join(lines)
