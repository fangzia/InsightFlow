"""dbt 模型生成器的提示词模板与响应解析。

LLM 被要求按如下格式输出，每个模型文件一个围栏代码块::

    ```sql file=models/<model_name>.sql
    select ...
    ```

:func:`parse_sql_blocks` 从响应中恢复出 ``{相对路径: 内容}`` 的映射，
并能容忍模型在代码块外围输出推理性的叙述文字。
"""

from __future__ import annotations

import re

SYSTEM_PROMPT = """\
You are an expert dbt analytics engineer working on a DuckDB project.

Task: implement the missing dbt model(s) for the project, as listed in the
"TARGET MODELS" section. Each target model must become a materialized object
(a table named exactly as specified) after `dbt run` succeeds.

Rules:
- Write idiomatic dbt SQL. You may use Jinja (`{{ ref(...) }}`, `{{ source(...) }}`)
  or plain SQL referencing existing tables/views in the same database.
- NEVER reference a target model itself, neither via `{{ ref('<target>') }}` nor via
  a raw table name like `main.<target>` or `<target>`: those tables are being built
  by this very run and referencing them is an error. Always rebuild from the
  original source tables and upstream models listed in the schema section.
- Never invent source tables: only reference tables listed in the database
  schema section, or models defined in the project.
- The project ships pre-installed dbt packages (e.g. dbt_utils) in
  `dbt_packages/`; you may use their macros without installing anything.
- If a target model design needs staging/intermediate models beyond the listed
  targets, INLINE them as CTEs inside one single SQL file. Do NOT reference
  models that you have not emitted in your answer, and do NOT add extra model
  files beyond the targets.
- If model specs (dbt schema.yml) are provided, follow their column names,
  semantics and descriptions as precisely as you can, and always include at
  least every spec column. Extra columns beyond the spec are allowed,
  especially when the column contracts list them.
- Do not add LIMIT/OFFSET unless explicitly requested; produce full results.
- Output ONLY fenced SQL blocks, one per model file, in this exact format:

```sql file=models/<model_name>.sql
-- your SQL here
```

No prose outside the blocks. Each block's `file=` path must be exactly
`models/<model_name>.sql` with the target model name.
"""

USER_TEMPLATE = """\
# TASK
{instruction}

{redesign_note}# TARGET MODELS (implement exactly these)
{targets}

{model_specs}# COLUMN CONTRACTS (the final SELECT must include AT LEAST these columns; extra columns are free; order does not matter)
{contracts}

# DATABASE SCHEMA
{schema}

{existing_models}# YOUR ANSWER
Provide the SQL for every target model listed above, using the fenced-block
format specified in the system message. Each model's output MUST include
every column of its column contract (verbatim names). Extra columns beyond
the contract are free and encouraged when plausibly useful for the task;
never drop a contract column.
"""

REPAIR_TEMPLATE = """\
# TASK
{instruction}

{redesign_note}# TARGET MODELS (implement exactly these)
{targets}

{model_specs}# COLUMN CONTRACTS (the final SELECT must include AT LEAST these columns; extra columns are free; order does not matter)
{contracts}

# DATABASE SCHEMA
{schema}

# PREVIOUS ATTEMPT (failed)
{previous_sql}

{failure_feedback}# YOUR ANSWER
Fix the failing SQL. Re-emit the SQL for every target model listed above in
the fenced-block format specified in the system message. Focus on fixing the
reported problem; do not redesign parts that already work. When the
fix concerns columns, only ADDITION is ever required: the contract is a
minimum, extra columns are free, and column order does not matter.
"""


def build_failure_feedback(error_digest: str, column_issues: str) -> str:
    """组装修复轮的失败反馈段：dbt 错误与/或列校验差异。

    两者都为空时返回通用占位段（不应发生——进入修复轮必有其一）。
    """
    sections: list[str] = []
    if error_digest:
        sections.append("# DBT ERROR OUTPUT\n" + error_digest)
    if column_issues:
        sections.append(
            "# COLUMN VERIFICATION FAILURES\n"
            "dbt ran successfully, but some materialized tables are MISSING\n"
            "required contract columns. Extra columns are fine - keep them all;\n"
            "only ADD the missing columns (order does not matter):\n"
            + column_issues
        )
    if not sections:
        sections.append("# PREVIOUS RUN FAILED\n(no diagnostics available)")
    return "\n\n".join(sections) + "\n\n"


# --------------------------------------------------------------------------- M3 critic

CRITIC_SYSTEM_PROMPT = """\
You are a strict but pragmatic data reviewer. A dbt pipeline has materialized
tables for an analytics task. Your job is to review whether the RESULT actually
answers the task - a structural check already passed (columns are present),
so focus on SEMANTICS: grain, measure definitions, and value plausibility.

Check for:
- Wrong grain: the task asks for e.g. monthly/customer-level rows but sampled
  rows suggest a different granularity.
- Implausible values: all-NULL key columns, negative quantities where the
  domain forbids them, dates far outside the task's period, ratios above 1
  when they are shares, duplicated identical rows.
- Missing semantics the instruction explicitly asks for.

Be conservative: this is a last-resort gate, and a wrong CHALLENGE costs a
full redesign. Only challenge when the evidence clearly shows the table
fails the task; when in doubt, APPROVE.

Reply with EXACTLY one line, either:
APPROVE
or
CHALLENGE: <one-sentence reason>
"""

CRITIC_TEMPLATE = """\
# TASK
{instruction}

# MATERIALIZED TABLE SAMPLES (actual data from the run)
{samples}

# YOUR VERDICT
APPROVE or CHALLENGE: <reason>
"""

# --------------------------------------------------------------------------- M3 critic v2（统计级证据）

CRITIC_V2_SYSTEM_PROMPT = """\
You are a strict data reviewer working from STATISTICAL EVIDENCE. A dbt
pipeline has materialized tables for an analytics task; the structural check
already passed (all contract columns are present). Your job: read the
statistics computed from the ACTUAL data and decide whether the result
plausibly answers the task.

Check, in order:

1. GRAIN: compare each table's row count with its source tables and with what
   the task asks for (a monthly summary has ~12-24 rows per year and key; a
   customer-level table has one row per customer; full order lines keep the
   source row count).
2. KEY MAPPING: columns named <prefix>_id should map to the id column of the
   table named <prefix>. Their value range must overlap that existing table's
   id range. A "NO OVERLAP" flag is strong evidence the join/mapping is wrong
   (a common defect: recomputing surrogate keys instead of looking them up).
3. VALUE SANITY: null-percentage, distinct counts, numeric ranges. All-NULL
   key columns, shares/ratios above 1, negative quantities, implausible
   magnitudes are defects.
4. TASK SEMANTICS: measures the instruction explicitly names (revenue, MRR,
   count of X, ...) - do the columns' magnitude and grain plausibly match?

Be conservative but not blind: with NO OVERLAP key evidence, a grain that
clearly contradicts the task, or an explicitly named measure that is visibly
wrong, CHALLENGE. When the evidence is consistent with the task, APPROVE. A
wrong CHALLENGE costs a full redesign; a blind APPROVE submits a wrong answer.

Reply with EXACTLY one line, either:
APPROVE
or
CHALLENGE: column '<col>': <observed statistic> conflicts with <task requirement>; likely cause: <cause>; fix hint: <suggestion>
"""

CRITIC_V2_TEMPLATE = """\
# TASK
{instruction}

# MATERIALIZED TABLE STATISTICS (computed from the actual run)
{stats}

# YOUR VERDICT
APPROVE or CHALLENGE: column '<col>': <stat> conflicts with <requirement>; likely cause: <cause>; fix hint: <hint>
"""


# --------------------------------------------------------------------------- M3 行数锚定

ANCHOR_SYSTEM_PROMPT = """\
You extract EXPLICIT output row-count constraints from an analytics task
instruction. A constraint counts ONLY when the instruction states the number
directly and unambiguously, e.g. "the top 10 actors", "the 5 most recent
orders", "list the first 20 customers", "exactly 12 rows".

Do NOT extract:
- counts of input rows or time windows ("last 12 months", "orders in 2020")
  unless they clearly mean the OUTPUT row count;
- implicit quantities (per month, per customer) that depend on data;
- numbers you computed or guessed yourself.

Output ONE fenced JSON block and nothing else:

```json
{"<model_name>": <row_count>}
```

Only models from TARGET MODELS; omit models without an explicit row count
(an empty object is a valid answer).
"""

ANCHOR_TEMPLATE = """\
# TASK
{instruction}

# TARGET MODELS
{targets}

# YOUR ANSWER
One JSON block mapping model names to their explicitly required row counts.
"""


# --------------------------------------------------------------------------- M3 粒度词锚定

GRAIN_SYSTEM_PROMPT = """\
You extract the OUTPUT GRAIN of each target model from an analytics task
instruction. The grain is what ONE output row represents:

- time_unit: the temporal aggregation the instruction explicitly asks for -
  one of "day", "week", "month", "quarter", "year", or null when the
  instruction does not state a time dimension.
- entity: the subject one row is about, as a snake_case column-style hint
  (e.g. "account_id", "customer", "region", "driver"), or null when the
  output is one row per time unit (global totals).

Examples:
"Monthly recurring revenue for each account"
  -> {"time_unit": "month", "entity": "account_id"}
"Number of orders per customer"
  -> {"time_unit": null, "entity": "customer_id"}
"Daily active user counts"
  -> {"time_unit": "day", "entity": null}
"Top rated movies overall"
  -> {"time_unit": null, "entity": null} (omit this model)

Only extract what the instruction states; do not infer from table names or
guess. Omit models whose grain the instruction does not determine.

Output ONE fenced JSON block and nothing else:

```json
{"<model_name>": {"time_unit": "month", "entity": "account_id"}}
```
"""

GRAIN_TEMPLATE = """\
# TASK
{instruction}

# TARGET MODELS
{targets}

# YOUR ANSWER
One JSON block with the output grain per model (omit undetermined models).
"""

_BLOCK_RE = re.compile(r"```sql\s+file=(\S+)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_FALLBACK_RE = re.compile(r"```(?:sql)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_sql_blocks(response: str, expected_models: list[str]) -> dict[str, str]:
    """从 LLM 响应中解析围栏 SQL 代码块。

    返回 ``models/<名称>.sql`` -> 内容 的映射。当无标签代码块数量与
    期望模型数一致时，回退到按顺序匹配无标签代码块。
    """
    files: dict[str, str] = {}
    for path, content in _BLOCK_RE.findall(response):
        path = path.strip().strip("`")
        if not path.startswith("models/"):
            path = "models/" + path.lstrip("/").removeprefix("models/")
        content = content.strip()
        if path and content:
            files[path] = content

    if not files and len(expected_models) == 1:
        untagged = [c.strip() for c in _FALLBACK_RE.findall(response) if c.strip()]
        if len(untagged) == 1:
            return {f"models/{expected_models[0]}.sql": untagged[0]}

    # 只保留与目标模型对应的文件（忽略多余的代码块）。
    expected_paths = {f"models/{name}.sql" for name in expected_models}
    return {path: content for path, content in files.items() if path in expected_paths}
