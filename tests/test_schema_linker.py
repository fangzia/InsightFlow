"""Schema Linker 与 Verifier 的单元测试（不依赖 LLM 端点）。"""

from pathlib import Path

from insightflow.agent.graph import _column_diff
from insightflow.agent.prompts import build_failure_feedback
from insightflow.agent.schema_linker import (
    _parse_contract_blocks,
    build_contracts,
    parse_yml_contracts,
    render_contracts,
)

YML = """
models:
  - name: dim_customer
    description: 客户维度表
    columns:
      - name: customer_id
        description: 主键
      - name: full_name
  - name: other_model
    columns:
      - name: x
"""


def test_parse_yml_contracts(tmp_path: Path):
    (tmp_path / "schema.yml").write_text(YML, encoding="utf-8")
    contracts = parse_yml_contracts(tmp_path, ["dim_customer", "missing_model"])
    assert list(contracts) == ["dim_customer"]  # 未声明的模型不产生契约
    c = contracts["dim_customer"]
    assert c["source"] == "yml"
    assert [col["name"] for col in c["columns"]] == ["customer_id", "full_name"]


def test_parse_contract_blocks_with_tags():
    response = """Here is my design.

```json file=fct_sales
{"columns": [{"name": "region", "type": "VARCHAR"}, {"name": "total", "type": "DOUBLE"}]}
```
"""
    contracts = _parse_contract_blocks(response, ["fct_sales"])
    assert contracts["fct_sales"]["source"] == "llm"
    assert [c["name"] for c in contracts["fct_sales"]["columns"]] == ["region", "total"]


def test_parse_contract_blocks_ignores_wrong_and_bad_blocks():
    response = """
```json file=not_target
{"columns": [{"name": "a"}]}
```
```json file=fct_sales
not valid json
```
"""
    assert _parse_contract_blocks(response, ["fct_sales"]) == {}


def test_parse_contract_blocks_single_untagged_fallback():
    response = '```json\n{"columns": [{"name": "k", "type": "INTEGER"}]}\n```'
    contracts = _parse_contract_blocks(response, ["dim_key"])
    assert contracts["dim_key"]["columns"][0]["name"] == "k"


def test_render_contracts():
    text = render_contracts(
        {
            "m": {
                "source": "llm",
                "columns": [{"name": "a", "description": "desc a"}, {"name": "b", "description": ""}],
            }
        }
    )
    assert "## model 'm'" in text
    assert "  - a: desc a" in text
    assert "  - b" in text and ": b" not in text.split("  - b")[1].split("\n")[0]
    assert render_contracts({}) == ""


def test_column_diff_superset():
    # 超集语义：只有缺失列算失败；多余列与顺序均免费
    msg = _column_diff("m", ["a", "b"], ["a", "x"])
    assert "MISSING required columns ['b']" in msg
    assert "unexpected" not in msg
    assert "order" not in msg.split("Extra columns")[0]  # 顺序不再是问题
    # 修复提示只引导补列、不引导裁剪
    msg2 = _column_diff("m", ["a"], ["a", "x", "y"])
    assert "Extra columns are allowed" in msg2


class _FakeLLM:
    """不依赖端点的假 LLM：固定响应或抛错。"""

    def __init__(self, response: str | None = None):
        self.response = response

    def chat(self, system: str, user: str) -> str:
        if self.response is None:
            raise RuntimeError("llm down")
        return self.response


def test_build_contracts_yml_is_floor(tmp_path: Path):
    # v2：yml 草稿是最小覆盖，LLM 审阅只加不删；删掉的草稿列被程序化补回
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "schema.yml").write_text(YML, encoding="utf-8")
    llm = _FakeLLM(
        '```json file=dim_customer\n'
        '{"columns": [{"name": "customer_id"}, {"name": "email"}]}\n'
        "```\n"
    )
    contracts = build_contracts(llm, tmp_path, "instr", "schema", ["dim_customer"])
    names = [c["name"] for c in contracts["dim_customer"]["columns"]]
    assert names == ["customer_id", "email", "full_name"]  # 草稿列 full_name 强制补回
    assert contracts["dim_customer"]["source"] == "llm"


def test_build_contracts_llm_failure_falls_back_to_draft(tmp_path: Path):
    # LLM 挂掉时回退 yml 草稿
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "schema.yml").write_text(YML, encoding="utf-8")
    contracts = build_contracts(_FakeLLM(None), tmp_path, "instr", "schema", ["dim_customer"])
    assert [c["name"] for c in contracts["dim_customer"]["columns"]] == ["customer_id", "full_name"]
    assert contracts["dim_customer"]["source"] == "yml"


def test_build_failure_feedback_both_and_none():
    both = build_failure_feedback("dbt boom", "model 'm': mismatch")
    assert "# DBT ERROR OUTPUT" in both
    assert "# COLUMN VERIFICATION FAILURES" in both
    none = build_failure_feedback("", "")
    assert "no diagnostics" in none
