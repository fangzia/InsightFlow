"""Tests for M1 agent components (no network / dbt required)."""

from pathlib import Path

import pytest

from insightflow.agent.prompts import parse_sql_blocks
from insightflow.connectors.duckdb_executor import extract_schema, run_dbt
from insightflow.eval.run_benchmark import render_model_specs

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAYBOOK = REPO_ROOT / "third_party" / "Spider2" / "spider2-dbt" / "examples" / "playbook001"


class TestParseSqlBlocks:
    def test_tagged_blocks(self):
        resp = """Some reasoning text.

```sql file=models/foo.sql
select 1
```

```sql file=models/bar.sql
select 2
```
"""
        files = parse_sql_blocks(resp, ["foo", "bar"])
        assert files == {
            "models/foo.sql": "select 1",
            "models/bar.sql": "select 2",
        }

    def test_ignores_unexpected_model(self):
        resp = "```sql file=models/other.sql\nselect 1\n```"
        assert parse_sql_blocks(resp, ["foo"]) == {}

    def test_single_untagged_block_fallback(self):
        resp = "Blah blah.\n```\nselect 42\n```\n"
        assert parse_sql_blocks(resp, ["foo"]) == {"models/foo.sql": "select 42"}

    def test_reasoning_prose_around_blocks(self):
        resp = "I should implement foo.\n```sql file=models/foo.sql\nselect 1\n```\nDone."
        assert parse_sql_blocks(resp, ["foo"]) == {"models/foo.sql": "select 1"}


@pytest.mark.skipif(not PLAYBOOK.exists(), reason="Spider2 data not downloaded")
class TestExtractSchema:
    def test_playbook_schema(self):
        db = PLAYBOOK / "playbook.duckdb"
        text = extract_schema(db, max_sample_rows=2)
        assert "-- table playbook.playbook" not in text  # schema is `main`
        assert "sessions" in text
        assert "customer_id" in text
        assert "sample:" in text

    def test_zero_samples(self):
        text = extract_schema(PLAYBOOK / "playbook.duckdb", max_sample_rows=0)
        assert "sample:" not in text


@pytest.mark.skipif(not PLAYBOOK.exists(), reason="Spider2 data not downloaded")
class TestRenderModelSpecs:
    def test_playbook_specs(self):
        specs = render_model_specs(PLAYBOOK / "models", ["cpa_and_roas", "attribution_touches"])
        assert "spec for model 'cpa_and_roas'" in specs
        assert "cost_per_acquisition" in specs
        # sources are not model specs
        assert "sessions" not in specs.split("columns")[0]

    def test_no_matching_specs(self):
        assert render_model_specs(PLAYBOOK / "models", ["nonexistent"]) == ""


class TestDbtResult:
    def test_error_digest_truncates(self):
        from insightflow.connectors.duckdb_executor import DbtResult

        r = DbtResult(False, "x" * 10, "y")
        assert r.error_digest(max_chars=5).startswith("...")
        assert len(r.error_digest(max_chars=5)) <= 5 + 4

    def test_triage_digest_extracts_error_block(self):
        from insightflow.connectors.duckdb_executor import DbtResult

        stdout = "\n".join(
            [
                "09:00:00  Running with dbt=1.8.10",
                "09:00:00  Registered adapter: duckdb=1.8.4",
                "09:00:01  1 of 2 START sql table model main.stg [RUN]",
                "09:00:01  1 of 2 OK created sql table model main.stg [SELECT 123 in 0.1s]",
                "09:00:02  2 of 2 START sql table model main.fct [RUN]",
                "09:00:02  2 of 2 ERROR creating sql table model main.fct [ERROR in 0.2s]",
                "09:00:02  Completed with 1 error and 0 warnings:",
                "09:00:02  Database Error in model fct (models/fct.sql)",
                "  Catalog Error: Table with name stg2 does not exist!",
                "  Did you mean \"stg\"?",
                "09:00:02  Done. PASS=1 WARN=0 ERROR=1 SKIP=0 TOTAL=2",
            ]
        )
        d = DbtResult(False, stdout, "").triage_digest()
        assert "DBT RUN FAILED (triaged)" in d
        assert "ok models: ['stg']" in d
        assert "failed models: ['fct']" in d
        assert "Catalog Error: Table with name stg2 does not exist!" in d
        assert "Running with dbt" not in d

    def test_triage_digest_falls_back_without_error_marker(self):
        from insightflow.connectors.duckdb_executor import DbtResult

        r = DbtResult(False, "", "TIMEOUT after 5s\nboom")
        assert "TIMEOUT" in r.triage_digest()


class TestLint:
    def test_dangling_ref_detected_with_hint(self):
        from insightflow.agent.graph import _find_dangling_refs

        files = {"models/fct.sql": "select * from {{ ref('staging_events') }}"}
        issues = _find_dangling_refs(files, {"stg_events", "raw_events"})
        assert len(issues) == 1
        assert "staging_events" in issues[0]
        assert "stg_events" in issues[0]  # difflib 近似名提示

    def test_refs_to_emitted_models_are_valid(self):
        from insightflow.agent.graph import _find_dangling_refs

        files = {
            "models/a.sql": "select * from {{ ref('b') }}",
            "models/b.sql": "select 1",
        }
        assert _find_dangling_refs(files, set()) == []

    def test_self_ref_is_not_dangling(self):
        from insightflow.agent.graph import _find_dangling_refs

        files = {"models/a.sql": "select * from {{ ref('a') }}"}  # 自引用由专门检查负责
        assert _find_dangling_refs(files, set()) == []

    def test_ref_to_bare_table_is_rejected(self):
        """dbt 语义：ref() 只指向模型节点，裸源表必须直接用表名。"""
        from insightflow.agent.graph import _find_dangling_refs

        files = {"models/fct.sql": "select * from {{ ref('zuora_account_data') }}"}
        issues = _find_dangling_refs(files, {"zuora_account_data_model"})
        assert len(issues) == 1
        assert "raw source table" in issues[0]


class TestMissingAncestorClosure:
    def test_missing_ancestors_added_recursively(self, tmp_path):
        from insightflow.connectors.duckdb_executor import _expand_missing_ancestors

        # models/a -> b -> c，全部未物化 → 闭包补全 b、c
        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "a.sql").write_text("select * from {{ ref('b') }}")
        (tmp_path / "models" / "b.sql").write_text("select * from {{ ref('c') }}")
        (tmp_path / "models" / "c.sql").write_text("select 1")
        import duckdb

        duckdb.connect(str(tmp_path / "x.duckdb")).close()  # 空库：无表
        out = _expand_missing_ancestors(tmp_path, ["a"])
        assert out[0] == "a"
        assert set(out) == {"a", "b", "c"}

    def test_no_ancestors_when_all_materialized(self, tmp_path):
        from insightflow.connectors.duckdb_executor import _expand_missing_ancestors

        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "a.sql").write_text("select * from {{ ref('b') }}")
        (tmp_path / "models" / "b.sql").write_text("select 1")
        import duckdb

        con = duckdb.connect(str(tmp_path / "x.duckdb"))
        con.execute("create table b as select 1")
        con.close()
        # 已物化的祖先绝不进 select（superstore001 回归：重建会打乱 row_number 代理键）
        assert _expand_missing_ancestors(tmp_path, ["a"]) == ["a"]


class TestPruneParseBlockers:
    def _mk(self, tmp_path):
        from insightflow.connectors.duckdb_executor import DbtResult

        (tmp_path / "models" / "mart").mkdir(parents=True, exist_ok=True)
        blocker = tmp_path / "models" / "mart" / "bad.sql"
        blocker.write_text("select * from {{ ref('int_missing') }}")
        return blocker, DbtResult(
            False,
            "Model 'model.p.bad' (models\\mart\\bad.sql) depends on a node named "
            "'int_missing' which was not found",
            "",
        )

    def test_prune_deletes_unreferenced_blocker(self, tmp_path):
        from insightflow.connectors.duckdb_executor import prune_parse_blockers

        blocker, result = self._mk(tmp_path)
        (tmp_path / "models" / "good.sql").write_text("select 1")
        deleted = prune_parse_blockers(tmp_path, result)
        assert len(deleted) == 1 and deleted[0].replace("\\", "/") == "models/mart/bad.sql"
        assert not blocker.exists()

    def test_prune_protects_agent_files(self, tmp_path):
        from insightflow.connectors.duckdb_executor import prune_parse_blockers

        blocker, result = self._mk(tmp_path)
        deleted = prune_parse_blockers(tmp_path, result, protect={blocker})
        assert deleted == []
        assert blocker.exists()

    def test_prune_skips_referenced_blocker(self, tmp_path):
        from insightflow.connectors.duckdb_executor import prune_parse_blockers

        blocker, result = self._mk(tmp_path)
        # 另一个模型引用了 bad → 删除会级联，不剪
        (tmp_path / "models" / "other.sql").write_text("select * from {{ ref('bad') }}")
        deleted = prune_parse_blockers(tmp_path, result)
        assert deleted == []
        assert blocker.exists()


class TestRepairRefsToTables:
    def test_rewrites_jinja_ref_to_direct_table(self, tmp_path):
        from insightflow.agent.graph import _repair_refs_to_tables

        (tmp_path / "models" / "staging").mkdir(parents=True)
        f = tmp_path / "models" / "staging" / "stg.sql"
        f.write_text("select * from {{ ref('circuits') }}")
        schema_text = "-- table main.circuits (BASE TABLE)\n--   columns: id INT"
        assert _repair_refs_to_tables(tmp_path, schema_text) == 1
        assert f.read_text(encoding="utf-8") == 'select * from "main"."circuits"'  # 无残留 Jinja 定界符

    def test_keeps_refs_to_real_models(self, tmp_path):
        from insightflow.agent.graph import _repair_refs_to_tables

        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "stg_events.sql").write_text("select 1")
        f = tmp_path / "models" / "fct.sql"
        f.write_text("select * from {{ ref('stg_events') }}")
        schema_text = "-- table main.stg_events (BASE TABLE)\n--   columns: id INT"
        assert _repair_refs_to_tables(tmp_path, schema_text) == 0
        assert "ref('stg_events')" in f.read_text(encoding="utf-8")


class TestCollectValueEvidence:
    """critic v2 的统计级证据采集（纯查询，无 LLM）。"""

    def _mk_db(self, tmp_path):
        import duckdb

        con = duckdb.connect(str(tmp_path / "x.duckdb"))
        # 代理键漂移场景：fct 的 dim_products_id 被重算成 [10001,19994]
        # （错误偏移），shipped dim_products.id 为 [101,4444] → 值域不相交
        con.execute("create table dim_products as select 101 + i as id from range(4344) t(i)")
        con.execute(
            "create table fct_sales as "
            "select 10001 + i as dim_products_id, 1 + i as row_num from range(9994) t(i)"
        )
        con.execute("create table src_orders as select * from range(100) t(i)")
        return con

    def test_key_no_overlap_flagged(self, tmp_path):
        from insightflow.agent.graph import _collect_value_evidence

        con = self._mk_db(tmp_path)
        sql = "select * from {{ ref('src_orders') }}"
        out = _collect_value_evidence(con, "fct_sales", sql, {"fct_sales", "dim_products", "src_orders"})
        assert "key check: 'dim_products_id' range=[10001,19994] vs existing 'dim_products'.id range=[101,4444]" in out
        assert "NO OVERLAP" in out
        con.close()

    def test_grain_ratio_rendered(self, tmp_path):
        from insightflow.agent.graph import _collect_value_evidence

        con = self._mk_db(tmp_path)
        sql = "select * from {{ ref('src_orders') }}"
        out = _collect_value_evidence(con, "fct_sales", sql, {"fct_sales", "dim_products", "src_orders"})
        assert "grain: 9994 rows vs source 'src_orders' 100 rows (ratio" in out
        con.close()

    def test_null_and_distinct_stats(self, tmp_path):
        from insightflow.agent.graph import _collect_value_evidence

        con = self._mk_db(tmp_path)
        out = _collect_value_evidence(con, "fct_sales", "", {"fct_sales", "dim_products"})
        assert "distinct=9994" in out
        assert "null=0%" in out
        con.close()


class TestRowAnchors:
    def test_parse_valid_anchors(self):
        from insightflow.agent.graph import _parse_row_anchors

        resp = '```json\n{"most_podiums": 10}\n```'
        assert _parse_row_anchors(resp, ["most_podiums", "other"]) == {"most_podiums": 10}

    def test_parse_drops_invalid(self):
        from insightflow.agent.graph import _parse_row_anchors

        # 非目标模型 / 非正数 / 布尔 / 坏 JSON 全部丢弃
        resp = '```json\n{"unknown_model": 5, "m": -1, "n": true, "k": "10"}\n```'
        assert _parse_row_anchors(resp, ["m", "n", "k"]) == {}

    def test_parse_no_block(self):
        from insightflow.agent.graph import _parse_row_anchors

        assert _parse_row_anchors("no json here", ["m"]) == {}


class TestDiffTableColumns:
    def _mk(self, tmp_path):
        import duckdb

        con = duckdb.connect(str(tmp_path / "x.duckdb"))
        con.execute("create table a as select 1 as id, 'x' as name, 1.5 as val")
        con.execute("create table b as select 1 as id, 'y' as name, 1.5 as val")  # 仅 name 列不同
        return con

    def test_weak_column_isolated(self, tmp_path):
        from insightflow.agent.graph import _diff_table_columns

        con = self._mk(tmp_path)
        rows_a, rows_b, flag, weak = _diff_table_columns(con, "a", "b")
        assert (rows_a, rows_b, flag) == (1, 1, None)
        assert weak == ["name"]
        con.close()

    def test_row_mismatch_flagged(self, tmp_path):
        from insightflow.agent.graph import _diff_table_columns

        con = self._mk(tmp_path)
        con.execute("create table c as select 1 as id, 'x' as name, 1.5 as val union all select 2 as id, 'x' as name, 1.5 as val")
        _, _, flag, _ = _diff_table_columns(con, "a", "c")
        assert flag == "ROWS"
        con.close()

    def test_identical_tables_no_weak(self, tmp_path):
        from insightflow.agent.graph import _diff_table_columns

        con = self._mk(tmp_path)
        con.execute("create table d as select * from a")
        _, _, flag, weak = _diff_table_columns(con, "a", "d")
        assert flag is None and weak == []
        con.close()

    def test_missing_column_in_b_is_weak(self, tmp_path):
        from insightflow.agent.graph import _diff_table_columns

        con = self._mk(tmp_path)
        con.execute("create table e as select 1 as id, 'x' as name")
        _, _, flag, weak = _diff_table_columns(con, "a", "e")
        assert flag is None and weak == ["val"]
        con.close()


class TestGrainAnchor:
    def test_parse_grain_specs(self):
        from insightflow.agent.graph import _parse_grain_specs

        resp = '```json\n{"mrr": {"time_unit": "month", "entity": "account_id"}, "x": {"time_unit": "bad", "entity": 5}}\n```'
        out = _parse_grain_specs(resp, ["mrr", "x"])
        assert out == {"mrr": {"time_unit": "month", "entity": "account_id"}}

    def test_parse_drops_both_null(self):
        from insightflow.agent.graph import _parse_grain_specs

        resp = '```json\n{"m": {"time_unit": null, "entity": null}}\n```'
        assert _parse_grain_specs(resp, ["m"]) == {}

    def test_check_grain_monthly_per_entity_ok(self, tmp_path):
        import duckdb

        from insightflow.agent.graph import _check_grain_anchor

        con = duckdb.connect(str(tmp_path / "x.duckdb"))
        # 12 个月 × 5 个 account = 60 行，符合 monthly-per-account
        con.execute(
            "create table mrr as select "
            "date '2024-01-15' + interval (m) month as month_date, "
            "a as account_id, 1.0 as amount "
            "from range(12) t(m), range(5) u(a)"
        )
        assert _check_grain_anchor(con, "mrr", {"time_unit": "month", "entity": "account_id"}) is None
        con.close()

    def test_check_grain_row_excess_flagged(self, tmp_path):
        import duckdb

        from insightflow.agent.graph import _check_grain_anchor

        con = duckdb.connect(str(tmp_path / "x.duckdb"))
        # 全局 monthly（无实体）→ 期望 12 行；实际 120 行（10x 过细）
        con.execute(
            "create table bad as select "
            "date '2024-01-15' + interval (m) month as month_date, "
            "a as account_id, 1.0 as amount "
            "from range(12) t(m), range(10) u(a)"
        )
        issue = _check_grain_anchor(con, "bad", {"time_unit": "month", "entity": None})
        assert issue is not None and "120 rows" in issue and "month" in issue
        con.close()

    def test_check_grain_row_deficit_flagged_from_source(self, tmp_path):
        import duckdb

        from insightflow.agent.graph import _check_grain_anchor

        con = duckdb.connect(str(tmp_path / "x.duckdb"))
        # 源表 50 个 customer；输出过粗聚合只 5 行 → 亏缺（源表基数可见）
        con.execute("create table src as select i as customer_id from range(50) t(i)")
        con.execute("create table agg as select i % 5 as customer_id from range(5) t(i)")
        sql = "select customer_id, count(*) from {{ ref('src') }} group by 1"
        issue = _check_grain_anchor(con, "agg", {"time_unit": None, "entity": "customer_id"}, sql)
        assert issue is not None and "fewer" in issue and "50" in issue
        con.close()

    def test_check_grain_no_matching_columns_skips(self, tmp_path):
        import duckdb

        from insightflow.agent.graph import _check_grain_anchor

        con = duckdb.connect(str(tmp_path / "x.duckdb"))
        con.execute("create table t as select 1 as a, 2 as b")
        assert _check_grain_anchor(con, "t", {"time_unit": "month", "entity": "account_id"}) is None
        con.close()


@pytest.mark.skipif(not PLAYBOOK.exists(), reason="Spider2 data not downloaded")
def test_run_dbt_missing_project(tmp_path):
    result = run_dbt(tmp_path)
    assert not result.success
