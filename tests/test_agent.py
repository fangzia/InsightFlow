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


@pytest.mark.skipif(not PLAYBOOK.exists(), reason="Spider2 data not downloaded")
def test_run_dbt_missing_project(tmp_path):
    result = run_dbt(tmp_path)
    assert not result.success
