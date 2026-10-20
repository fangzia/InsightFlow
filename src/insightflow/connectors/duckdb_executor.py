"""Spider 2.0-DBT 任务的 DuckDB + dbt 执行适配器。

两个职责：

- :func:`extract_schema` —— 将 DuckDB 文件中所有表/视图的 schema（含少量
  值采样）导出为紧凑的 SQL 风格文本，用于 LLM 上下文窗口。
- :func:`run_dbt` —— 在任务工作区中执行 ``dbt run``，返回结构化结果，
  供修复循环回传给 LLM。
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

_INTERNAL_SCHEMAS = ("information_schema", "pg_catalog", "main_pg_temp")


@dataclass
class DbtResult:
    """一次 ``dbt run`` 调用的结果。"""

    success: bool
    stdout: str
    stderr: str
    command: list[str] = field(default_factory=list)

    def error_digest(self, max_chars: int = 4000) -> str:
        """截取 stdout/stderr 尾部，作为修复提示词中的紧凑错误摘要。"""
        text = self.stdout
        if self.stderr.strip():
            text += "\n" + self.stderr
        if len(text) > max_chars:
            text = "...\n" + text[-max_chars:]
        return text


def extract_schema(db_path: Path, *, max_sample_rows: int = 3, max_value_len: int = 60) -> str:
    """导出 DuckDB 文件中所有表/视图的紧凑 schema 文本。

    每张表的格式::

        -- table <schema>.<name> (<table_type>)
        --   col1 TYPE, col2 TYPE, ...
        --   sample: col1=..., col2=...
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = con.execute(
            """
            select table_schema, table_name, table_type
            from information_schema.tables
            where table_schema not in ('information_schema', 'pg_catalog')
            order by table_schema, table_name
            """
        ).fetchall()
        parts: list[str] = []
        for schema, table, table_type in tables:
            cols = con.execute(
                """
                select column_name, data_type
                from information_schema.columns
                where table_schema = ? and table_name = ?
                order by ordinal_position
                """,
                [schema, table],
            ).fetchall()
            header = f"-- table {schema}.{table} ({table_type})"
            col_defs = ", ".join(f"{name} {dtype}" for name, dtype in cols)
            lines = [header, f"--   columns: {col_defs}"]
            if max_sample_rows > 0:
                try:
                    quoted = f'"{schema}"."{table}"'
                    rows = con.execute(f"select * from {quoted} limit {max_sample_rows}").fetchall()  # noqa: S608
                    names = [name for name, _ in cols]
                    for row in rows:
                        sample = ", ".join(
                            f"{name}={_clip(value, max_value_len)}" for name, value in zip(names, row, strict=False)
                        )
                        lines.append(f"--   sample: {sample}")
                except duckdb.Error:
                    # 例如依赖的基表已被删除的视图；schema 信息本身仍有用
                    lines.append("--   sample: <unavailable>")
            parts.append("\n".join(lines))
        return "\n\n".join(parts) if parts else "-- (no tables found)"
    finally:
        con.close()


def _clip(value: object, max_len: int) -> str:
    text = "NULL" if value is None else str(value)
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return text


def run_dbt(project_dir: Path, *, models: list[str] | None = None, timeout: int = 300) -> DbtResult:
    """在指定项目目录中执行 ``dbt run``。"""
    project_dir = Path(project_dir).resolve()
    cmd = [
        "dbt",
        "run",
        "--project-dir",
        str(project_dir),
        "--profiles-dir",
        str(project_dir),
        "--no-use-colors",
    ]
    if models:
        cmd += ["--select", " ".join(models)]
    try:
        proc = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(project_dir),
        )
    except subprocess.TimeoutExpired as e:
        out = e.stdout or ""
        err = e.stderr or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        if isinstance(err, bytes):
            err = err.decode(errors="replace")
        return DbtResult(False, out, f"TIMEOUT after {timeout}s\n{err}", cmd)
    return DbtResult(proc.returncode == 0, proc.stdout, proc.stderr, cmd)
