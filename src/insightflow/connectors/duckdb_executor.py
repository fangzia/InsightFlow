"""Spider 2.0-DBT 任务的 DuckDB + dbt 执行适配器。

两个职责：

- :func:`extract_schema` —— 将 DuckDB 文件中所有表/视图的 schema（含少量
  值采样）导出为紧凑的 SQL 风格文本，用于 LLM 上下文窗口。
- :func:`run_dbt` —— 在任务工作区中执行 ``dbt run``，返回结构化结果，
  供修复循环回传给 LLM。
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import yaml

_INTERNAL_SCHEMAS = ("information_schema", "pg_catalog", "main_pg_temp")

# dbt 1.8 --no-use-colors 的模型进度行：`N of M OK/ERROR/START ...`（triage 用）
_OK_MODEL_RE = re.compile(r"^\d+ of \d+ OK created \w+ .*? model (?:main\.)?(\S+)", re.IGNORECASE)
_ERR_MODEL_RE = re.compile(r"^\d+ of \d+ ERROR creating \w+ .*? model (?:main\.)?(\S+)", re.IGNORECASE)
_PROGRESS_RE = re.compile(r"^\d+ of \d+ (?:START|CREATED) ", re.IGNORECASE)
_TS_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\s+")
# 其余噪声行前缀（时间戳剥离后）
_NOISE_PREFIXES = (
    "Completed ",
    "Finished ",
    "Started ",
    "Running with ",
    "This version ",
    "Update available ",
    "Registered adapter ",
    "Unable to do partial",
    "Dependency Error",
    "all logs ",
)
_ERROR_MARKERS = (
    "encountered an error",
    "database error",
    "compilation error",
    "runtime error",
    "parsing error",
    "catalog error",
    "binder error",
    "parser error",
    "syntax error",
)
# 模型 SQL 中的 ref() 调用（缺失祖先闭包展开用）
_REF_CALL_RE = re.compile(r"ref\(\s*['\"]([^'\"]+)['\"]\s*\)")


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

    def triage_digest(self, max_chars: int = 3000) -> str:
        """把 dbt 原始输出提炼为结构化错误摘要（M3 triage）。

        相比 :meth:`error_digest` 的原始尾部日志，这里主动过滤进度噪声，
        并先行提取"哪些模型物化成功 / 失败"，让修复轮 LLM 一眼看到：

        - ok / failed 模型清单（多模型项目中部分成功是常见状态）；
        - 自第一个错误标记行起的去时间戳错误块（含 dbt 的缩进细节）。

        提炼失败（如 timeout 的 stderr-only 场景）回退到原始摘要。
        """
        raw = self.stdout + (("\n" + self.stderr) if self.stderr.strip() else "")
        ok_models: list[str] = []
        err_models: list[str] = []
        cleaned: list[str] = []
        for line in raw.splitlines():
            s = _TS_RE.sub("", line).rstrip()
            if m := _OK_MODEL_RE.match(s):
                ok_models.append(m.group(1))
                continue
            if m := _ERR_MODEL_RE.match(s):
                err_models.append(m.group(1))
                cleaned.append(s)
                continue
            if not s or _PROGRESS_RE.match(s):
                continue
            if any(s.startswith(p) for p in _NOISE_PREFIXES):
                continue
            cleaned.append(s)
        # 自第一个错误标记行起截取（错误总在日志尾部；其后的行全是错误上下文）
        start = next(
            (i for i, s in enumerate(cleaned) if any(m in s.lower() for m in _ERROR_MARKERS)),
            None,
        )
        if start is None:
            return self.error_digest(2000)
        block = "\n".join(cleaned[start:])
        if len(block) > max_chars:
            block = "...\n" + block[-max_chars:]
        header = "DBT RUN FAILED (triaged)"
        if ok_models:
            header += f"\nok models: {ok_models}"
        if err_models:
            header += f"\nfailed models: {err_models}"
        return f"{header}\n--- error output ---\n{block}"


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


def needs_deps(project_dir: Path) -> bool:
    """需要执行 ``dbt deps`` 的机械判定。

    - packages.yml 声明的 hub 包缺失对应目录（git/local 包无法廉价校验
      revision，不参与判断——缺失时 dbt 会自带报错）；
    - **破损的 vendored 包**：dbt_packages/ 下存在目录但缺 dbt_project.yml
      （chinook001 的 dbt_external_tables 即此——dbt 直接拒载项目），
      重新 deps 可恢复。
    """
    pkg_dir = Path(project_dir) / "dbt_packages"
    if pkg_dir.is_dir():
        for d in pkg_dir.glob("*"):
            if d.is_dir() and not (d / "dbt_project.yml").exists():
                return True
    pkg_yml = Path(project_dir) / "packages.yml"
    if not pkg_yml.exists():
        return False
    try:
        pkgs = yaml.safe_load(pkg_yml.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return False
    hub_names = [str(p["package"]) for p in pkgs.get("packages") or [] if isinstance(p, dict) and p.get("package")]
    if not hub_names:
        return False
    installed = {d.name for d in pkg_dir.glob("*")}
    # hub 包安装目录不带 org 前缀（dbt-labs/dbt_utils -> dbt_packages/dbt_utils）
    return any(name not in installed and name.split("/")[-1] not in installed for name in hub_names)


def _expand_missing_ancestors(project_dir: Path, models: list[str]) -> list[str]:
    """返回 models ∪ {ref 闭包中**表未物化**的祖先模型}（M3）。

    为什么不是 ``dbt --select +model``：加号选择器会**重建**全部祖先，
    而已物化的祖先往往就是 gold 正确形态——shipped dim 表的代理键
    （``row_number()`` 无稳定排序）重建后会重排，下游 FK 全错
    （m3-step1 冒烟中 superstore001 的回归即此）。因此本函数只**补缺**：

    - 沿 ref 闭包 BFS（models/ 与 dbt_packages/ 下的 .sql 文件）；
    - 表已存在于工作区 .duckdb 的祖先**不进 select**（保持 shipped 状态）；
    - 顺带的效果：坏分支（如引用不存在源表的 tmp 模型）不在闭包内
      就不会被物化，run 得以绕开它们。
    """
    model_files: dict[str, Path] = {}
    for base in (project_dir / "models", project_dir / "dbt_packages"):
        if base.is_dir():
            for f in base.rglob("*.sql"):
                model_files.setdefault(f.stem, f)
    present: set[str] = set()
    db_file = next(project_dir.glob("*.duckdb"), None)
    if db_file is not None:
        con = duckdb.connect(str(db_file), read_only=True)
        try:
            present = {
                r[0]
                for r in con.execute(
                    "select table_name from information_schema.tables "
                    "where table_schema not in ('information_schema', 'pg_catalog')"
                ).fetchall()
            }
        finally:
            con.close()
    extra: list[str] = []
    seen: set[str] = set(models)
    queue = list(models)
    while queue:
        name = queue.pop()
        f = model_files.get(name)
        if f is None:
            continue
        try:
            sql = f.read_text(encoding="utf-8")
        except OSError:
            continue
        for m in _REF_CALL_RE.finditer(sql):
            ref = m.group(1)
            if ref in seen or ref in present:
                continue
            seen.add(ref)
            if ref in model_files:
                extra.append(ref)
                queue.append(ref)
    return [*models, *extra]


_YML_PARSE_ERR_RE = re.compile(r"Error reading \S+: (.+?\.ya?ml) - Runtime Error", re.IGNORECASE)
_MISSING_NODE_RE = re.compile(
    r"Model '([\w.]+)' \((models[/\\][^)]+?\.sql)\) depends on a node named '(\S+)' which was not found"
)


def _resolve_reported_path(project_dir: Path, reported: str, package: str | None = None) -> Path | None:
    """解析 dbt 错误中报告的文件路径为绝对路径（不存在返回 None）。

    - yml 的 *Error reading* 路径相对 ``models/``；
    - sql 的 *not found* 路径含 ``models/`` 前缀；
    - **包内模型**（命名空间 ``model.<pkg>.<name>``）的路径相对包根
      ``dbt_packages/<pkg>/``（recharge001 的 recharge_source.tmp 即此）。
    """
    rel = reported.replace("\\", "/").lstrip("/")
    candidates = []
    if package:
        candidates.append(project_dir / "dbt_packages" / package / rel)
        candidates.append(project_dir / "dbt_packages" / package / "models" / rel)
    candidates.append(project_dir / "models" / rel)
    candidates.append(project_dir / rel)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def prune_parse_blockers(project_dir: Path, result: DbtResult, protect: set[Path] | None = None) -> list[str]:
    """删除 shipped 且**无依赖者**的 parse 阻塞模型文件（M3）。

    shipped 项目可能带引用不存在节点的坏模型（如 netflix001 的
    ``mart/netflix_original_programs.sql``），dbt 整项目 parse 失败，agent
    又无法覆盖子目录文件——只能剪除。安全条件：

    - 仅删 *depends on a node which was not found* 点名的 .sql；
    - ``protect`` 中的路径（agent 本轮生成的文件——修复轮的职责）不删；
    - 删前检查全项目无其他模型 ref 该文件（有依赖者删除会级联出新
      "node not found"，此时留给修复轮/放弃）；
    - 每次最多删 2 个。
    """
    out = result.stdout + "\n" + result.stderr
    deleted: list[str] = []
    matches = _MISSING_NODE_RE.findall(out)
    if not matches:
        return deleted
    protect_resolved = {p.resolve() for p in (protect or set())}
    all_models: dict[str, Path] = {}
    for base in (project_dir / "models", project_dir / "dbt_packages"):
        if base.is_dir():
            for f in base.rglob("*.sql"):
                all_models.setdefault(f.stem, f)
    for full_name, rel_reported, _node in matches[:2]:
        # 命名空间 model.<pkg>.<name> → 包内路径
        parts = full_name.split(".")
        package = parts[1] if len(parts) >= 3 and parts[0] == "model" else None
        f = _resolve_reported_path(project_dir, rel_reported, package=package)
        if f is None or f.resolve() in protect_resolved:
            continue
        stem = f.stem
        # 有其他模型引用该文件 → 删除会级联，不剪
        referenced = False
        for other in all_models.values():
            if other == f:
                continue
            try:
                text = other.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if any(m.group(1) == stem for m in _REF_CALL_RE.finditer(text)):
                referenced = True
                break
        if referenced:
            continue
        f.unlink()
        deleted.append(str(f.relative_to(project_dir)))
        all_models.pop(stem, None)
    return deleted


def run_dbt(
    project_dir: Path,
    *,
    models: list[str] | None = None,
    timeout: int = 300,
    install_deps: bool = False,
    select_ancestors: bool = False,
    quarantine_yml: bool = False,
) -> DbtResult:
    """在指定项目目录中执行 ``dbt run``。

    - ``install_deps``：packages.yml 声明的 hub 包缺失时先执行 ``dbt deps``
      （best-effort，失败不阻塞——真正的缺包错误由 run 本身报告）。
    - ``select_ancestors``：select 追加**缺失祖先闭包**（见
      :func:`_expand_missing_ancestors`）——补齐未物化的上游模型，
      但绝不重建已物化的表。
    - ``quarantine_yml``：dbt 报 *Parsing Error reading <file>.yml* 时把
      该 yml 移到 ``_quarantined_yml/`` 后重试（官方 shipped 项目存在坏
      yml——未加引号的 Jinja / 缩进断裂；移出后 dbt 不再解析它，模型
      SQL 不受影响）。最多隔离 3 个文件。
    """
    project_dir = Path(project_dir).resolve()
    if install_deps and needs_deps(project_dir):
        subprocess.run(  # noqa: S603
            [
                "dbt",
                "deps",
                "--project-dir",
                str(project_dir),
                "--profiles-dir",
                str(project_dir),
                "--no-use-colors",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(project_dir),
        )
    quarantined = 0
    result: DbtResult | None = None
    while True:
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
            selector = " ".join(_expand_missing_ancestors(project_dir, models) if select_ancestors else models)
            cmd += ["--select", selector]
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
        result = DbtResult(proc.returncode == 0, proc.stdout, proc.stderr, cmd)
        # yml 隔离重试：Parsing Error 指名坏 yml → 移出 models/ 再跑
        if result.success or not quarantine_yml or quarantined >= 3:
            return result
        m = _YML_PARSE_ERR_RE.search(result.stdout + "\n" + result.stderr)
        if m is None:
            return result
        # yml 的 Error reading 路径相对 models/（也可能带前缀），两种都试
        bad = _resolve_reported_path(project_dir, m.group(1))
        if bad is None:
            return result
        quarantine_dir = project_dir / "_quarantined_yml"
        quarantine_dir.mkdir(exist_ok=True)
        shutil.move(str(bad), str(quarantine_dir / bad.name))
        quarantined += 1
        print(f"[run_dbt] quarantined broken yml: {bad.relative_to(project_dir)}")
    return result
