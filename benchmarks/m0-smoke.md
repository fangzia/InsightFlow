# M0 冒烟记录：官方评测 harness 跑通（Spider 2.0-DBT / DuckDB 子集）

日期：2026-10-10
环境：Windows / Python 3.12 / uv / DuckDB（无 Docker、无云凭证）

## 结果

- 数据：`third_party/Spider2`（shallow clone）+ 官方两个 Google Drive 包
  （DBT_start_db.zip 360MB → examples/，dbt_gold.zip 652MB → evaluation_suite/gold/），经官方 `setup.py` 分发
- 任务：68 个 DuckDB/dbt 任务（`examples/spider2-dbt.jsonl`）
- Gold 提交自评（`uv run python -m insightflow.eval.gold_submission` → 官方 `evaluate.py`）：
  **62 / 64 可评实例 = 96.9%**（4 个实例官方未发布 gold，不计入）

## 官方数据/harness 的已知问题（评测工程素材）

1. **chinook001**：发布的 gold duckdb 只含原始表（customer 等 11 张），
   eval spec 却要求 dbt 模型表（dim_customer / fct_invoice / obt_invoice）——gold 错版，无法自评。
2. **f1001**：gold 中受检表是非物化 VIEW，定义为 `ORDER BY ... LIMIT 20`，
   边界并列导致非确定性。**gold 自评 5 次结果 1,1,0,0,1** ——即使完美答案也 flaky。
   （复现：`uv run python scripts/probe_eval_failures.py`）
3. **xero_new001 / xero_new002 / social_media001**：gold 文件名与 eval spec 不一致，
   gold_submission 构建器已自动规范化（见 WARNING 日志）。
4. **airbnb002 / biketheft001 / google_ads001 / gitcoin001**：官方未发布 gold，无法本地自评。

## 对 M1+ 的设计输入

- 我们的回归评测需在比较前物化视图（`CREATE TABLE AS SELECT ...`）消除执行非确定性；
  匹配器需容忍 LIMIT 边界的并列行。
- 评测协议约定：报告"可评实例"口径（64），并注明上游 flaky 实例。

## 复现命令

```powershell
uv sync --all-groups
# 数据（约 1GB，需代理）：见 .env.example / README
uv run python third_party/Spider2/spider2-dbt/setup.py   # 在 spider2-dbt 目录下
uv run python -m insightflow.eval.gold_submission
uv run python evaluate.py --result_dir gold_submission_smoke --gold_dir gold
# （在 third_party/Spider2/spider2-dbt/evaluation_suite 下）
```
