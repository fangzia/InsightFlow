# InsightFlow

**Spider2-DBT 上的 graph-based dbt agent —— 一套度量驱动的 LLM agent 优化方法论。**

在业界最难的企业级 text-to-workflow 基准（Spider2-DBT，64 任务，官方榜单
SOTA 75% / claude-opus-5）上，用内网 qwen3.8 单模型构建 LangGraph agent，
做到 **23.4%（15/64，≈ 官方 Spider-Agent 换用 o1 级模型的分数段）**、
**dbt 编译成功 55/64**。项目的核心价值不在分数，而在方法论：**每个机制
一个消融开关、每轮改动先冒烟再全量、负面结果与正面结果同权重记录**——
最终用 12+ 个机制的受控实验钉死了"graph 优化为什么不该继续做、真正
的瓶颈在哪"。

## 指标一览

| 指标 | M1 | M2 v2 | M3（两轮全量） |
|---|---|---|---|
| 官方得分 | 15/64 | 15/64 | 15 / 14（±1 方差） |
| dbt 编译成功 | 36/64 | 40/64 | **55 / 54** |
| 消融开关数 | 3 | 6 | **16** |

15=15 的平局账本（诚实归因）：+2 机械修复正当收益、−2 机械修复副作用
（覆盖"碰巧正确的旧物化状态"）、±2-3 单 seed 方差——dbt 层 +15 是
不依赖 LLM 的确定性收益。

## 架构（M3 终形态）

```
START → prepare → link ⇄ generate → execute → verify ─┬→(verified)→ column_review → critic ─┬→(approve)→ END
           ↑        ↑    (lint 静态拦截)                 │        (双版本对照)   (统计证据) └→(challenge)→ link
           │        │                                    ├→(有进展 & 预算内)→ generate 定向修复
           │        └←──(replan：无进展 → 重设计)────────┤
           └──────────────(critic challenge)            └→(耗尽)→ restore_best / select_best → END

   预实现模型 dbt 失败 → 纳入图修复（池 0 救援）
   预算耗尽后 → 3 条独立候选轨迹 → 程序化选择器（超集语义完全序）
```

生成质量之外的六类官方 shipped 项目缺陷（缺包/破包/坏 yml/坏模型/
ref 指裸表/祖先重建）全部由门控机械修复处理——这是 dbt 层 +15 的来源。

## 这个项目展示了什么

1. **消融方法论**：16 个 `AgentOptions` 开关（机械修复/闭包/replan/
   多候选/critic 两代/行数锚定/粒度锚定/单列重审），基线字节级可复现。
2. **协议逆向与认知翻案**：重读官方评测器源码发现沿用结论是错的
   （逐列 OR-any 匹配，非精确匹配），用消融把 7 分倒退定量归因到单一
   原因再修复（M2）。
3. **回归取证**：`+select` 重建祖先导致代理键值域错位（superstore001
   1.0→0），对比四方数据逐列定位，修复为"只补缺失闭包"。
4. **负面结果的系统性产出**：43 个"dbt 绿 × eval 红"的逐列 autopsy
   （49% 行数错 / 33% 单列错）+ 图内信号穷举实验，证明值层正确性在
   没有 gold 的图内不可判定——并用 leaderboard 证据把剩余瓶颈定位到
   生成模型层（同方法换模型 13%→40%）。

## 快速开始

```powershell
# 安装
uv sync

# 配置（OpenAI 兼容端点）
copy .env.example .env   # 填 INSIGHTFLOW_OPENAI_BASE_URL / API_KEY / DEFAULT_MODEL

# 单元测试（50 个，含全部机制的回归锚定）
uv run pytest tests -q

# 冒烟
uv run python -m insightflow.eval.run_benchmark --mode agent-m3-grain --instances f1001,superstore001

# 全量（64 实例，约 4-5 小时）
uv run python -m insightflow.eval.run_benchmark --mode agent-m3-grain --run-name my-run
```

模式阶梯（逐级叠加开关，全部可复现）：`zeroshot` → `agent` →
`agent-m2-v2` → `agent-m3` → `agent-m3-replan` → `agent-m3-fanout` →
`agent-m3-critic` → `agent-m3-critic-v2` → `agent-m3-anchor` → `agent-m3-grain`。

## 文档

- [`benchmarks/m0-baseline.md`](benchmarks/m0-baseline.md) —— 基线与评测协议取证
- [`benchmarks/m1-baseline.md`](benchmarks/m1-baseline.md) —— agentic 循环与错误反馈
- [`benchmarks/m2-progress.md`](benchmarks/m2-progress.md) —— 契约/校验、一次诚实的倒退与消融归因
- [`benchmarks/m3-graph-optimization.md`](benchmarks/m3-graph-optimization.md) —— graph 优化、值层信号边界测定、失败池 autopsy

## 结构

```
src/insightflow/
├── agent/          # LangGraph 状态机：link/generate/execute/verify/repair/
│   │               # replan/candidates/critic/column_review + 提示词
├── connectors/     # DuckDB + dbt 执行器（机械修复家族、错误提炼、闭包）
├── eval/           # 基准 harness（模式矩阵、run_records、逐实例归因）
└── core/           # LLM 客户端与配置
tests/              # 50 个单元测试（含 v2 基线回归锚定）
benchmarks/         # 里程碑文档（M0-M3）
```
