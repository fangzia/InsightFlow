# M3：Graph 结构优化与值层信号边界测定

**状态：M3 完成** —— 四步 graph 演进 + 三类值层信号实验全部落地并消融。
全量结果：**eval 15/64 与 M2 v2 打平，dbt 成功 40/64 → 55/64（+15）**。
本文档记录机制清单、逐机制的实测贡献（含大量**负面结果**）、失败池
autopsy，以及最终用实验钉死的结论：**值层正确性在没有 gold 的图内
不存在可判定信号**——剩余瓶颈在生成模型，不在 graph。

## 头条结果

| 模式 | 官方得分 | dbt 运行成功 | 说明 |
|---|---|---|---|
| `agent-m2-v2`（M2 基线） | **15/64 = 23.4%** | 40/64 | 起点 |
| `agent-m3-critic-v2`（全机制） | **15/64 = 23.4%** | **55/64** | 全量验证 |
| `agent-m3-grain`（终形态） | **14/64 = 21.9%** | 54/64 | 与 critic-v2 唯一 eval 差异 = quickbooks003 单 seed 漂移（v2 已知脆弱实例），机制零回归 |

dbt 层 +15 是确定性收益（机械修复，不依赖 LLM）；eval 层平局的账本：
+2 机械修复正当收益（workday001、recharge002）、−2 机械修复副作用
（f1003、retail001，见"覆盖旧状态"一节）、±2-3 单 seed 方差
（mrr002/google_play002 冒烟翻正全量翻负、superstore001 保住）。

外部对照：Spider2-DBT 榜单 SOTA = 75%（48/64，SignalPilot，claude-opus-5）；
GPT-5 系开源方法约 40% 档；官方 Spider-Agent 换用 o1 级模型约 23-26%。
本项目用 qwen3.8 单模型做到 23.4% —— 约等于别家换一代旗舰模型的分数段。

## 起点：v2 graph 的三个结构性盲区

```
START → prepare → link → generate → execute → verify ─┬→ END
                        ↑__________________|          │
                          (预算内定向修补)          └→ restore_best → END
```

1. **修复循环只有"定向修补"一种模式**——22 个实例耗满 3 轮预算仍失败，
   反馈信号是 dbt stdout 尾部 4000 字符原始日志；
2. **单路径生成**——无候选竞争，无重设计通道；
3. **验证闭环自我强化**——契约由 link 设计、verify 用契约校验、修复轮
   执行契约，闭环内没有任何环节消费"任务真正要什么"。机制 B
   （google_play001/hive001/mrr002：dbt 绿、verify 首轮通过、零修复、eval 0）
   即此形态。

## 四步演进与冒烟矩阵

| 步骤 | 机制 | 冒烟结论 |
|---|---|---|
| 1 | 错误提炼（triage）+ ref 静态 lint | 诊断工具；真正收益来自顺带修复的六类 shipped 缺陷 |
| 2 | 无进展检测 → replan（限 1 次） | 机制验证通过（6+ 次触发全正确），但残余池里设计层错误罕见——无翻盘 |
| 3 | 延迟多候选（预算耗尽后 3 条独立轨迹）+ 程序化选择器 | 机制验证通过（superstore001 回退轨迹被候选救回），候选自身值漂移——无 eval 增量 |
| 4 | critic（verify 全绿后独立审查物化数据，限 1 次） | 安全但保守：v1 采样 3 行 → 10/10 APPROVE，机制 B 一个没抓到 |

值层追加实验：

| 信号 | 设计 | 实测 |
|---|---|---|
| critic v2 统计证据 | NULL 率/distinct/值域/源表对账 | 全量 52/52 APPROVE 零挑战——"值域内错误指派"统计上不可见 |
| 行数锚定（显式数字） | "top 10"→10 行 | **覆盖率 1/64**——Spider2-dbt 指令是工程描述，无 NL2SQL 式显式数 |
| 粒度词锚定 | "monthly per X"→行数 vs distinct(时间)×distinct(实体，源表基数) | 触发率高（6/6），但期望基数三误差源（源表 distinct≠gold、实体列定位错、输出表 distinct≤行数）把修复**带偏**（analytics_engineering001 从 55 行拉向 28，gold 是 103）——净贡献 ≈0 |
| 单列重审（双版本对照） | verify-green 后高温重掷探针，逐列 multiset 对照，弱列定向重写 | 零误伤（控制组全保持）；但独立重掷**一致地复现同样的错误口径**——一致性信号对"一致地错"失明 |

## 机械修复家族（dbt 层 +15 的来源）

六类官方 shipped 项目缺陷，全部门控、可消融：

1. **缺 hub 包**（dbt_deps 缺 `dbt_utils` 等 4 实例）→ needs_deps 扩展
2. **破损 vendored 包**（chinook001）→ 重装
3. **坏 yml 隔离**（flicks/tickit/google_play002）→ 坏 yml 移出编译路径，保住 4 实例 dbt 绿 + google_play002 冒烟净新增 1 分
4. **坏模型剪除**（netflix/app_reporting/provider/activity）→ 阻塞 parse 的模型移出编译路径（含包内路径）
5. **`ref()` 指裸表三形态**（f1001 系模型内/recharge 系 dbt_project.yml vars/包内）→ 预检改写
6. **`+select` 重建已物化祖先**（原方案缺陷）→ 改为只补缺失闭包——superstore001 的 FK 代理键因此保住（gold 的 `dim_products_id` 依赖已有物化状态）

两个非显然的工程细节：

- **回归取证**：`+select` 最初版重建了 dims 表，代理键被重新计算与
  shipped `dim_products.id` 值域错位 → superstore001 从 1.0 回归 0。
  通过对比"新旧模型 SQL、物化表、db 源表、eval spec"四方数据逐列
  定位。修复 = 闭包语义（只补缺的祖先）。
- **快照择优按严重度**（非计数）：superstore001 第 4 轮的部分回退
  不再覆盖第 3 轮最优——这保住了它。

## 全量换血四例（15=15 的构成）

- **新增 workday001、recharge002**：机械修复跑通 dbt 后首次得分——正当收益
- **丢失 f1003、retail001**：v2 里它们靠"早期成功轮物化的旧表 + final dbt
  失败不动它"得分（m2-progress 记录过的负负得正）；m3 机械修复跑通了
  dbt，新 SQL 的表**覆盖**了碰巧正确的旧表——机械修复的副作用
- **方差互换**：mrr002、google_play002（冒烟翻正、全量翻负）、
  superstore001（保住）、quickbooks003（critic-v2 轮通过、grain 终形态
  轮翻负——两轮全量间唯一的 eval 差异，再次确认其为单 seed 脆弱实例）

`agent-m3-grain` 终形态全量（14/64，dbt 54/64）与 `agent-m3-critic-v2`
（15/64，dbt 55/64）的逐实例对比：通过集完全一致除 quickbooks003、
dbt 集差 1（inzight001 波动）——**两类终形态机制等价，分差全部落在
±1 方差带内**。粒度锚定/单列重审在全量上既无正贡献也无回归，
与冒烟阶段"净贡献 ≈0"的归因一致。

## 43 个 dbt-green-eval-red 的 autopsy（全量 m3-critic-v2）

用官方评测函数（`sys.path` 注入 evaluation_suite，非手写复刻——见
方法论教训）逐实例取证，按官方失败语义分类：

| 失败类别 | 数量 | 典型 | 含义 |
|---|---|---|---|
| 行数不匹配 | 21（49%） | reddit001 ±1 行；zuora001 6 vs 334 | 评测器唯一硬约束直接判负 |
| ├ 接近型 | ~8 | shopify001 ±6、f1003 ±1 | SQL 99% 对，差一个 WHERE 边界/去重 |
| └ 粒度型 | ~13 | zuora/sap/apple_store | 聚合粒度错一个量级 |
| 少数列错 | 14（33%） | **~7 个只差一列**（f1001: 30 列对 1 列错） | 单一 join 口径/名称拼接错 |
| 表缺失等 | 8 | chinook/tickit 系 | 复刻读取问题，官方判 0 原因待逐个确认 |

结论：**49% 是行数（评测器唯一硬约束）、33% 是单列级语义口径**——
两类都不是 graph 结构问题，是生成质量问题。

## 负面结果专节：值层信号穷举

系统性地尝试了图内可得的一切信号源，每个都量化了死因：

| 信号 | 死因 |
|---|---|
| 契约自洽（verify） | 闭环内——契约错即盖章（机制 B） |
| fan-out / join 键诊断 | 只见结构膨胀，不知任务要几行 |
| 统计证据（critic v2） | "值域内错误指派、看似合理的错误聚合"统计上不可见 |
| LLM 采样裁决（critic v1） | 3 行采样无信息量 |
| 指令显式数字锚定 | 覆盖率 1/64——指令风格假设不成立 |
| 粒度词锚定 | 期望基数的三个估计源全部有偏差，反馈方向错误 |
| 双版本一致性 | 独立重掷复现同样的错误口径——只抓随机不稳，不抓一致错 |

**边界结论**：在没有 gold 的图内，值层正确性不可判定。这不是实现
问题，是信息论边界——每个信号都测到了上限。

## 方法论教训

1. **本地评测必须直接调用官方函数**。手写复刻 `fetchall()` 版 compare
   把 f1001 判成全对（官方 `fetchdf()` 的 pandas 类型语义：NaN/None、
   numpy 数值、Decimal 精确比较）。假阴性/假阳性都出现过。
2. **"跑通"不总是收益**：机械修复覆盖旧状态导致 −2（f1003/retail001
   家族）。修复类改动要考虑与"碰巧正确的现状"的交互。
3. **单 seed 噪声带 ±2-3**：任何 ≤3 分的主张需要多 seed 或机制级
   归因支撑（v2 即确立，M3 再次验证——方差互换贡献了账本的一大块）。
4. **duckdb 的 `DROP TABLE/VIEW IF EXISTS` 对类型不匹配照样报错**：
   恢复/重建物化对象需按 `information_schema` 类型精确 drop。

## 机制开关总表（`AgentOptions`）

| 开关 | 机制 | 全量实测触发 |
|---|---|---|
| `use_lint` | ref 存在性静态检查 | 活跃 |
| `use_triage` | dbt 错误结构化提炼 | 活跃 |
| `use_deps` | 缺/破包修复 | 4+ 实例 |
| `use_ancestors` | 缺失祖先闭包（不重建已有） | 活跃 |
| `use_pool0_rescue` | 预实现模型 dbt 失败时纳入图 | 2 实例 |
| `use_yml_quarantine` | 坏 yml 隔离 | 5 实例 |
| `use_prune_blockers` | 坏模型剪除 | 10 实例 |
| `use_ref_repair` | ref-指-裸表三形态改写 | 12 实例 |
| `replan_limit` | 无进展 → 重设计 | 6 次 |
| `n_candidates` | 延迟多候选 + 选择器 | 16 实例触发，select_best 兜底 1 次 |
| `use_critic` / `use_critic_v2` | 结果审查（采样/统计证据） | 52 次裁决全 APPROVE |
| `use_anchor` | 行数锚定 + 重复行检测 | 锚定 1/64；重复行检测活跃 |
| `use_column_review` | 双版本单列重审 | 触发正常，一致性盲区 |
| `use_grain_anchor` | 粒度词锚定 | 触发率高，期望基数带偏 |

所有开关默认关闭；`agent-m2-v2` 行为字节级保持（回归测试锚定）。

## 结论与下一步

1. **dbt 层已到顶**：55/64，剩余 9 个是数据不可达（tpch 家族）或
   LLM 输出质量（activity001 类）。
2. **eval 层的钥匙不在 graph**：四步结构优化 + 三类值层信号全部
   量化了上限。剩余 43 个失败 = 49% 行数错 + 33% 少数列错 +
   LLM 质量墙，全部指向单次生成质量分布。
3. **最大杠杆是模型接入**：leaderboard 证据——Spider-Agent 同方法
   换模型 12.98%（GPT-4o）→ 23.58%（o1）→ 25.78%（Claude-4-Sonnet）→
   ~40%（GPT-5 系）；SOTA 75% 底座是 claude-opus-5。当前网关仅
   qwen3.8 可用。M3 基础设施（管道全通 + 消融开关）已为模型升级
   准备完毕。
4. qwen3.8 约束下的现实路径：pass@3 方差收割（mrr002/google_play002/
   f1001 的翻转证明方差分存在）+ f1003/retail001 保护（预计 +2），
   目标区间 25-30%。

## 复现

```powershell
# 全量（任一模式，约 4-5 小时）
uv run python -m insightflow.eval.run_benchmark --mode agent-m3-grain --run-name <name>
# 目标实例冒烟
uv run python -m insightflow.eval.run_benchmark --mode agent-m3-critic --instances f1001,google_play002
# 机制单元测试（50 个）
uv run pytest tests -q
```

逐实例记录：`runs/<run-name>/run_records.json`；冒烟日志：
`runs/m3-*.log`。模式定义：`src/insightflow/eval/run_benchmark.py`
（`agent-m3` → `agent-m3-critic` → `agent-m3-anchor` → `agent-m3-grain`
逐级叠加开关）。
