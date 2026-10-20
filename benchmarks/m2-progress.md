# M2 进展：Schema Linker、Verifier —— 以及一次诚实的倒退

**状态：M2 v2 闭环完成** —— `agent-m2-v2` 全量跑分 **15/64**，与 M1
`agent` 打平，且严格包含 nosize 消融的全部 14 个通过。本文档记录 v1
的**负面结果**、其根因、消融对根因的量化确认，以及 v2 修正的实施
与结果——负面结果部分比之前的冒烟成功更有教学价值。

## 头条结果

| 模式 | 官方得分 | dbt 运行成功 | 语义失败 |
|---|---|---|---|
| `zeroshot`（M1 baseline） | **14/64 = 21.9%** | 28/64 | 14 |
| `agent`（M1 v0 单循环） | **15/64 = 23.4%** | 36/64 | 21 |
| `agent-m2`（契约 + 精确校验） | **8/64 = 12.5%** | 39/64 | ~31 |
| `agent-m2-nosize`（消融，无列数提示） | **14/64 = 21.9%** | 40/64 | ~26 |
| `agent-m2-v2`（超集契约 + 超集校验 + 快照） | **15/64 = 23.4%** | 40/64 | ~25 |

M1 的两项头条指标在 v2 中都保持住了——dbt 成功率并列新高（40/64），
superstore001 保住 1.0，机制 A 的 4 个受害实例全部追回。v1 的 7 分
倒退已完全修复；但 v2 与 M1 `agent` **打平而非胜出**（详见 v2 结果
节的构成分析）。下文解释 v1 倒退的原因；消融实验将倒退**定量归因到
列数提示**（见下文专节）。

## M2 v1 引入的机制

- 两步生成：先生成 SQL 前锁定列**契约**（列名 + 顺序 + 语义）；yml 规格
  在列数与 eval spec 的期望列数（`condition_cols`）一致时被直接采信，
  否则由 LLM 在目标列数约束下设计契约。
- 运行时 Verifier：物化表实际列 vs 契约（精确匹配），再做 join fan-out
  诊断（join 键唯一性检查、重复行采样、差异列自动推导）——正是它把
  superstore001 从 25667 → 10277 → 9994 行两轮修复收敛。
- 静态自引用拦截、修复预算扩到 3 轮。

## 相对 M1 `agent` 的逐实例变化

- 保留（11 个共有通过中的 7 个）：divvy001、f1003、lever001、mrr001、
  playbook001、retail001、shopify002
- **新增通过（1 个）：superstore001** —— yml 可信契约（12 列）+ fan-out
  修复，与冒烟预测完全一致
- **丢掉的共有通过（4 个）：** greenhouse001、qualtrics001、
  quickbooks002、workday002
- **丢掉的 agent 独有通过（4 个）：** google_play001、hive001、
  marketo001、mrr002

## 根因：M0 时对评测器的取证是错的

M2 的整套设计建立在一个从 M0 沿用至今的结论上：*"pred 用全部列与 gold
的 `condition_cols` 比对——pred 多一列即判负"*。重读官方
`compare_pandas_table`（`third_party/Spider2/spider2-dbt/evaluation_suite/
eval_utils.py`，L132–147）后发现完全相反：

```python
gold_cols = gold.iloc[:, condition_cols]   # gold 只取被查列
pred_cols = pred                            # pred 保留全部列

for gold_vec in gold_cols:                  # 每个被查列只需匹配
    if not any(vectors_match(gold_vec, pred_vec) for pred_vec in pred_cols):
        return 0                            # # pred 的"任意"一列（与顺序无关）
```

真实协议是**逐列子集匹配**：

- gold 的每个被查列只需在 pred 的**任意**一列中出现即可（列顺序无关，
  按向量相等匹配，容差 1e-2）；
- **pred 多余的列完全免费**；
- 唯一的硬结构约束是**行数**。

### 对设计的颠覆性影响

在这套语义下，最优策略是**超集**：把所有合理可能的列都输出——多列
不花任何代价，漏一个被查列则全盘皆输。M1 的自由生成恰好无意中做到了
这一点——那些未裁剪的宽表（Fivetran 类数据源上动辄 25–100+ 列）顺带
覆盖了 gold 的被查列，这正是它 15 分的主要来源。而 M2 v1 是**有意
反着做**：

1. **最小化契约**：linker 提示词明确要求列集最小化，且 `condition_cols`
   列数提示把契约钉死在恰好 N 列——**零冗余空间**：组合错一列，该实例
   就无法挽回。
2. **精确匹配校验门**：`actual == expected` 惩罚多余的列，把免费资产
   当成"缺陷"去修复。
3. **遵从性修复**：修复循环随后把这套反转真正执行了——首轮生成的宽表
   （按 M1 证据本可得分）被裁剪到契约。

## 从运行日志确认的两种回退机制

**机制 A —— 修复轮裁掉了能得分的表（最恶劣的一类）。** 首次生成产出了
宽表，verify 按最小契约标记多列为差异，修复轮把它们删了。首轮形态与
M1 得分形态一致、终轮却丢分的实例：

- `workday002`：首轮 11 列（含 `source_relation`、`job_profile_code`、
  `job_title` 等）→ 契约只要求 5 列 → 终表是 5 列版 → 丢分。
  M1 agent 本可通过此实例。
- `google_play001`（M1 agent 独有通过）：首轮 26 列 → 被裁到 13 列契约。
  `qualtrics001` 同理（23 → 16，丢掉了如 `count_surveys_sent_30d`）。

**机制 B —— 契约本身是错的，还被忠实执行。** 契约首轮即满足、dbt 绿、
verify 绿、评测器红。契约本身错了（LLM 设计的组合不对，或被采信的 yml
与 gold 不一致——与 nba001 同属官方规格缺陷一族）。
`greenhouse001`、`quickbooks002`、`marketo001`、`hive001`、`mrr002`
都符合此模式。verifier 是**自洽性**检查；契约错了它就变成盖章机器——
更糟的是，它把一个猜测固化成规格，让修复循环继续强制执行。

一个值得记录的细节：`retail001` **最终 dbt 运行失败却拿了 1.0**——
更早的轮次已把表物化进了工作区 `.duckdb`，而那才是评测器看到的。
"dbt 成功 39/64"因此低估了可用提交数，且最后一轮状态 ≠ 提交质量。

## `condition_cols` 列数提示：既是披露问题，也是 bug

把 `condition_cols` 的长度当期望列数，此前已被标记为灰色地带（评测
spec 的结构元数据；真实部署的 agent 不该有它）。这次重新取证给出了
不同性质的结论：该提示不只是有争议——在真实匹配协议下它是**主动
有害的**，因为钉死列数抹掉了 M1 一直在隐性受益的超集冗余。

**`agent-m2-nosize` 消融将其量化为 −6 分**（其余配置与 `agent-m2`
完全一致，仅不传期望列数）：

- 14/64 vs 8/64，**完整保留了 agent-m2 的全部 8 个通过**（含
  superstore001）；
- **追回 5 个被列数提示毁掉的实例**：greenhouse001、workday002、
  marketo001、quickbooks002、qualtrics001；
- **新增 1 个 M1 从未通过的实例**：maturity001；
- 松开钉死后，linker 自然设计出宽表契约（marketo 35 列、sap 172 列、
  quickbooks 28 列）——"最小化"提示词压不住 LLM，**恰好是列数钉死
  这个硬约束才是主凶**；
- dbt 成功率同步新高（40/64）。

仍相对 M1 `agent` 丢 3 个（google_play001、hive001、mrr002），属
机制 B（契约组合错误被忠实执行，如 mrr002 的 LLM 12 列契约首轮
满足、评测仍红）。移除该提示同时满足方法论洁净（生成过程不再消费
eval-spec 元数据）与机械正确两个标准。由 `condition_tabs` 推导目标
模型的做法保留——已在本文与 `m1-baseline.md` 中披露。

消融还贡献了 v2 第 2 项的直接证据：`reddit001` 首轮 17 列**全部相同、
仅顺序不同**，verify 按精确匹配判失败，烧掉一整轮修复把顺序调对——
而评测器与列顺序无关。这轮预算纯属浪费。

## M2 v1 中真正有价值的部分

- **fan-out 诊断**与评测协议无关（行数是评测器唯一钉死的结构量），
  且被证明能救回粒度错误——nosize 冒烟中它独自扛住了 superstore001。
- yml 可信契约作为**最小覆盖**（而非精确规格）仍有价值，前提是 yml
  忠实（superstore001、mrr001）。
- dbt 成功率 39/64（新高），得益于静态自引用拦截与更丰富的修复反馈。
- 负面结果本身：**agent 架构必须由评测协议的真实代码驱动，而不是对
  它的转述**。M0 的结论从未对照源码复核；本文即勘误。

## M2 v2 修正方案（已全部实施）

消融已经证明第 1 项的大半收益（去掉钉死即从 8 回到 14）；v2 的目标
是把剩余缺口（相对 M1 `agent` 的 −3/+2）也吃掉。四项修正全部落地：

1. **超集契约**：yml 变为*最小*列覆盖；LLM 提示词从"最小化"反转为
   "最大化合理覆盖——多余的列是免费的"（程序化兜底：LLM 审阅删掉的
   草稿列强制补回）。彻底移除精确列数钉死（消融量化：该项单独值
   6 个实例；v2 模式不消费 `condition_cols`）。
2. **超集校验门**：检查 `expected ⊆ actual` 而非 `actual == expected`；
   只有缺失列才触发修复，修复反馈只引导补列、绝不引导裁剪；列顺序
   不再算失败（`reddit001` 已展示顺序检查白烧一轮修复预算）。
3. **fan-out 证据式措辞**：保留全部诊断机制（join 键唯一性、重复行
   采样、差异列推导），但反馈从指令式改为证据呈现——重复键组 +
   样本 + 差异列 + 契约声明的粒度，由修复轮 LLM 裁决一对多是否
   任务本意。
4. **最优轮快照**：新增 `best_generated` 状态与 `restore_best` 节点——
   每个"dbt 成功且列门通过"的轮次被快照；预算耗尽仍未收敛时写回
   最优轮、重跑一次 dbt 再结束（针对机制 A 的保险；retail001 证明
   工作区状态的确跨轮存活）。

## M2 v2 结果

**15/64 = 23.4%**，dbt 成功 40/64（并列新高）。构成分析：

- **严格包含 `agent-m2-nosize` 的全部 14 个通过**，并追回
  `quickbooks003`（此前仅 zeroshot 通过过、agent 与 v1 系均丢）。
- 机制 A 的受害实例全部追回：workday002、greenhouse001、
  quickbooks002、qualtrics001、marketo001（消融已追回，v2 保持）。
- 预期的诚实排序 `v2 > agent > zeroshot` **只实现了一半**：v2 与
  M1 `agent` 打平（15 = 15），通过集并不重合：

  - 共有通过 12 个；
  - `agent` 独有（v2 仍丢的 3 个）：google_play001、hive001、
    mrr002——运行记录显示三者 dbt 绿、verify **首轮即通过**、零
    修复轮，即机制 B 的纯粹形态：契约组合本身错了，还被忠实
    执行。超集语义救不了"错误地自洽"——这是契约式两步生成的
    剩余税。
  - v2 独有 3 个：superstore001（fan-out 修复三轮收敛）、
    maturity001、quickbooks003。

- 冒烟注记：superstore001 首次冒烟失败过一次——LLM 把代理键 `id`
  生成成源表 `row_id` 而非项目惯例 `100 + ROW_NUMBER()`；重跑即恢复
  1.0。属轨迹漂移而非 v2 系统性缺陷，也是单 seed 噪声的又一直接证据。
- **最优轮快照 `restore_best` 全量 0 次触发**——预算耗尽的实例要么
  从未有过列门通过的轮次（无快照可恢复），要么最后一轮已是最优。
  保险未付出代价、也未见收益；保留它是因为 retail001 一族
  （"最终 dbt 失败但 .duckdb 保留了早期得分形态"）的教训仍然成立。
- 修复轮使用分布：attempts=1（首轮 verify 即收敛）21 个——超集门
  把 v1 大量"多余列/顺序"假阳性修复轮直接省掉了；4 轮耗尽 22 个。

## 局限与披露

- 单 seed、temperature 0.2 —— −7 的倒退远超 M1 两模式间观察到的
  ±2–3 噪声带，倒退归因的结论稳健。但 v2 的**胜出**主张不成立：
  与 `agent` 的 15 = 15 在单 seed 噪声带之内，且 v2 冒烟中
  superstore001 还出现过一次轨迹漂移失分。多 seed 之前，诚实的表述
  是"v2 ≥ nosize（严格超集）且追平 agent"，而非"v2 > agent"。
- `nba001` 仍是官方数据缺陷（yml 声明 10 列 vs gold 12 列；
  `condition_cols [0,3,4,…,11]` 因列数巧合掩盖了不一致）。
- `condition_tabs`（目标模型推导）是 harness 消费的评测 spec 元数据，
  全部模式一致使用并已在本文与 `m1-baseline.md` 中披露；
  `condition_cols` 长度（期望列数）已在 v2 中移除——v2 模式不再消费
  任何 eval spec 元数据。
- 68 个基准实例中有 4 个无可用 gold，各处均排除（分母 64，见
  `m0-smoke.md`）。
- v1 的 `agent-m2` / `agent-m2-nosize` 模式名保留仅为历史命令兼容；
  复现 8/64 与 14/64 需 checkout 对应历史版本的代码。

## 如何复现

```powershell
uv run python -m insightflow.eval.run_benchmark --mode zeroshot --run-name zeroshot-m1v2
uv run python -m insightflow.eval.run_benchmark --mode agent      --run-name agent-m1v2
uv run python -m insightflow.eval.run_benchmark --mode agent-m2   --run-name agent-m2-full
uv run python -m insightflow.eval.run_benchmark --mode agent-m2-nosize --run-name agent-m2-nosize-full
uv run python -m insightflow.eval.run_benchmark --mode agent-m2-v2 --run-name agent-m2-v2-full
uv run python scripts\diff_pred_gold.py <instance_id> <submission_dir>   # 逐表/逐列差异
```

控制台日志：`runs/agent-m2-full-console.log`、
`runs/agent-m2-nosize-full-console.log`、
`runs/agent-m2-v2-full-console.log`；逐实例记录：
`runs/<run-name>/run_records.json`。