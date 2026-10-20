# M1 Baseline: LangGraph Agent v0 vs Zero-shot

**Status: complete** — full 64-instance runs for both modes, scored with the
official Spider 2.0-DBT harness.

## Headline result

| Mode | Official score | dbt run success |
|---|---|---|
| `zeroshot` (baseline) | **14/64 = 21.9%** | 28/64 |
| `agent` (v0 single loop) | **15/64 = 23.4%** | 36/64 |

M1 acceptance criterion met: agent v0 exceeds the zero-shot baseline.

## Setup

- Benchmark: Spider 2.0-DBT subset (68 tasks; 64 with usable gold answers, see `m0-smoke.md`)
- Model: `qwen3.8` (Qwen3.6-27B, FP8, vLLM via Xinference gateway), temperature 0.2
- Harness: official `evaluate.py` with `duckdb_match`, unchanged
- Target models: derived from the gold eval spec (`condition_tabs`), applied identically to both modes
- dbt: **core 1.8.10 + dbt-duckdb 1.8.4** — see "Environment forensics" below for why this pin is load-bearing

## Modes (ablation)

| Mode | Schema context | Repair rounds | Description |
|---|---|---|---|
| `zeroshot` | DDL only (no value samples) | 0 (single shot) | Baseline: one prompt, one shot, run or fail |
| `agent` | DDL + 3 sample rows per table | 1 | v0 single loop: generate → execute → repair once on dbt failure |

Both modes share the same LangGraph pipeline (`insightflow.agent.graph`); the
differences are two knobs in `AgentOptions`, keeping the comparison controlled.

## Per-instance breakdown

- Common passes (11): divvy001, f1003, greenhouse001, lever001, mrr001,
  playbook001, qualtrics001, quickbooks002, retail001, shopify002, workday002
- **Agent-only passes (4)** — all rescued by the repair loop:
  google_play001, hive001, marketo001, mrr002
- Zeroshot-only passes (3): quickbooks003, workday001, xero001 — in the agent
  run the first generation happened to fail and the repair round fixed dbt
  but drifted semantics away from gold

Repair-loop dbt effects (agent vs zeroshot): fixed 10 (apple_store001,
asana001, google_play001, hive001, intercom001, marketo001, playbook002,
sap001, shopify001, xero_new002), regressed 2 (jira001, twilio001).
Attempts histogram (agent): 1×25 first-shot success, 2×34 with one repair.

## Analysis: where the missing 78% lives

Error budget for the agent run (64 instances):

1. **Semantic mismatch dominates**: 36 instances pass `dbt run` but only 15
   score — 21 instances execute cleanly yet produce tables whose
   columns/values don't match gold. This is the single largest bucket and is
   exactly what the M2 schema-linking milestone targets (align generated
   output schema to task intent, not just make it run).
2. **dbt execution failures**: 28 instances still fail even with one repair
   round (Jinja/source errors, missing relations, multi-model references).
   The CTE-inlining prompt rule (added in v2) reduced but did not eliminate
   phantom-model references.
3. **Non-determinism noise**: with temperature 0.2 the zeroshot↔agent delta
   (+4 agent-only, −3 zeroshot-only) partly reflects sampling variance. The
   +1 net score is modest; multi-seed averaging is noted as a to-do for M2
   comparisons.

## Environment forensics (v1 → v2 lesson)

The first full run scored only 7/64 for both modes. Root cause was **not** the
agent: dbt-core 1.9+ enforces a `packages.yml` ↔ `dbt_packages/` consistency
check that rejects the pre-vendored packages shipped inside the Spider 2.0-DBT
examples (the official baseline image installs dbt of the 1.8 era, which never
runs `dbt deps`). Pinning `dbt-core>=1.8,<1.9` (plus explicitly, since
dbt-duckdb 1.8.x does not upper-bound dbt-core) doubled the score to 14–15/64
with zero agent changes. Takeaway recorded for the project: validate the
harness environment with a gold self-check (M0) before trusting any baseline.

## How to reproduce

```powershell
uv run python -m insightflow.eval.run_benchmark --mode zeroshot --run-name zeroshot-m1v2
uv run python -m insightflow.eval.run_benchmark --mode agent --run-name agent-m1v2
uv run python scripts\compare_runs.py   # per-instance diff of the two runs
```

Each run writes `runs/<run-name>/submission/` (standard harness layout) and
prints the official score. Per-instance diagnostics land in
`runs/<run-name>/run_records.json`.

## Next (M2)

- Schema linker: close the 21-instance semantic-mismatch bucket
- Multi-seed / temperature-0 comparisons to separate signal from noise
- Failure-driven prompt iteration for the 28 remaining dbt failures
- Materialized snapshots + tie-tolerant matching for the flaky gold views
  discovered in M0 (affects f1001-type instances)
