"""One-off probe: why do chinook001 / f1001 gold submissions fail duckdb_match?"""

import json
from pathlib import Path

import duckdb

BASE = Path(__file__).parents[1] / "third_party" / "Spider2" / "spider2-dbt" / "evaluation_suite"
GOLD = BASE / "gold"
PRED = BASE / "gold_submission_smoke"


def _vec_match(v1, v2, ignore_order_=False, tol=1e-2):
    """Replica of the official vectors_match logic (compare per-column vectors)."""
    import math

    import pandas as pd

    try:
        if ignore_order_:
            v1, v2 = (
                sorted(v1, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))),
                sorted(v2, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))),
            )
        if len(v1) != len(v2):
            return False
        for a, b in zip(v1, v2, strict=False):
            if pd.isna(a) and pd.isna(b):
                continue
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if not math.isclose(float(a), float(b), abs_tol=tol):
                    return False
            elif a != b:
                return False
        return True
    except Exception:
        return False


def table_info(db_path: Path, table: str) -> tuple[int, int]:
    con = duckdb.connect(database=str(db_path), read_only=True)
    try:
        df = con.execute(f"SELECT * FROM {table}").fetchdf()  # noqa: S608
        return len(df.index), len(df.columns)
    finally:
        con.close()


def main() -> None:
    eval_jsonl = GOLD / "spider2_eval.jsonl"
    specs = {
        json.loads(line)["instance_id"]: json.loads(line)["evaluation"]
        for line in eval_jsonl.read_text().splitlines()
        if line.strip()
    }

    for instance in ["chinook001", "f1001"]:
        spec = specs[instance]["parameters"]
        print(f"\n=== {instance} (gold={spec['gold']}) ===")
        gold_db = GOLD / instance / spec["gold"]
        con = duckdb.connect(database=str(gold_db), read_only=True)
        available = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        con.close()
        print(f"  available tables: {available}")
        for tab, cols in zip(spec["condition_tabs"], spec["condition_cols"], strict=False):
            if tab not in available:
                print(f"  {tab}: MISSING from gold db (condition indices up to {max(cols)})")
                continue
            rows, ncols = table_info(gold_db, tab)
            status = "OK" if max(cols) < ncols else f"OUT OF RANGE (needs {max(cols) + 1} cols, has {ncols})"
            print(f"  {tab}: rows={rows}, cols={ncols}, max condition_idx={max(cols)} -> {status}")

    # --- f1001: locate the exact failing table in compare_pandas_table ---
    import sys

    sys.path.insert(0, str(BASE))
    import pandas as pd  # noqa: F401
    from eval_utils import compare_pandas_table  # type: ignore  # noqa: E402

    def load_df(db: Path, table: str):
        con = duckdb.connect(database=str(db), read_only=True)
        df = con.execute(f'SELECT * FROM "{table}"').fetchdf()  # noqa: S608
        con.close()
        return df

    spec = specs["f1001"]["parameters"]
    gold_db = GOLD / "f1001" / spec["gold"]
    pred_db = PRED / "f1001" / spec["gold"]
    print("\n=== f1001 per-table compare ===")
    for tab, cols, ig in zip(spec["condition_tabs"], spec["condition_cols"], spec["ignore_orders"], strict=False):
        gold_df = load_df(gold_db, tab)
        pred_df = load_df(pred_db, tab)
        ok = compare_pandas_table(pred_df, gold_df, condition_cols=cols, ignore_order=ig)
        print(f"  {tab}: score={ok}, gold dtypes={[str(t) for t in gold_df.dtypes.tolist()]}")
        if not ok:
            gold_sel = gold_df.iloc[:, cols]
            for ci, gold_col in enumerate(gold_sel.columns):
                gold_vec = gold_sel[gold_col].tolist()
                pred_cols_list = pred_df.transpose().values.tolist()
                if not any(
                    _vec_match(gold_vec, pv, ig) for pv in pred_cols_list
                ):
                    print(f"    unmatched gold col #{ci} ({gold_col}): sample={gold_vec[:5]}")


if __name__ == "__main__":
    main()
