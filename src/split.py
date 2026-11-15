"""Phase 3: the frozen time split, walk-forward folds and reporting bands. Run once; results/split.json is the contract.

Usage: python src/split.py          (writes results/split.json; refuses to overwrite a frozen split unless --force)

Rule set (fixed 2026-11-04, before any model was fit):
- CUT = 70th percentile of submit_time over all jobs with a submit time (job-count based, as in the proposal).
- GAP = 24 h embargo. Jobs submitted in [CUT, CUT + GAP) are used by no model or metric.
- TEST = jobs submitted at or after CUT + GAP.
- DEVELOPMENT (train) = jobs submitted before CUT whose outcome was known before CUT + GAP
  (a model deployed at CUT + GAP could only have learned from outcomes known by then).
- Walk-forward folds inside development: fold k fits on jobs submitted before c_k with outcome known before
  c_k + GAP, validates on jobs submitted in [c_k + GAP, c_{k+1}), with c_4 = CUT.
- Test bands: the test period split into 4 bands of equal duration.
"""
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import INTERIM, RESULTS

DAY = 86400
GAP = DAY
FOLD_CUT_DAYS = [34, 41, 48]   # fold cut points (days since first submit); validation windows are ~6 days each
N_BANDS = 4


def compute(df: pd.DataFrame) -> dict:
    t0 = float(df.submit_time.min())
    cut = float(np.quantile(df.submit_time, 0.70))
    end = float(df.submit_time.max())
    cuts = [t0 + d * DAY for d in FOLD_CUT_DAYS] + [cut]
    folds = [{"fold": k + 1, "fit_submit_before": cuts[k], "fit_known_before": cuts[k] + GAP,
              "val_from": cuts[k] + GAP, "val_to": cuts[k + 1]} for k in range(len(FOLD_CUT_DAYS))]
    test_from = cut + GAP
    edges = list(np.linspace(test_from, end + 1, N_BANDS + 1))
    return {"frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "t0": t0, "cut": cut, "gap_s": GAP, "test_from": test_from, "trace_end": end,
            "cut_day": (cut - t0) / DAY, "test_from_day": (test_from - t0) / DAY, "trace_end_day": (end - t0) / DAY,
            "folds": folds, "band_edges": edges,
            "band_edges_day": [(e - t0) / DAY for e in edges]}


def load_split() -> dict:
    return json.loads((RESULTS / "split.json").read_text())


def assign(df: pd.DataFrame, s: dict) -> pd.DataFrame:
    """Add columns: part in {dev, embargo, test, late_label}, band (1..4 for test), fold_val (1..3 for dev rows in a val window)."""
    df = df.copy()
    st, kn = df.submit_time.to_numpy(), df.known_time.to_numpy()
    part = np.full(len(df), "dev", dtype=object)
    part[st >= s["cut"]] = "embargo"
    part[st >= s["test_from"]] = "test"
    part[(st < s["cut"]) & (kn >= s["cut"] + s["gap_s"]) & df.y_fail.notna().to_numpy()] = "late_label"
    df["part"] = part
    df["band"] = np.where(part == "test", np.searchsorted(s["band_edges"], st, side="right"), 0)
    fv = np.zeros(len(df), int)
    for f in s["folds"]:
        fv[(st >= f["val_from"]) & (st < f["val_to"])] = f["fold"]
    df["fold_val"] = np.where(part == "dev", fv, 0)
    return df


def fold_masks(df: pd.DataFrame, f: dict):
    """Fit and validation masks for one walk-forward fold (dev rows only)."""
    dev = df.part == "dev"
    fit = dev & (df.submit_time < f["fit_submit_before"]) & (df.known_time < f["fit_known_before"])
    val = dev & (df.submit_time >= f["val_from"]) & (df.submit_time < f["val_to"])
    return fit, val


if __name__ == "__main__":
    out = RESULTS / "split.json"
    if out.exists() and "--force" not in sys.argv:
        print("split.json already frozen; not overwriting. Use --force only with a DECISIONS.md entry.")
        s = load_split()
    else:
        df = pd.read_parquet(INTERIM / "jobs.parquet", columns=["submit_time", "known_time", "y_fail", "y_logrt", "user", "status"])
        s = compute(df)
        df = assign(df, s)
        lab = df.y_fail.notna()
        s["counts"] = {p: {"jobs": int((df.part == p).sum()), "failure_labelled": int(((df.part == p) & lab).sum()),
                           "runtime_labelled": int(((df.part == p) & df.y_logrt.notna()).sum())}
                       for p in ["dev", "late_label", "embargo", "test"]}
        s["late_label_runtime_dropped"] = int(((df.part == "late_label") & df.y_logrt.notna()).sum())
        s["test_band_counts"] = {int(b): int(n) for b, n in df[(df.part == "test") & lab].band.value_counts().sort_index().items()}
        s["fold_counts"] = []
        for f in s["folds"]:
            fit, val = fold_masks(df, f)
            s["fold_counts"].append({"fold": f["fold"], "fit_failure_labelled": int((fit & lab).sum()),
                                     "val_failure_labelled": int((val & lab).sum()),
                                     "val_runtime_labelled": int((val & df.y_logrt.notna()).sum())})
        dev_users = set(df.loc[(df.part == "dev") & lab, "user"])
        tl = df[(df.part == "test") & lab]
        s["test_unseen_user_share_%"] = round(100 * float((~tl.user.isin(dev_users)).mean()), 2)
        out.write_text(json.dumps(s, indent=1))
    print(json.dumps({k: v for k, v in s.items() if k not in ("folds", "band_edges")}, indent=1))
