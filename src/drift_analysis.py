"""Phase 6 (Suchi): does the model hold up over time? Per-band and per-day error for model vs baseline, after the
single final test run. Reads saved test predictions; fits nothing.

Usage: python src/drift_analysis.py   -> results/drift.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from config import INTERIM, RESULTS
from dataset import load

if __name__ == "__main__":
    df, s = load()
    meta = df[df.part == "test"][["job_name", "band", "day", "user"]]
    P = pd.read_parquet(INTERIM / "test_preds_runtime.parquet").merge(meta, on="job_name")
    F = pd.read_parquet(INTERIM / "test_preds_failure.parquet").merge(meta, on="job_name")
    best = json.loads((RESULTS / "test_run_log.json").read_text())["runtime_model"]
    out = {"runtime_by_day": [], "failure_by_day": []}
    for d, g in P.groupby("day"):
        out["runtime_by_day"].append({"day": int(d), "n": len(g),
                                      "mae_model": float(np.abs(g[best] - g.y).mean()),
                                      "mae_baseline": float(np.abs(g.baseline - g.y).mean()),
                                      "mse_model_signed": float((g[best] - g.y).mean()),
                                      "mse_baseline_signed": float((g.baseline - g.y).mean())})
    for d, g in F.groupby("day"):
        ok = 0 < g.y.mean() < 1
        out["failure_by_day"].append({"day": int(d), "n": len(g), "base_rate": float(g.y.mean()),
                                      "mean_pred_model": float(g.prob.mean()), "mean_pred_baseline": float(g.baseline.mean()),
                                      "pr_auc_model": float(average_precision_score(g.y, g.prob)) if ok else None,
                                      "pr_auc_baseline": float(average_precision_score(g.y, g.baseline)) if ok else None})
    # drift in the inputs: share of test jobs whose user/task type was rare in development
    dev = df[(df.part == "dev") & df.y_fail.notna()]
    test = df[(df.part == "test") & df.y_fail.notna()]
    out["input_shift"] = {
        "role_set_dev_%": (dev.role_set.value_counts(normalize=True).head(6) * 100).round(2).to_dict(),
        "role_set_test_%": (test.role_set.value_counts(normalize=True).head(6) * 100).round(2).to_dict(),
        "fail_rate_dev": float(dev.y_fail.mean()), "fail_rate_test": float(test.y_fail.mean()),
        "fail_rate_test_by_band": test.groupby("band").y_fail.mean().round(4).to_dict(),
        "median_logrt_dev": float(df[(df.part == "dev")].y_logrt.median()),
        "median_logrt_test_by_band": df[df.part == "test"].groupby("band").y_logrt.median().round(4).to_dict(),
        "heavy_user_share_test_by_band_%": (test.groupby("band").user.agg(lambda u: (u == dev.user.value_counts().index[0]).mean()) * 100).round(2).to_dict(),
    }
    (RESULTS / "drift.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["input_shift"], indent=1, default=float))
