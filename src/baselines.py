"""Phase 4: baselines (no learning beyond a global median/rate), scored on the walk-forward folds.

Runtime:  global median of the fit set; per-user past median (finished-before-submit), global median fallback.
Failure:  global rate of the fit set; per-user past failure rate, global fallback; and a smoothed per-user rate
          (k pseudo-jobs at the global rate) chosen by fold score if it is stronger.

Usage: python src/baselines.py   -> results/baselines_cv.json
"""
import json

import numpy as np

from config import RESULTS
from dataset import folds, load
from metrics import failure_metrics, runtime_metrics

SMOOTH_K = 10  # pseudo-counts for the smoothed failure baseline (fixed, not tuned per fold)


def runtime_global(fit, rows):
    return np.full(len(rows), fit.y_logrt.median())


def runtime_user(fit, rows):
    return rows.user_median_logrt.fillna(fit.y_logrt.median()).to_numpy()


def failure_global(fit, rows):
    return np.full(len(rows), fit.y_fail.mean())


def failure_user(fit, rows):
    return rows.user_fail_rate.fillna(fit.y_fail.mean()).to_numpy()


def failure_user_smoothed(fit, rows, k=SMOOTH_K):
    g = fit.y_fail.mean()
    n = rows.user_n_finished.fillna(0).to_numpy()
    r = rows.user_fail_rate.fillna(g).to_numpy()
    return (r * n + g * k) / (n + k)


RUNTIME = {"global_median": runtime_global, "user_median": runtime_user}
FAILURE = {"global_rate": failure_global, "user_rate": failure_user, "user_rate_smoothed": failure_user_smoothed}


def mean_over(folds_out, key):
    return float(np.mean([f[key] for f in folds_out]))


if __name__ == "__main__":
    df, s = load()
    out = {"runtime": {}, "failure": {}}
    for name, fn in RUNTIME.items():
        res = []
        for k, fit, val in folds(df, s, "y_logrt"):
            m = runtime_metrics(val.y_logrt.to_numpy(), fn(fit, val)); m["fold"] = k; res.append(m)
        out["runtime"][name] = {"folds": res, "mean_mae": mean_over(res, "mae"), "mean_rmse": mean_over(res, "rmse")}
    for name, fn in FAILURE.items():
        res = []
        for k, fit, val in folds(df, s, "y_fail"):
            m = failure_metrics(val.y_fail.to_numpy(), fn(fit, val)); m["fold"] = k; res.append(m)
        out["failure"][name] = {"folds": res, "mean_pr_auc": mean_over(res, "pr_auc"), "mean_brier": mean_over(res, "brier"),
                                "mean_base_rate": mean_over(res, "base_rate")}
    best_f = max(["user_rate", "user_rate_smoothed"], key=lambda n: out["failure"][n]["mean_pr_auc"])
    out["chosen"] = {"runtime": "user_median", "failure": best_f}
    (RESULTS / "baselines_cv.json").write_text(json.dumps(out, indent=1))
    for t in out:
        if t == "chosen":
            continue
        for n, v in out[t].items():
            print(t, n, {k: round(x, 4) for k, x in v.items() if k.startswith("mean")})
    print("chosen:", out["chosen"])
