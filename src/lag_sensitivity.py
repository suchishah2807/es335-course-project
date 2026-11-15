"""Sensitivity of the 24 h history rule (development data only; the test set is not scored).

Failed jobs with no recorded end are treated as known at submit + 24 h. Of failed jobs that do record an end, 2.77%
ran longer than 24 h (0.92% longer than 72 h), so the rule may let a few failures into history early. Here we rebuild
the history features with a 72 h lag and refit the two chosen models (frozen settings) on the same three walk-forward
folds. If fold scores barely move, the possible leak is not driving the results.

Usage: python src/lag_sensitivity.py  -> results/lag_sensitivity.json
"""
import json

import numpy as np
import pandas as pd

from baselines import failure_user, runtime_user
from build_table import history
from config import INTERIM, RESULTS
from features import ALL, add_user_id
from metrics import failure_metrics, runtime_metrics
from model_common import LinearModel, TreeModel
from split import assign, fold_masks, load_split

LAGS_H = [24, 72]


def with_lag(base, raw, lag_h):
    df = base.drop(columns=[c for c in base.columns if c.startswith(("user_", "group_"))]).copy()
    end = df.job_name.map(raw.end_time.fillna(raw.t_end_max))
    finished = df.y_fail.notna()
    df["known_time"] = np.where(finished, end.fillna(df.submit_time + lag_h * 3600), np.inf)
    df["finished"] = finished
    df = pd.concat([df, history(df, "user", "user"), history(df, "group", "group")], axis=1)
    return df


if __name__ == "__main__":
    s = load_split()
    base = pd.read_parquet(INTERIM / "jobs.parquet").sort_values("submit_time", kind="stable").reset_index(drop=True)
    raw = pd.read_parquet(INTERIM / "audit_jobs.parquet", columns=["job_name", "end_time", "t_end_max"]).set_index("job_name")
    rc = json.loads((RESULTS / "runtime_config.json").read_text())
    fc = json.loads((RESULTS / "failure_config.json").read_text())
    out = {"note": __doc__.strip().splitlines()[0], "lags": {}}
    for lag in LAGS_H:
        df = add_user_id(assign(with_lag(base, raw, lag), s))
        res = []
        for f in s["folds"]:
            fit, val = fold_masks(df, f)
            fr, vr = df[fit & df.y_logrt.notna()], df[val & df.y_logrt.notna()]
            ff, vf = df[fit & df.y_fail.notna()], df[val & df.y_fail.notna()]
            pr = TreeModel(ALL, "reg", max_iter=rc["max_iter"], **rc["tree_params"]).fit(fr, fr.y_logrt.to_numpy()).predict(vr)
            pf = LinearModel(ALL, "clf", C=fc["logit_C"]).fit(ff, ff.y_fail.to_numpy()).predict(vf)
            r = {"fold": f["fold"],
                 "runtime_mae": runtime_metrics(vr.y_logrt.to_numpy(), pr)["mae"],
                 "runtime_baseline_mae": runtime_metrics(vr.y_logrt.to_numpy(), runtime_user(fr, vr))["mae"],
                 "failure_pr_auc": failure_metrics(vf.y_fail.to_numpy(), pf)["pr_auc"],
                 "failure_baseline_pr_auc": failure_metrics(vf.y_fail.to_numpy(), failure_user(ff, vf))["pr_auc"]}
            res.append(r)
            print(lag, "h", {k: round(v, 4) for k, v in r.items()}, flush=True)
        out["lags"][f"{lag}h"] = {"folds": res, **{f"mean_{k}": float(np.mean([r[k] for r in res])) for k in res[0] if k != "fold"}}
    a, b = out["lags"]["24h"], out["lags"]["72h"]
    out["change_72_minus_24"] = {k: b[k] - a[k] for k in a if k.startswith("mean_")}
    (RESULTS / "lag_sensitivity.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["change_72_minus_24"], indent=1))
