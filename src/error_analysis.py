"""Phase 6: pick representative test errors (runtime and failure) from saved predictions. Fits nothing.
Selection rules are fixed in code so the cases are not cherry-picked by hand; the written explanation of each case
lives in results/error_notes.json (added after reading the cases).

Usage: python src/error_analysis.py  -> results/errors.json
"""
import json

import numpy as np
import pandas as pd

from config import INTERIM, RESULTS, SEED
from dataset import load

SHOW = ["band", "role_set", "main_role", "gpu_type_spec", "workload", "n_tasks", "inst_total", "gpu_total", "cpu_total",
        "mem_total", "submit_hour", "submit_weekday", "user_n_finished", "user_median_logrt", "user_fail_rate",
        "group_n_finished", "group_median_logrt", "group_fail_rate", "user_submits_last_1h"]


def rows(sub, kind, rule, cols):
    out = []
    for _, r in sub.iterrows():
        d = {"kind": kind, "rule": rule}
        d.update({c: (None if pd.isna(r[c]) else (r[c].item() if hasattr(r[c], "item") else r[c])) for c in cols})
        out.append(d)
    return out


if __name__ == "__main__":
    df, s = load()
    test = df[df.part == "test"]
    log = json.loads((RESULTS / "test_run_log.json").read_text())
    best = log["runtime_model"]
    P = pd.read_parquet(INTERIM / "test_preds_runtime.parquet").merge(test[["job_name", "user"] + SHOW], on="job_name")
    F = pd.read_parquet(INTERIM / "test_preds_failure.parquet").merge(test[["job_name", "user"] + SHOW], on="job_name")
    P["err"] = P[best] - P.y
    P["abs_err"] = P.err.abs()
    P["base_abs"] = (P.baseline - P.y).abs()
    rng = np.random.default_rng(SEED)
    rc = ["y", best, "q05", "q95", "baseline"] + SHOW
    cases = []
    cases += rows(P.nlargest(2, "err"), "runtime", "largest over-prediction", rc)
    cases += rows(P.nsmallest(2, "err"), "runtime", "largest under-prediction", rc)
    med = P["abs_err"].median()
    typ = P.iloc[(P["abs_err"] - med).abs().argsort()[:200]]
    cases += rows(typ.sample(2, random_state=SEED), "runtime", "typical miss (|error| near the median)", rc)
    new = P[P.user_n_finished.fillna(0) == 0]
    cases += rows(new.sample(1, random_state=SEED), "runtime", "user with no finished history", rc)
    rare = P[P.gpu_type_spec != "unspecified"]
    if len(rare):
        cases += rows(rare.nlargest(1, "abs_err"), "runtime", "explicit GPU type requested (rare), largest miss", rc)
    lost = P[(P["abs_err"] > P.base_abs + 0.5)]
    cases += rows(lost.sample(1, random_state=SEED), "runtime", "baseline much better than model", rc)
    fc_ = ["y", "prob", "baseline"] + SHOW
    cases += rows(F[F.y == 0].nlargest(1, "prob"), "failure", "most confident false alarm", fc_)
    cases += rows(F[F.y == 1].nsmallest(1, "prob"), "failure", "most confident missed failure", fc_)
    fnew = F[F.user_n_finished.fillna(0) == 0]
    cases += rows(fnew.sample(1, random_state=SEED), "failure", "user with no finished history", fc_)
    for i, c in enumerate(cases, 1):
        c["case"] = i
    summary = {
        "runtime_share_model_beats_baseline": float((P["abs_err"] < P.base_abs).mean()),
        "runtime_abs_err_quantiles": P["abs_err"].quantile([.5, .75, .9, .99]).round(4).to_dict(),
        "runtime_mae_by_role_set": P.groupby("role_set").agg(n=("abs_err", "size"), mae_model=("abs_err", "mean"), mae_base=("base_abs", "mean"))
                                    .query("n >= 300").round(4).reset_index().to_dict("records"),
        "runtime_mae_by_gpu_type_spec": P.groupby("gpu_type_spec").agg(n=("abs_err", "size"), mae_model=("abs_err", "mean"), mae_base=("base_abs", "mean"))
                                         .round(4).reset_index().to_dict("records"),
        "runtime_mae_by_true_length": P.assign(bin=pd.cut(P.y, [0, 1, 2, 3, 4, 5, 7], labels=["<10s", "10s-100s", "100s-17m", "17m-2.8h", "2.8h-28h", ">28h"]))
                                      .groupby("bin", observed=True).agg(n=("abs_err", "size"), mae_model=("abs_err", "mean"), mse_model=("err", "mean"),
                                                                         mae_base=("base_abs", "mean")).round(4).reset_index().to_dict("records"),
    }
    (RESULTS / "errors.json").write_text(json.dumps({"cases": cases, "summary": summary}, indent=1, default=str))
    for c in cases:
        print(c["case"], c["kind"], c["rule"], {k: c[k] for k in ["y", "baseline", "user_n_finished", "role_set"] if k in c},
              c.get(best, c.get("prob")))
    print(json.dumps(summary, indent=1, default=str))
