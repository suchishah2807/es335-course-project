"""Phase 6: THE final test run. Models and settings are frozen in results/runtime_config.json and
results/failure_config.json (chosen on walk-forward folds). This script refits them once on all development rows,
scores the test set once, and records that it ran in results/test_run_log.json. It refuses to run twice.

Usage: python src/final_test.py --reproduce  (fresh clone: rerun the frozen pipeline, compare, do not overwrite)
       python src/final_test.py --dry    (code-path check: fold 3 stands in for test; test rows untouched)
       python src/final_test.py          (the real run; add --force only with a DECISIONS.md entry)
Writes: results/final_metrics.json, data/interim/test_preds.parquet, results/test_run_log.json
"""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from baselines import failure_global, failure_user, runtime_global, runtime_user
from config import INTERIM, RESULTS
from dataset import dev_and_test, load
from features import ALL, LADDER, LEAKY
from metrics import (calibration_bins, failure_boot_stat, failure_metrics, runtime_boot_stat, runtime_metrics,
                     threshold_stats, user_bootstrap)
from model_common import LinearModel, TreeModel
from split import fold_masks

LOG = RESULTS / "test_run_log.json"
B_MAIN, B_SLICE = 1000, 300


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def rt_block(y, p, base, users, lo=None, hi=None, B=B_MAIN):
    out = {"model": runtime_metrics(y, p, lo, hi), "baseline": runtime_metrics(y, base)}
    out["ci"] = user_bootstrap(users, runtime_boot_stat(y, p, base), B=B)
    return out


def fl_block(y, p, base, users, thr, B=B_MAIN):
    out = {"model": failure_metrics(y, p, thr), "baseline": failure_metrics(y, base, thr)}
    out["ci"] = user_bootstrap(users, failure_boot_stat(y, p, base), B=B)
    return out


def main():
    dry = "--dry" in sys.argv
    repro = "--reproduce" in sys.argv  # fresh-clone check: same frozen run, results compared, log untouched
    if not dry and not repro:
        prev = json.loads(LOG.read_text()) if LOG.exists() else None
        if prev and prev["status"] == "completed" and "--force" not in sys.argv:
            sys.exit(f"Final test already completed ({prev['ran_at_utc']}). Not running again.")
        attempts = (prev or {}).get("attempts", []) + [{"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                                        "git_commit": git_commit()}]
        LOG.write_text(json.dumps({"status": "started", "ran_at_utc": None, "attempts": attempts}, indent=1))
    t_start = time.time()
    rc = json.loads((RESULTS / "runtime_config.json").read_text())
    fc = json.loads((RESULTS / "failure_config.json").read_text())
    df, s = load()
    if dry:  # fast code-path check: tiny models and bootstraps
        global B_MAIN, B_SLICE
        B_MAIN, B_SLICE = 20, 10
        rc["max_iter"] = fc["max_iter"] = 15
    if dry:  # code-path check on development data only: fold-3 fit set as "dev", fold-3 validation as "test"
        fit3, val3 = fold_masks(df, s["folds"][-1])
        df.loc[val3, "band"] = 1 + (df.loc[val3, "submit_time"].rank(pct=True) * 4).clip(upper=3.999).astype(int)
        dev_r, test_r = df[fit3 & df.y_logrt.notna()], df[val3 & df.y_logrt.notna()]
        dev_f, test_f = df[fit3 & df.y_fail.notna()], df[val3 & df.y_fail.notna()]
    else:
        dev_r, test_r = dev_and_test(df, "y_logrt")
        dev_f, test_f = dev_and_test(df, "y_fail")
    yr, yf = dev_r.y_logrt.to_numpy(), dev_f.y_fail.to_numpy()
    tp = {k: v for k, v in rc["tree_params"].items() if k != "loss"}
    R = {}  # all results

    # ---------------- runtime ----------------
    print("runtime models...", flush=True)
    m_ridge = LinearModel(ALL, "reg", alpha=rc["ridge_alpha"]).fit(dev_r, yr)
    m_tree = TreeModel(ALL, "reg", max_iter=rc["max_iter"], **rc["tree_params"]).fit(dev_r, yr)
    qm = {a: TreeModel(ALL, "reg", loss="quantile", quantile=a, max_iter=rc["max_iter"], **tp).fit(dev_r, yr)
          for a in (0.05, 0.5, 0.95)}
    q = {a: m.predict(test_r) for a, m in qm.items()}
    lo, hi = np.minimum(q[0.05], q[0.95]), np.maximum(q[0.05], q[0.95])
    off = rc.get("cqr_offset_log10", 0.0) if rc.get("use_cqr") else 0.0  # conformal widening learned on fold 3
    R["interval_raw"] = runtime_metrics(test_r.y_logrt.to_numpy(), q[0.5], lo, hi)
    lo, hi = lo - off, hi + off
    P = pd.DataFrame({"job_name": test_r.job_name.values})
    P["y"] = test_r.y_logrt.values
    P["ridge"] = m_ridge.predict(test_r)
    P["tree"] = m_tree.predict(test_r)
    P["q05"], P["q50"], P["q95"] = lo, q[0.5], hi
    P["baseline"] = runtime_user(dev_r, test_r)
    P["global"] = runtime_global(dev_r, test_r)
    best = "tree" if rc["keep_tree"] else "ridge"
    R["runtime_model"] = best
    y, users = P.y.to_numpy(), test_r.user.to_numpy()
    print("runtime bootstrap...", flush=True)
    R["runtime"] = {"overall": rt_block(y, P[best].to_numpy(), P.baseline.to_numpy(), users, lo, hi),
                    "ridge": runtime_metrics(y, P.ridge.to_numpy()), "tree": runtime_metrics(y, P.tree.to_numpy()),
                    "q50": runtime_metrics(y, P.q50.to_numpy(), lo, hi), "global": runtime_metrics(y, P["global"].to_numpy())}
    R["runtime"]["ci_vs_ridge"] = user_bootstrap(users, runtime_boot_stat(y, P[best].to_numpy(), P.ridge.to_numpy()), B=B_MAIN)
    # per band, seen/unseen, heavy user
    bands, seen = test_r.band.to_numpy(), test_r.user.isin(set(dev_r.user) | set(dev_f.user)).to_numpy()
    hist = (test_r.user_n_finished.fillna(0) > 0).to_numpy()
    heavy = df[df.part == "dev"].user.value_counts().index[0]
    slices = {f"band_{b}": bands == b for b in range(1, 5)}
    slices.update({"seen_user": seen, "unseen_user": ~seen, "has_history": hist, "no_history": ~hist,
                   "heavy_user_excluded": users != heavy, "heavy_user_only": users == heavy})
    R["runtime"]["slices"] = {k: rt_block(y[m], P[best].to_numpy()[m], P.baseline.to_numpy()[m], users[m], lo[m], hi[m], B=B_SLICE)
                              for k, m in slices.items() if m.sum() >= 50}

    # ---------------- failure ----------------
    print("failure models...", flush=True)
    m_log = LinearModel(ALL, "clf", C=fc["logit_C"]).fit(dev_f, yf)
    m_ftree = TreeModel(ALL, "clf", max_iter=fc["max_iter"], **fc["tree_params"]).fit(dev_f, yf)
    fbest = "tree" if fc["keep_tree"] else "logistic"
    def make_family(cols):  # the chosen failure model family, used for the ablation ladder and the leakage run
        if fc["keep_tree"]:
            return TreeModel(cols, "clf", max_iter=fc["max_iter"], **fc["tree_params"])
        return LinearModel(cols, "clf", C=fc["logit_C"])
    make_best = (lambda: TreeModel(ALL, "clf", max_iter=fc["max_iter"], **fc["tree_params"])) if fc["keep_tree"] \
        else (lambda: LinearModel(ALL, "clf", C=fc["logit_C"]))
    F = pd.DataFrame({"job_name": test_f.job_name.values, "y": test_f.y_fail.values})
    F["logistic"] = m_log.predict(test_f)
    F["tree"] = m_ftree.predict(test_f)
    F["raw"] = F[fbest]
    if fc["use_isotonic"]:
        # calibrator fit on the last walk-forward fold (development data only)
        lab = df.y_fail.notna()
        fit3, val3 = fold_masks(df, s["folds"][-1])
        m3 = make_best().fit(df[fit3 & lab], df[fit3 & lab].y_fail.to_numpy())
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(m3.predict(df[val3 & lab]), df[val3 & lab].y_fail)
        F["prob"] = iso.predict(F.raw)
        R["calibrator"] = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
    else:
        F["prob"] = F.raw
    F["baseline"] = failure_user(dev_f, test_f)
    F["global"] = failure_global(dev_f, test_f)
    thr = fc["threshold"]
    R["failure_model"] = fbest
    yv, fu = F.y.to_numpy(), test_f.user.to_numpy()
    print("failure bootstrap...", flush=True)
    R["failure"] = {"overall": fl_block(yv, F.prob.to_numpy(), F.baseline.to_numpy(), fu, thr),
                    "raw": failure_metrics(yv, F.raw.to_numpy(), thr), "logistic": failure_metrics(yv, F.logistic.to_numpy()),
                    "tree": failure_metrics(yv, F.tree.to_numpy()), "global": failure_metrics(yv, F["global"].to_numpy())}
    R["failure"]["ci_vs_logistic"] = user_bootstrap(fu, failure_boot_stat(yv, F.prob.to_numpy(), F.logistic.to_numpy()), B=B_MAIN)
    fb = test_f.band.to_numpy()
    fseen = test_f.user.isin(set(dev_r.user) | set(dev_f.user)).to_numpy()
    fhist = (test_f.user_n_finished.fillna(0) > 0).to_numpy()
    fsl = {f"band_{b}": fb == b for b in range(1, 5)}
    fsl.update({"seen_user": fseen, "unseen_user": ~fseen, "has_history": fhist, "no_history": ~fhist,
                "heavy_user_excluded": fu != heavy, "heavy_user_only": fu == heavy})
    R["failure"]["slices"] = {k: fl_block(yv[m], F.prob.to_numpy()[m], F.baseline.to_numpy()[m], fu[m], thr, B=B_SLICE)
                              for k, m in fsl.items() if m.sum() >= 50 and 0 < yv[m].mean() < 1}
    R["failure"]["calibration_bins"] = {"model": calibration_bins(yv, F.prob.to_numpy()),
                                        "raw": calibration_bins(yv, F.raw.to_numpy()),
                                        "baseline": calibration_bins(yv, F.baseline.to_numpy())}
    gpus = test_f.gpu_total.to_numpy()
    R["failure"]["threshold_curve"] = [
        threshold_stats(yv, F.prob.to_numpy(), t) | {
            "gpus_flagged_share": float(gpus[F.prob.to_numpy() >= t].sum() / gpus.sum()),
            "failed_gpus_caught_share": float(gpus[(F.prob.to_numpy() >= t) & (yv == 1)].sum() / gpus[yv == 1].sum()),
            "baseline": threshold_stats(yv, F.baseline.to_numpy(), t)}
        for t in np.round(np.arange(0.02, 0.99, 0.02), 2)]

    # ---------------- ablation ladder (chosen family; tree settings) ----------------
    print("ablation...", flush=True)
    R["ablation"] = {"runtime": [], "failure": []}
    cols = []
    for name, group in LADDER:
        cols = cols + group
        pr = TreeModel(cols, "reg", max_iter=rc["max_iter"], **rc["tree_params"]).fit(dev_r, yr).predict(test_r)
        r = rt_block(y, pr, P.baseline.to_numpy(), users, B=B_SLICE)
        R["ablation"]["runtime"].append({"rung": name, "n_features": len(cols), **r})
        pf = make_family(cols).fit(dev_f, yf).predict(test_f)
        r = fl_block(yv, pf, F.baseline.to_numpy(), fu, thr, B=B_SLICE)
        R["ablation"]["failure"].append({"rung": name, "n_features": len(cols), **r})
        print("  ", name, flush=True)

    # ---------------- sensitivity: assigned GPU type (leakage) ----------------
    pr = TreeModel(ALL + LEAKY, "reg", max_iter=rc["max_iter"], **rc["tree_params"]).fit(dev_r, yr).predict(test_r)
    pf = make_family(ALL + LEAKY).fit(dev_f, yf).predict(test_f)
    R["sensitivity_assigned_gpu"] = {"runtime": runtime_metrics(y, pr), "runtime_tree_without": runtime_metrics(y, P.tree.to_numpy()),
                                     "failure": failure_metrics(yv, pf), "failure_without": failure_metrics(yv, F.raw.to_numpy())}

    R["threshold"] = thr
    if dry:
        (INTERIM / "dry_run_metrics.json").write_text(json.dumps(R, indent=1, default=float))
        print("dry run OK (development data only)")
        return
    if repro:
        (INTERIM / "reproduce_metrics.json").write_text(json.dumps(R, indent=1, default=float))
        ref = json.loads((RESULTS / "final_metrics.json").read_text())
        for task, key in [("runtime", "mae"), ("failure", "pr_auc"), ("failure", "brier")]:
            a, b = ref[task]["overall"]["model"][key], R[task]["overall"]["model"][key]
            print(f"{task} {key}: committed {a:.4f}  reproduced {b:.4f}  {'MATCH' if abs(a - b) < 5e-4 else 'DIFFERS'}")
        return
    import joblib
    (INTERIM / "models").mkdir(exist_ok=True)
    joblib.dump({"runtime": m_tree if best == "tree" else m_ridge, "quantiles": qm,
                 "failure": m_ftree if fbest == "tree" else m_log, "calibrator": iso if fc["use_isotonic"] else None,
                 "dev_runtime_median": float(np.median(yr)), "dev_fail_rate": float(yf.mean())},
                INTERIM / "models" / "final_models.joblib")
    P.to_parquet(INTERIM / "test_preds_runtime.parquet", index=False)
    F.to_parquet(INTERIM / "test_preds_failure.parquet", index=False)
    (RESULTS / "final_metrics.json").write_text(json.dumps(R, indent=1, default=float))
    LOG.write_text(json.dumps({"status": "completed", "ran_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "attempts": attempts, "completed_runs": 1, "runtime_model": best, "failure_model": fbest,
                               "runtime_config": rc, "failure_config": fc, "seconds": round(time.time() - t_start)},
                              indent=1, default=float))
    print("done in", round(time.time() - t_start), "s")


if __name__ == "__main__":
    main()
