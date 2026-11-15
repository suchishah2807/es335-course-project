"""Phase 5a (Saanvi): runtime model selection on walk-forward folds only. The test set is never loaded here.

Candidates: ridge (alpha grid), HistGradientBoosting with squared or absolute loss (small grid), and quantile HGB
(5/50/95%) with the chosen tree settings. Selection = lowest mean fold MAE. The tree is kept only if it clearly
beats ridge (mean MAE lower by > 0.02 log10 AND lower on every fold).

Usage: python src/runtime_models.py  -> results/runtime_cv.json, results/runtime_config.json
"""
import itertools
import json
import time

import numpy as np

from baselines import runtime_user
from config import RESULTS
from dataset import folds, load
from features import ALL
from metrics import runtime_metrics
from model_common import LinearModel, TreeModel

RIDGE_ALPHAS = [0.1, 10, 1000]
TREE_GRID = {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [31, 127], "min_samples_leaf": [100]}
LOSSES = ["squared_error", "absolute_error"]
MAX_ITER = 300


def cv(make, df, s, label):
    res, t = [], time.time()
    for k, fit, val in folds(df, s, "y_logrt"):
        m = make().fit(fit, fit.y_logrt.to_numpy())
        p = m.predict(val)
        r = runtime_metrics(val.y_logrt.to_numpy(), p)
        r["fold"] = k
        r["beats_baseline_mae"] = r["mae"] < runtime_metrics(val.y_logrt.to_numpy(), runtime_user(fit, val))["mae"]
        res.append(r)
    out = {"label": label, "folds": res, "mean_mae": float(np.mean([r["mae"] for r in res])),
           "mean_rmse": float(np.mean([r["rmse"] for r in res])), "seconds": round(time.time() - t)}
    print(f"{label:60s} MAE {out['mean_mae']:.4f}  RMSE {out['mean_rmse']:.4f}  ({out['seconds']} s)", flush=True)
    return out


def quantile_cv(df, s, params):
    res = []
    for k, fit, val in folds(df, s, "y_logrt"):
        y = fit.y_logrt.to_numpy()
        q = {a: TreeModel(ALL, "reg", loss="quantile", quantile=a, max_iter=MAX_ITER, **params).fit(fit, y).predict(val)
             for a in (0.05, 0.5, 0.95)}
        r = runtime_metrics(val.y_logrt.to_numpy(), q[0.5], q[0.05], q[0.95])
        r["fold"] = k
        res.append(r)
        print(f"  quantile fold {k}: coverage {r['coverage_90']:.3f} width {r['interval_width_log10']:.3f} "
              f"MAE(q50) {r['mae']:.4f}", flush=True)
    return res


if __name__ == "__main__":
    df, s = load()
    out = {"ridge": [], "tree": []}
    for a in RIDGE_ALPHAS:
        out["ridge"].append(cv(lambda: LinearModel(ALL, "reg", alpha=a), df, s, f"ridge alpha={a}") | {"alpha": a})
    for loss, (lr, leaves, msl) in itertools.product(LOSSES, itertools.product(*TREE_GRID.values())):
        params = {"loss": loss, "learning_rate": lr, "max_leaf_nodes": leaves, "min_samples_leaf": msl}
        out["tree"].append(cv(lambda: TreeModel(ALL, "reg", max_iter=MAX_ITER, **params), df, s,
                              f"hgb {params}") | {"params": params})
    best_r = min(out["ridge"], key=lambda r: r["mean_mae"])
    best_t = min(out["tree"], key=lambda r: r["mean_mae"])
    per_fold_better = all(t["mae"] < r["mae"] for t, r in zip(best_t["folds"], best_r["folds"]))
    keep_tree = (best_r["mean_mae"] - best_t["mean_mae"] > 0.02) and per_fold_better
    tree_params = {k: v for k, v in best_t["params"].items() if k != "loss"}
    out["quantile"] = quantile_cv(df, s, tree_params)
    cfg = {"ridge_alpha": best_r["alpha"], "tree_params": best_t["params"], "max_iter": MAX_ITER,
           "keep_tree": bool(keep_tree), "tree_beats_ridge_every_fold": bool(per_fold_better),
           "ridge_mean_mae": best_r["mean_mae"], "tree_mean_mae": best_t["mean_mae"],
           "quantile_mean_coverage": float(np.mean([r["coverage_90"] for r in out["quantile"]]))}
    (RESULTS / "runtime_cv.json").write_text(json.dumps(out, indent=1))
    (RESULTS / "runtime_config.json").write_text(json.dumps(cfg, indent=1))
    print(json.dumps(cfg, indent=1))
