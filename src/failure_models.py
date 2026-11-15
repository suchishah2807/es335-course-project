"""Phase 5b (Suchi): failure model selection on walk-forward folds only. The test set is never loaded here.

Candidates: logistic regression (C grid); HistGradientBoosting classifier (small grid) with and without
class weighting. Selection = highest mean fold PR-AUC; tree kept only if it clearly beats logistic (PR-AUC higher by
> 0.02 on the mean AND on every fold).
Calibration: isotonic regression fit on the PREVIOUS fold's validation predictions, applied to the next fold
(fold 2 uses fold 1's, fold 3 uses fold 2's) — kept only if it lowers Brier on both of those folds.
Threshold: minimises cost (C_FN = 3, C_FP = 1) pooled over folds 2-3 with the final probability pipeline.

Usage: python src/failure_models.py  -> results/failure_cv.json, results/failure_config.json
"""
import itertools
import json
import time

import numpy as np
from sklearn.isotonic import IsotonicRegression

from baselines import failure_user
from config import RESULTS
from dataset import folds, load
from features import ALL
from metrics import C_FN, C_FP, calibration_bins, failure_metrics, threshold_stats
from model_common import LinearModel, TreeModel

LOGIT_C = [0.01, 0.1, 1.0]
TREE_GRID = {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [31, 127], "min_samples_leaf": [100]}
WEIGHTS = [None, "balanced"]
MAX_ITER = 300


def cv(make, df, s, label, keep_preds=False):
    res, preds, t = [], [], time.time()
    for k, fit, val in folds(df, s, "y_fail"):
        m = make().fit(fit, fit.y_fail.to_numpy())
        p = m.predict(val)
        y = val.y_fail.to_numpy()
        r = failure_metrics(y, p)
        r["fold"] = k
        r["baseline_pr_auc"] = failure_metrics(y, failure_user(fit, val))["pr_auc"]
        res.append(r)
        if keep_preds:
            preds.append((y, p))
    out = {"label": label, "folds": res, "mean_pr_auc": float(np.mean([r["pr_auc"] for r in res])),
           "mean_brier": float(np.mean([r["brier"] for r in res])), "seconds": round(time.time() - t)}
    print(f"{label:70s} PR-AUC {out['mean_pr_auc']:.4f}  Brier {out['mean_brier']:.4f}  ({out['seconds']} s)", flush=True)
    return out, preds


def calibration_check(preds):
    """preds[k] = (y, p) for fold k+1. Fit isotonic on fold k, evaluate on fold k+1."""
    res = []
    for k in range(1, len(preds)):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(preds[k - 1][1], preds[k - 1][0])
        y, p = preds[k]
        pc = iso.predict(p)
        res.append({"fold": k + 1, "brier_raw": float(np.mean((p - y) ** 2)), "brier_iso": float(np.mean((pc - y) ** 2)),
                    "ece_raw": failure_metrics(y, p)["ece"], "ece_iso": failure_metrics(y, pc)["ece"],
                    "bins_raw": calibration_bins(y, p), "bins_iso": calibration_bins(y, pc)})
    return res


def choose_threshold(pairs):
    y = np.concatenate([a for a, _ in pairs])
    p = np.concatenate([b for _, b in pairs])
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    costs = [threshold_stats(y, p, t)["cost_per_job"] for t in grid]
    t = float(grid[int(np.argmin(costs))])
    return t, [{"threshold": float(g), "cost_per_job": float(c)} for g, c in zip(grid, costs)]


if __name__ == "__main__":
    df, s = load()
    out = {"logistic": [], "tree": []}
    for C in LOGIT_C:
        r, _ = cv(lambda: LinearModel(ALL, "clf", C=C), df, s, f"logistic C={C}")
        out["logistic"].append(r | {"C": C})
    for cw, (lr, leaves, msl) in itertools.product(WEIGHTS, itertools.product(*TREE_GRID.values())):
        params = {"class_weight": cw, "learning_rate": lr, "max_leaf_nodes": leaves, "min_samples_leaf": msl}
        r, _ = cv(lambda: TreeModel(ALL, "clf", max_iter=MAX_ITER, **params), df, s, f"hgb {params}")
        out["tree"].append(r | {"params": params})
    best_l = max(out["logistic"], key=lambda r: r["mean_pr_auc"])
    best_t = max(out["tree"], key=lambda r: r["mean_pr_auc"])
    per_fold = all(t["pr_auc"] > l["pr_auc"] for t, l in zip(best_t["folds"], best_l["folds"]))
    keep_tree = (best_t["mean_pr_auc"] - best_l["mean_pr_auc"] > 0.02) and per_fold

    # calibration + threshold with the chosen model family
    if keep_tree:
        make = lambda: TreeModel(ALL, "clf", max_iter=MAX_ITER, **best_t["params"])
    else:
        make = lambda: LinearModel(ALL, "clf", C=best_l["C"])
    _, preds = cv(make, df, s, "chosen model (re-run for predictions)", keep_preds=True)
    cal = calibration_check(preds)
    use_iso = all(c["brier_iso"] < c["brier_raw"] for c in cal)
    pairs = []
    for k in range(1, len(preds)):
        y, p = preds[k]
        if use_iso:
            p = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(preds[k - 1][1], preds[k - 1][0]).predict(p)
        pairs.append((y, p))
    thr, curve = choose_threshold(pairs)
    out.update({"calibration": cal, "threshold_curve": curve})
    cfg = {"logit_C": best_l["C"], "tree_params": best_t["params"], "max_iter": MAX_ITER, "keep_tree": bool(keep_tree),
           "tree_beats_logit_every_fold": bool(per_fold), "logit_mean_pr_auc": best_l["mean_pr_auc"],
           "tree_mean_pr_auc": best_t["mean_pr_auc"], "use_isotonic": bool(use_iso), "threshold": thr,
           "bayes_threshold": C_FP / (C_FP + C_FN)}
    (RESULTS / "failure_cv.json").write_text(json.dumps(out, indent=1))
    (RESULTS / "failure_config.json").write_text(json.dumps(cfg, indent=1))
    print(json.dumps(cfg, indent=1))
