"""Phase 5b (Suchi), follow-up on walk-forward folds only: in the first grid the failure tree got WORSE as capacity
grew (PR-AUC 0.676 -> 0.644) and never beat logistic (0.709). To give the tree a fair chance we extend the grid in
the direction the folds point to (smaller, more regularised trees). Same keep rule: beat logistic by > 0.02 mean PR-AUC
and on every fold. Updates results/failure_config.json only if the rule passes.

Usage: python src/failure_tree_extra.py  -> results/failure_cv_extra.json
"""
import json

from config import RESULTS
from dataset import load
from failure_models import cv
from features import ALL
from model_common import TreeModel

EXTRA = [{"learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 2000, "l2_regularization": 1.0, "max_iter": 150},
         {"learning_rate": 0.03, "max_leaf_nodes": 7, "min_samples_leaf": 5000, "l2_regularization": 1.0, "max_iter": 200}]

if __name__ == "__main__":
    df, s = load()
    cfg = json.loads((RESULTS / "failure_config.json").read_text())
    logit = json.loads((RESULTS / "failure_cv.json").read_text())["logistic"]
    best_l = max(logit, key=lambda r: r["mean_pr_auc"])
    res = []
    for p in EXTRA:
        r, _ = cv(lambda: TreeModel(ALL, "clf", **p), df, s, f"hgb extra {p}")
        res.append(r | {"params": p})
    best = max(res, key=lambda r: r["mean_pr_auc"])
    per_fold = all(t["pr_auc"] > l["pr_auc"] for t, l in zip(best["folds"], best_l["folds"]))
    keep = (best["mean_pr_auc"] - best_l["mean_pr_auc"] > 0.02) and per_fold
    out = {"extra": res, "best_extra_mean_pr_auc": best["mean_pr_auc"], "logit_mean_pr_auc": best_l["mean_pr_auc"],
           "keep_tree": keep, "tree_beats_logit_every_fold": per_fold}
    (RESULTS / "failure_cv_extra.json").write_text(json.dumps(out, indent=1))
    # the reported tree (comparator, or the model if it wins) is the best tree over both grids
    if best["mean_pr_auc"] > cfg["tree_mean_pr_auc"]:
        p = dict(best["params"]); cfg["max_iter"] = p.pop("max_iter"); cfg["tree_params"] = p
        cfg["tree_mean_pr_auc"] = best["mean_pr_auc"]
        cfg["tree_note"] = ("tree_params = best tree over both grids (regularised extension); kept only as a comparator "
                            "because it does not clearly beat logistic") if not keep else "regularised tree clears the keep rule"
    cfg["keep_tree"] = bool(keep)
    (RESULTS / "failure_config.json").write_text(json.dumps(cfg, indent=1))
    print({k: v for k, v in out.items() if k != "extra"})
