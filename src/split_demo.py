"""Why we split by time: the same model scored with a time-ordered split vs a random split of the SAME development
rows (fold 3 fit + validation). Development data only; the test set is not touched.

Usage: python src/split_demo.py   -> results/split_demo.json
"""
import json

import numpy as np

from baselines import failure_user, runtime_user
from config import RESULTS, SEED
from dataset import load
from features import ALL
from metrics import failure_metrics, runtime_metrics
from model_common import TreeModel
from split import fold_masks


def run(task, fit, val, make):
    y = task
    m = make().fit(fit, fit[y].to_numpy())
    p = m.predict(val)
    if y == "y_logrt":
        return {"model": runtime_metrics(val[y].to_numpy(), p), "baseline": runtime_metrics(val[y].to_numpy(), runtime_user(fit, val))}
    return {"model": failure_metrics(val[y].to_numpy(), p), "baseline": failure_metrics(val[y].to_numpy(), failure_user(fit, val))}


if __name__ == "__main__":
    df, s = load()
    rc = json.loads((RESULTS / "runtime_config.json").read_text())
    fc = json.loads((RESULTS / "failure_config.json").read_text())
    fit, val = fold_masks(df, s["folds"][-1])
    out = {}
    rng = np.random.default_rng(SEED)
    for task, make in [("y_logrt", lambda: TreeModel(ALL, "reg", max_iter=rc["max_iter"], **rc["tree_params"])),
                       ("y_fail", lambda: TreeModel(ALL, "clf", max_iter=fc["max_iter"], **fc["tree_params"]))]:
        lab = df[task].notna()
        f_t, v_t = df[fit & lab], df[val & lab]
        pool = df[(fit | val) & lab]
        is_val = rng.random(len(pool)) < len(v_t) / len(pool)
        f_r, v_r = pool[~is_val], pool[is_val]
        out[task] = {"time_split": run(task, f_t, v_t, make), "random_split": run(task, f_r, v_r, make),
                     "n_fit_time": len(f_t), "n_val_time": len(v_t), "n_fit_random": len(f_r), "n_val_random": len(v_r)}
        print(task, json.dumps({k: {kk: round(vv["model"].get("mae", vv["model"].get("pr_auc")), 4) for kk, vv in v.items()
                                    if isinstance(vv, dict)} for k, v in [("x", out[task])]}))
    out["note"] = ("Random split: rows from the fold-3 fit and validation windows shuffled together; validation size kept. "
                   "Note that history features are time-safe in both cases; the inflation comes from same-period jobs, "
                   "bursts and repeated scripts appearing on both sides.")
    (RESULTS / "split_demo.json").write_text(json.dumps(out, indent=1))
