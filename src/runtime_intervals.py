"""Phase 5a (Saanvi): the raw 5–95% quantile interval under-covers on walk-forward folds (~0.78–0.80 vs 0.90 target).
Check a conformalised fix (CQR, Romano et al. 2019) on development data only:
  score_i = max(q05_i - y_i, y_i - q95_i) on a calibration week; offset = 90% quantile of the scores;
  interval = [q05 - offset, q95 + offset].
The offset is learned on fold k's validation week and evaluated on fold k+1's validation week (forward in time).

Usage: python src/runtime_intervals.py -> results/runtime_intervals_cv.json (updates runtime_config.json with use_cqr)
"""
import json

import numpy as np

from config import RESULTS
from dataset import folds, load
from features import ALL
from model_common import TreeModel


def offset(y, lo, hi, level=0.90):
    sc = np.maximum(lo - y, y - hi)
    n = len(sc)
    return float(np.quantile(sc, min(1.0, np.ceil((n + 1) * level) / n)))


if __name__ == "__main__":
    df, s = load()
    cfg = json.loads((RESULTS / "runtime_config.json").read_text())
    tp = {k: v for k, v in cfg["tree_params"].items() if k != "loss"}
    preds = []
    for k, fit, val in folds(df, s, "y_logrt"):
        y = fit.y_logrt.to_numpy()
        lo = TreeModel(ALL, "reg", loss="quantile", quantile=0.05, max_iter=cfg["max_iter"], **tp).fit(fit, y).predict(val)
        hi = TreeModel(ALL, "reg", loss="quantile", quantile=0.95, max_iter=cfg["max_iter"], **tp).fit(fit, y).predict(val)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        preds.append((val.y_logrt.to_numpy(), lo, hi))
        print("fold", k, "raw coverage", round(float(np.mean((preds[-1][0] >= lo) & (preds[-1][0] <= hi))), 3), flush=True)
    res = []
    for k in range(1, len(preds)):
        off = offset(*preds[k - 1])
        y, lo, hi = preds[k]
        res.append({"fold": k + 1, "offset_from_fold": k, "offset_log10": off,
                    "coverage_raw": float(np.mean((y >= lo) & (y <= hi))), "width_raw": float(np.mean(hi - lo)),
                    "coverage_cqr": float(np.mean((y >= lo - off) & (y <= hi + off))), "width_cqr": float(np.mean(hi - lo + 2 * off))})
        print(res[-1], flush=True)
    use = all(abs(r["coverage_cqr"] - 0.9) < abs(r["coverage_raw"] - 0.9) for r in res)
    final_offset = offset(*preds[-1])  # learned on fold 3's validation week, applied to the final model
    cfg.update({"use_cqr": bool(use), "cqr_offset_log10": final_offset if use else 0.0})
    (RESULTS / "runtime_intervals_cv.json").write_text(json.dumps({"folds": res, "final_offset_from_fold3": final_offset}, indent=1))
    (RESULTS / "runtime_config.json").write_text(json.dumps(cfg, indent=1))
    print("use_cqr", use, "offset", final_offset)
