"""Metrics and the paired user-level bootstrap, defined in Phase 3 before any model existed."""
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from config import SEED

# Failure alert costs: a missed failure (FN) wastes the job's GPU allocation and the user's time;
# a false alert (FP) costs a quick manual check. Ratio fixed before modelling: C_FN / C_FP = 3,
# so the Bayes-optimal threshold for calibrated probabilities is C_FP / (C_FP + C_FN) = 0.25.
C_FP, C_FN = 1.0, 3.0


# ---------------- runtime (targets and predictions in log10 seconds) ----------------
def runtime_metrics(y, p, lo=None, hi=None):
    e = p - y
    out = {"n": int(len(y)), "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
           "mean_signed_error": float(np.mean(e)), "median_abs_error": float(np.median(np.abs(e)))}
    out["error_factor"] = float(10 ** out["mae"])
    if lo is not None:
        out["coverage_90"] = float(np.mean((y >= lo) & (y <= hi)))
        out["interval_width_log10"] = float(np.mean(hi - lo))
    return out


# ---------------- failure ----------------
def failure_metrics(y, p, threshold=None):
    out = {"n": int(len(y)), "base_rate": float(np.mean(y)), "pr_auc": float(average_precision_score(y, p)),
           "roc_auc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else None,
           "brier": float(brier_score_loss(y, p)), "brier_base": float(np.mean((y - y.mean()) ** 2)),
           "mean_pred": float(np.mean(p))}
    out["ece"] = float(ece(y, p))
    if threshold is not None:
        out.update(threshold_stats(y, p, threshold))
    return out


def threshold_stats(y, p, t):
    flag = p >= t
    tp, fp = int(np.sum(flag & (y == 1))), int(np.sum(flag & (y == 0)))
    fn = int(np.sum(~flag & (y == 1)))
    return {"threshold": float(t), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "flag_rate": float(flag.mean()), "cost_per_job": (C_FP * fp + C_FN * fn) / len(y)}


def ece(y, p, bins=10):
    """Expected calibration error with equal-frequency bins."""
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return tot


def calibration_bins(y, p, bins=10):
    q = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, len(q) - 2)
    return [{"mean_pred": float(p[idx == b].mean()), "frac_pos": float(y[idx == b].mean()), "n": int((idx == b).sum())}
            for b in range(len(q) - 1) if (idx == b).any()]


# ---------------- paired bootstrap by user ----------------
def user_bootstrap(users, stat_fn, B=1000, seed=SEED):
    """Resample USERS with replacement (cluster bootstrap); stat_fn(weights) -> dict of statistics, where
    weights[i] = how many times row i's user was drawn. Returns {name: (lo, hi)} 95% percentile intervals."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(users, return_inverse=True)
    draws = []
    for _ in range(B):
        cnt = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        draws.append(stat_fn(cnt[inv].astype(float)))
    keys = draws[0].keys()
    return {k: (float(np.percentile([d[k] for d in draws], 2.5)), float(np.percentile([d[k] for d in draws], 97.5)))
            for k in keys}


def runtime_boot_stat(y, p_model, p_base):
    em, eb = np.abs(p_model - y), np.abs(p_base - y)
    sm, sb = (p_model - y) ** 2, (p_base - y) ** 2

    def f(w):
        W = w.sum()
        mae_m, mae_b = (w @ em) / W, (w @ eb) / W
        return {"mae_model": mae_m, "mae_base": mae_b, "mae_diff": mae_m - mae_b,
                "rmse_diff": np.sqrt((w @ sm) / W) - np.sqrt((w @ sb) / W)}
    return f


def failure_boot_stat(y, p_model, p_base):
    def f(w):
        keep = w > 0
        a = average_precision_score(y[keep], p_model[keep], sample_weight=w[keep])
        b = average_precision_score(y[keep], p_base[keep], sample_weight=w[keep])
        bm = np.average((p_model[keep] - y[keep]) ** 2, weights=w[keep])
        bb = np.average((p_base[keep] - y[keep]) ** 2, weights=w[keep])
        return {"prauc_model": a, "prauc_base": b, "prauc_diff": a - b, "brier_diff": bm - bb,
                "base_rate": np.average(y[keep], weights=w[keep])}
    return f
