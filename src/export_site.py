"""Phase 7: export small, derived, anonymised data for the static site (docs/data/*.js). Fits nothing on test data.

- site_metrics.js : headline numbers, per-band, slices, ablation, calibration, threshold curve, split demo, audit numbers
- site_sample.js  : ~3,000 test jobs (runtime-labelled), stratified by band; derived fields only, IDs re-numbered
- site_grid.js    : prediction grid from the FINAL models over the "Try a job" controls
- site_stepper.js : one real test job walked through every step of a prediction

Files are JS (window.SITE_* = {...}) rather than JSON so the site also works when opened from disk.
Usage: python src/export_site.py
"""
import itertools
import json

import joblib
import numpy as np
import pandas as pd

from config import INTERIM, RESULTS, SEED, SITE_DATA
from dataset import load
from features import ALL

R = json.loads((RESULTS / "final_metrics.json").read_text(encoding="utf8"))
LOG = json.loads((RESULTS / "test_run_log.json").read_text(encoding="utf8"))


def write_js(name, var, obj):
    txt = json.dumps(obj, separators=(",", ":"), default=float, allow_nan=False)
    (SITE_DATA / f"{name}.js").write_text(f"window.{var}={txt};\n", encoding="utf8")
    print(f"{name}.js  {len(txt) / 1024:.0f} KB")


def r3(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 4)


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return r3(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def metrics_payload(df, s):
    audit = json.loads((RESULTS / "audit.json").read_text(encoding="utf8"))
    figs = json.loads((RESULTS / "audit_figures.json").read_text(encoding="utf8"))
    pay = {k: R[k] for k in ["runtime_model", "failure_model", "threshold", "runtime", "failure", "ablation",
                             "sensitivity_assigned_gpu", "interval_raw"] if k in R}
    pay["split"] = {k: s[k] for k in ["cut_day", "test_from_day", "trace_end_day", "band_edges_day", "counts",
                                      "fold_counts", "test_unseen_user_share_%"]}
    pay["split"]["folds_day"] = [{"fold": f["fold"], "fit_to": (f["fit_submit_before"] - s["t0"]) / 86400,
                                  "val_from": (f["val_from"] - s["t0"]) / 86400, "val_to": (f["val_to"] - s["t0"]) / 86400}
                                 for f in s["folds"]]
    daily = pd.DataFrame(audit["daily"])
    pay["daily"] = {"day": daily.day.tolist(), "n": daily.n.tolist(), "fail_rate": daily.fail_rate.tolist()}
    pay["split_demo"] = json.loads((RESULTS / "split_demo.json").read_text(encoding="utf8"))
    pay["baselines_cv"] = {t: {n: {k: v for k, v in d.items() if k.startswith("mean")} for n, d in b.items()}
                           for t, b in json.loads((RESULTS / "baselines_cv.json").read_text(encoding="utf8")).items() if t != "chosen"}
    rcv = json.loads((RESULTS / "runtime_cv.json").read_text(encoding="utf8"))
    fcv = json.loads((RESULTS / "failure_cv.json").read_text(encoding="utf8"))
    pay["cv"] = {"runtime": [{"label": r["label"], "mae": r["mean_mae"], "rmse": r["mean_rmse"]} for r in rcv["ridge"] + rcv["tree"]],
                 "failure": [{"label": r["label"], "pr_auc": r["mean_pr_auc"], "brier": r["mean_brier"]} for r in fcv["logistic"] + fcv["tree"]],
                 "intervals": json.loads((RESULTS / "runtime_intervals_cv.json").read_text(encoding="utf8")),
                 "calibration": [{k: v for k, v in c.items() if not k.startswith("bins")} for c in fcv["calibration"]]}
    pay["configs"] = {"runtime": LOG["runtime_config"], "failure": LOG["failure_config"]}
    pay["test_run"] = {"ran_at_utc": LOG["ran_at_utc"], "completed_runs": LOG["completed_runs"]}
    pay["audit"] = {"status": audit["job_status_counts"], "rows": audit["rows"], "users": audit["users"],
                    "span_days": audit["submit_span_days"], "availability": audit["availability_by_status"],
                    "missing": figs["05"], "runtime_dist": figs["04"], "users_conc": figs["08"],
                    "tasks_per_job": audit["tasks_per_job"], "gpu_type_spec": audit["gpu_type_spec_values"]}
    pay["drift"] = json.loads((RESULTS / "drift.json").read_text(encoding="utf8"))
    pay["errors"] = json.loads((RESULTS / "errors.json").read_text(encoding="utf8"))
    notes = RESULTS / "error_notes.json"
    pay["error_notes"] = json.loads(notes.read_text(encoding="utf8")) if notes.exists() else {}
    # threshold curve at 0.01 steps from the saved (frozen) test predictions, so the chosen threshold sits on the grid
    from metrics import threshold_stats
    Fp = pd.read_parquet(INTERIM / "test_preds_failure.parquet").merge(df[["job_name", "gpu_total"]], on="job_name")
    yv, pv, bv, g = Fp.y.to_numpy(), Fp.prob.to_numpy(), Fp.baseline.to_numpy(), Fp.gpu_total.to_numpy()
    pay["failure"]["threshold_curve"] = [threshold_stats(yv, pv, t) | {
        "gpus_flagged_share": float(g[pv >= t].sum() / g.sum()),
        "failed_gpus_caught_share": float(g[(pv >= t) & (yv == 1)].sum() / g[yv == 1].sum()),
        "baseline": threshold_stats(yv, bv, t)} for t in np.round(np.arange(0.01, 1.0, 0.01), 2)]
    pay["leakage"] = pd.read_csv(RESULTS / "leakage_table.csv", keep_default_na=False).to_dict("records")
    return clean(pay)


def sample_payload(df):
    best = R["runtime_model"]
    test = df[df.part == "test"]
    P = pd.read_parquet(INTERIM / "test_preds_runtime.parquet").merge(
        test[["job_name", "user", "band", "day", "submit_time", "role_set", "gpu_type_spec", "gpu_total", "cpu_total",
              "mem_total", "n_tasks", "submit_hour", "submit_weekday", "user_n_finished"]], on="job_name")
    F = pd.read_parquet(INTERIM / "test_preds_failure.parquet")[["job_name", "prob", "baseline"]].rename(
        columns={"prob": "p_fail", "baseline": "p_fail_base"})
    P = P.merge(F, on="job_name", how="left")
    dev_users = set(df[(df.part == "dev") & df.y_fail.notna()].user)
    P["seen"] = P.user.isin(dev_users)
    S = pd.concat([g.sample(min(len(g), 750), random_state=SEED) for _, g in P.groupby("band")])
    S = S.sort_values("submit_time").reset_index(drop=True)
    t0 = df.submit_time.min()
    cols = {"id": (S.index + 1).tolist(), "band": S.band.tolist(), "day": ((S.submit_time - t0) / 86400).round(2).tolist(),
            "role": S.role_set.tolist(), "gtype": S.gpu_type_spec.tolist(), "gpu": S.gpu_total.round(2).tolist(),
            "cpu": S.cpu_total.round(1).tolist(), "mem": S.mem_total.round(1).tolist(), "ntask": S.n_tasks.astype(int).tolist(),
            "hour": S.submit_hour.tolist(), "wday": S.submit_weekday.tolist(), "seen": S.seen.astype(int).tolist(),
            "hist": (S.user_n_finished.fillna(0) > 0).astype(int).tolist(), "y": S.y.round(3).tolist(),
            "pred": S[best].round(3).tolist(), "lo": S.q05.round(3).tolist(), "hi": S.q95.round(3).tolist(),
            "base": S.baseline.round(3).tolist(), "pf": S.p_fail.round(3).tolist(), "pfb": S.p_fail_base.round(3).tolist()}
    cols = {k: [None if (isinstance(v, float) and not np.isfinite(v)) else v for v in vals] for k, vals in cols.items()}
    return {"n": len(S), "cols": cols, "note": "Stratified sample of runtime-labelled test jobs (750 per band). IDs re-numbered."}


ROLES = {  # role_set -> (main_role, n_tasks, inst_total)
    "tensorflow": ("tensorflow", 1, 1), "PyTorchWorker": ("PyTorchWorker", 1, 1), "worker": ("worker", 1, 1),
    "ps+worker": ("worker", 2, 2), "xComputeWorker": ("xComputeWorker", 1, 1), "JupyterTask": ("JupyterTask", 1, 1)}
GPUS = [0, 0.5, 1, 2, 4, 8]
GTYPES = ["unspecified", "T4", "P100", "V100", "V100M32"]
HOURS = [0, 4, 8, 12, 16, 20]
DAYS = {"weekday": 2, "weekend": 5}


def grid_payload(df):
    M = joblib.load(INTERIM / "models" / "final_models.joblib")
    dev = df[(df.part == "dev") & df.y_logrt.notna()]
    sizes = {}
    for name, q in [("small", 0.25), ("medium", 0.5), ("large", 0.75)]:
        sizes[name] = (float(dev.cpu_per_inst_max.quantile(q)), float(dev.mem_per_inst_max.quantile(q)))
    hist_dev = df[(df.part == "dev") & (df.user_n_finished > 0)]
    hcols = [c for c in ALL if c.startswith(("user_", "group_")) and c != "user_id"]
    typical_hist = {c: float(hist_dev[c].median()) for c in hcols}
    rows, keys = [], []
    for role, g, gt, size, h, dname, user in itertools.product(ROLES, GPUS, GTYPES, sizes, HOURS, DAYS, ["returning", "new"]):
        mr, nt, inst = ROLES[role]
        cpu, mem = sizes[size]
        r = {"n_tasks": nt, "inst_total": inst, "cpu_total": cpu * inst, "mem_total": mem * inst, "gpu_total": g,
             "cpu_per_inst_max": cpu, "mem_per_inst_max": mem, "gpu_per_inst_max": g / max(inst - (1 if role == "ps+worker" else 0), 1),
             "has_gpu_request": int(g > 0), "gpu_type_spec": gt, "main_role": mr, "role_set": role, "workload": "untagged",
             "user_id": "OTHER", "submit_hour": h, "submit_weekday": DAYS[dname]}
        if user == "returning":
            r.update(typical_hist)
        else:
            r.update({c: (0.0 if ("_n_" in c or "submits" in c) else np.nan) for c in hcols})
        rows.append(r)
        keys.append((role, g, gt, size, h, dname, user))
    X = pd.DataFrame(rows)
    q05, q50, q95 = (M["quantiles"][a].predict(X) for a in (0.05, 0.5, 0.95))
    off = LOG["runtime_config"].get("cqr_offset_log10", 0.0) if LOG["runtime_config"].get("use_cqr") else 0.0
    lo, hi = np.minimum(q05, q95) - off, np.maximum(q05, q95) + off
    point = M["runtime"].predict(X)
    pf = M["failure"].predict(X)
    if M["calibrator"] is not None:
        pf = M["calibrator"].predict(pf)
    base_rt = np.where(X.user_n_finished > 0, X.user_median_logrt, M["dev_runtime_median"])
    base_f = np.where(X.user_n_finished > 0, X.user_fail_rate, M["dev_fail_rate"])
    out = {"axes": {"role": list(ROLES), "gpu": GPUS, "gtype": GTYPES, "size": list(sizes), "hour": HOURS,
                    "day": list(DAYS), "user": ["returning", "new"]},
           "sizes": {k: {"cpu_cores": round(v[0], 1), "mem_gb": round(v[1], 1)} for k, v in sizes.items()},
           "typical_history": {k: round(v, 3) for k, v in typical_hist.items()},
           "pred": np.round(point, 3).tolist(), "lo": np.round(lo, 3).tolist(), "hi": np.round(hi, 3).tolist(),
           "pf": np.round(pf, 3).tolist(), "base_rt": np.round(base_rt, 3).tolist(), "base_f": np.round(base_f, 3).tolist()}
    test = df[(df.part == "test") & df.y_logrt.notna()]
    out["preset"] = {"role": test.role_set.mode()[0], "gpu": float(min(GPUS, key=lambda g: abs(g - test.gpu_total.median()))),
                     "gtype": "unspecified", "size": "medium", "hour": 12, "day": "weekday", "user": "returning"}
    return out


def stepper_payload(df):
    """Walk one real, typical test job (has history, |error| near the median) through each step."""
    best = R["runtime_model"]
    test = df[df.part == "test"]
    P = pd.read_parquet(INTERIM / "test_preds_runtime.parquet").merge(test, on="job_name")
    F = pd.read_parquet(INTERIM / "test_preds_failure.parquet")[["job_name", "prob", "baseline"]].rename(
        columns={"prob": "p_fail", "baseline": "p_fail_base"})
    P = P.merge(F, on="job_name")
    P["abs_err"] = (P[best] - P.y).abs()
    cand = P[(P.user_n_finished >= 20) & (P.n_tasks == 2) & (P.gpu_total > 0)]
    if not len(cand):
        cand = P[P.user_n_finished >= 20]
    cand = cand.iloc[(cand.abs_err - P.abs_err.median()).abs().argsort()[:50]]
    j = cand.sample(1, random_state=SEED).iloc[0]
    same = df[df.user == j.user]
    before = same[same.submit_time < j.submit_time]
    known = before[before.known_time < j.submit_time]
    pending = before[before.known_time >= j.submit_time]
    t0 = df.submit_time.min()
    return clean({
        "request": {"role_set": j.role_set, "main_role": j.main_role, "n_tasks": int(j.n_tasks), "inst_total": float(j.inst_total),
                    "gpu_total": float(j.gpu_total), "cpu_total": r3(j.cpu_total), "mem_total": r3(j.mem_total),
                    "gpu_type_spec": j.gpu_type_spec, "workload": j.workload, "submit_hour": int(j.submit_hour),
                    "submit_weekday": int(j.submit_weekday), "day": round((j.submit_time - t0) / 86400, 2),
                    "gpu_type_assigned": j.gpu_type_assigned},
        "history": {"earlier_submissions": len(before), "finished_before_submit": len(known),
                    "still_running_at_submit": len(pending), "user_n_finished": float(j.user_n_finished),
                    "user_median_logrt": j.user_median_logrt, "user_fail_rate": j.user_fail_rate,
                    "group_n_finished": j.group_n_finished, "group_median_logrt": j.group_median_logrt,
                    "user_submits_last_1h": float(j.user_submits_last_1h)},
        "prediction": {"point": j[best], "lo": j.q05, "hi": j.q95, "baseline": j.baseline, "truth": j.y,
                       "p_fail": j.p_fail, "p_fail_base": j.p_fail_base},
    })


if __name__ == "__main__":
    df, s = load()
    write_js("site_metrics", "SITE_METRICS", metrics_payload(df, s))
    write_js("site_sample", "SITE_SAMPLE", sample_payload(df))
    write_js("site_grid", "SITE_GRID", grid_payload(df))
    write_js("site_stepper", "SITE_STEPPER", stepper_payload(df))
