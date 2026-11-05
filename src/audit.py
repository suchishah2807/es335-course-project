"""Phase 1 data audit. Settles the 7 open questions with numbers, written to results/audit.json.

Usage: python src/audit.py      (needs data/interim/*.parquet from src/load_raw.py)
Also writes data/interim/audit_jobs.parquet (one row per job with timing fields) for the plots.
"""
import json

import numpy as np
import pandas as pd

from config import INTERIM, RESULTS, TZ

DAY = 86400


def pct(x):
    return round(100 * float(x), 2)


def clean(o):
    """Make nested dicts JSON-safe (numpy keys/values, NaN -> None)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if o is pd.NA:
        return None
    return o


def main():
    job = pd.read_parquet(INTERIM / "pai_job_table.parquet")
    task = pd.read_parquet(INTERIM / "pai_task_table.parquet")
    tag = pd.read_parquet(INTERIM / "pai_group_tag_table.parquet")
    out = {}

    # ---------- basic counts and keys ----------
    out["rows"] = {"job_table": len(job), "task_table": len(task), "group_tag_table": len(tag)}
    out["keys"] = {
        "job_name_unique_in_job_table": bool(job.job_name.is_unique),
        "inst_id_unique_in_job_table": bool(job.inst_id.is_unique),
        "inst_id_unique_in_group_tag": bool(tag.inst_id.is_unique),
        "jobs_with_any_task_row": pct(job.job_name.isin(task.job_name).mean()),
        "task_rows_whose_job_exists": pct(task.job_name.isin(job.job_name).mean()),
        "jobs_with_group_tag_row": pct(job.inst_id.isin(tag.inst_id).mean()),
        "group_tag_rows_whose_job_exists": pct(tag.inst_id.isin(job.inst_id).mean()),
        "group_tag_user_matches_job_user": pct(
            job.merge(tag[["inst_id", "user"]], on="inst_id", suffixes=("", "_tag"))
               .eval("user == user_tag").mean()),
    }
    out["users"] = int(job.user.nunique())
    span = job.start_time.max() - job.start_time.min()
    out["submit_span_days"] = round(span / DAY, 2)

    # ---------- Q7 status counts ----------
    out["job_status_counts"] = job.status.value_counts(dropna=False).to_dict()
    out["task_status_counts"] = task.status.value_counts(dropna=False).to_dict()

    # ---------- Q3 multi-task structure ----------
    ntask = task.groupby("job_name").size()
    out["tasks_per_job"] = {
        "share_1": pct((ntask == 1).mean()), "share_2": pct((ntask == 2).mean()),
        "share_3plus": pct((ntask >= 3).mean()), "max": int(ntask.max()),
    }
    out["task_name_top"] = task.task_name.value_counts(dropna=False).head(15).to_dict()
    out["task_name_nunique"] = int(task.task_name.nunique())
    multi = task[task.job_name.map(ntask) > 1]
    out["multi_task_role_sets_top"] = (multi.groupby("job_name").task_name
                                       .agg(lambda s: "+".join(sorted(s.astype(str)))).value_counts().head(10).to_dict())
    # do tasks within one job request the same GPU type / differ in resources?
    g = task.groupby("job_name")
    out["multi_task_gpu_type_varies"] = pct((g.gpu_type.nunique() > 1)[ntask > 1].mean())

    # ---------- Q1/Q2/Q4 timing ----------
    agg = g.agg(t_start_min=("start_time", "min"), t_end_max=("end_time", "max"),
                t_start_na=("start_time", lambda s: s.isna().sum()),
                t_end_na=("end_time", lambda s: s.isna().sum()), n_tasks=("task_name", "size"))
    j = job.merge(agg, left_on="job_name", right_index=True, how="left")
    j["queue_wait"] = j.t_start_min - j.start_time
    all_ended = (j.t_end_na == 0) & (j.t_start_na == 0) & j.n_tasks.notna()
    j["exec_time"] = np.where(all_ended, j.t_end_max - j.t_start_min, np.nan)
    j["job_duration"] = j.end_time - j.start_time
    j["job_end_minus_task_end"] = j.end_time - j.t_end_max

    by = j.groupby("status")
    out["availability_by_status"] = pd.DataFrame({
        "n": by.size(),
        "job_end_time_present_%": by.end_time.apply(lambda s: pct(s.notna().mean())),
        "all_task_ends_present_%": by.exec_time.apply(lambda s: pct(s.notna().mean())),
        "exec_time_le0_%": by.exec_time.apply(lambda s: pct((s <= 0).mean())),
        "task_start_present_%": by.t_start_min.apply(lambda s: pct(s.notna().mean())),
    }).T.to_dict()
    fin = j[j.status.isin(["Terminated", "Failed"])]
    out["timing_finished_jobs"] = {
        "queue_wait_negative_%": pct((fin.queue_wait < 0).mean()),
        "queue_wait_median_s": float(fin.queue_wait.median()),
        "queue_wait_p90_s": float(fin.queue_wait.quantile(.9)),
        "exec_time_median_s": float(fin.exec_time.median()),
        "exec_time_p90_s": float(fin.exec_time.quantile(.9)),
        "exec_time_zero_%": pct((fin.exec_time == 0).mean()),
        "job_duration_median_s": float(fin.job_duration.median()),
        "job_end_minus_task_end_median_s": float(fin.job_end_minus_task_end.median()),
        "job_end_equals_task_end_%": pct((fin.job_end_minus_task_end.abs() <= 1).mean()),
        "share_of_job_duration_that_is_wait_median": float((fin.queue_wait / fin.job_duration)
                                                           .replace([np.inf, -np.inf], np.nan).median()),
        "spearman_wait_vs_exec": float(fin[["queue_wait", "exec_time"]].corr("spearman").iloc[0, 1]),
    }
    for s in ["Terminated", "Failed"]:
        e = fin.loc[fin.status == s, "exec_time"].dropna()
        out["timing_finished_jobs"][f"exec_median_{s}"] = float(e.median())

    # ---------- Q5 missingness ----------
    tf = task.merge(job[["job_name", "status", "start_time"]].rename(columns={"status": "job_status", "start_time": "submit"}),
                    on="job_name", how="left")
    fields = ["plan_cpu", "plan_mem", "plan_gpu", "gpu_type", "inst_num", "task_name", "start_time", "end_time"]
    out["task_missing_%"] = {f: pct(task[f].isna().mean()) for f in fields}
    out["task_missing_by_job_status_%"] = {f: tf.groupby("job_status")[f].apply(lambda s: pct(s.isna().mean())).to_dict()
                                           for f in fields}
    tf["day"] = ((tf.submit - job.start_time.min()) // DAY).astype("Int64")
    out["task_missing_plan_gpu_by_day_%"] = tf.groupby("day").plan_gpu.apply(lambda s: pct(s.isna().mean())).to_dict()
    tagm = job[["inst_id", "status"]].merge(tag, on="inst_id", how="left")
    for f in ["gpu_type_spec", "group", "workload"]:
        tagm[f] = tagm[f].replace("", np.nan)
    out["tag_missing_% (over all jobs, incl. no tag row)"] = {f: pct(tagm[f].isna().mean()) for f in ["gpu_type_spec", "group", "workload"]}
    out["tag_missing_by_status_%"] = {f: tagm.groupby("status")[f].apply(lambda s: pct(s.isna().mean())).to_dict()
                                      for f in ["gpu_type_spec", "group", "workload"]}
    out["gpu_type_spec_values"] = tag.gpu_type_spec.replace("", np.nan).value_counts(dropna=False).to_dict()
    out["gpu_type_values"] = task.gpu_type.value_counts(dropna=False).to_dict()
    out["cardinality"] = {"user": int(job.user.nunique()), "group": int(tag.group.nunique()),
                          "workload": int(tag.workload.nunique()), "task_name": int(task.task_name.nunique()),
                          "gpu_type_spec": int(tag.gpu_type_spec.nunique()), "gpu_type": int(task.gpu_type.nunique())}
    # agreement between requested and assigned GPU type (single-task jobs with both present)
    one = task[task.job_name.map(ntask) == 1][["job_name", "gpu_type"]]
    cmp_ = one.merge(job[["job_name", "inst_id"]], on="job_name").merge(tag[["inst_id", "gpu_type_spec"]], on="inst_id")
    cmp_ = cmp_[cmp_.gpu_type_spec.notna() & (cmp_.gpu_type_spec != "") & cmp_.gpu_type.notna()]
    out["gpu_type_equals_spec_when_spec_given_%"] = pct((cmp_.gpu_type == cmp_.gpu_type_spec).mean())
    out["gpu_type_missing_by_status_%"] = out["task_missing_by_job_status_%"]["gpu_type"]

    # ---------- Q6 drift ----------
    out["job_submit_time_missing"] = int(j.start_time.isna().sum())
    out["job_submit_time_missing_by_status"] = j[j.start_time.isna()].status.value_counts().to_dict()
    j["day"] = ((j.start_time - j.start_time.min()) // DAY).astype("Int64")
    j["local"] = pd.to_datetime(j.start_time, unit="s", utc=True).dt.tz_convert(TZ)
    daily = j.groupby("day").agg(n=("status", "size"),
                                 failed=("status", lambda s: (s == "Failed").sum()),
                                 term=("status", lambda s: (s == "Terminated").sum()),
                                 running=("status", lambda s: s.isin(["Running", "Waiting"]).sum()))
    daily["fail_rate"] = daily.failed / (daily.failed + daily.term)
    out["daily"] = daily.reset_index().to_dict("list")
    out["fail_rate_overall_finished"] = float((fin.status == "Failed").mean())
    tn = task.drop_duplicates("job_name").merge(j[["job_name", "day"]], on="job_name")
    tn["week"] = tn.day // 7
    top = tn.task_name.value_counts().head(5).index
    out["task_mix_by_week_%"] = (pd.crosstab(tn.week, tn.task_name.where(tn.task_name.isin(top), "other"),
                                             normalize="index") * 100).round(2).to_dict()
    # ---------- Q7 censoring ----------
    j["status_bucket"] = j.status.where(j.status.isin(["Terminated", "Failed"]), "Running/Waiting")
    out["status_by_day_%"] = (pd.crosstab(j.day, j.status_bucket, normalize="index") * 100).round(2).to_dict()
    late = j[j.day >= j.day.max() - 6]
    out["running_waiting_share_last_7_days_%"] = pct(late.status.isin(["Running", "Waiting"]).mean())
    out["running_waiting_share_first_60_days_%"] = pct(j[j.day < 60].status.isin(["Running", "Waiting"]).mean())
    out["running_waiting_with_end_time_%"] = pct(j[j.status.isin(["Running", "Waiting"])].end_time.notna().mean())

    # users
    upj = j.groupby("user").size()
    out["jobs_per_user"] = {"median": float(upj.median()), "p90": float(upj.quantile(.9)), "max": int(upj.max()),
                            "top10_users_share_%": pct(upj.nlargest(10).sum() / len(j))}

    keep = ["job_name", "inst_id", "user", "status", "start_time", "end_time", "t_start_min", "t_end_max",
            "n_tasks", "queue_wait", "exec_time", "job_duration", "day"]
    j[keep].to_parquet(INTERIM / "audit_jobs.parquet", index=False)
    out = clean(out)
    (RESULTS / "audit.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k not in ("daily", "status_by_day_%", "task_missing_plan_gpu_by_day_%",
                                                                   "task_mix_by_week_%")}, indent=1, default=str))


if __name__ == "__main__":
    main()
