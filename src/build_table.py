"""Phase 2: build one row per job with submission-time features, time-safe history features and both labels.

Usage: python src/build_table.py   -> data/interim/jobs.parquet, results/build_report.json

Rules (see DECISIONS.md, Phase 2):
- Only fields known at submission are features. The assigned gpu_type is kept as `gpu_type_assigned`
  for a labelled leakage sensitivity run only.
- History features for job i use only jobs of the same user / group that had FINISHED strictly before
  job i's submit time. A finished job's outcome becomes known at its end time; failed jobs with no
  recorded end become known at submit + 24 h (conservative: later than the true end for 97.2% of
  failed jobs that do record an end).
- Submission-count features (e.g. submits in the last hour) use only earlier SUBMISSIONS, which a
  scheduler sees immediately.
"""
import heapq
import json

import numpy as np
import pandas as pd

from config import INTERIM, RESULTS, TZ

UNKNOWN_FINISH_LAG_S = 24 * 3600


def aggregate_tasks(task: pd.DataFrame) -> pd.DataFrame:
    t = task.copy()
    t["has_gpu"] = t.plan_gpu.notna()
    inst = t.inst_num.fillna(1)
    t["cpu_tot"] = t.plan_cpu / 100 * inst
    t["mem_tot"] = t.plan_mem * inst
    t["gpu_tot"] = t.plan_gpu.fillna(0) / 100 * inst
    t["inst"] = inst
    # main role = task with the largest GPU share, ties broken by instance count
    t = t.sort_values(["job_name", "gpu_tot", "inst"], ascending=[True, False, False])
    g = t.groupby("job_name", sort=False)
    agg = g.agg(n_tasks=("task_name", "size"), inst_total=("inst", "sum"),
                cpu_total=("cpu_tot", lambda s: s.sum(min_count=1)), mem_total=("mem_tot", lambda s: s.sum(min_count=1)),
                gpu_total=("gpu_tot", "sum"), cpu_per_inst_max=("plan_cpu", "max"),
                mem_per_inst_max=("plan_mem", "max"), gpu_per_inst_max=("plan_gpu", "max"),
                has_gpu_request=("has_gpu", "max"), main_role=("task_name", "first"),
                gpu_type_assigned=("gpu_type", "first"),
                task_start_min=("start_time", "min"), task_end_max=("end_time", "max"),
                task_end_na=("end_time", lambda s: int(s.isna().sum())))
    agg["cpu_per_inst_max"] /= 100
    agg["gpu_per_inst_max"] = (agg.gpu_per_inst_max / 100).fillna(0)
    agg["role_set"] = g.task_name.agg(lambda s: "+".join(sorted(s.astype(str))))
    agg["has_gpu_request"] = agg.has_gpu_request.astype(int)
    return agg.reset_index()


def count_before(key, t_query, key_ev, t_ev, weights=None, side="left"):
    """For each query (key, t) return the number (or weighted sum) of events with the same key and t_ev < t.
    Vectorised with a composite sort key."""
    big = 1e8  # timestamps are below 1e8 s
    comp_ev = key_ev.astype(np.float64) * big + t_ev
    order = np.argsort(comp_ev, kind="stable")
    comp_ev = comp_ev[order]
    cum = np.concatenate([[0], np.cumsum(np.ones(len(order)) if weights is None else weights[order])])
    hi = np.searchsorted(comp_ev, key.astype(np.float64) * big + t_query, side=side)
    lo = np.searchsorted(comp_ev, key.astype(np.float64) * big, side="left")
    return cum[hi] - cum[lo]


def running_median_before(key, t_query, key_ev, t_ev, v_ev):
    """Median of event values with the same key and t_ev < t_query. Exact, two-heap streaming per key."""
    n_q = len(key)
    # events: (key, time, kind) with kind 0 = query, 1 = add value, so ties put queries first (strict <)
    keys = np.concatenate([key, key_ev])
    times = np.concatenate([t_query, t_ev])
    kind = np.concatenate([np.zeros(n_q, np.int8), np.ones(len(key_ev), np.int8)])
    idx = np.concatenate([np.arange(n_q), np.arange(len(key_ev))])
    order = np.lexsort((kind, times, keys))
    out = np.full(n_q, np.nan)
    cur, lo, hi = None, [], []  # lo = max-heap (negated), hi = min-heap
    for o in order:
        k = keys[o]
        if k != cur:
            cur, lo, hi = k, [], []
        if kind[o] == 0:
            if lo:
                out[idx[o]] = -lo[0] if len(lo) > len(hi) else (-lo[0] + hi[0]) / 2
        else:
            v = v_ev[idx[o]]
            if not lo or v <= -lo[0]:
                heapq.heappush(lo, -v)
            else:
                heapq.heappush(hi, v)
            if len(lo) > len(hi) + 1:
                heapq.heappush(hi, -heapq.heappop(lo))
            elif len(hi) > len(lo):
                heapq.heappush(lo, -heapq.heappop(hi))
    return out


def history(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    key = df[col].astype("category").cat.codes.to_numpy().astype(np.int64)
    s = df.submit_time.to_numpy()
    fin = df.finished.to_numpy()
    ft = df.known_time.to_numpy()
    out = pd.DataFrame(index=df.index)
    valid = key >= 0
    kq = np.where(valid, key, -1)
    n_fin = count_before(kq, s, key[fin], ft[fin])
    n_failed = count_before(kq, s, key[fin], ft[fin], weights=df.y_fail.to_numpy()[fin])
    term = (df.status == "Terminated").to_numpy()
    out[f"{prefix}_n_finished"] = n_fin
    out[f"{prefix}_fail_rate"] = np.where(n_fin > 0, n_failed / np.maximum(n_fin, 1), np.nan)
    out[f"{prefix}_n_success"] = count_before(kq, s, key[term], ft[term])
    out[f"{prefix}_median_logrt"] = running_median_before(kq, s, key[term], ft[term], df.y_logrt.to_numpy()[term])
    # submissions are visible immediately: count earlier submissions overall and in the last hour
    out[f"{prefix}_n_prev_submits"] = count_before(kq, s, key, s)
    out[f"{prefix}_submits_last_1h"] = out[f"{prefix}_n_prev_submits"] - count_before(kq, s - 3600, key, s)
    out.loc[~valid, :] = np.nan
    return out


def main():
    job = pd.read_parquet(INTERIM / "pai_job_table.parquet")
    task = pd.read_parquet(INTERIM / "pai_task_table.parquet")
    tag = pd.read_parquet(INTERIM / "pai_group_tag_table.parquet")
    rep = {"jobs_in": len(job), "tasks_in": len(task), "tags_in": len(tag)}

    agg = aggregate_tasks(task)
    rep["task_agg_rows"] = len(agg)
    df = job.merge(agg, on="job_name", how="left", validate="one_to_one")
    rep["after_task_join"] = len(df)
    tag = tag.drop(columns="user")
    for c in ["gpu_type_spec", "group", "workload"]:
        tag[c] = tag[c].replace("", np.nan)
    df = df.merge(tag, on="inst_id", how="left", validate="one_to_one")
    rep["after_tag_join"] = len(df)
    rep["jobs_without_tag_row"] = int(df.group.isna().sum())

    rep["dropped_no_submit_time"] = int(df.start_time.isna().sum())
    df = df[df.start_time.notna()].copy()
    df = df.rename(columns={"start_time": "submit_time"})
    df["gpu_type_spec"] = df.gpu_type_spec.fillna("unspecified")
    df["workload"] = df.workload.fillna("untagged")

    # labels
    df["finished"] = df.status.isin(["Terminated", "Failed"])
    df["y_fail"] = np.where(df.status == "Failed", 1.0, np.where(df.status == "Terminated", 0.0, np.nan))
    rt = df.task_end_max - df.task_start_min
    df["runtime_s"] = np.where((df.status == "Terminated") & (df.task_end_na == 0), rt, np.nan)
    df["y_logrt"] = np.log10(np.clip(df.runtime_s, 1, None))
    rep["terminated_runtime_lt_1s"] = int(((df.runtime_s < 1) & df.runtime_s.notna()).sum())

    # when does the outcome become known (for history features)?
    end = df.end_time.fillna(df.task_end_max)
    df["known_time"] = np.where(df.finished, end.fillna(df.submit_time + UNKNOWN_FINISH_LAG_S), np.inf)
    rep["failed_known_time_imputed"] = int((df.finished & end.isna()).sum())

    # time-of-day features (UTC+8 preserved per README)
    loc = pd.to_datetime(df.submit_time, unit="s", utc=True).dt.tz_convert(TZ)
    df["submit_hour"] = loc.dt.hour
    df["submit_weekday"] = loc.dt.weekday
    df["day"] = ((df.submit_time - df.submit_time.min()) // 86400).astype(int)

    df = df.sort_values("submit_time", kind="stable").reset_index(drop=True)
    df = pd.concat([df, history(df, "user", "user"), history(df, "group", "group")], axis=1)

    # sanity: history never uses the job itself or future outcomes
    first = df.groupby("user").head(1)
    rep["first_job_of_user_has_history"] = int((first.user_n_finished > 0).sum())

    keep = ["job_name", "inst_id", "user", "group", "status", "submit_time", "day", "known_time",
            "y_fail", "runtime_s", "y_logrt",
            "n_tasks", "inst_total", "cpu_total", "mem_total", "gpu_total", "cpu_per_inst_max", "mem_per_inst_max",
            "gpu_per_inst_max", "has_gpu_request", "gpu_type_spec", "main_role", "role_set", "workload",
            "submit_hour", "submit_weekday", "gpu_type_assigned"] + \
           [c for c in df.columns if c.startswith(("user_", "group_"))]
    df = df[keep]
    df.to_parquet(INTERIM / "jobs.parquet", index=False)
    rep["rows_out"] = len(df)
    rep["label_counts"] = {"failure_labelled": int(df.y_fail.notna().sum()), "failed": int((df.y_fail == 1).sum()),
                           "runtime_labelled": int(df.y_logrt.notna().sum())}
    rep["missing_%"] = (df.isna().mean() * 100).round(2).to_dict()
    rep["cardinality"] = {c: int(df[c].nunique()) for c in ["user", "group", "main_role", "role_set", "workload",
                                                            "gpu_type_spec", "gpu_type_assigned"]}
    (RESULTS / "build_report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
