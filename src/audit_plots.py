"""Phase 1 audit figures: eight plots, each answering one question. Writes figures/01_*.png ... 08_*.png
and results/audit_figures.json (the numbers behind each figure, for the site's text alternatives).

Usage: python src/audit_plots.py   (after src/audit.py)
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotstyle as ps
from config import FIGURES, INTERIM, RESULTS

ps.apply()
A = json.loads((RESULTS / "audit.json").read_text())
J = pd.read_parquet(INTERIM / "audit_jobs.parquet")
T = pd.read_parquet(INTERIM / "pai_task_table.parquet")
alt = {}

# 1. Which jobs have a usable runtime label? -------------------------------------------------
av = A["availability_by_status"]
order = ["Terminated", "Failed", "Running", "Waiting"]
vals = [av[s]["all_task_ends_present_%"] for s in order]
ns = [int(av[s]["n"]) for s in order]
fig, ax = plt.subplots(figsize=(6.4, 2.6))
ax.barh(order[::-1], vals[::-1], color=ps.MODEL, height=0.55)
for y, (v, n) in enumerate(zip(vals[::-1], ns[::-1])):
    ax.text(v + 1.5, y, f"{v:.1f}%  (n = {n:,})", va="center", fontsize=8.5, color=ps.INK)
ax.set_xlim(0, 125); ax.set_xticks([0, 25, 50, 75, 100]); ax.grid(axis="y", visible=False)
ax.set_xlabel("Jobs whose every task records an end time (%)")
ax.set_title("Q4. Only successful jobs have a measurable runtime")
ps.takeaway(fig, "Failed jobs almost never record task end times, so the runtime model is scoped to Terminated jobs.")
ps.save(fig, FIGURES / "01_runtime_label_by_status.png")
alt["01"] = dict(zip(order, vals))

# 2. Job volume per day ----------------------------------------------------------------------
d = pd.DataFrame(A["daily"])
fig, ax = plt.subplots(figsize=(6.4, 2.6))
ax.plot(d.day, d.n / 1000, color=ps.MODEL)
ax.set_xlabel("Day of trace (submit time, shifted dates)"); ax.set_ylabel("Jobs submitted (thousands)")
ax.set_ylim(0, None); ax.set_title("Q6a. Job volume per day varies several-fold")
ps.takeaway(fig, f"Median {d.n.median():,.0f} jobs/day, range {d.n.min():,}–{d.n.max():,}; weekly rhythm and bursts.")
ps.save(fig, FIGURES / "02_jobs_per_day.png")
alt["02"] = d[["day", "n"]].to_dict("list")

# 3. Failure rate per day --------------------------------------------------------------------
fr = A["fail_rate_overall_finished"]
dd = d[(d.failed + d.term) >= 200]
fig, ax = plt.subplots(figsize=(6.4, 2.6))
ax.plot(dd.day, dd.fail_rate * 100, color=ps.ACCENT2)
ax.axhline(fr * 100, color=ps.BASELINE, lw=1)
ax.text(dd.day.max(), fr * 100 + 1, f"overall {fr*100:.1f}%", ha="right", fontsize=8, color=ps.INK2)
ax.set_ylim(0, None); ax.set_xlabel("Day of trace"); ax.set_ylabel("Failed / (Failed + Terminated) (%)")
ax.set_title("Q6b. The failure rate drifts across the trace")
ps.takeaway(fig, "Days with fewer than 200 finished jobs omitted. A time split is needed so drift shows up in the test score.")
ps.save(fig, FIGURES / "03_failure_rate_per_day.png")
alt["03"] = {"day": dd.day.tolist(), "fail_rate_%": (dd.fail_rate * 100).round(2).tolist()}

# 4. Runtime distribution --------------------------------------------------------------------
rt = J.loc[J.status == "Terminated", "exec_time"].dropna()
lr = np.log10(rt.clip(lower=1))
fig, ax = plt.subplots(figsize=(6.4, 2.6))
ax.hist(lr, bins=80, color=ps.MODEL, edgecolor="white", linewidth=0.3)
for s, lab in [(60, "1 min"), (3600, "1 h"), (86400, "1 day")]:
    ax.axvline(np.log10(s), color=ps.BASELINE, lw=0.8)
    ax.text(np.log10(s), ax.get_ylim()[1] * 0.95, " " + lab, fontsize=8, color=ps.INK2, va="top")
ax.set_xlabel("Runtime, log10 seconds (Terminated jobs)"); ax.set_ylabel("Jobs")
ax.set_title("Q1. Runtime spans six orders of magnitude, so we model log10 seconds")
ps.takeaway(fig, f"Median {rt.median():,.0f} s; 90th percentile {rt.quantile(.9):,.0f} s; skew in raw seconds is extreme.")
ps.save(fig, FIGURES / "04_runtime_distribution.png")
alt["04"] = {"median_s": float(rt.median()), "p10_s": float(rt.quantile(.1)), "p90_s": float(rt.quantile(.9)),
             "max_s": float(rt.max()), "skew_raw": float(rt.skew()), "skew_log10": float(lr.skew())}

# 5. Missingness by field and status ---------------------------------------------------------
fields = ["plan_cpu", "plan_mem", "plan_gpu", "gpu_type", "inst_num", "start_time", "end_time"]
m = pd.DataFrame(A["task_missing_by_job_status_%"]).loc[order, fields]
tagm = pd.DataFrame(A["tag_missing_by_status_%"]).loc[order, ["gpu_type_spec", "workload"]]
m = pd.concat([m, tagm], axis=1)
fig, ax = plt.subplots(figsize=(7.2, 2.4))
ax.imshow(m.values, cmap="Blues", vmin=0, vmax=110, aspect="auto")
ax.set_xticks(range(m.shape[1]), m.columns, rotation=30, ha="right"); ax.set_yticks(range(4), order)
for i in range(m.shape[0]):
    for k in range(m.shape[1]):
        v = m.values[i, k]
        ax.text(k, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white" if v > 60 else ps.INK)
ax.grid(False); ax.set_title("Q5. Missing values (%) depend strongly on job status")
ps.takeaway(fig, "gpu_type_spec is empty for ~98% (no type requested). end_time is missing for failed/running jobs.")
ps.save(fig, FIGURES / "05_missingness_by_status.png")
alt["05"] = m.round(2).to_dict()

# 6. Unfinished share by submit day (censoring) ----------------------------------------------
sb = pd.DataFrame(A["status_by_day_%"])
fig, ax = plt.subplots(figsize=(6.4, 2.6))
ax.plot(sb.index.astype(int), sb["Running/Waiting"], color=ps.MODEL)
ax.set_ylim(0, None); ax.set_xlabel("Day of trace (submit)"); ax.set_ylabel("Still Running/Waiting (%)")
ax.set_title("Q7. Unfinished jobs are spread across the trace, not piled at the end")
ps.takeaway(fig, f"Last 7 days {A['running_waiting_share_last_7_days_%']}% vs first 60 days "
                 f"{A['running_waiting_share_first_60_days_%']}%: these are not simply censored by the trace end.")
ps.save(fig, FIGURES / "06_unfinished_by_day.png")
alt["06"] = sb["Running/Waiting"].round(2).to_dict()

# 7. Task-type mix by week -------------------------------------------------------------------
mix = pd.DataFrame(A["task_mix_by_week_%"])
mix.index = mix.index.astype(int)
cols = [c for c in mix.sum().sort_values(ascending=False).index if c != "other"] + ["other"]
mix = mix[cols]
fig, ax = plt.subplots(figsize=(6.4, 2.8))
bottom = np.zeros(len(mix))
for c, col in zip(cols, ps.CATS[:len(cols) - 1] + [ps.CATS[-1]]):
    ax.bar(mix.index, mix[c], bottom=bottom, color=col, label=c, width=0.85, edgecolor="white", linewidth=1)
    bottom += mix[c].values
ax.set_ylim(0, 100); ax.set_xlabel("Week of trace"); ax.set_ylabel("Share of jobs (first task) (%)")
ax.legend(ncol=6, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, 1.13)); ax.grid(axis="x", visible=False)
ax.set_title("Q6c. The task-type mix shifts week to week", pad=22)
ps.takeaway(fig, "The workload mix is not stationary: another reason to evaluate on the latest weeks only.")
ps.save(fig, FIGURES / "07_task_mix_by_week.png")
alt["07"] = mix.round(2).to_dict()

# 8. Jobs per user concentration -------------------------------------------------------------
upj = J.groupby("user").size().sort_values(ascending=False)
cum = upj.cumsum() / upj.sum() * 100
fig, ax = plt.subplots(figsize=(6.4, 2.6))
ax.plot(np.arange(1, len(cum) + 1), cum.values, color=ps.MODEL)
ax.set_xscale("log"); ax.set_ylim(0, 100)
ax.set_xlabel("Users ranked by job count (log scale)"); ax.set_ylabel("Cumulative share of jobs (%)")
ax.set_title("Users are very unequal: a few users submit most jobs")
ps.takeaway(fig, f"Top 10 of {len(upj):,} users submit {A['jobs_per_user']['top10_users_share_%']}% of jobs: "
                 "so CIs are bootstrapped by user, not by job.")
ps.save(fig, FIGURES / "08_jobs_per_user.png")
alt["08"] = {"top1_%": float(cum.iloc[0]), "top10_%": float(cum.iloc[9]), "top100_%": float(cum.iloc[99]), "users": len(upj)}

(RESULTS / "audit_figures.json").write_text(json.dumps(alt, indent=1, default=float))
print("wrote 8 figures")
