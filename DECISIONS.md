# Decision log

What we decided while building this project, why, and what went wrong along the way. Newest entries are at the bottom. When we changed our minds, we added a note instead of deleting the old line.

Project: predicting how long a GPU job runs and whether it fails, using the Alibaba PAI GPU trace (2020). ES 335, Prof. Udit Bhatia.

## Choosing the project (Oct 2026)

- We wanted a current, AI-related topic, so we skipped finance, call-centre and credit-card datasets.
- Shortlist: Azure VM, PM100, Alibaba PAI GPU and the Parallel Workloads Archive. We dropped the archive (data from the 1990s) and Azure (less interesting), then picked Alibaba over PM100 because it is real AI workloads and supports two tasks. PM100 had a cleaner licence, which we gave up.
- Two tasks from the same job request: how long it runs (regression) and whether it fails (classification). The two models are independent.
- One row = one job, predicted at the moment it is submitted, using only what is known then.
- Saanvi owns runtime, Suchi owns failure and the drift analysis. The shared pipeline we built together.

## First plan

- Join the job, task and group tables into one row per job.
- Use only fields known at submission. Anything known after the job ends (end time, status) would leak the answer.
- Main model: gradient-boosted trees, since the data mixes numbers, categories and missing values. A simple model (ridge for runtime, logistic regression for failure) as a check. We only keep the trees if they clearly win.
- Baselines: each user's past median runtime and each user's past failure rate. Our models have to beat "this user's next job will look like their last ones".
- Split by time, never randomly. In real use a model only has the past.
- Tune only on folds inside the training period. Use the test period once, at the end.
- Metrics: MAE on log10 seconds for runtime, PR-AUC for failure (accuracy is misleading when most jobs succeed).
- Jobs from the same user are related, so confidence intervals resample users, not single jobs.
- Seed 335 everywhere.
- We do not re-host the trace. It has no formal licence; research use is allowed if the NSDI 2022 paper is cited.

## Data audit (4 Nov)

- Checksums matched Alibaba's README. 1,055,501 jobs, 1,433 users, about 69 days.
- Status: 732,355 ended normally, 256,555 failed, 62,928 still running, 3,663 waiting.
- **Queue wait is always 0.** The README describes a wait before the job starts, but in this data it is 0 for every finished job. So we dropped it and defined runtime as last task end minus first task start.
- **Failed jobs have no usable runtime.** Only 0.38% of failed jobs have an end time on every task (10% have a job-level end time), against 100% of successful jobs. So the runtime model trains on successful jobs only and answers "how long will it run if it works". Failure risk is the other model's job.
- **The GPU type column is filled in after scheduling.** The requested type is blank for 98% of jobs, and the filled-in type is what the scheduler assigned. We use only the requested type, with "unspecified" as a value. We tested the assigned type once, separately, to see how much it would leak.
- Missing GPU request means "no GPU asked for", so we kept it as a signal instead of filling it in.
- 82% of jobs have one task. Most multi-task jobs are a parameter server plus workers, so we sum resources over tasks instead of making a column per task.
- The daily failure rate swings from about 9% to 77% (ignoring the first 5 near-empty days).
- **One user sent 27% of all jobs**, including a burst on days 59 and 60 where 93% of their jobs failed. The 70/30 split puts this burst in the test period. We kept the split anyway, since moving it after seeing the data would be cheating. Instead we planned to report results with and without that user.
- Running and waiting jobs are dropped, because we don't know how they end. Waiting jobs also have no submit time.

## Building the table (4 Nov)

- Row counts stay at 1,055,501 through every join. After dropping jobs with no submit time: 988,910 jobs for failure, 732,355 for runtime.
- History features (user's past runtime, past failure rate, and the same for the job's script group) use only jobs that had ended before this one was submitted. Checked by hand on 300 random jobs: no mismatches.
- **When is a failure "known"?** 90% of failed jobs never record an end time. We treat them as known 24 hours after submission, because most failures that do record an end finish sooner. (We first wrote 93% here; the right figure is 97.2%. See the 24-hour check below.)
- Only the 200 busiest users get their own category; the rest share "OTHER". Script groups (314,487 of them) are used only through their history.
- Anything learned from data (category lists, fill-in values, scaling) is fitted on training rows only.

## The split (4 Nov, frozen at 13:51 UTC before any model was trained)

- Cut at the 70th percentile of submit time (day 55.16), then a 24-hour gap, then the test period (days 56 to 69).
- The gap stops a long job, or near-copies sent minutes apart, from landing on both sides.
- Development: 685,529 jobs with a failure label, 545,646 with a runtime label. Test: 285,306 and 171,672.
- Three walk-forward folds inside development (cuts at days 34, 41, 48) for all tuning.
- The test period is split into 4 bands to see whether the models get worse over time.
- Alert cost: a missed failure costs 3, a false alarm costs 1. That gives a theoretical threshold of 0.25. The real threshold is picked on the folds.
- Planned checks, never used to choose anything: the assigned GPU type, the heavy user left out, and a random split for comparison.

## Baselines (folds only)

- Runtime: one median for everyone gives MAE 0.766; each user's past median gives 0.569.
- Failure: random guessing scores 0.194; each user's past failure rate gives PR-AUC 0.494. A smoothed version was no better.

## Models (folds only, test never touched)

- **Runtime:** ridge 0.352, trees with squared loss about 0.317, trees with absolute loss 0.2995. We chose absolute loss because MAE rewards the median, and that loss aims at the median. The tree beat ridge on every fold, so it stays.
- **Problem: the 90% ranges were too narrow.** They caught only about 79% of next-week jobs, because next week never looks exactly like the training weeks. Fix: measure the miss on the latest fold and widen both ends by that amount (0.183 log10, about 1.5x).
- **Failure:** logistic regression 0.709. Bigger trees did worse (0.644 to 0.676); they learned one-off bursts that didn't repeat. A small regularised tree scored 0.7105, a tie, so we kept the simpler logistic model.
- Recalibrating the probabilities helped on one fold and hurt on another, so we didn't use it.
- The best alert threshold on the folds was 0.17, not 0.25. The model under-predicts risk when failures rise.
- We dry-ran the whole pipeline on the last fold first, to catch bugs before touching the test set.

## The test run (4 Nov, 14:56 to 15:12 UTC, run once)

- **Runtime:** MAE 0.338 vs 0.601 for the baseline, typically off by 2.2x instead of 4.0x. Interval for the difference: −0.324 to −0.176. Ridge got 0.385.
- **Ranges:** 75.6% coverage before widening, 88.6% after. Only 72.5% for the heaviest user.
- **Failure:** PR-AUC 0.923 vs 0.542 (random guessing 0.398). Interval: +0.089 to +0.419. At threshold 0.17, 77% of alerts were real and 90% of failures were caught.
- **The heavy user dominates the failure score.** They are 48% of the test jobs. Without them: 0.614 vs 0.475. We first called that "the representative number", but we chose that framing after seeing results, so we dropped it. The full test set is the main result, and the heavy-user split is a side check.
- **Drift:** failure PR-AUC drops in the last two bands (0.58 and 0.51), mostly because failures were rarer then.
- **What helps:** the request alone doesn't beat the runtime baseline (0.622). Adding history brings it to 0.352. For failure, history gives the big jump (0.774 to 0.924). Time of day barely matters.
- Adding the assigned GPU type changed runtime MAE only from 0.338 to 0.332, so leaving it out costs almost nothing.
- The tree we didn't pick scored 0.938 on test. We didn't switch, because choosing on the test would turn it into a tuning set.
- New users: runtime MAE 0.350 vs 0.402. Only 1.45% of test jobs, so this is rough.
- A random split would have looked much better (runtime MAE 0.203, failure 0.838, against 0.331 and 0.702 with the time split).
- The 12 error cases were picked by fixed rules in code, not by hand. We guessed burst jobs might be short; the data said no (median 21 vs 11 min), so we left that miss unexplained.
- The model guesses the longest jobs too short (about 13x for jobs over 28 hours), and does worse than the baseline on rare GPU types.

## Reproducing without reusing the test

- `final_test.py --reproduce` reruns the frozen pipeline and compares with our saved numbers. It doesn't overwrite anything, so the test still counts as used once.
- A rerun from a fresh copy matched 1,661 of 1,666 numbers. The 5 that differed were all for the tree we didn't pick: we had edited one of its settings by hand. The script now writes that setting itself, and the rerun matches.

## Checking the 24-hour rule

- Of failures that do record an end, 2.77% ran longer than 24 hours. So the correct figure is 97.2%, not the 93% we first wrote. It also means a few failures may enter a user's history slightly early.
- We can't check the other 90%. The heavy user is 54% of those, so the measured jobs may not be typical.
- We tried 72 hours instead. The model's lead over the baseline stayed about the same (+0.215 vs +0.221 on the folds). We kept 24 hours, since a longer wait also throws away information that really was known.

## Website

- Each interaction explains one result: the alert threshold slider, the heavy-user toggle, time vs random split, the boosting step-through, following one job through the failure model, the feature ladder, the per-day view, and a box for checking reproduced numbers.
- The scheduler example uses 8 made-up jobs and says so. We built predictions, not a scheduler.

## What we concluded

- Both models beat the per-user baseline on the later test period.
- A user's past jobs carry almost all the useful information.
- How we split the data mattered more than which model we used.
- A simple model was enough for failure; trees helped for runtime.
- Ranges learned on past weeks need widening before they work on new weeks.
- Limits: one cluster, 69 days, 2020. Runtime only for successful jobs. The 24-hour rule can't be checked for most failures. The 3 to 1 cost is our assumption. The models find patterns, not causes.