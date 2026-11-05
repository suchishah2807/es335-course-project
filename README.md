# Predicting Runtime and Failure of AI Jobs on a Shared GPU Cluster

ES 335 (Machine Learning, Prof. Udit Bhatia) course project by **Saanvi Mundra** and **Suchi Shah**.

**Question.** For a job submitted to a shared GPU cluster, using only information known at submission, can we predict
(a) how long it will run if it succeeds and (b) whether it will fail, better than simple per-user baselines?
And which information (resources, task type, history, time of day) actually helps?

- Live site: `https://<repo-owner>.github.io/ES335-gpu-job-prediction/` (GitHub Pages from `main` / `docs`)
- Decision log: [`DECISIONS.md`](DECISIONS.md)
- Data: Alibaba PAI GPU cluster trace 2020 (Weng et al., NSDI '22). **Not redistributed here**; fetched by script.

## Repository layout

| Path | What it holds |
|---|---|
| `scripts/fetch_data.py` | Downloads the three trace tables, verifies SHA-256 against the official README, attaches column names |
| `src/config.py` | Paths, seed (`SEED = 335`), time zone |
| `src/load_raw.py` | Raw CSV → parquet (no cleaning) |
| `src/audit.py`, `src/audit_plots.py` | Phase 1 data audit (`results/audit.json`, `figures/01–08`) |
| `src/build_table.py` | One row per job, submission-time features, time-safe history features, labels |
| `src/leakage_table.py` | Column-by-column leakage audit (`results/leakage_table.csv`) |
| `src/split.py` | **The frozen time split** (`results/split.json`), walk-forward folds, test bands |
| `src/features.py`, `src/model_common.py` | Feature groups, train-only category encoder, model wrappers |
| `src/metrics.py` | Metrics, alert cost, paired bootstrap by user |
| `src/baselines.py` | Global and per-user baselines (walk-forward folds) |
| `src/dataset.py` | Loads the job table with the frozen split attached |
| `src/runtime_models.py`, `src/runtime_intervals.py` | Runtime model selection and conformal interval check (Saanvi) |
| `src/failure_models.py`, `src/failure_tree_extra.py` | Failure model selection, calibration, threshold (Suchi) |
| `src/final_test.py` | The single final test run (`--dry`, `--reproduce` modes) |
| `src/split_demo.py` | Random vs time split on development data (site panel 2) |
| `src/drift_analysis.py` | Per-day / per-band drift analysis (Suchi) |
| `src/error_analysis.py` | Rule-based selection of representative errors |
| `src/export_site.py` | Small derived data files for the website (`docs/data/`) |
| `scripts/build_notebooks.py` | Builds the three walk-through notebooks from saved results |
| `notebooks/` | Readable walk-throughs of the audit, runtime and failure results |
| `results/` | Small derived JSON/CSV only |
| `docs/` | The static GitHub Pages site |

## Reproduce

Requires Python 3.11+ (tested on 3.13) and about 3 GB of free disk.

```bash
pip install -r requirements.txt
python scripts/fetch_data.py        # ~150 MB download, checksums verified
cd src
python load_raw.py                  # raw CSV -> data/interim/*.parquet
python audit.py && python audit_plots.py
python build_table.py               # ~10 min (exact streaming medians for history features)
python leakage_table.py
python split.py                     # prints the frozen split; will not overwrite results/split.json
python baselines.py
python runtime_models.py            # walk-forward tuning only (~25 min)
python runtime_intervals.py         # conformal widening check on folds
python failure_models.py            # walk-forward tuning only (~30 min)
python failure_tree_extra.py        # regularised-tree extension on folds
python final_test.py --reproduce    # reruns the frozen test pipeline and compares with results/final_metrics.json
python split_demo.py && python drift_analysis.py && python error_analysis.py
python export_site.py               # writes docs/data/site_*.js
cd .. && python scripts/build_notebooks.py && jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

`final_test.py` without flags is the single real test run. It already ran (see `results/test_run_log.json`) and refuses
to run again, so a fresh clone uses `--reproduce`, which recomputes everything and prints MATCH/DIFFERS without
overwriting the committed results. `final_test.py --dry` checks the code path on development data only.
The whole pipeline takes about 1.5 hours on our laptop (18 CPU threads). Close other memory-heavy programs while it runs.

Then open `docs/index.html` (or serve it: `python -m http.server -d docs 8000`).

**Where the split is defined:** `src/split.py` → `results/split.json` (frozen 2026-11-04 13:51 UTC, commit `51e686b`, before any model was fit).
Earliest 70% of jobs by submit time = development; 24 h embargo; the rest = test. All tuning uses three walk-forward folds inside development.

**Seeds:** `SEED = 335` everywhere (`src/config.py`).

## Data access and licence

The trace is published by Alibaba at <https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020>.
There is no formal licence file; the maintainers allow use for research and study and ask that the NSDI '22 paper be cited.
This repository does not contain any raw data. The site shows only derived results and a small sample of test jobs
with re-numbered (anonymised) identifiers.

## AI-use disclosure

We used Claude Code (Anthropic) as a coding assistant for writing and debugging scripts, drafting documentation,
and building the static site. All modelling decisions, the evaluation design and every reported number were checked
by us against the code and the data; the reasoning is recorded in `DECISIONS.md`. We are responsible for all results.

## Citations

- Q. Weng, W. Xiao, Y. Yu, W. Wang, C. Wang, J. He, Y. Li, L. Zhang, W. Lin, Y. Ding. *MLaaS in the Wild: Workload
  Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters.* USENIX NSDI 2022.
- F. Pedregosa et al. *Scikit-learn: Machine Learning in Python.* JMLR 12, 2011.
- pandas, NumPy, Matplotlib (data handling and figures).
