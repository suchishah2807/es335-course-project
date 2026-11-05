"""Create the three readable walk-through notebooks from saved results (they never refit models or touch raw test data).

Usage: python scripts/build_notebooks.py && jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
"""
from pathlib import Path

import nbformat as nbf

NB = Path(__file__).resolve().parents[1] / "notebooks"
SETUP = """import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
R = ROOT / "results"
load = lambda n: json.loads((R / n).read_text())
pd.set_option("display.precision", 4)"""


def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c) for c in cells]
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return n


audit = nb([
    "md:# 01 · Data audit (shared)\nAnswers the seven open questions from the proposal with numbers from `src/audit.py`. Figures are in `figures/`.",
    SETUP,
    "a = load('audit.json')\npd.Series(a['job_status_counts'], name='jobs')",
    "md:## Q4: which jobs have a usable runtime label?",
    "pd.DataFrame(a['availability_by_status']).T",
    "md:## Q1/Q2: queue wait and the runtime definition\nThe gap between job submission and the first task start is 0 for every finished job, so runtime = task span = job duration for Terminated jobs.",
    "pd.Series(a['timing_finished_jobs'])",
    "md:## Q5: missingness by status (%)",
    "pd.DataFrame(a['task_missing_by_job_status_%'])",
    "md:## Requested vs assigned GPU type",
    "pd.DataFrame({'gpu_type_spec (requested)': pd.Series(a['gpu_type_spec_values']), 'gpu_type (assigned)': pd.Series(a['gpu_type_values'])})",
    "md:## Q6: drift — failure rate by day",
    "d = pd.DataFrame(a['daily']); d[['day','n','fail_rate']].describe()",
    "md:## Leakage table",
    "pd.read_csv(R / 'leakage_table.csv')",
    "md:## The frozen split",
    "s = load('split.json'); {k: s[k] for k in ['frozen_at_utc','cut_day','test_from_day','trace_end_day','counts']}",
])

runtime = nb([
    "md:# 02 · Runtime (Saanvi)\nBaseline → ridge → gradient-boosted trees → quantile range with a conformal correction. Tuning on walk-forward folds only; test scored once.",
    SETUP,
    "md:## Walk-forward results (development data only)",
    "cv = load('runtime_cv.json'); pd.DataFrame([{'model': r['label'], 'mean MAE': r['mean_mae'], 'mean RMSE': r['mean_rmse']} for r in cv['ridge'] + cv['tree']])",
    "pd.Series(load('runtime_config.json'))",
    "md:## Interval coverage: raw vs conformal (folds)",
    "pd.DataFrame(load('runtime_intervals_cv.json')['folds'])",
    "md:## Final test (run once)",
    "F = load('final_metrics.json'); o = F['runtime']['overall']\npd.DataFrame({'model': o['model'], 'baseline': o['baseline']})",
    "pd.DataFrame(o['ci'], index=['2.5%', '97.5%']).T",
    "md:## Per band, seen/unseen, heavy user",
    "pd.DataFrame({k: {'n': v['model']['n'], 'MAE model': v['model']['mae'], 'MAE baseline': v['baseline']['mae'], 'diff lo': v['ci']['mae_diff'][0], 'diff hi': v['ci']['mae_diff'][1], 'signed error': v['model']['mean_signed_error'], 'coverage': v['model'].get('coverage_90')} for k, v in F['runtime']['slices'].items()}).T",
    "md:## Ablation ladder",
    "pd.DataFrame([{'rung': a['rung'], 'MAE': a['model']['mae'], 'diff lo': a['ci']['mae_diff'][0], 'diff hi': a['ci']['mae_diff'][1]} for a in F['ablation']['runtime']])",
    "md:## Representative errors",
    "e = load('errors.json'); pd.DataFrame([c for c in e['cases'] if c['kind'] == 'runtime'])",
    "pd.DataFrame(e['summary']['runtime_mae_by_true_length'])",
])

failure = nb([
    "md:# 03 · Failure and drift (Suchi)\nBaseline → logistic → gradient-boosted classifier (planned grid + regularised extension) → calibration and threshold. Then per-band drift.",
    SETUP,
    "cv = load('failure_cv.json'); pd.DataFrame([{'model': r['label'], 'mean PR-AUC': r['mean_pr_auc'], 'mean Brier': r['mean_brier']} for r in cv['logistic'] + cv['tree']])",
    "x = load('failure_cv_extra.json'); pd.DataFrame([{'model': r['label'], 'mean PR-AUC': r['mean_pr_auc']} for r in x['extra']])",
    "md:## Calibration check on folds (isotonic fitted on the previous fold)",
    "pd.DataFrame([{k: v for k, v in c.items() if not k.startswith('bins')} for c in cv['calibration']])",
    "pd.Series(load('failure_config.json'))",
    "md:## Final test (run once)",
    "F = load('final_metrics.json'); o = F['failure']['overall']\npd.DataFrame({'model': o['model'], 'baseline': o['baseline']})",
    "pd.DataFrame(o['ci'], index=['2.5%', '97.5%']).T",
    "pd.DataFrame(F['failure']['calibration_bins']['model'])",
    "md:## Drift: per band and per day",
    "pd.DataFrame({k: {'PR-AUC model': v['model']['pr_auc'], 'PR-AUC baseline': v['baseline']['pr_auc'], 'base rate': v['model']['base_rate'], 'mean pred': v['model']['mean_pred']} for k, v in F['failure']['slices'].items()}).T",
    "d = load('drift.json'); pd.DataFrame(d['failure_by_day'])",
    "pd.Series(d['input_shift'])",
])

if __name__ == "__main__":
    NB.mkdir(exist_ok=True)
    for name, n in [("01_audit", audit), ("02_runtime", runtime), ("03_failure", failure)]:
        nbf.write(n, NB / f"{name}.ipynb")
    print("notebooks written")
