"""Build the two contributor bundles and the verification manifest from the tested staging repo.

Usage: python scripts/make_handoff.py
  -> handoff/skeleton.zip, handoff/saanvi_bundle.zip, handoff/suchi_bundle.zip (gitignored; send each to its owner)
  -> HANDOFF_MANIFEST.txt (git blob hash + path for every file that main should contain after both PRs merge)

Check a merged clone with:  python scripts/make_handoff.py --verify
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "handoff"

# Files pushed by Suchi in the first commit on main (repo skeleton) — she owns the repository.
SKELETON = [".gitignore", ".gitattributes", "DECISIONS.md", "HANDOFF_MANIFEST.txt", "scripts/make_handoff.py"]

SAANVI = [  # runtime task + her half of the shared pipeline
    "README.md", "requirements.txt", "scripts/fetch_data.py", "scripts/build_notebooks.py",
    "src/config.py", "src/load_raw.py", "src/audit.py", "src/build_table.py", "src/features.py", "src/model_common.py",
    "src/metrics.py", "src/dataset.py", "src/baselines.py",
    "src/runtime_models.py", "src/runtime_intervals.py", "src/error_analysis.py", "src/export_site.py",
    "src/lag_sensitivity.py", "results/lag_sensitivity.json",
    "results/audit.json", "results/build_report.json", "results/baselines_cv.json",
    "results/runtime_cv.json", "results/runtime_config.json", "results/runtime_intervals_cv.json",
    "results/errors.json", "results/error_notes.json",
    "notebooks/01_audit.ipynb", "notebooks/02_runtime.ipynb",
    "docs/css/site.css", "docs/js/charts.js", "docs/vendor/",
    "docs/data/site_grid.js", "docs/data/site_sample.js", "docs/data/site_stepper.js",
]
SUCHI = [  # failure task + drift + her half of the shared pipeline
    "src/plotstyle.py", "src/audit_plots.py", "src/leakage_table.py", "src/split.py", "src/final_test.py",
    "src/split_demo.py", "src/drift_analysis.py", "src/failure_models.py", "src/failure_tree_extra.py",
    "src/failure_duration_check.py", "results/failure_duration_check.json", "results/reproduction_check.json",
    "results/audit_figures.json", "results/leakage_table.csv", "results/split.json", "results/failure_cv.json",
    "results/failure_config.json", "results/failure_cv_extra.json", "results/final_metrics.json",
    "results/test_run_log.json", "results/drift.json", "results/split_demo.json",
    "notebooks/03_failure.ipynb", "figures/", "docs/figures/",
    "docs/index.html", "docs/js/app.js", "docs/data/site_metrics.js", "docs/data/leakage_table.json",
]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def tracked():
    return {line.split("\t")[1]: line.split()[1] for line in git("ls-files", "-s").splitlines()}


def expand(entries, files):
    out = set()
    for e in entries:
        out |= {f for f in files if f == e or (e.endswith("/") and f.startswith(e))}
    return out


def main():
    files = tracked()
    groups = {"skeleton": expand(SKELETON, files), "saanvi": expand(SAANVI, files), "suchi": expand(SUCHI, files)}
    allg = set().union(*groups.values())
    missing = sorted(set(files) - allg - {"HANDOFF_MANIFEST.txt"})
    overlap = (groups["saanvi"] & groups["suchi"]) | (groups["skeleton"] & (groups["saanvi"] | groups["suchi"]))
    if missing or overlap:
        sys.exit(f"Unassigned files: {missing}\nAssigned twice: {sorted(overlap)}")
    if "--verify" in sys.argv:
        want = dict(l.split("  ", 1)[::-1] for l in (ROOT / "HANDOFF_MANIFEST.txt").read_text().splitlines() if l.strip())
        bad = [p for p, h in want.items() if files.get(p) != h]
        extra = sorted(set(files) - set(want) - {"HANDOFF_MANIFEST.txt"})
        print("VERIFIED: main matches the tested staging repo" if not bad and not extra else f"MISMATCH: {bad[:20]} extra: {extra[:20]}")
        return
    (ROOT / "HANDOFF_MANIFEST.txt").write_text("".join(f"{files[p]}  {p}\n" for p in sorted(files) if p != "HANDOFF_MANIFEST.txt"))
    OUT.mkdir(exist_ok=True)
    for who in ("skeleton", "saanvi", "suchi"):
        z = OUT / (f"{who}.zip" if who == "skeleton" else f"{who}_bundle.zip")
        git("archive", "--format=zip", "-o", str(z), "HEAD", "--", *sorted(groups[who]))
        print(f"{z.name}: {len(groups[who])} files")
    print(f"skeleton: {len(groups['skeleton'])} files; manifest: {len(files)} entries")


if __name__ == "__main__":
    main()
