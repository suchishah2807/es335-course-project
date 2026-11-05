"""Shared paths and constants. Every script imports from here so seeds and locations stay in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"      # big derived tables (parquet), gitignored
RESULTS = ROOT / "results"               # small derived JSON/CSV, committed
FIGURES = ROOT / "figures"               # audit plots, committed
SITE_DATA = ROOT / "docs" / "data"       # JSON consumed by the website

SEED = 335
TZ = "Asia/Shanghai"                     # trace timestamps keep UTC+8 time-of-day and weekday

for d in (INTERIM, RESULTS, FIGURES, SITE_DATA):
    d.mkdir(parents=True, exist_ok=True)


def read_table(name: str):
    """Read one raw CSV with its .header column names."""
    import pandas as pd
    cols = (RAW / f"{name}.header").read_text().strip().split(",")
    return pd.read_csv(RAW / f"{name}.csv", header=None, names=cols, low_memory=False)
