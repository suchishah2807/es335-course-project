"""Convert the three raw CSVs to parquet in data/interim/ (faster to reload). No cleaning happens here.

Usage: python src/load_raw.py
"""
from config import INTERIM, read_table

TABLES = ["pai_job_table", "pai_task_table", "pai_group_tag_table"]

if __name__ == "__main__":
    for t in TABLES:
        df = read_table(t)
        df.to_parquet(INTERIM / f"{t}.parquet", index=False)
        print(f"{t}: {len(df):,} rows, columns {list(df.columns)}")
