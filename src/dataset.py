"""Load the job table with the frozen split attached. Every modelling script starts here."""
import pandas as pd

from config import INTERIM
from features import add_user_id
from split import assign, fold_masks, load_split


def load():
    s = load_split()
    df = pd.read_parquet(INTERIM / "jobs.parquet")
    df = add_user_id(assign(df, s))
    return df, s


def folds(df, s, target):
    """Yield (fold_no, fit_df, val_df) restricted to rows labelled for `target` ('y_fail' or 'y_logrt')."""
    lab = df[target].notna()
    for f in s["folds"]:
        fit, val = fold_masks(df, f)
        yield f["fold"], df[fit & lab], df[val & lab]


def dev_and_test(df, target):
    lab = df[target].notna()
    return df[(df.part == "dev") & lab], df[(df.part == "test") & lab]
