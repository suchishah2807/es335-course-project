"""Feature groups (ablation ladder order) and the train-only categorical preprocessing shared by every model."""
import numpy as np
import pandas as pd

RESOURCES = ["n_tasks", "inst_total", "cpu_total", "mem_total", "gpu_total", "cpu_per_inst_max",
             "mem_per_inst_max", "gpu_per_inst_max", "has_gpu_request", "gpu_type_spec"]
TASK_TYPE = ["main_role", "role_set", "workload"]
HISTORY = ["user_id", "user_n_finished", "user_fail_rate", "user_n_success", "user_median_logrt",
           "user_n_prev_submits", "user_submits_last_1h", "group_n_finished", "group_fail_rate",
           "group_n_success", "group_median_logrt", "group_n_prev_submits", "group_submits_last_1h"]
TIME = ["submit_hour", "submit_weekday"]
LEAKY = ["gpu_type_assigned"]          # sensitivity run only, never headline

LADDER = [("resources", RESOURCES), ("+ task type", TASK_TYPE), ("+ history", HISTORY), ("+ time", TIME)]
ALL = RESOURCES + TASK_TYPE + HISTORY + TIME
CATEGORICAL = ["gpu_type_spec", "main_role", "role_set", "workload", "user_id", "gpu_type_assigned"]
TOP_N = 200


class CategoryEncoder:
    """Fit on TRAIN rows only: keep each categorical's top-N train values, map the rest (and unseen values) to OTHER.
    Produces integer codes (for trees) and the fitted category lists (for one-hot in linear models)."""

    def __init__(self, top_n: int = TOP_N):
        self.top_n = top_n
        self.levels = {}

    def fit(self, df: pd.DataFrame, cols):
        for c in cols:
            vc = df[c].astype(str).value_counts()
            self.levels[c] = list(vc.index[: self.top_n]) + ["OTHER"]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c, lv in self.levels.items():
            if c in out:
                s = out[c].astype(str)
                s = s.where(s.isin(lv[:-1]), "OTHER")
                out[c] = pd.Categorical(s, categories=lv)
        return out


def add_user_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["user_id"] = df["user"]
    return df


def frame(df: pd.DataFrame, cols) -> pd.DataFrame:
    return df[cols]


# heavy-tailed counts/sizes: linear models see log1p of these
HEAVY = ["inst_total", "cpu_total", "mem_total", "gpu_total", "cpu_per_inst_max", "mem_per_inst_max",
         "user_n_finished", "user_n_success", "user_n_prev_submits", "user_submits_last_1h",
         "group_n_finished", "group_n_success", "group_n_prev_submits", "group_submits_last_1h"]


def numeric_cols(cols):
    return [c for c in cols if c not in CATEGORICAL]


def safe_log1p(x):
    return np.log1p(np.clip(x, 0, None))
