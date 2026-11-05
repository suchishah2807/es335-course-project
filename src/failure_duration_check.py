"""How safe is the 24 h history rule? Duration of Failed jobs that DO record an end time (Phase 2 check).

Usage: python src/failure_duration_check.py  -> results/failure_duration_check.json
"""
import json

import pandas as pd

from config import INTERIM, RESULTS

if __name__ == "__main__":
    j = pd.read_parquet(INTERIM / "audit_jobs.parquet", columns=["status", "start_time", "end_time", "t_end_max", "user"])
    f = j[j.status == "Failed"]
    end = f.end_time.fillna(f.t_end_max)
    has = end.notna()
    dur = (end - f.start_time)[has]
    heavy = j.user.value_counts().index[0]
    out = {"failed_jobs": int(len(f)), "with_end_time": int(has.sum()), "with_end_time_%": round(100 * has.mean(), 2),
           "share_longer_than_%": {f"{h}h": round(100 * float((dur > h * 3600).mean()), 2) for h in (1, 6, 12, 24, 48, 72)},
           "count_longer_than": {f"{h}h": int((dur > h * 3600).sum()) for h in (1, 6, 12, 24, 48, 72)},
           "duration_quantiles_h": {str(q): round(float(dur.quantile(q)) / 3600, 2) for q in (0.5, 0.9, 0.95, 0.99)},
           "heavy_user_share_with_end_%": round(100 * float((f[has].user == heavy).mean()), 2),
           "heavy_user_share_without_end_%": round(100 * float((f[~has].user == heavy).mean()), 2),
           "longer_than_24h_excluding_heavy_user_%": round(100 * float((dur[f[has].user != heavy] > 86400).mean()), 2)}
    (RESULTS / "failure_duration_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
