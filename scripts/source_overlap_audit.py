"""Describe source-exact duplicates without repairing or excluding them."""
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd

rows = []
for path in sorted(Path("outputs/source_exact/runs").glob("*/data.npz")):
    with np.load(path) as data:
        groups = {name: {tuple(row) for row in data[key]} for name, key in (("pretraining", "pre_bases"), ("main", "main_bases"), ("generalization", "gen_bases"))}
        row = {"run_id": int(path.parent.name.split("_")[1])}
        for name, expected in (("pretraining", 10), ("main", 3), ("generalization", 100)):
            row[name + "_duplicate_count"] = expected - len(groups[name])
        for first, second in itertools.combinations(groups, 2):
            row[first + "_" + second + "_exact_matches"] = len(groups[first] & groups[second])
        rows.append(row)
df = pd.DataFrame(rows)
df.to_csv("outputs/source_exact/analysis/sequence_overlap_summary.csv", index=False)
summary = {column: int(df[column].sum()) for column in df if column != "run_id"}
summary["runs_with_generalization_duplicates"] = int((df.generalization_duplicate_count > 0).sum())
summary["runs_with_any_generalization_training_overlap"] = int(((df.pretraining_generalization_exact_matches + df.main_generalization_exact_matches) > 0).sum())
Path("outputs/source_exact/analysis/sequence_overlap_totals.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
