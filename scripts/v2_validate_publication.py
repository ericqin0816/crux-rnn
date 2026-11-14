"""Independent checks of published numbers, artifacts and preserved v1 bytes."""

import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from cruxrnn.reproducibility import file_hash, seed, utc_now, write_json


def main():
    baseline = json.loads(Path("provenance/v2_preservation_baseline.json").read_text())
    for name, checksum in baseline["preserved_files"].items():
        if file_hash(name) != checksum:
            raise AssertionError("Preserved v1 artifact changed: " + name)
    summary = pd.read_csv("results/v2/main_summary.csv")
    paired = pd.read_csv("results/v2/paired_differences.csv")
    matrix = paired.pivot(index="run_id", columns="distance", values="difference_pp")
    assert matrix.shape == (100, 7) and len(paired) == 700 and not matrix.isna().any().any()
    for row in summary.itertuples():
        values = matrix[row.distance].to_numpy()
        np.testing.assert_allclose(
            [row.difference_pp, row.median_pp, row.sd_difference_pp],
            [values.mean(), np.median(values), values.std(ddof=1)],
            atol=1e-12,
        )
    interactions = json.loads(Path("results/v2/interactions.json").read_text())
    for label, columns in (("original_four", [1, 2, 4, 7]), ("all_distances", list(range(1, 8)))):
        with np.load(Path("outputs/v2/main/analysis") / f"bootstrap_{label}.npz") as saved:
            assert saved["indices"].shape == (10000, 100)
            x = np.asarray(columns, float)
            x -= x.mean()
            slopes = matrix[columns].to_numpy() @ x / (x @ x)
            recalculated = slopes[saved["indices"]].mean(axis=1)
            np.testing.assert_allclose(recalculated, saved["slopes"], atol=1e-12)
            np.testing.assert_allclose(
                np.percentile(recalculated, [2.5, 97.5]),
                [interactions[label]["ci_low"], interactions[label]["ci_high"]],
                atol=1e-12,
            )
    english = Path("README.md").read_text(encoding="utf-8")
    chinese = Path("README.zh-CN.md").read_text(encoding="utf-8")
    number_pattern = (
        r"^\| (\d+) \| (\d+) \| ([-\d.]+) \| ([-\d.]+) \| ([-\d.]+) \| ([-\d.]+) \| ([-\d.]+) \|$"
    )
    en = re.findall(number_pattern, english, re.M)
    zh = re.findall(number_pattern, chinese, re.M)
    assert len(en) == 4 and en == zh
    for values in en:
        k = int(values[0])
        row = summary[summary.distance == k].iloc[0]
        np.testing.assert_allclose(
            np.array(values[1:], float),
            [
                row.n,
                row.blocked_pct,
                row.interleaved_pct,
                row.difference_pp,
                row.ci_low_pp,
                row.ci_high_pp,
            ],
            atol=0.00051,
        )
    documents = (
        [Path("README.md"), Path("README.zh-CN.md"), Path("CONTRIBUTING.md"), Path("NOTICE.md")]
        + list(Path("docs").glob("*.md"))
        + [Path("reports/v2_research_report.md")]
    )
    missing = []
    for path in documents:
        text = path.read_text(encoding="utf-8")
        refs = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text) + re.findall(
            r'<img[^>]+src="([^"]+)"', text
        )
        for ref in refs:
            ref = ref.strip("<>").split("#")[0]
            if not ref or re.match(r"[a-zA-Z]+:", ref):
                continue
            if not (path.parent / ref).exists():
                missing.append((path.as_posix(), ref))
    if missing:
        raise AssertionError("Missing local document links: " + str(missing))
    figure_manifest = json.loads(Path("results/v2/figures/figure_manifest.json").read_text())
    for name, checksum in figure_manifest["figures"].items():
        path = Path("results/v2/figures") / name
        assert file_hash(path) == checksum
        if path.suffix == ".svg":
            ET.parse(str(path))
        else:
            with Image.open(path) as image:
                assert image.width > 1000 and image.height > 500
                assert abs(image.info["dpi"][0] - 300) < 1
    assert len(figure_manifest["figures"]) == 24
    ET.parse("docs/assets/cruxrnn-logo.svg")
    data = Path("docs/demo/data.js").read_text(encoding="utf-8")
    payload = json.loads(data.split("window.CRUX_DATA = ", 1)[1].rstrip(";\n"))
    assert len(payload["main"]) == 7
    for item in payload["main"]:
        row = summary[summary.distance == item["distance"]].iloc[0]
        np.testing.assert_allclose(
            [
                item["blocked"],
                item["interleaved"],
                item["difference"],
                item["ci_low"],
                item["ci_high"],
            ],
            [
                row.blocked_pct,
                row.interleaved_pct,
                row.difference_pp,
                row.ci_low_pp,
                row.ci_high_pp,
            ],
            atol=1e-12,
        )
    assignments = defaultdict(list)
    for path in Path("outputs/v2/main/runs").glob("rnn_h7_*/manifest.json"):
        manifest = json.loads(path.read_text())
        job = manifest["recipe"]["job"]
        for stream, value in manifest["seeds"].items():
            assignments[value].append((job["run_id"], job["distance"], stream))
    collisions = [items for items in assignments.values() if len(items) > 1]
    report = {
        "verified_at_utc": utc_now(),
        "preserved_v1_files_verified": len(baseline["preserved_files"]),
        "main_pair_matrix": [100, 7],
        "paired_rows": 700,
        "independently_recalculated_mean_median_sd": True,
        "whole_ID_bootstrap_slopes_and_intervals_verified": True,
        "bilingual_primary_tables_equal": True,
        "missing_local_links": missing,
        "verified_figure_files": 24,
        "offline_main_values_match_saved_csv": True,
        "distinct_primary_seed_assignments": len(assignments),
        "primary_seed_collisions": collisions,
        "analysis_seed_derived_from_frozen_namespace": seed(
            "cruxrnn_v2_main_20261114", "analysis", 0, "bootstrap_analysis"
        ),
    }
    write_json("results/v2_validation/publication_checks.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
