"""Curate only the approved fresh confirmation and write reports from verified data.

This script does not train models or change statistical methods. Large per-update
CSVs are stored as lossless gzip files; original arrays/checkpoints stay local.
"""
import datetime
import gzip
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from motor_rnn.experiment import verify_done
from motor_rnn.provenance import file_hash, implementation_hash, write_json

CONFIG = Path("configs/final_proposed.json")
EXPECTED = "26f8cc56ad115f9e4bb4bb47b30af4997c205cd2906cb8e5195bf24fda8ff006"
PREFINAL = "fa3b7c34496274569944b85c74d2c6c4cd443fe4"
TITLE = "CruxRNN: Sequence Similarity, Practice Scheduling, and Interference in a Minimal Recurrent Neural Network"


def copy(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(source), str(destination))


def table(frame, columns=None, decimals=3):
    frame = frame[columns] if columns else frame
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return "NaN" if np.isnan(value) else ("{:." + str(decimals) + "f}").format(value)
        return str(value)
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    rows.extend("| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def main():
    cfg = json.loads(CONFIG.read_text())
    assert file_hash(CONFIG) == EXPECTED
    assert implementation_hash() == cfg["implementation_sha256"]
    approval = json.loads(Path("provenance/final_approval.json").read_text())
    assert approval["pre_final_git_commit"] == PREFINAL
    root = Path(cfg["output_directory"])
    analysis = root / "analysis"
    verification = json.loads((analysis / "verification.json").read_text())
    design = json.loads((analysis / "design_verification.json").read_text())
    assert len(verification["completed"]) == design["verified_cells"] == 120
    assert not verification["missing"] and verification["failure_records"] == 0
    records = [json.loads(p.read_text()) for p in sorted((root / "execution_records").glob("*.json"))]
    producing = [record for record in records if record.get("newly_completed_cells", 0)]
    assert len(producing) == 1 and producing[0]["newly_completed_cells"] == 120, "Account explicitly for any resumed execution before publishing"
    execution = producing[0]
    assert execution["state"] == "completed" and execution["working_tree_clean_before_execution"]
    assert execution["config_file_sha256"] == EXPECTED
    # Verify byte identity of the frozen numerical modules at both Git snapshots.
    for path in sorted(Path("src/motor_rnn").glob("*.py")):
        pre = subprocess.check_output(["git", "show", PREFINAL + ":" + path.as_posix()])
        launch = subprocess.check_output(["git", "show", execution["code_git_commit"] + ":" + path.as_posix()])
        assert pre == launch == path.read_bytes()
    target = Path("results/confirmatory")
    target.mkdir(parents=True, exist_ok=True)
    orders = json.loads((root / "blocked_order_assignment.json").read_text())
    assert file_hash(root / "blocked_order_assignment.json") == approval["blocked_order_assignment_sha256"]
    cells = []
    for run_id in cfg["run_ids"]:
        for k in cfg["distances"]:
            directory = root / "runs" / "run_{:03d}_k{}".format(run_id, k)
            cell = {**cfg, "run_id": run_id, "distance": k, "blocked_order": orders[str(k)][str(run_id)]}
            assert verify_done(directory, cell)
            done = json.loads((directory / "complete.json").read_text())
            manifest = json.loads((directory / "manifest.json").read_text())
            dest = target / "manifests" / (directory.name + ".json")
            copy(directory / "manifest.json", dest)
            cells.append({"run_id": run_id, "distance": k, "directory": directory.as_posix(),
                          "manifest_in_repository": dest.as_posix(), "manifest_sha256": file_hash(dest),
                          "data_sha256": manifest["data_sha256"], "config_hash": done["config_hash"],
                          "complete_record_sha256": file_hash(directory / "complete.json"),
                          "artifact_hashes": done["files"], "elapsed_seconds": done["elapsed_seconds"],
                          "producing_git_commit": execution["code_git_commit"]})
            copy(directory / "complete.json", target / "completion_records" / (directory.name + ".json"))
    for source in analysis.iterdir():
        if source.suffix in (".csv", ".json"):
            if source.name in ("training.csv", "development.csv", "D_development.csv"):
                with open(target / (source.name + ".gz"), "wb") as raw:
                    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                        zipped.write(source.read_bytes())
            else:
                copy(source, target / ("raw_metrics.csv" if source.name == "metrics.csv" else source.name))
    for source in (root / "figures").iterdir():
        if source.suffix in (".png", ".pdf"):
            copy(source, target / "figures" / source.name)
    for name in ("phase_config.json", "blocked_order_assignment.json", "last_execution.json", "events.jsonl"):
        copy(root / name, target / name)
    for name in ("analysis_execution.json", "analysis_execution.log"):
        if (root / name).exists():
            copy(root / name, target / name)
    for source in (root / "execution_records").glob("*.json"):
        copy(source, target / "execution_records" / source.name)
    directory = root / "runs/run_000_k2"
    with np.load(directory / "data.npz") as data, np.load(directory / "orders.npz") as saved:
        example = {"manifest": json.loads((directory / "manifest.json").read_text()),
                   "pair_checks": json.loads((directory / "pair_checks.json").read_text()),
                   "train_cue_shape": list(data["train_x"].shape), "final_probe_cue_shape": list(data["final_noisy_x"].shape),
                   "training_example_cue": data["train_x"][0, 0].tolist(), "training_example_target": data["train_y"][0].tolist(),
                   "blocked_first_ids": saved["blocked"][:12].tolist(), "interleaved_first_ids": saved["interleaved"][:12].tolist(),
                   "example_id_multisets_verified_equal": bool(np.array_equal(np.sort(saved["blocked"]), np.sort(saved["interleaved"])))}
    write_json(target / "example.json", example)
    stage = {"project": "CruxRNN", "phase": "confirmatory_final", "pre_final_git_commit": PREFINAL,
             "generating_git_commit": execution["code_git_commit"], "numerical_modules_identical_to_pre_final": True,
             "config_file": CONFIG.as_posix(), "config_file_sha256": EXPECTED,
             "numerical_implementation_sha256": implementation_hash(), "seed_namespace": cfg["namespace"],
             "started_at_utc": execution["started_at_utc"], "finished_at_utc": execution["finished_at_utc"],
             "expected_cells": 120, "completed_cells": 120, "main_models": 240,
             "failure_records": 0, "excluded_cells": 0, "resumed_cells": 0, "replaced_successful_cells": 0,
             "phase_wall_seconds": execution["wall_seconds"],
             "analysis_wall_seconds": json.loads((root / "analysis_execution.json").read_text())["wall_seconds"],
             "cell_runtime_seconds": verification["measured_cell_runtime_seconds"],
             "run_artifact_bytes": design["run_artifact_bytes"],
             "environment_sha256": execution["environment_sha256"],
             "pip_freeze_sha256": execution["pip_freeze_sha256"],
             "raw_directory": root.as_posix(), "published_directory": target.as_posix(), "cells": cells}
    write_json(target / "artifact_index.json", stage)
    stages = json.loads(Path("provenance/stages.json").read_text())
    stages["confirmatory_final"] = {key: value for key, value in stage.items() if key != "cells"}
    write_json("provenance/stages.json", stages)
    published = {p.relative_to(target).as_posix(): file_hash(p) for p in target.rglob("*") if p.is_file() and p.name != "published_file_hashes.json"}
    write_json(target / "published_file_hashes.json", published)
    write_reports(cfg, stage, target)
    historical_path = Path("reports/prior_execution.md")
    history = historical_path.read_text(encoding="utf-8")
    if "## Subsequent approved confirmation" not in history:
        history = history.replace("# CruxRNN: prior execution and the pilot approval gate", "# CruxRNN: prior execution and the pilot approval gate\n\nThe following review-state account describes the period before the user's subsequent approval. The final section records the approved confirmation; the earlier provenance gap remains unchanged.")
        history += "\n## Subsequent approved confirmation\n\nThe user approved the pilot and the unchanged fresh configuration on October 13, 2026 (local date). The exact pre-final commit `" + PREFINAL + "` was pushed to ericqin0816/crux-rnn and remotely verified first. The producing execution commit was `" + stage["generating_git_commit"] + "`, with byte-identical numerical modules. All 120 new paired cells completed in namespace `cruxrnn_final_proposed_v1`, with zero failures/exclusions and contemporaneous Git/config/environment/UTC/hash records. Only these new cells enter the final inference and figures. The earlier 120 cells remain exploratory/provenance-incomplete at `outputs/final/` and are not pooled.\n"
        historical_path.write_text(history, encoding="utf-8")
    deviations = Path("reports/deviations.md")
    text = deviations.read_text(encoding="utf-8")
    if "## Approved confirmatory execution" not in text:
        text += "\n## Approved confirmatory execution\n\nThe paragraphs above record the earlier execution and pilot-review periods. After explicit user approval, the exact pre-final checkpoint was pushed and verified. The frozen config and numerical modules were unchanged. Before prospective training, the requested descriptive sample SD (ddof=1), comprehensive group summary and 300-dpi export were added; bootstrap inference was unchanged. Commit `" + stage["generating_git_commit"] + "` contains those approval/reporting changes and is the recorded producing commit. Reporting/curation helpers were prepared while training ran, without changing the runner, analysis procedure, numerical modules or config. All 120 new pairs completed once; there were no execution failures, exclusions, resumes, forced overrides, or replacements. Source/smoke/pilot and the earlier exploratory final remain separate. Current full environment evidence is retained separately from the historical snapshot.\n"
        deviations.write_text(text, encoding="utf-8")
    Path("docs/github_setup.md").write_text("# CruxRNN: GitHub delivery\n\nRepository: [ericqin0816/crux-rnn](https://github.com/ericqin0816/crux-rnn), public. `origin` is `https://github.com/ericqin0816/crux-rnn.git`. The user's browser verified owner ericqin0816; the mismatched connector was not used for writes. Git authenticated normally for pushes; no credential was extracted.\n\nThe clean pre-final commit `" + PREFINAL + "` was pushed and independently checked against remote main before training. The approved producing commit `" + stage["generating_git_commit"] + "` was also pushed. Final result and one explicit SHA-recording documentation commit are published after verification; see RUN_REPORT for that relationship. Release target: `v1.0-final-experiment`, **CruxRNN v1.0 — Final Experiment**.\n\nCurated tables, gzip CSV histories, five PNG/PDF figures, 120 manifests/completion records and hash indexes are tracked. Large arrays and checkpoints are excluded from Git and supplied in the separate confirmatory data release asset. Author checkout, environments, credentials and temporary files remain ignored. The prior exploratory data are not included in that asset. No Git LFS is required. GitHub provides transparency, not independent scientific validation.\n", encoding="utf-8")
    print("Published 120 verified pairs from the fresh namespace only.")


def write_reports(cfg, stage, target):
    primary = pd.read_csv(target / "primary_summary.csv")
    paired = pd.read_csv(target / "paired_differences.csv")
    runs = pd.read_csv(target / "run_level.csv")
    metrics = pd.read_csv(target / "group_metrics_summary.csv")
    trend = json.loads((target / "trend.json").read_text())
    assert len(runs) == 240 and len(paired) == 120
    display = primary.rename(columns={"distance": "k", "paired_runs": "pairs", "blocked_noisy_accuracy_pct": "Blocked %",
                                      "IP_noisy_accuracy_pct": "IP %", "mean_difference_pp": "Mean IP−blocked pp",
                                      "median_difference_pp": "Median pp", "sd_difference_pp": "SD pp",
                                      "ci_low_pp": "95% CI low pp", "ci_high_pp": "95% CI high pp"})
    primary_table = table(display, ["k", "pairs", "Blocked %", "IP %", "Mean IP−blocked pp", "Median pp", "SD pp", "95% CI low pp", "95% CI high pp"])
    crossing = primary[(primary.ci_low_pp <= 0) & (primary.ci_high_pp >= 0)].distance.tolist()
    ci_statement = ("The intervals include zero at k=" + ", ".join(map(str, crossing)) + "; the direction at these distances remains uncertain.") if crossing else "All four prespecified bootstrap intervals are above zero in this simulation."
    monotonic = np.all(np.diff(primary.mean_difference_pp.to_numpy()) > 0)
    shape_statement = "The categorical mean differences increase across the four distances." if monotonic else "The categorical mean differences do not increase monotonically across the four distances; the slope alone does not capture that shape."
    trend_text = "The prespecified slope is **{:.3f} pp per Hamming unit**, with a run-ID bootstrap 95% CI **[{:.3f}, {:.3f}]**.".format(trend["slope_pp_per_hamming"], trend["ci_low"], trend["ci_high"])
    if trend["ci_low"] <= 0 <= trend["ci_high"]:
        trend_text += " The slope interval includes zero."
    secondary = pd.read_csv(target / "secondary_summary.csv")
    sec = secondary.copy()
    for column in sec.columns:
        if column not in ("distance", "schedule", "final_cross_entropy"):
            sec[column] *= 100
    sec_table = table(sec)
    secondary_pairs = secondary.pivot(index="distance", columns="schedule")
    lower_sequence = [k for k in cfg["distances"] if secondary_pairs["final_noisy_sequence_accuracy"]["interleaved"].loc[k] < secondary_pairs["final_noisy_sequence_accuracy"]["blocked"].loc[k]]
    larger_forgetting = [k for k in cfg["distances"] if secondary_pairs["forgetting_noisy"]["interleaved"].loc[k] > secondary_pairs["forgetting_noisy"]["blocked"].loc[k]]
    secondary_interpretation = "IP has lower mean noisy complete-sequence accuracy at k={}, despite its higher mean action accuracy. Mean noisy D forgetting is greater under IP at k={}; baseline accuracy limits how much a model can lose. Mean clean unseen-base action accuracy spans {:.2f}–{:.2f}% and noisy accuracy spans {:.2f}–{:.2f}%, close to the 25% four-action baseline. These descriptive results limit broad claims of better learning, faster learning or transfer.".format(
        ",".join(map(str, lower_sequence)), ",".join(map(str, larger_forgetting)),
        100 * secondary.generalization_clean_accuracy.min(), 100 * secondary.generalization_clean_accuracy.max(),
        100 * secondary.generalization_noisy_accuracy.min(), 100 * secondary.generalization_noisy_accuracy.max())
    before = metrics[(metrics.stage == "before_D") & metrics.split.isin(["final_clean", "final_noisy"]) & metrics.group.isin(list("ABC"))].copy()
    abc = before[["distance", "schedule", "split", "group", "action_accuracy", "sequence_accuracy", "cross_entropy"]].copy()
    abc[["action_accuracy", "sequence_accuracy"]] *= 100
    speed = []
    for (k, schedule), part in runs.groupby(["distance", "schedule"]):
        reached = pd.to_numeric(part.threshold_start, errors="coerce").dropna()
        speed.append({"k": k, "schedule": schedule, "reached / 30": "{} / 30".format(len(reached)),
                      "median first qualifying update (reached only)": float(reached.median()) if len(reached) else np.nan})
    speed_table = table(pd.DataFrame(speed))
    limitation = "The complete target sequence is supplied in the initial cue. These results describe numerical cue decoding by one small RNN with a fixed budget, ordinal action codes, and unmatched action frequencies/repeats. D similarity is observed rather than controlled. Thirty simulation IDs quantify uncertainty within this model/generator; noisy copies and positions are not independent subjects. Prior exploratory outcomes were seen, so this is a disclosed independent confirmation, not a retroactively untouched preregistration. No human brain mechanism or climbing-performance effect is demonstrated."
    methods = "Thirty independent replicate IDs each span k=1,2,4,7: 120 paired cells and 240 main models. Each cell pretrains once, clones identical weights, and presents the same 15,000 noisy examples in blocked or interleaved order. Each of six blocked orders occurs five times per k. SGD .02, zero momentum, batch 20; 750 main updates; probes every 25; 100 matched D updates. Primary: equally weighted A/B/C fresh-noisy final action accuracy before D. Ten thousand percentile bootstrap resamples use whole replicate IDs, retaining all four distances for the prespecified linear slope. SD uses ddof=1. No copies/positions serve as independent runs."
    runtime = "Phase wall time: **{:.3f} seconds ({:.2f} minutes)**. Summed cell execution: **{:.3f} seconds**. These measured times cover data generation, pretraining, main training, probes, generalization, D and cell saves; they exclude environment setup, analysis, plotting, reporting, and GitHub delivery. Analysis/figure generation took an additional **{:.3f} seconds**, for **{:.3f} seconds** of experiment plus analysis/plotting; independent verification/reporting/archive/delivery time is separate.".format(stage["phase_wall_seconds"], stage["phase_wall_seconds"] / 60, stage["cell_runtime_seconds"], stage["analysis_wall_seconds"], stage["phase_wall_seconds"] + stage["analysis_wall_seconds"])
    phases = "| Stage | Completed paired cells | Role |\n| --- | --- | --- |\n| Source reproduction | 100 (first five audited) | Audited source behavior; uncontrolled similarity |\n| Smoke | 4 | Pipeline only |\n| Pilot | 20 (five IDs × four k) | Diagnostic only |\n| Earlier exploratory | 120 | Exposed outcomes; producing SHA/timestamps missing; excluded from new inference |\n| New confirmatory final | 120 (30 IDs × four k) | Fresh approved namespace, complete contemporary provenance |"
    provenance = "Pre-final checkpoint: `{}` (pushed and remotely verified before execution). Producing execution commit: `{}`; it adds approval and reporting metadata, and every numerical module is byte-identical to the pre-final checkpoint. Config SHA-256: `{}`. Namespace: `{}`. Frozen numerical fingerprint: `{}`. Start/end UTC: `{}` / `{}`. Expected/completed: 120/120 pairs, 240 main models. Failed/excluded/resumed/replaced-successful cells: 0/0/0/0. All artifact hashes, seeds, orders, disjoint bases, pair initializations, metric identities and bootstrap row resampling were independently checked.".format(PREFINAL, stage["generating_git_commit"], EXPECTED, cfg["namespace"], cfg["implementation_sha256"], stage["started_at_utc"], stage["finished_at_utc"])
    paths = "Full raw arrays, checkpoints, seeds, per-cell CSVs and completion hashes: `outputs/cruxrnn_final_proposed_v1/runs/run_NNN_kK/` (Git-ignored). Full analysis including bootstrap indices/arrays: `outputs/cruxrnn_final_proposed_v1/analysis/`. Tracked results: `results/confirmatory/`; all 120 individual differences are in `paired_differences.csv`. Per-update CSVs are losslessly compressed as `training.csv.gz`, `development.csv.gz`, and `D_development.csv.gz` (pandas `read_csv` detects gzip). The artifact index, 120 manifests/completion records, published-file hashes and contemporary execution record link those products to code/config/data. The earlier exploratory files remain separately in `outputs/final/` and `reports/history/prior_*`; none were pooled. Historical source/smoke/pilot metadata gaps remain explicit."
    figures = "\n".join("- `results/confirmatory/figures/{}.[png,pdf]`".format(name) for name in ("learning_curves", "final_accuracy", "paired_differences", "D_forgetting", "position_errors"))
    results = "# CruxRNN: confirmatory final results\n\n" + methods + "\n\n" + phases + "\n\n## Primary outcome\n\n" + primary_table + "\n\n" + ci_statement + " " + shape_statement + "\n\n" + trend_text + " No p-values or multiplicity-adjusted familywise claims are made.\n\n## Secondary outcomes\n\nAccuracy and forgetting columns below are percentages or percentage points; cross-entropy is an unscaled mean loss.\n\n" + sec_table + "\n\nClean action scores, complete-sequence success, cross-entropy, unseen-base generalization and D forgetting answer different questions. The raw/group tables additionally retain clean complete-sequence accuracy, generalization sequence accuracy, every A/B/C score, positions 1–7, and before/after-D evaluations. These secondary summaries are descriptive.\n\n### A/B/C before D\n\nAction/sequence accuracy are shown in percent.\n\n" + table(abc) + "\n\n### Learning speed\n\nThreshold: at least 90% noisy mean action accuracy for three consecutive fixed probes. Medians condition on reaching; failures to reach are not assigned a fabricated time.\n\n" + speed_table + "\n\n### Positions and interference\n\n`position_summary.csv` separates shared/differing scores and counts; `group_metrics_summary.csv` includes each position. At k=7, shared accuracy stays NaN and action count is zero. Forgetting is before-D minus after-D accuracy on the same held-out set. Raw drops depend on baseline accuracy and are not direct evidence of a consolidation mechanism. `D_development.csv.gz` preserves A/B/C after every D update.\n\n## Interpretation and limits\n\n" + limitation + "\n\n## Provenance and data\n\n" + provenance + "\n\n" + runtime + "\n\n" + paths + "\n"
    results = results.replace("These secondary summaries are descriptive.", "These secondary summaries are descriptive. " + secondary_interpretation)
    Path("reports/results.md").write_text(results, encoding="utf-8")
    plain_rows = []
    for row in primary.itertuples():
        plain_rows.append("- At k={}: blocked {:.2f}%, interleaved {:.2f}%; mean advantage {:.2f} percentage points, 95% bootstrap interval [{:.2f}, {:.2f}].".format(row.distance, row.blocked_noisy_accuracy_pct, row.IP_noisy_accuracy_pct, row.mean_difference_pp, row.ci_low_pp, row.ci_high_pp))
    plain = "# CruxRNN in plain English\n\nWe asked whether practicing three artificial seven-action patterns in separate blocks or shuffled together changes how this small network reproduces them. Larger k means the patterns differ at more positions and are less similar. We completed 30 paired simulations at each of four distances, using fresh seeds after the pilot approval.\n\n" + "\n".join(plain_rows) + "\n\n" + ci_statement + " " + shape_statement + "\n\n" + trend_text + " The straight line summarizes the measured distances, and does not establish a rule for other tasks.\n\nThe uncertainty comes from independent simulations. Hundreds of noisy examples from one model do not become hundreds of independent participants. A confidence interval above zero supports a positive mean difference within this computational protocol; it does not settle biological or behavioral questions.\n\nShared positions can make action accuracy high even when a model does not reproduce a whole sequence correctly. We therefore also report complete-sequence scores, clean cues, unseen bases and performance after learning D. A high initial score leaves more accuracy available to lose during D.\n\n" + limitation + "\n\n" + phases + "\n\nThe prior exploratory study is disclosed and kept separate. The current results use only `cruxrnn_final_proposed_v1`. Read `results.md` for all metrics and `../RUN_REPORT.md` for hashes, runtime and reproduction steps.\n"
    plain = plain.replace("The prior exploratory study is disclosed", secondary_interpretation + "\n\nThe prior exploratory study is disclosed")
    Path("reports/plain_english.md").write_text(plain, encoding="utf-8")
    report = "# " + TITLE + "\n\n**Approved confirmatory final experiment complete.**\n\n## Phase accounting\n\n" + phases + "\n\n## Approval and frozen provenance\n\n" + provenance + "\n\nThe approval record is `provenance/final_approval.json`. The original proposal remains unchanged as historical pre-execution evidence in `provenance/final_proposal.json`. The launch record is under `results/confirmatory/execution_records/`.\n\n## Design and analysis\n\n" + methods + "\n\n" + primary_table + "\n\n" + ci_statement + " " + shape_statement + "\n\n" + trend_text + "\n\n## Secondary outcomes\n\n" + sec_table + "\n\n" + speed_table + "\n\nFull A/B/C, clean/sequence/CE/position/generalization/D summaries and interpretation: `reports/results.md`. k=7 shared accuracy is NaN with count zero.\n\n## Tests and verification\n\nThe preflight suite passed 17 tests with no failures/skips, including reference model output/gradient parity, exact tiny interrupted/resumed training, pairing, exposure, targets, disjointness, seed namespaces, metrics, thresholds and approval/provenance controls. Final test log: `reports/unit_tests_confirmatory_final.txt`. Independent full-data audit: `results/confirmatory/design_verification.json`. All 120 paired cells and completion hashes were accounted for before final inference. No failure, exclusion, resume, force override, or replacement of successful cells occurred.\n\n## Environment and measured runtime\n\nWindows 11 Home; AMD Ryzen 7 8700F (8 cores / 16 logical processors). Python 3.8.10; PyTorch 1.13.1+cpu; NumPy 1.24.3; pandas 2.0.3; SciPy 1.10.1; Matplotlib 3.7.2; scikit-learn 1.3.0; joblib 1.4.2. CPU float32, one intra-op and one inter-op thread, deterministic algorithms. The system snapshot reports interpreter defaults separately; the runner explicitly sets both experiment thread counts to one. Current freeze/system evidence: `provenance/environment/confirmatory/`. Conda was unavailable; an isolated local Windows Python and pip provided the principal requested versions.\n\n" + runtime + "\n\n## Files and figures\n\n" + paths + "\n\n" + figures + "\n\nAll five figures use only the new confirmation and export 300-dpi PNG plus vector PDF. Learning curves show mean ± standard error across 30 IDs (descriptive); paired outcome intervals are the prespecified bootstrap intervals.\n\n## Source audit and earlier phases\n\nSource repository inspected at `31f58bb6de90d1fffcb752c0b97afb7df51479a7`; six inspected file hashes in `provenance/author_source_files.json`. First five and full 100-run numerical reproduction comparisons are in `results/source_reproduction/first_five/` and `full_100/`. Original uniqueness and training-input retention limitations were preserved in source-exact mode. Smoke checks every pipeline stage; the 20-pair pilot supplied diagnostics and a runtime estimate. Neither enters these final figures or inference. Earlier execution did not record producing Git SHAs/start/end timestamps; later snapshots/reverifications are honestly distinguished in `provenance/stages.json`. No missing historical provenance was invented.\n\n## Reproduction\n\nSet up `envs/reproduction.yml` or `scripts/setup_windows.ps1`, activate the environment, and use its Python. On this machine:\n\n```powershell\n$taskPython = Join-Path (Get-Location) 'envs/tools/python3810/tools/python.exe'\n& $taskPython scripts/system_info.py\n& $taskPython -m unittest discover -s tests -v\n& $taskPython scripts/run_experiment.py --config configs/final_proposed.json --resume\n& $taskPython scripts/analyze.py --config configs/final_proposed.json\n& $taskPython scripts/verify_design.py --config configs/final_proposed.json\n& $taskPython scripts/publish_confirmatory.py\n& $taskPython scripts/create_walkthrough.py\n```\n\nStart the runner from a clean Git checkout. On a fresh checkout it generates all 120 cells with the same frozen namespace; locally `--resume` verifies/skips completed cells. Only incomplete computation may continue. Retain raw outputs for bitwise artifact comparison. A producing Git SHA in a new reproduction will identify that new checkout, while the scientific config/numerical fingerprint remain fixed.\n\n## Deviations and limitations\n\nOnly descriptive SD/group tables and 300-dpi export were added before this new experiment. Scientific settings, numerical modules, blocked assignment and bootstrap procedure were unchanged. Optional source vulnerability and 10,000-pretraining sensitivity remain unexecuted. Full ledger: `reports/deviations.md`.\n\n" + limitation + "\n\n## Final commit and release provenance\n\nThe results commit SHA will be recorded in one explicit follow-up documentation commit after results are committed. The release tag will identify the final documentation snapshot; a commit cannot contain its own hash.\n"
    report = report.replace("Full A/B/C, clean/sequence/CE/position/generalization/D summaries", secondary_interpretation + "\n\nFull A/B/C, clean/sequence/CE/position/generalization/D summaries")
    archive_metadata = Path("artifacts/confirmatory_archive.json")
    if archive_metadata.exists():
        archive = json.loads(archive_metadata.read_text())
        report += "\n## Full raw-data release asset\n\n`cruxrnn-confirmatory-data.zip` contains only the new confirmation's raw outputs, including arrays, checkpoints, CSVs, bootstrap indices/values and figures. Size: {} bytes; {} members; CRC verified. SHA-256: `{}`. Every payload is indexed by SHA-256 inside `archive_manifest.json`. Extract at the release checkout root; no earlier exploratory data, author checkout or Python runtime is included. External receipt: `artifacts/confirmatory_archive.json`.\n".format(archive["bytes"], archive["file_count"], archive["sha256"])
    commit_receipt = Path("provenance/final_results_commit.json")
    if commit_receipt.exists():
        receipt = json.loads(commit_receipt.read_text())
        report = report.replace("The results commit SHA will be recorded in one explicit follow-up documentation commit after results are committed. The release tag will identify the final documentation snapshot; a commit cannot contain its own hash.",
                                "Results commit: `{}` (**Add CruxRNN confirmatory final experiment results**). This SHA is recorded in the explicitly separate follow-up documentation commit, along with `provenance/final_results_commit.json`. The follow-up adds only this provenance relationship and its receipt; the results commit cannot contain its own SHA. Release tag `v1.0-final-experiment` identifies the follow-up documentation snapshot. Resolve that snapshot's exact SHA with `git rev-parse v1.0-final-experiment^{{}}`.".format(receipt["results_commit"]))
    Path("RUN_REPORT.md").write_text(report, encoding="utf-8")
    notes = "Research question: does sequence similarity change the relative effects of blocked and interleaved practice in a minimal RNN?\n\n" + methods + "\n\n" + primary_table + "\n\n" + ci_statement + " " + shape_statement + "\n\n" + trend_text + "\n\nCompleted 120/120 pairs (240 main models); zero failures/exclusions. " + runtime + "\n\n" + limitation + "\n\nReproduce with the pinned environment and `scripts/run_experiment.py --config configs/final_proposed.json --resume`, then `scripts/analyze.py --config configs/final_proposed.json` and `scripts/verify_design.py --config configs/final_proposed.json`. Start from a clean checkout. The frozen config SHA is `" + EXPECTED + "`. See README and RUN_REPORT for full setup, provenance and data locations. Tracked CSVs, figures and manifests are in `results/confirmatory/`; local full arrays/checkpoints are indexed separately. Source/smoke/pilot and earlier exploratory data were not pooled into final inference.\n"
    Path("reports/release_notes.md").write_text(notes, encoding="utf-8")
    update_readme(primary_table, trend_text, ci_statement, shape_statement, stage)


def update_readme(primary_table, trend_text, ci_statement, shape_statement, stage):
    path = Path("README.md")
    text = path.read_text(encoding="utf-8")
    status = "**Current status: approved confirmatory final experiment complete.** All 120 fresh paired cells / 240 main models are verified, with zero failures or exclusions. Source reproduction (100 pairs), smoke (4) and pilot (20) remain separate. The earlier 120-pair exploratory study has incomplete producing-SHA/timestamp provenance and is excluded from these final results; [prior execution](reports/prior_execution.md) discloses that history."
    text = re.sub(r"^\*\*Current status:.*$", status, text, flags=re.M)
    text = re.sub(r"^Start with.*$", "Start with [RUN_REPORT.md](RUN_REPORT.md), the [student notebook](notebooks/walkthrough.ipynb), and [confirmatory results](reports/results.md). Code, tables, figures and provenance are published in [ericqin0816/crux-rnn](https://github.com/ericqin0816/crux-rnn). Release tag: `v1.0-final-experiment`.", text, flags=re.M)
    text = text.replace("The proposed final assignment gives", "The frozen final assignment gives")
    text = text.replace("source/smoke/pilot and proposed final configurations", "source/smoke/pilot and frozen final configurations")
    text = text.replace("tracked source/smoke/pilot CSVs, figures, manifests", "tracked source/smoke/pilot/final CSVs, figures, manifests")
    text = text.replace("Review suite: 17 passed", "Final suite: 17 passed")
    text = text.replace("reports/unit_tests_pilot_review.txt", "reports/unit_tests_confirmatory_final.txt")
    final_section = "## 16. Running the final experiment\n\nThe user approved the fresh confirmation after pilot review. Frozen config: `configs/final_proposed.json`, SHA-256 `" + EXPECTED + "`; namespace `cruxrnn_final_proposed_v1`. Completed 30 IDs at each k=1,2,4,7, with five instances of each blocked order per k. Pre-final Git checkpoint `" + PREFINAL + "` was pushed before execution; producing commit `" + stage["generating_git_commit"] + "` has byte-identical numerical modules. [Approval](provenance/final_approval.json) and [execution records](results/confirmatory/execution_records/) record the relationship.\n\n```powershell\n& $taskPython scripts/run_experiment.py --config configs/final_proposed.json --resume\n& $taskPython scripts/analyze.py --config configs/final_proposed.json\n& $taskPython scripts/verify_design.py --config configs/final_proposed.json\n& $taskPython scripts/publish_confirmatory.py\n& $taskPython scripts/create_walkthrough.py\n```\n\nStart from a clean Git tree; the runner verifies existing complete cells and only continues incomplete work. A fresh checkout regenerates all cells from the frozen seeds. [Prior execution disclosure](reports/prior_execution.md) distinguishes the earlier exploratory study.\n"
    result_section = "## 17. Results\n\nFresh-noisy final action accuracy, before D; differences/intervals are percentage points:\n\n" + primary_table + "\n\n" + ci_statement + " " + shape_statement + " " + trend_text + "\n\n[Full results](reports/results.md) include clean/complete-sequence/CE/A-B-C/position/generalization/speed/D outcomes. [Every individual pair](results/confirmatory/paired_differences.csv) is retained. Source, smoke, pilot and earlier exploratory data were not pooled. Measured final phase runtime: {:.3f} seconds ({:.2f} minutes); zero failures/exclusions.\n".format(stage["phase_wall_seconds"], stage["phase_wall_seconds"] / 60)
    figure_section = "## 18. Figures\n\nFive original figures use only the new confirmation: learning curves, final accuracy, paired differences with bootstrap CIs, D forgetting, and shared/differing position errors. All export 300-dpi PNG plus PDF in [results/confirmatory/figures](results/confirmatory/figures/). Clean/noisy panels are separate; k=7 shared positions have NaN/count zero. `scripts/analyze.py` regenerates them. Pilot/smoke figures retain diagnostic labels in their separate directories.\n\n![Confirmatory paired differences and bootstrap CIs](results/confirmatory/figures/paired_differences.png)\n"
    for number, following, replacement in ((16, 17, final_section), (17, 18, result_section), (18, 19, figure_section)):
        text = re.sub(r"## " + str(number) + r"\..*?(?=\n## " + str(following) + r"\.)", lambda match: replacement, text, flags=re.S)
    text = text.replace("Original pre-execution Git/timestamp records are missing.", "Original exploratory/source/smoke/pilot pre-execution Git/timestamp records are missing; the new confirmation records them contemporaneously.")
    text = text.replace("Prior final outcomes were seen, so a later run is a disclosed independent confirmation", "Prior exploratory outcomes were seen, so this run is a disclosed independent confirmation")
    text = text.replace("[GitHub setup and later push command](docs/github_setup.md). No Git LFS or final release/tag is created.", "[GitHub delivery](docs/github_setup.md). No Git LFS is required. Full raw confirmation arrays/checkpoints are supplied as `cruxrnn-confirmatory-data.zip` with a SHA-256/CRC record in `artifacts/confirmatory_archive.json`; extract at the release checkout root. The release provides this asset separately from Git history. Historical outputs stay separate.")
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
