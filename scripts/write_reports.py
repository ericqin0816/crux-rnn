"""Write reports only from verified saved outputs, without placeholder results."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from motor_rnn.provenance import file_hash

root = Path.cwd()
final = root / "outputs/final/analysis"
primary = pd.read_csv(final / "primary_summary.csv")
secondary = pd.read_csv(final / "secondary_summary.csv")
trend = json.loads((final / "trend.json").read_text())
verification = json.loads((final / "verification.json").read_text())
design = json.loads((final / "design_verification.json").read_text())
assert design["verified_cells"] == 120 and not verification["missing"]
assert len(primary) == 4 and (primary.paired_runs == 30).all()
environment = json.loads(Path("outputs/environment/system_info.json").read_text())
benchmark = json.loads(Path("outputs/environment/benchmark.json").read_text())
config_hash = file_hash("configs/final.json")
assert config_hash == Path("configs/final.sha256").read_text().strip()
counts = {}
runtime_rows = []
for phase in ("source_exact", "smoke", "pilot", "final"):
    saved = json.loads(Path("outputs", phase, "analysis/verification.json").read_text())
    counts[phase] = {"completed": len(saved["completed"]), "missing": len(saved["missing"]), "failures": saved["failure_records"]}
    runtime_rows.append((phase, saved["measured_cell_runtime_seconds"]))
source_comparison = pd.read_csv("outputs/source_exact/comparison/full_100/aggregate_comparison.csv")
source_directions = json.loads(Path("outputs/source_exact/comparison/full_100/qualitative_directions.json").read_text())
source_overlap = json.loads(Path("outputs/source_exact/analysis/sequence_overlap_totals.json").read_text())
tests = Path("reports/unit_tests_final.txt").read_text()
assert "Ran 14 tests" in tests and "\nOK" in tests
assert not any(counts[phase]["missing"] for phase in counts)


def table(headers, rows):
    return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in rows)


primary_table = table(["k", "Paired runs", "Blocked noisy (%)", "IP noisy (%)", "IP−blocked mean (pp)", "Median (pp)", "95% CI (pp)"],
                      [[int(r.distance), int(r.paired_runs), "{:.2f}".format(r.blocked_noisy_accuracy_pct), "{:.2f}".format(r.IP_noisy_accuracy_pct),
                        "{:.2f}".format(r.mean_difference_pp), "{:.2f}".format(r.median_difference_pp), "[{:.2f}, {:.2f}]".format(r.ci_low_pp, r.ci_high_pp)] for r in primary.itertuples()])
secondary_table = table(["k", "Schedule", "Clean trained (%)", "Noisy whole-sequence (%)", "Clean unseen (%)", "Noisy unseen (%)", "Noisy CE", "Clean D drop (pp)", "Noisy D drop (pp)"],
                        [[int(r.distance), r.schedule, "{:.2f}".format(100 * r.final_clean_action_accuracy), "{:.2f}".format(100 * r.final_noisy_sequence_accuracy),
                          "{:.2f}".format(100 * r.generalization_clean_accuracy), "{:.2f}".format(100 * r.generalization_noisy_accuracy), "{:.3f}".format(r.final_cross_entropy),
                          "{:.2f}".format(100 * r.forgetting_clean), "{:.2f}".format(100 * r.forgetting_noisy)] for r in secondary.itertuples()])
count_table = table(["Phase", "Completed paired cells", "Missing", "Failure records", "Measured cell runtime (s)"],
                    [[phase, counts[phase]["completed"], counts[phase]["missing"], counts[phase]["failures"], "{:.2f}".format(seconds)] for phase, seconds in runtime_rows])
source_table = table(["Metric", "New n=100 loss", "Author n=100 loss", "New shape", "Author shape"],
                     [[r.metric, "{:.4f}".format(r.new_mean_loss), "{:.4f}".format(r.author_mean_loss), r.new_shape, r.author_shape] for r in source_comparison.itertuples()])
speed = pd.read_csv(final / "threshold_reached_counts.csv")
speed_table = table(["k", "Schedule", "Reached sustained 90% (out of 30)"], [[int(r.distance), r.schedule, int(r.reached_count)] for r in speed.itertuples()])
limitation_text = """The complete sequence is provided as the initial cue, so generalization is decoding unseen numeric patterns rather than acquiring an unobserved physical skill. Seven hidden units, default initialization, one optimizer, and one fixed budget limit generality. Action frequencies, adjacent repeats, and longest runs are recorded but not matched across distances; they can accompany this generator's manipulation. Numeric cue values impose an ordinal geometry on action labels. Low-distance performance can approach a ceiling, and whole-sequence accuracy can be at a floor. Bootstrap uncertainty is across artificial simulation IDs, not people. D distances were observed, not manipulated, and raw forgetting depends on baseline accuracy. There is no offline time-dependent consolidation or direct biological evidence. These findings do not establish how human brains work and are not a broad novelty claim."""

results = """# Verified simulation results

All claims below refer to newly generated outputs except the explicitly labeled author saved-array comparison. Final simulation pairs: **30 at each k=1,2,4,7**, or 120 run-by-distance pairs / 240 main-trained schedule models. There are zero missing final cells, zero failure records, and zero exclusions. Each controlled cell also completed the paired 100-update D experiment. `complete.json` hashes, CSV counts, checkpoint update counts, manifests, and bootstrap indices were verified.

## Primary held-out fresh-noisy action accuracy

{primary}

Confidence intervals are percentile intervals from 10,000 resamples of simulation IDs. The observed mean schedule differences are positive at all four distances. The categorical values must be inspected alongside the linear trend.

The prespecified OLS trend is **{slope:.3f} percentage points per Hamming unit**, with 95% bootstrap CI **[{low:.3f}, {high:.3f}]**. Full replicate IDs, including all four distances, are resampled together. This is a descriptive model-dependent trend, not a biological dose-response law. No p-values or significance tests were added.

## Secondary measures

{secondary}

Clean and noisy conditions remain separate. Unseen-base performance is poor in absolute terms under both schedules (about 25–29% action accuracy; the analytical uniform-guess expectation for four classes is 25%). Higher relative performance does not imply reliable new-sequence competence. Despite better primary action accuracy, IP has **lower mean complete-sequence accuracy at k=2,4,7**; the secondary metrics do not uniformly favor IP. Blocked practice can retain one entire recent sequence while IP gets more individual actions correct across the family. At k=1 the median primary difference is zero despite a positive mean. D drops use the same final held-out old-sequence set before and after training. Mean noisy D drops are larger under IP at k=1,4,7 and slightly smaller at k=2; they should be interpreted in the context of higher initial accuracy and the blocked floor. Raw A/B/C and position 1–7 metrics are available in `outputs/final/analysis/metrics.csv`. At k=7 every shared-position entry is NaN with count zero. Learning and D curves use fixed development probes; final data were not used to choose a threshold.

{speed}

All threshold start/confirmation updates, including `not_reached`, are saved in the run-level CSV. Results from only successful threshold reaches are not substituted for missing thresholds.

## Source-compatible reproduction and phases

{counts}

Five source runs were inspected first; their per-run shapes and main directional comparisons agreed with the saved arrays. The measured runtime supported extending to 100 new main-training runs. The five are included in those 100 and were verified/skipped on resume, not retrained. Main aggregate comparison:

{source}

New blocked training loss is lower, with abrupt increases at the two block switches; final training-input retention and source generalization loss favor IP. These directions match the saved arrays. Aggregate values differ, as expected with new explicit seed streams and a Windows CPU environment. No exact historical seed replication is claimed. Source uniqueness/novelty were deliberately not enforced: {duplicate_runs} of 100 new runs had duplicate generalization bases, and {overlap_runs} had at least one exact generalization/pretraining-or-main overlap. These runs were retained; see `sequence_overlap_summary.csv`. All controlled generalization bases are unique and disjoint. The optional source vulnerability protocols were not newly run. The authors' saved last-sequence noise, pruning, and interference arrays were audited and show larger mean loss increases for blocked practice; those are reference data, not new controlled outcomes. The Figure 2 10,000-pretraining sensitivity config is provided but unexecuted.

Smoke used reduced budgets and has no scientific interpretation. Pilot used five disjoint IDs at each k with real budgets. No scientific settings changed following pilot. Neither phase contributes to final intervals.

## Limits

{limitations}
""".format(primary=primary_table, slope=trend["slope_pp_per_hamming"], low=trend["ci_low"], high=trend["ci_high"],
           secondary=secondary_table, speed=speed_table, counts=count_table, source=source_table, limitations=limitation_text,
           duplicate_runs=source_overlap["runs_with_generalization_duplicates"], overlap_runs=source_overlap["runs_with_any_generalization_training_overlap"])
Path("reports/results.md").write_text(results, encoding="utf-8")

gap_text = ", ".join("k={}: {:.1f} points".format(int(r.distance), r.mean_difference_pp) for r in primary.itertuples())
plain = """# What the simulations tell us

For this small artificial network, mixing the practice examples produced higher average final accuracy across the three trained patterns than practicing each pattern in a separate block. The average interleaved advantage was {gaps}. Each comparison used 30 paired simulations, and only fresh noisy final inputs were used for the primary score. The uncertainty intervals and exact figures are in `results.md`.

The advantage in individual-action accuracy was larger when the three patterns differed at more positions. The fitted trend was {slope:.2f} percentage points of additional advantage per differing position, with a bootstrap interval [{low:.2f}, {high:.2f}]. This summarizes four chosen distances. It does not imply that every individual simulation improved or that the relation must be linear outside this design. At k=1 the median advantage was zero. Requiring all seven actions to be correct gave lower mean success under IP at k=2,4,7, even though IP got more individual actions correct. The choice of learning measure matters.

Why can blocked practice look good while leaving earlier patterns poorly reproduced? During a block, the network repeatedly adjusts to the same pattern. Its current-example training error can fall rapidly. Those same weight changes can make earlier patterns harder to produce. The fixed probes look at all three patterns, so they expose that tradeoff.

Learning an additional D pattern often reduced old-pattern accuracy. This secondary test used identical D examples for both schedules. A network starting with higher old-pattern accuracy has more accuracy available to lose, while a network already near a floor can show a smaller numerical drop. We should not call one schedule biologically more robust from raw drops alone. D similarity was recorded rather than controlled.

Think of climbing only as a way to picture the question: repeatedly drilling one move pattern versus alternating several partly overlapping patterns. We did not measure climbing performance, muscles, timing, people, or brains. The model receives the full correct action pattern as its initial numeric cue. Even its “generalization” test simply asks it to decode unseen complete cues. It has no sleep, delayed memory, replay, or biological consolidation.

The project reproduces the reference study's main computational pattern, then asks a narrow similarity question with better-controlled data and evaluation. Recorded action frequencies and repeated-action structure could still contribute to differences among distance conditions. The result is evidence about this precise simulation, not advice that every learner should always interleave every skill.
""".format(gaps=gap_text, slope=trend["slope_pp_per_hamming"], low=trend["ci_low"], high=trend["ci_high"])
Path("reports/plain_english.md").write_text(plain, encoding="utf-8")

commands = """# Commands actually executed

Commands below are the experiment-relevant commands, with the local interpreter abbreviated as `$taskPython = 'C:\\Users\\Eric Qin\\Desktop\\proj1\\envs\\tools\\python3810\\tools\\python.exe'`. They were invoked through PowerShell; outputs are preserved in `reports/*_log.txt`, comparison/verification files, and per-phase execution metadata. File creation used the Codex patch tool. This list omits routine directory listings and progress reads.

```powershell
git init
git clone https://github.com/YJ-0000/RNN_during_RP_and_IP external/author_repo
git -C external/author_repo checkout 31f58bb6de90d1fffcb752c0b97afb7df51479a7
conda env create -f envs/reproduction.yml  # failed: executable unavailable
```

The system Python 3.12 downloaded/extracted the Python 3.8.10 NuGet package using `urllib.request.urlretrieve` and `zipfile.ZipFile.extractall` (URL: `https://www.nuget.org/api/v2/package/python/3.8.10`). The local runtime's attempted `-m venv .venv` failed (ensurepip exit 101). Its direct `-m ensurepip --upgrade` succeeded. Exact packages were then installed:

```powershell
& $taskPython -m pip install torch==1.13.1+cpu numpy==1.24.3 pandas==2.0.3 scipy==1.10.1 matplotlib==3.7.2 scikit-learn==1.3.0 joblib==1.4.2 --extra-index-url https://download.pytorch.org/whl/cpu
& $taskPython -m unittest discover -s tests -v
& $taskPython scripts/system_info.py
& $taskPython scripts/benchmark.py
& $taskPython scripts/record_provenance.py
& $taskPython scripts/attempt_environment.py
& $taskPython scripts/run_experiment.py --config configs/source_exact.json --limit 5
& $taskPython scripts/run_experiment.py --config configs/smoke.json
& $taskPython scripts/compare_source.py --limit 5 --label first_five
& $taskPython scripts/analyze.py --config configs/smoke.json
& $taskPython scripts/run_experiment.py --config configs/pilot.json
& $taskPython scripts/run_experiment.py --config configs/source_exact.json --resume
& $taskPython scripts/analyze.py --config configs/pilot.json
& $taskPython scripts/freeze_final.py
& $taskPython scripts/run_experiment.py --config configs/final.json
& $taskPython scripts/analyze.py --config configs/source_exact.json
& $taskPython scripts/compare_source.py --label full_100
& $taskPython scripts/source_overlap_audit.py
& $taskPython scripts/verify_design.py --config configs/pilot.json
& $taskPython scripts/verify_design.py --config configs/source_exact.json
& $taskPython scripts/analyze.py --config configs/final.json
& $taskPython scripts/verify_design.py --config configs/final.json
& $taskPython -m unittest discover -s tests -v
& $taskPython scripts/run_experiment.py --config configs/final.json --resume
& $taskPython scripts/create_walkthrough.py
& $taskPython scripts/write_reports.py
& $taskPython scripts/package_project.py
```

The report writer records only verified completions. See per-phase `events.jsonl` for new versus skipped runs and `complete.json` for hashes. Final resume is a verification-only pass; the original final-training wall time is preserved in `outputs/final/final_training_execution.json`.
"""
Path("reports/commands.md").write_text(commands, encoding="utf-8")

cpu = json.loads(environment["cpu"]["stdout"])
memory = json.loads(environment["memory"]["stdout"])
training_execution = json.loads(Path("outputs/final/final_training_execution.json").read_text())
figures = sorted(Path("outputs/final/figures").glob("*.png"))
assert len(figures) == 5
figure_links = "\n".join("- [{}]({}) and [PDF]({})".format(p.stem, p.as_posix(), p.with_suffix(".pdf").as_posix()) for p in figures)
package_text = "Python 3.8.10; Torch 1.13.1+cpu; NumPy 1.24.3; pandas 2.0.3; SciPy 1.10.1; Matplotlib 3.7.2; scikit-learn 1.3.0; joblib 1.4.2."
total_seconds = sum(seconds for _, seconds in runtime_rows)
report = """# Run report

Completion is confirmed from saved artifacts, counts, logs, and hashes, not from a command being launched. **All 120 final run-by-distance pairs completed**, with 30 paired blocked/IP observations at each k=1,2,4,7. No final missing runs, recorded mechanical failures, or exclusions.

## 1. Machine and environment

{os}; platform string `{platform}`. CPU: {cpu}, {cores} cores / {logical} logical processors. OS-reported total visible RAM {ram:.2f} GiB; available RAM at environment capture {free:.2f} GiB. All runs used CPU, float32, one Torch intra-op and one inter-op thread, deterministic algorithms. CUDA and MPS availability were both false in the CPU build. No accelerator was used. The 100-update CPU benchmark measured {bench} seconds for 1/2/4 threads respectively; one thread was retained.

{packages} Full freeze and Torch build details: `outputs/environment/pip_freeze.txt`, `outputs/environment/system_info.json`. Requested Conda specification: `envs/reproduction.yml`. Conda export was unavailable (WinError 2), and no Conda environment is falsely claimed. Exact-version Windows alternative: `envs/requirements-exact.txt` and `scripts/setup_windows.ps1`.

## 2. Source provenance

Exact inspected source commit: `31f58bb6de90d1fffcb752c0b97afb7df51479a7`. Repository https://github.com/YJ-0000/RNN_during_RP_and_IP . Audit and file hashes: `reports/audit.md`, `outputs/provenance/source_files.json`. The author checkout is ignored and remains only in `external/author_repo`. No root LICENSE was found. Independent source and original Matplotlib figures are supplied; author source/figures are not distributed.

## 3. Frozen final configuration

Byte-level SHA-256 of `configs/final.json`: **`{hash}`**. Frozen marker: `configs/final.sha256`. Training implementation fingerprint: `{implementation}`. Both were checked before final training and again during verification; no final settings were changed after results became visible. Saved blocked-order assignment has each of the six orders exactly five times at each distance.

## 4. Commands

See [the executed command ledger](reports/commands.md) for exact commands. Commands include pinned checkout, environment attempts, system capture, benchmark, unittest, five-run source audit, 100-run resume, smoke/pilot training and analysis, final freeze/train/analyze, independent artifact verification, final verification-only resume, and executed notebook generation. Per-phase execution metadata and console logs are saved.

## 5. Unit tests

`python -m unittest discover -s tests -v`: **14/14 passed** before real simulation and again at delivery. Logs: `reports/unit_tests_initial.txt`, `reports/unit_tests_final.txt`. Coverage includes distances, overlap, paired states, exposure/update equality, noise targets, toy metrics, evaluation immutability, exact eight-transition timing, balanced orders, seed namespaces, sustained threshold, and interrupted-versus-uninterrupted equality. Local author-model forward outputs and parameter gradients match exactly in the parity test.

## 6–9. Source, smoke, pilot, and final status

{counts}

Source status: five-run audit followed by **100 newly seeded source-compatible main-training runs**, with all expected saved shapes and aggregate comparisons. Historical seed identity is not reconstructable. Optional source perturbation/vulnerability protocols and Figure 2 sensitivity were **not executed**; reference vulnerability arrays were audited separately. Comparison evidence: `outputs/source_exact/comparison/first_five/` and `full_100/`.

Smoke: one reduced-budget run at each distance, all stages and five figure pairs completed; smoke supplies no findings. Pilot: five disjoint IDs at each distance, full main/D/evaluation budgets, 20 cells verified. No scientific settings were adjusted after pilot. Final: 120 paired cells / 240 schedule models, each with 750 main updates and 100 D updates. Every final ID is from `final_v1`, disjoint from `smoke_v1` and `pilot_v1`. Numeric IDs may repeat across phase names, but their seed namespaces do not.

## 10–11. Failures, fixes, and exclusions

Conda could not launch; the local NuGet venv attempt failed at ensurepip. Direct use of the isolated Python 3.8.10 runtime solved environment setup while retaining all requested versions. The notebook's first builder execution had a newline-escaping syntax error, repaired by using raw code-template strings and then successfully executing all eight code cells. Figure QA found one annotation/legend overlap; moving the annotation fixed it without changing data or analysis. No simulation seeds were rerun for an undesirable scientific outcome. There were zero recorded simulation failure records and zero exclusions. The tiny test intentionally interrupts and resumes; final resume re-verifies/skips all completed cells without optimizer steps. Full ledger: `reports/deviations.md`.

## 12. Measured runtime

Summed cell execution time across source, smoke, pilot, and final: **{total:.2f} seconds ({minutes:.2f} minutes)**. Per-phase measured values are in the status table. Original final-training wall time: **{finalwall:.2f} seconds**. These are measured simulation execution times; they exclude environment downloads, code-writing time, and report/figure QA. No GPU speed estimate is asserted. Per-run times: `outputs/<phase>/analysis/execution_times.csv`; original final execution: `outputs/final/final_training_execution.json`.

## 13. Raw data and resumability

`outputs/final/runs/run_NNN_kK/` stores all base sequences, compressed generated data, named seed manifests, identical example orders, pretrained weights, main/D checkpoints, raw CSVs, pair checks, and complete artifact hashes. Every checkpoint includes model/optimizer/RNG states, current update, config and data-manifest hashes, run ID, distance, schedule, blocked order, and source commit. Latest checkpoints were saved at least every 50 main updates. Resume guards reject mismatched metadata by default; an explicit `--force` is recorded as a deviation. Final config/implementation guards remain enforced independently. Use `scripts/run_experiment.py --config configs/final.json --resume` to verify existing or continue incomplete work.

Merged raw CSVs, run-level metrics, paired differences, bootstrap indices and values, summary tables, trend, threshold counts, position summaries, and verification reports are in `outputs/final/analysis/`. These generated outputs are preserved locally but Git-ignored; retain/copy `outputs/` when distributing results, since source-only checkout does not include them. The distributable `artifacts/motor_rnn_project.zip` includes generated outputs and own project source, with an internal per-file SHA-256 manifest and archive CRC verification. It excludes the author checkout, local Python/package runtime, venv, and Git metadata; its external checksum is in `artifacts/delivery_manifest.json`.

## 14–16. Primary paired differences, intervals, and trend

{primary}

The prespecified fitted slope is **{slope:.3f} pp per Hamming unit**, bootstrap 95% CI **[{low:.3f}, {high:.3f}]**. Per-distance CIs and slope CI use 10,000 resamples of simulation IDs; the slope preserves all four distances within each sampled ID. Every individual paired difference is in `outputs/final/analysis/paired_differences.csv`. Full resampling arrays and indices are in `bootstrap.npz`. No noisy-copy/position pseudoreplication, p-value, or invented statistical-significance claim is used.

## 17. Secondary results

{secondary}

{speed}

A/B/C and positions 1–7 are retained in raw metrics. Shared/differing errors are separate; k=7 shared accuracy is NaN with zero observations. Clean/noisy scores remain distinct. IP has lower mean complete-sequence accuracy at k=2,4,7 despite higher mean action accuracy. Mean noisy D forgetting is greater under IP at k=1,4,7; these secondary drops depend on baseline, with greater accuracy available to lose under IP. D similarity was not manipulated. See `reports/results.md` for interpretation.

## 18. Figure paths and notebook

{figures}

PNG/PDF pairs were generated from saved data and visually inspected. Notebook: `notebooks/walkthrough.ipynb`, with sequentially executed saved-data cells, explanations for a Python beginner, and no expensive training.

## 19. Known deviations

The exact numerical environment was achieved via pip/local Windows Python instead of Conda; the source's Linux build strings were not imported. Reproduction adds deterministic named seed bundles and sequential one-thread execution, and does not claim exact author random-stream identity. Source uniqueness and training-input retention are deliberately preserved; strict disjointness and fresh noise apply to the controlled extension. Optional source vulnerability and 10,000-pretraining sensitivity were not rerun. Reduced smoke budgets are intentional, and smoke/pilot outputs are excluded from final inference. Latest resumable checkpoints overwrite earlier checkpoints rather than archiving every 50-update state. There were no post-pilot scientific parameter changes. The complete deviations ledger is `reports/deviations.md`.

## 20. Limitations

{limitations}
""".format(os=memory["Caption"], platform=environment["os"], cpu=cpu["Name"].strip(), cores=cpu["NumberOfCores"], logical=cpu["NumberOfLogicalProcessors"],
           ram=memory["TotalVisibleMemorySize"] / 1024 ** 2, free=memory["FreePhysicalMemory"] / 1024 ** 2,
           bench="/".join("{:.4f}".format(r["seconds"]) for r in benchmark["measurements"]), packages=package_text,
           hash=config_hash, implementation=json.loads(Path("configs/final.json").read_text())["implementation_sha256"],
           counts=count_table, total=total_seconds, minutes=total_seconds / 60, finalwall=training_execution["wall_seconds"],
           primary=primary_table, slope=trend["slope_pp_per_hamming"], low=trend["ci_low"], high=trend["ci_high"],
           secondary=secondary_table, speed=speed_table, figures=figure_links, limitations=limitation_text)
Path("RUN_REPORT.md").write_text(report, encoding="utf-8")
print("Reports written from verified final outputs; SHA-256", config_hash)
