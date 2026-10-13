# Run report

Completion is confirmed from saved artifacts, counts, logs, and hashes, not from a command being launched. **All 120 final run-by-distance pairs completed**, with 30 paired blocked/IP observations at each k=1,2,4,7. No final missing runs, recorded mechanical failures, or exclusions.

## 1. Machine and environment

Microsoft Windows 11 Home; platform string `Windows-10-10.0.26200-SP0`. CPU: AMD Ryzen 7 8700F 8-Core Processor, 8 cores / 16 logical processors. OS-reported total visible RAM 15.59 GiB; available RAM at environment capture 2.16 GiB. All runs used CPU, float32, one Torch intra-op and one inter-op thread, deterministic algorithms. CUDA and MPS availability were both false in the CPU build. No accelerator was used. The 100-update CPU benchmark measured 0.1274/0.1447/0.1392 seconds for 1/2/4 threads respectively; one thread was retained.

Python 3.8.10; Torch 1.13.1+cpu; NumPy 1.24.3; pandas 2.0.3; SciPy 1.10.1; Matplotlib 3.7.2; scikit-learn 1.3.0; joblib 1.4.2. Full freeze and Torch build details: `outputs/environment/pip_freeze.txt`, `outputs/environment/system_info.json`. Requested Conda specification: `envs/reproduction.yml`. Conda export was unavailable (WinError 2), and no Conda environment is falsely claimed. Exact-version Windows alternative: `envs/requirements-exact.txt` and `scripts/setup_windows.ps1`.

## 2. Source provenance

Exact inspected source commit: `31f58bb6de90d1fffcb752c0b97afb7df51479a7`. Repository https://github.com/YJ-0000/RNN_during_RP_and_IP . Audit and file hashes: `reports/audit.md`, `outputs/provenance/source_files.json`. The author checkout is ignored and remains only in `external/author_repo`. No root LICENSE was found. Independent source and original Matplotlib figures are supplied; author source/figures are not distributed.

## 3. Frozen final configuration

Byte-level SHA-256 of `configs/final.json`: **`907b6540c9f49d8554105803fad58966c900c67eb260e9f510d379f4730ceecc`**. Frozen marker: `configs/final.sha256`. Training implementation fingerprint: `e16274af61696a90c379fb16165ca1ff8309e0f1f02d8d4526da050588d4d168`. Both were checked before final training and again during verification; no final settings were changed after results became visible. Saved blocked-order assignment has each of the six orders exactly five times at each distance.

## 4. Commands

See [the executed command ledger](reports/commands.md) for exact commands. Commands include pinned checkout, environment attempts, system capture, benchmark, unittest, five-run source audit, 100-run resume, smoke/pilot training and analysis, final freeze/train/analyze, independent artifact verification, final verification-only resume, and executed notebook generation. Per-phase execution metadata and console logs are saved.

## 5. Unit tests

`python -m unittest discover -s tests -v`: **14/14 passed** before real simulation and again at delivery. Logs: `reports/unit_tests_initial.txt`, `reports/unit_tests_final.txt`. Coverage includes distances, overlap, paired states, exposure/update equality, noise targets, toy metrics, evaluation immutability, exact eight-transition timing, balanced orders, seed namespaces, sustained threshold, and interrupted-versus-uninterrupted equality. Local author-model forward outputs and parameter gradients match exactly in the parity test.

## 6–9. Source, smoke, pilot, and final status

| Phase | Completed paired cells | Missing | Failure records | Measured cell runtime (s) |
| --- | --- | --- | --- | --- |
| source_exact | 100 | 0 | 0 | 235.75 |
| smoke | 4 | 0 | 0 | 0.52 |
| pilot | 20 | 0 | 0 | 60.51 |
| final | 120 | 0 | 0 | 358.03 |

Source status: five-run audit followed by **100 newly seeded source-compatible main-training runs**, with all expected saved shapes and aggregate comparisons. Historical seed identity is not reconstructable. Optional source perturbation/vulnerability protocols and Figure 2 sensitivity were **not executed**; reference vulnerability arrays were audited separately. Comparison evidence: `outputs/source_exact/comparison/first_five/` and `full_100/`.

Smoke: one reduced-budget run at each distance, all stages and five figure pairs completed; smoke supplies no findings. Pilot: five disjoint IDs at each distance, full main/D/evaluation budgets, 20 cells verified. No scientific settings were adjusted after pilot. Final: 120 paired cells / 240 schedule models, each with 750 main updates and 100 D updates. Every final ID is from `final_v1`, disjoint from `smoke_v1` and `pilot_v1`. Numeric IDs may repeat across phase names, but their seed namespaces do not.

## 10–11. Failures, fixes, and exclusions

Conda could not launch; the local NuGet venv attempt failed at ensurepip. Direct use of the isolated Python 3.8.10 runtime solved environment setup while retaining all requested versions. The notebook's first builder execution had a newline-escaping syntax error, repaired by using raw code-template strings and then successfully executing all eight code cells. Figure QA found one annotation/legend overlap; moving the annotation fixed it without changing data or analysis. No simulation seeds were rerun for an undesirable scientific outcome. There were zero recorded simulation failure records and zero exclusions. The tiny test intentionally interrupts and resumes; final resume re-verifies/skips all completed cells without optimizer steps. Full ledger: `reports/deviations.md`.

## 12. Measured runtime

Summed cell execution time across source, smoke, pilot, and final: **654.80 seconds (10.91 minutes)**. Per-phase measured values are in the status table. Original final-training wall time: **370.10 seconds**. These are measured simulation execution times; they exclude environment downloads, code-writing time, and report/figure QA. No GPU speed estimate is asserted. Per-run times: `outputs/<phase>/analysis/execution_times.csv`; original final execution: `outputs/final/final_training_execution.json`.

## 13. Raw data and resumability

`outputs/final/runs/run_NNN_kK/` stores all base sequences, compressed generated data, named seed manifests, identical example orders, pretrained weights, main/D checkpoints, raw CSVs, pair checks, and complete artifact hashes. Every checkpoint includes model/optimizer/RNG states, current update, config and data-manifest hashes, run ID, distance, schedule, blocked order, and source commit. Latest checkpoints were saved at least every 50 main updates. Resume guards reject mismatched metadata by default; an explicit `--force` is recorded as a deviation. Final config/implementation guards remain enforced independently. Use `scripts/run_experiment.py --config configs/final.json --resume` to verify existing or continue incomplete work.

Merged raw CSVs, run-level metrics, paired differences, bootstrap indices and values, summary tables, trend, threshold counts, position summaries, and verification reports are in `outputs/final/analysis/`. These generated outputs are preserved locally but Git-ignored; retain/copy `outputs/` when distributing results, since source-only checkout does not include them. The distributable `artifacts/motor_rnn_project.zip` includes generated outputs and own project source, with an internal per-file SHA-256 manifest and archive CRC verification. It excludes the author checkout, local Python/package runtime, venv, and Git metadata; its external checksum is in `artifacts/delivery_manifest.json`.

## 14–16. Primary paired differences, intervals, and trend

| k | Paired runs | Blocked noisy (%) | IP noisy (%) | IP−blocked mean (pp) | Median (pp) | 95% CI (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 30 | 88.45 | 90.56 | 2.11 | 0.00 | [0.10, 4.26] |
| 2 | 30 | 78.86 | 82.98 | 4.13 | 3.57 | [1.27, 6.91] |
| 4 | 30 | 60.88 | 75.91 | 15.04 | 11.69 | [10.89, 19.17] |
| 7 | 30 | 41.00 | 77.72 | 36.73 | 36.64 | [32.53, 40.77] |

The prespecified fitted slope is **5.933 pp per Hamming unit**, bootstrap 95% CI **[5.199, 6.655]**. Per-distance CIs and slope CI use 10,000 resamples of simulation IDs; the slope preserves all four distances within each sampled ID. Every individual paired difference is in `outputs/final/analysis/paired_differences.csv`. Full resampling arrays and indices are in `bootstrap.npz`. No noisy-copy/position pseudoreplication, p-value, or invented statistical-significance claim is used.

## 17. Secondary results

| k | Schedule | Clean trained (%) | Noisy whole-sequence (%) | Clean unseen (%) | Noisy unseen (%) | Noisy CE | Clean D drop (pp) | Noisy D drop (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | blocked | 88.41 | 30.00 | 25.19 | 25.20 | 0.401 | 43.17 | 43.27 |
| 1 | interleaved | 90.63 | 41.71 | 25.40 | 25.40 | 0.260 | 46.03 | 45.69 |
| 2 | blocked | 78.89 | 25.19 | 25.21 | 25.21 | 0.702 | 37.62 | 37.73 |
| 2 | interleaved | 83.02 | 20.69 | 25.49 | 25.53 | 0.446 | 37.30 | 37.26 |
| 4 | blocked | 60.79 | 30.00 | 26.47 | 26.41 | 1.249 | 22.70 | 22.86 |
| 4 | interleaved | 75.87 | 14.87 | 27.82 | 27.77 | 0.628 | 31.27 | 31.22 |
| 7 | blocked | 40.79 | 22.56 | 25.98 | 25.97 | 1.837 | 7.78 | 7.71 |
| 7 | interleaved | 77.78 | 20.52 | 28.77 | 28.81 | 0.675 | 36.98 | 37.46 |

| k | Schedule | Reached sustained 90% (out of 30) |
| --- | --- | --- |
| 1 | blocked | 27 |
| 1 | interleaved | 28 |
| 2 | blocked | 0 |
| 2 | interleaved | 7 |
| 4 | blocked | 0 |
| 4 | interleaved | 5 |
| 7 | blocked | 0 |
| 7 | interleaved | 3 |

A/B/C and positions 1–7 are retained in raw metrics. Shared/differing errors are separate; k=7 shared accuracy is NaN with zero observations. Clean/noisy scores remain distinct. IP has lower mean complete-sequence accuracy at k=2,4,7 despite higher mean action accuracy. Mean noisy D forgetting is greater under IP at k=1,4,7; these secondary drops depend on baseline, with greater accuracy available to lose under IP. D similarity was not manipulated. See `reports/results.md` for interpretation.

## 18. Figure paths and notebook

- [D_forgetting](outputs/final/figures/D_forgetting.png) and [PDF](outputs/final/figures/D_forgetting.pdf)
- [final_accuracy](outputs/final/figures/final_accuracy.png) and [PDF](outputs/final/figures/final_accuracy.pdf)
- [learning_curves](outputs/final/figures/learning_curves.png) and [PDF](outputs/final/figures/learning_curves.pdf)
- [paired_differences](outputs/final/figures/paired_differences.png) and [PDF](outputs/final/figures/paired_differences.pdf)
- [position_errors](outputs/final/figures/position_errors.png) and [PDF](outputs/final/figures/position_errors.pdf)

PNG/PDF pairs were generated from saved data and visually inspected. Notebook: `notebooks/walkthrough.ipynb`, with sequentially executed saved-data cells, explanations for a Python beginner, and no expensive training.

## 19. Known deviations

The exact numerical environment was achieved via pip/local Windows Python instead of Conda; the source's Linux build strings were not imported. Reproduction adds deterministic named seed bundles and sequential one-thread execution, and does not claim exact author random-stream identity. Source uniqueness and training-input retention are deliberately preserved; strict disjointness and fresh noise apply to the controlled extension. Optional source vulnerability and 10,000-pretraining sensitivity were not rerun. Reduced smoke budgets are intentional, and smoke/pilot outputs are excluded from final inference. Latest resumable checkpoints overwrite earlier checkpoints rather than archiving every 50-update state. There were no post-pilot scientific parameter changes. The complete deviations ledger is `reports/deviations.md`.

## 20. Limitations

The complete sequence is provided as the initial cue, so generalization is decoding unseen numeric patterns rather than acquiring an unobserved physical skill. Seven hidden units, default initialization, one optimizer, and one fixed budget limit generality. Action frequencies, adjacent repeats, and longest runs are recorded but not matched across distances; they can accompany this generator's manipulation. Numeric cue values impose an ordinal geometry on action labels. Low-distance performance can approach a ceiling, and whole-sequence accuracy can be at a floor. Bootstrap uncertainty is across artificial simulation IDs, not people. D distances were observed, not manipulated, and raw forgetting depends on baseline accuracy. There is no offline time-dependent consolidation or direct biological evidence. These findings do not establish how human brains work and are not a broad novelty claim.
