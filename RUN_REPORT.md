# CruxRNN: Sequence Similarity, Practice Scheduling, and Interference in a Minimal Recurrent Neural Network

**Approved confirmatory final experiment complete.**

## Phase accounting

| Stage | Completed paired cells | Role |
| --- | --- | --- |
| Source reproduction | 100 (first five audited) | Audited source behavior; uncontrolled similarity |
| Smoke | 4 | Pipeline only |
| Pilot | 20 (five IDs × four k) | Diagnostic only |
| Earlier exploratory | 120 | Exposed outcomes; producing SHA/timestamps missing; excluded from new inference |
| New confirmatory final | 120 (30 IDs × four k) | Fresh approved namespace, complete contemporary provenance |

## Approval and frozen provenance

Pre-final checkpoint: `fa3b7c34496274569944b85c74d2c6c4cd443fe4` (pushed and remotely verified before execution). Producing execution commit: `28a2b07bb40afcf7af9b907612e01ce722902bcc`; it adds approval and reporting metadata, and every numerical module is byte-identical to the pre-final checkpoint. Config SHA-256: `26f8cc56ad115f9e4bb4bb47b30af4997c205cd2906cb8e5195bf24fda8ff006`. Namespace: `cruxrnn_final_proposed_v1`. Frozen numerical fingerprint: `e16274af61696a90c379fb16165ca1ff8309e0f1f02d8d4526da050588d4d168`. Start/end UTC: `2026-10-13T23:51:43.738254+00:00` / `2026-10-13T23:58:05.047976+00:00`. Expected/completed: 120/120 pairs, 240 main models. Failed/excluded/resumed/replaced-successful cells: 0/0/0/0. All artifact hashes, seeds, orders, disjoint bases, pair initializations, metric identities and bootstrap row resampling were independently checked.

The approval record is `provenance/final_approval.json`. The original proposal remains unchanged as historical pre-execution evidence in `provenance/final_proposal.json`. The launch record is under `results/confirmatory/execution_records/`.

## Design and analysis

Thirty independent replicate IDs each span k=1,2,4,7: 120 paired cells and 240 main models. Each cell pretrains once, clones identical weights, and presents the same 15,000 noisy examples in blocked or interleaved order. Each of six blocked orders occurs five times per k. SGD .02, zero momentum, batch 20; 750 main updates; probes every 25; 100 matched D updates. Primary: equally weighted A/B/C fresh-noisy final action accuracy before D. Ten thousand percentile bootstrap resamples use whole replicate IDs, retaining all four distances for the prespecified linear slope. SD uses ddof=1. No copies/positions serve as independent runs.

| k | pairs | Blocked % | IP % | Mean IP−blocked pp | Median pp | SD pp | 95% CI low pp | 95% CI high pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 30 | 89.002 | 91.116 | 2.114 | 0.000 | 9.097 | -1.086 | 5.437 |
| 2 | 30 | 80.221 | 86.651 | 6.430 | 4.357 | 7.930 | 3.878 | 9.418 |
| 4 | 30 | 61.006 | 76.019 | 15.013 | 14.238 | 9.451 | 11.722 | 18.459 |
| 7 | 30 | 40.163 | 79.029 | 38.865 | 38.905 | 12.254 | 34.549 | 43.122 |

The intervals include zero at k=1; the direction at these distances remains uncertain. The categorical mean differences increase across the four distances.

The prespecified slope is **6.124 pp per Hamming unit**, with a run-ID bootstrap 95% CI **[5.383, 6.853]**.

## Secondary outcomes

| distance | schedule | final_clean_action_accuracy | final_noisy_sequence_accuracy | final_cross_entropy | generalization_clean_accuracy | generalization_noisy_accuracy | forgetting_clean | forgetting_noisy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | blocked | 89.048 | 31.111 | 0.392 | 25.257 | 25.284 | 47.937 | 47.741 |
| 1 | interleaved | 91.270 | 43.367 | 0.251 | 25.633 | 25.658 | 42.857 | 42.587 |
| 2 | blocked | 80.159 | 31.111 | 0.667 | 25.357 | 25.334 | 40.476 | 40.532 |
| 2 | interleaved | 86.667 | 29.978 | 0.377 | 26.610 | 26.614 | 36.349 | 36.251 |
| 4 | blocked | 60.952 | 30.000 | 1.206 | 25.262 | 25.271 | 25.556 | 25.889 |
| 4 | interleaved | 76.032 | 12.056 | 0.638 | 27.062 | 27.034 | 34.286 | 34.475 |
| 7 | blocked | 39.841 | 27.044 | 1.919 | 26.462 | 26.458 | 9.206 | 9.167 |
| 7 | interleaved | 79.048 | 24.633 | 0.656 | 29.248 | 29.203 | 42.857 | 42.744 |

| k | schedule | reached / 30 | median first qualifying update (reached only) |
| --- | --- | --- | --- |
| 1 | blocked | 28 / 30 | 287.500 |
| 1 | interleaved | 28 / 30 | 350.000 |
| 2 | blocked | 0 / 30 | NaN |
| 2 | interleaved | 9 / 30 | 550.000 |
| 4 | blocked | 0 / 30 | NaN |
| 4 | interleaved | 1 / 30 | 650.000 |
| 7 | blocked | 0 / 30 | NaN |
| 7 | interleaved | 4 / 30 | 575.000 |

IP has lower mean noisy complete-sequence accuracy at k=2,4,7, despite its higher mean action accuracy. Mean noisy D forgetting is greater under IP at k=4,7; baseline accuracy limits how much a model can lose. Mean clean unseen-base action accuracy spans 25.26–29.25% and noisy accuracy spans 25.27–29.20%, close to the 25% four-action baseline. These descriptive results limit broad claims of better learning, faster learning or transfer.

Full A/B/C, clean/sequence/CE/position/generalization/D summaries and interpretation: `reports/results.md`. k=7 shared accuracy is NaN with count zero.

## Tests and verification

The preflight suite passed 17 tests with no failures/skips, including reference model output/gradient parity, exact tiny interrupted/resumed training, pairing, exposure, targets, disjointness, seed namespaces, metrics, thresholds and approval/provenance controls. Final test log: `reports/unit_tests_confirmatory_final.txt`. Independent full-data audit: `results/confirmatory/design_verification.json`. All 120 paired cells and completion hashes were accounted for before final inference. No failure, exclusion, resume, force override, or replacement of successful cells occurred.

## Environment and measured runtime

Windows 11 Home; AMD Ryzen 7 8700F (8 cores / 16 logical processors). Python 3.8.10; PyTorch 1.13.1+cpu; NumPy 1.24.3; pandas 2.0.3; SciPy 1.10.1; Matplotlib 3.7.2; scikit-learn 1.3.0; joblib 1.4.2. CPU float32, one intra-op and one inter-op thread, deterministic algorithms. The system snapshot reports interpreter defaults separately; the runner explicitly sets both experiment thread counts to one. Current freeze/system evidence: `provenance/environment/confirmatory/`. Conda was unavailable; an isolated local Windows Python and pip provided the principal requested versions.

Phase wall time: **381.342 seconds (6.36 minutes)**. Summed cell execution: **369.015 seconds**. These measured times cover data generation, pretraining, main training, probes, generalization, D and cell saves; they exclude environment setup, analysis, plotting, reporting, and GitHub delivery. Analysis/figure generation took an additional **9.943 seconds**, for **391.285 seconds** of experiment plus analysis/plotting; independent verification/reporting/archive/delivery time is separate.

## Files and figures

Full raw arrays, checkpoints, seeds, per-cell CSVs and completion hashes: `outputs/cruxrnn_final_proposed_v1/runs/run_NNN_kK/` (Git-ignored). Full analysis including bootstrap indices/arrays: `outputs/cruxrnn_final_proposed_v1/analysis/`. Tracked results: `results/confirmatory/`; all 120 individual differences are in `paired_differences.csv`. Per-update CSVs are losslessly compressed as `training.csv.gz`, `development.csv.gz`, and `D_development.csv.gz` (pandas `read_csv` detects gzip). The artifact index, 120 manifests/completion records, published-file hashes and contemporary execution record link those products to code/config/data. The earlier exploratory files remain separately in `outputs/final/` and `reports/history/prior_*`; none were pooled. Historical source/smoke/pilot metadata gaps remain explicit.

- `results/confirmatory/figures/learning_curves.[png,pdf]`
- `results/confirmatory/figures/final_accuracy.[png,pdf]`
- `results/confirmatory/figures/paired_differences.[png,pdf]`
- `results/confirmatory/figures/D_forgetting.[png,pdf]`
- `results/confirmatory/figures/position_errors.[png,pdf]`

All five figures use only the new confirmation and export 300-dpi PNG plus vector PDF. Learning curves show mean ± standard error across 30 IDs (descriptive); paired outcome intervals are the prespecified bootstrap intervals.

## Source audit and earlier phases

Source repository inspected at `31f58bb6de90d1fffcb752c0b97afb7df51479a7`; six inspected file hashes in `provenance/author_source_files.json`. First five and full 100-run numerical reproduction comparisons are in `results/source_reproduction/first_five/` and `full_100/`. Original uniqueness and training-input retention limitations were preserved in source-exact mode. Smoke checks every pipeline stage; the 20-pair pilot supplied diagnostics and a runtime estimate. Neither enters these final figures or inference. Earlier execution did not record producing Git SHAs/start/end timestamps; later snapshots/reverifications are honestly distinguished in `provenance/stages.json`. No missing historical provenance was invented.

## Reproduction

Set up `envs/reproduction.yml` or `scripts/setup_windows.ps1`, activate the environment, and use its Python. On this machine:

```powershell
$taskPython = Join-Path (Get-Location) 'envs/tools/python3810/tools/python.exe'
& $taskPython scripts/system_info.py
& $taskPython -m unittest discover -s tests -v
& $taskPython scripts/run_experiment.py --config configs/final_proposed.json --resume
& $taskPython scripts/analyze.py --config configs/final_proposed.json
& $taskPython scripts/verify_design.py --config configs/final_proposed.json
& $taskPython scripts/publish_confirmatory.py
& $taskPython scripts/create_walkthrough.py
```

Start the runner from a clean Git checkout. On a fresh checkout it generates all 120 cells with the same frozen namespace; locally `--resume` verifies/skips completed cells. Only incomplete computation may continue. Retain raw outputs for bitwise artifact comparison. A producing Git SHA in a new reproduction will identify that new checkout, while the scientific config/numerical fingerprint remain fixed.

## Deviations and limitations

Only descriptive SD/group tables and 300-dpi export were added before this new experiment. Scientific settings, numerical modules, blocked assignment and bootstrap procedure were unchanged. Optional source vulnerability and 10,000-pretraining sensitivity remain unexecuted. Full ledger: `reports/deviations.md`.

The complete target sequence is supplied in the initial cue. These results describe numerical cue decoding by one small RNN with a fixed budget, ordinal action codes, and unmatched action frequencies/repeats. D similarity is observed rather than controlled. Thirty simulation IDs quantify uncertainty within this model/generator; noisy copies and positions are not independent subjects. Prior exploratory outcomes were seen, so this is a disclosed independent confirmation, not a retroactively untouched preregistration. No human brain mechanism or climbing-performance effect is demonstrated.

## Final commit and release provenance

The results commit SHA will be recorded in one explicit follow-up documentation commit after results are committed. The release tag will identify the final documentation snapshot; a commit cannot contain its own hash.

## Full raw-data release asset

`cruxrnn-confirmatory-data.zip` contains only the new confirmation: arrays, all checkpoints, CSVs, bootstrap indices/values, figures and contemporary execution records. Size: 117456062 bytes; 1956 members; CRC verified. SHA-256: `092bfe2324a77e6782d80034b3cf4ec5fb79017990d625be8803f0eb5f96b676`. Each payload is indexed by SHA-256 in `archive_manifest.json`. Extract at the release checkout root. No earlier exploratory data, author checkout or runtime is included. External receipt: `artifacts/confirmatory_archive.json`.

Final delivery checks also verified every published-file hash, lossless gzip payloads, independently recomputed mean/median/sample-SD/bootstrap-CI columns, and all eight notebook cells. All five final PNGs were visually inspected without finding layout issues.
