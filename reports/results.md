# CruxRNN: confirmatory final results

Thirty independent replicate IDs each span k=1,2,4,7: 120 paired cells and 240 main models. Each cell pretrains once, clones identical weights, and presents the same 15,000 noisy examples in blocked or interleaved order. Each of six blocked orders occurs five times per k. SGD .02, zero momentum, batch 20; 750 main updates; probes every 25; 100 matched D updates. Primary: equally weighted A/B/C fresh-noisy final action accuracy before D. Ten thousand percentile bootstrap resamples use whole replicate IDs, retaining all four distances for the prespecified linear slope. SD uses ddof=1. No copies/positions serve as independent runs.

| Stage | Completed paired cells | Role |
| --- | --- | --- |
| Source reproduction | 100 (first five audited) | Audited source behavior; uncontrolled similarity |
| Smoke | 4 | Pipeline only |
| Pilot | 20 (five IDs × four k) | Diagnostic only |
| Earlier exploratory | 120 | Exposed outcomes; producing SHA/timestamps missing; excluded from new inference |
| New confirmatory final | 120 (30 IDs × four k) | Fresh approved namespace, complete contemporary provenance |

## Primary outcome

| k | pairs | Blocked % | IP % | Mean IP−blocked pp | Median pp | SD pp | 95% CI low pp | 95% CI high pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 30 | 89.002 | 91.116 | 2.114 | 0.000 | 9.097 | -1.086 | 5.437 |
| 2 | 30 | 80.221 | 86.651 | 6.430 | 4.357 | 7.930 | 3.878 | 9.418 |
| 4 | 30 | 61.006 | 76.019 | 15.013 | 14.238 | 9.451 | 11.722 | 18.459 |
| 7 | 30 | 40.163 | 79.029 | 38.865 | 38.905 | 12.254 | 34.549 | 43.122 |

The intervals include zero at k=1; the direction at these distances remains uncertain. The categorical mean differences increase across the four distances.

The prespecified slope is **6.124 pp per Hamming unit**, with a run-ID bootstrap 95% CI **[5.383, 6.853]**. No p-values or multiplicity-adjusted familywise claims are made.

## Secondary outcomes

Accuracy and forgetting columns below are percentages or percentage points; cross-entropy is an unscaled mean loss.

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

Clean action scores, complete-sequence success, cross-entropy, unseen-base generalization and D forgetting answer different questions. The raw/group tables additionally retain clean complete-sequence accuracy, generalization sequence accuracy, every A/B/C score, positions 1–7, and before/after-D evaluations. These secondary summaries are descriptive. IP has lower mean noisy complete-sequence accuracy at k=2,4,7, despite its higher mean action accuracy. Mean noisy D forgetting is greater under IP at k=4,7; baseline accuracy limits how much a model can lose. Mean clean unseen-base action accuracy spans 25.26–29.25% and noisy accuracy spans 25.27–29.20%, close to the 25% four-action baseline. These descriptive results limit broad claims of better learning, faster learning or transfer.

### A/B/C before D

Action/sequence accuracy are shown in percent.

| distance | schedule | split | group | action_accuracy | sequence_accuracy | cross_entropy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | blocked | final_clean | A | 89.048 | 30.000 | 0.399 |
| 1 | blocked | final_clean | B | 88.571 | 30.000 | 0.424 |
| 1 | blocked | final_clean | C | 89.524 | 33.333 | 0.353 |
| 1 | blocked | final_noisy | A | 89.048 | 30.000 | 0.399 |
| 1 | blocked | final_noisy | B | 88.552 | 30.000 | 0.424 |
| 1 | blocked | final_noisy | C | 89.405 | 33.333 | 0.353 |
| 1 | interleaved | final_clean | A | 91.905 | 50.000 | 0.250 |
| 1 | interleaved | final_clean | B | 90.000 | 36.667 | 0.271 |
| 1 | interleaved | final_clean | C | 91.905 | 46.667 | 0.232 |
| 1 | interleaved | final_noisy | A | 91.805 | 49.300 | 0.250 |
| 1 | interleaved | final_noisy | B | 89.710 | 34.633 | 0.272 |
| 1 | interleaved | final_noisy | C | 91.833 | 46.167 | 0.233 |
| 2 | blocked | final_clean | A | 79.048 | 30.000 | 0.677 |
| 2 | blocked | final_clean | B | 80.952 | 33.333 | 0.622 |
| 2 | blocked | final_clean | C | 80.476 | 30.000 | 0.702 |
| 2 | blocked | final_noisy | A | 79.095 | 30.000 | 0.677 |
| 2 | blocked | final_noisy | B | 80.943 | 33.333 | 0.622 |
| 2 | blocked | final_noisy | C | 80.624 | 30.000 | 0.702 |
| 2 | interleaved | final_clean | A | 87.143 | 30.000 | 0.369 |
| 2 | interleaved | final_clean | B | 87.619 | 36.667 | 0.361 |
| 2 | interleaved | final_clean | C | 85.238 | 23.333 | 0.399 |
| 2 | interleaved | final_noisy | A | 87.305 | 30.767 | 0.370 |
| 2 | interleaved | final_noisy | B | 87.495 | 35.867 | 0.362 |
| 2 | interleaved | final_noisy | C | 85.152 | 23.300 | 0.400 |
| 4 | blocked | final_clean | A | 62.381 | 33.333 | 1.231 |
| 4 | blocked | final_clean | B | 59.048 | 23.333 | 1.217 |
| 4 | blocked | final_clean | C | 61.429 | 33.333 | 1.168 |
| 4 | blocked | final_noisy | A | 62.176 | 33.333 | 1.231 |
| 4 | blocked | final_noisy | B | 59.524 | 23.333 | 1.217 |
| 4 | blocked | final_noisy | C | 61.319 | 33.333 | 1.169 |
| 4 | interleaved | final_clean | A | 75.238 | 10.000 | 0.665 |
| 4 | interleaved | final_clean | B | 77.619 | 6.667 | 0.625 |
| 4 | interleaved | final_clean | C | 75.238 | 20.000 | 0.621 |
| 4 | interleaved | final_noisy | A | 75.376 | 9.667 | 0.666 |
| 4 | interleaved | final_noisy | B | 77.600 | 7.700 | 0.625 |
| 4 | interleaved | final_noisy | C | 75.081 | 18.800 | 0.622 |
| 7 | blocked | final_clean | A | 43.333 | 23.333 | 1.701 |
| 7 | blocked | final_clean | B | 38.095 | 26.667 | 2.008 |
| 7 | blocked | final_clean | C | 38.095 | 30.000 | 2.045 |
| 7 | blocked | final_noisy | A | 43.914 | 23.333 | 1.704 |
| 7 | blocked | final_noisy | B | 38.248 | 27.800 | 2.008 |
| 7 | blocked | final_noisy | C | 38.329 | 30.000 | 2.046 |
| 7 | interleaved | final_clean | A | 76.667 | 23.333 | 0.682 |
| 7 | interleaved | final_clean | B | 81.905 | 26.667 | 0.625 |
| 7 | interleaved | final_clean | C | 78.571 | 26.667 | 0.656 |
| 7 | interleaved | final_noisy | A | 76.862 | 22.700 | 0.684 |
| 7 | interleaved | final_noisy | B | 81.443 | 25.267 | 0.626 |
| 7 | interleaved | final_noisy | C | 78.781 | 25.933 | 0.658 |

### Learning speed

Threshold: at least 90% noisy mean action accuracy for three consecutive fixed probes. Medians condition on reaching; failures to reach are not assigned a fabricated time.

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

### Positions and interference

`position_summary.csv` separates shared/differing scores and counts; `group_metrics_summary.csv` includes each position. At k=7, shared accuracy stays NaN and action count is zero. Forgetting is before-D minus after-D accuracy on the same held-out set. Raw drops depend on baseline accuracy and are not direct evidence of a consolidation mechanism. `D_development.csv.gz` preserves A/B/C after every D update.

## Interpretation and limits

The complete target sequence is supplied in the initial cue. These results describe numerical cue decoding by one small RNN with a fixed budget, ordinal action codes, and unmatched action frequencies/repeats. D similarity is observed rather than controlled. Thirty simulation IDs quantify uncertainty within this model/generator; noisy copies and positions are not independent subjects. Prior exploratory outcomes were seen, so this is a disclosed independent confirmation, not a retroactively untouched preregistration. No human brain mechanism or climbing-performance effect is demonstrated.

## Provenance and data

Pre-final checkpoint: `fa3b7c34496274569944b85c74d2c6c4cd443fe4` (pushed and remotely verified before execution). Producing execution commit: `28a2b07bb40afcf7af9b907612e01ce722902bcc`; it adds approval and reporting metadata, and every numerical module is byte-identical to the pre-final checkpoint. Config SHA-256: `26f8cc56ad115f9e4bb4bb47b30af4997c205cd2906cb8e5195bf24fda8ff006`. Namespace: `cruxrnn_final_proposed_v1`. Frozen numerical fingerprint: `e16274af61696a90c379fb16165ca1ff8309e0f1f02d8d4526da050588d4d168`. Start/end UTC: `2026-10-13T23:51:43.738254+00:00` / `2026-10-13T23:58:05.047976+00:00`. Expected/completed: 120/120 pairs, 240 main models. Failed/excluded/resumed/replaced-successful cells: 0/0/0/0. All artifact hashes, seeds, orders, disjoint bases, pair initializations, metric identities and bootstrap row resampling were independently checked.

Phase wall time: **381.342 seconds (6.36 minutes)**. Summed cell execution: **369.015 seconds**. These measured times cover data generation, pretraining, main training, probes, generalization, D and cell saves; they exclude environment setup, analysis, plotting, reporting, and GitHub delivery. Analysis/figure generation took an additional **9.943 seconds**, for **391.285 seconds** of experiment plus analysis/plotting; independent verification/reporting/archive/delivery time is separate.

Full raw arrays, checkpoints, seeds, per-cell CSVs and completion hashes: `outputs/cruxrnn_final_proposed_v1/runs/run_NNN_kK/` (Git-ignored). Full analysis including bootstrap indices/arrays: `outputs/cruxrnn_final_proposed_v1/analysis/`. Tracked results: `results/confirmatory/`; all 120 individual differences are in `paired_differences.csv`. Per-update CSVs are losslessly compressed as `training.csv.gz`, `development.csv.gz`, and `D_development.csv.gz` (pandas `read_csv` detects gzip). The artifact index, 120 manifests/completion records, published-file hashes and contemporary execution record link those products to code/config/data. The earlier exploratory files remain separately in `outputs/final/` and `reports/history/prior_*`; none were pooled. Historical source/smoke/pilot metadata gaps remain explicit.
