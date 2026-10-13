# Verified simulation results

All claims below refer to newly generated outputs except the explicitly labeled author saved-array comparison. Final simulation pairs: **30 at each k=1,2,4,7**, or 120 run-by-distance pairs / 240 main-trained schedule models. There are zero missing final cells, zero failure records, and zero exclusions. Each controlled cell also completed the paired 100-update D experiment. `complete.json` hashes, CSV counts, checkpoint update counts, manifests, and bootstrap indices were verified.

## Primary held-out fresh-noisy action accuracy

| k | Paired runs | Blocked noisy (%) | IP noisy (%) | IP−blocked mean (pp) | Median (pp) | 95% CI (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 30 | 88.45 | 90.56 | 2.11 | 0.00 | [0.10, 4.26] |
| 2 | 30 | 78.86 | 82.98 | 4.13 | 3.57 | [1.27, 6.91] |
| 4 | 30 | 60.88 | 75.91 | 15.04 | 11.69 | [10.89, 19.17] |
| 7 | 30 | 41.00 | 77.72 | 36.73 | 36.64 | [32.53, 40.77] |

Confidence intervals are percentile intervals from 10,000 resamples of simulation IDs. The observed mean schedule differences are positive at all four distances. The categorical values must be inspected alongside the linear trend.

The prespecified OLS trend is **5.933 percentage points per Hamming unit**, with 95% bootstrap CI **[5.199, 6.655]**. Full replicate IDs, including all four distances, are resampled together. This is a descriptive model-dependent trend, not a biological dose-response law. No p-values or significance tests were added.

## Secondary measures

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

Clean and noisy conditions remain separate. Unseen-base performance is poor in absolute terms under both schedules (about 25–29% action accuracy; the analytical uniform-guess expectation for four classes is 25%). Higher relative performance does not imply reliable new-sequence competence. Despite better primary action accuracy, IP has **lower mean complete-sequence accuracy at k=2,4,7**; the secondary metrics do not uniformly favor IP. Blocked practice can retain one entire recent sequence while IP gets more individual actions correct across the family. At k=1 the median primary difference is zero despite a positive mean. D drops use the same final held-out old-sequence set before and after training. Mean noisy D drops are larger under IP at k=1,4,7 and slightly smaller at k=2; they should be interpreted in the context of higher initial accuracy and the blocked floor. Raw A/B/C and position 1–7 metrics are available in `outputs/final/analysis/metrics.csv`. At k=7 every shared-position entry is NaN with count zero. Learning and D curves use fixed development probes; final data were not used to choose a threshold.

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

All threshold start/confirmation updates, including `not_reached`, are saved in the run-level CSV. Results from only successful threshold reaches are not substituted for missing thresholds.

## Source-compatible reproduction and phases

| Phase | Completed paired cells | Missing | Failure records | Measured cell runtime (s) |
| --- | --- | --- | --- | --- |
| source_exact | 100 | 0 | 0 | 235.75 |
| smoke | 4 | 0 | 0 | 0.52 |
| pilot | 20 | 0 | 0 | 60.51 |
| final | 120 | 0 | 0 | 358.03 |

Five source runs were inspected first; their per-run shapes and main directional comparisons agreed with the saved arrays. The measured runtime supported extending to 100 new main-training runs. The five are included in those 100 and were verified/skipped on resume, not retrained. Main aggregate comparison:

| Metric | New n=100 loss | Author n=100 loss | New shape | Author shape |
| --- | --- | --- | --- | --- |
| blocked_training | 0.7496 | 0.7576 | (100, 750) | (100, 750) |
| interleaved_training | 0.9858 | 0.9752 | (100, 750) | (100, 750) |
| blocked_retention | 1.5319 | 1.4787 | (100, 15000) | (100, 15000) |
| interleaved_retention | 0.6277 | 0.6342 | (100, 15000) | (100, 15000) |
| blocked_generalization | 2.5145 | 2.4751 | (100, 10000) | (100, 10000) |
| interleaved_generalization | 1.9733 | 1.9858 | (100, 10000) | (100, 10000) |
| pre_generalization | 1.4097 | 1.4110 | (100, 10000) | (100, 10000) |

New blocked training loss is lower, with abrupt increases at the two block switches; final training-input retention and source generalization loss favor IP. These directions match the saved arrays. Aggregate values differ, as expected with new explicit seed streams and a Windows CPU environment. No exact historical seed replication is claimed. Source uniqueness/novelty were deliberately not enforced: 28 of 100 new runs had duplicate generalization bases, and 6 had at least one exact generalization/pretraining-or-main overlap. These runs were retained; see `sequence_overlap_summary.csv`. All controlled generalization bases are unique and disjoint. The optional source vulnerability protocols were not newly run. The authors' saved last-sequence noise, pruning, and interference arrays were audited and show larger mean loss increases for blocked practice; those are reference data, not new controlled outcomes. The Figure 2 10,000-pretraining sensitivity config is provided but unexecuted.

Smoke used reduced budgets and has no scientific interpretation. Pilot used five disjoint IDs at each k with real budgets. No scientific settings changed following pilot. Neither phase contributes to final intervals.

## Limits

The complete sequence is provided as the initial cue, so generalization is decoding unseen numeric patterns rather than acquiring an unobserved physical skill. Seven hidden units, default initialization, one optimizer, and one fixed budget limit generality. Action frequencies, adjacent repeats, and longest runs are recorded but not matched across distances; they can accompany this generator's manipulation. Numeric cue values impose an ordinal geometry on action labels. Low-distance performance can approach a ceiling, and whole-sequence accuracy can be at a floor. Bootstrap uncertainty is across artificial simulation IDs, not people. D distances were observed, not manipulated, and raw forgetting depends on baseline accuracy. There is no offline time-dependent consolidation or direct biological evidence. These findings do not establish how human brains work and are not a broad novelty claim.
