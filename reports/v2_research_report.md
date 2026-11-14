# CruxRNN v2 research report

**Computational replication and extension; review candidate, not a published paper or human-subject study.**

## Abstract

We examined whether sequence similarity moderates the effect of practice
schedule in complete-cue recurrent decoding. The main study completed 100
pairs at every nonzero seven-position Hamming distance. Architecture,
capacity and granularity extensions reused matched 20-ID reference subsets.
All 2760 planned scientific main models and the
same number of D-adapted clones completed in 1637.45 seconds wall time.
All seven pointwise intervals exclude zero. Original-four slope: 5.600 pp per Hamming unit, 95% CI [5.131, 6.058], Holm p=0.0005999. IP has lower mean exact-sequence accuracy at k=3,4,5,6,7, despite any action-accuracy advantage.
The result concerns a fixed artificial protocol and cannot establish human
neural mechanisms or climbing-performance effects.

## Background and question

Song et al. (2026) studied blocked and interleaved practice in a minimal Elman
network. CruxRNN independently reproduced the audited computation, then
controlled pairwise sequence distance. Climbing supplies motivation, not a
validation dataset. v2 asks whether the schedule-by-similarity relationship
persists with more seeds and under architecture, capacity, granularity,
interference and cue-noise changes.

## Hypotheses and prospective timing

H1 tests a paired schedule effect at original k=1,2,4,7. H2 tests moderation
by distance, with linear and categorical interaction analyses. The versioned
[plan](../docs/experiment_plan_v2.md) was committed at `9c64a49` before final
data generation. Prior v1 outcomes were seen; this is not a claim that earlier
outcome exposure can be undone. Smoke, pilot, parity and custom/demo runs are
separate and excluded.

## Methods

The complete seven-action cue is shifted from 0–3 targets to 1–4 inputs with
Gaussian SD .1 noise. Exactly k positions contain three distinct A/B/C actions;
all other positions agree. Pretraining, main, unseen and D bases are unique
and disjoint. One shared pretrained network per variant is cloned into
schedules using identical examples, targets and initial state hashes.
SGD .02, zero momentum, batch 20, 1000 pretraining examples, 15,000 main
examples/750 updates and 2000 D examples/100 updates are fixed.

All six blocked orders occur 16 or 17 times per distance. Additional schedules
change granularity while preserving exposure/update counts. GRU/LSTM h=7 and
h=3 and RNN h=16/32/64 use the same budgets/families as the matched references.
Approximate matching compares RNN7=179 parameters to GRU3=159 and LSTM3=195;
same-hidden GRU7/LSTM7 have 403/515. D and robustness reuse trained models.
[Full methods](../docs/methodology.md).

## Main results

All seven conditions contain the planned 100 complete pairs. Accuracy is the
fresh-noisy final action score before D; differences/intervals are pp.
Raw/standardized bootstrap intervals are pointwise. Full precision, SDs and
sensitivity columns are retained in `results/v2/main_summary.csv`.

| k | Pairs | Blocked % | IP % | IP−blocked pp | Median pp | SD pp | CI low | CI high | paired_dz | Primary Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 100 | 87.448 | 89.304 | 1.857 | 0.000 | 6.007 | 0.651 | 2.976 | 0.309 | 0.0017 |
| 2 | 100 | 79.203 | 84.897 | 5.693 | 4.762 | 8.469 | 4.050 | 7.377 | 0.672 | 0.0005999 |
| 3 | 100 | 69.782 | 80.655 | 10.873 | 9.524 | 9.028 | 9.188 | 12.675 | 1.204 | undefined |
| 4 | 100 | 60.508 | 77.634 | 17.126 | 17.000 | 11.011 | 15.023 | 19.283 | 1.555 | 0.0005999 |
| 5 | 100 | 53.132 | 74.854 | 21.722 | 23.238 | 11.949 | 19.358 | 24.022 | 1.818 | undefined |
| 6 | 100 | 46.283 | 77.748 | 31.465 | 32.214 | 11.753 | 29.151 | 33.778 | 2.677 | undefined |
| 7 | 100 | 40.249 | 75.170 | 34.921 | 35.048 | 13.121 | 32.358 | 37.524 | 2.662 | 0.0005999 |

All seven pointwise intervals exclude zero. Primary distance tests passing Holm correction: 1, 2, 4, 7. Do not equate unadjusted pointwise intervals with Holm
familywise decisions or confuse a small supported effect with practical value.

## Interaction and sensitivity

Original-four slope: 5.600 pp per Hamming unit, 95% CI [5.131, 6.058], Holm p=0.0005999.
Original-four categorical interaction: adjusted p=0.0005999.
All-seven slope: 5.771 pp/unit,
95% CI [5.386, 6.162], secondary-family
Holm p=0.0008999. Original-four leave-one-ID-out slopes
range [5.542, 5.657].
Paired t-tests and 10% trimmed means are retained as sensitivities, not
replacements for the prespecified endpoint. The categorical plot remains
necessary because a slope need not describe every intermediate distance.

## Exploratory architecture and capacity results

The following raw variant effects use the same first 20 family IDs at each
original distance, including the reused RNN7 reference. Intervals are
descriptive pointwise intervals; the next table tests corrected matched
differences in schedule effect against RNN7.

| architecture | hidden_size | distance | n | difference_pp | ci_low_pp | ci_high_pp | paired_dz |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gru | 3 | 1 | 20 | 0.788 | -1.576 | 3.050 | 0.146 |
| gru | 3 | 2 | 20 | 2.771 | -1.700 | 7.179 | 0.268 |
| gru | 3 | 4 | 20 | 5.598 | 1.495 | 9.791 | 0.579 |
| gru | 3 | 7 | 20 | 9.217 | 6.395 | 12.052 | 1.406 |
| gru | 7 | 1 | 20 | 1.002 | -3.491 | 5.450 | 0.096 |
| gru | 7 | 2 | 20 | 1.224 | -1.760 | 4.757 | 0.159 |
| gru | 7 | 4 | 20 | 6.048 | 3.005 | 9.243 | 0.814 |
| gru | 7 | 7 | 20 | 20.010 | 13.611 | 26.912 | 1.295 |
| lstm | 3 | 1 | 20 | 2.260 | 0.071 | 5.283 | 0.371 |
| lstm | 3 | 2 | 20 | 3.088 | -0.241 | 6.417 | 0.398 |
| lstm | 3 | 4 | 20 | 4.219 | 1.043 | 7.319 | 0.569 |
| lstm | 3 | 7 | 20 | 4.433 | 1.850 | 6.860 | 0.758 |
| lstm | 7 | 1 | 20 | 0.462 | -1.905 | 2.857 | 0.080 |
| lstm | 7 | 2 | 20 | 1.971 | -1.962 | 5.781 | 0.219 |
| lstm | 7 | 4 | 20 | 8.602 | 3.709 | 13.624 | 0.732 |
| lstm | 7 | 7 | 20 | 12.310 | 8.590 | 16.345 | 1.348 |
| rnn | 7 | 1 | 20 | 1.198 | -0.231 | 2.950 | 0.318 |
| rnn | 7 | 2 | 20 | 5.786 | 1.205 | 11.239 | 0.490 |
| rnn | 7 | 4 | 20 | 14.571 | 11.302 | 18.198 | 1.802 |
| rnn | 7 | 7 | 20 | 37.248 | 31.700 | 42.724 | 2.936 |
| rnn | 16 | 1 | 20 | 3.050 | 1.707 | 4.452 | 0.939 |
| rnn | 16 | 2 | 20 | 12.314 | 9.509 | 14.910 | 1.939 |
| rnn | 16 | 4 | 20 | 29.948 | 25.721 | 33.614 | 3.175 |
| rnn | 16 | 7 | 20 | 54.129 | 50.950 | 57.188 | 7.437 |
| rnn | 32 | 1 | 20 | 5.414 | 3.733 | 7.024 | 1.391 |
| rnn | 32 | 2 | 20 | 14.005 | 11.240 | 16.510 | 2.286 |
| rnn | 32 | 4 | 20 | 35.419 | 33.302 | 37.252 | 7.589 |
| rnn | 32 | 7 | 20 | 62.448 | 58.840 | 65.445 | 8.110 |
| rnn | 64 | 1 | 20 | 6.048 | 4.186 | 7.760 | 1.455 |
| rnn | 64 | 2 | 20 | 16.086 | 13.747 | 18.076 | 3.206 |
| rnn | 64 | 4 | 20 | 37.843 | 37.355 | 38.095 | 35.607 |
| rnn | 64 | 7 | 20 | 64.110 | 62.283 | 65.652 | 16.106 |

Difference-in-differences (variant effect minus reference effect), with Holm
correction within architecture or capacity families:

| family | architecture | hidden_size | distance | difference_pp | ci_low_pp | ci_high_pp | p_holm_family |
| --- | --- | --- | --- | --- | --- | --- | --- |
| architecture | gru | 3 | 1 | -0.410 | -2.700 | 1.893 | 1 |
| architecture | gru | 3 | 2 | -3.014 | -9.169 | 2.400 | 1 |
| architecture | gru | 3 | 4 | -8.974 | -12.952 | -4.388 | 0.021 |
| architecture | gru | 3 | 7 | -28.031 | -34.376 | -21.576 | 0.0016 |
| architecture | gru | 7 | 1 | -0.195 | -4.769 | 4.500 | 1 |
| architecture | gru | 7 | 2 | -4.562 | -9.498 | -0.162 | 0.6151 |
| architecture | gru | 7 | 4 | -8.524 | -13.048 | -4.505 | 0.005999 |
| architecture | gru | 7 | 7 | -17.238 | -25.353 | -9.033 | 0.0121 |
| architecture | lstm | 3 | 1 | 1.062 | -2.026 | 4.652 | 1 |
| architecture | lstm | 3 | 2 | -2.698 | -9.793 | 4.050 | 1 |
| architecture | lstm | 3 | 4 | -10.352 | -14.960 | -5.886 | 0.0026 |
| architecture | lstm | 3 | 7 | -32.814 | -38.717 | -26.993 | 0.0016 |
| architecture | lstm | 7 | 1 | -0.736 | -3.907 | 2.398 | 1 |
| architecture | lstm | 7 | 2 | -3.814 | -10.524 | 3.176 | 1 |
| architecture | lstm | 7 | 4 | -5.969 | -11.162 | -1.090 | 0.3087 |
| architecture | lstm | 7 | 7 | -24.938 | -31.286 | -18.324 | 0.0016 |
| capacity | rnn | 16 | 1 | 1.852 | -0.438 | 3.798 | 0.1129 |
| capacity | rnn | 16 | 2 | 6.529 | 1.536 | 10.974 | 0.0344 |
| capacity | rnn | 16 | 4 | 15.376 | 10.493 | 20.302 | 0.0012 |
| capacity | rnn | 16 | 7 | 16.881 | 10.783 | 22.776 | 0.0012 |
| capacity | rnn | 32 | 1 | 4.217 | 2.086 | 6.074 | 0.005 |
| capacity | rnn | 32 | 2 | 8.219 | 2.707 | 12.867 | 0.0177 |
| capacity | rnn | 32 | 4 | 20.848 | 16.909 | 24.374 | 0.0012 |
| capacity | rnn | 32 | 7 | 25.200 | 18.926 | 30.993 | 0.0012 |
| capacity | rnn | 64 | 1 | 4.850 | 3.014 | 6.643 | 0.0012 |
| capacity | rnn | 64 | 2 | 10.300 | 4.879 | 15.245 | 0.005999 |
| capacity | rnn | 64 | 4 | 23.271 | 19.557 | 26.621 | 0.0012 |
| capacity | rnn | 64 | 7 | 26.862 | 20.902 | 32.829 | 0.0012 |

Negative mean raw variant effects, retained without exclusion:

No variant has a negative mean raw paired difference; interval uncertainty still applies.

In the matched 20-ID k=7 subset, RNN capacity effects were h=7: 37.25 pp; h=16: 54.13 pp; h=32: 62.45 pp; h=64: 64.11 pp. Larger capacity did not remove the schedule gap under this fixed protocol. Gated-model effects were GRU h=3: 9.22 pp; GRU h=7: 20.01 pp; LSTM h=3: 4.43 pp; LSTM h=7: 12.31 pp; several low-distance intervals include zero. These are exploratory protocol comparisons, not optimized architecture rankings.
These equal-budget comparisons do not estimate optimized architecture
performance. Parameter counts, pretraining behavior and gate dynamics differ.

## Exploratory granularity

Each alternative is compared with fully blocked practice on the same 20 IDs
and examples. Four alternatives × four distances form its Holm family.

| distance | schedule | n | difference_pp | ci_low_pp | ci_high_pp | p_holm_family |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | large_blocks | 20 | -0.098 | -2.207 | 2.045 | 1 |
| 1 | small_blocks | 20 | 1.136 | -0.010 | 2.802 | 1 |
| 1 | cyclic | 20 | 1.583 | 0.238 | 3.405 | 0.6464 |
| 1 | interleaved | 20 | 1.198 | -0.231 | 2.948 | 1 |
| 2 | large_blocks | 20 | 0.755 | -1.833 | 3.771 | 1 |
| 2 | small_blocks | 20 | 4.800 | 0.338 | 10.083 | 0.3342 |
| 2 | cyclic | 20 | 6.312 | 1.412 | 11.962 | 0.164 |
| 2 | interleaved | 20 | 5.786 | 1.281 | 11.362 | 0.164 |
| 4 | large_blocks | 20 | 5.571 | 1.400 | 9.350 | 0.1287 |
| 4 | small_blocks | 20 | 12.643 | 9.495 | 15.995 | 0.0016 |
| 4 | cyclic | 20 | 14.483 | 11.036 | 18.036 | 0.0016 |
| 4 | interleaved | 20 | 14.571 | 11.221 | 18.155 | 0.0016 |
| 7 | large_blocks | 20 | 18.431 | 13.024 | 23.772 | 0.0016 |
| 7 | small_blocks | 20 | 35.757 | 30.800 | 40.783 | 0.0016 |
| 7 | cyclic | 20 | 36.924 | 31.700 | 42.236 | 0.0016 |
| 7 | interleaved | 20 | 37.248 | 31.862 | 42.702 | 0.0016 |

## Secondary outcomes and contradictions

Accuracy/generalization columns below are percent, forgetting is pp, and
cross-entropy is unscaled loss. Clean/noisy, action/exact-sequence outcomes
answer different questions.

| distance | schedule | clean_accuracy | sequence_accuracy | cross_entropy | generalization_clean | generalization_noisy | forgetting_noisy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | blocked | 87.429 | 27.730 | 0.426 | 25.476 | 25.480 | 43.540 |
| 1 | interleaved | 89.333 | 36.147 | 0.309 | 25.794 | 25.777 | 44.738 |
| 2 | blocked | 79.190 | 28.667 | 0.705 | 25.451 | 25.440 | 37.609 |
| 2 | interleaved | 84.952 | 29.010 | 0.419 | 26.386 | 26.378 | 41.448 |
| 3 | blocked | 69.857 | 27.523 | 0.981 | 25.353 | 25.390 | 33.384 |
| 3 | interleaved | 80.762 | 19.600 | 0.521 | 26.367 | 26.362 | 40.323 |
| 4 | blocked | 60.429 | 21.897 | 1.205 | 25.619 | 25.638 | 23.062 |
| 4 | interleaved | 77.762 | 16.420 | 0.590 | 27.510 | 27.477 | 37.533 |
| 5 | blocked | 53.143 | 26.667 | 1.497 | 25.524 | 25.516 | 16.627 |
| 5 | interleaved | 74.952 | 15.843 | 0.678 | 27.461 | 27.438 | 33.302 |
| 6 | blocked | 46.238 | 23.983 | 1.694 | 26.027 | 26.026 | 14.991 |
| 6 | interleaved | 77.857 | 20.493 | 0.672 | 28.929 | 28.923 | 38.187 |
| 7 | blocked | 40.000 | 23.993 | 1.882 | 25.871 | 25.844 | 9.392 |
| 7 | interleaved | 75.143 | 14.620 | 0.713 | 28.797 | 28.777 | 36.030 |

IP has lower mean exact-sequence accuracy at k=3,4,5,6,7, despite any action-accuracy advantage. Learning-speed attainment counts and reached-only medians are
in `thresholds.csv`; nonattainment is censored. A/B/C and position groups,
shared/differing scores, before/after-D metrics and all noise levels remain
in the raw/group tables. Shared k=7 accuracy is NaN with count zero. Per-seed
per-sequence forgetting distributions are in `forgetting_distributions.csv`.
Raw D drops must be interpreted with baseline/floor effects and observed D
distances; they do not identify biological consolidation.

At held-out SD=1 the raw mean differences were k=1: 0.89 pp; k=2: 2.89 pp; k=3: 6.53 pp; k=4: 11.06 pp; k=5: 14.72 pp; k=6: 21.80 pp; k=7: 23.95 pp. These robustness means are descriptive; no additional corrected significance claim is made.

## Statistical analysis

Independent IDs are the units. Ten thousand paired percentile bootstrap
resamples give raw and paired-dz intervals. Sign-flip tests use the stated
symmetry assumption and plus-one Monte Carlo correction. Primary Holm family:
four original-distance effects plus linear/categorical interactions (six).
The all-seven secondary family has nine. Interaction resamples/sign flips
retain complete ID rows. Uncertainty concerns this model/generator.
[Detailed assumptions](../docs/statistical_analysis.md).

## Reproducibility, failures and engineering validation

Completed cells: 1260/1260.
Main models: 2760; D clones: 2760.
Recorded final execution failures: 0; exclusions:
0. Producing Git commit(s):
`47b32340142cce8cc1dd46056d0283d43c356943`.
Namespace: `cruxrnn_v2_main_20261114`. Config SHA-256:
`cb66c371a2b377273490a3421cd8bff2bf267d3c7e6a590c6d2750b9df621693`. Phase start/end UTC:
`2026-11-14T20:38:41.404937+00:00` /
`2026-11-14T21:05:59.353070+00:00`.
Wall time 1637.445 s is not the sum of parallel cell times and excludes
analysis/figures/reporting. Full environment/freeze and execution records are
saved alongside exact seeds, sequence sets, schedule IDs, metrics, checkpoints
and SHA-256 indexes.

All 38 recorded-environment tests passed. Eight full-budget v1 models reproduced
every final weight and all 750 losses bitwise. Smoke: 64 cells / 152 main
models; full-budget pilot: 32 / 76; quick-start demo: 16 / 38. A fresh modern
clone passed 21 v2 tests and its separate 38-model demo. These diagnostics
are not pooled. GitHub Actions checks the recorded and modern environments.

Prototype integration exposed a Windows index-dtype issue and old pip lacked
editable-build support. Both were fixed before final data, without changing
the experimental design. CUDA remains untested on this CPU-only setup.

## Discussion and limits

Read effect magnitude, intervals, corrected tests and secondary outcomes
together. A difference within this controlled network does not establish a
general advantage of one curriculum. Full target cues, ordinal coding,
unmatched action frequencies/repeats, fixed optimization, unequal capacities,
observed D similarity, smaller exploratory samples and prior outcome exposure
limit conclusions. [Full limitations](../docs/limitations.md).

The primary mean gap grows from 1.86 pp at k=1
to 34.92 pp at k=7. Both schedules have lower
absolute action accuracy at k=7 than at k=1; blocked accuracy declines more.
The k=1 paired median is 0.00 pp, so a positive mean
does not imply an advantage for every seed. In this dataset IP's mean raw D
forgetting is greater at every distance, despite its higher pre-D accuracy;
baseline/floor effects and final retained scores constrain that interpretation.

## Future directions

Independently replicate with new namespaces; match frequencies/repeats; test
categorical relabeling, one-hot/partial cues and justified architecture-specific
optimization; manipulate D similarity; and compare predictions with suitable
behavioral measurements. These require new plans rather than post hoc claims.

## Data, figures and references

Compact outputs: `results/v2/`; full local arrays/checkpoints/bootstrap arrays:
`outputs/v2/main/`. Twelve original PNG/SVG figure types are under
`results/v2/figures/`; learning bands are ±SE, paired intervals are bootstrap
95% CIs. [Figure captions](../docs/figures.md) identify samples, endpoints and
uncertainty. The offline viewer reads only exported actual tables and recorded
families; schedule illustrations are separately labeled. Full raw-data archive
integrity and extraction instructions are in the
[reproduction guide](../docs/reproducibility.md) and
[archive receipt](../artifacts/v2_archive_receipt.json). The archive remains
local pending owner review; no final v2 release has been published.

Song, Y., Kim, H., & Kim, T. (2026). A minimal recurrent neural network models
the robustness of interleaved practice on motor sequence learning. Scientific
Reports, 16, 10068. https://doi.org/10.1038/s41598-026-40162-w .
Pinned source: `31f58bb6de90d1fffcb752c0b97afb7df51479a7`.
Original source code/figures are excluded; CruxRNN visuals are original.

AI-assisted development and documentation are disclosed. Owner review is
required before merge/final v2 release; this report is not publication approval.
