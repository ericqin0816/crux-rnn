# CruxRNN v2: prospective experiment and analysis plan

Status: versioned before new final execution. Date: 2026-11-14 (America/Chicago).
Prior v1 outcomes were seen; this is an independently seeded extension after
that observation, not a claim of untouched historical preregistration.

## Hypotheses and sample sizes

H1: interleaving changes fresh-noisy final A/B/C action accuracy relative to
blocked practice. H2: that paired difference increases as Hamming distance k
increases (less similarity). Both are tested two-sided; contrary effects and
nonlinear categorical patterns are retained. Primary distances: 1,2,4,7.
Target: **100 independent paired repetitions per distance** (800 primary main
models). The prospectively planned extension covers k=1 through 7, also 100
pairs per k: **700 pairs / 1400 primary schedule models** total.

The original 7-unit RNN, 1000 pretraining examples (10 bases ×100), 15,000 main
examples (A/B/C ×5000), SGD .02, momentum 0, batch 20, 750 main updates, noise
SD .1, clean plus 100 noisy evaluation copies per base, 100 unique unseen
bases, probes every 25 updates, and 2000 D examples / 100 updates are retained.
Threshold: noisy mean A/B/C accuracy >=.9 at three consecutive probes.
All 6 blocked orders occur 16 or 17 times per k; rotating the remainder across
k avoids always favoring the same orders. 100 cannot be exactly divisible by 6.

## Exploratory extensions (planned, smaller samples)

For the first **20** replicate IDs at each original distance, using the same
generated families/examples/noise/order streams as the matched main reference:

| Extension | Additional main models | Controls |
| --- | ---: | --- |
| Architecture: GRU/LSTM, each h=7 and h=3 | 640 | Each variant paired internally; same budgets and data. h=3 gives approximate parameter matching to RNN h=7; report actual counts. |
| Capacity: RNN h=16,32,64 | 480 | Same budgets/data; RNN h=7 reference reused. |
| Granularity: 1000-example blocks, 100-example blocks, deterministic trial cycling | 240 | Fully blocked and uniformly shuffled IP references reused; identical example-ID multisets and 750 updates. |
| Forgetting and learning dynamics | 0 | Reuse every main model, with a cloned D-adapted model and all A/B/C probes. |
| Cue noise robustness: SD=0,.1,.3,.6,1 | 0 | New held-out sets, one named stream per SD, identical across paired schedules. Training noise remains .1. |

Total planned new scientific main models: **2760**, plus **2760 D-adapted
clones** and **1260 shared pretrained initializations**. D clones are reported
separately, never counted as independent main repetitions. Smoke/pilot/parity
validation are separately named and never pooled. No new hyperparameters are
selected by comparing final outcomes. Equal hidden size does not imply equal
parameter count; approximate matching is explicitly approximate.

## Primary outcome and statistical units

Primary: equally weighted A/B/C fresh-noisy action accuracy **before D**.
All individual IP-minus-blocked differences are retained in pp. Units are
independent replicate IDs; copies, positions and D updates are repeated
measurements. Reuse across architecture/capacity conditions enables matched
contrasts, not additional independent subjects.

Report mean/median/sample SD, blocked/IP means and SDs, paired Cohen dz,
10,000 paired percentile bootstrap CIs for raw and standardized effects, and
two-sided paired sign-flip tests (10,000 Monte Carlo samples, plus-one p-value
correction). A paired t-test is a sensitivity result. Sign-flip inference
assumes sign exchangeability/symmetry under its null; it is not a human
randomized trial. Undefined dz/degenerate bootstrap denominators are reported.

The practice×similarity interaction is tested on paired differences: linear
slope versus k, and a categorical heterogeneity statistic using row-centered
effects. Bootstrap resamples and interaction sign flips retain whole IDs over
all distances. Inspect categorical effects rather than assuming linearity.

Primary family: four original-distance mean-difference tests plus original-
distance slope and categorical interaction: **Holm correction over six tests**,
alpha .05. Pointwise 95% CIs are not familywise simultaneous intervals.
Secondary all-seven-distance family: seven effects plus slope/categorical
interaction (Holm nine). Architecture/capacity difference-in-differences
contrasts use the matched 20-ID RNN reference, Holm within each extension.
Granularity contrasts against fully blocked practice use Holm across its
four alternatives × four original distances. Remaining secondary metrics
are descriptive unless explicitly labeled with a correction family.

Sensitivity: paired t-tests, 10% trimmed paired means, original-four versus
all-seven slope, and leave-one-ID-out slope range. Report sequence accuracy,
cross-entropy, A/B/C, positions, shared/differing, clean/noisy unseen-base
accuracy, censored threshold attainment and per-sequence D forgetting.
k=7 shared-position accuracy is NaN and its count is zero.

## Execution, failures and integrity

Before final data: run existing tests, output/gradient and full-budget v1
parity checks, tiny smoke, then small full-budget pilots. Pilot namespaces
must differ from the final namespace. Choose CPU worker count from resources,
not outcomes; CPU is the reference, optional CUDA is separately labeled.
Record UTC times, producing Git SHA, numerical code/config/plan hashes,
environment, seeds, generated sequences, schedule IDs, checkpoints and metrics.

No scientific outcome warrants exclusion or replacement. Retain every
mechanical failure; resume only incomplete checkpoints with identical recipe
and data hashes. Completed successful cells cannot be replaced. No exclusions
are planned. If resources prevent completion, report missing cells and actual
sample counts; do not fabricate results or silently lower the target.

## Limits established before observing v2 outcomes

Complete cues already contain targets. Ordinal labels, unmatched action
frequencies/repeats, fixed optimization budgets and observed D distances limit
interpretation. Larger models/gated models can need different optimization;
equal-budget performance is not an architecture's optimal achievable score.
No claim about human neural mechanisms or climbing performance is supported.
v1 results, source-paper attribution and original release remain unchanged.
