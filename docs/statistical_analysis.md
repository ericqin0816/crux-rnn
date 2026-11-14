# Statistical analysis

The versioned [prospective plan](experiment_plan_v2.md) fixes the primary
endpoint and tests before final v2 execution. Prior v1 outcomes were already
known; this timing is disclosed.

## Paired outcomes and uncertainty

For independent ID i and distance k, compute
`d_ik = 100 × (accuracy_IP − accuracy_blocked)` in percentage points (pp).
The primary endpoint is equally weighted A/B/C fresh-noisy action accuracy,
before D. A noisy copy, action position or checkpoint is not another subject.

Report each schedule's mean/sample SD, every paired difference, paired
mean/median/sample SD, and Cohen's paired `dz = mean(d)/SD(d)`. Ten thousand
ID resamples produce percentile 95% CIs for raw and standardized effects.
Undefined dz and bootstrap draws with zero SD are reported explicitly.
These are pointwise intervals, not simultaneous familywise intervals.

Two-sided sign-flip tests use 10,000 Monte Carlo draws and a plus-one p-value
correction. They assume sign exchangeability/symmetry under the null. They
are not assumption-free human randomization tests. The paired t-test is a
sensitivity check; neither method treats thousands of cue copies as subjects.

## Schedule × similarity interaction

Regress paired differences on k. Each ID supplies a slope across distances;
the average equals the balanced-data least-squares slope. Bootstrap whole
rows of the n×k difference matrix, preserving all distances within sampled
IDs. Also test categorical heterogeneity: subtract each ID's across-k mean,
then sign-flip whole residual rows. This removes the overall schedule effect
before asking whether it varies across similarity conditions. A straight line
can miss nonlinearity, so categorical effects and raw distributions accompany it.

## Correction families

The primary Holm family has six tests: effects at k=1,2,4,7, their linear
interaction slope, and their categorical interaction. Alpha is .05. The
prospectively secondary all-seven-distance family has nine tests (seven
effects plus both interactions). Architecture and capacity comparisons test
differences of paired schedule effects against their matched 20-ID reference,
with Holm correction within each extension. Granularity compares four
alternatives against blocked practice over four original distances: 16 tests.

Other outcomes are descriptive unless a correction family is explicitly
attached. A CI excluding zero is not by itself a multiplicity-adjusted claim.
P-values address compatibility with the stated null; effect sizes and
uncertainty address magnitude. Statistical significance does not imply a
useful climbing or biological effect.

## Sensitivity and transparency

Report paired t-test p-values, 10% trimmed paired means, original-four versus
all-seven slopes and leave-one-ID-out slope ranges. Trimming is a separately
labeled sensitivity summary, never deletion from the primary data. Learning
curves display mean ± SE, not bootstrap CIs. Threshold medians condition on
actually reaching the threshold; report attainment counts and leave
`not_reached` censored rather than assigning an invented convergence time.

Per-sequence D drops depend on baseline performance. Raw loss of accuracy
cannot establish better or worse consolidation when one model starts near a
floor. Clean/noisy, action/whole-sequence and generalization outcomes can
disagree; all are retained. Implementation: [statistics.py](../src/cruxrnn/statistics.py)
and [analysis.py](../src/cruxrnn/analysis.py).
