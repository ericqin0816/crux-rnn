# Researcher guide for the project owner

CruxRNN asks a narrow question: when three seven-action patterns are more or
less similar, how does changing their practice order affect a recurrent
network? Speed climbing motivated the question, but the data are artificial
action labels, not measurements of climbers or brains. You should be able to
explain that boundary before explaining any result.

Read this alongside [methodology](methodology.md), [the statistics guide](statistical_analysis.md)
and the [v2 report](../reports/v2_research_report.md). The saved-data
[v1 notebook](../notebooks/walkthrough.ipynb) is still useful for the basics;
its numerical results remain explicitly v1.

## 1. Where the code lives

| File and functions | What to understand |
| --- | --- |
| `src/cruxrnn/config.py`: `ExperimentConfig`, `load_config` | Budgets, noise, device, replicate counts, threshold and analysis settings are operational values, validated before execution. Unknown fields and unsafe historical output paths fail early. |
| `sequences.py`: `family`, `generate` | Builds A/B/C with exact equal pairwise distances, unique disjoint base sets and separately seeded noisy cues. It reuses audited, unchanged helper functions from our own frozen v1 package. |
| `models.py`: `SequenceModel`, `parameter_count` | Complete cue → recurrent state → seven four-logit outputs. RNN, GRU and LSTM share the cue/readout/feedback task, with configurable hidden size. |
| `schedules.py`: `make_order`, `blocked_order` | Produces different permutations of exactly the same example IDs. The order assignment is deterministic and near-balanced at n=100. |
| `training.py`: `optimizer_for`, `step`, `train` | SGD, one pass, development probes, atomic checkpoints, metadata-guarded resume and finite-loss checks. Actual configured learning rate/momentum/batch size are used. |
| `evaluation.py`: `evaluate` | No optimizer steps; clean/noisy scores, explicit groups, state-hash protection and correct NaN for empty position groups. |
| `reproducibility.py`: `execution_identity`, `numeric_fingerprint`, `set_seed` | Producing Git identity, code hashes, named streams and backend settings. Numerical/config/plan bytes must match Git for research runs; unrelated documentation edits are recorded honestly. |
| `engine.py`: `make_jobs`, `execute_cell`, `run_study`, `verify_complete` | Expands the planned design into independent cells, pretrains once per variant, clones paired schedules, runs D, retains failures and verifies completed artifacts. CPU workers share no mutable model state. |
| `statistics.py`: `paired_summary`, `similarity_interaction`, `sign_flip`, `holm` | Independent-ID inference, raw/standardized effects, interaction and correction families. Read its assumptions, not just its output columns. |
| `analysis.py`: `verify`, `analyze` | Checks saved data/orders/checkpoints, gathers seed-level outputs and streams curve summaries without loading millions of rows at once. v1 is never pooled. |
| `visualization.py`: `diagrams`, `figures` | Original design/model diagrams and figures generated directly from verified saved outputs. PNG and SVG exports have consistent labels/colors. |
| `cli.py`: `main` | Installed `cruxrnn` commands for demo, planned studies, custom cells, original v1 verification, analysis and figures. |
| `src/motor_rnn/` | Frozen v1 numerical package. Do not reformat it casually: its exact bytes are part of the original config freeze. |

Tests are in `tests/test_v2_core.py` and `test_v2_statistics.py`; historical
controls remain in the other two test files. `docs/demo/` is an offline viewer,
not a training service. Its `data.js` is exported from actual analysis tables.

## 2. The model, step by step

An example has a cue shaped `(1,7)`, such as `[1,2,3,4,1,2,3]`. The targets
are the corresponding zero-based classes `[0,1,2,3,0,1,2]`. A training batch
contains 20 examples, so Torch receives `(20,1,7)`. The entire pattern enters
at the first recurrent step. There is no hidden human movement sequence.

For an Elman RNN, the hidden state follows

`h_t = tanh(W_in z_t + b_in + W_rec h_(t−1) + b_rec)`.

The readout is `logits_t = W_out h_t + b_out`. Logits are raw scores, not
probabilities. A feedback projection makes the next recurrent input:
`z_(t+1) = W_feedback logits_t + b_feedback`. We repeat until seven outputs
form `(batch,7,4)`. Argmax gives a predicted class; softmax is used implicitly
inside cross-entropy. Hidden state resets for every example. The eighth,
unused recurrent transition is preserved to match the original task exactly.

Cross-entropy averages the negative log probability of the correct class over
all examples and seven positions. SGD changes parameters by
`theta <- theta − learning_rate × gradient(loss)`. The 15,000 main examples
give 750 batch updates during **one pass**, not 750 passes through the data.

GRU and LSTM add learned gates to the recurrent state update. LSTM also has a
cell state. They do not introduce human memory biology. At h=7 the RNN/GRU/LSTM
have 179/403/515 parameters, including projections. GRU h=3 has 159 and LSTM
h=3 has 195, so those comparisons approximately match the 179-parameter RNN;
they do not claim exact equivalence in capacity.

## 3. Why pairing matters

For one family and model variant, pretrain once and clone the weights. All
schedules receive the same noisy training arrays and targets; only the order
of their IDs changes. If you generated new noise for each schedule, an
accuracy difference could reflect different examples rather than practice.

Each of three sequences has 5000 exposures. Fully blocked practice has three
long blocks; 1000- and 100-example blocks switch more often; deterministic
cycling rotates bases trial by trial; shuffled IP randomly mixes the exact
IDs without replacement. Hidden state resets between examples, so the
important change is batch composition and the sequence of optimizer updates.

At k changed positions, A/B/C have three different actions; elsewhere all
agree. Thus larger k means **less** sequence similarity. The familiar original
conditions are 1,2,4,7. v2 extends them to every nonzero distance. Frequencies
and repeated-action patterns are not matched, an important limitation.

## 4. What each study does

The main v2 study plans 100 paired IDs at each of seven distances: 1400 main
models, including 800 at the original four distances. The first 20 IDs at the
original distances supply matched reference data for exploratory extensions.
GRU/LSTM h=7 and h=3 add 640 models; RNN h=16/32/64 adds 480; three additional
practice schedules add 240. That is 2760 scientific main models, with D-adapted
clones counted separately. Shared references are reused, not retrained and
pretended to be extra independent evidence.

Every main model is retained intact, then a clone learns the same new D cues
as its paired schedule. Before/after A/B/C drops measure interference under
this fixed protocol. D's similarity is recorded rather than experimentally
varied. Noise robustness changes only held-out cue noise, not training noise.
Learning dynamics reuse fixed development probes throughout training.

Tiny demos and smoke runs check software. Full-budget pilots check runtime,
resources and learning behavior. Their namespaces differ from final seeds;
their estimates are not final scientific findings. The exact plan was committed
before final v2 data generation. The historical v1 release remains accessible.

## 5. Reading the actual results

Start with `results/v2/main_summary.csv` and `paired_differences.csv`, then
read the generated [v2 report](../reports/v2_research_report.md). The figure
`main_accuracy.png` shows schedule means; `paired_effects.png` shows every
seed pair and uncertainty. The report states actual completion counts and
failures; do not infer them from the planned table above.

An action score can be high because many positions are shared, while a model
still gets the whole sequence wrong. Always compare complete-sequence scores.
Unseen-base performance asks whether an already supplied complete cue can be
decoded, not whether the network learned a new physical skill. D forgetting
depends on baseline: a low-scoring model has less accuracy available to lose.

Use the confidence interval to discuss the range of effects compatible with
run-to-run variation under this model. An interval containing zero leaves the
direction uncertain. A positive interval alone is not a multiplicity-adjusted
claim. Read the Holm-adjusted primary-family p-value and the actual pp effect.
Even a statistically supported 1 pp difference may be practically small.

For reference, v1 at k=1 estimated +2.11 pp with CI [−1.09,5.44]; its direction
was uncertain. The v1 interaction slope was positive, but whole-sequence
accuracy and D forgetting gave important counterexamples to a blanket “IP is
better” claim. v2 must be interpreted from its own new records, not copied
from those historical numbers.

## 6. Why these statistics

Each independently seeded model/family realization is a subject. The two
practice schedules within it are paired; hundreds of noisy copies are repeated
measurements of the same trained network. Bootstrap IDs, not copies or action
positions. For the interaction, bootstrap whole rows across distances.

Paired dz scales the mean difference by its between-ID SD. Raw pp remains
easier to interpret. Sign-flip tests assume null sign exchangeability; the
t-test is a sensitivity check. Holm correction limits familywise false
positives across the explicitly planned test families. Categorical effects,
trimmed means and leave-one-ID-out slopes help reveal outlier/nonlinearity
dependence. They do not justify deleting an inconvenient successful run.

## 7. Reproduce and modify without editing package source

From a fresh checkout and activated Python environment:

```text
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -p "test_v2*.py" -v
cruxrnn demo --output outputs/my-demo
```

For exact recorded numerical versions, follow [reproducibility.md](reproducibility.md)
before running the complete suite. The reference experiment command is:

```text
cruxrnn run --config configs/v2/main.json --workers 4
cruxrnn verify --config configs/v2/main.json
cruxrnn analyze --config configs/v2/main.json
cruxrnn figures --config configs/v2/main.json
```

Use `--resume` only to verify complete cells or continue incomplete checkpoints.
To inspect the original protocol, `cruxrnn original` prints the preserved v1
command; `--execute` runs its guarded verification/resume in the recorded v1
environment. Do not run it to replace old successful outcomes.

Individual extension selectors are `main`, `architecture`, `capacity`,
`granularity`, `forgetting`, `noise`, and `dynamics`; the last four reuse the
appropriate main jobs. Start with `main`, then run required extensions with
`--resume`; full-suite analysis expects every planned cell. Raw per-cell CSVs
are available immediately, even before aggregation.

A custom pair needs no package edits:

```text
cruxrnn cell --config configs/v2/smoke.json --architecture gru --hidden-size 16 --distance 3 --run-id 0 --namespace my-exploratory-study --output outputs/my-exploration
```

This is explicitly exploratory and outside the fixed main plan. Change budgets,
noise, threshold and distances by copying/editing a JSON config, with a new
namespace/output directory and a new prospective plan for serious inference.
Do not edit the frozen main config to fit observed results. CUDA can be requested
with `run --device cuda --workers 1`; it needs an appropriate PyTorch build and
hardware and was not validated on this CPU-only machine.

## 8. Questions a professor may ask

**What is new here?** A controlled similarity extension, larger independent
sample, several exploratory dimensions, corrected reusable engineering and
transparent paired statistics. It is not a claim of inventing RNNs or proving
a biological memory theory.

**Is this data leakage?** The full cue is an intentional task design, openly
described. Fresh evaluation noise and disjoint unseen bases prevent those
specific overlaps, but full-cue decoding remains an easy/limited task.

**Is schedule assignment randomized?** Block orders are deterministic and
near-balanced. Paired inputs/weights control the intended contrast. Sign-flip
tests require their stated symmetry assumption; this is not a human RCT.

**Why not tune GRU/LSTM separately?** This comparison holds the training budget
and optimizer fixed. It tests robustness of the observed phenomenon under one
protocol, not each architecture's optimized upper performance limit.

**Why can IP have higher action accuracy but worse exact-sequence scores?**
It may spread errors across more examples while producing more correct
individual actions. Those objectives differ; both outcomes must be shown.

**Does a bigger D drop prove weaker memory?** No. Compare starting/final scores,
per-sequence patterns, floors and D distance before interpreting raw drops.

**Why only 20 pairs for extensions?** They are resource-bounded exploratory
comparisons with common reference families, not the 100-pair primary study.

**What would strengthen the work next?** Balance action frequencies/repeats,
test relabeling/one-hot or partial cues, independently replicate with new seeds,
justify architecture-specific optimization and compare with appropriate
behavioral measurements. Each requires a new plan, not a post hoc claim.

## 9. Decisions and independent review

v2 keeps v1 frozen, uses a separate package, preserves full-cue/eight-transition
timing, extends exact k constraints, uses near-balance at n=100, reuses matched
20-ID references and separates all phase namespaces. The audit explains the
ignored v1 config fields and other real shortcomings without inventing a
failure in its executed primary arithmetic.

Code, original visual assets, analysis implementation and documentation were
AI-assisted. Before merging/releasing, independently inspect model/task
semantics, a generated family and its schedules, one checkpoint/metric pair,
bootstrap/sign-flip assumptions, correction families, license scope and every
scientific interpretation. Re-run the documented demo/tests and explain one
figure without relying on generated prose. A functioning pipeline and passing
tests are evidence for implementation behavior, not proof of scientific truth.
