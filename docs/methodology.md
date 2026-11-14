# Methodology

CruxRNN independently reproduces a complete-cue sequence task from
[Song, Kim and Kim (2026)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13022130/)
and extends it by controlling sequence similarity and practice order. The
climbing motivation is an analogy about repetition and related move patterns;
no climbing measurements or human participants enter these experiments.

## Task and model

Each sequence has seven categorical actions, encoded as integers 0–3. The
entire target sequence, shifted to cue values 1–4, enters the network at the
start. Gaussian noise alters cues, never target classes. The network decodes
and unfolds that full cue; it does not predict actions whose identities were
withheld. Outputs are seven sets of four raw logits.

For the Elman network, h starts at zero and

`h_t = tanh(W_in z_t + b_in + W_rec h_(t−1) + b_rec)`,
`logits_t = W_out h_t + b_out`,
`z_(t+1) = W_feedback logits_t + b_feedback`.

Seven outputs contribute to the loss. An eighth recurrent transition is
computed but unused, preserving audited source timing. GRU/LSTM variants
replace the recurrent layer; readout, feedback and task remain the same. LSTM
resets both hidden and cell states per example. Parameter counts include every
projection, bias and recurrent gate.

## Exact similarity and independence

For k=1,…,7, select k positions. B and C each use an action different from A
and each other at those positions; all other positions agree. Every pair
therefore has distance k. A/B/C, ten pretraining bases, 100 unseen bases and
one D base are unique and disjoint within a cell. Action frequencies/repeated
runs are recorded, not matched across distances.

Named deterministic streams generate initialization, base families, training,
development/final/generalization noise, interleaving, D and robustness probes.
Development and final noise are independent. Cross-architecture/capacity
comparisons deliberately reuse families/examples from the matched main IDs;
those repeated conditions do not create extra independent subjects.

## Budgets and pairing

Pretrain on 10×100 examples. Clone once into schedules with equal initial
weight hashes and fresh SGD optimizers. Each schedule sees the same 3×5000
master noisy examples exactly once: 750 updates of batch size 20, learning
rate .02, zero momentum. Evaluate fixed clean/noisy A/B/C probes at updates
0,25,…,750. The main endpoint uses separate fresh-noisy cues before D.

Fully blocked practice finishes one base before switching. Large/small blocks
switch after 1000/100 examples. Trial cycling deterministically rotates bases;
shuffled IP uniformly permutes all master example IDs. Within-batch example
order does not create recurrent carryover: hidden state resets for each cue.
Different batch composition and the sequence of SGD updates are the intended
schedule manipulation. All exposure and update counts remain equal.

All six blocked orders appear 16 or 17 times for n=100; the remainder rotates
across distances. Exact six-way balance at n=100 is impossible. Orders are
fixed by run ID and distance and saved before their models train.

## Evaluation and D

Action accuracy averages all seven predictions, with equal A/B/C weight.
Exact-sequence accuracy requires all seven predictions correct. Cross-entropy,
A/B/C and position scores provide complementary views. Clean cues and 100
fresh noisy copies/base are separate endpoints. Generalization decodes 100
unseen complete cues; it is not transfer to unobserved physical dynamics.
At k=7, shared positions have count zero and accuracy NaN.

Noise robustness uses separately seeded held-out sets at SD 0,.1,.3,.6,1;
training SD remains .1. Each main model is kept intact while a clone learns
2000 identical D examples (100 updates). Old A/B/C development probes run at
baseline and every D update. Final forgetting is before minus after accuracy
on the same held-out final set. D distances are observed, not manipulated.

## Versions

The frozen v1 implementation/results/release are preserved. Its source-exact
mode intentionally retains audited source limitations; see
[the original audit](../reports/audit.md). The v2 plan and distinct namespaces
are in [experiment_plan_v2.md](experiment_plan_v2.md). New results are never
pooled with v1, pilot or smoke observations.
