# CruxRNN: pilot-review run report

**CruxRNN: Sequence Similarity, Practice Scheduling, and Interference in a Minimal Recurrent Neural Network**

Implementation, source reproduction, smoke, and five-ID pilot are complete and reverified. **The new proposed final confirmation has NOT started and is blocked pending approval.** Earlier final execution had already completed before this later gate arrived; it is preserved and disclosed in [prior execution](reports/prior_execution.md). We cannot truthfully claim that no final experiment ever ran. No new proposed final data/model/optimizer steps, final release, or tag were created during this review.

## Project and test status

Numerical implementation is complete and unchanged from its verified historical snapshot. Current suite: **17 run / 17 passed / 0 failed / 0 skipped / 0 unresolved failures**. Fourteen tests cover the scientific controls; three verify final approval blocking and committed-code provenance. [Full test log](reports/unit_tests_pilot_review.txt). The earlier suite passed 14/14 before original simulation. Notebook: eight code cells executed from curated saved pilot data, without training.

| Phase | Verified pairs | Failures | Exclusions | Summed cell seconds |
| --- | --- | --- | --- | --- |
| Source first-five audit | 5 | 0 | 0 | 12.045307 |
| Source main reproduction (including first five) | 100 | 0 | 0 | 235.75 |
| Smoke: one ID × four distances | 4 | 0 | 0 | 0.516126 |
| Pilot: five IDs × four distances | 20 | 0 | 0 | 60.509655 |
| New proposed final confirmation | 0 | 0 | 0 | Not executed |

Counts are verified from completion-record hashes, datasets/manifests, CSVs and checkpoints. No experiment was rerun in this review merely to relabel results or improve an outcome. Read-only artifact checks are in `reports/source_review_verification.txt`, `smoke_review_verification.txt`, and `pilot_review_verification.txt`.

## Source reproduction and audit

The source first-five audit completed in **12.045307 summed cell seconds**; this is measured per-run execution time, not an invented phase wall clock. The first-five phase wall-clock record was not retained when the original resume metadata was overwritten. A later 100-run main reproduction includes those five and took 235.75 summed cell seconds. All historical author array shapes were checked; main matched per-run dimensions:

| Metric | New n=5 CE | Author n=100 CE | New shape | Author shape |
| --- | --- | --- | --- | --- |
| blocked_training | 0.791488 | 0.757624 | (5, 750) | (100, 750) |
| interleaved_training | 0.985135 | 0.975204 | (5, 750) | (100, 750) |
| blocked_retention | 1.285424 | 1.478705 | (5, 15000) | (100, 15000) |
| interleaved_retention | 0.605468 | 0.634231 | (5, 15000) | (100, 15000) |
| blocked_generalization | 2.452984 | 2.475091 | (5, 10000) | (100, 10000) |
| interleaved_generalization | 1.975530 | 1.985807 | (5, 10000) | (100, 10000) |
| pre_generalization | 1.412459 | 1.410960 | (5, 10000) | (100, 10000) |

Both new and author arrays show lower blocked current-example training loss, positive loss jumps at the two block switches, and lower IP training-input retention and source-generalization CE. The first-five blocked switch jumps were about 1.517 and 2.127 CE. Aggregates differ; exact author random streams were not saved, so exact numerical/historical replication is not claimed. Optional source vulnerability and Figure 2 sensitivity were not rerun. Author saved vulnerability arrays were audited separately.

[Source audit](reports/audit.md) explicitly retains the pretraining caption discrepancy, absent uniqueness enforcement, training-input retention, last-third-only vulnerability, missing run-level seeds, wrong CI sample-count axis, batch-update “epochs,” and root LICENSE uncertainty. Source code remains only in ignored `external/author_repo/`; independent implementation and original figures are version controlled.

## Smoke status

All four reduced cells completed every stage, including pretraining, paired schedules, fixed evaluation, D, checkpoint saving, analysis, and five PNG/PDF figure pairs. Runtime: **0.516126 summed cell seconds**. No recorded simulation failures. Resume equality is checked by the tiny interrupted/uninterrupted unittest. Smoke has no scientific interpretation. Evidence: `results/smoke/`, original logs, and its artifact index.

## Pilot status and diagnostics

Five independent IDs from `pilot_v1` each cover k=1,2,4,7: **20 paired cells / 40 main schedule models**, with 750 main and 100 D updates per schedule. Failed/restarted/invalidated/excluded pilot runs: **zero**. No numerical implementation flaw invalidated a pilot cell. Original phase wall time: **62.592907 seconds**. Summed cell time: **60.509655 seconds**; each distance cell ranged 2.941637–3.081741 seconds, mean 3.025483.

| Pilot ID | Four-distance cell seconds |
| --- | --- |
| 0 | 11.945132 |
| 1 | 12.089134 |
| 2 | 12.188326 |
| 3 | 12.086735 |
| 4 | 12.200328 |

| k | Schedule | Pilot IDs | Held-out action range (%) | Whole-sequence range (%) | Mean training CE | Sustained 90% reached |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | blocked | 5 | 90.48–90.48 | 33.33–33.33 | 0.547 | 5 |
| 1 | interleaved | 5 | 93.86–99.90 | 57.00–99.33 | 0.574 | 5 |
| 2 | blocked | 5 | 62.14–80.95 | 0.00–33.33 | 0.701 | 0 |
| 2 | interleaved | 5 | 66.67–93.52 | 0.00–61.00 | 0.803 | 0 |
| 4 | blocked | 5 | 58.10–66.67 | 0.00–33.33 | 0.693 | 0 |
| 4 | interleaved | 5 | 68.24–98.86 | 0.00–92.00 | 0.929 | 1 |
| 7 | blocked | 5 | 28.57–47.62 | 0.00–33.33 | 0.770 | 0 |
| 7 | interleaved | 5 | 56.38–95.24 | 0.00–66.67 | 1.091 | 0 |

These are diagnostics, not a final effect estimate. Learning curves are finite and show acquisition under both schedules. k=1 IP approaches a ceiling; k=1 blocked held-out action accuracy was 90.48% in every pilot. Some whole-sequence scores were zero at other distances. Higher distances retain substantial action error. These are limitations of the specified task/budget; no architecture, budget, noise, optimizer, metric, distances, threshold, or exclusion rules were changed to favor a condition. Generalization remains limited. D similarity is observed, not manipulated; raw forgetting also depends on initial accuracy.

Pilot run artifacts total **26,252,005 bytes (25.04 MiB)** before aggregate analysis/figure copies. Peak training RAM was not instrumented; no peak-memory claim is made. Per-cell resources/times: `results/pilot/execution_times.csv`, `artifact_index.json`.

## Environment and machine

Microsoft Windows 11 Home; platform string `Windows-10-10.0.26200-SP0`. AMD Ryzen 7 8700F 8-Core Processor, 8 cores / 16 logical processors. Total visible RAM 15.59 GiB; available at original capture 2.16 GiB.

Python 3.8.10; PyTorch 1.13.1+cpu; NumPy 1.24.3; pandas 2.0.3; SciPy 1.10.1; Matplotlib 3.7.2; scikit-learn 1.3.0; joblib 1.4.2. CPU, float32, deterministic algorithms, one intra-op/inter-op thread. CUDA/MPS unavailable in the CPU build; no accelerator used. A 100-update thread benchmark measured .1274/.1447/.1392 seconds for 1/2/4 threads, favoring one. Exact environment/freeze/build/system evidence: `provenance/environment/` and `envs/`.

## Deviations and repairs

- Conda was missing (WinError 2); local Python 3.8.10 NuGet plus pip achieved all requested principal versions.
- NuGet venv creation failed at ensurepip with exit 101; using the isolated local interpreter directly worked.
- Named BLAKE2b streams and deterministic sequential one-thread execution add provenance to the source's unsaved worker streams. Exact original seeds cannot be reconstructed.
- Source duplicates/overlaps and evaluation reuse are deliberately preserved; strict disjointness applies to the controlled extension.
- Optional source perturbation/sensitivity protocols are provided but unexecuted. Source main reproduction was expanded from five to 100 based on measured runtime under the earlier request.
- Reduced smoke budgets are intentional. Latest 50-update checkpoints replace earlier checkpoint files; full metric/probe histories remain.
- Notebook newline escaping and one plot annotation overlap were repaired after historical simulation; no numerical results changed.
- The original phases ran before their first Git snapshot, and pre-execution Git SHAs plus phase start/end timestamps were not recorded. This is an unresolved historical provenance gap, not repaired by newer commits.
- The later pilot-only instruction arrived after the earlier final phase had already run. Existing final output/history is preserved; additional final execution is now gated. A fresh proposed namespace/output directory avoids recycling prior final seeds; scientific parameters are unchanged.
- Browser GitHub identity `ericqin0816` was verified and used to create the empty public repository; the mismatched connector identity was not used for writes.

[Full deviations ledger](reports/deviations.md). No seed was selected or excluded because of its scientific outcome.

## Git and phase provenance

Repository: `crux-rnn`, local folder remains `C:/Users/Eric Qin/Desktop/proj1`; branch **main**. This report is generated using code/publication base commit **`1e184c6aed075117c0578ba71b5ee5a2e14a9569`**. The final documentation commit and clean-tree status are recorded after committing in ignored `provenance/delivery_status.json` and reported to the user; `git rev-parse HEAD` gives the current delivery SHA. A report cannot embed its own resulting commit SHA without changing that SHA.

Historical source/smoke/pilot/final generating commits: **not recorded**. Their first post-execution snapshot is `2673c9ca11ca9fc701a1d13129f0cb371385ea70`; subsequent review verification is labeled separately in `provenance/stages.json`. Do not imply old simulations were executed from the newer CruxRNN commits. Numerical package fingerprint remains `e16274af61696a90c379fb16165ca1ff8309e0f1f02d8d4526da050588d4d168`. Configs, seed manifests, data/checkpoint hashes, metrics, analysis and figures are linked in `results/*/artifact_index.json`. Current orchestration records a clean producing Git commit and UTC pre/post execution metadata for any future phase.

Remote: **`https://github.com/ericqin0816/crux-rnn.git`**, verified empty through browser and `git ls-remote`. The public repository exists under `ericqin0816`; **nothing has been pushed**. No unrelated repository was overwritten. No LFS, final tag or final release was created. [GitHub setup and later push command](docs/github_setup.md).

## Exact proposed final configuration

Path: **[configs/final_proposed.json](configs/final_proposed.json)**. SHA-256: **`26f8cc56ad115f9e4bb4bb47b30af4997c205cd2906cb8e5195bf24fda8ff006`**. Balanced orders: `configs/final_proposed_blocked_orders.json`, assignment SHA-256 `7510596d71df132ec3779c85c9bb9375c7cadb90d160308a6ed6fc2d5e943967`. Namespace: `cruxrnn_final_proposed_v1`; output: `outputs/cruxrnn_final_proposed_v1`; status: **unexecuted**.

30 independent IDs at each k=1,2,4,7 → 120 paired cells / 240 main-trained schedule models, plus 120 pretraining states and 240 D-clone states. Per cell: 10×100 pretraining examples; 3×5000 main examples; 100 noisy probes/base per independent split and one clean/base; 100 unique unseen bases; 2000 D examples; batch 20; 750 main +100 D updates; SD .1; 7/7/4 tanh model; SGD .02/momentum 0; development interval 25; sustained .9 threshold for three checkpoints; 10,000 run-ID bootstrap samples. All six blocked orders appear exactly five times per distance. Approval is required; no prospective outcomes were consulted.

Forecast based **only on pilot measurements**: 363.06 summed cell seconds (~6.05 min), or 375.56 phase wall seconds (~6.26 min) by scaling the original pilot wall time sixfold. Raw run-artifact estimate: 157,512,030 bytes (150.22 MiB), excluding analysis/figures, environment and safety margin. These are estimates, not completed final runtime/storage. See `provenance/pilot_runtime_forecast.json`.

## Files and unresolved issues

- [README](README.md), [audit](reports/audit.md), [methods](reports/methods.md), [deviations](reports/deviations.md), [professor questions](reports/professor_questions.md).
- [Pilot raw metrics](results/pilot/raw_metrics.csv), [run-level metrics](results/pilot/run_level.csv), [summary](results/pilot/primary_summary.csv), [timings](results/pilot/execution_times.csv).
- Pilot PNG/PDF figures: `results/pilot/figures/learning_curves`, `final_accuracy`, `paired_differences`, `D_forgetting`, `position_errors`.
- [Complete tests](reports/unit_tests_pilot_review.txt), [notebook](notebooks/walkthrough.ipynb), [proposal](configs/final_proposed.json).
- Source first-five and 100-run comparisons: `results/source_reproduction/`; smoke: `results/smoke/`.
- Full raw datasets/checkpoints remain local and ignored under `outputs/<phase>/runs/`; locations/hashes/manifests are in each tracked artifact index. Original author source is excluded.

Unresolved: original producing Git/timestamps cannot be reconstructed; earlier final exposure cannot be undone; user approval is still needed for the proposed confirmation; publishing the committed repository remains pending. Scientific uncertainty includes cue decoding, a single narrow architecture/budget, frequency/repetition covariation with distance, pilot floor/ceiling behavior, and secondary D baseline dependence. No human-brain mechanism, climbing-performance effect, statistical significance, or broad novelty is asserted.
