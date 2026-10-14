# CruxRNN: deviations and implementation repair ledger

This file records environment attempts, mechanical repairs, and any departures from the requested design. It is updated from execution evidence. Scientific outcomes never justify rerunning or excluding a seed.

## Environment, before simulation

- `conda env create -f envs/reproduction.yml` could not launch because Conda is absent. PowerShell reported: “The term 'conda' is not recognized as a name of a cmdlet, function, script file, or executable program.” The subprocess attempt is saved in `envs/pinned_attempt.json`, including `FileNotFoundError: [WinError 2] The system cannot find the file specified`.
- Downloaded the Python 3.8.10 Windows NuGet runtime into ignored `envs/tools/python3810`. Its attempted `-m venv .venv` failed: `ensurepip --upgrade --default-pip` returned exit status 101. The resulting venv launcher reported `Unable to create process using ...python.exe -m pip install ...`. Used the isolated local NuGet runtime directly instead; no system Python packages were changed.
- All requested principal versions installed exactly: Python 3.8.10, PyTorch 1.13.1 **CPU** (wheel version string `1.13.1+cpu`), NumPy 1.24.3, pandas 2.0.3, SciPy 1.10.1, Matplotlib 3.7.2, scikit-learn 1.3.0, joblib 1.4.2. This is a packaging/runtime-location deviation, not a numerical package-version relaxation. No Conda environment exists; its unavailable status is saved, alongside a full pip freeze and system report. `envs/reproduction.yml` remains the requested portable Conda specification; `envs/requirements-exact.txt` records the executed Windows alternative.

## Reproduction provenance additions

- Named BLAKE2b run-level seeds replace the author's unsaved per-worker random streams. Sequential CPU execution uses one Torch intra-op and one inter-op thread, following measured tiny-model thread benchmarks. Exact historical run identities cannot be recovered. Therefore comparisons are distributional and qualitative, not exact historical reruns.
- Source-compatible main retention, lack of uniqueness enforcement, and eight-transition forward timing are retained. Controlled overlap prevention applies to the extension; applying it to `source_exact` would silently change the audited source design.
- The optional source weight-noise, pruning, and last-sequence interference protocols are implemented but are disabled in the main reproduction configuration. Saved author vulnerability arrays are audited and compared descriptively; the controlled D experiment is a separate paired secondary protocol.
- The optional Figure 2 pretraining sensitivity configuration is provided, with 1000 copies of each of ten bases (10,000 total); it is not part of the primary final experiment.

## After pilot

All 20 pilot run-by-distance cells were hash-verified, with 40 schedule observations and no missing cells or recorded mechanical failures. Mean primary accuracies spanned approximately 38–96%; k=1 interleaved performance approached a ceiling, while larger distances left substantial error. The blocked k=1 accuracy was 90.48% in all five pilots, consistent with six shared positions plus one last-block differing position. This is a property of the prespecified task, not an implementation failure. Complete-sequence accuracy had floors in some cells. No training budget, architecture, optimizer, noise, generator, evaluation, threshold, or exclusion settings were changed after pilot. Final config adds explicit documentation of already-fixed dimensions and SGD parameters and an implementation fingerprint; these do not change executed behavior. Pilot inference is diagnostic only and is not pooled into final results.

## Delivery repairs after simulation

- The first notebook-builder execution failed with `SyntaxError: EOL while scanning string literal`: its Python strings interpreted an intended notebook `\n` escape too early. Changed notebook-code template strings to raw strings and executed all cells again. Initial failure evidence is retained in `reports/notebook_initial_failure.txt`; the successful execution log is separate. No simulation or analysis values changed.
- Visual inspection found that the k=7 shared-position annotation overlapped the legend in the noisy error panel. Moved only that annotation into empty space. The frozen training implementation/configuration is unchanged; figures were regenerated from the same saved data and deterministic bootstrap indices.
- Checkpoints are saved atomically every 50 main updates by replacing the latest checkpoint; every historical 50-update file is not archived. This satisfies restart cadence but limits retrospective inspection of intermediate weight states. Full loss and probe histories remain saved.

## Later pilot-review instruction and CruxRNN provenance

The earlier request had already been executed through 120 final paired cells when the later instruction to stop at pilot review arrived. Existing outputs and original documents are preserved. This review launches no additional final seeds, and an explicit execution gate blocks the final phase. The new proposal retains scientific parameters but uses a fresh fixed namespace/output location because the old final namespace has already been executed. Prior outcome exposure is disclosed; it cannot be retroactively undone.

Original producing Git SHAs and phase UTC start/end times were not recorded before execution. The first commit `2673c9ca11ca9fc701a1d13129f0cb371385ea70` was a later snapshot, not a contemporaneous producing commit. New meaningful commits record actual review changes without rewriting that history. Future executions require a clean committed tree and capture Git/environment/config/time and outcome metadata. Existing numerical modules are unchanged, so old outputs are not attributed to changed experiment code. The newly retitled notebook reads curated pilot copies; historical final notebook/report bytes remain archived.

The user identified `ericqin0816` as the intended GitHub owner. The connector reported another account; it was not used for writes. Following the user's browser instruction, the browser's owner selector verified `ericqin0816`, and an empty public `crux-rnn` repository was created. The local remote uses its verified clone URL. No code/results were pushed, and no final release/tag was made.

## Approved confirmatory execution

The paragraphs above record the earlier execution and pilot-review periods. After explicit user approval, the exact pre-final checkpoint was pushed and verified. The frozen config and numerical modules were unchanged. Before prospective training, the requested descriptive sample SD (ddof=1), comprehensive group summary and 300-dpi export were added; bootstrap inference was unchanged. Commit `28a2b07bb40afcf7af9b907612e01ce722902bcc` contains those approval/reporting changes and is the recorded producing commit. Reporting/curation helpers were prepared while training ran, without changing the runner, analysis procedure, numerical modules or config. All 120 new pairs completed once; there were no execution failures, exclusions, resumes, forced overrides, or replacements. Source/smoke/pilot and the earlier exploratory final remain separate. Current full environment evidence is retained separately from the historical snapshot.
