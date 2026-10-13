"""CruxRNN: report historical verified phases and the unexecuted proposal."""
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from motor_rnn.provenance import file_hash, implementation_hash, write_json

stages = json.loads(Path("provenance/stages.json").read_text(encoding="utf-8"))
by_phase = {stage["phase"]: stage for stage in stages["stages"]}
proposal = json.loads(Path("configs/final_proposed.json").read_text(encoding="utf-8"))
proposal_hash = file_hash("configs/final_proposed.json")
assert proposal_hash == Path("configs/final_proposed.sha256").read_text().strip()
assert not Path(proposal["output_directory"]).exists(), "Do not label an executed proposal unstarted"
gate = json.loads(Path("provenance/execution_gate.json").read_text())
assert gate["final_training_allowed"] is False
assert implementation_hash() == proposal["implementation_sha256"]
environment = json.loads(Path("provenance/environment/system_info.json").read_text())
cpu = json.loads(environment["cpu"]["stdout"])
memory = json.loads(environment["memory"]["stdout"])
test_text = Path("reports/unit_tests_pilot_review.txt").read_text()
test_status = re.findall(r"^test_.*\.\.\. (.+)$", test_text, re.M)
assert len(test_status) == 17 and all(status == "ok" for status in test_status)
report_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
remote = subprocess.check_output(["git", "remote", "get-url", "origin"], text=True).strip()
assert remote == "https://github.com/ericqin0816/crux-rnn.git"
pilot = pd.read_csv("results/pilot/run_level.csv")
times = pd.read_csv("results/pilot/execution_times.csv")
comparison = pd.read_csv("results/source_reproduction/first_five/aggregate_comparison.csv")
source = by_phase["source_exact"]
smoke = by_phase["smoke"]
pilot_stage = by_phase["pilot"]
assert len(pilot) == 40 and len(times) == 20
pilot_wall = json.loads(Path("results/pilot/original_last_execution.json").read_text())["wall_seconds"]
source_index = json.loads(Path("results/source_reproduction/artifact_index.json").read_text())
five_seconds = sum(cell["elapsed_seconds"] for cell in source_index["cells"] if cell["run_id"] < 5)
mean_seconds = times.seconds.mean()
projected_cell_seconds = 120 * mean_seconds
projected_wall_seconds = pilot_wall * 6
projected_bytes = pilot_stage["raw_run_artifact_bytes"] * 6


def table(headers, rows):
    return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + "\n".join("| " + " | ".join(str(x) for x in row) + " |" for row in rows)


phase_table = table(["Phase", "Verified pairs", "Failures", "Exclusions", "Summed cell seconds"],
                    [["Source first-five audit", 5, 0, 0, "{:.6f}".format(five_seconds)],
                     ["Source main reproduction (including first five)", source["completed_cells"], source["failure_records"], 0, "{:.2f}".format(source["cell_runtime_seconds"])],
                     ["Smoke: one ID × four distances", 4, 0, 0, "{:.6f}".format(smoke["cell_runtime_seconds"])],
                     ["Pilot: five IDs × four distances", 20, 0, 0, "{:.6f}".format(times.seconds.sum())],
                     ["New proposed final confirmation", 0, 0, 0, "Not executed"]])
comparison_table = table(["Metric", "New n=5 CE", "Author n=100 CE", "New shape", "Author shape"],
                         [[row.metric, "{:.6f}".format(row.new_mean_loss), "{:.6f}".format(row.author_mean_loss), row.new_shape, row.author_shape] for row in comparison.itertuples()])
diagnostics = []
for (k, schedule), group in pilot.groupby(["distance", "schedule"]):
    diagnostics.append([int(k), schedule, len(group), "{:.2f}–{:.2f}".format(100 * group.final_noisy_action_accuracy.min(), 100 * group.final_noisy_action_accuracy.max()),
                        "{:.2f}–{:.2f}".format(100 * group.final_noisy_sequence_accuracy.min(), 100 * group.final_noisy_sequence_accuracy.max()),
                        "{:.3f}".format(group.training_cross_entropy.mean()), int(sum(str(value) != "not_reached" for value in group.threshold_start))])
diagnostic_table = table(["k", "Schedule", "Pilot IDs", "Held-out action range (%)", "Whole-sequence range (%)", "Mean training CE", "Sustained 90% reached"], diagnostics)
per_id = times.groupby("run_id").seconds.sum()
runtime_table = table(["Pilot ID", "Four-distance cell seconds"], [[int(i), "{:.6f}".format(v)] for i, v in per_id.items()])
forecast = {"project": "CruxRNN", "basis": "pilot only; earlier final runtime not used", "pilot_cells": 20,
            "pilot_mean_seconds_per_cell": float(mean_seconds), "projected_final_cells": 120,
            "projected_cell_seconds": float(projected_cell_seconds), "projected_phase_wall_seconds": float(projected_wall_seconds),
            "pilot_raw_artifact_bytes": pilot_stage["raw_run_artifact_bytes"], "projected_raw_artifact_bytes": projected_bytes,
            "excludes": ["environment setup", "analysis and figure generation", "reporting", "unmeasured peak RAM"],
            "proposal_not_executed": True}
write_json("provenance/pilot_runtime_forecast.json", forecast)

report = f"""# CruxRNN: pilot-review run report

**CruxRNN: Sequence Similarity, Practice Scheduling, and Interference in a Minimal Recurrent Neural Network**

Implementation, source reproduction, smoke, and five-ID pilot are complete and reverified. **The new proposed final confirmation has NOT started and is blocked pending approval.** Earlier final execution had already completed before this later gate arrived; it is preserved and disclosed in [prior execution](reports/prior_execution.md). We cannot truthfully claim that no final experiment ever ran. No new proposed final data/model/optimizer steps, final release, or tag were created during this review.

## Project and test status

Numerical implementation is complete and unchanged from its verified historical snapshot. Current suite: **17 run / 17 passed / 0 failed / 0 skipped / 0 unresolved failures**. Fourteen tests cover the scientific controls; three verify final approval blocking and committed-code provenance. [Full test log](reports/unit_tests_pilot_review.txt). The earlier suite passed 14/14 before original simulation. Notebook: eight code cells executed from curated saved pilot data, without training.

{phase_table}

Counts are verified from completion-record hashes, datasets/manifests, CSVs and checkpoints. No experiment was rerun in this review merely to relabel results or improve an outcome. Read-only artifact checks are in `reports/source_review_verification.txt`, `smoke_review_verification.txt`, and `pilot_review_verification.txt`.

## Source reproduction and audit

The source first-five audit completed in **{five_seconds:.6f} summed cell seconds**; this is measured per-run execution time, not an invented phase wall clock. The first-five phase wall-clock record was not retained when the original resume metadata was overwritten. A later 100-run main reproduction includes those five and took {source['cell_runtime_seconds']:.2f} summed cell seconds. All historical author array shapes were checked; main matched per-run dimensions:

{comparison_table}

Both new and author arrays show lower blocked current-example training loss, positive loss jumps at the two block switches, and lower IP training-input retention and source-generalization CE. The first-five blocked switch jumps were about 1.517 and 2.127 CE. Aggregates differ; exact author random streams were not saved, so exact numerical/historical replication is not claimed. Optional source vulnerability and Figure 2 sensitivity were not rerun. Author saved vulnerability arrays were audited separately.

[Source audit](reports/audit.md) explicitly retains the pretraining caption discrepancy, absent uniqueness enforcement, training-input retention, last-third-only vulnerability, missing run-level seeds, wrong CI sample-count axis, batch-update “epochs,” and root LICENSE uncertainty. Source code remains only in ignored `external/author_repo/`; independent implementation and original figures are version controlled.

## Smoke status

All four reduced cells completed every stage, including pretraining, paired schedules, fixed evaluation, D, checkpoint saving, analysis, and five PNG/PDF figure pairs. Runtime: **{smoke['cell_runtime_seconds']:.6f} summed cell seconds**. No recorded simulation failures. Resume equality is checked by the tiny interrupted/uninterrupted unittest. Smoke has no scientific interpretation. Evidence: `results/smoke/`, original logs, and its artifact index.

## Pilot status and diagnostics

Five independent IDs from `pilot_v1` each cover k=1,2,4,7: **20 paired cells / 40 main schedule models**, with 750 main and 100 D updates per schedule. Failed/restarted/invalidated/excluded pilot runs: **zero**. No numerical implementation flaw invalidated a pilot cell. Original phase wall time: **{pilot_wall:.6f} seconds**. Summed cell time: **{times.seconds.sum():.6f} seconds**; each distance cell ranged {times.seconds.min():.6f}–{times.seconds.max():.6f} seconds, mean {mean_seconds:.6f}.

{runtime_table}

{diagnostic_table}

These are diagnostics, not a final effect estimate. Learning curves are finite and show acquisition under both schedules. k=1 IP approaches a ceiling; k=1 blocked held-out action accuracy was 90.48% in every pilot. Some whole-sequence scores were zero at other distances. Higher distances retain substantial action error. These are limitations of the specified task/budget; no architecture, budget, noise, optimizer, metric, distances, threshold, or exclusion rules were changed to favor a condition. Generalization remains limited. D similarity is observed, not manipulated; raw forgetting also depends on initial accuracy.

Pilot run artifacts total **{pilot_stage['raw_run_artifact_bytes']:,} bytes ({pilot_stage['raw_run_artifact_bytes']/1024**2:.2f} MiB)** before aggregate analysis/figure copies. Peak training RAM was not instrumented; no peak-memory claim is made. Per-cell resources/times: `results/pilot/execution_times.csv`, `artifact_index.json`.

## Environment and machine

{memory['Caption']}; platform string `{environment['os']}`. {cpu['Name'].strip()}, {cpu['NumberOfCores']} cores / {cpu['NumberOfLogicalProcessors']} logical processors. Total visible RAM {memory['TotalVisibleMemorySize']/1024**2:.2f} GiB; available at original capture {memory['FreePhysicalMemory']/1024**2:.2f} GiB.

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

Repository: `crux-rnn`, local folder remains `C:/Users/Eric Qin/Desktop/proj1`; branch **{branch}**. This report is generated using code/publication base commit **`{report_commit}`**. The final documentation commit and clean-tree status are recorded after committing in ignored `provenance/delivery_status.json` and reported to the user; `git rev-parse HEAD` gives the current delivery SHA. A report cannot embed its own resulting commit SHA without changing that SHA.

Historical source/smoke/pilot/final generating commits: **not recorded**. Their first post-execution snapshot is `2673c9ca11ca9fc701a1d13129f0cb371385ea70`; subsequent review verification is labeled separately in `provenance/stages.json`. Do not imply old simulations were executed from the newer CruxRNN commits. Numerical package fingerprint remains `{implementation_hash()}`. Configs, seed manifests, data/checkpoint hashes, metrics, analysis and figures are linked in `results/*/artifact_index.json`. Current orchestration records a clean producing Git commit and UTC pre/post execution metadata for any future phase.

Remote: **`{remote}`**, verified empty through browser and `git ls-remote`. The public repository exists under `ericqin0816`; **nothing has been pushed**. No unrelated repository was overwritten. No LFS, final tag or final release was created. [GitHub setup and later push command](docs/github_setup.md).

## Exact proposed final configuration

Path: **[configs/final_proposed.json](configs/final_proposed.json)**. SHA-256: **`{proposal_hash}`**. Balanced orders: `configs/final_proposed_blocked_orders.json`, assignment SHA-256 `{file_hash('configs/final_proposed_blocked_orders.json')}`. Namespace: `{proposal['namespace']}`; output: `{proposal['output_directory']}`; status: **unexecuted**.

30 independent IDs at each k=1,2,4,7 → 120 paired cells / 240 main-trained schedule models, plus 120 pretraining states and 240 D-clone states. Per cell: 10×100 pretraining examples; 3×5000 main examples; 100 noisy probes/base per independent split and one clean/base; 100 unique unseen bases; 2000 D examples; batch 20; 750 main +100 D updates; SD .1; 7/7/4 tanh model; SGD .02/momentum 0; development interval 25; sustained .9 threshold for three checkpoints; 10,000 run-ID bootstrap samples. All six blocked orders appear exactly five times per distance. Approval is required; no prospective outcomes were consulted.

Forecast based **only on pilot measurements**: {projected_cell_seconds:.2f} summed cell seconds (~{projected_cell_seconds/60:.2f} min), or {projected_wall_seconds:.2f} phase wall seconds (~{projected_wall_seconds/60:.2f} min) by scaling the original pilot wall time sixfold. Raw run-artifact estimate: {projected_bytes:,} bytes ({projected_bytes/1024**2:.2f} MiB), excluding analysis/figures, environment and safety margin. These are estimates, not completed final runtime/storage. See `provenance/pilot_runtime_forecast.json`.

## Files and unresolved issues

- [README](README.md), [audit](reports/audit.md), [methods](reports/methods.md), [deviations](reports/deviations.md), [professor questions](reports/professor_questions.md).
- [Pilot raw metrics](results/pilot/raw_metrics.csv), [run-level metrics](results/pilot/run_level.csv), [summary](results/pilot/primary_summary.csv), [timings](results/pilot/execution_times.csv).
- Pilot PNG/PDF figures: `results/pilot/figures/learning_curves`, `final_accuracy`, `paired_differences`, `D_forgetting`, `position_errors`.
- [Complete tests](reports/unit_tests_pilot_review.txt), [notebook](notebooks/walkthrough.ipynb), [proposal](configs/final_proposed.json).
- Source first-five and 100-run comparisons: `results/source_reproduction/`; smoke: `results/smoke/`.
- Full raw datasets/checkpoints remain local and ignored under `outputs/<phase>/runs/`; locations/hashes/manifests are in each tracked artifact index. Original author source is excluded.

Unresolved: original producing Git/timestamps cannot be reconstructed; earlier final exposure cannot be undone; user approval is still needed for the proposed confirmation; publishing the committed repository remains pending. Scientific uncertainty includes cue decoding, a single narrow architecture/budget, frequency/repetition covariation with distance, pilot floor/ceiling behavior, and secondary D baseline dependence. No human-brain mechanism, climbing-performance effect, statistical significance, or broad novelty is asserted.
"""
Path("RUN_REPORT.md").write_text(report, encoding="utf-8")
Path("reports/results.md").write_text("# CruxRNN: pilot diagnostics\n\nThese are five-ID pilot diagnostics, not final scientific findings. Earlier final execution is separately disclosed in `prior_execution.md`. The proposed confirmation is unexecuted.\n\n" + diagnostic_table + "\n\nFinite learning curves and successful complete stages support pipeline readiness. k=1 performance approaches a ceiling and some complete-sequence scores reach a floor. No settings were tuned to obtain a preferred effect. See `../RUN_REPORT.md`, raw CSVs and original pilot figures for evidence and limitations.\n", encoding="utf-8")
Path("reports/plain_english.md").write_text("""# CruxRNN: what the pilot tells us

The pilot shows that the program can generate the required data, pair the two practice schedules, train the tiny network, evaluate unchanged weights, learn an additional D pattern, save resumable checkpoints, and produce analysis and figures. It used five independent simulation IDs at each of four distances. It had no failed or excluded simulation cells.

This checks software readiness and reveals limits: some similar patterns approach a ceiling, while getting every action correct can be difficult even when individual-action accuracy is fairly high. The settings were not changed to favor an outcome. Five pilot IDs are not the planned 30-ID scientific analysis.

Climbing is an analogy for alternating movement patterns. There are no human participants, brain measurements or climbing measurements. The complete correct pattern is supplied as an initial numeric cue; even generalization tests cue decoding rather than a new physical skill. D similarity is observed rather than manipulated.

An earlier final experiment already ran before the later request to stop at pilot review. Its outputs are preserved and disclosed. The new proposed independent confirmation has not started and requires approval. That history cannot be changed into a claim that final outcomes were never seen.
""", encoding="utf-8")
print(json.dumps({"report": "RUN_REPORT.md", "report_generation_code_commit": report_commit,
                  "tests_passed": len(test_status), "proposal_config_sha256": proposal_hash,
                  "proposed_final_started": False, "historical_final_execution_disclosed": True}, indent=2))
