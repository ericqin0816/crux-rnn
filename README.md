# Practice schedules and movement-sequence similarity

An independently implemented, executed student research project based on [Song, Kim, and Kim (2026)](https://doi.org/10.1038/s41598-026-40162-w). It compares blocked and interleaved practice in a 7-unit recurrent neural network and extends the study with exact A/B/C Hamming distances 1,2,4,7. Results concern this artificial system only.

Start with [RUN_REPORT.md](RUN_REPORT.md), [reports/plain_english.md](reports/plain_english.md), and [notebooks/walkthrough.ipynb](notebooks/walkthrough.ipynb). Methods, audit, deviations, and professor questions are in `reports/`. Full generated data/checkpoints/metrics/figures remain locally in ignored `outputs/`; they are not silently included in a source-only Git distribution.

The deliverable archive `artifacts/motor_rnn_project.zip` contains this project's source and all generated outputs, plus an internal file-hash manifest. It excludes the author checkout and downloaded Python/package runtime. Archive checksum and CRC verification are recorded in `artifacts/delivery_manifest.json`.

## Environment

The requested portable Conda target is `envs/reproduction.yml`. On this Windows machine Conda was unavailable; all requested Python/package versions were nevertheless installed exactly using a local Python 3.8.10 runtime. To reproduce that setup:

```powershell
./scripts/setup_windows.ps1
$taskPython = Join-Path (Get-Location) 'envs/tools/python3810/tools/python.exe'
& $taskPython scripts/system_info.py
& $taskPython -m unittest discover -s tests -v
```

For a Conda machine, use `conda env create -f envs/reproduction.yml`, activate `motor-rnn-reproduction`, and substitute `python` below. Source and tests use the standard scientific stack and standard-library unittest. The notebook only reads saved results. Notebook display requires a Jupyter installation of your choice; Jupyter is not part of the numerical experiment environment.

## Commands

```powershell
& $taskPython scripts/run_experiment.py --config configs/source_exact.json --limit 5
& $taskPython scripts/compare_source.py --limit 5 --label first_five
& $taskPython scripts/run_experiment.py --config configs/source_exact.json --resume
& $taskPython scripts/run_experiment.py --config configs/smoke.json
& $taskPython scripts/analyze.py --config configs/smoke.json
& $taskPython scripts/run_experiment.py --config configs/pilot.json
& $taskPython scripts/analyze.py --config configs/pilot.json
& $taskPython scripts/freeze_final.py
& $taskPython scripts/run_experiment.py --config configs/final.json
& $taskPython scripts/analyze.py --config configs/final.json
```

The final freeze script intentionally refuses to overwrite an existing final config. To continue an interrupted phase, add `--resume` to its run command. Completed cells are hash-verified and skipped; mismatched checkpoints raise errors. Retain failed outputs and inspect `failures.jsonl`. Do not restart a seed to obtain a preferred result. To run a separate new experiment, use a new phase directory and namespace; do not overwrite the frozen final study.

The optional `configs/source_figure2_sensitivity.json` uses 10,000 pretraining examples; it is provided separately. `configs/source_vulnerability.json` enables optional source vulnerability protocols in a separate namespace/directory. Both configurations are unexecuted. The main controlled D experiment always runs. Neither an unexecuted configuration nor author saved arrays should be reported as a new simulation.

## Where to find data

- `outputs/<phase>/runs/run_NNN_kK/`: immutable `data.npz`, sequence/seed `manifest.json`, example orders, pretrained weights, resumable checkpoints, raw metrics, training/development/D curves, paired checks, and `complete.json` with artifact hashes.
- `outputs/final/analysis/`: merged raw metrics, `run_level.csv`, `paired_differences.csv`, `primary_summary.csv`, bootstrap resampling arrays, trend and secondary summaries, execution times, and verification counts.
- `outputs/final/figures/`: five original PNG/PDF figure pairs, with clean/noisy panels labeled separately.
- `outputs/source_exact/comparison/`: author saved-array hashes/shapes, new versus author aggregates, qualitative comparisons, and our training plots.
- `outputs/environment/` and `outputs/provenance/`: system information, pip freeze, CPU benchmark, source hashes, and commit.

Only the ignored `external/author_repo` contains author source; no repository root license was found. Obtain that local audit reference with `git clone https://github.com/YJ-0000/RNN_during_RP_and_IP external/author_repo`, then `git -C external/author_repo checkout 31f58bb6de90d1fffcb752c0b97afb7df51479a7`. The project's model parity test uses it only if available. No author figures are copied.
