# Commands actually executed

Commands below are the experiment-relevant commands, with the local interpreter abbreviated as `$taskPython = 'C:\Users\Eric Qin\Desktop\proj1\envs\tools\python3810\tools\python.exe'`. They were invoked through PowerShell; outputs are preserved in `reports/*_log.txt`, comparison/verification files, and per-phase execution metadata. File creation used the Codex patch tool. This list omits routine directory listings and progress reads.

```powershell
git init
git clone https://github.com/YJ-0000/RNN_during_RP_and_IP external/author_repo
git -C external/author_repo checkout 31f58bb6de90d1fffcb752c0b97afb7df51479a7
conda env create -f envs/reproduction.yml  # failed: executable unavailable
```

The system Python 3.12 downloaded/extracted the Python 3.8.10 NuGet package using `urllib.request.urlretrieve` and `zipfile.ZipFile.extractall` (URL: `https://www.nuget.org/api/v2/package/python/3.8.10`). The local runtime's attempted `-m venv .venv` failed (ensurepip exit 101). Its direct `-m ensurepip --upgrade` succeeded. Exact packages were then installed:

```powershell
& $taskPython -m pip install torch==1.13.1+cpu numpy==1.24.3 pandas==2.0.3 scipy==1.10.1 matplotlib==3.7.2 scikit-learn==1.3.0 joblib==1.4.2 --extra-index-url https://download.pytorch.org/whl/cpu
& $taskPython -m unittest discover -s tests -v
& $taskPython scripts/system_info.py
& $taskPython scripts/benchmark.py
& $taskPython scripts/record_provenance.py
& $taskPython scripts/attempt_environment.py
& $taskPython scripts/run_experiment.py --config configs/source_exact.json --limit 5
& $taskPython scripts/run_experiment.py --config configs/smoke.json
& $taskPython scripts/compare_source.py --limit 5 --label first_five
& $taskPython scripts/analyze.py --config configs/smoke.json
& $taskPython scripts/run_experiment.py --config configs/pilot.json
& $taskPython scripts/run_experiment.py --config configs/source_exact.json --resume
& $taskPython scripts/analyze.py --config configs/pilot.json
& $taskPython scripts/freeze_final.py
& $taskPython scripts/run_experiment.py --config configs/final.json
& $taskPython scripts/analyze.py --config configs/source_exact.json
& $taskPython scripts/compare_source.py --label full_100
& $taskPython scripts/source_overlap_audit.py
& $taskPython scripts/verify_design.py --config configs/pilot.json
& $taskPython scripts/verify_design.py --config configs/source_exact.json
& $taskPython scripts/analyze.py --config configs/final.json
& $taskPython scripts/verify_design.py --config configs/final.json
& $taskPython -m unittest discover -s tests -v
& $taskPython scripts/run_experiment.py --config configs/final.json --resume
& $taskPython scripts/create_walkthrough.py
& $taskPython scripts/write_reports.py
& $taskPython scripts/package_project.py
```

The report writer records only verified completions. See per-phase `events.jsonl` for new versus skipped runs and `complete.json` for hashes. Final resume is a verification-only pass; the original final-training wall time is preserved in `outputs/final/final_training_execution.json`.
