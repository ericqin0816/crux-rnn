# Reproducibility and installation

## Quick software check

Use a fresh checkout, a virtual environment and a CPU PyTorch wheel:

```text
git clone https://github.com/ericqin0816/crux-rnn.git
cd crux-rnn
git switch v2-research-upgrade
python -m venv .venv
```

Activate `.venv` (`.venv\Scripts\Activate.ps1` on Windows or
`source .venv/bin/activate` on Linux/macOS), then:

```text
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
python -m ruff check
python -m ruff format --check
python -m unittest discover -s tests -p "test_v2*.py" -v
cruxrnn demo --output outputs/my-demo
```

Modern v2 tests and the demo do not promise bitwise agreement with the
historical numerical environment. The complete historical tests require that
environment; old checkpoint-loading behavior differs in newer PyTorch.
GitHub Actions separately checks the recorded environment and modern v2.
Use a fresh demo directory or `--resume` to avoid replacing successful data.

## Recorded research environment

The recorded scientific runs use Python 3.8.10, PyTorch 1.13.1+cpu,
NumPy 1.24.3, pandas 2.0.3, SciPy 1.10.1, Matplotlib 3.7.2,
scikit-learn 1.3.0 and joblib 1.4.2. The latter two are historical dependencies,
not needed by the new core. Principal versions were retained from v1.
CPU float32, one intra-op and one inter-op thread per process are used.

`envs/reproduction.yml` specifies the portable Conda target. Conda was absent
on the actual Windows machine, so an isolated NuGet Python runtime and pip
were used. The v2 editable build required updating local build tools to
pip 25.0.1, setuptools 75.3.2 and wheel 0.45.1; this did not change numerical
package versions. Ruff is pinned to 0.9.10 for repeatable lint behavior.

With Python 3.8.10 active:

```text
python -m pip install --upgrade pip==25.0.1
python -m pip install -r envs/requirements-exact.txt
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
```

On the original Windows checkout, the interpreter is
`envs/tools/python3810/tools/python.exe`; use `python -m cruxrnn` if its
Scripts directory is not on PATH. Do not commit the runtime or a virtualenv.

## Research execution

The [experiment plan](experiment_plan_v2.md) and config must be committed
before final data generation. Run from the repository root. Research checks
actual numerical/config/plan bytes against Git; unrelated documentation edits
are recorded rather than mislabeled as a clean whole tree. Each cell has:

- Full recipe, exact architecture/budgets and raw/canonical config hashes.
- Named seed assignments, generated base sequences and changed positions.
- Compressed master cues/targets and saved complete schedule orders.
- Initial paired state hashes and main/D checkpoints with optimizer/RNG state.
- Training loss, fixed probe histories, held-out metrics and run-level results.
- Producing Git SHA, numerical fingerprint, UTC times, environment and attempt
  records, followed by a complete per-file SHA-256 index.

CPU parallelism is configurable with `--workers`; four were used here. Changing
worker count does not change named cell seeds. The first 20 main IDs are reused
as references across extensions; this is intentional matching, not accidental
independence. All final data use a namespace distinct from smoke/pilot/v1.

## Resume and failures

Checkpoints are atomic and saved at least every 50 main/D updates, with final
loss/probe histories. `--resume` verifies and skips complete cells, or continues
incomplete checkpoints only when recipe/data/numerical hashes match. No force
override exists in v2. Keep failure records and successful outputs. Never
restart an unfavorable successful seed to seek a more attractive outcome.
Per-execution wall time and summed cell times differ under parallelism.

## Outputs and verification

Full v2 raw data live in `outputs/v2/main/` (ignored). Compact public tables,
distributions, gzip metric/manifests/checksum indexes, execution records and
PNG/SVG figures live in `results/v2/`. Bootstrap arrays remain in the full raw
archive. `cruxrnn verify` checks all expected cells, exact distances,
independence, exposures, initial weights, budgets, checkpoint histories,
k=7 empty masks and metric identities before inference.

The v2 raw-data archive is a local delivery artifact pending owner review;
its receipt records bytes, CRC and SHA-256. The v1 release retains its existing
public archive. V2 does not publish a final release or move the v1 tag.
Source-only Git downloads provide compact outputs and regeneration commands,
not an implied copy of every historical checkpoint.

Optional `--device cuda --workers 1` requires a compatible CUDA PyTorch build.
The CLI enables deterministic cuBLAS workspace settings, but GPU execution was
not tested on this CPU-only setup and need not be numerically identical.
