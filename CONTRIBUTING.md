# Contributing to CruxRNN

Open a focused issue or PR explaining the scientific or software problem.
Preserve the `v1.0-final-experiment` tag and frozen `motor_rnn` implementation.
New experiments need a new namespace/output directory and a versioned plan
before running final seeds. Never replace successful unfavorable outcomes.

Install the package with `python -m pip install -e ".[dev]"`, then run
`python -m ruff format --check`, `python -m ruff check` and
`python -m unittest discover -s tests -p 'test_v2*.py' -v`.
The exact recorded Python 3.8.10/PyTorch 1.13.1 environment additionally runs
the complete historical suite: `python -m unittest discover -s tests -v`.
Historical numerical files are deliberately excluded from autoformatting;
their raw byte hashes are part of v1 provenance.

Run `cruxrnn demo --output outputs/my-demo` for a lightweight integration check.
Real experiments require committed numerical/config/plan bytes. Checkpoints
contain local trusted Python/PyTorch state; do not load untrusted checkpoints.
Commit compact metrics, seed distributions, manifests and original figures;
keep environments, caches, credentials and large checkpoints out of Git.

Scientific changes should state hypotheses, units, pairing, budgets, exclusions,
statistical correction families and actual completed sample sizes. Distinguish
planned, exploratory, smoke and pilot results. Retain negative/inconclusive
results and document deviations. See [the researcher guide](docs/researcher_guide.md).

Use focused milestone commits, run relevant checks, and push the completed branch.
Do not merge the v2 review branch or publish its final release without owner review.
Disclose AI assistance and identify claims that need independent verification.
