# CruxRNN: curated reproducibility outputs

These files are copied from hash-verified historical execution artifacts. They are not newly executed simulations. `source_reproduction/` contains five-run and 100-run comparisons, summary metrics, timings, manifest copies, and a full artifact hash index. `smoke/` contains reduced-budget pipeline checks. `pilot/` contains the five replicate IDs at all four distances, including raw metrics, learning/D curves, summaries, and original figure pairs.

Pilot and smoke statistics are diagnostic. Their nominal intervals must not be presented as final inferential results. The current approval gate blocks any additional final experiment. Earlier final execution is disclosed in `reports/prior_execution.md`; its original local outputs and historical documentation are preserved.

Large generated arrays, individual model checkpoints, and bootstrap resampling arrays remain under ignored `outputs/`. Each phase's `artifact_index.json` specifies their locations, SHA-256 hashes, original namespace, configuration, and completeness. Dataset generation and runner code are version controlled; use the exact recorded environment, configuration, and seeds to reproduce them. Source-exact novelty/uniqueness limitations are retained.

Historical phases have no contemporaneously recorded producing Git commit. Their later verified snapshot and review commit are explicitly labeled; they must not be substituted for an unknown generating commit.
