# CruxRNN package

**CruxRNN: Sequence Similarity, Practice Scheduling, and Interference in a Minimal Recurrent Neural Network**

The distribution name is `crux-rnn`. The import namespace remains `motor_rnn`, as required by the original specification. The independently implemented numerical modules are preserved byte-for-byte from the verified original snapshot `2673c9ca11ca9fc701a1d13129f0cb371385ea70`. Renaming the distribution does not change model behavior, saved weights, data, or deterministic seed derivation.

`model.py` defines the recurrent model; `data.py` generates bases, independent splits, and paired example orders; `metrics.py` evaluates action/sequence accuracy and loss; `training.py` implements SGD and hash-guarded checkpoints; `experiment.py` orchestrates one run-by-distance pair. `provenance.py` supplies SHA-256 hashes and BLAKE2b seed streams.

The fixed seed-domain label inside the implementation is retained for reproducibility. Do not rename it merely to match product branding. Future code changes must be committed and their results associated with that producing commit; existing outputs must retain their historical provenance.
