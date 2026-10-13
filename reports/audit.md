# Audit of the pinned reference

Reference: Song, Youngjo; Kim, Hakjoo; Kim, Taewon (2026), [Scientific Reports](https://www.nature.com/articles/s41598-026-40162-w), DOI 10.1038/s41598-026-40162-w. [Accessible article text](https://pmc.ncbi.nlm.nih.gov/articles/PMC13022130/).

Repository: https://github.com/YJ-0000/RNN_during_RP_and_IP . Inspected commit: `31f58bb6de90d1fffcb752c0b97afb7df51479a7`. Checkout exists only in ignored `external/author_repo`; its contents are not part of this project's distributable source. This project implements the mathematical behavior independently. No author figures are reused.

The paper Figure 2 caption states **10,000 pretraining trials**. The paper Methods (“Training and model evaluation”) states **1,000 trials**, consistent with the inspected code's 10 sequences × 100 examples. Main reproduction uses 1,000; a separate optional configuration tests the Figure 2 value. This discrepancy is unresolved by the authors' material inspected here.

## File-level inspection

| File | Observations |
|---|---|
| `Code01_Interleaved_vs_Repetitive.py` | Learning rate .02, batch size 20, 7/7/4 dimensions, pretraining 10×100, main training 3×5000, and generalization 100×100. Pretrained model is copied, fresh SGD optimizers use zero momentum. Retention inputs are the same blocked training inputs for both schedules. Global NumPy/Torch seeds are set before `Parallel(n_jobs=-1)` with 100 simulations; explicit run-level seed bundles are not saved. Saved arrays appear in `results_lr_0_02_save`. |
| `utils/DataGen.py` | Random action classes 0–3, seven positions, cues shifted to 1–4. Gaussian noise SD .1 modifies inputs alone. Interleaved examples are an exact permutation of blocked examples. No uniqueness or cross-set overlap rejection is performed for supposedly unique/novel base sequences. |
| `utils/NetworkModels.py` | One batch-first tanh RNN, hidden-to-four-logit linear map and four-logit-to-seven-input feedback. Hidden reset every example. One initial recurrent transition, then seven outputs and seven feedback transitions: **eight RNN calls**, with the last feedback transition unused by the loss. Raw logits are fed back. |
| `utils/TorchFunctions.py` | Consecutive batches, cross-entropy, no epoch recycling. Main retention uses training inputs rather than fresh evaluation noise. Vulnerability retention slices the **last third**, hence only the last main sequence. Weight noise SD .1 and random pruning .05 accumulate; index zero is an unperturbed baseline. Interference uses 2000 inputs/100 SGD updates plus baseline. Separate vulnerability calls generate different interference data for RP and IP. |
| `Code02_Plot.py` | Loads saved arrays and draws loss plots and comparisons. Learning-curve bands at lines 113 and 116 divide run-axis standard deviation by sqrt(`shape[1]`), the update-axis length (750), instead of run count (`shape[0]`, 100): wrong effective sample count. Other vulnerability bands use the run count. Our analysis uses run-level resampling. |
| `environment.yml` | Export of Linux-specific Conda build strings, Python 3.8.10 and CPU PyTorch 1.13.1. This cannot be imported verbatim as a Windows environment. Portable version targets are provided separately. |

**No repository root LICENSE was found during the audit.** The article's license does not establish a license for repository source. No author source was copied into the independent package. Local numerical parity tests import the ignored reference only when it is present.

## Preservation versus controlled extension

`source_exact` retains duplicate/overlapping bases if randomly drawn, training-input retention, final unused recurrent transition, and last-sequence-only optional vulnerability evaluation. It does not silently repair those source facts. Deterministic named seed bundles and reproducible sequential execution are explicit provenance additions; exact historical random seeds cannot be reconstructed from the saved arrays. The plotting-axis bug is documented, not used to fabricate comparable confidence intervals.

`controlled_similarity` enforces disjoint unique bases, independent held-out noise, fixed all-sequence development probes, balanced blocked orders, and paired identical D exposures. These controls are part of the extension, not assertions about the original experiment. Per-file SHA-256 provenance is saved in `outputs/provenance/source_files.json` by the setup script.
