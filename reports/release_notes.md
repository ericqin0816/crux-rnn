Research question: does sequence similarity change the relative effects of blocked and interleaved practice in a minimal RNN?

Thirty independent replicate IDs each span k=1,2,4,7: 120 paired cells and 240 main models. Each cell pretrains once, clones identical weights, and presents the same 15,000 noisy examples in blocked or interleaved order. Each of six blocked orders occurs five times per k. SGD .02, zero momentum, batch 20; 750 main updates; probes every 25; 100 matched D updates. Primary: equally weighted A/B/C fresh-noisy final action accuracy before D. Ten thousand percentile bootstrap resamples use whole replicate IDs, retaining all four distances for the prespecified linear slope. SD uses ddof=1. No copies/positions serve as independent runs.

| k | pairs | Blocked % | IP % | Mean IP−blocked pp | Median pp | SD pp | 95% CI low pp | 95% CI high pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 30 | 89.002 | 91.116 | 2.114 | 0.000 | 9.097 | -1.086 | 5.437 |
| 2 | 30 | 80.221 | 86.651 | 6.430 | 4.357 | 7.930 | 3.878 | 9.418 |
| 4 | 30 | 61.006 | 76.019 | 15.013 | 14.238 | 9.451 | 11.722 | 18.459 |
| 7 | 30 | 40.163 | 79.029 | 38.865 | 38.905 | 12.254 | 34.549 | 43.122 |

The intervals include zero at k=1; the direction at these distances remains uncertain. The categorical mean differences increase across the four distances.

The prespecified slope is **6.124 pp per Hamming unit**, with a run-ID bootstrap 95% CI **[5.383, 6.853]**.

Completed 120/120 pairs (240 main models); zero failures/exclusions. Phase wall time: **381.342 seconds (6.36 minutes)**. Summed cell execution: **369.015 seconds**. These measured times cover data generation, pretraining, main training, probes, generalization, D and cell saves; they exclude environment setup, analysis, plotting, reporting, and GitHub delivery. Analysis/figure generation took an additional **9.943 seconds**, for **391.285 seconds** of experiment plus analysis/plotting; independent verification/reporting/archive/delivery time is separate.

The complete target sequence is supplied in the initial cue. These results describe numerical cue decoding by one small RNN with a fixed budget, ordinal action codes, and unmatched action frequencies/repeats. D similarity is observed rather than controlled. Thirty simulation IDs quantify uncertainty within this model/generator; noisy copies and positions are not independent subjects. Prior exploratory outcomes were seen, so this is a disclosed independent confirmation, not a retroactively untouched preregistration. No human brain mechanism or climbing-performance effect is demonstrated.

Reproduce with the pinned environment and `scripts/run_experiment.py --config configs/final_proposed.json --resume`, then `scripts/analyze.py --config configs/final_proposed.json` and `scripts/verify_design.py --config configs/final_proposed.json`. Start from a clean checkout. The frozen config SHA is `26f8cc56ad115f9e4bb4bb47b30af4997c205cd2906cb8e5195bf24fda8ff006`. See README and RUN_REPORT for full setup, provenance and data locations. Tracked CSVs, figures and manifests are in `results/confirmatory/`; local full arrays/checkpoints are indexed separately. Source/smoke/pilot and earlier exploratory data were not pooled into final inference.

Raw-data asset: `cruxrnn-confirmatory-data.zip` (117456062 bytes), SHA-256 `092bfe2324a77e6782d80034b3cf4ec5fb79017990d625be8803f0eb5f96b676`. It contains the new confirmation only, including all checkpoints and bootstrap arrays; extract at the release checkout root. An internal SHA-256 manifest and external CRC/hash receipt support verification. All 17 tests passed, all 120 cells were audited, and the saved-data notebook executed all eight cells.
