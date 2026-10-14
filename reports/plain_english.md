# CruxRNN in plain English

We asked whether practicing three artificial seven-action patterns in separate blocks or shuffled together changes how this small network reproduces them. Larger k means the patterns differ at more positions and are less similar. We completed 30 paired simulations at each of four distances, using fresh seeds after the pilot approval.

- At k=1: blocked 89.00%, interleaved 91.12%; mean advantage 2.11 percentage points, 95% bootstrap interval [-1.09, 5.44].
- At k=2: blocked 80.22%, interleaved 86.65%; mean advantage 6.43 percentage points, 95% bootstrap interval [3.88, 9.42].
- At k=4: blocked 61.01%, interleaved 76.02%; mean advantage 15.01 percentage points, 95% bootstrap interval [11.72, 18.46].
- At k=7: blocked 40.16%, interleaved 79.03%; mean advantage 38.87 percentage points, 95% bootstrap interval [34.55, 43.12].

The intervals include zero at k=1; the direction at these distances remains uncertain. The categorical mean differences increase across the four distances.

The prespecified slope is **6.124 pp per Hamming unit**, with a run-ID bootstrap 95% CI **[5.383, 6.853]**. The straight line summarizes the measured distances, and does not establish a rule for other tasks.

The uncertainty comes from independent simulations. Hundreds of noisy examples from one model do not become hundreds of independent participants. A confidence interval above zero supports a positive mean difference within this computational protocol; it does not settle biological or behavioral questions.

Shared positions can make action accuracy high even when a model does not reproduce a whole sequence correctly. We therefore also report complete-sequence scores, clean cues, unseen bases and performance after learning D. A high initial score leaves more accuracy available to lose during D.

The complete target sequence is supplied in the initial cue. These results describe numerical cue decoding by one small RNN with a fixed budget, ordinal action codes, and unmatched action frequencies/repeats. D similarity is observed rather than controlled. Thirty simulation IDs quantify uncertainty within this model/generator; noisy copies and positions are not independent subjects. Prior exploratory outcomes were seen, so this is a disclosed independent confirmation, not a retroactively untouched preregistration. No human brain mechanism or climbing-performance effect is demonstrated.

| Stage | Completed paired cells | Role |
| --- | --- | --- |
| Source reproduction | 100 (first five audited) | Audited source behavior; uncontrolled similarity |
| Smoke | 4 | Pipeline only |
| Pilot | 20 (five IDs × four k) | Diagnostic only |
| Earlier exploratory | 120 | Exposed outcomes; producing SHA/timestamps missing; excluded from new inference |
| New confirmatory final | 120 (30 IDs × four k) | Fresh approved namespace, complete contemporary provenance |

IP has lower mean noisy complete-sequence accuracy at k=2,4,7, despite its higher mean action accuracy. Mean noisy D forgetting is greater under IP at k=4,7; baseline accuracy limits how much a model can lose. Mean clean unseen-base action accuracy spans 25.26–29.25% and noisy accuracy spans 25.27–29.20%, close to the 25% four-action baseline. These descriptive results limit broad claims of better learning, faster learning or transfer.

The prior exploratory study is disclosed and kept separate. The current results use only `cruxrnn_final_proposed_v1`. Read `results.md` for all metrics and `../RUN_REPORT.md` for hashes, runtime and reproduction steps.
