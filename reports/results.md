# CruxRNN: pilot diagnostics

These are five-ID pilot diagnostics, not final scientific findings. Earlier final execution is separately disclosed in `prior_execution.md`. The proposed confirmation is unexecuted.

| k | Schedule | Pilot IDs | Held-out action range (%) | Whole-sequence range (%) | Mean training CE | Sustained 90% reached |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | blocked | 5 | 90.48–90.48 | 33.33–33.33 | 0.547 | 5 |
| 1 | interleaved | 5 | 93.86–99.90 | 57.00–99.33 | 0.574 | 5 |
| 2 | blocked | 5 | 62.14–80.95 | 0.00–33.33 | 0.701 | 0 |
| 2 | interleaved | 5 | 66.67–93.52 | 0.00–61.00 | 0.803 | 0 |
| 4 | blocked | 5 | 58.10–66.67 | 0.00–33.33 | 0.693 | 0 |
| 4 | interleaved | 5 | 68.24–98.86 | 0.00–92.00 | 0.929 | 1 |
| 7 | blocked | 5 | 28.57–47.62 | 0.00–33.33 | 0.770 | 0 |
| 7 | interleaved | 5 | 56.38–95.24 | 0.00–66.67 | 1.091 | 0 |

Finite learning curves and successful complete stages support pipeline readiness. k=1 performance approaches a ceiling and some complete-sequence scores reach a floor. No settings were tuned to obtain a preferred effect. See `../RUN_REPORT.md`, raw CSVs and original pilot figures for evidence and limitations.
