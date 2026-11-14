# v2 code audit and baseline

Audited baseline: `c1fdcadaca0ace46cf590bbf0859ef8c523cad68`, the unchanged
`v1.0-final-experiment` release. Audit started 2026-11-14 (America/Chicago).
The 17 existing tests passed, including the locally available author-model
output/gradient comparison and interrupted/resumed training equivalence.

## Scope and evidence

Inspected the seven numerical modules, runner/provenance/analysis scripts,
configuration variants, tests, notebook cells, source audit, methods, results,
deviations, historical execution disclosure, professor questions, packaging,
report builders, dependencies, Git history/remotes and tracked output indexes.
The paper is [Song et al. (2026)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13022130/).
The source reference is pinned to `31f58bb6de90d1fffcb752c0b97afb7df51479a7`.
This audit does not equate qualitative author-array agreement with recovering
the authors' unavailable random streams.

## Findings and response

| Actual finding | Consequence | v2 response |
| --- | --- | --- |
| `experiment.execute_cell` calls `optimizer_for` without the configured learning rate; batch sizes and sustained-threshold defaults are also hardcoded. | Editing a descriptive config can leave actual behavior unchanged. The executed v1 values equal these defaults, so this does not invalidate its reported runs. | Validated typed v2 config; pass every operational parameter explicitly; test nondefault settings. |
| `data.family` accepts only 1,2,4,7; `balanced_assignment` requires a multiple of six. | Cannot run all distances or exactly 100 replicates. | Generalize 1–7; assign all six orders with counts differing by at most one, rotate the remainder across distances, save assignments. |
| `analyze.bootstrap` requires a 30×4 matrix and hardcodes 10,000 resamples. | Cannot analyze an expanded sample or another planned study. | Reusable statistics with validated matrices, configured resamples, paired effects, hypothesis tests and Holm correction. |
| Model/evaluation/training are CPU-specific; dimensions, data cardinalities and paths are tightly coupled. | Architecture/capacity changes and acceleration are difficult to validate. | A separate typed `cruxrnn` package with explicit architecture, device, schedules and configuration. Keep frozen `motor_rnn` byte-for-byte. |
| Evaluation infers A/B/C from divisibility by three; generalization also gets the old family's shared/differing position masks. | Callers can mislabel arbitrary datasets/groups. Overall v1 generalization accuracy is unaffected. | Explicit `main_groups` intent; suppress shared/differing and A/B/C for unseen bases. |
| Several report scripts execute on import and target historical paths; the methods introduction still says the already completed confirmation is unexecuted. | Easy to regenerate the wrong report or read inconsistent status. | Preserve historical outputs; v2 CLI/report pipeline uses dedicated paths and versioned documentation. Correct the old introduction's status explicitly, without changing its numerical methods. |
| No console entry point, Ruff config, Actions workflow, install verification, bilingual overview or public interactive explorer. | Reproduction and review require knowing project-specific scripts. | Installable CLI, modern documented packaging, scoped lint/format checks, CPU CI smoke, original offline HTML/SVG demonstration. |
| v1 uses pointwise bootstrap intervals without multiplicity correction, effect sizes or hypothesis tests. | Its estimates are useful but cannot be advertised as adjusted familywise significance. | Leave v1 claims unchanged; prospectively define v2 tests, correction families, interaction/trend and sensitivity analyses. |
| Most functions have no annotations and validation uses assertions. | Public API errors are unclear and checks can disappear under `python -O`. | Explicit v2 validation exceptions and useful types/docstrings; assertions retained only in frozen historical code. |
| Local raw checkpoints are ignored, with a separately verified release archive. | Source checkout alone cannot inspect those weight files. | Retain v1 asset, publish v2 seed-level metrics and checksum indexes, supply a local v2 raw-data archive without placing large binaries in Git. |

## Controls that already work

Paired models share pretrained weights, complete noisy training examples,
exposures and update counts. Development/final noise streams are separate;
controlled main/pretraining/generalization/D bases are unique and disjoint.
Evaluation preserves weights. Bootstrap subjects are run IDs, not predictions.
k=7 shared accuracy is NaN with count zero. No new leakage or arithmetic error
was found in the reported v1 primary outcome. Full target cues, ordinal labels,
unmatched action frequencies and random D distance are disclosed design limits,
not newly discovered software bugs.

## Preservation and review

No v1 scientific file or published numerical result will be rewritten to make
the upgrade look successful. The v2 package and new outputs live alongside it.
The original source repository has no root license; its code/figures remain
excluded. Any proposed license applies only to independently authored CruxRNN
material. The owner must review scientific interpretations, licensing and the
AI-assisted implementation before merging the v2 PR or publishing a release.
