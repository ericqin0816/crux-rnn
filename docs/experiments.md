# Experiments and version boundaries

See [the prospective plan](experiment_plan_v2.md) for hypotheses, sample sizes,
exclusions and correction families. [The final report](../reports/v2_research_report.md)
states actual completion counts and measured results; planned counts below
must not be substituted for completion evidence.

| Study | Planned sample / conditions | Main models | Purpose |
| --- | --- | ---: | --- |
| Expanded RNN | 100 pairs at each k=1–7; h=7 | 1400 | Primary original distances plus all-seven extension |
| Architecture | 20 pairs at k=1,2,4,7; GRU/LSTM h=7 and h=3 | 640 additional | Same hidden size and approximate parameter matching |
| Capacity | 20 pairs at original k; RNN h=16/32/64 | 480 additional | Equal-budget capacity sensitivity |
| Granularity | First 20 original-k RNN references, plus three new schedules | 240 additional | 1000/100-example blocks and deterministic trial cycling |
| D interference | Clone every main model, 100 identical D updates | 2760 adapted clones | Old A/B/C before/after and during interference |
| Noise | SD=0,.1,.3,.6,1 on fresh held-out cues | 0 additional | Robustness of existing trained models |
| Dynamics | Fixed probes and loss throughout main/D training | 0 additional | Speed, per-sequence interference and learning curves |

The 1260 model-variant cells use 700 main family realizations. Architecture
and capacity variants reuse the same first-20-ID families at original
distances. They do not provide extra independent subjects to the main effect.
Five schedules in the granularity subset share one pretrained initialization;
ordinary cells have two schedules. All reference results are reused once.

## Commands

Run the complete committed design:

```text
cruxrnn run --config configs/v2/main.json --workers 4
cruxrnn verify --config configs/v2/main.json
cruxrnn analyze --config configs/v2/main.json
cruxrnn figures --config configs/v2/main.json
```

Or execute incrementally, retaining all completed cells:

```text
cruxrnn run --config configs/v2/main.json --experiment main --workers 4 --resume
cruxrnn run --config configs/v2/main.json --experiment architecture --workers 4 --resume
cruxrnn run --config configs/v2/main.json --experiment capacity --workers 4 --resume
```

`granularity`, `forgetting`, `noise` and `dynamics` select the corresponding
main jobs, because those outcomes are already collected there. They do not
retrain completed cells when `--resume` is used. Full-suite analysis requires
all planned jobs; individual cell CSVs are inspectable immediately.

`configs/v2/smoke.json` uses tiny budgets to exercise every stage;
`configs/v2/pilot.json` uses full budgets at the extremes with two IDs. Both
are diagnostics in separate namespaces. `cruxrnn demo` is a one-ID lightweight
demonstration, not an inferential experiment. The eight-model full-budget
v1 parity check reproduces saved initial data/orders and compares every main
weight and all 750 losses, not a new independent scientific sample.

For a new custom model/condition without editing package code:

```text
cruxrnn cell --config configs/v2/smoke.json --architecture lstm --hidden-size 16 --distance 3 --run-id 0 --namespace custom-pilot --output outputs/custom-pilot
```

This custom pair is labeled exploratory and kept outside the planned main
study. A multi-run research extension needs its own versioned plan and
independent-unit analysis. Do not treat a single custom pair as a population
estimate. Model checkpoints are local trusted files, not a public upload API.

## Historical work

The original [v1 release](https://github.com/ericqin0816/crux-rnn/releases/tag/v1.0-final-experiment)
and [v1 README](../README.v1.md) preserve the 30-pair × four-distance
confirmation. Source reproduction (100 pairs), smoke (4 cells), pilot
(20 cells), earlier provenance-incomplete exploratory work (120 pairs) and
the later v1 confirmation (120 pairs) remain distinct. Their metadata gaps
are not repaired by attributing them to newer commits. Nothing is pooled
with `cruxrnn_v2_main_20261114`.
