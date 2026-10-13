"""Build and execute a student notebook using saved outputs only."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path

cells = []
def markdown(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})
def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "source": text.splitlines(True), "execution_count": None, "outputs": []})

markdown("""# Movement patterns, practice order, and a tiny RNN

This notebook walks through **saved, verified simulations**. It does not train models. You need basic Python, NumPy, and pandas. The numerical experiments used Python 3.8.10 and PyTorch 1.13.1 CPU.

We compare practicing three movement patterns in separate blocks with practicing the same examples in a shuffled order. Climbing offers an analogy: you might repeat one move pattern or alternate several. The data here are artificial action labels, not climbing measurements. The results concern this network and do not prove anything about human brains.

Read `../RUN_REPORT.md` for completion counts, uncertainty, runtime, and deviations.
""")
code(r"""from pathlib import Path
import json
import numpy as np
import pandas as pd

# This supports launching Jupyter from the project or notebooks folder.
root = Path.cwd()
if not (root / 'configs' / 'final.json').exists():
    root = root.parent
assert (root / 'configs' / 'final.json').exists()
analysis = root / 'outputs' / 'final' / 'analysis'
verification = json.loads((analysis / 'verification.json').read_text())
print('Verified final run pairs:', len(verification['completed']))
print('Missing final run pairs:', len(verification['missing']))
""")
markdown("""## A, B, and C

Each pattern is a list of seven actions. Internally the actions are 0, 1, 2, and 3. A family contains three bases, called A, B, and C. The family is regenerated independently for each simulation and distance.

**Hamming distance** counts positions where two lists differ. At k changed positions, B and C each use an action different from A and from each other. Thus A–B, A–C, and B–C all have the same distance k. At all remaining positions the actions agree.
""")
code(r"""run_folder = root / 'outputs' / 'final' / 'runs' / 'run_000_k2'
manifest = json.loads((run_folder / 'manifest.json').read_text())
for name in ['A', 'B', 'C']:
    print(name, manifest['main_sequences'][name]['actions_0_based'])
print('Changed positions (Python starts at 0):', manifest['changed_positions_0_based'])

def hamming(first, second):
    return sum(a != b for a, b in zip(first, second))

A = manifest['main_sequences']['A']['actions_0_based']
B = manifest['main_sequences']['B']['actions_0_based']
C = manifest['main_sequences']['C']['actions_0_based']
print('Pairwise distances:', hamming(A, B), hamming(A, C), hamming(B, C))
""")
markdown("""## Input noise and unchanged targets

The input cue uses **base + 1**, so values are 1–4 before noise. We add Gaussian input noise with standard deviation 0.1. The correct target remains the original action class 0–3; noise never changes the answer.

Development and final evaluation use newly generated noise, with separate deterministic seeds. They contain the same three bases as training but fresh cues. Generalization uses different, unseen bases. The controlled generator also prevents exact overlap with pretraining and D.
""")
code(r"""with np.load(run_folder / 'data.npz') as saved:
    example_cue = saved['train_x'][0, 0].copy()
    example_target = saved['train_y'][0].copy()
    print('Training cue shape:', saved['train_x'].shape)
    print('Final noisy cue shape:', saved['final_noisy_x'].shape)
print('Clean cue:', example_target + 1)
print('Noisy cue:', np.round(example_cue, 3))
print('Target:', example_target)
""")
markdown("""## Model shapes and the complete cue

The cue tensor is `(batch, 1, 7)`: 20 examples in a training batch, one initial time step, and seven features. The **entire intended action pattern is supplied at the beginning**. This task asks the model to decode and unfold a complete cue, not to guess a missing future action.

The RNN has seven hidden units and tanh activation. A linear layer gives four raw logits for the first action. Another linear layer maps these logits back to seven input features, which the recurrent layer uses for the next action. Hidden state resets to zero for each example. Seven action outputs make `(batch, 7, 4)`. Argmax chooses the predicted action. The source also computes one final feedback transition whose result is unused; this timing is preserved.

Cross-entropy measures how much probability the model assigns to the correct targets. **An optimizer update** computes a loss on 20 examples and adjusts weights with SGD. With 15,000 examples, there are 750 main updates. There is one pass through the data, not 750 full passes.
""")
markdown("""## Practice order and pairing

Blocked training presents all 5000 A examples, then all 5000 B examples, then all 5000 C examples, or another prespecified permutation of the three blocks. Interleaved training shuffles all 15,000 example IDs. Both schedules use exactly the same noisy cues and targets.

For each simulation, one pretrained network is cloned into both schedules, with identical weight hashes and fresh identical SGD optimizers. This **paired design** helps isolate order. In the final study each of the six block orders appears five times at each distance. Thirty independent simulation IDs are used at each k. Smoke and pilot IDs come from different seed namespaces and are never pooled into final inference.
""")
code(r"""with np.load(run_folder / 'orders.npz') as orders:
    blocked = orders['blocked']
    interleaved = orders['interleaved']
    assert np.array_equal(np.sort(blocked), np.sort(interleaved))
    print('First blocked example IDs:', blocked[:12])
    print('First interleaved example IDs:', interleaved[:12])
pair = json.loads((run_folder / 'pair_checks.json').read_text())
print('Initial hashes match:', len(set(pair['initial_state_hashes'].values())) == 1)
print('Updates per schedule:', pair['updates_each_schedule'])
""")
markdown("""## Accuracy and learning curves

Action accuracy is the fraction of the seven action predictions that are correct. Complete-sequence accuracy requires all seven to be correct. A model can get most shared positions right yet fail the differing action, so these metrics answer different questions.

The primary outcome averages **fresh-noisy final action accuracy equally over A/B/C**. Clean accuracy is reported separately. Development probes track all three bases at updates 0,25,…,750 and never modify weights. The speed threshold requires at least 90% noisy mean accuracy for three consecutive checkpoints; otherwise it is `not_reached`.

![Fixed development probes](../outputs/final/figures/learning_curves.png)
""")
code(r"""runs = pd.read_csv(analysis / 'run_level.csv')
summary = pd.read_csv(analysis / 'primary_summary.csv')
print(summary[['distance', 'paired_runs', 'blocked_noisy_accuracy_pct',
               'IP_noisy_accuracy_pct', 'mean_difference_pp']].round(2).to_string(index=False))
print('\nThreshold reached counts:')
print(pd.read_csv(analysis / 'threshold_reached_counts.csv').to_string(index=False))
""")
markdown("""## Paired differences and bootstrap confidence intervals

For each run and k, subtract blocked accuracy from interleaved accuracy. Multiplying by 100 expresses the result in **percentage points**. Positive means IP was higher in that run.

The independent unit is the simulation run. A noisy copy or a position shares the same model and training history, so it cannot be treated as another independent subject.

To estimate uncertainty we resample 30 simulation IDs with replacement 10,000 times, then average their paired differences. The 2.5th and 97.5th percentiles form a percentile 95% confidence interval. The slope analysis resamples whole IDs with **all four distances together**. This uncertainty concerns the artificial model/generator; it does not cover biological uncertainty.
""")
code(r"""paired = pd.read_csv(analysis / 'paired_differences.csv')
print(paired.head(8).round(3).to_string(index=False))
with np.load(analysis / 'bootstrap.npz') as bootstrap:
    print('Saved bootstrap resample shape:', bootstrap['indices'].shape)
    print('Example resampled indices:', bootstrap['indices'][0, :10])
print('\nPer-distance bootstrap 95% intervals:')
print(summary[['distance', 'mean_difference_pp', 'median_difference_pp',
               'ci_low_pp', 'ci_high_pp']].round(2).to_string(index=False))
trend = json.loads((analysis / 'trend.json').read_text())
print('Slope, percentage points per Hamming unit:', round(trend['slope_pp_per_hamming'], 3))
print('Slope 95% CI:', round(trend['ci_low'], 3), round(trend['ci_high'], 3))
""")
markdown("""![All categorical paired effects](../outputs/final/figures/paired_differences.png)

Inspect all four values rather than relying only on a straight-line slope. A slope can conceal a nonlinear pattern. No p-values are required for this project.
""")
markdown("""## Learning D and forgetting

After main practice, both trained models are cloned and given the same 2000 examples of one unused D pattern, in the same order (100 updates). The old A/B/C probes are evaluated before D and after every update.

**Forgetting = old final accuracy before D − old final accuracy after D.** The identical held-out final set is used twice. Positive means an accuracy drop; negative means improvement. D distance is recorded but not manipulated, making this a secondary experiment. A model that starts near a floor has less accuracy available to lose, so raw forgetting differences need baseline context.

![A/B/C during D](../outputs/final/figures/D_forgetting.png)
""")
code(r"""print('D distances to A/B/C for this example:', manifest['D_distances_to_ABC'])
secondary = pd.read_csv(analysis / 'secondary_summary.csv')
print(secondary[['distance', 'schedule', 'forgetting_clean', 'forgetting_noisy']]
      .round(3).to_string(index=False))
""")
markdown("""## Shared positions, unseen bases, and limits

Shared and differing positions are scored separately. For k=7 there are **zero shared positions**: shared accuracy is NaN, and count is zero. It is not a zero-percent accuracy result.

Generalization here means decoding **100 previously unseen complete cues**, not acquiring a new physical skill. Clean and noisy generalization are separate. Action frequencies and repeated-action patterns are recorded but not matched across k. The task uses ordinal cue values 1–4, only seven hidden units, a fixed training budget, and no offline consolidation process.

The final conclusions and limitations are in `../reports/results.md` and `../reports/plain_english.md`. Use `../reports/professor_questions.md` to prepare a discussion about what the model can and cannot establish.
""")
code(r"""positions = pd.read_csv(analysis / 'position_summary.csv')
print(positions[(positions['distance'] == 7) & (positions['group'] == 'shared')]
      [['schedule', 'split', 'action_accuracy', 'action_count']].to_string(index=False))
print('\nClean and noisy unseen-base accuracies:')
print(secondary[['distance', 'schedule', 'generalization_clean_accuracy',
                 'generalization_noisy_accuracy']].round(3).to_string(index=False))
""")

namespace = {"__name__": "__notebook__"}
execution_count = 0
for cell in cells:
    cell["id"] = hashlib.blake2b("".join(cell["source"]).encode(), digest_size=6).hexdigest()
    if cell["cell_type"] != "code":
        continue
    execution_count += 1
    source = "".join(cell["source"])
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(ast.parse(source), "walkthrough-cell-{}".format(execution_count), "exec"), namespace)
    cell["execution_count"] = execution_count
    if stream.getvalue():
        cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": stream.getvalue().splitlines(True)}]
notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.8.10"},
                         "execution_provenance": "Each code cell executed sequentially against saved final outputs; no training code in notebook."}}
directory = Path("notebooks")
directory.mkdir(exist_ok=True)
(directory / "walkthrough.ipynb").write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
print("Executed", execution_count, "notebook code cells and wrote", directory / "walkthrough.ipynb")
