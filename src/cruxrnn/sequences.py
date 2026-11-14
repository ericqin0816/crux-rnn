"""Controlled families at all seven distances, with fresh held-out noise."""

import itertools

import numpy as np

from motor_rnn.data import assert_no_overlap, copies, hamming, unique_bases

from .config import ExperimentConfig


def family(rng: np.random.RandomState, distance: int) -> tuple:
    """Construct A/B/C with all three pairwise Hamming distances exactly k."""
    if type(distance) is not int or not 1 <= distance <= 7:
        raise ValueError("Hamming distance must be an integer from 1 through 7")
    first = rng.randint(0, 4, size=7)
    bases = np.tile(first, (3, 1))
    changed = np.sort(rng.choice(7, size=distance, replace=False))
    for position in changed:
        alternatives = np.array([action for action in range(4) if action != first[position]])
        bases[1:, position] = rng.choice(alternatives, size=2, replace=False)
    if any(hamming(bases[i], bases[j]) != distance for i, j in itertools.combinations(range(3), 2)):
        raise RuntimeError("Failed exact pairwise Hamming constraint")
    return bases, changed


def generate(config: ExperimentConfig, distance: int, seeds: dict) -> dict:
    """Generate once per paired cell; only schedule order may change afterward."""

    def rng(name):
        return np.random.RandomState(seeds[name])

    main, changed = family(rng("sequence_family_generation"), distance)
    pre = unique_bases(rng("pretraining_sequences"), 10, main)
    unseen = unique_bases(rng("generalization_generation"), 100, np.concatenate([main, pre]))
    interference = unique_bases(rng("D_selection"), 1, np.concatenate([main, pre, unseen]))
    assert_no_overlap({"main": main, "pre": pre, "unseen": unseen, "D": interference})
    data = {
        "main_bases": main,
        "pre_bases": pre,
        "gen_bases": unseen,
        "D_base": interference,
        "changed_positions": changed,
    }
    specifications = [
        ("pre", pre, config.pre_per_sequence, "pretraining_noise", config.noise_sd),
        ("train", main, config.train_per_sequence, "training_noise", config.noise_sd),
        ("gen_noisy", unseen, config.eval_per_sequence, "generalization_noise", config.noise_sd),
        ("gen_clean", unseen, 1, "generalization_noise", 0),
        ("D", interference, config.d_examples, "D_noise", config.noise_sd),
    ]
    for split, stream in (("dev", "development_noise"), ("final", "final_noise")):
        specifications += [
            (split + "_noisy", main, config.eval_per_sequence, stream, config.noise_sd),
            (split + "_clean", main, 1, stream, 0),
        ]
    for name, bases, count, stream, sd in specifications:
        data[name + "_x"], data[name + "_y"] = copies(bases, count, rng(stream), sd)
    return data
