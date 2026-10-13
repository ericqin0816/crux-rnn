"""Generate immutable master data; schedules index exactly the same examples."""
import itertools
import numpy as np
from .provenance import canonical_hash, file_hash, write_json

ORDERS = ["".join(p) for p in itertools.permutations("ABC")]


def hamming(a, b):
    return int(np.count_nonzero(np.asarray(a) != np.asarray(b)))


def unique_bases(rng, count, forbidden=()):
    used = {tuple(x) for x in forbidden}
    chosen = []
    if count + len(used) > 4 ** 7:
        raise ValueError("Not enough unused sequences")
    while len(chosen) < count:
        candidate = tuple(rng.randint(0, 4, size=7).tolist())
        if candidate not in used:
            chosen.append(candidate)
            used.add(candidate)
    return np.asarray(chosen, dtype=np.int64)


def family(rng, distance):
    if distance not in (1, 2, 4, 7):
        raise ValueError("Distance must be one of 1, 2, 4, 7")
    a = rng.randint(0, 4, size=7)
    bases = np.tile(a, (3, 1))
    positions = np.sort(rng.choice(7, size=distance, replace=False))
    for pos in positions:
        alternatives = np.asarray([v for v in range(4) if v != a[pos]])
        bc = rng.choice(alternatives, size=2, replace=False)
        bases[1:, pos] = bc
    assert all(hamming(bases[i], bases[j]) == distance for i, j in itertools.combinations(range(3), 2))
    return bases, positions


def assert_no_overlap(groups):
    seen = set()
    for name, bases in groups.items():
        current = {tuple(row) for row in bases}
        assert len(current) == len(bases), "Duplicate within " + name
        assert not (current & seen), "Forbidden overlap in " + name
        seen.update(current)


def sequence_description(base):
    longest = current = 1
    for i in range(1, 7):
        current = current + 1 if base[i] == base[i - 1] else 1
        longest = max(longest, current)
    return {"actions_0_based": base.tolist(), "cue_1_based": (base + 1).tolist(),
            "action_frequencies": np.bincount(base, minlength=4).tolist(),
            "adjacent_repeated_actions": int(np.sum(base[1:] == base[:-1])),
            "longest_repeated_run": longest}


def copies(bases, count, rng, sd):
    targets = np.repeat(bases, count, axis=0).astype(np.int64)
    cue = (targets[:, None, :] + 1).astype(np.float64)
    if sd:
        cue += rng.normal(0, sd, cue.shape)
    return cue.astype(np.float32), targets


def schedule_orders(samples_per_sequence, blocked_order, permutation_seed):
    n = samples_per_sequence
    blocked = np.concatenate([np.arange("ABC".index(letter) * n, ("ABC".index(letter) + 1) * n) for letter in blocked_order])
    interleaved = np.random.RandomState(permutation_seed).permutation(3 * n)
    assert np.array_equal(np.sort(blocked), np.sort(interleaved))
    for order in (blocked, interleaved):
        assert np.array_equal(np.bincount(order // n, minlength=3), [n, n, n])
    return {"blocked": blocked, "interleaved": interleaved}


def balanced_assignment(run_ids, distances):
    assert len(run_ids) % 6 == 0, "Final count must be a multiple of six"
    result = {str(k): {str(r): ORDERS[i % 6] for i, r in enumerate(run_ids)} for k in distances}
    for orders in result.values():
        assert all(list(orders.values()).count(order) == len(run_ids) // 6 for order in ORDERS)
    return result


def generate(config, seeds):
    rng = lambda name: np.random.RandomState(seeds[name])
    controlled = config["mode"] == "controlled_similarity"
    if controlled:
        main, changed = family(rng("sequence_family_generation"), config["distance"])
        pre = unique_bases(rng("pretraining_sequences"), 10, main)
        gen = unique_bases(rng("generalization_generation"), 100, np.concatenate([main, pre]))
        d = unique_bases(rng("D_selection"), 1, np.concatenate([main, pre, gen]))
        assert_no_overlap({"main": main, "pre": pre, "generalization": gen, "D": d})
    else:
        # Preserve lack of uniqueness/novelty enforcement in source_exact.
        main = rng("sequence_family_generation").randint(4, size=(3, 7))
        pre = rng("pretraining_sequences").randint(4, size=(10, 7))
        gen = rng("generalization_generation").randint(4, size=(100, 7))
        d = rng("D_selection").randint(4, size=(1, 7))
        changed = np.flatnonzero(np.any(main != main[0], axis=0))
    arrays = {"main_bases": main, "pre_bases": pre, "gen_bases": gen, "D_base": d,
              "changed_positions": changed, "train_ids": np.arange(3 * config["train_per_sequence"]),
              "train_family": np.repeat(np.arange(3), config["train_per_sequence"])}
    def add(name, bases, n, stream, sd):
        arrays[name + "_x"], arrays[name + "_y"] = copies(bases, n, rng(stream), sd)
    add("pre", pre, config["pre_per_sequence"], "pretraining_noise", config["noise_sd"])
    add("train", main, config["train_per_sequence"], "training_noise", config["noise_sd"])
    add("gen_noisy", gen, config["eval_per_sequence"], "generalization_noise", config["noise_sd"])
    add("gen_clean", gen, 1, "generalization_noise", 0)
    add("D", d, config["D_examples"], "D_noise", config["noise_sd"])
    if controlled:
        for split, stream in (("dev", "development_noise"), ("final", "final_noise")):
            add(split + "_noisy", main, config["eval_per_sequence"], stream, config["noise_sd"])
            add(split + "_clean", main, 1, stream, 0)
    return arrays


def save_data(directory, config, seeds):
    directory.mkdir(parents=True, exist_ok=True)
    data = generate(config, seeds)
    np.savez_compressed(directory / "data.npz", **data)
    main, pre, gen, d = (data[name] for name in ("main_bases", "pre_bases", "gen_bases", "D_base"))
    counts = {"main_unique": len({tuple(x) for x in main}), "pre_unique": len({tuple(x) for x in pre}),
              "generalization_unique": len({tuple(x) for x in gen})}
    manifest = {"config_hash": canonical_hash(config), "seeds": seeds,
                "data_sha256": file_hash(directory / "data.npz"), "source_exact_overlap_counts": counts,
                "changed_positions_0_based": data["changed_positions"].tolist(),
                "main_sequences": {letter: sequence_description(main[i]) for i, letter in enumerate("ABC")},
                "pretraining_bases": pre.tolist(), "generalization_bases": gen.tolist(),
                "D": sequence_description(d[0]), "D_distances_to_ABC": [hamming(d[0], x) for x in main],
                "pairwise_main_distances": [hamming(main[i], main[j]) for i, j in itertools.combinations(range(3), 2)]}
    write_json(directory / "manifest.json", manifest)
    return data, manifest
