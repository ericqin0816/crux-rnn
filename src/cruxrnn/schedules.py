"""Practice orders permute the identical master examples without replacement."""

import itertools

import numpy as np

ORDERS = tuple("".join(order) for order in itertools.permutations("ABC"))
SCHEDULES = ("blocked", "large_blocks", "small_blocks", "cyclic", "interleaved")


def blocked_order(run_id: int, distance: int) -> str:
    """Near balance for any n; rotate remainder orders over distance conditions."""
    return ORDERS[(run_id + distance - 1) % 6]


def make_order(samples: int, schedule: str, order: str, seed: int) -> np.ndarray:
    """Return all 3*samples IDs once; cyclic means deterministic trial cycling.

    Large/small chunks contain 1000/100 examples (last chunks may be shorter).
    Interleaved means uniformly shuffled mixed practice, matching v1 IP.
    """
    if type(samples) is not int or samples < 1 or order not in ORDERS:
        raise ValueError("Expected positive samples and a permutation of ABC")
    if schedule not in SCHEDULES:
        raise ValueError(f"Unknown practice schedule: {schedule}")
    if schedule == "interleaved":
        ids = np.random.RandomState(seed).permutation(3 * samples)
    else:
        chunk = {"blocked": samples, "large_blocks": 1000, "small_blocks": 100, "cyclic": 1}[
            schedule
        ]
        chunks = []
        for start in range(0, samples, chunk):
            for letter in order:
                offset = "ABC".index(letter) * samples
                chunks.append(np.arange(offset + start, offset + min(start + chunk, samples)))
        ids = np.concatenate(chunks)
    if not np.array_equal(np.sort(ids), np.arange(3 * samples)):
        raise ValueError("Schedule must contain every master example exactly once")
    return ids
