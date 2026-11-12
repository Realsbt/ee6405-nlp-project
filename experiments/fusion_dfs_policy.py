"""Validation-only continuation gates; thresholds are practical rules, not tests."""

import math
import statistics


PILOT_SEED = 3407
SEEDS = (3407, 42, 2026)
MIN_GAIN = 0.002
MAX_SEED_REGRESSION = 0.002


def continuation_gate(candidate, reference, pilot=False):
    expected = (PILOT_SEED,) if pilot else SEEDS
    if set(candidate) != set(expected) or set(reference) != set(expected):
        raise ValueError("Continuation gate requires matched predeclared seeds")
    scores = list(candidate.values()) + list(reference.values())
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores):
        raise ValueError("Invalid validation Macro-F1")
    differences = [candidate[seed] - reference[seed] for seed in expected]
    mean = statistics.mean(differences)
    positive = sum(value > 0 for value in differences)
    passed = mean + 1e-12 >= MIN_GAIN and (pilot or (
        positive >= 2 and min(differences) + 1e-12 >= -MAX_SEED_REGRESSION))
    return {
        "decision": "continue" if passed else "prune",
        "phase": "single_seed_screen" if pilot else "three_seed_confirmation",
        "validation_differences_pp": dict(zip(map(str, expected), [100 * d for d in differences])),
        "mean_validation_gain_pp": 100 * mean,
        "positive_seeds": positive, "minimum_mean_gain_pp": 100 * MIN_GAIN,
        "maximum_allowed_seed_regression_pp": 100 * MAX_SEED_REGRESSION,
        "test_used_for_decision": False,
        "interpretation": "Practical compute-allocation heuristic, not statistical significance",
    }


def extension_tree(parent, explore, prefer=None):
    reference = "ext_edu_ce2"
    incumbent = reference
    if explore("ext_edu_consistent", reference):
        incumbent = "ext_edu_consistent"
    if parent != "fusion_cls":
        if incumbent == "ext_edu_consistent":
            if explore("ext_chunks_consistent", incumbent):
                incumbent = "ext_chunks_consistent"
            elif explore("ext_chunks_ce2", reference):
                if prefer is not None and prefer("ext_chunks_ce2", incumbent):
                    incumbent = "ext_chunks_ce2"
        elif explore("ext_chunks_ce2", reference):
            incumbent = "ext_chunks_ce2"
    return incumbent
