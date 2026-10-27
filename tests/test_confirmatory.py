"""Confirmatory tests: Holm, the sign test, the exact dose randomization test and the H2 pairing."""

import itertools

import pandas as pd
import pytest
import yaml

from src.analysis.confirmatory import dose_null, h2_test, h3_test, tier_rank, two_sided
from src.analysis.stats import binomial_two_sided, holm, rewarded_advantage
from src.paths import ROOT
from src.train import queue

UNITS = ROOT / "results" / "analysis" / "units.csv"
needs_units = pytest.mark.skipif(not UNITS.exists(), reason="run src.analysis.primary first")


def test_rewarded_advantage_matches_group_normalization_with_the_sample_std():
    import numpy as np
    for k in range(1, 8):                                       # k of 8 responses rewarded
        r = np.array([1.0] * k + [0.0] * (8 - k))
        direct = (1 - r.mean()) / r.std(ddof=1)                 # TRL: (r - mean) / sample std
        assert rewarded_advantage(k / 8, 8) == pytest.approx(direct)


def test_holm_and_the_sign_test():
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert binomial_two_sided(7, 7) == pytest.approx(2 / 2 ** 7)
    assert binomial_two_sided(0, 0) == 1.0 and binomial_two_sided(3, 6) == 1.0


def test_dose_null_matches_brute_force_enumeration():
    fams = list("abcdef")
    sizes = [2, 2, 2]
    events = [("a", 0), ("b", 1), ("a", 2), ("c", 0)]
    brute = {}
    for order in itertools.permutations(fams):            # every labelled split, each counted equally often
        group = {f: next(g for g in range(3) if f in order[2 * g:2 * g + 2]) for f in fams}
        t = sum(tier_rank(group[f], r) for f, r in events)
        brute[t] = brute.get(t, 0) + 1
    n = sum(brute.values())
    assert dose_null(events, sizes, len(fams)) == pytest.approx({t: c / n for t, c in sorted(brute.items())})


def test_single_event_rank_follows_group_sizes():
    dist = dose_null([("x", 0)], [5, 5, 4], 14)
    assert dist == pytest.approx({1: 5 / 14, 2: 5 / 14, 3: 4 / 14})
    p, mean = two_sided(dist, 3)
    assert mean == pytest.approx((5 + 10 + 12) / 14)
    assert p == pytest.approx(4 / 14)                   # rank 1 is 0.93 from the mean, closer than rank 3 (1.07)


def test_h2_pairs_hard_and_easy_runs_of_the_same_judge_and_seed():
    rows = []
    for judge, seed, hard, easy in (("a", 0, True, False), ("a", 1, True, False), ("b", 0, False, False)):
        for pool, ev in (("hard", hard), ("easy", easy)):
            rows.append({"run": f"{judge}-{pool}-{seed}", "group": "core", "judge": judge, "pool": pool,
                         "seed": seed, "family": "f", "event": ev})
    h2 = h2_test(pd.DataFrame(rows))
    assert (h2["pairs"], h2["hard_only"], h2["easy_only"]) == (3, 2, 0) and h2["p"] == pytest.approx(0.5)


@needs_units
def test_h3_on_the_main_runs_uses_the_seeded_tiers():
    units = pd.read_csv(UNITS)
    h3 = h3_test(units, yaml.safe_load(queue.PROTOCOL.read_text())["seeding"])
    assert h3["events"] == int(units.loc[units["group"] == "core", "event"].sum())
    assert 0 < h3["p"] <= 1 and h3["p_one_sided"] <= h3["p"]
