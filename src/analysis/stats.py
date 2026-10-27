"""Small statistics helpers shared by the audits and the analysis."""

import math


def wilson_interval(k, n, z=1.96):
    """Wilson score interval for a binomial proportion k/n (95% by default).

    Unlike the normal approximation it stays inside [0, 1] and behaves at k = 0,
    which matters for judges such as xVerify that almost never accept an attack.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def holm(pvalues):
    """Holm step-down adjusted p-values, in the order given."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def rewarded_advantage(q, group_size):
    """Advantage of a rewarded response (reward 1) in a group of binary rewards with acceptance rate q.

    GRPO with scale_rewards: group (configs/grpo.yaml) uses (r - mean) / std over the group, and TRL's std is the
    sample standard deviation (Bessel's correction, trl.trainer.utils.nanstd). For a group of G responses with a
    share q rewarded this is sqrt((G - 1) / G * (1 - q) / q). Evaluated at a mean acceptance rate it is a plug-in
    value, not the mean advantage over groups, which the logs cannot give (they keep only flagged responses).
    """
    return math.sqrt((group_size - 1) / group_size * (1 - q) / q)


def binomial_two_sided(k, n):
    """Exact two-sided binomial test of k successes in n trials against one half (sign test)."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)
