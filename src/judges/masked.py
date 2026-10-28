"""A judge patched by a rule: responses in a masked family are rejected without asking the judge.

Experiment B (configs/masking.yaml) removes one family's reward to see which family wins instead. The family
detectors (src.detectors) decide membership on the full response, as they do for the outcome measures, so a
masked family earns nothing in training and its evaluation outputs are never accepted, while every other
response gets the wrapped judge's own verdict. The same patched judge serves the reward and the evaluations.
"""

import time

from src.detectors.families import FAMILIES, classify
from src.judges.local import JudgeResult


class MaskedJudge:
    def __init__(self, inner, families):
        unknown = sorted(set(families) - set(FAMILIES))
        if unknown:
            raise ValueError(f"unknown families {unknown}; choose from {FAMILIES}")
        if not families:
            raise ValueError("MaskedJudge needs at least one family")
        self.inner, self.families = inner, frozenset(families)
        self.cfg = {**getattr(inner, "cfg", {}), "masked_families": sorted(self.families)}

    def masked(self, response):
        return classify(response or "").family in self.families

    def judge(self, items):
        """Same contract as the wrapped judge: (results in the order of items, seconds)."""
        start = time.time()
        hit = [self.masked(it["response"]) for it in items]
        passed = [it for it, h in zip(items, hit) if not h]
        inner_results, _ = self.inner.judge(passed) if passed else ([], 0.0)
        inner_iter = iter(inner_results)
        results = [JudgeResult(answer="", output="", accepted=False, parse_fail=False, label="masked") if h
                   else next(inner_iter) for h in hit]
        return results, time.time() - start
