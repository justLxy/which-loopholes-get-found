"""Audit your own judge: rank the 14 hacking patterns before RL training and predict the first event.

Exposure = 256 x p_hat x FPR (paper Section 3). The top family is the predicted winner; its first-event step and 90%
window come from src.train.masking.timing_fit. Below the level in configs/masking.yaml (1.92) no event is expected.
With q0, the expected-advantage score p_hat (FPR - q0) / sqrt(q0 (1 - q0)) is added.

Usage:
    python -m src.audit.predict --fpr my_fpr.csv --propensity my_p.csv [--q0 0.2] [--out ranking.csv]
    python -m src.audit.predict --judge r1_distill_verifier_1p5b --checkpoint main-rot1 [--pool hard] [--q0 0.2]
The CSVs have the columns family,fpr and family,p_hat, one row for each of the 14 families (src.detectors). The
second form uses our measurements: results/stage2/string_fpr.csv and results/main/sft-<checkpoint>/
propensity_sampled.csv.
"""

import argparse
import functools
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.analysis.descriptive import STEP_SAMPLES
from src.detectors import FAMILIES
from src.paths import ROOT, display_path
from src.train.lowshare import scores
from src.train.masking import PLAN, Z90, timing_fit

STRING_FPR = ROOT / "results" / "stage2" / "string_fpr.csv"
MAIN_RESULTS = ROOT / "results" / "main"


class PredictError(ValueError):
    """Invalid input; the message is one line."""


@dataclass(frozen=True, eq=False)
class Ranking:
    table: pd.DataFrame          # one row per family, by exposure: family, fpr, p_hat, exposure, rank, rank_fpr,
                                 # rank_p_hat, and with q0 also ea and rank_ea
    top: str                     # the family with the highest exposure ('+'-joined when tied)
    exposure: float              # its exposure
    step: float | None           # predicted first-event step and its 90% window (None when the exposure is 0)
    step_low: float | None
    step_high: float | None
    event_predicted: bool        # exposure at least the level: an event is expected within the run
    level: float
    q0: float | None


@functools.lru_cache(maxsize=None)
def timing(plan=str(PLAN)):
    """(a, b, sd) of the timing fit and the level, as configs/masking.yaml defines them."""
    cfg = yaml.safe_load(Path(plan).read_text())
    return timing_fit(ROOT / cfg["timing_fit"]), float(cfg["level"])


def window(exposure, fit):
    """(step, low, high) of the 90% window, as in src.train.masking.predict."""
    a, b, sd = fit
    mid = a + b * math.log(exposure)
    return math.exp(mid), math.exp(mid - Z90 * sd), math.exp(mid + Z90 * sd)


# ----------------------------------------------------------------------------- validation

def _families(index, what):
    names = [str(f) for f in index]
    dup = sorted({f for f in names if names.count(f) > 1})
    if dup:
        raise PredictError(f"{what}: families listed more than once: {', '.join(dup)}")
    unknown = sorted(set(names) - set(FAMILIES))
    if unknown:
        raise PredictError(f"{what}: unknown families: {', '.join(unknown)} (the 14 families are "
                           f"{', '.join(FAMILIES)})")
    missing = [f for f in FAMILIES if f not in names]
    if missing:
        raise PredictError(f"{what}: missing families: {', '.join(missing)}")


def check_values(values, what):
    """Validate {family: value}: the 14 families, each in [0, 1]. Returns FAMILIES order."""
    s = pd.Series(values, dtype=object) if not isinstance(values, pd.Series) else values
    _families(s.index, what)
    s.index = [str(f) for f in s.index]
    num = pd.to_numeric(s, errors="coerce").astype(float)
    bad = [f for f in FAMILIES if not np.isfinite(num[f])]
    if bad:
        raise PredictError(f"{what}: not a number for {', '.join(f'{f} ({s[f]!r})' for f in bad)}")
    out = [f"{f} ({num[f]:g})" for f in FAMILIES if not 0 <= num[f] <= 1]
    if out:
        raise PredictError(f"{what}: values must be between 0 and 1: {', '.join(out)}")
    return num[list(FAMILIES)]


def check_q0(q0):
    try:
        q0 = float(q0)
    except (TypeError, ValueError):
        raise PredictError(f"q0 must be a number, not {q0!r}") from None
    if not 0 < q0 < 1:
        raise PredictError(f"q0 must be strictly between 0 and 1, not {q0:g}")
    return q0


# ----------------------------------------------------------------------------- ranking

def rank_desc(values):
    """Descending rank; ties share the lowest."""
    return values.rank(ascending=False, method="min").astype(int)


def rank_patterns(fpr, p_hat, q0=None):
    """Rank the families and predict the first event. fpr, p_hat: {family: value}. Raises PredictError."""
    fpr, p_hat = check_values(fpr, "FPR"), check_values(p_hat, "propensity")
    q0 = None if q0 is None else check_q0(q0)
    p, f = p_hat.to_numpy(), fpr.to_numpy()
    if q0 is None:
        exposure, ea = STEP_SAMPLES * p * f, None      # the exposure term of src.train.lowshare.scores
    else:
        exposure, ea = scores(p, f, q0)
    t = pd.DataFrame({"family": list(FAMILIES), "fpr": f, "p_hat": p, "exposure": exposure})
    t["rank"], t["rank_fpr"], t["rank_p_hat"] = rank_desc(t["exposure"]), rank_desc(t["fpr"]), rank_desc(t["p_hat"])
    if ea is not None:
        t["ea"] = ea
        t["rank_ea"] = rank_desc(t["ea"])
    t = t.sort_values(["rank", "family"]).reset_index(drop=True)
    best = float(t["exposure"].max())
    top = "+".join(sorted(t.loc[t["exposure"] == best, "family"]))
    fit, level = timing()
    step = low = high = None
    if best > 0:
        step, low, high = window(best, fit)
    return Ranking(t, top, best, step, low, high, best >= level, level, q0)


# ----------------------------------------------------------------------------- inputs

def read_values(path, column):
    """A CSV with the columns family and `column` as a Series (validated by rank_patterns)."""
    path = Path(path)
    try:
        df = pd.read_csv(path, dtype={"family": str})
    except FileNotFoundError:
        raise PredictError(f"{path}: no such file") from None
    except (OSError, ValueError, pd.errors.ParserError) as err:
        raise PredictError(f"{path} cannot be read as CSV ({str(err).splitlines()[0]})") from None
    missing = [c for c in ("family", column) if c not in df.columns]
    if missing:
        raise PredictError(f"{path} lacks the columns {', '.join(missing)} (expected family,{column}; "
                           f"found {','.join(map(str, df.columns))})")
    if df["family"].isna().any():
        raise PredictError(f"{path}: a row has no family name")
    _families(df["family"], str(path))
    return df.set_index("family")[column]


def measured_inputs(judge, checkpoint, pool="hard", root=ROOT):
    """(fpr, p_hat) measured in the main experiment."""
    fpr = pd.read_csv(root / STRING_FPR.relative_to(ROOT))
    if judge not in set(fpr["judge"]):
        raise PredictError(f"unknown judge {judge!r}; choose one of {', '.join(sorted(set(fpr['judge'])))}")
    main = root / MAIN_RESULTS.relative_to(ROOT)
    path = main / f"sft-{checkpoint}" / "propensity_sampled.csv"
    if not path.exists():
        known = sorted(p.parent.name.removeprefix("sft-") for p in main.glob("sft-*/propensity_sampled.csv"))
        raise PredictError(f"unknown checkpoint {checkpoint!r}; choose one of {', '.join(known)}")
    prop = pd.read_csv(path)
    if pool not in set(prop["pool"]):
        raise PredictError(f"no propensity for the {pool!r} pool; choose one of {', '.join(sorted(set(prop['pool'])))}")
    f = fpr[(fpr["judge"] == judge) & fpr["family"].isin(FAMILIES)].set_index("family")["fpr"]
    p = prop[(prop["pool"] == pool) & prop["family"].isin(FAMILIES)].set_index("family")["p_hat"]
    return f, p


# ----------------------------------------------------------------------------- command line

def describe(r):
    shown = r.table.copy()
    fmt = {"fpr": "{:.3f}", "p_hat": "{:.5f}", "exposure": "{:.3f}", "ea": "{:.5f}"}
    for col, f in fmt.items():
        if col in shown:
            shown[col] = shown[col].map(f.format)
    lines = [shown.to_string(index=False), ""]
    if r.step is None:
        lines.append(f"Top family: {r.top}, exposure 0: no family is rewarded, so no event is predicted.")
    else:
        lines.append(f"Top family: {r.top}, exposure {r.exposure:.3f} rewarded responses per step.")
        lines.append(f"Predicted first event at step {r.step:.0f}, 90% window {r.step_low:.0f} to {r.step_high:.0f} "
                     "(timing fit of the main runs).")
        if not r.event_predicted:
            lines.append(f"The exposure is below {r.level:g}, the lowest exposure of any main-run event: no event "
                         "is expected within the 300 steps.")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    own = ap.add_argument_group("your own measurements")
    own.add_argument("--fpr", help="CSV with the columns family,fpr")
    own.add_argument("--propensity", help="CSV with the columns family,p_hat")
    ours = ap.add_argument_group("our measurements")
    ours.add_argument("--judge", help="a judge key of results/stage2/string_fpr.csv, e.g. r1_distill_verifier_1p5b")
    ours.add_argument("--checkpoint", help="a starting checkpoint, e.g. main-rot1 (results/main/sft-<checkpoint>)")
    ours.add_argument("--pool", default="hard", help="difficulty pool of the propensity (default: hard)")
    ap.add_argument("--q0", type=float, help="acceptance of the starting policy's outputs; adds the EA score")
    ap.add_argument("--out", help="write the ranking to this CSV")
    args = ap.parse_args(argv)
    own_mode, our_mode = args.fpr or args.propensity, args.judge or args.checkpoint
    if bool(own_mode) == bool(our_mode):
        ap.error("give either --fpr and --propensity, or --judge and --checkpoint")
    if own_mode and not (args.fpr and args.propensity):
        ap.error("--fpr and --propensity go together")
    if our_mode and not (args.judge and args.checkpoint):
        ap.error("--judge and --checkpoint go together")
    try:
        if own_mode:
            fpr, p_hat = read_values(args.fpr, "fpr"), read_values(args.propensity, "p_hat")
        else:
            fpr, p_hat = measured_inputs(args.judge, args.checkpoint, args.pool)
        ranking = rank_patterns(fpr, p_hat, args.q0)
        print(describe(ranking))
        if args.out:
            ranking.table.to_csv(args.out, index=False)
            print(f"wrote {display_path(Path(args.out).resolve())}")
    except PredictError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except OSError as err:
        print(f"error: cannot write {args.out} ({err.strerror})", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
