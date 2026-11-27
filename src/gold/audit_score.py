"""Score the gold-checker hand audit against the labels the team filled in.

Usage:
    python -m src.gold.audit_score LABELED.csv [MORE.csv ...] [--key audits/gold/key.csv] [--out result.json]

Each labeled CSV has the columns of audits/gold/gold_audit_sheet.csv with "label (Y/N/?)"
and "labeler" filled; rows may come from one combined sheet or one sheet per labeler.
Reports agreement between gold and the human labels per stratum, an overall agreement
weighted by each stratum's share of all responses (the sample over-represents rare
cases), Cohen's kappa between labelers on shared items, and every disagreement for
the error analysis. The pilot requires overall agreement of at least 95%.

The team's labels are in audits/gold/labels/ (one file per labeler, labels only). The key was
committed after labeling ended (src/gold/rebuild_key.py).
"""

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from src.analysis.stats import wilson_interval
from src.gold.audit_sheet import SHEET_DIR

LABEL_COLUMN = "label (Y/N/?)"
YES, NO = {"y", "yes", "1", "true", "correct"}, {"n", "no", "0", "false", "wrong", "incorrect"}


def parse_label(text):
    t = (text or "").strip().lower()
    return True if t in YES else False if t in NO else None


def read_labels(paths):
    """audit_id -> {labeler: True/False/None}; rows without a label are skipped."""
    labels = defaultdict(dict)
    for path in paths:
        with Path(path).open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not (row.get(LABEL_COLUMN) or "").strip():
                    continue
                labeler = (row.get("labeler") or row.get("assigned_to") or "?").strip() or "?"
                labels[row["audit_id"]][labeler] = parse_label(row[LABEL_COLUMN])
    return labels


def read_key(path):
    with Path(path).open(encoding="utf-8") as f:
        return {row["audit_id"]: {**row, "gold_correct": row["gold_correct"] == "True"} for row in csv.DictReader(f)}


def human_verdict(votes):
    """Majority of decided votes; None when undecided or tied."""
    decided = [v for v in votes.values() if v is not None]
    if not decided or decided.count(True) == decided.count(False):
        return None
    return decided.count(True) > decided.count(False)


def cohen_kappa(pairs):
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if not pairs:
        return None
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def score(labels, key, population):
    per_stratum = defaultdict(lambda: {"agree": 0, "n": 0, "undecided": 0})
    disagreements = []
    for audit_id, votes in labels.items():
        k = key[audit_id]
        verdict = human_verdict(votes)
        s = per_stratum[k["stratum"]]
        if verdict is None:
            s["undecided"] += 1
            continue
        s["n"] += 1
        if verdict == k["gold_correct"]:
            s["agree"] += 1
        else:
            disagreements.append({"audit_id": audit_id, "stratum": k["stratum"], "gold": k["gold_correct"],
                                  "human": verdict, "gold_extracted": k["gold_extracted"], "votes": votes})
    rows = {}
    for name, s in per_stratum.items():
        lo, hi = wilson_interval(s["agree"], s["n"])
        rows[name] = {**s, "agreement": s["agree"] / s["n"] if s["n"] else None, "ci": [round(lo, 3), round(hi, 3)]}
    total = sum(population.get(name, 0) for name in rows if rows[name]["n"])
    weighted = (
        sum(population.get(name, 0) / total * r["agreement"] for name, r in rows.items() if r["n"]) if total else None
    )
    raw_n = sum(r["n"] for r in rows.values())
    raw = sum(r["agree"] for r in rows.values()) / raw_n if raw_n else None

    labelers = sorted({labeler for v in labels.values() for labeler in v})
    kappas = {}
    for a, b in combinations(labelers, 2):
        pairs = [(v[a], v[b]) for v in labels.values() if a in v and b in v]
        if pairs:
            kappas[f"{a}-{b}"] = {"items": len(pairs), "kappa": cohen_kappa(pairs)}
    return {"per_stratum": rows, "weighted_agreement": weighted, "raw_agreement": raw, "items_scored": raw_n,
            "pass": weighted is not None and weighted >= 0.95, "inter_rater": kappas,
            "disagreements": disagreements}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labeled", nargs="+")
    ap.add_argument("--key", default=str(SHEET_DIR / "key.csv"))
    ap.add_argument("--out", help="also write the full result as JSON")
    args = ap.parse_args()
    key = read_key(args.key)
    population = json.loads((Path(args.key).parent / "strata.json").read_text())["population"]
    result = score(read_labels(args.labeled), key, population)
    print(json.dumps({k: v for k, v in result.items() if k != "disagreements"}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, default=str))
    print(f"\n{len(result['disagreements'])} disagreements:")
    for d in result["disagreements"]:
        print(f"  {d['audit_id']} [{d['stratum']}] gold={d['gold']} human={d['human']} "
              f"extracted={d['gold_extracted']!r} votes={d['votes']}")


if __name__ == "__main__":
    main()
