"""Make the blind hand-audit sheet for the gold checker (pilot check: agreement >= 95%).

Usage:
    python -m src.gold.audit_sheet --difficulty-run runs/difficulty [--n 200] [--shared 30]

Samples policy responses from a difficulty run, stratified so the cases where a
rule-based checker can go wrong are well represented:
  correct_plain       gold says correct and the boxed answer is literally the reference
  correct_nontrivial  gold says correct through an equivalent form (1,000 / 3-1 / \\frac{12}{2})
  wrong_answered      gold says wrong and a boxed answer was given
  no_answer           no boxed answer, not truncated (a correct unboxed answer would be a miss)
  truncated           hit the length limit
Labelers see the problem, the reference and the full response, never the gold
verdict or the stratum. Items are split among labelers A, B and C, and `--shared`
items go to all three for inter-rater agreement.

Writes (blind, safe to share): audits/gold/gold_audit_sheet.csv and gold_audit.html
(rebuild the page alone with --html-from-csv audits/gold/gold_audit_sheet.csv).
Writes (not for labelers): runs/gold-audit/key.csv and strata.json.
"""

import argparse
import csv
import html
import json
import random
from collections import Counter
from pathlib import Path

import yaml

from src.data.pools import load_candidates
from src.paths import ROOT, display_path
from src.pipeline.shards import read_shards

STRATA = (("correct_plain", 0.35), ("correct_nontrivial", 0.15), ("wrong_answered", 0.30),
          ("no_answer", 0.10), ("truncated", 0.10))
LABELERS = ("A", "B", "C")
SHEET_DIR = ROOT / "audits" / "gold"
KEY_DIR = ROOT / "runs" / "gold-audit"
INSTRUCTIONS = (
    "Label each response: Y if the final answer it commits to equals the reference answer "
    "(any equivalent form counts, boxed or not), N if it is wrong or gives no final answer, "
    "? if you cannot decide. Judge only the final answer, not the reasoning. Add a note for "
    "anything unusual.")


def stratum(r):
    if r["truncated"]:
        return "truncated"
    if r["no_answer"]:
        return "no_answer"
    if not r["gold_correct"]:
        return "wrong_answered"
    plain = (r["gold_extracted"] or "").strip().strip("$").replace(" ", "")
    return "correct_plain" if plain == r["reference"] else "correct_nontrivial"


def sample_items(records, n, shared, seed):
    rng = random.Random(seed)
    by_stratum = {}
    for r in records:
        by_stratum.setdefault(stratum(r), []).append(r)
    population = {name: len(by_stratum.get(name, [])) for name, _ in STRATA}
    quotas = {name: round(n * share) for name, share in STRATA}
    quotas[STRATA[0][0]] += n - sum(quotas.values())
    # A stratum with fewer responses than its quota gives the shortfall to the next strata with spare ones.
    shortfall = 0
    for name, _ in STRATA:
        if quotas[name] > population[name]:
            shortfall += quotas[name] - population[name]
            quotas[name] = population[name]
    for name, _ in STRATA:
        extra = min(shortfall, population[name] - quotas[name])
        quotas[name] += extra
        shortfall -= extra
    chosen = []
    for name, _ in STRATA:
        pool = sorted(by_stratum.get(name, []), key=lambda r: (r["uid"], r["sample_idx"]))
        chosen += [(name, r) for r in rng.sample(pool, quotas[name])]
    rng.shuffle(chosen)
    items = []
    for i, (name, r) in enumerate(chosen):
        assigned = "ALL" if i < shared else LABELERS[(i - shared) % len(LABELERS)]
        items.append({"audit_id": f"g{i + 1:03d}", "assigned_to": assigned, "stratum": name, "record": r})
    return items, population


def write_outputs(items, population, candidates, sheet_dir=SHEET_DIR, key_dir=KEY_DIR):
    sheet_dir.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)
    with (sheet_dir / "gold_audit_sheet.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "assigned_to", "problem", "reference", "response", "label (Y/N/?)", "labeler", "notes"])
        for it in items:
            r = it["record"]
            w.writerow([it["audit_id"], it["assigned_to"], candidates[r["uid"]]["problem"], r["reference"],
                        r["response"], "", "", ""])
    with (key_dir / "key.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "uid", "sample_idx", "stratum", "gold_correct", "gold_extracted", "truncated"])
        for it in items:
            r = it["record"]
            w.writerow([it["audit_id"], r["uid"], r["sample_idx"], it["stratum"], r["gold_correct"],
                        r["gold_extracted"], r["truncated"]])
    (key_dir / "strata.json").write_text(json.dumps(
        {"population": population, "sampled": dict(Counter(it["stratum"] for it in items))}, indent=2))

    rows = [{"audit_id": it["audit_id"], "assigned_to": it["assigned_to"],
             "problem": candidates[it["record"]["uid"]]["problem"], "reference": it["record"]["reference"],
             "response": it["record"]["response"]} for it in items]
    (sheet_dir / "gold_audit.html").write_text(render_html(rows), encoding="utf-8")


def render_html(rows):
    """Readable page for labelers. Math in problems and responses is rendered with MathJax;
    each response also keeps its raw text, in case the model's LaTeX renders misleadingly."""
    cards = []
    for r in rows:
        resp = html.escape(r["response"])
        cards.append(
            f'<section><h2>{html.escape(r["audit_id"])} <small>labeler {html.escape(r["assigned_to"])}</small></h2>'
            f'<h3>Problem</h3><div class="text">{html.escape(r["problem"])}</div>'
            f'<h3>Reference answer: <code>{html.escape(r["reference"])}</code></h3>'
            f'<h3>Response</h3><div class="text response">{resp}</div>'
            f'<details><summary>raw text of the response</summary><pre>{resp}</pre></details></section>')
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Gold audit</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script>window.MathJax = {{tex: {{inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
  displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']], processEscapes: true}}}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>
<style>body{{font-family:system-ui,sans-serif;max-width:900px;margin:0 auto;padding:16px;line-height:1.55}}
section{{border:1px solid #ccc;border-radius:8px;padding:12px 16px;margin:16px 0}}
.text{{white-space:pre-wrap;overflow-wrap:anywhere;overflow-x:auto}}.response{{background:#f6f6f6;padding:8px;border-radius:6px}}
pre{{white-space:pre-wrap;background:#fbfbfb;border:1px dashed #ccc;padding:8px;border-radius:6px}}
h3{{margin:10px 0 4px}}summary{{cursor:pointer;color:#555;margin-top:6px}}
mjx-container[display="true"]{{max-width:100%;overflow-x:auto;overflow-y:hidden}}</style>
</head><body><h1>Gold checker hand audit ({len(rows)} responses)</h1><p>{html.escape(INSTRUCTIONS)}</p>
<p>Record labels in gold_audit_sheet.csv (column "label"), one row per audit_id.</p>
{''.join(cards)}</body></html>"""


def html_from_csv(csv_path, html_path):
    with Path(csv_path).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    Path(html_path).write_text(render_html(rows), encoding="utf-8")
    return len(rows)


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty-run", default="runs/difficulty")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--shared", type=int, default=30, help="items labeled by all three labelers")
    ap.add_argument("--html-from-csv", default=None,
                    help="only rebuild the HTML page from an existing sheet (no difficulty run needed)")
    args = ap.parse_args()
    if args.html_from_csv:
        out = Path(args.html_from_csv).with_name("gold_audit.html")
        print(f"wrote {display_path(out)} ({html_from_csv(args.html_from_csv, out)} responses)")
        return
    records = read_shards(ROOT / args.difficulty_run / "samples")
    items, population = sample_items(records, args.n, args.shared, cfg["seed"])
    write_outputs(items, population, load_candidates())
    print(f"population by stratum: {population}")
    print(f"sampled: {dict(Counter(it['stratum'] for it in items))}; "
          f"assigned: {dict(Counter(it['assigned_to'] for it in items))}")
    print(
        f"labeler files: {display_path(SHEET_DIR / 'gold_audit_sheet.csv')}, "
        f"{display_path(SHEET_DIR / 'gold_audit.html')}"
    )
    print(f"answer key (do not share with labelers): {display_path(KEY_DIR / 'key.csv')}")


if __name__ == "__main__":
    main()
