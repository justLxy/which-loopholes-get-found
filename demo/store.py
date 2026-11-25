"""The demo's data bundle (demo/data): load it, check it, and cache the per-run files.

load_bundle(path) raises BundleError with a one-line message if the bundle is unusable. Run files are read on
demand and cached.
"""

import functools
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_DATA = Path(__file__).resolve().parent / "data"
REBUILD = "build it with `python -m src.demo.build_data` from the repository root"
REQUIRED = ("judges.json", "families.json", "risk.csv", "runs.csv")
COLUMNS = {
    "risk.csv": ["checkpoint", "judge", "pool", "family", "tier", "fpr", "p_hat", "exposure", "rank_risk", "rank_fpr"],
    "runs.csv": ["run", "judge", "pool", "seed", "checkpoint", "group", "first_event_family", "first_event_step"],
    "prereg.csv": ["run", "experiment", "kind", "judge", "pool", "seed", "condition", "q0", "first_by_exposure",
                   "exposure_top", "first_by_fpr", "first_by_propensity", "predicted_step", "step_low", "step_high",
                   "winner", "step", "hit", "in_band"]}
TIMING_KEYS = ("a", "b", "sd", "z90", "level", "step_samples")
RUN_CACHE = 64                      # cached run files


class BundleError(Exception):
    """Unusable data bundle; the message is one line."""


@dataclass(frozen=True, eq=False)
class Store:
    path: Path
    judges: dict
    families: dict
    risk: pd.DataFrame
    runs: pd.DataFrame
    results: dict | None             # results.json: optional
    prereg: pd.DataFrame | None      # prereg.csv: optional
    timing: dict | None = None       # timing.json: optional

    def run(self, name):
        """runs/<name>.json, cached. Callers must not modify the returned dict."""
        return read_run(str(self.path), name)


@functools.lru_cache(maxsize=RUN_CACHE)
def read_run(data_dir, name):
    path = Path(data_dir) / "runs" / f"{name}.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as err:
        raise BundleError(f"{path} cannot be read ({err}); {REBUILD}") from err


def _read(path, kind):
    """kind: json, csv, or text-csv (empty cells stay empty strings, as in runs.csv and prereg.csv)."""
    try:
        if kind == "json":
            return json.loads(path.read_text())
        return pd.read_csv(path, keep_default_na=kind != "text-csv")
    except (OSError, ValueError, pd.errors.ParserError) as err:
        raise BundleError(f"{path} cannot be read ({err}); {REBUILD}") from err


def _check_columns(table, path):
    missing = [c for c in COLUMNS[path.name] if c not in table.columns]
    if missing:
        raise BundleError(f"{path} lacks the columns {', '.join(missing)}; {REBUILD}")


def load_bundle(path=DEFAULT_DATA):
    path = Path(path)
    if not path.is_dir():
        raise BundleError(f"the demo data {path} does not exist; {REBUILD}")
    missing = [f for f in REQUIRED if not (path / f).is_file()] + ([] if (path / "runs").is_dir() else ["runs/"])
    if missing:
        what = "is empty" if not any(path.iterdir()) else f"is incomplete (missing {', '.join(missing)})"
        raise BundleError(f"the demo data {path} {what}; {REBUILD}")
    judges, families = _read(path / "judges.json", "json"), _read(path / "families.json", "json")
    risk, runs = _read(path / "risk.csv", "csv"), _read(path / "runs.csv", "text-csv")
    _check_columns(risk, path / "risk.csv")
    _check_columns(runs, path / "runs.csv")
    if runs.empty:
        raise BundleError(f"{path / 'runs.csv'} lists no runs; {REBUILD}")
    results = _read(path / "results.json", "json") if (path / "results.json").is_file() else None
    prereg = None
    if (path / "prereg.csv").is_file():
        prereg = _read(path / "prereg.csv", "text-csv")
        _check_columns(prereg, path / "prereg.csv")
    timing = None
    if (path / "timing.json").is_file():
        timing = _read(path / "timing.json", "json")
        timing = timing if isinstance(timing, dict) else {}
        bad = [k for k in TIMING_KEYS if not isinstance(timing.get(k), (int, float)) or isinstance(timing.get(k), bool)]
        if bad:
            raise BundleError(f"{path / 'timing.json'} lacks the numbers {', '.join(bad)}; {REBUILD}")
    names = list(runs["run"]) + ([] if prereg is None else list(prereg["run"]))
    absent = [n for n in names if not (path / "runs" / f"{n}.json").is_file()]
    if absent:
        more = f" and {len(absent) - 1} more" if len(absent) > 1 else ""
        raise BundleError(f"{path / 'runs'} lacks {absent[0]}.json{more}; {REBUILD}")
    unknown = sorted(set(runs["judge"]) - set(judges))
    if unknown:
        raise BundleError(f"{path / 'judges.json'} lacks the judges {', '.join(unknown)}; {REBUILD}")
    return Store(path, judges, families, risk, runs, results, prereg, timing)
