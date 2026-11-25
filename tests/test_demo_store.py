"""The demo's bundle loader (demo/store.py): validation messages and the per-run cache."""

import importlib.util
import json
import shutil

import pytest

from src.paths import ROOT

DATA = ROOT / "demo" / "data"
if not (DATA / "runs.csv").exists():
    pytest.skip("build demo/data first (python -m src.demo.build_data)", allow_module_level=True)

spec = importlib.util.spec_from_file_location("demo_store", ROOT / "demo" / "store.py")
store = importlib.util.module_from_spec(spec)
spec.loader.exec_module(store)


@pytest.fixture
def bundle(tmp_path):
    """A writable copy of the committed bundle."""
    return shutil.copytree(DATA, tmp_path / "data")


def error(path):
    with pytest.raises(store.BundleError) as info:
        store.load_bundle(path)
    message = str(info.value)
    assert "\n" not in message and "python -m src.demo.build_data" in message
    return message


def test_the_committed_bundle_loads():
    s = store.load_bundle(DATA)
    assert len(s.runs) == 36 and len(s.prereg) == 10 and s.results is not None
    assert len(s.families) == 14 and set(s.runs["judge"]) <= set(s.judges)


def test_missing_and_empty_bundles(tmp_path):
    assert "does not exist" in error(tmp_path / "nothing")
    (tmp_path / "empty").mkdir()
    assert "is empty" in error(tmp_path / "empty")


@pytest.mark.parametrize("name", ["judges.json", "families.json", "risk.csv", "runs.csv", "runs"])
def test_each_required_part_is_checked(bundle, name):
    target = bundle / name
    shutil.rmtree(target) if target.is_dir() else target.unlink()
    assert "is incomplete" in error(bundle) and (name if name != "runs" else "runs/") in error(bundle)


def test_missing_columns_and_rows(bundle):
    risk = (bundle / "risk.csv").read_text().splitlines()
    (bundle / "risk.csv").write_text("\n".join([risk[0].replace("rank_fpr", "rank_x")] + risk[1:]))
    assert "risk.csv lacks the columns rank_fpr" in error(bundle)


def test_a_runs_csv_without_runs(bundle):
    header = (bundle / "runs.csv").read_text().splitlines()[0]
    (bundle / "runs.csv").write_text(header + "\n")
    assert "lists no runs" in error(bundle)


def test_a_missing_run_file_and_unreadable_files(bundle):
    name = (bundle / "runs.csv").read_text().splitlines()[1].split(",")[0]
    (bundle / "runs" / f"{name}.json").unlink()
    assert f"lacks {name}.json" in error(bundle)
    (bundle / "runs" / f"{name}.json").write_text("{}")
    (bundle / "judges.json").write_text("{not json")
    assert "judges.json cannot be read" in error(bundle)


def test_prereg_is_optional_but_checked(bundle):
    (bundle / "prereg.csv").unlink()
    assert store.load_bundle(bundle).prereg is None
    (bundle / "prereg.csv").write_text("run,experiment\nx,B\n")
    assert "prereg.csv lacks the columns" in error(bundle)


def test_run_files_are_read_once(bundle):
    s = store.load_bundle(bundle)
    name = s.runs["run"][0]
    store.read_run.cache_clear()
    first = s.run(name)
    (bundle / "runs" / f"{name}.json").write_text(json.dumps({"changed": True}))
    assert s.run(name) is first and "changed" not in first
    info = store.read_run.cache_info()
    assert (info.hits, info.misses) == (1, 1)
