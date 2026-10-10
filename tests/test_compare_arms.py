import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location(
    "compare_arms", REPO / "scripts" / "compare_arms.py")
ca = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ca)


def _arm(root, name, episodes):
    """episodes: {id: n_replan}"""
    (root / name / "log").mkdir(parents=True)
    (root / name / "events").mkdir(parents=True)
    for ep, n in episodes.items():
        json.dump({"id": ep, "success": 1.0, "spl": 0.5},
                  open(root / name / "log" / f"stats_{ep}.json", "w"))
        json.dump({"id": ep, "events": [{"kind": "replan", "step": 3}] * n
                                       + [{"kind": "stop_accept", "step": 9}]},
                  open(root / name / "events" / f"events_{ep}.json", "w"))


def test_alarm_ids_are_the_episodes_that_replanned(tmp_path):
    _arm(tmp_path, "c1", {"7": 1, "9": 0, "11": 2, "13": 0})
    out = tmp_path / "a05_alarmed_ids.json"
    ca.write_alarm_ids(tmp_path, "c1", out, tau=3.96)
    d = json.load(open(out))
    assert d["episode_ids"] == ["7", "11"]
    assert "c1" in d["source"] and "3.96" in d["source"]


def test_alarm_ids_are_sorted_numerically_and_stringly(tmp_path):
    _arm(tmp_path, "c1", {"100": 1, "9": 1, "11": 1})
    out = tmp_path / "ids.json"
    ca.write_alarm_ids(tmp_path, "c1", out, tau=1.0)
    assert json.load(open(out))["episode_ids"] == ["9", "11", "100"]


def test_alarm_ids_rejects_an_arm_with_no_events(tmp_path):
    (tmp_path / "c1" / "log").mkdir(parents=True)
    (tmp_path / "c1" / "events").mkdir(parents=True)
    try:
        ca.write_alarm_ids(tmp_path, "c1", tmp_path / "ids.json", tau=1.0)
    except SystemExit:
        return
    raise AssertionError("expected SystemExit on an empty arm")


def _run(root, name, episodes):
    """episodes: {id: (steps, alarm_step or None)}"""
    (root / name / "log").mkdir(parents=True)
    (root / name / "events").mkdir(parents=True)
    for ep, (steps, alarm) in episodes.items():
        json.dump({"id": ep, "success": 0.0},
                  open(root / name / "log" / f"stats_{ep}.json", "w"))
        events = [{"kind": "replan", "step": alarm}] if alarm is not None else []
        json.dump({"id": ep, "steps": steps,
                   "events": events + [{"kind": "stop_accept", "step": steps}]},
                  open(root / name / "events" / f"events_{ep}.json", "w"))


def test_trigger_schedule_matches_alarm_count_and_relative_depth(tmp_path):
    # the judge fires halfway through each of three long episodes
    _run(tmp_path, "c1", {"2": (100, 50), "5": (60, 30), "7": (80, 40),
                          "3": (100, None), "4": (100, None)})
    _run(tmp_path, "a0", {ep: (steps, None) for ep, steps in
                          {"2": 100, "3": 100, "4": 100, "5": 60, "7": 80}.items()})
    doc = ca.write_trigger_schedule(tmp_path, "c1", tmp_path / "sched.json", seed=1)
    assert len(doc["episode_ids"]) == 3
    assert doc["episode_ids"] == sorted(doc["trigger_steps"], key=int)
    # every trigger sits at half the episode's own length, so all three are reachable
    assert all(abs(step / 100.0 - 0.5) < 0.02 if ep in ("2", "3", "4") else True
               for ep, step in doc["trigger_steps"].items())
    again = ca.write_trigger_schedule(tmp_path, "c1", tmp_path / "again.json", seed=1)
    assert again["trigger_steps"] == doc["trigger_steps"]


def test_trigger_schedule_never_schedules_past_the_episode(tmp_path):
    """A depth drawn from long episodes must not land beyond a short one: the
    arm would then run to the end without ever triggering."""
    _run(tmp_path, "c1", {"2": (200, 180), "5": (200, 170), "7": (200, 190),
                          "3": (12, None), "4": (14, None)})
    _run(tmp_path, "a0", {"2": (200, None), "3": (12, None), "4": (14, None),
                          "5": (200, None), "7": (200, None)})
    doc = ca.write_trigger_schedule(tmp_path, "c1", tmp_path / "sched.json", seed=3)
    lengths = {"2": 200, "3": 12, "4": 14, "5": 200, "7": 200}
    for ep, step in doc["trigger_steps"].items():
        assert 1 <= step <= lengths[ep] - 2, (ep, step, lengths[ep])


def test_the_episodes_to_compare_must_be_named(tmp_path):
    import pytest
    with pytest.raises(SystemExit, match="episode-list"):
        ca.main(["--root", str(tmp_path), "--arms", "a0", "c1"])
    with pytest.raises(SystemExit, match="go together"):
        ca.main(["--root", str(tmp_path), "--base-log", str(tmp_path)])

