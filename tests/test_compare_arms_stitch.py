import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_arms as ca


def test_agg_tolerates_absent_metrics():
    """R2R stats carry ndtw: null; summing them raised TypeError."""
    stats = {"1": {"success": 1.0, "spl": 0.8, "ndtw": None},
             "2": {"success": 0.0, "spl": None, "ndtw": None}}
    g = ca.agg(stats)
    assert g["success"] == 0.5
    assert g["spl"] == pytest.approx(0.4)
    assert "ndtw" not in g


def write_log(d: Path, rows):
    (d / "log").mkdir(parents=True)
    for r in rows:
        json.dump(r, open(d / "log" / f"stats_{r['id']}.json", "w"))


def test_stitch_takes_alarmed_from_arm_and_rest_from_base(tmp_path):
    base = {"1": 1.0, "2": 0.0, "3": 0.0, "4": 1.0}
    base_stats = {e: {"id": e, "success": s, "spl": s} for e, s in base.items()}
    arm = {"2": {"id": "2", "success": 1.0, "spl": 0.5},   # rescued
           "3": {"id": "3", "success": 0.0, "spl": 0.0},
           "4": {"id": "4", "success": 0.0, "spl": 0.0}}   # harmed
    out = ca.stitch(base_stats, arm, {"2", "3", "4"})
    assert [out[e]["success"] for e in ("1", "2", "3", "4")] == [1.0, 1.0, 0.0, 0.0]
    assert ca.rescue_harm(base_stats, out, {"2", "3", "4"}) == (1, 1)


def test_stitch_ignores_arm_episodes_outside_the_alarm_set(tmp_path):
    base_stats = {"1": {"id": "1", "success": 0.0}, "2": {"id": "2", "success": 0.0}}
    arm = {"1": {"id": "1", "success": 1.0}, "2": {"id": "2", "success": 1.0}}
    out = ca.stitch(base_stats, arm, {"1"})
    assert out["1"]["success"] == 1.0
    assert out["2"]["success"] == 0.0

