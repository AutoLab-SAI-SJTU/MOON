#!/usr/bin/env python
"""Compare evaluation arms on a pre-registered episode list.

Per-arm aggregates; the episode-paired comparison against the control arm with
exact McNemar; and, for the judged arms, the alarm rate, the unnecessary-restart
rate on control successes and the rescue rate on control failures. Partial
runs are tolerated: paired statistics use only the episodes every arm finished,
and report that n.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="directory holding one sub-directory per arm")
    ap.add_argument("--arms", nargs="+", default=["a0", "c1"])
    ap.add_argument("--episode-list", type=Path, default=None,
                    help="json with episode_ids to compare on; not needed with --base-log")
    ap.add_argument("--out", type=Path, default=None,
                    help="output path; defaults to <root>/analysis.json")
    ap.add_argument("--control", default="a0", help="arm every other arm is paired against")
    ap.add_argument("--require-complete", action="store_true",
                    help="abort unless every arm finished every episode in --episode-list")
    ap.add_argument("--write-alarm-ids", type=Path, default=None,
                    help="write the first --arms entry's alarm set here and exit")
    ap.add_argument("--tau", type=float, default=float("nan"),
                    help="threshold recorded in the alarm-set file's source line")
    ap.add_argument("--write-trigger-schedule", type=Path, default=None,
                    help="write a random-trigger control matched to the first --arms entry's "
                         "alarm set here and exit: as many random episodes, each triggered at "
                         "the relative depth of one of the arm's alarms")
    ap.add_argument("--schedule-seed", type=int, default=2026)
    ap.add_argument("--base-arm", default="a0",
                    help="arm holding the un-intervened run, for --write-trigger-schedule")
    ap.add_argument("--base-log", type=Path, default=None,
                    help="flat log directory of the base run; with --alarm-ids, every arm is "
                         "stitched onto it (alarmed episodes from the arm, the rest from the base) "
                         "so an arm that replayed only its alarm set reports a full-split number")
    ap.add_argument("--alarm-ids", type=Path, default=None,
                    help="the alarm set the arms replayed, as written by --write-alarm-ids")
    return ap.parse_args(argv)


def load_arm(root: Path, arm: str):
    stats, events = {}, {}
    for p in sorted((root / arm / "log").glob("stats_*.json")):
        r = json.load(open(p))
        stats[str(r["id"])] = r
    for p in sorted((root / arm / "events").glob("events_*.json")):
        r = json.load(open(p))
        events[str(r["id"])] = r
    return stats, events


def mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact binomial: b = arm only, c = control only."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mcnemar_one_sided(b: int, c: int) -> float:
    """P(at least b of b+c discordant pairs favour the arm | fair coin)."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(b, n + 1)) / 2 ** n


def metric(s: dict, key: str) -> float:
    """A missing, null or NaN metric counts as zero: those episodes ran and failed."""
    v = s.get(key)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 0.0
    return float(v)


def agg(stats: dict) -> dict:
    n = len(stats)
    if n == 0:
        return {"n": 0}
    keys = ["success", "spl", "oracle_success", "ndtw", "distance_to_goal", "path_length"]
    out = {"n": n}
    for k in keys:
        # nDTW is null on R2R; a key nobody reports is left out rather than shown as 0
        if any(s.get(k) is not None for s in stats.values()):
            out[k] = sum(metric(s, k) for s in stats.values()) / n
    return out


def restarted(ev: dict) -> bool:
    return any(e.get("kind") == "replan" for e in ev.get("events", []))


def episode_sort_key(ep: str):
    return (0, int(ep)) if ep.isdigit() else (1, ep)


def write_alarm_ids(root: Path, arm: str, out: Path, tau: float) -> list:
    """The alpha's alarm set: the episodes the restart-only arm intervened on.
    Adapter arms replay exactly this list, so it must come from an arm that ran
    the whole split without a recovery adapter."""
    _, events = load_arm(root, arm)
    if not events:
        raise SystemExit(f"no events under {root / arm / 'events'}")
    ids = sorted((e for e, ev in events.items() if restarted(ev)), key=episode_sort_key)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"episode_ids": ids, "source": f"{arm} first-alarm set (tau {tau})"},
              open(out, "w"), ensure_ascii=False)
    return ids


def write_trigger_schedule(root: Path, arm: str, out: Path, seed: int,
                           base_arm: str = "a0") -> dict:
    """A random-trigger control for the arm's alarm set: as many episodes drawn
    at random, each triggered at the same *relative* depth as one of the arm's
    alarms.

    Depth is matched relative to the episode's own length rather than in steps:
    an alarm 60 steps into a long episode is not reachable in a 40-step one, and
    an arm that never triggers is not a matched control. The un-intervened base
    run supplies both the episode pool and the lengths, so the schedule uses only
    what is known before the arm runs."""
    _, events = load_arm(root, arm)
    base_stats, base_events = load_arm(root, base_arm)
    if not base_events:
        raise SystemExit(f"no base events under {root / base_arm / 'events'}")
    steps = {e: int(ev["steps"]) for e, ev in base_events.items()}
    depths = []
    for e, ev in events.items():
        if not restarted(ev) or steps.get(e, 0) <= 0:
            continue
        first = next(x["step"] for x in ev["events"] if x.get("kind") == "replan")
        depths.append(first / steps[e])
    if not depths:
        raise SystemExit(f"no alarms under {root / arm / 'events'}")
    pool = [e for e in sorted(steps, key=episode_sort_key) if steps[e] >= 3]
    if len(pool) < len(depths):
        raise SystemExit(f"{base_arm} has {len(pool)} usable episodes for {len(depths)} alarms")
    rng = random.Random(seed)
    ids = sorted(rng.sample(pool, len(depths)), key=episode_sort_key)
    rng.shuffle(depths)
    trigger = {e: max(1, min(round(r * steps[e]), steps[e] - 2))
               for e, r in zip(ids, depths)}
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"episode_ids": ids, "trigger_steps": trigger,
           "source": f"{arm} alarm depths, relative to each episode's own length in "
                     f"{base_arm}, on {len(ids)} random episodes (seed {seed})"}
    json.dump(doc, open(out, "w"), ensure_ascii=False)
    return doc


def load_log(d: Path) -> dict:
    """Per-episode stats from a flat log directory, e.g. the official base run."""
    return {str(r["id"]): r
            for r in (json.load(open(p)) for p in sorted(d.glob("stats_*.json")))}


def stitch(base_stats: dict, arm_stats: dict, ids: set) -> dict:
    """The deployed result of an arm that only replayed the alarm set.

    Alarmed episodes come from the arm, every other episode from the base run;
    an arm episode outside the alarm set is ignored, so a partially reused
    directory cannot leak into the full-split number."""
    return {e: (arm_stats[e] if e in ids and e in arm_stats else r)
            for e, r in base_stats.items()}


def rescue_harm(base_stats: dict, stitched_stats: dict, ids: set):
    """Alarm-set episodes the arm turned from failure to success, and the reverse."""
    rescued = harmed = 0
    for e in ids & set(base_stats):
        was = metric(base_stats[e], "success") >= 0.5
        now = metric(stitched_stats[e], "success") >= 0.5
        rescued += (not was) and now
        harmed += was and not now
    return rescued, harmed


def main(argv=None):
    args = parse_args(argv)
    if args.write_alarm_ids is not None:
        ids = write_alarm_ids(args.root, args.arms[0], args.write_alarm_ids, args.tau)
        print(f"[alarm-ids] {len(ids)} episodes -> {args.write_alarm_ids}")
        return
    if args.write_trigger_schedule is not None:
        doc = write_trigger_schedule(args.root, args.arms[0], args.write_trigger_schedule,
                                     args.schedule_seed, args.base_arm)
        print(f"[trigger-schedule] {len(doc['episode_ids'])} episodes -> "
              f"{args.write_trigger_schedule}")
        return
    if (args.base_log is None) != (args.alarm_ids is None):
        raise SystemExit("--base-log and --alarm-ids go together")
    if args.base_log is None and args.episode_list is None:
        raise SystemExit("name the episodes to compare: --episode-list, or --base-log with --alarm-ids")
    base_stats = load_log(args.base_log) if args.base_log else None
    alarm_ids = set(json.load(open(args.alarm_ids))["episode_ids"]) if args.alarm_ids else None
    want = set(base_stats) if base_stats else set(json.load(open(args.episode_list))["episode_ids"])
    arms = {}
    for a in args.arms:
        st, ev = load_arm(args.root, a)
        if base_stats is not None:
            done = alarm_ids & set(st)
            if done < alarm_ids:
                print(f"[partial] {a}: {len(done)}/{len(alarm_ids)} alarm episodes replayed")
            st = stitch(base_stats, st, alarm_ids)
        st = {k: v for k, v in st.items() if k in want}
        ev = {k: v for k, v in ev.items() if k in want}
        arms[a] = (st, ev)
        print(f"[load] {a}: {len(st)}/{len(want)} episodes")
        missing = want - set(st)
        if missing:
            head = ", ".join(sorted(missing)[:10])
            msg = (f"[incomplete] {a} is missing {len(missing)} episode(s): {head}"
                   f"{' ...' if len(missing) > 10 else ''}")
            if args.require_complete:
                raise SystemExit(msg + "\n[abort] --require-complete: a dead shard would "
                                       "otherwise be reported as a full-set result")
            print(msg)

    out = {"per_arm": {}, "paired": {}, "intervention": {}, "paired_skipped": {}}

    if base_stats is not None:
        g = agg(base_stats)
        out["per_arm"]["base"] = g
        print(f"\n== base run ==\n  base: n={g['n']} SR={g['success']*100:.2f} "
              f"SPL={g['spl']*100:.2f}  alarm set {len(alarm_ids)} episodes")

    print("\n== per-arm aggregates (each over its finished episodes) ==")
    for a, (st, _) in arms.items():
        g = agg(st)
        out["per_arm"][a] = g
        if g["n"]:
            cells = [f"SR={g['success']*100:.2f}"]
            for key, label, scale in (("spl", "SPL", 100), ("oracle_success", "oSR", 100),
                                      ("ndtw", "nDTW", 100), ("distance_to_goal", "dtg", 1)):
                if key in g:
                    cells.append(f"{label}={g[key]*scale:.2f}")
            print(f"  {a}: n={g['n']} " + " ".join(cells))

    if base_stats is not None:
        print("\n== rescue / harm on the alarm set (vs the base run) ==")
        n_fail = sum(1 for e in alarm_ids if metric(base_stats[e], "success") < 0.5)
        n_succ = len(alarm_ids) - n_fail
        for a, (st, _) in arms.items():
            resc, harm = rescue_harm(base_stats, st, alarm_ids)
            out["per_arm"].setdefault(a, {}).update({"rescued": resc, "harmed": harm})
            print(f"  {a}: rescued {resc}/{n_fail} ({resc/max(n_fail,1)*100:.0f}%) "
                  f"harmed {harm}/{n_succ} ({harm/max(n_succ,1)*100:.0f}%)")

    ctrl = args.control
    # A comparison that is not computed is said so, in the printout and in the
    # JSON; silence would read as "no significant difference".
    if ctrl not in arms:
        reason = f"control arm {ctrl!r} not among --arms {sorted(arms)}"
        out["paired_skipped"]["*"] = reason
        print(f"\n[skip] no paired block computed: {reason}")
    elif not arms[ctrl][0]:
        reason = (f"control arm {ctrl!r} finished 0 of {len(want)} episodes, so "
                  "every comparison against it is undefined")
        out["paired_skipped"]["*"] = reason
        print(f"\n[skip] no paired block computed: {reason}")
    else:
        ctrl_st = arms[ctrl][0]
        print(f"\n== episode-paired vs {ctrl.upper()} ==")
        for a, (st, _) in arms.items():
            if a == ctrl:
                continue
            common = sorted(set(st) & set(ctrl_st))
            if not common:
                reason = (f"no episode finished by both ({len(st)} finished in {a}, "
                          f"{len(ctrl_st)} in {ctrl}, 0 shared)")
                out["paired_skipped"][a] = reason
                print(f"  [skip] {a} vs {ctrl} NOT compared: {reason}")
                continue
            b = sum(1 for e in common if st[e]["success"] > ctrl_st[e]["success"])
            c = sum(1 for e in common if st[e]["success"] < ctrl_st[e]["success"])
            sr_a = sum(st[e]["success"] for e in common) / len(common)
            sr_0 = sum(ctrl_st[e]["success"] for e in common) / len(common)
            p1, p2 = mcnemar_one_sided(b, c), mcnemar_p(b, c)
            out["paired"][a] = {"n": len(common), "sr": sr_a, f"sr_{ctrl}": sr_0,
                                "control": ctrl, "rescued": b, "harmed": c,
                                "mcnemar_p_one_sided": p1, "mcnemar_p_two_sided": p2,
                                "discordant": b + c}
            print(f"  {a} vs {ctrl} (n={len(common)}): SR {sr_0*100:.2f}->{sr_a*100:.2f} "
                  f"({(sr_a-sr_0)*100:+.2f}) rescued={b} harmed={c} "
                  f"discordant={b+c} McNemar p={p1:.4f} (one-sided) / {p2:.4f} (two-sided)")
            if b + c:
                sr_d_a = sum(st[e]["success"] for e in common
                             if st[e]["success"] != ctrl_st[e]["success"]) / (b + c)
                print(f"     on the {b+c} discordant episodes only: "
                      f"{a} {sr_d_a*100:.1f}% vs {ctrl} {(1-sr_d_a)*100:.1f}%")

    print("\n== intervention accounting ==")
    if base_stats is not None:
        print("  (rates below are over the alarm set only: a stitched arm replayed nothing else)")
    for a, (st, ev) in arms.items():
        if a == ctrl or not ev:
            continue
        rec = {}
        int_eps = {e for e, v in ev.items() if restarted(v)}
        rec["intervened_eps"] = len(int_eps)
        print(f"  {a}: restarted={len(int_eps)} / {len(ev)} episodes")

        if ctrl in arms and arms[ctrl][0]:
            a0_st = arms[ctrl][0]
            common = set(ev) & set(a0_st)
            a0_succ = {e for e in common if a0_st[e]["success"] >= 1}
            a0_fail = common - a0_succ
            if a0_succ:
                rec["false_alarm_rate"] = len(int_eps & a0_succ) / len(a0_succ)
                print(f"     unnecessary-restart rate (control successes, n={len(a0_succ)}): "
                      f"{rec['false_alarm_rate']*100:.2f}%")
            if a0_fail:
                cov = len(int_eps & a0_fail) / len(a0_fail)
                hit = [e for e in (int_eps & a0_fail) if e in st]
                resc = sum(st[e]["success"] for e in hit) / len(hit) if hit else 0.0
                rec["fail_coverage"] = cov
                rec["rescue_rate_on_intervened_fail"] = resc
                print(f"     coverage of control failures (n={len(a0_fail)}): {cov*100:.2f}% restarted; "
                      f"rescue rate among them {resc*100:.2f}%")
            flips = [e for e in (set(st) & set(a0_st) & set(ev))
                     if st[e]["success"] != a0_st[e]["success"]]
            attrib = {k: [0, 0] for k in ("restart", "none")}
            for e in flips:                     # [rescued, harmed]
                key = "restart" if e in int_eps else "none"
                attrib[key][0 if st[e]["success"] > a0_st[e]["success"] else 1] += 1
            rec["flip_attrib"] = attrib
            print(f"     flip attribution (rescued/harmed, n={len(flips)}): "
                  f"restart={attrib['restart'][0]}/{attrib['restart'][1]} "
                  f"no-intervention={sum(attrib['none'])}"
                  f"{' (flips without an intervention: check determinism)' if sum(attrib['none']) else ''}")
        out["intervention"][a] = rec

    dst = args.out or args.root / "analysis.json"
    json.dump(out, open(dst, "w"), ensure_ascii=False, indent=1)
    print(f"\n[wrote] {dst}")


if __name__ == "__main__":
    main()

