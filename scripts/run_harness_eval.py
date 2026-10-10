"""Closed-loop evaluation: the frozen base alone (a0), under the harness (c1),
or the recovery adapter always on without a monitor (a1).

Shards share one output directory; every episode writes its metrics to
log/stats_<id>.json (the resume marker) and its decision record to
events/events_<id>.json.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from moon.harness import HarnessConfig, HarnessLoop  # noqa: E402
from moon.harness.chunking import ACTION_NAMES  # noqa: E402
from moon.harness.env import (DATASET_CONFIGS, MAX_TOTAL_STEPS, SPIN_LIMIT,  # noqa: E402
                              build_env, reset_to, seed_all, set_navida_root)
from moon.harness.judge import (NavidaJudge, NoJudge, OracleJudge,  # noqa: E402
                                PeriodicJudge, ScheduledJudge, SpinJudge)
from moon.harness.restart import IdentityRestart  # noqa: E402
from moon.harness.settings import DECODE_MODES, ORACLE_ROUTE_MARGIN  # noqa: E402
from moon.monitor import ALPHAS, DEPLOYED_ALPHA, calibrated_tau  # noqa: E402
from moon.navida_io import RESOLUTION_RATIO  # noqa: E402

TARGET_KEYS = {"distance_to_goal", "success", "spl", "path_length",
               "oracle_success", "ndtw"}

OFFICIAL_SEED = 41      # the upstream evaluator's single global seed

# What defines the condition, as opposed to which slice of it a process runs;
# a resume compares these and nothing else.
PINNED_KEYS = ("arm", "split", "dataset", "model", "resolution_ratio", "judge_lora",
               "alpha", "drift_threshold", "risk_window", "judge_every",
               "seed_mode", "seed_base", "episode_list",
               "restart_lora", "restart_keeps_history", "trigger_schedule",
               "restart_decode", "restart_temperature", "restart_top_p",
               "decode_mode", "max_recoveries",
               "trigger", "trigger_every", "trigger_window", "trigger_route_margin",
               "merge_restart", "stop_judge_after_budget")


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["a0", "c1", "a1"],
                    help="a0 = frozen base; c1 = monitor + clear-history restart; "
                         "a1 = recovery adapter driving from the first step, no monitor")
    ap.add_argument("--split", default="val_unseen", choices=["val_seen", "val_unseen"])
    ap.add_argument("--dataset", default="r2r", choices=sorted(DATASET_CONFIGS))
    ap.add_argument("--navida-root", type=Path, default=None,
                    help="NaVIDA checkout holding config/ and data/; default $NAVIDA_ROOT")
    ap.add_argument("--model", required=True, help="frozen backbone checkpoint")
    ap.add_argument("--judge-lora", default=None, help="judge adapter; required by c1")
    ap.add_argument("--restart-lora", default=None,
                    help="recovery adapter driving the episode after a restart (c1) or "
                         "throughout (a1); absent on c1 = the frozen base restarts")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--episodes", type=int, default=-1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=None, choices=ALPHAS,
                    help="target false-alarm rate selecting the calibrated threshold of "
                         f"--dataset; default {DEPLOYED_ALPHA}")
    ap.add_argument("--drift-threshold", type=float, default=None,
                    help="threshold on the rolled judge score; overrides --alpha")
    ap.add_argument("--risk-window", type=int, default=3,
                    help="rolling-mean window over judge scores")
    ap.add_argument("--judge-every", type=int, default=1,
                    help="one judge forward per N decision points")
    ap.add_argument("--decode", dest="decode_mode", choices=DECODE_MODES, default="greedy",
                    help="policy decoding: greedy, or the upstream sampler (official)")
    ap.add_argument("--resolution-ratio", type=float, default=RESOLUTION_RATIO,
                    help="sensor-frame downscale before the policy resize")
    ap.add_argument("--restart-keeps-history", action="store_true",
                    help="c1 ablation: mount the recovery adapter at the alarm but keep "
                         "the frame history instead of restarting on the current frame")
    ap.add_argument("--restart-decode", choices=["greedy", "sample"], default="greedy",
                    help="c1: decoding after the restart; sample draws a continuation with "
                         "the recovery trainer's sampler while the pre-alarm run stays greedy")
    ap.add_argument("--restart-temperature", type=float, default=1.6)
    ap.add_argument("--restart-top-p", type=float, default=1.0)
    ap.add_argument("--max-recoveries", type=int, default=1,
                    help="restarts allowed per episode. A second alarm can only fire after "
                         "the refractory period and resets the rolling risk, so k > 1 "
                         "re-clears under the recovery adapter")
    ap.add_argument("--merge-restart", action="store_true",
                    help="c1: merge the recovery adapter into a plain copy of the backbone "
                         "at the alarm, restored exactly afterwards, and run the frozen base "
                         "on that copy without the LoRA wrappers (a deployment-speed build)")
    ap.add_argument("--stop-judge-after-budget", action="store_true",
                    help="c1: no judge calls once the episode has used its restarts; "
                         "their scores can no longer trigger anything")
    ap.add_argument("--trigger", choices=["judge", "periodic", "spin", "oracle"],
                    default="judge",
                    help="what fires the restart on a c1 arm: the calibrated judge, a fixed "
                         "step interval, a stuck detector over the recent actions, or the "
                         "ground-truth distance to the reference path")
    ap.add_argument("--trigger-every", type=int, default=42,
                    help="periodic: fire once the episode reaches this step")
    ap.add_argument("--trigger-window", type=int, default=6,
                    help="spin: fire when the last N executed actions contain no forward move")
    ap.add_argument("--trigger-route-margin", type=float, default=ORACLE_ROUTE_MARGIN,
                    help="oracle: fire beyond this geodesic distance from the reference path")
    ap.add_argument("--trigger-schedule", type=Path, default=None,
                    help="c1 control: json with trigger_steps {episode: step}; the alarm "
                         "fires at the first decision point on or after that step "
                         "instead of coming from the judge (window 1, threshold 0)")
    ap.add_argument("--seed-mode", choices=["official", "paired"], default="paired",
                    help="paired (default) seeds every episode from --seed-base so arms "
                         "can be compared episode by episode; official reproduces the "
                         "upstream evaluator's single global seed")
    ap.add_argument("--seed-base", type=int, default=OFFICIAL_SEED,
                    help="paired mode: episode seed = seed_base + episode_id %% 100000")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--episode-list", type=Path, default=None,
                    help="json with episode_ids; restricts the run to those episodes")
    return ap.parse_args(argv)


def resolve_threshold(args):
    """The drift threshold of a judge-triggered c1 arm: explicit, or the calibrated
    tau of the dataset at --alpha (its deployed alpha when unset)."""
    if args.drift_threshold is not None:
        return float(args.drift_threshold)
    if args.arm != "c1" or args.trigger != "judge" or args.trigger_schedule:
        return None
    try:
        return calibrated_tau(args.dataset, args.alpha)
    except ValueError as e:
        raise SystemExit(f"{e}; pass --drift-threshold")


def write_run_config(out_dir, args, resolved):
    """Record the condition; refuse to resume an output directory under another one.

    The stats files double as the resume marker, so a second invocation with
    different settings would quietly mix two conditions in one directory.
    Paths are recorded as given on the command line.
    """
    cfg = {k: (v.as_posix() if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg.update({k: (v.as_posix() if isinstance(v, Path) else v) for k, v in resolved.items()})
    path = Path(out_dir) / "run_config.json"
    if path.exists():
        # Shards race through here at start-up and can catch a sibling's
        # half-written file; only a readable copy can prove a mismatch.
        old = None
        for _ in range(10):
            try:
                text = path.read_text()
                if text.strip():
                    old = json.loads(text)
                    break
            except (json.JSONDecodeError, OSError):
                pass
            time.sleep(0.5)
        if old is not None:
            # a key absent from an older run_config predates the setting; only a
            # recorded value that disagrees means two conditions in one directory
            drift = {k: (old.get(k), cfg.get(k)) for k in PINNED_KEYS
                     if k in old and old.get(k) != cfg.get(k)}
            if drift:
                raise SystemExit(
                    f"{path} records a different condition {drift}; resuming here "
                    "would mix two settings in one output directory")
    tmp = path.with_suffix(".json.tmp%d" % os.getpid())
    tmp.write_text(json.dumps(cfg, indent=2, sort_keys=True, default=str))
    tmp.replace(path)
    return {k: cfg.get(k) for k in PINNED_KEYS}


def route_distance(env, episode):
    """Geodesic distance from the agent to the episode's reference path: the
    ground-truth deviation signal the learned monitor has to predict."""
    path = getattr(episode, "reference_path", None)
    if not path:
        return None
    pos = env.sim.get_agent(0).get_state().position
    d = env.sim.geodesic_distance(pos, [list(p) for p in path])
    return None if d is None or math.isinf(d) else float(d)


def run_episode(env, loop, base, ep_id, episode, with_route_dist=False):
    obs = reset_to(env, episode)
    st = loop.reset()
    base.reset(env.current_episode.instruction.instruction_text)
    base.observe(obs["rgb"])
    spin, last_dtg, iter_step, done = 0, 999.0, 0, False

    while not env.episode_over and not done:
        # The judge reads the driver's own frame list and action record, so a
        # clear-history restart resets both together.
        st.judge_ctx = {"frames": base.judge_rgb_list,
                        "recent": base.recent_actions,
                        "instruction": base.instruction}
        if with_route_dist:
            st.judge_ctx["route_dist"] = route_distance(env, episode)
        atoms = loop.decide(st)
        if not atoms:
            break
        for a in atoms:
            if env.episode_over:
                break
            info = env.get_metrics()
            if info["distance_to_goal"] != last_dtg:
                last_dtg, spin = info["distance_to_goal"], 0
            else:
                spin += 1
            if spin > SPIN_LIMIT or iter_step > MAX_TOTAL_STEPS:
                env.step({"action": 0})     # upstream forces STOP here
                done = True
                break
            t0 = time.perf_counter()
            obs = env.step({"action": a})
            st.sim_seconds += time.perf_counter() - t0
            st.sim_steps += 1
            iter_step += 1
            base.note_action(ACTION_NAMES[a])
            base.observe(obs["rgb"])
            loop.record_executed(st, [a])

    info = env.get_metrics()
    result = {k: info[k] for k in TARGET_KEYS if k in info}
    result["id"] = ep_id
    return result, st


def events_payload(ep_id, arm, st, run_cfg, wall):
    """What the episode leaves behind: the record an offline replay reads back.

    ``executed_atoms`` and ``decision_steps`` are the simulator's own action
    record, so a later pass can reproduce the trajectory frame by frame without
    the policy."""
    return {"id": ep_id, "arm": arm, "steps": st.step,
            "config": run_cfg,
            "events": st.events,
            "recoveries": st.recoveries_used,
            "mid_scores": st.mid_scores,
            "executed_atoms": st.executed_atoms,
            "decision_steps": st.decision_steps,
            "timing": {"wall_seconds": round(wall, 4),
                       "sim_seconds": round(st.sim_seconds, 4), "sim_steps": st.sim_steps,
                       "policy_seconds": [round(t, 4) for t in st.policy_seconds],
                       "judge_seconds": [round(t, 4) for t in st.judge_seconds]}}


def check_args(args):
    if args.restart_lora and args.arm == "a0":
        raise SystemExit("--restart-lora applies to c1 (mounted at the alarm) or a1 (always on)")
    if args.arm == "a1" and not args.restart_lora:
        raise SystemExit("a1 is the recovery adapter always on: --restart-lora missing")
    if (args.arm == "c1" and not args.judge_lora and not args.trigger_schedule
            and args.trigger == "judge"):
        raise SystemExit("c1 is monitor + clear-history restart: --judge-lora missing")
    if args.trigger_schedule and (args.arm != "c1" or args.judge_lora):
        raise SystemExit("--trigger-schedule replaces the judge of a c1 arm")
    if args.trigger != "judge":
        if args.arm != "c1" or args.judge_lora or args.trigger_schedule:
            raise SystemExit(f"--trigger {args.trigger} replaces the judge of a c1 arm")
    if args.restart_keeps_history and args.arm != "c1":
        raise SystemExit("--restart-keeps-history applies to c1 only")
    if args.merge_restart and (args.arm != "c1" or not args.restart_lora):
        raise SystemExit("--merge-restart applies to c1 with --restart-lora")
    if args.restart_decode == "sample" and args.arm != "c1":
        raise SystemExit("--restart-decode sample applies to c1 only")
    if args.seed_mode == "official" and args.seed_base != OFFICIAL_SEED:
        raise SystemExit("--seed-base applies to --seed-mode paired only")


def main(argv=None):
    args = parse_args(argv)
    check_args(args)
    from moon.harness.host import NavidaHost
    from moon.harness.navida_driver import NavidaDriver, sampling_config
    set_navida_root(args.navida_root)
    # build_env chdirs into the NaVIDA root, so every path is resolved first;
    # run_config.json keeps them as given.
    out_dir = args.out_dir.resolve()
    model = str(Path(args.model).resolve())
    judge_lora = str(Path(args.judge_lora).resolve()) if args.judge_lora else None
    restart_lora = str(Path(args.restart_lora).resolve()) if args.restart_lora else None
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "log").mkdir(exist_ok=True)
    (out_dir / "events").mkdir(exist_ok=True)

    drift_threshold = resolve_threshold(args)
    run_cfg = write_run_config(out_dir, args, {"drift_threshold": drift_threshold})

    if args.seed_mode == "official":
        random.seed(OFFICIAL_SEED)
        np.random.seed(OFFICIAL_SEED)
        print("[seed] official mode: torch unseeded, as upstream; episodes are not "
              "paired with other arms", flush=True)
    else:
        print(f"[seed] paired mode: episode seed = {args.seed_base} + id % 100000", flush=True)

    adapters = {}
    if restart_lora:
        adapters["restart"] = restart_lora
    if judge_lora:
        adapters["judge"] = judge_lora
    host = NavidaHost(model, device=f"cuda:{args.gpu}", adapters=adapters or None,
                      merge_adapters=("restart",) if args.merge_restart else ())
    restart_gen_cfg = None
    if args.restart_decode == "sample":
        restart_gen_cfg = sampling_config(args.restart_temperature, args.restart_top_p)
    base = NavidaDriver(host, decode=args.decode_mode, resolution_ratio=args.resolution_ratio,
                        restart_gen_cfg=restart_gen_cfg)
    schedule = None
    cheap_trigger = args.trigger != "judge"
    if args.trigger_schedule:
        schedule = json.load(open(args.trigger_schedule))["trigger_steps"]
        judge = ScheduledJudge(schedule)
    elif args.trigger == "periodic":
        judge = PeriodicJudge(args.trigger_every)
    elif args.trigger == "spin":
        judge = SpinJudge(args.trigger_window)
    elif args.trigger == "oracle":
        judge = OracleJudge(args.trigger_route_margin)
    else:
        judge = NavidaJudge(host) if judge_lora else NoJudge()

    if args.arm != "c1":
        cfg = HarnessConfig(drift_threshold=float("inf"), risk_window=1)
    elif schedule is not None or cheap_trigger:
        # A single-decision trigger: a rolling mean would dilute the spike.
        cfg = HarnessConfig(drift_threshold=0.0, risk_window=1, judge_every=1,
                            max_recoveries=args.max_recoveries,
                            restart_clears_history=not args.restart_keeps_history,
                            judge_after_budget=not args.stop_judge_after_budget)
    else:
        cfg = HarnessConfig(drift_threshold=drift_threshold,
                            risk_window=args.risk_window, risk_mode="mean",
                            judge_every=args.judge_every, max_recoveries=args.max_recoveries,
                            restart_clears_history=not args.restart_keeps_history,
                            judge_after_budget=not args.stop_judge_after_budget)

    episode_list = None
    if args.episode_list:
        episode_list = json.load(open(args.episode_list))["episode_ids"]

    env = build_env(args.split, args.dataset)
    by_id = {str(e.episode_id): e for e in env.episodes}
    ids = [e.episode_id for e in env.episodes]
    if episode_list is not None:
        want = set(episode_list)
        ids = [i for i in ids if str(i) in want]
    if args.num_shards > 1:
        ids = ids[args.shard::args.num_shards]
    if args.episodes > 0:
        ids = ids[: args.episodes]

    loop = HarnessLoop(cfg, base, judge)
    done = 0
    for ep_id in [str(i) for i in ids]:
        stats_p = out_dir / "log" / f"stats_{ep_id}.json"
        if stats_p.exists():
            continue
        if args.seed_mode == "paired":
            seed_all(args.seed_base + (int(ep_id) % 100000))
        base.policy_role = "restart" if args.arm == "a1" else None
        if schedule is not None or cheap_trigger:
            judge.begin(ep_id)
        if args.arm == "c1":
            instruction = by_id[ep_id].instruction.instruction_text
            loop.restart = (IdentityRestart(instruction, base) if restart_lora
                            else IdentityRestart(instruction))
        t_ep = time.perf_counter()
        result, st = run_episode(env, loop, base, ep_id, by_id[ep_id],
                                 with_route_dist=args.trigger == "oracle")
        wall = time.perf_counter() - t_ep
        with open(out_dir / "events" / f"events_{ep_id}.json", "w") as f:
            json.dump(events_payload(ep_id, args.arm, st, run_cfg, wall), f)
        with open(stats_p, "w") as f:      # written last: the resume marker
            json.dump(result, f, indent=4)
        done += 1
        print(f"[{args.arm}] ep={ep_id} success={result.get('success')} "
              f"steps={st.step} triggers={st.recoveries_used}", flush=True)
    print(f"DONE arm={args.arm} episodes={done} -> {args.out_dir}")


if __name__ == "__main__":
    main()

