#!/usr/bin/env python3
"""Longer ASM PPO on the 4-cell RLC UM scenario.

Same placement, traffic, and handover as asm_state_s2_4enb.py. Training is
16384 steps: about 20 episodes on each of 8 parallel simulations. The 2048-step
model stays in output_asm_state_s2_4enb_um. all_on and alternate are copied
from that table.
"""
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))
from asm_ppo import eval_ppo, load_config, policy_init_modes, train_ppo, write_table

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GYM = os.path.join(ROOT, "nsorangym_gym")
EX = os.path.join(GYM, "examples")
KIND = "asm"
HANDOVER = "Threshold"
N_GNB = 4
STEPS = 16384


def latest_checkpoint(train_dir):
    ckpt_dir = os.path.join(train_dir, "checkpoints")
    best = None
    best_steps = -1
    if not os.path.isdir(ckpt_dir):
        return None, None
    prefix = f"ppo_{KIND}_"
    suffix = "_steps.zip"
    for name in os.listdir(ckpt_dir):
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        if "vecnormalize" in name:
            continue
        try:
            steps = int(name[len(prefix):-len(suffix)])
        except ValueError:
            continue
        if steps > best_steps:
            best_steps = steps
            best = os.path.join(ckpt_dir, name)
    norm = os.path.join(ckpt_dir, f"ppo_{KIND}_vecnormalize_{best_steps}_steps.pkl")
    if not os.path.isfile(norm):
        norm = None
    return best, norm


def main():
    os.environ.setdefault("NS3_SKIP_BUILD", "1")
    out = os.path.join(EX, "output_asm_state_s2_4enb_um_t16384")
    os.makedirs(out, exist_ok=True)
    args = SimpleNamespace(
        config=os.path.join(
            GYM, "src/environments/scenario_configurations/asm_s2_asm_isolated.json"
        ),
        ns3_path=os.path.join(ROOT, "nsorangym_mmwave"),
        optimized=False,
        skip_configuration=True,
        omega=0.0,
        kind=KIND,
        simTime=10.0,
        seed=0,
        n_eval_episodes=2,
        total_timesteps=STEPS,
        n_steps=32,
        batch_size=16,
        n_envs=8,
        until_converge=False,
        chunk_timesteps=STEPS,
        max_timesteps=STEPS,
        ent_frac=0.8,
        plateau_rel=0.05,
        plateau_chunks=2,
        net_arch="64,64",
        output_folder=out,
        left_on_bonus=0.0,
        init_all_on=False,
    )
    cfg = load_config(args)
    cfg["nMmWaveEnb"] = [N_GNB]
    cfg["nUe"] = [60]
    cfg["ues"] = [60]
    cfg["gnbDistance"] = [120]
    cfg["ueClusterRadius"] = [25]
    cfg["rateLeft"] = ["2Mbps"]
    cfg["rateMid"] = ["2Mbps"]
    cfg["rateRight"] = ["2Mbps"]
    cfg["handoverMode"] = [HANDOVER]
    cfg["rlcAmEnabled"] = [0]

    prev_path = os.path.join(EX, "output_asm_state_s2_4enb_um", "compare_summary.json")
    with open(prev_path) as handle:
        summaries = json.load(handle)
    summaries.pop("ppo", None)

    train_dir = os.path.join(out, "ppo")
    final_zip = os.path.join(train_dir, f"ppo_{KIND}.zip")
    resume_from, resume_norm = (None, None)
    if not os.path.isfile(final_zip):
        resume_from, resume_norm = latest_checkpoint(train_dir)
    print(f"\n===== train kind={KIND} steps={STEPS} n_envs=8 nGnb=4 rlc=UM =====", flush=True)
    model_path, stats_path, train_s, converged, steps = train_ppo(
        args,
        cfg,
        train_dir,
        KIND,
        resume_from=resume_from,
        resume_vecnormalize=resume_norm,
    )
    init = policy_init_modes("ppo", N_GNB, KIND)
    print(
        f"\n===== eval ppo init={init} train_s={train_s:.1f} steps={steps} =====",
        flush=True,
    )
    rows, summary = eval_ppo(
        args,
        cfg,
        model_path,
        stats_path,
        os.path.join(out, "eval_ppo"),
        args.n_eval_episodes,
        KIND,
        init_modes=list(init),
    )
    summary["elapsed_s"] = train_s
    summary["timesteps"] = steps
    summary["converged"] = converged
    summary["handoverMode"] = HANDOVER
    summary["nUe"] = 60
    summary["nMmWaveEnb"] = N_GNB
    os.makedirs(os.path.join(out, "eval_ppo"), exist_ok=True)
    with open(os.path.join(out, "eval_ppo", "eval_steps.json"), "w") as handle:
        json.dump(rows, handle)
    summaries["ppo"] = summary
    with open(os.path.join(out, "compare_summary.json"), "w") as handle:
        json.dump(summaries, handle, indent=2)
    write_table(summaries, os.path.join(out, "compare_table.csv"))


if __name__ == "__main__":
    main()
