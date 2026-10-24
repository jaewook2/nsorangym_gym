#!/usr/bin/env python3
"""Train ASM-state PPO on scenario 2 with Threshold handover.

Same budget and scenario as asm_state_s2_ppo.py. Results are merged into
the existing Threshold fixed-policy table. The NoAuto run is left as-is.
"""
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))
from asm_ppo import (  # noqa: E402
    eval_ppo,
    load_config,
    policy_init_modes,
    train_ppo,
    write_table,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GYM = os.path.join(ROOT, "nsorangym_gym")
EX = os.path.join(GYM, "examples")
KIND = "asm"
HANDOVER = "Threshold"


def main():
    os.environ.setdefault("NS3_SKIP_BUILD", "1")
    out = os.path.join(EX, "output_asm_state_s2_threshold")
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
        simTime=3.3,
        seed=0,
        n_eval_episodes=2,
        total_timesteps=2048,
        n_steps=32,
        batch_size=16,
        n_envs=8,
        until_converge=False,
        chunk_timesteps=2048,
        max_timesteps=2048,
        ent_frac=0.8,
        plateau_rel=0.05,
        plateau_chunks=2,
        net_arch="64,64",
        output_folder=out,
        left_on_bonus=0.0,
        init_all_on=False,
    )
    cfg = load_config(args)
    cfg["handoverMode"] = [HANDOVER]
    train_dir = os.path.join(out, "ppo")
    print(f"\n===== train kind={KIND} handover={HANDOVER} =====", flush=True)
    model_path, stats_path, train_s, converged, steps = train_ppo(args, cfg, train_dir, KIND)
    init = policy_init_modes("ppo", 2, KIND)
    print(
        f"\n===== eval ppo init={init} handover={HANDOVER} train_s={train_s:.1f} steps={steps} =====",
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
    with open(os.path.join(out, "eval_ppo", "eval_steps.json"), "w") as handle:
        json.dump(rows, handle)

    summary_path = os.path.join(out, "compare_summary.json")
    with open(summary_path) as handle:
        summaries = json.load(handle)
    summaries["ppo"] = summary
    with open(summary_path, "w") as handle:
        json.dump(summaries, handle, indent=2)
    write_table(summaries, os.path.join(out, "compare_table.csv"))


if __name__ == "__main__":
    main()
