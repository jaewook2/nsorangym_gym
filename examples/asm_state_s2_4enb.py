#!/usr/bin/env python3
"""Four mmWave cells, 15 UEs each, 10 s, Threshold handover, RLC UM.

gNBs sit on a line from -120 m to 120 m, so adjacent cells are 80 m apart
and a UE cluster of radius 25 m can see its neighbor. Every cluster is
offered 2 Mbps. Comparison: all cells Active, alternate cells OFF, and
ASM PPO. The two-cell tables are left untouched.
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from asm_ppo import (  # noqa: E402
    eval_ppo,
    load_config,
    policy_init_modes,
    run_policy,
    train_ppo,
    write_table,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GYM = os.path.join(ROOT, "nsorangym_gym")
EX = os.path.join(GYM, "examples")
KIND = "asm"
HANDOVER = "Threshold"
N_GNB = 4


def main():
    os.environ.setdefault("NS3_SKIP_BUILD", "1")
    out = os.path.join(EX, "output_asm_state_s2_4enb_um")
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

    on = np.zeros(N_GNB, dtype=np.int64)
    alternate = np.array([0, 5, 0, 5], dtype=np.int64)
    policies = {
        "all_on": (lambda _obs, _rng: on.copy(), [1, 1, 1, 1]),
        "alternate": (lambda _obs, _rng: alternate.copy(), [1, 6, 1, 6]),
    }
    summaries = {}
    all_rows = []
    for name, (action, init) in policies.items():
        print(f"\n===== {name} init={init} handover={HANDOVER} nGnb=4 sim=10s =====", flush=True)
        rows, summary = run_policy(
            name,
            action,
            args,
            cfg,
            os.path.join(out, name),
            args.n_eval_episodes,
            args.seed,
            KIND,
            init_modes=list(init),
        )
        summary["handoverMode"] = HANDOVER
        summary["nUe"] = 60
        summary["nMmWaveEnb"] = N_GNB
        summaries[name] = summary
        all_rows.extend(rows)
        with open(os.path.join(out, name, "eval_steps.json"), "w") as handle:
            json.dump(rows, handle)
        print(f"===== {name} return={summary.get('ep_return_mean')} =====", flush=True)
    with open(os.path.join(out, "compare_summary.json"), "w") as handle:
        json.dump(summaries, handle, indent=2)
    with open(os.path.join(out, "eval_steps.json"), "w") as handle:
        json.dump(all_rows, handle)
    write_table(summaries, os.path.join(out, "compare_table.csv"))

    train_dir = os.path.join(out, "ppo")
    print(f"\n===== train kind={KIND} handover={HANDOVER} nGnb=4 =====", flush=True)
    model_path, stats_path, train_s, converged, steps = train_ppo(args, cfg, train_dir, KIND)
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
    with open(os.path.join(out, "eval_ppo", "eval_steps.json"), "w") as handle:
        json.dump(rows, handle)
    summaries["ppo"] = summary
    with open(os.path.join(out, "compare_summary.json"), "w") as handle:
        json.dump(summaries, handle, indent=2)
    write_table(summaries, os.path.join(out, "compare_table.csv"))


if __name__ == "__main__":
    main()
