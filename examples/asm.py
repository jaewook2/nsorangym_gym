#!/usr/bin/env python3
"""ASM component state machine on scenario-asm-verify.

Actions choose a target: Active, ASM1, ASM2, ASM3, ASM4, or OFF.
ns-3 turns PA, RF, baseband, and control on or off in order.
Duty-cycle control is duty.py, not this script.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from environments.asm_env import AsmEnv

MODE_NAME = {
    1: "Active",
    2: "ASM1",
    3: "ASM2",
    4: "ASM3",
    5: "ASM4",
    6: "OFF",
}


def main():
    parser = argparse.ArgumentParser(description="Random per-gNB ASM state-machine example")
    parser.add_argument(
        "--config",
        default=os.path.join(
            os.path.dirname(__file__),
            "../src/environments/scenario_configurations/asm_use_case.json",
        ),
    )
    parser.add_argument("--output_folder", default="output_asm_state")
    parser.add_argument(
        "--ns3_path",
        default=os.environ.get(
            "NS3_PATH",
            "/home/pknu-boo/nsorangym_mmwave_test_0914/nsorangym_mmwave",
        ),
    )
    parser.add_argument("--num_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optimized", action="store_true")
    parser.add_argument("--skip_configuration", action="store_true", default=True)
    parser.add_argument("--no_skip_configuration", action="store_false", dest="skip_configuration")
    args = parser.parse_args()

    with open(args.config) as params_file:
        scenario_configuration = json.load(params_file)

    env = AsmEnv(
        ns3_path=args.ns3_path,
        scenario_configuration=scenario_configuration,
        output_folder=args.output_folder,
        optimized=args.optimized,
        skip_configuration=args.skip_configuration,
        verbose=True,
        kind="asm",
    )
    rng = np.random.default_rng(args.seed)
    print("Launch reset ", end="", flush=True)
    obs, info = env.reset()
    print("done", flush=True)
    print(f"First obs {dict(zip(env.COLUMNS, obs.tolist()))}")
    print(f"kind={env.kind} cells {env.mmwave_cell_ids} "
          f"ues={env.n_ues} offeredKbps={env.offered_kbps} sim_path {env.sim_path}")
    if env.kind != "asm":
        print("ASM example did not enable the state machine", flush=True)
        env.close()
        sys.exit(1)

    completed = 0
    saw_target = False
    try:
        for step in range(1, args.num_steps + 1):
            action = rng.integers(0, 6, size=env.n_gnbs)
            modes = env._modes_from_action(action)
            pretty = ", ".join(
                f"cell {cid} target {mode} ({MODE_NAME[mode]})"
                for cid, mode in zip(env.mmwave_cell_ids, modes)
            )
            print(f"Step {step} {pretty} ", end="", flush=True)
            obs, reward, terminated, truncated, info = env.step(action)
            print("done", flush=True)
            metrics = env.last_metrics
            state = metrics.get("asmState") or {}
            print(
                f"  target={dict(zip(env.mmwave_cell_ids, modes))} "
                f"state={state} applied={metrics.get('appliedModes')} "
                f"nUe={metrics.get('nUe')} nUeLte={metrics.get('nUeLte')} "
                f"appKbps={metrics.get('appKbps', 0):.1f} "
                f"asmPowerW={metrics.get('asmPowerW', 0):.3f} "
                f"reward={reward:.3f}"
            )
            print(f"  terminated={terminated} truncated={truncated} ts={env.last_timestamp}")
            completed += 1
            saw_target = saw_target or any(
                str(key).startswith("target") and float(value) > 0.0
                for key, value in metrics.items()
            )
            if terminated or truncated:
                break
    finally:
        env.close()

    print(f"ASM steps {completed} target_observed={saw_target}")
    if completed == 0 or not saw_target:
        sys.exit(1)


if __name__ == "__main__":
    main()
