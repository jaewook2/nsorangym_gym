#!/usr/bin/env python3
"""Duty-cycle control on scenario-asm-verify.

Levels 1-6 are data-symbol duty: 100, 75, 50, 25, 0 percent, then TurnOff.
This is not the ASM component state machine. Use asm.py for that.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from environments.asm_env import AsmEnv

MODE_NAME = {
    1: "100% mmWave data",
    2: "75% mmWave data",
    3: "50% mmWave data",
    4: "25% mmWave data",
    5: "0% mmWave data (cell on)",
    6: "TurnOff / LTE fallback",
}


def main():
    parser = argparse.ArgumentParser(description="Random per-gNB duty-cycle example")
    parser.add_argument(
        "--config",
        default=os.path.join(
            os.path.dirname(__file__),
            "../src/environments/scenario_configurations/asm_use_case.json",
        ),
    )
    parser.add_argument("--output_folder", default="output_duty")
    parser.add_argument(
        "--ns3_path",
        default=os.environ.get(
            "NS3_PATH",
            "/home/pknu-boo/nsorangym_mmwave_test_0914/nsorangym_mmwave",
        ),
    )
    parser.add_argument(
        "--kind",
        choices=("on_off", "duty_mode", "duty"),
        default="duty_mode",
        help="on_off jumps between 100%% and TurnOff. duty_mode changes duty by at most one level.",
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
        kind=args.kind,
    )
    rng = np.random.default_rng(args.seed)
    print("Launch reset ", end="", flush=True)
    obs, info = env.reset()
    print("done", flush=True)
    print(f"First obs {dict(zip(env.COLUMNS, obs.tolist()))}")
    print(f"kind={env.kind} sequential={env.sequential} cells {env.mmwave_cell_ids} "
          f"ues={env.n_ues} offeredKbps={env.offered_kbps} sim_path {env.sim_path}")

    matches = 0
    evaluated = 0
    prev_modes = None
    try:
        for step in range(1, args.num_steps + 1):
            if env.kind == "on_off":
                action = rng.integers(0, 2, size=env.n_gnbs)
            elif prev_modes is None:
                action = np.zeros(env.n_gnbs, dtype=int)
            else:
                action = np.array(
                    [int(np.clip(int(m) - 1 + int(rng.integers(-1, 2)), 0, 5)) for m in prev_modes],
                    dtype=int,
                )
            modes = env._modes_from_action(action)
            pretty = ", ".join(
                f"cell {cid} duty {mode} ({MODE_NAME[mode]})"
                for cid, mode in zip(env.mmwave_cell_ids, modes)
            )
            print(f"Step {step} {pretty} ", end="", flush=True)
            obs, reward, terminated, truncated, info = env.step(action)
            print("done", flush=True)
            metrics = env.last_metrics
            applied = metrics.get("appliedModes", {})
            tput_modes = metrics.get("tputModes", {})
            commanded = dict(zip(env.mmwave_cell_ids, modes))
            cell_ok = True
            for cid, mode in commanded.items():
                seen = applied.get(cid)
                if seen is None:
                    seen = tput_modes.get(cid, 1)
                cell_ok = cell_ok and int(seen) == mode
            if commanded:
                evaluated += 1
                matches += int(cell_ok)
            print(
                f"  commanded={commanded} applied={applied} tput={tput_modes} "
                f"nUe={metrics.get('nUe')} nUeLte={metrics.get('nUeLte')} "
                f"appKbps={metrics.get('appKbps', 0):.1f} "
                f"eeNorm={metrics.get('eeNorm', 0):.3f} "
                f"pMmw={metrics.get('pMmw', 0):.2f} "
                f"pLte={metrics.get('pLte', 0):.2f} "
                f"satP5={metrics.get('satP5', 0):.3f} "
                f"reward={reward:.3f} match={cell_ok} bothOff={metrics.get('bothOff')}"
            )
            print(f"  terminated={terminated} truncated={truncated} ts={env.last_timestamp}")
            prev_modes = modes
            if terminated or truncated:
                break
    finally:
        env.close()

    print(f"Mode match {matches}/{evaluated}")
    if evaluated == 0 or matches < evaluated:
        sys.exit(1)


if __name__ == "__main__":
    main()
