#!/usr/bin/env python3
"""Short gym <-> LENA handshake test (semaphores + control file)."""

import argparse
import json
import sys
import traceback

from environments.lena_ts_env import LenaTrafficSteeringEnv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LENA Traffic Steering gym test")
    parser.add_argument(
        "--config",
        type=str,
        default="../src/environments/scenario_configurations/lena_ts_use_case.json",
    )
    parser.add_argument("--output_folder", type=str, default="output_lena")
    parser.add_argument("--ns3_path", type=str, default="../../ns-3-dev")
    parser.add_argument("--num_steps", type=int, default=3)
    parser.add_argument("--optimized", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--configure", action="store_true")
    parser.add_argument("--scenario", type=str, default="lena-oran-test")
    args = parser.parse_args()

    with open(args.config) as f:
        scenario_configuration = json.load(f)

    print("Creating LenaTrafficSteeringEnv")
    env = LenaTrafficSteeringEnv(
        ns3_path=args.ns3_path,
        scenario_configuration=scenario_configuration,
        output_folder=args.output_folder,
        optimized=args.optimized,
        scenario=args.scenario,
        verbose=args.verbose,
        skip_configuration=not args.configure,
    )
    print(f"  executable={env.script_executable}")
    print(f"  ns3_params={env.scenario_configuration}")

    try:
        print("reset() ...", flush=True)
        obs, info = env.reset()
        print(f"reset OK  obs_shape={getattr(obs, 'shape', type(obs))}")

        for i in range(args.num_steps):
            action = env.action_space.sample() * 0  # no-op actions
            print(f"step {i} ...", flush=True)
            obs, reward, terminated, truncated, info = env.step(action)
            print(
                f"step {i} OK  reward={reward} terminated={terminated} truncated={truncated}",
                flush=True,
            )
            if terminated or truncated:
                print("episode ended")
                break
        print("TEST PASS")
    except Exception as exc:
        print("TEST FAIL:", exc, file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    finally:
        try:
            env.close()
        except Exception:
            pass
