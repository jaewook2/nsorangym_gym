#!/usr/bin/env python3
"""Minimal launcher for LenaTrafficSteeringEnv (5gnr_gym stub)."""

import argparse
import json
from environments.lena_ts_env import LenaTrafficSteeringEnv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LENA Traffic Steering stub env")
    parser.add_argument(
        "--config",
        type=str,
        default="../src/environments/scenario_configurations/lena_ts_use_case.json",
    )
    parser.add_argument("--output_folder", type=str, default="output_lena")
    parser.add_argument(
        "--ns3_path",
        type=str,
        default="../../ns-3-dev",
        help="Path to ns-3-dev (with contrib/nr LENA)",
    )
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--optimized", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--scenario",
        type=str,
        default="lena-oran-test",
        help="ns-3 scratch/program name",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        scenario_configuration = json.load(f)

    print("Creating LenaTrafficSteeringEnv (stub)")
    env = LenaTrafficSteeringEnv(
        ns3_path=args.ns3_path,
        scenario_configuration=scenario_configuration,
        output_folder=args.output_folder,
        optimized=args.optimized,
        scenario=args.scenario,
        verbose=args.verbose,
    )
    print("Note: full step loop needs semaphore + LENA KPM wiring; env object created OK.")
    print(f"  control_file={env.control_file} scenario={env.scenario}")
    print(f"  action_space={env.action_space} observation_space={env.observation_space}")
