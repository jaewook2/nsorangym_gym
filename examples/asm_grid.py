#!/usr/bin/env python3
"""Static ASM pair sweep: all left/right modes 1-6 except 6-6. No PPO."""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from asm_ppo import load_config, run_policy, write_table


def parse_args():
    parser = argparse.ArgumentParser(description="Static ASM 1-6 x 1-6 grid (skip 6-6)")
    parser.add_argument(
        "--config",
        default=os.path.join(
            os.path.dirname(__file__),
            "../src/environments/scenario_configurations/asm_s2_asm_isolated.json",
        ),
    )
    parser.add_argument("--output_folder", default="examples/output_asm_s2_static_grid")
    parser.add_argument(
        "--ns3_path",
        default=os.environ.get(
            "NS3_PATH",
            "/home/pknu-boo/nsorangym_mmwave_test_0914/nsorangym_mmwave",
        ),
    )
    parser.add_argument("--omega", type=float, default=0.0)
    parser.add_argument("--n_eval_episodes", type=int, default=1)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--simTime", type=float, default=None)
    parser.add_argument("--skip_configuration", action="store_true", default=True)
    parser.add_argument("--optimized", action="store_true")
    return parser.parse_args()


def mode_pairs():
    pairs = []
    for left in range(1, 7):
        for right in range(1, 7):
            if left == 6 and right == 6:
                continue
            pairs.append((left, right))
    return pairs


def eval_pair(payload: dict):
    left = int(payload["left"])
    right = int(payload["right"])
    name = f"{left}-{right}"
    action = np.array([left - 1, right - 1], dtype=np.int64)
    env_args = SimpleNamespace(
        ns3_path=payload["ns3_path"],
        omega=payload["omega"],
        skip_configuration=payload["skip_configuration"],
        optimized=payload["optimized"],
        kind="asm",
    )
    rows, summary = run_policy(
        name,
        lambda _obs, _rng, a=action: a.copy(),
        env_args,
        payload["cfg"],
        os.path.join(payload["output_folder"], f"eval_{name}"),
        payload["n_eval_episodes"],
        payload["seed"],
        "asm",
    )
    return name, rows, summary


def print_matrix(summaries: dict, key: str):
    print(f"\n===== {key} matrix (rows=left ASM, cols=right ASM) =====", flush=True)
    header = "     " + "".join(f"{c:>10}" for c in range(1, 7))
    print(header, flush=True)
    for left in range(1, 7):
        row = f"{left:<5}"
        for right in range(1, 7):
            if left == 6 and right == 6:
                row += f"{'skip':>10}"
                continue
            name = f"{left}-{right}"
            val = (summaries.get(name) or {}).get(key)
            row += f"{'n/a':>10}" if val is None else f"{val:10.3f}"
        print(row, flush=True)


def main():
    args = parse_args()
    os.environ.setdefault("NS3_SKIP_BUILD", "1")
    cfg = load_config(args)
    os.makedirs(args.output_folder, exist_ok=True)
    env_args = SimpleNamespace(
        ns3_path=args.ns3_path,
        omega=args.omega,
        skip_configuration=args.skip_configuration,
        optimized=args.optimized,
        kind="asm",
    )
    pairs = mode_pairs()
    n_workers = max(1, min(int(args.n_workers), len(pairs)))
    print(
        f"===== static ASM grid n={len(pairs)} skip=6-6 "
        f"episodes={args.n_eval_episodes} workers={n_workers} omega={args.omega} =====",
        flush=True,
    )
    payloads = [
        {
            "left": left,
            "right": right,
            "ns3_path": args.ns3_path,
            "omega": args.omega,
            "skip_configuration": args.skip_configuration,
            "optimized": args.optimized,
            "cfg": cfg,
            "output_folder": args.output_folder,
            "n_eval_episodes": args.n_eval_episodes,
            "seed": args.seed,
        }
        for left, right in pairs
    ]
    summaries = {}
    all_rows = []
    done = 0
    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futs = [pool.submit(eval_pair, payload) for payload in payloads]
        for fut in as_completed(futs):
            name, rows, summary = fut.result()
            summaries[name] = summary
            all_rows.extend(rows)
            done += 1
            print(f"----- done {done}/{len(pairs)} {name} return={summary.get('ep_return_mean')} -----", flush=True)
            ordered = {f"{a}-{b}": summaries[f"{a}-{b}"] for a, b in pairs if f"{a}-{b}" in summaries}
            with open(os.path.join(args.output_folder, "compare_summary.json"), "w") as handle:
                json.dump(ordered, handle, indent=2)
            write_table(ordered, os.path.join(args.output_folder, "compare_table.csv"))

    with open(os.path.join(args.output_folder, "eval_steps.json"), "w") as handle:
        json.dump(all_rows, handle)
    ranked = sorted(
        summaries.items(),
        key=lambda item: (item[1].get("ep_return_mean") is None, -(item[1].get("ep_return_mean") or 0)),
    )
    print("\n===== ranked by return =====", flush=True)
    for name, stats in ranked:
        print(
            f"{name:<8} return={stats.get('ep_return_mean'):.3f} "
            f"eeNorm={stats.get('eeNorm'):.3f} appKbps={stats.get('appKbps'):.1f} "
            f"satP5={stats.get('satP5'):.3f}",
            flush=True,
        )
    print_matrix(summaries, "ep_return_mean")
    print_matrix(summaries, "appKbps")
    print(f"wrote {args.output_folder}", flush=True)


if __name__ == "__main__":
    main()
