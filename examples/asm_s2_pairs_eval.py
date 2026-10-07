#!/usr/bin/env python3
"""Evaluate a subset of static ASM pairs on scenario 2."""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from asm_ppo import load_config, run_policy, write_table


def pairs():
    chosen = [(1, right) for right in range(2, 7)]
    chosen += [(left, right) for left in range(2, 6) for right in range(1, 7)]
    return chosen


def eval_pair(payload):
    left = int(payload["left"])
    right = int(payload["right"])
    name = f"{left}-{right}"
    action = np.array([left - 1, right - 1], dtype=np.int64)
    rows, summary = run_policy(
        name,
        lambda _obs, _rng, a=action: a.copy(),
        payload["args"],
        payload["cfg"],
        os.path.join(payload["out"], f"eval_{name}"),
        2,
        0,
        "asm",
        init_modes=[left, right],
    )
    return name, rows, summary


def main():
    out = "examples/output_asm_s2_asm_pairs"
    os.makedirs(out, exist_ok=True)
    args = SimpleNamespace(
        config="src/environments/scenario_configurations/asm_s2_asm_isolated.json",
        simTime=3.3,
        ns3_path="/home/pknu-boo/nsorangym_mmwave_test_0914/nsorangym_mmwave",
        optimized=False,
        skip_configuration=True,
        omega=0.0,
        kind="asm",
    )
    cfg = load_config(args)
    chosen = pairs()
    print(f"===== asm pairs n={len(chosen)} workers=16 episodes=2 =====", flush=True)
    payloads = [
        {"left": left, "right": right, "args": args, "cfg": cfg, "out": out}
        for left, right in chosen
    ]
    summaries = {}
    all_rows = []
    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=16, mp_context=ctx) as pool:
        futs = [pool.submit(eval_pair, item) for item in payloads]
        done = 0
        for fut in as_completed(futs):
            name, rows, summary = fut.result()
            summaries[name] = summary
            all_rows.extend(rows)
            done += 1
            ret = summary.get("ep_return_mean")
            print(f"----- done {done}/{len(chosen)} {name} return={ret} -----", flush=True)
    ordered = {f"{a}-{b}": summaries[f"{a}-{b}"] for a, b in chosen if f"{a}-{b}" in summaries}
    with open(os.path.join(out, "compare_summary.json"), "w") as handle:
        json.dump(ordered, handle, indent=2)
    with open(os.path.join(out, "eval_steps.json"), "w") as handle:
        json.dump(all_rows, handle)
    write_table(ordered, os.path.join(out, "compare_table.csv"))
    print("pairs eval done", flush=True)


if __name__ == "__main__":
    main()
