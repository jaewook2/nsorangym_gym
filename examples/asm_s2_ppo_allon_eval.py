#!/usr/bin/env python3
"""Eval saved scenario-2 PPO models from an all-ON start. No training.

init_modes=None keeps both cells at ASM mode 1 after the warmup, so the
first scored action is sequential from mode 1 instead of jumping to 2-4.
"""
import csv
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))
from asm_ppo import eval_ppo, load_config  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GYM = os.path.join(ROOT, "nsorangym_gym")
EX = os.path.join(GYM, "examples")

RUNS = (
    ("2048", "output_asm_s2_bestinit"),
    ("6144", "output_asm_s2_bestinit_x3"),
    ("12288", "output_asm_s2_bestinit_x6"),
    ("18432", "output_asm_s2_bestinit_t18432"),
    ("36864", "output_asm_s2_bestinit_t36864"),
)


def main():
    out = os.path.join(EX, "output_asm_s2_ppo_allon_eval")
    os.makedirs(out, exist_ok=True)
    args = SimpleNamespace(
        config=os.path.join(
            GYM, "src/environments/scenario_configurations/asm_s2_asm_isolated.json"
        ),
        ns3_path=os.path.join(ROOT, "nsorangym_mmwave"),
        optimized=False,
        skip_configuration=True,
        omega=0.0,
        kind="duty_mode",
        simTime=3.3,
        seed=0,
        n_eval_episodes=2,
    )
    cfg = load_config(args)
    rows_out = []
    for steps, folder in RUNS:
        model = os.path.join(EX, folder, "asm", "ppo", "ppo_asm.zip")
        stats = os.path.join(EX, folder, "asm", "ppo", "vecnormalize.pkl")
        log_dir = os.path.join(out, f"t{steps}")
        print(f"\n===== ppo all-on start steps={steps} =====", flush=True)
        rows, summary = eval_ppo(
            args, cfg, model, stats, log_dir, args.n_eval_episodes, "duty_mode", init_modes=None
        )
        summary = dict(summary)
        summary["timesteps"] = int(steps)
        summary["policy"] = f"ppo_{steps}"
        rows_out.append(summary)
        for row in rows:
            row["timesteps"] = int(steps)
        with open(os.path.join(log_dir, "eval_steps.json"), "w") as handle:
            json.dump(rows, handle)
        print(
            f"===== ppo_{steps} return={summary.get('ep_return_mean')} =====",
            flush=True,
        )
    path = os.path.join(out, "compare_table.csv")
    fields = [
        "policy",
        "timesteps",
        "ep_return_mean",
        "reward",
        "eeNorm",
        "satP5",
        "appKbps",
        "pMmw",
        "pLte",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
