#!/usr/bin/env python3
"""Eval scenario-2 baseline policies from an ASM 2-4 start. No training.

Every policy is forced to modes [2, 4] after warmup. Later actions stay
sequential, so a policy that wants another pair has to walk there.
"""
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))
from asm_ppo import load_config, policy_actions, run_policy, write_table  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GYM = os.path.join(ROOT, "nsorangym_gym")
EX = os.path.join(GYM, "examples")
INIT = [2, 4]


def main():
    out = os.path.join(EX, "output_asm_s2_baselines_init24")
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
    policies = policy_actions(2, "duty_mode")
    summaries = {}
    all_rows = []
    for name, choose in policies.items():
        log_dir = os.path.join(out, name)
        print(f"\n===== baseline {name} init={INIT} =====", flush=True)
        rows, summary = run_policy(
            name,
            choose,
            args,
            cfg,
            log_dir,
            args.n_eval_episodes,
            args.seed,
            "duty_mode",
            init_modes=list(INIT),
        )
        summaries[name] = summary
        all_rows.extend(rows)
        with open(os.path.join(log_dir, "eval_steps.json"), "w") as handle:
            json.dump(rows, handle)
        print(
            f"===== {name} return={summary.get('ep_return_mean')} =====",
            flush=True,
        )
    with open(os.path.join(out, "compare_summary.json"), "w") as handle:
        json.dump(summaries, handle, indent=2)
    with open(os.path.join(out, "eval_steps.json"), "w") as handle:
        json.dump(all_rows, handle)
    write_table(summaries, os.path.join(out, "compare_table.csv"))


if __name__ == "__main__":
    main()
