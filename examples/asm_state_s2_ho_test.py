#!/usr/bin/env python3
"""Scenario-2 ASM comparison with automatic secondary-cell handover.

Same placement, traffic, and policies as asm_state_s2_test.py. The only
change is SecondaryCellHandoverMode=Threshold, so a UE on LTE can return
to mmWave once it leaves outage. NoAuto results stay in
output_asm_state_s2_test.
"""
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))
from asm_ppo import load_config, policy_actions, policy_init_modes, run_policy, write_table

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GYM = os.path.join(ROOT, "nsorangym_gym")
EX = os.path.join(GYM, "examples")
KIND = "asm"
HANDOVER = "Threshold"
POLICIES = ("all_on", "one_on_cell2", "one_on_cell3")


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
    )
    cfg = load_config(args)
    cfg["handoverMode"] = [HANDOVER]
    actions = policy_actions(2, KIND)
    summaries = {}
    all_rows = []
    for name in POLICIES:
        init = policy_init_modes(name, 2, KIND)
        print(f"\n===== {name} init={init} kind={KIND} handover={HANDOVER} =====", flush=True)
        rows, summary = run_policy(
            name,
            actions[name],
            args,
            cfg,
            os.path.join(out, name),
            args.n_eval_episodes,
            args.seed,
            KIND,
            init_modes=list(init),
        )
        summary["handoverMode"] = HANDOVER
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


if __name__ == "__main__":
    main()
