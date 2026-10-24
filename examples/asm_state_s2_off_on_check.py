#!/usr/bin/env python3
"""Turn the right mmWave cell off, then back on, under two HO modes.

Cell 2 stays Active. Cell 3 goes Active -> OFF -> Active. One episode each
for NoAuto and Threshold. Prints where the cell-3 UEs are before the off,
after the off has settled, and after the cell is back on.
"""
import csv
import glob
import os
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from asm_ppo import load_config, make_env

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GYM = os.path.join(ROOT, "nsorangym_gym")
EX = os.path.join(GYM, "examples")
KIND = "asm"
ON = np.array([0, 0], dtype=np.int64)
OFF = np.array([0, 5], dtype=np.int64)
# 100 ms steps after the 0.5 s all-on warmup.
HOLD_ON = 10
HOLD_OFF = 30


def choose(step):
    if step < HOLD_ON:
        return ON.copy()
    if step < HOLD_ON + HOLD_OFF:
        return OFF.copy()
    return ON.copy()


def latest_sim(log_dir):
    paths = sorted(glob.glob(os.path.join(log_dir, "ns3", "env0", "*", "ue-app-sat.csv")))
    if not paths:
        raise FileNotFoundError(f"no ue-app-sat.csv under {log_dir}")
    return os.path.dirname(paths[-1])


def load_sat(path):
    by_t = {}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            t = int(row["timeMs"])
            by_t.setdefault(t, []).append(
                {
                    "imsi": int(row["imsi"]),
                    "cell": int(row["servingNrCellId"]),
                    "mmw": int(row["useMmWave"]),
                }
            )
    return by_t


def nearest(times, target):
    return min(times, key=lambda t: abs(t - target))


def snapshot(by_t, t, imsis):
    rows = [r for r in by_t[t] if r["imsi"] in imsis]
    return Counter((r["cell"], r["mmw"]) for r in rows), len(rows)


def report(mode, sim_dir):
    stderr_path = os.path.join(sim_dir, "stderr")
    events = []
    with open(stderr_path) as handle:
        for line in handle:
            if any(
                key in line
                for key in (
                    "ASM target",
                    "ASM 6 TurnOff",
                    "leave ASM 6",
                    "resume user-plane",
                    "switch user-plane to LTE",
                    "ASM ",
                )
            ):
                if "mmWave cell" in line or "IMSI" in line:
                    events.append(line.strip())
    print(f"\n----- {mode} events -----", flush=True)
    for line in events:
        if "ASM target" in line or "TurnOff" in line or "leave ASM" in line or "ASM " in line and "->" in line:
            print(line, flush=True)
    resumes = [line for line in events if "resume user-plane" in line]
    lte_switches = [line for line in events if "switch user-plane to LTE" in line]
    print(f"lte_switches={len(lte_switches)} resumes={len(resumes)}", flush=True)
    for line in resumes:
        print(line, flush=True)

    by_t = load_sat(os.path.join(sim_dir, "ue-app-sat.csv"))
    times = sorted(by_t)
    # Home cell is the cell each UE is on while both cells are still up.
    home_t = nearest(times, 1400)
    home = {}
    for row in by_t[home_t]:
        if row["cell"] in (2, 3) and row["mmw"] == 1:
            home[row["imsi"]] = row["cell"]
    right = {imsi for imsi, cell in home.items() if cell == 3}
    print(f"home_t={home_t} cell2={sum(c==2 for c in home.values())} cell3={len(right)}", flush=True)

    off_done = None
    on_done = None
    changes = os.path.join(sim_dir, "asm-mode-changes.txt")
    if os.path.exists(changes):
        with open(changes) as handle:
            for row in csv.DictReader(handle):
                if int(row["cellId"]) != 3:
                    continue
                t = int(row["timeMs"])
                new = int(row["newMode"])
                print(f"cell3 mode {row['oldMode']} -> {new} at {t} ms", flush=True)
                if new == 6:
                    off_done = t
                if off_done is not None and new == 1 and t > off_done:
                    on_done = t
    marks = [("before_off", (off_done or 2000) - 200)]
    if off_done is not None:
        marks.append(("after_off", off_done + 1000))
    if on_done is not None:
        marks.append(("just_on", on_done + 200))
        marks.append(("after_on", max(times)))
    else:
        marks.append(("end", max(times)))
    for label, target in marks:
        t = nearest(times, target)
        counts, n = snapshot(by_t, t, right)
        print(f"{label} t={t} n={n} (cell,useMmWave)={dict(counts)}", flush=True)


def run_one(mode, out):
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
        simTime=7.0,
        seed=0,
    )
    cfg = load_config(args)
    cfg["handoverMode"] = [mode]
    env = make_env(args, cfg, out, KIND, init_modes=None)()
    print(f"\n===== {mode} cell3 off then on =====", flush=True)
    try:
        obs, _info = env.reset()
        done = False
        step = 0
        while not done:
            action = choose(step)
            obs, reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)
            step += 1
            if step % 5 == 0 or done:
                print(f"  {mode} step={step} action={action.tolist()} reward={reward:.3f}", flush=True)
    finally:
        sim_dir = latest_sim(out)
        env.close()
    print(f"sim_dir={sim_dir}", flush=True)
    report(mode, sim_dir)


def main():
    os.environ.setdefault("NS3_SKIP_BUILD", "1")
    base = os.path.join(EX, "output_asm_state_s2_off_on")
    for mode in ("NoAuto", "Threshold"):
        run_one(mode, os.path.join(base, mode))


if __name__ == "__main__":
    main()
