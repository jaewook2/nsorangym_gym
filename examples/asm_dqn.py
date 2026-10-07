#!/usr/bin/env python3
"""Train DQN on the ON/OFF ASM scenario with parallel ns-3 environments.

SB3 DQN accepts only Discrete actions. Each mmWave cell is on or off, so the
joint action is one integer in 0 .. 2**n_cells - 1.
"""
import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(__file__))
from asm_ppo import (  # noqa: E402
    load_config,
    make_env,
    parse_args,
    policy_actions,
    policy_init_modes,
    run_policy,
    unwrap_metrics,
    write_table,
)


class JointOnOff(gym.ActionWrapper):
    """Map Discrete(2**n) onto the env's per-cell MultiDiscrete([2] * n)."""

    def __init__(self, env):
        super().__init__(env)
        nvec = np.asarray(env.action_space.nvec, dtype=int)
        if np.any(nvec != 2):
            raise ValueError(f"JointOnOff expects binary cells, got {nvec}")
        self.n_cells = int(nvec.size)
        self.action_space = spaces.Discrete(2**self.n_cells)

    def action(self, act):
        value = int(np.asarray(act).reshape(-1)[0])
        cells = [(value >> i) & 1 for i in range(self.n_cells)]
        return np.asarray(cells, dtype=np.int64)

    @property
    def last_metrics(self):
        return unwrap_metrics(self.env)


def make_dqn_env(args, cfg, log_dir, kind, rank, init_modes):
    def _init():
        env = make_env(args, cfg, log_dir, kind, rank=rank, init_modes=init_modes)()
        return JointOnOff(env)

    return _init


def train_dqn(args, cfg, log_dir, kind):
    import time

    from stable_baselines3 import DQN
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    os.makedirs(log_dir, exist_ok=True)
    model_path = os.path.join(log_dir, f"dqn_{kind}.zip")
    n_envs = max(1, int(args.n_envs))
    hidden = [int(part) for part in str(args.net_arch).split(",") if part.strip()]
    n_gnbs = int(cfg.get("nMmWaveEnb", [3])[0])
    init_modes = None if args.init_all_on else policy_init_modes("ppo", n_gnbs, kind)

    def _worker(rank):
        return make_dqn_env(args, cfg, log_dir, kind, rank, init_modes)

    if n_envs == 1:
        vec = DummyVecEnv([_worker(0)])
    else:
        vec = SubprocVecEnv([_worker(i) for i in range(n_envs)], start_method="fork")
    model = DQN(
        "MlpPolicy",
        vec,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=100,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=hidden),
        seed=args.seed,
        device="cpu",
        verbose=1,
        tensorboard_log=os.path.join(args.output_folder, "tb", kind),
    )
    print(
        f"\n===== train DQN {kind} n_envs={n_envs} budget={args.total_timesteps} "
        f"net={hidden} actions={vec.action_space.n} eps_frac=0.5 lr=1e-4 "
        f"omega={args.omega} =====",
        flush=True,
    )
    t0 = time.time()
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
    model.save(model_path)
    vec.close()
    elapsed = time.time() - t0
    print(f"saved {model_path} elapsed_s={elapsed:.1f}", flush=True)
    return model_path, elapsed


def eval_dqn(args, cfg, model_path, log_dir, n_episodes, kind, init_modes=None):
    from stable_baselines3 import DQN
    from stable_baselines3.common.vec_env import DummyVecEnv

    from asm_ppo import summarize_steps, vec_metrics

    os.makedirs(log_dir, exist_ok=True)
    vec = DummyVecEnv([make_dqn_env(args, cfg, log_dir, kind, 0, init_modes)])
    model = DQN.load(model_path, env=vec, device="cpu")
    rows = []
    ep_rewards = []
    print(f"\n===== eval dqn episodes={n_episodes} =====", flush=True)
    try:
        for ep in range(n_episodes):
            obs = vec.reset()
            done = False
            step = 0
            ep_r = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, _infos = vec.step(action)
                reward = float(rewards[0])
                done = bool(dones[0])
                metrics = vec_metrics(vec)
                rows.append(
                    {
                        "policy": "dqn",
                        "episode": ep,
                        "step": step,
                        "reward": reward,
                        "eeNorm": metrics.get("eeNorm"),
                        "satP5": metrics.get("satP5"),
                        "appKbps": metrics.get("appKbps"),
                        "pMmw": metrics.get("pMmw"),
                        "pLte": metrics.get("pLte"),
                        "power": metrics.get("power"),
                        "appliedModes": str(metrics.get("appliedModes")),
                    }
                )
                ep_r += reward
                step += 1
                print(
                    f"  dqn ep={ep} step={step} reward={reward:.3f} "
                    f"eeNorm={metrics.get('eeNorm', 0):.3f} satP5={metrics.get('satP5', 0):.3f} "
                    f"appKbps={metrics.get('appKbps', 0):.1f} modes={metrics.get('appliedModes')}",
                    flush=True,
                )
            ep_rewards.append(ep_r)
            print(f"  dqn episode {ep} return={ep_r:.3f} steps={step}", flush=True)
    finally:
        vec.close()
    summary = summarize_steps(rows)
    summary["ep_return_mean"] = float(np.mean(ep_rewards)) if ep_rewards else None
    summary["ep_return_std"] = float(np.std(ep_rewards)) if ep_rewards else None
    return rows, summary


def main():
    import json

    args = parse_args()
    os.environ.setdefault("NS3_SKIP_BUILD", "1")
    cfg = load_config(args)
    os.makedirs(args.output_folder, exist_ok=True)
    kind = "on_off"
    kind_dir = os.path.join(args.output_folder, kind)
    train_dir = os.path.join(kind_dir, "dqn")
    model_path, train_s = train_dqn(args, cfg, train_dir, kind)
    summaries = {"train": {"elapsed_s": train_s, "timesteps": args.total_timesteps}}
    all_rows = []
    n_gnbs = int(cfg.get("nMmWaveEnb", [3])[0])
    for name, choose in policy_actions(n_gnbs, kind).items():
        rows, summary = run_policy(
            name,
            choose,
            args,
            cfg,
            os.path.join(kind_dir, f"eval_{name}"),
            args.n_eval_episodes,
            args.seed,
            kind,
            init_modes=None if args.init_all_on else policy_init_modes(name, n_gnbs, kind),
        )
        all_rows.extend(rows)
        summaries[name] = summary
    rows, summary = eval_dqn(
        args,
        cfg,
        model_path,
        os.path.join(kind_dir, "eval_dqn"),
        args.n_eval_episodes,
        kind,
        init_modes=None if args.init_all_on else policy_init_modes("all_on", n_gnbs, kind),
    )
    all_rows.extend(rows)
    summaries["dqn"] = summary
    with open(os.path.join(kind_dir, "compare_summary.json"), "w") as handle:
        json.dump(summaries, handle, indent=2)
    with open(os.path.join(kind_dir, "eval_steps.json"), "w") as handle:
        json.dump(all_rows, handle)
    write_table(summaries, os.path.join(kind_dir, "compare_table.csv"))
    with open(os.path.join(args.output_folder, "compare_summary.json"), "w") as handle:
        json.dump({kind: summaries}, handle, indent=2)


if __name__ == "__main__":
    main()
