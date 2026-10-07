#!/usr/bin/env python3
"""Train PPO on AsmEnv until the policy settles, then compare baselines.

Two techniques (separate action spaces, same state/reward):
  on_off  ASM 1 vs 6 per mmWave cell, no sequential constraint
  asm     ASM 1-6 per mmWave cell, each step may change by at most ±1
"""
import argparse
import json
import os
import sys
import time

import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from environments.asm_env import AsmEnv


class SafeNs3Wrapper(gym.Wrapper):
    def _clean_obs(self, obs):
        arr = np.asarray(obs, dtype=np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)

    @property
    def last_metrics(self):
        return dict(getattr(self.unwrapped, "last_metrics", {}) or {})

    def reset(self, **kwargs):
        try:
            obs, info = self.env.reset(**kwargs)
            return self._clean_obs(obs), info
        except Exception as exc:
            print(f"[SafeNs3] reset failed: {exc}", flush=True)
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, {"error": str(exc)}

    def step(self, action):
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
            reward = float(np.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0))
            return self._clean_obs(obs), reward, terminated, truncated, info
        except Exception as exc:
            print(f"[SafeNs3] step failed: {exc}", flush=True)
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, False, True, {"error": str(exc)}


def parse_args():
    parser = argparse.ArgumentParser(description="PPO + baseline comparison on ASM")
    parser.add_argument(
        "--config",
        default=os.path.join(
            os.path.dirname(__file__),
            "../src/environments/scenario_configurations/asm_use_case.json",
        ),
    )
    parser.add_argument("--kind", choices=("on_off", "asm", "duty", "both"), default="both")
    parser.add_argument("--output_folder", default="examples/output_asm_s1_onoff_overlap")
    parser.add_argument(
        "--omega",
        type=float,
        default=0.0,
        help="Reward mix: 0=EE only, 1=satP5 only (default 0).",
    )
    parser.add_argument(
        "--ns3_path",
        default=os.environ.get(
            "NS3_PATH",
            "/home/pknu-boo/nsorangym_mmwave_test_0914/nsorangym_mmwave",
        ),
    )
    parser.add_argument("--simTime", type=float, default=None)
    parser.add_argument("--total_timesteps", type=int, default=256)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--n_eval_episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--until_converge", action="store_true", default=True)
    parser.add_argument("--no_until_converge", action="store_false", dest="until_converge")
    parser.add_argument("--chunk_timesteps", type=int, default=256)
    parser.add_argument("--max_timesteps", type=int, default=2048)
    parser.add_argument("--ent_frac", type=float, default=0.8)
    parser.add_argument("--plateau_rel", type=float, default=0.05)
    parser.add_argument("--plateau_chunks", type=int, default=2)
    parser.add_argument("--skip_configuration", action="store_true", default=True)
    parser.add_argument("--no_skip_configuration", action="store_false", dest="skip_configuration")
    parser.add_argument("--optimized", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument(
        "--init_all_on",
        action="store_true",
        help="After warmup, leave every cell ON instead of a per-policy start mode.",
    )
    parser.add_argument(
        "--left_on_bonus",
        type=float,
        default=0.0,
        help="Added to the training reward when the leftmost mmWave cell is ON. Eval stays 0.",
    )
    parser.add_argument(
        "--net_arch",
        default="64,64",
        help="Hidden widths for both the policy and value MLPs, comma-separated.",
    )
    return parser.parse_args()


def load_config(args):
    with open(args.config) as handle:
        cfg = json.load(handle)
    if args.simTime is not None:
        cfg["simTime"] = [args.simTime]
    return cfg


def _normalize_kind(kind: str) -> str:
    if kind in ("duty", "asm_ppo", "asm"):
        return "asm"
    if kind in ("onoff", "onoff_ppo", "on_off"):
        return "on_off"
    return kind


def make_env(args, cfg, log_dir, kind=None, rank: int = 0, init_modes=None, left_on_bonus: float = 0.0):
    kind = _normalize_kind(kind or args.kind)
    def _init():
        env = AsmEnv(
            ns3_path=args.ns3_path,
            scenario_configuration=cfg,
            output_folder=os.path.join(log_dir, "ns3", f"env{rank}"),
            optimized=args.optimized,
            skip_configuration=args.skip_configuration,
            kind=kind,
            omega=args.omega,
            init_modes=init_modes,
            left_on_bonus=left_on_bonus,
        )
        env = SafeNs3Wrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return _init


def unwrap_metrics(env) -> dict:
    cur = env
    while hasattr(cur, "env"):
        cur = cur.env
    return dict(getattr(cur, "last_metrics", {}) or {})


def vec_metrics(vec) -> dict:
    cur = vec
    while hasattr(cur, "venv"):
        cur = cur.venv
    env = cur.envs[0]
    if hasattr(env, "last_metrics"):
        return dict(env.last_metrics or {})
    return unwrap_metrics(env)


class SequentialRandom:
    """Unconstrained 0/1 for on_off; adjacent ASM walk for asm."""

    def __init__(self, n_gnbs: int, n_choices: int):
        self.n_gnbs = n_gnbs
        self.n_choices = n_choices
        self.prev = None

    def reset(self):
        self.prev = None

    def __call__(self, _obs, rng):
        if self.n_choices <= 2:
            action = rng.integers(0, self.n_choices, size=self.n_gnbs, dtype=np.int64)
            if self.n_gnbs >= 2 and np.all(action == self.n_choices - 1):
                action[int(rng.integers(0, self.n_gnbs))] = 0
            return action
        if self.prev is None:
            self.prev = np.zeros(self.n_gnbs, dtype=np.int64)
            return self.prev.copy()
        nxt = np.array(
            [
                int(np.clip(int(prev) + int(rng.integers(-1, 2)), 0, self.n_choices - 1))
                for prev in self.prev
            ],
            dtype=np.int64,
        )
        if self.n_gnbs >= 2 and np.all(nxt == self.n_choices - 1):
            nxt[int(rng.integers(0, self.n_gnbs))] = self.n_choices - 2
        self.prev = nxt
        return nxt.copy()


def policy_actions(n_gnbs: int, kind: str):
    kind = _normalize_kind(kind)
    n_choices = 2 if kind == "on_off" else 6
    off_idx = n_choices - 1
    all_on = np.zeros(n_gnbs, dtype=np.int64)
    policies = {
        "all_on": lambda _obs, _rng: all_on.copy(),
    }
    if n_gnbs == 2:
        left = np.array([0, off_idx], dtype=np.int64)
        right = np.array([off_idx, 0], dtype=np.int64)
        policies["one_on_cell2"] = lambda _obs, _rng: left.copy()
        policies["one_on_cell3"] = lambda _obs, _rng: right.copy()
        if kind == "asm":
            policies["asm2_4"] = lambda _obs, _rng: np.array([1, 3], dtype=np.int64)
            policies["asm4_2"] = lambda _obs, _rng: np.array([3, 1], dtype=np.int64)
    elif n_gnbs == 3:
        off = off_idx
        policies["left_only"] = lambda _obs, _rng, off=off: np.array([0, off, off], dtype=np.int64)
        policies["center_only"] = lambda _obs, _rng, off=off: np.array([off, 0, off], dtype=np.int64)
        policies["right_only"] = lambda _obs, _rng, off=off: np.array([off, off, 0], dtype=np.int64)
        policies["sides_on"] = lambda _obs, _rng, off=off: np.array([0, off, 0], dtype=np.int64)
        policies["left_center"] = lambda _obs, _rng, off=off: np.array([0, 0, off], dtype=np.int64)
        policies["right_center"] = lambda _obs, _rng, off=off: np.array([off, 0, 0], dtype=np.int64)
    policies["random"] = SequentialRandom(n_gnbs, n_choices)
    return policies


def policy_init_modes(name: str, n_gnbs: int, kind: str):
    """ASM modes applied once after warmup, before the scored episode."""
    kind = _normalize_kind(kind)
    off = 6 if kind == "on_off" else 6
    on = 1
    if name == "ppo":
        if kind == "on_off" and n_gnbs == 3:
            return [off, off, on]
        if kind == "on_off" and n_gnbs == 2:
            return [on, off]
        if kind == "asm" and n_gnbs == 2:
            return [2, 4]
        return [on] * n_gnbs
    if name in ("all_on", "random"):
        return [on] * n_gnbs
    if n_gnbs == 2:
        if name == "one_on_cell2":
            return [on, off]
        if name == "one_on_cell3":
            return [off, on]
        if name == "asm2_4":
            return [2, 4]
        if name == "asm4_2":
            return [4, 2]
    if n_gnbs == 3:
        table = {
            "left_only": [on, off, off],
            "center_only": [off, on, off],
            "right_only": [off, off, on],
            "sides_on": [on, off, on],
            "left_center": [on, on, off],
            "right_center": [off, on, on],
        }
        if name in table:
            return table[name]
    return [on] * n_gnbs


def summarize_steps(rows: list[dict]) -> dict:
    if not rows:
        return {"n_steps": 0, "n_episodes": 0}
    keys = ("reward", "eeNorm", "satP5", "appKbps", "pMmw", "pLte", "power")
    out = {
        "n_steps": len(rows),
        "n_episodes": len({row["episode"] for row in rows}),
    }
    for key in keys:
        vals = [float(row[key]) for row in rows if row.get(key) is not None]
        out[key] = float(np.mean(vals)) if vals else None
        if vals:
            out[f"{key}_std"] = float(np.std(vals))
    return out


def run_policy(name, choose_action, args, cfg, log_dir, n_episodes, seed, kind, init_modes=None):
    from stable_baselines3.common.monitor import Monitor

    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(make_env(args, cfg, log_dir, kind, init_modes=init_modes)(), log_dir)
    rng = np.random.default_rng(seed)
    rows = []
    ep_rewards = []
    print(f"\n===== eval {name} episodes={n_episodes} =====", flush=True)
    try:
        for ep in range(n_episodes):
            obs, _info = env.reset()
            if hasattr(choose_action, "reset"):
                choose_action.reset()
            done = False
            step = 0
            ep_r = 0.0
            while not done:
                action = choose_action(obs, rng)
                obs, reward, terminated, truncated, _info = env.step(action)
                done = bool(terminated or truncated)
                metrics = unwrap_metrics(env)
                row = {
                    "policy": name,
                    "episode": ep,
                    "step": step,
                    "reward": float(reward),
                    "eeNorm": metrics.get("eeNorm"),
                    "satP5": metrics.get("satP5"),
                    "appKbps": metrics.get("appKbps"),
                    "pMmw": metrics.get("pMmw"),
                    "pLte": metrics.get("pLte"),
                    "power": metrics.get("power"),
                    "appliedModes": str(metrics.get("appliedModes")),
                }
                rows.append(row)
                ep_r += float(reward)
                step += 1
                print(
                    f"  {name} ep={ep} step={step} reward={reward:.3f} "
                    f"eeNorm={metrics.get('eeNorm', 0):.3f} satP5={metrics.get('satP5', 0):.3f} "
                    f"appKbps={metrics.get('appKbps', 0):.1f} modes={metrics.get('appliedModes')}",
                    flush=True,
                )
            ep_rewards.append(ep_r)
            print(f"  {name} episode {ep} return={ep_r:.3f} steps={step}", flush=True)
    finally:
        env.close()
    summary = summarize_steps(rows)
    summary["ep_return_mean"] = float(np.mean(ep_rewards)) if ep_rewards else None
    summary["ep_return_std"] = float(np.std(ep_rewards)) if ep_rewards else None
    return rows, summary


def eval_ppo(args, cfg, model_path, stats_path, log_dir, n_episodes, kind, init_modes=None):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    os.makedirs(log_dir, exist_ok=True)

    def _monitored():
        return Monitor(make_env(args, cfg, log_dir, kind, init_modes=init_modes)(), log_dir)

    vec = DummyVecEnv([_monitored])
    vec = VecNormalize.load(stats_path, vec)
    vec.training = False
    vec.norm_reward = False
    model = PPO.load(model_path, env=vec)
    rows = []
    ep_rewards = []
    print(f"\n===== eval ppo episodes={n_episodes} =====", flush=True)
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
                rows.append({
                    "policy": "ppo",
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
                })
                ep_r += reward
                step += 1
                print(
                    f"  ppo ep={ep} step={step} reward={reward:.3f} "
                    f"eeNorm={metrics.get('eeNorm', 0):.3f} satP5={metrics.get('satP5', 0):.3f} "
                    f"appKbps={metrics.get('appKbps', 0):.1f} modes={metrics.get('appliedModes')}",
                    flush=True,
                )
            ep_rewards.append(ep_r)
            print(f"  ppo episode {ep} return={ep_r:.3f} steps={step}", flush=True)
    finally:
        vec.close()
    summary = summarize_steps(rows)
    summary["ep_return_mean"] = float(np.mean(ep_rewards)) if ep_rewards else None
    summary["ep_return_std"] = float(np.std(ep_rewards)) if ep_rewards else None
    return rows, summary


def summarize_monitor(log_dir):
    import glob
    import pandas as pd

    paths = sorted(
        glob.glob(os.path.join(log_dir, "**", "*monitor*.csv"), recursive=True),
        key=os.path.getmtime,
    )
    if not paths:
        return {"n_episodes": 0}
    frames = []
    for path in paths:
        try:
            frames.append(pd.read_csv(path, skiprows=1))
        except Exception:
            continue
    if not frames:
        return {"n_episodes": 0}
    df = pd.concat(frames, ignore_index=True)
    if df.empty or "r" not in df.columns:
        return {"n_episodes": 0}
    tail = df.tail(max(1, min(10, len(df))))
    return {
        "n_episodes": int(len(df)),
        "ep_rew_mean": float(df["r"].mean()),
        "ep_rew_last10": float(tail["r"].mean()),
        "ep_len_mean": float(df["l"].mean()) if "l" in df.columns else None,
    }


def train_ppo(args, cfg, log_dir, kind):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    os.makedirs(log_dir, exist_ok=True)
    model_path = os.path.join(log_dir, f"ppo_{kind}.zip")
    stats_path = os.path.join(log_dir, "vecnormalize.pkl")
    n_envs = max(1, int(args.n_envs))
    rollout = args.n_steps * n_envs
    if rollout % args.batch_size != 0:
        raise ValueError(
            f"n_steps*n_envs={rollout} must be divisible by batch_size={args.batch_size}"
        )

    def _worker(rank: int):
        def _init():
            mon_dir = os.path.join(log_dir, f"env{rank}")
            os.makedirs(mon_dir, exist_ok=True)
            env = make_env(
                args,
                cfg,
                log_dir,
                kind,
                rank=rank,
                init_modes=None
                if args.init_all_on
                else policy_init_modes("ppo", int(cfg.get("nMmWaveEnb", [2])[0]), kind),
                left_on_bonus=float(args.left_on_bonus),
            )()
            return Monitor(env, mon_dir)
        return _init

    if n_envs == 1:
        vec = DummyVecEnv([_worker(0)])
    else:
        vec = SubprocVecEnv([_worker(i) for i in range(n_envs)], start_method="fork")
    vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    hidden = [int(part) for part in str(args.net_arch).split(",") if part.strip()]
    policy_kwargs = dict(net_arch=dict(pi=hidden, vf=hidden))
    model = PPO(
        "MlpPolicy",
        vec,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=3e-4,
        ent_coef=0.01,
        max_grad_norm=0.5,
        seed=args.seed,
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log=os.path.join(args.output_folder, "tb", kind),
    )
    ckpt = CheckpointCallback(
        save_freq=max(args.n_steps, 64),
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix=f"ppo_{kind}",
    )
    budget = args.max_timesteps if args.until_converge else args.total_timesteps
    chunk = args.chunk_timesteps if args.until_converge else args.total_timesteps
    print(
        f"\n===== train PPO {kind} n_envs={n_envs} budget={budget} chunk={chunk} "
        f"rollout={rollout} net={hidden} omega={args.omega} until_converge={args.until_converge} =====",
        flush=True,
    )
    t0 = time.time()
    elapsed = 0
    plateau_hits = 0
    prev_last10 = None
    first = True
    converged = False
    while elapsed < budget:
        n = min(chunk, budget - elapsed)
        model.learn(
            total_timesteps=n,
            callback=ckpt,
            progress_bar=False,
            reset_num_timesteps=first,
        )
        first = False
        elapsed += n
        model.save(model_path)
        vec.save(stats_path)
        stats = summarize_monitor(log_dir)
        last10 = stats.get("ep_rew_last10")
        if prev_last10 is not None and last10 is not None:
            rel = abs(last10 - prev_last10) / (abs(prev_last10) + 1e-6)
            if rel <= args.plateau_rel and last10 >= prev_last10 - 0.5:
                plateau_hits += 1
            else:
                plateau_hits = 0
        prev_last10 = last10
        print(
            f"===== {kind} extra={elapsed}/{budget} last10={last10} "
            f"plateau={plateau_hits} =====",
            flush=True,
        )
        if args.until_converge and plateau_hits >= args.plateau_chunks:
            converged = True
            print(f"===== converged {kind} at {elapsed} =====", flush=True)
            break
    train_s = time.time() - t0
    vec.close()
    print(f"saved {model_path} elapsed_s={train_s:.1f} converged={converged}", flush=True)
    return model_path, stats_path, train_s, converged, elapsed


def write_table(summaries: dict, path: str):
    keys = (
        "ep_return_mean",
        "reward",
        "eeNorm",
        "satP5",
        "appKbps",
        "pMmw",
        "pLte",
        "n_steps",
        "n_episodes",
    )
    lines = ["policy," + ",".join(keys)]
    for name, stats in summaries.items():
        row = [name]
        for key in keys:
            val = stats.get(key)
            row.append("" if val is None else f"{val:.6g}" if isinstance(val, float) else str(val))
        lines.append(",".join(row))
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n===== comparison =====", flush=True)
    print(f"{'policy':<16} {'return':>8} {'rew':>7} {'eeNorm':>7} {'satP5':>7} {'appKbps':>9}")
    for name, stats in summaries.items():
        def fmt(key, width, digits=3):
            val = stats.get(key)
            if val is None:
                return f"{'n/a':>{width}}"
            return f"{val:{width}.{digits}f}"
        print(
            f"{name:<16}{fmt('ep_return_mean', 8)}{fmt('reward', 7)}"
            f"{fmt('eeNorm', 7)}{fmt('satP5', 7)}{fmt('appKbps', 9, 1)}"
        )
    print(f"wrote {path}", flush=True)


def main():
    args = parse_args()
    os.environ.setdefault("NS3_SKIP_BUILD", "1")
    cfg = load_config(args)
    os.makedirs(args.output_folder, exist_ok=True)
    kinds = ("on_off", "asm") if args.kind == "both" else (_normalize_kind(args.kind),)
    all_summaries = {}

    for kind in kinds:
        print(f"\n########## {kind} ##########", flush=True)
        kind_dir = os.path.join(args.output_folder, kind)
        train_dir = os.path.join(kind_dir, "ppo")
        model_path = os.path.join(train_dir, f"ppo_{kind}.zip")
        stats_path = os.path.join(train_dir, "vecnormalize.pkl")
        all_rows = []
        summaries = {}
        do_train = (not args.eval_only) and (not args.skip_train)
        if do_train:
            model_path, stats_path, train_s, converged, steps = train_ppo(
                args, cfg, train_dir, kind
            )
            summaries["train"] = {
                "elapsed_s": train_s,
                "timesteps": steps,
                "converged": converged,
            }

        n_gnbs = int(cfg.get("nMmWaveEnb", [2])[0])
        policies = policy_actions(n_gnbs, kind)
        for name, choose in policies.items():
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

        if os.path.isfile(model_path) and os.path.isfile(stats_path):
            rows, summary = eval_ppo(
                args,
                cfg,
                model_path,
                stats_path,
                os.path.join(kind_dir, "eval_ppo"),
                args.n_eval_episodes,
                kind,
                init_modes=None if args.init_all_on else policy_init_modes("ppo", n_gnbs, kind),
            )
            all_rows.extend(rows)
            summaries["ppo"] = summary
        else:
            print(f"skip ppo eval: missing {model_path} or {stats_path}", flush=True)

        if "one_on_cell2" in summaries and "one_on_cell3" in summaries:
            left = summaries["one_on_cell2"]
            right = summaries["one_on_cell3"]
            merged = {}
            for key in ("ep_return_mean", "reward", "eeNorm", "satP5", "appKbps", "pMmw", "pLte"):
                vals = [left.get(key), right.get(key)]
                vals = [v for v in vals if v is not None]
                merged[key] = float(np.mean(vals)) if vals else None
            summaries["one_on"] = merged

        out_json = os.path.join(kind_dir, "compare_summary.json")
        with open(out_json, "w") as handle:
            json.dump(summaries, handle, indent=2)
        with open(os.path.join(kind_dir, "eval_steps.json"), "w") as handle:
            json.dump(all_rows, handle)
        write_table(summaries, os.path.join(kind_dir, "compare_table.csv"))
        all_summaries[kind] = summaries

    merged_path = os.path.join(args.output_folder, "compare_summary.json")
    with open(merged_path, "w") as handle:
        json.dump(all_summaries, handle, indent=2)
    print(f"wrote {merged_path}", flush=True)


if __name__ == "__main__":
    main()
