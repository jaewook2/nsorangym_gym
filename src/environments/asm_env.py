import csv
import os

import numpy as np
from gymnasium import spaces

from nsoran.ns_env import NsOranEnv


class AsmEnv(NsOranEnv):
    """Gymnasium env for ASM duty / TurnOff on scenario-asm-verify.

    Center LTE eNB, mmWave gNBs on each side, 30 randomly placed UEs
    (10 high / 10 mid / 10 low CBR). Action is one ASM mode per mmWave cell.

    State, per mmWave cell:
      one-hot ASM (1-6), nUeFrac, offeredFrac, satMean
    plus global nUeFracLte, lteFrac, lteDuty, appNorm, satMean, satP5.

    Kinds:
      on_off  action 0/1 -> ASM 1 / ASM 6 (true radio off, not ASM 5)
      duty    action 0-5 -> ASM 1-6

    Reward:
      r = [(1-omega)*EE_norm + omega*sat_p5] * (1-C_trans) * (1-beta*lteFrac)
    C_trans mean: 0 stay, 0.1 duty, 0.3 off, 0.5 on.
    """

    TPUT_COLUMNS = ("lteKbps", "mmwKbps", "lteFrac", "appKbps")
    ASM_DUTY = {1: 1.0, 2: 0.75, 3: 0.5, 4: 0.25, 5: 0.0, 6: 0.0}
    P_SLEEP = 0.15
    P_STATIC = 0.40
    P_TX = 0.60
    # LTE eNB stays on; TX scales with delivered LTE PDCP rate vs 20 MHz capacity.
    P_LTE_STATIC = 0.40
    P_LTE_TX = 0.60
    LTE_CAP_KBPS = 50 * 1000.0
    OFFERED_KBPS = 10 * 10000.0 + 10 * 2000.0 + 10 * 500.0

    def __init__(self, ns3_path: str, scenario_configuration: dict, output_folder: str,
                 optimized: bool, verbose: bool = False, skip_configuration: bool = False,
                 scenario: str = "scenario-asm-verify", omega: float = 0.5,
                 beta: float = 0.8, kind: str = "on_off"):
        cfg = {}
        for key, value in scenario_configuration.items():
            cfg[key] = list(value) if isinstance(value, list) else [value]
        n_gnb = int(cfg.get("nMmWaveEnb", [2])[0])
        n_ue = cfg.get("ues", cfg.get("nUe", [30]))
        cfg["nMmWaveEnb"] = [n_gnb]
        cfg["ues"] = n_ue
        cfg["nUe"] = n_ue
        cfg.setdefault("controlMode", ["asm"])
        cfg.setdefault("controlFileName", ["asm_actions_for_ns3.csv"])
        cfg.setdefault("controlFileNameAsm", ["asm_actions_for_ns3.csv"])
        cfg.setdefault("useSemaphores", [1])
        cfg.setdefault("scheduleAsmControl", [0])
        cfg.setdefault("flushAsmControlFile", [0])

        super().__init__(
            ns3_path=ns3_path,
            scenario=scenario,
            scenario_configuration=cfg,
            output_folder=output_folder,
            optimized=optimized,
            skip_configuration=skip_configuration,
            control_header=["timestamp", "cellId", "asmMode"],
            log_file="AsmActions.txt",
            control_file="asm_actions_for_ns3.csv",
        )
        self.verbose = verbose
        self.n_gnbs = n_gnb
        self.n_ues = int(self.scenario_configuration["ues"])
        self.mmwave_cell_ids = list(range(2, 2 + self.n_gnbs))
        self.last_action_modes = {}
        self.last_metrics = {}
        self.omega = float(np.clip(omega, 0.0, 1.0))
        self.beta = float(np.clip(beta, 0.0, 1.0))
        self.kind = str(kind)
        if self.kind not in ("on_off", "duty"):
            raise ValueError(f"unknown ASM kind {self.kind}")
        self.prev_modes = {}
        self.p_mmw_both_on = self.n_gnbs * (self.P_STATIC + self.P_TX)
        self.p_lte_idle = self.P_LTE_STATIC
        self.p_ref = self.p_mmw_both_on + self.p_lte_idle
        self.ee_ref = self.OFFERED_KBPS / self.p_ref
        state = []
        for cid in self.mmwave_cell_ids:
            state.extend(f"asm{cid}_{mode}" for mode in range(1, 7))
            state.extend((f"nUeFrac{cid}", f"offeredFrac{cid}", f"satMean{cid}"))
        state.extend(("nUeFracLte", "lteFrac", "lteDuty", "appNorm", "satMean", "satP5"))
        self.COLUMNS = tuple(state)
        self.observation_space = spaces.Box(
            shape=(len(self.COLUMNS),),
            low=0.0,
            high=1.0,
            dtype=np.float64,
        )
        n_choices = 2 if self.kind == "on_off" else 6
        self.action_space = spaces.MultiDiscrete([n_choices] * self.n_gnbs)

    def reset(self, **kwargs):
        self.prev_modes = {}
        return super().reset(**kwargs)

    def configure_and_build_ns3(self):
        if self.skip_configuration:
            return
        super().configure_and_build_ns3()

    def _modes_from_action(self, action) -> list[int]:
        action = np.asarray(action, dtype=int).reshape(-1)
        if self.kind == "on_off":
            return [1 if int(a) == 0 else 6 for a in action]
        return [int(a) + 1 for a in action]

    def _compute_action(self, action) -> list[tuple]:
        modes = self._modes_from_action(action)
        self.last_action_modes = {
            cell_id: mode for cell_id, mode in zip(self.mmwave_cell_ids, modes)
        }
        return [(cell_id, mode) for cell_id, mode in self.last_action_modes.items()]

    def _read_tput_row(self) -> dict:
        path = os.path.join(self.sim_path, "asm-path-tput.csv")
        row = {}
        if not os.path.exists(path):
            return row
        with open(path, "r") as handle:
            for item in csv.DictReader(handle):
                row = item
        return row

    def _read_last_applied_modes(self) -> dict[int, int]:
        path = os.path.join(self.sim_path, "asm-mode-changes.txt")
        applied = {}
        if not os.path.exists(path):
            return applied
        with open(path, "r") as handle:
            for item in csv.DictReader(handle):
                try:
                    applied[int(item["cellId"])] = int(item["newMode"])
                except (KeyError, TypeError, ValueError):
                    continue
        return applied

    def _parse_tput_modes(self, raw: str) -> dict[int, int]:
        modes = {}
        for part in (raw or "").split(";"):
            if ":" not in part:
                continue
            cell_s, mode_s = part.split(":", 1)
            try:
                modes[int(cell_s)] = int(float(mode_s))
            except ValueError:
                continue
        return modes

    def _cell_power(self, mode: int) -> float:
        if int(mode) == 6:
            return self.P_SLEEP
        duty = self.ASM_DUTY.get(int(mode), 1.0)
        return self.P_STATIC + self.P_TX * duty

    def _lte_power(self, lte_kbps: float) -> float:
        duty = float(np.clip(lte_kbps / max(self.LTE_CAP_KBPS, 1e-6), 0.0, 1.0))
        return self.P_LTE_STATIC + self.P_LTE_TX * duty, duty

    def _current_modes(self) -> dict[int, int]:
        modes = dict(self.last_metrics.get("tputModes") or {})
        modes.update(self.last_metrics.get("appliedModes") or {})
        if not modes:
            modes = {cid: 1 for cid in self.mmwave_cell_ids}
        for cid in self.mmwave_cell_ids:
            modes.setdefault(cid, 1)
        return modes

    def _read_latest_ue_rows(self) -> list[dict]:
        path = os.path.join(self.sim_path, "ue-app-sat.csv")
        if not os.path.exists(path):
            return []
        with open(path, "r") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return []
        latest_ms = None
        parsed = []
        for row in rows:
            try:
                ts = int(float(row["timeMs"]))
            except (KeyError, TypeError, ValueError):
                continue
            latest_ms = ts if latest_ms is None else max(latest_ms, ts)
            parsed.append((ts, row))
        return [row for ts, row in parsed if ts == latest_ms]

    def _read_latest_sats(self) -> list[float]:
        sats = []
        for row in self._read_latest_ue_rows():
            try:
                sats.append(float(row["sat"]))
            except (KeyError, TypeError, ValueError):
                continue
        return sats

    @staticmethod
    def _row_float(row: dict, key: str, default: float = 0.0) -> float:
        try:
            return float(row.get(key, default) or default)
        except (TypeError, ValueError):
            return default

    def _path_occupancy(self, rows: list[dict]) -> dict:
        n_ue = {cid: 0 for cid in self.mmwave_cell_ids}
        offered = {cid: 0.0 for cid in self.mmwave_cell_ids}
        sats = {cid: [] for cid in self.mmwave_cell_ids}
        n_lte = 0
        for row in rows:
            sat = self._row_float(row, "sat")
            offered_i = self._row_float(row, "offeredKbps")
            use_mmw = int(self._row_float(row, "useMmWave"))
            nr_cell = int(self._row_float(row, "servingNrCellId"))
            if use_mmw and nr_cell in n_ue:
                n_ue[nr_cell] += 1
                offered[nr_cell] += offered_i
                sats[nr_cell].append(sat)
            else:
                n_lte += 1
        return {"nUe": n_ue, "offered": offered, "sats": sats, "nUeLte": n_lte}

    def _get_obs(self) -> np.ndarray:
        row = self._read_tput_row()
        applied = self._read_last_applied_modes()
        tput_modes = self._parse_tput_modes(row.get("asmModes", ""))
        ue_rows = self._read_latest_ue_rows()
        occ = self._path_occupancy(ue_rows)
        all_sats = [self._row_float(item, "sat") for item in ue_rows]
        sat_mean = float(np.mean(all_sats)) if all_sats else 0.0
        sat_p5 = float(np.clip(np.percentile(all_sats, 5), 0.0, 1.0)) if all_sats else 0.0
        app_kbps = self._row_float(row, "appKbps")
        lte_kbps = self._row_float(row, "lteKbps")
        mmw_kbps = self._row_float(row, "mmwKbps")
        lte_frac = self._row_float(row, "lteFrac")
        lte_duty = float(np.clip(lte_kbps / max(self.LTE_CAP_KBPS, 1e-6), 0.0, 1.0))
        app_norm = float(np.clip(app_kbps / max(self.OFFERED_KBPS, 1e-6), 0.0, 1.0))
        n_total = max(self.n_ues, 1)

        values = []
        for cell_id in self.mmwave_cell_ids:
            mode = int(applied.get(cell_id, tput_modes.get(cell_id, 1)))
            one_hot = [0.0] * 6
            if 1 <= mode <= 6:
                one_hot[mode - 1] = 1.0
            n_ue = occ["nUe"].get(cell_id, 0)
            cell_sats = occ["sats"].get(cell_id, [])
            sat_cell = float(np.mean(cell_sats)) if cell_sats else 1.0
            values.extend(one_hot)
            values.extend((
                float(np.clip(n_ue / n_total, 0.0, 1.0)),
                float(np.clip(occ["offered"].get(cell_id, 0.0) / max(self.OFFERED_KBPS, 1e-6), 0.0, 1.0)),
                float(np.clip(sat_cell, 0.0, 1.0)),
            ))
        values.extend((
            float(np.clip(occ["nUeLte"] / n_total, 0.0, 1.0)),
            float(np.clip(lte_frac, 0.0, 1.0)),
            lte_duty,
            app_norm,
            float(np.clip(sat_mean, 0.0, 1.0)),
            sat_p5,
        ))
        obs = np.array(values, dtype=np.float64)
        self.last_metrics = {
            "timestamp": self.last_timestamp,
            "appliedModes": applied,
            "tputModes": tput_modes,
            "lteKbps": lte_kbps,
            "mmwKbps": mmw_kbps,
            "lteFrac": lte_frac,
            "appKbps": app_kbps,
            "nUe": occ["nUe"],
            "nUeLte": occ["nUeLte"],
            **{name: float(obs[i]) for i, name in enumerate(self.COLUMNS)},
        }
        return obs

    @staticmethod
    def _jain_index(values: list[float]) -> float:
        if not values:
            return 0.0
        arr = np.asarray(values, dtype=np.float64)
        arr = np.clip(arr, 0.0, 1.0)
        s1 = float(np.sum(arr))
        s2 = float(np.sum(arr * arr))
        if s2 <= 0.0:
            return 0.0
        return (s1 * s1) / (len(arr) * s2)

    @staticmethod
    def _mode_transition_cost(prev: int, new: int) -> float:
        prev = int(prev)
        new = int(new)
        if prev == new:
            return 0.0
        if prev == 6:
            return 0.5
        if new == 6:
            return 0.3
        return 0.1

    def _transition_cost(self, modes: dict[int, int]) -> tuple[float, dict[int, float]]:
        if not self.prev_modes:
            costs = {cid: 0.0 for cid in self.mmwave_cell_ids}
            return 0.0, costs
        costs = {}
        for cid in self.mmwave_cell_ids:
            prev = int(self.prev_modes.get(cid, modes.get(cid, 1)))
            new = int(modes.get(cid, 1))
            costs[cid] = self._mode_transition_cost(prev, new)
        mean_cost = float(np.mean(list(costs.values()))) if costs else 0.0
        return mean_cost, costs

    def _compute_reward(self) -> float:
        modes = self._current_modes()
        p_mmw = sum(self._cell_power(modes[cid]) for cid in self.mmwave_cell_ids)
        lte_kbps = float(self.last_metrics.get("lteKbps", 0.0) or 0.0)
        p_lte, lte_duty = self._lte_power(lte_kbps)
        power = p_mmw + p_lte
        app_kbps = float(self.last_metrics.get("appKbps", 0.0) or 0.0)
        ee = app_kbps / max(power, 1e-6)
        ee_norm = float(np.clip(ee / max(self.ee_ref, 1e-6), 0.0, 1.0))
        sats = self._read_latest_sats()
        sat_p5 = float(np.clip(np.percentile(sats, 5), 0.0, 1.0)) if sats else 0.0
        trans, trans_cell = self._transition_cost(modes)
        lte_frac = float(np.clip(self.last_metrics.get("lteFrac", 0.0) or 0.0, 0.0, 1.0))
        quality = (1.0 - self.omega) * ee_norm + self.omega * sat_p5
        reward = float(quality * (1.0 - trans) * (1.0 - self.beta * lte_frac))
        self.prev_modes = {cid: int(modes[cid]) for cid in self.mmwave_cell_ids}
        self.last_metrics.update({
            "power": power,
            "pMmw": p_mmw,
            "pLte": p_lte,
            "lteDuty": lte_duty,
            "ee": ee,
            "eeNorm": ee_norm,
            "satP5": sat_p5,
            "jain": self._jain_index(sats),
            "transCost": trans,
            "transCostCell": trans_cell,
            "ltePenalty": self.beta * lte_frac,
            "omega": self.omega,
            "beta": self.beta,
            "nSatSamples": len(sats),
            "reward": reward,
        })
        return reward
