import csv
import os

import numpy as np
from gymnasium import spaces

from nsoran.ns_env import NsOranEnv


class AsmEnv(NsOranEnv):
    """Gymnasium env for ASM on/off and sequential duty on scenario-asm-verify.

    Center LTE eNB, mmWave gNBs on each side. Default layout clusters UEs at
    each gNB (heavier left ~ASM 2, lighter right ~ASM 4) so both cells should
    stay on mmWave rather than dumping to LTE.

    State, per mmWave cell:
      one-hot ASM (1-6), nUeFrac, offeredFrac, satMean, rlcFrac
    plus global nUeFracLte, lteFrac, lteDuty, appNorm, satMean, satP5, rlcFracLte.
    rlcFrac is cell RLC occupancy vs ~1% of MaxTxBuffer (busy if queues build).
    PRB usage is not in the observation: mmWave DU KPMs stay 0 in this stack.

    Kinds:
      on_off     action 0/1 -> mode 1 / mode 6, no sequential constraint
      duty_mode  action 0-5 -> duty modes 1-6, each step may change by at most ±1.
                 Symbol duty is 100/75/50/25/0%, and mode 6 turns the cell off.
                 duty is a previous name for duty_mode.
      asm        action 0-5 -> target Active, ASM1, ASM2, ASM3, ASM4, OFF.
                 ns-3 applies the target only while Stable. Direct jumps.
                 Both cells may be OFF.

    on_off and duty_mode cannot turn every mmWave cell off at once: the heavier
    (or last-on) cell is forced to stay on.

    Reward:
      r = [(1-omega)*EE_norm + omega*sat_p5] * (1-C_trans) * (1-beta*lteUse)
          * (1-bothOffPenalty if all cells ASM 6)
      omega=0 is EE only (satP5 still logged, not in r).
    C_trans: duty_mode mean of per-cell 0 stay / 0.1 duty change / 0.3 off / 0.5 on.
             on_off skips C_trans.
    Reset holds all mmWave cells at ASM 1 for warmup_s (0.5s) so traffic
    is up before PPO or any baseline acts; simTime is extended by that.
    init_modes / initAsmModes, when set, is applied once after warmup
    (ASM mode per cell) and becomes that run's start state.
    """

    TPUT_COLUMNS = ("lteKbps", "mmwKbps", "lteFrac", "appKbps")
    ASM_DUTY = {1: 1.0, 2: 0.75, 3: 0.5, 4: 0.25, 5: 0.0, 6: 0.0}
    P_SLEEP = 0.15
    P_STATIC = 0.40
    P_TX = 0.60
    # LTE eNB stays on; TX scales with delivered LTE PDCP rate vs 20 MHz capacity.
    P_LTE_STATIC = 0.40
    P_LTE_TX = 0.60
    LTE_CAP_PER_100RB_KBPS = 50 * 1000.0
    LTE_CAP_KBPS = LTE_CAP_PER_100RB_KBPS * 6 / 100.0
    OFFERED_KBPS = 15 * 2000.0 + 15 * 600.0
    KIND_ALIASES = {
        "on_off": "on_off",
        "onoff": "on_off",
        "onoff_ppo": "on_off",
        "duty_mode": "duty_mode",
        "asm_ppo": "duty_mode",
        "duty": "duty_mode",
        "asm": "asm",
    }

    def __init__(self, ns3_path: str, scenario_configuration: dict, output_folder: str,
                 optimized: bool, verbose: bool = False, skip_configuration: bool = False,
                 scenario: str = "scenario-asm-verify", omega: float = 0.0,
                 beta: float = 1.0, kind: str = "on_off", sequential: bool | None = None,
                 both_off_penalty: float = 1.0, init_modes: list | None = None,
                 left_on_bonus: float = 0.0):
        cfg = {}
        for key, value in scenario_configuration.items():
            cfg[key] = list(value) if isinstance(value, list) else [value]
        n_gnb = int(cfg.get("nMmWaveEnb", [2])[0])
        n_ue = cfg.get("ues", cfg.get("nUe", [30]))
        cfg["nMmWaveEnb"] = [n_gnb]
        cfg["ues"] = n_ue
        cfg["nUe"] = n_ue
        normalized_kind = self.KIND_ALIASES.get(str(kind), str(kind))
        if normalized_kind == "asm":
            cfg["asmStateMachine"] = [1]
        cfg.setdefault("controlMode", ["asm"])
        cfg.setdefault("controlFileName", ["asm_actions_for_ns3.csv"])
        cfg.setdefault("controlFileNameAsm", ["asm_actions_for_ns3.csv"])
        cfg.setdefault("useSemaphores", [1])
        cfg.setdefault("scheduleAsmControl", [0])
        cfg.setdefault("flushAsmControlFile", [0])
        cfg.setdefault("uePlacement", ["cluster"])
        cfg.setdefault("ueClusterRadius", [25])
        cfg.setdefault("rateLeft", ["2Mbps"])
        cfg.setdefault("rateRight", ["600kbps"])
        cfg.setdefault("lteBandwidthRbs", [6])
        period = float(self._cfg_first(cfg, "indicationPeriodicity", 0.1))
        warmup_s = 0.5
        warmup_steps = int(round(warmup_s / max(period, 1e-9)))
        cfg_init = cfg.pop("initAsmModes", None)
        if init_modes is None and cfg_init is not None:
            init_modes = [int(m) for m in cfg_init]
        elif init_modes is not None:
            init_modes = [int(m) for m in init_modes]
        init_s = period if init_modes else 0.0
        sim0 = float(self._cfg_first(cfg, "simTime", 3.0))
        cfg["simTime"] = [sim0 + warmup_s + init_s + period]

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
        self.left_on_bonus = float(left_on_bonus)
        self.beta = float(np.clip(beta, 0.0, 1.0))
        self.both_off_penalty = float(np.clip(both_off_penalty, 0.0, 1.0))
        self.kind = self.KIND_ALIASES.get(str(kind), str(kind))
        if self.kind not in ("on_off", "duty_mode", "asm"):
            raise ValueError(f"unknown ASM kind {kind}")
        self.sequential = (
            bool(self.kind == "duty_mode") if sequential is None else bool(sequential)
        )
        if self.kind == "asm":
            self.both_off_penalty = 0.0
        self.prev_modes = {}
        self.offered_kbps = self._offered_from_cfg(cfg, self.n_gnbs, self.n_ues)
        lte_rbs = int(self._cfg_first(cfg, "lteBandwidthRbs", 6))
        self.lte_cap_kbps = self.LTE_CAP_PER_100RB_KBPS * (lte_rbs / 100.0)
        max_rlc = float(int(self._cfg_first(cfg, "bufferSize", 10))) * 1024.0 * 1024.0
        self.rlc_busy_bytes = max(50_000.0, max_rlc * 0.01)
        if self.kind == "asm":
            tx_dbm = float(self._cfg_first(cfg, "txPower", 30.0))
            tx_w = 10 ** ((tx_dbm - 30.0) / 10.0)
            active_w = (
                tx_w
                + float(self._cfg_first(cfg, "asmRfPower", 0.0))
                + float(self._cfg_first(cfg, "asmBbPower", 0.0))
                + float(self._cfg_first(cfg, "asmControlPower", 0.0))
            )
            self.p_mmw_both_on = self.n_gnbs * active_w
        else:
            self.p_mmw_both_on = self.n_gnbs * (self.P_STATIC + self.P_TX)
        self.p_lte_idle = self.P_LTE_STATIC
        self.p_ref = self.p_mmw_both_on + self.p_lte_idle
        self.ee_ref = self.offered_kbps / self.p_ref
        state = []
        for cid in self.mmwave_cell_ids:
            state.extend(f"asm{cid}_{mode}" for mode in range(1, 7))
            if self.kind == "asm":
                state.extend((f"phase{cid}_stable", f"phase{cid}_deact", f"phase{cid}_wake"))
                state.extend(f"target{cid}_{mode}" for mode in range(1, 7))
            state.extend((f"nUeFrac{cid}", f"offeredFrac{cid}", f"satMean{cid}",
                          f"rlcFrac{cid}"))
        state.extend(("nUeFracLte", "lteFrac", "lteDuty", "appNorm", "satMean", "satP5",
                      "rlcFracLte"))
        self.COLUMNS = tuple(state)
        self.observation_space = spaces.Box(
            shape=(len(self.COLUMNS),),
            low=0.0,
            high=1.0,
            dtype=np.float64,
        )
        n_choices = 2 if self.kind == "on_off" else 6
        self.action_space = spaces.MultiDiscrete([n_choices] * self.n_gnbs)
        self.warmup_s = warmup_s
        self.warmup_steps = warmup_steps
        self.init_modes = init_modes

    @staticmethod
    def _cfg_first(cfg: dict, key: str, default):
        value = cfg.get(key, default)
        if isinstance(value, (list, tuple)):
            return value[0] if value else default
        return value

    @staticmethod
    def _parse_kbps(rate) -> float:
        text = str(rate).strip().lower().replace(" ", "")
        if text.endswith("gbps"):
            return float(text[:-4]) * 1e6
        if text.endswith("mbps"):
            return float(text[:-4]) * 1000.0
        if text.endswith("kbps"):
            return float(text[:-4])
        if text.endswith("bps"):
            return float(text[:-3]) / 1000.0
        return float(text)

    @classmethod
    def _offered_from_cfg(cls, cfg: dict, n_gnb: int, n_ue: int) -> float:
        placement = str(cls._cfg_first(cfg, "uePlacement", "cluster"))
        if placement != "random":
            left = cls._parse_kbps(cls._cfg_first(cfg, "rateLeft", "2Mbps"))
            right = cls._parse_kbps(cls._cfg_first(cfg, "rateRight", "600kbps"))
            sides_raw = cls._cfg_first(cfg, "ueClusterSidesOnly", 0)
            if isinstance(sides_raw, str):
                sides_only = sides_raw.strip().lower() in ("1", "true", "yes")
            else:
                sides_only = bool(sides_raw)
            if n_gnb <= 2 or sides_only:
                n_left = n_ue // 2
                n_right = n_ue - n_left
                return float(n_left * left + n_right * right)
            mid = cls._parse_kbps(cls._cfg_first(cfg, "rateMid", "2Mbps"))
            n_each = n_ue // max(n_gnb, 1)
            rem = n_ue % max(n_gnb, 1)
            total = 0.0
            for g in range(n_gnb):
                n_this = n_each + (1 if g < rem else 0)
                if g == 0:
                    total += n_this * left
                elif g + 1 == n_gnb:
                    total += n_this * right
                else:
                    total += n_this * mid
            return float(total)
        high = cls._parse_kbps(cls._cfg_first(cfg, "rateHigh", "10Mbps"))
        mid = cls._parse_kbps(cls._cfg_first(cfg, "rateMid", "2Mbps"))
        low = cls._parse_kbps(cls._cfg_first(cfg, "rateLow", "500kbps"))
        n_each = n_ue // 3
        n_high = n_each + (1 if n_ue % 3 > 0 else 0)
        n_mid = n_each + (1 if n_ue % 3 > 1 else 0)
        n_low = n_ue - n_high - n_mid
        return float(n_high * high + n_mid * mid + n_low * low)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        all_on = np.zeros(self.n_gnbs, dtype=np.int64)
        terminated = truncated = False
        for _ in range(self.warmup_steps):
            obs, _reward, terminated, truncated, info = super().step(all_on)
            if terminated or truncated:
                break
        if not (terminated or truncated) and self.init_modes:
            # Empty prev_modes so ASM can jump straight to the start duty.
            self.prev_modes = {}
            obs, _reward, terminated, truncated, info = super().step(
                self._action_from_modes(self.init_modes)
            )
        else:
            self.prev_modes = {cid: 1 for cid in self.mmwave_cell_ids}
        return obs, info

    def _action_from_modes(self, modes) -> np.ndarray:
        if len(modes) != self.n_gnbs:
            raise ValueError(f"initAsmModes needs {self.n_gnbs} entries, got {modes}")
        action = []
        for mode in modes:
            mode = int(mode)
            if self.kind == "on_off":
                action.append(0 if mode != 6 else 1)
            else:
                action.append(int(np.clip(mode, 1, 6)) - 1)
        return np.asarray(action, dtype=np.int64)

    def configure_and_build_ns3(self):
        if self.skip_configuration:
            return
        super().configure_and_build_ns3()

    def _keep_on_index(self) -> int:
        n_ue = (self.last_metrics or {}).get("nUe") or {}
        if n_ue:
            counts = [int(n_ue.get(cid, 0)) for cid in self.mmwave_cell_ids]
            if max(counts) > 0:
                return int(np.argmax(counts))
        for i, cid in enumerate(self.mmwave_cell_ids):
            if int(self.prev_modes.get(cid, 1)) != 6:
                return i
        return 0

    def _prevent_both_off(self, modes: list[int]) -> list[int]:
        modes = [int(m) for m in modes]
        if len(modes) < 2 or not all(m == 6 for m in modes):
            return modes
        keep = self._keep_on_index()
        cid = self.mmwave_cell_ids[keep]
        prev = int(self.prev_modes.get(cid, 1))
        if self.kind == "on_off":
            stay = 1
        elif self.sequential:
            stay = 5 if prev in (5, 6) else (prev if prev != 6 else 5)
        else:
            stay = 1 if prev == 6 else prev
        modes[keep] = stay
        return modes

    def _modes_from_action(self, action) -> list[int]:
        action = np.asarray(action, dtype=int).reshape(-1)
        if self.kind == "on_off":
            requested = [1 if int(a) == 0 else 6 for a in action]
        else:
            requested = [int(np.clip(int(a), 0, 5)) + 1 for a in action]
        if self.sequential:
            applied = []
            for cell_id, req in zip(self.mmwave_cell_ids, requested):
                if cell_id not in self.prev_modes:
                    applied.append(int(req))
                    continue
                prev = int(self.prev_modes[cell_id])
                lo = max(1, prev - 1)
                hi = min(6, prev + 1)
                applied.append(int(np.clip(req, lo, hi)))
            requested = applied
        if self.kind == "asm":
            return requested
        return self._prevent_both_off(requested)

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

    def _parse_asm_state(self, raw: str) -> dict[int, tuple[int, int]]:
        state = {}
        for part in (raw or "").split(";"):
            bits = part.split(":")
            if len(bits) < 3:
                continue
            try:
                state[int(bits[0])] = (int(float(bits[1])), int(float(bits[2])))
            except ValueError:
                continue
        return state

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
        duty = float(np.clip(lte_kbps / max(self.lte_cap_kbps, 1e-6), 0.0, 1.0))
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

    def _read_latest_du_rows(self, cell_id: int) -> list[dict]:
        path = os.path.join(self.sim_path, f"du-cell-{cell_id}.txt")
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
                ts = int(float(row["timestamp"]))
            except (KeyError, TypeError, ValueError):
                continue
            latest_ms = ts if latest_ms is None else max(latest_ms, ts)
            parsed.append((ts, row))
        return [row for ts, row in parsed if ts == latest_ms]

    def _read_kpm_asm(self):
        """Per-cell ASM KPMs from du-cell-*.txt. Power is read once per cell."""
        modes = {}
        state = {}
        power = 0.0
        for cell_id in self.mmwave_cell_ids:
            rows = self._read_latest_du_rows(cell_id)
            if not rows or "ASM.Mode" not in rows[0]:
                return None
            head = rows[0]
            mode = int(self._row_float(head, "ASM.Mode"))
            phase = int(self._row_float(head, "ASM.Phase"))
            target = int(self._row_float(head, "ASM.Target"))
            modes[cell_id] = mode
            state[cell_id] = (phase, target if target else mode)
            power += self._row_float(head, "ASM.PowerW")
        return modes, state, power

    def _du_occupancy(self, cell_id: int) -> tuple[float, float]:
        rows = self._read_latest_du_rows(cell_id)
        if not rows:
            return 0.0, 0.0
        head = rows[0]
        buf = self._row_float(head, "DRB.BufferSize.Qos")
        if buf <= 0.0:
            buf = sum(self._row_float(row, "DRB.BufferSize.Qos.UEID") for row in rows)
        prb_used = self._row_float(head, "RRU.PrbUsedDl")
        prb_avail = self._row_float(head, "dlAvailablePrbs")
        prb_pct = self._row_float(head, "dlPrbUsage")
        if prb_used <= 0.0:
            prb_used = sum(self._row_float(row, "RRU.PrbUsedDl.UEID") for row in rows)
        if prb_pct > 0.0:
            prb_frac = prb_pct / 100.0
        else:
            prb_frac = prb_used / max(prb_avail, 1.0)
        rlc_frac = buf / max(self.rlc_busy_bytes, 1.0)
        return (
            float(np.clip(rlc_frac, 0.0, 1.0)),
            float(np.clip(prb_frac, 0.0, 1.0)),
        )

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
        asm_state = self._parse_asm_state(row.get("asmState", ""))
        kpm = self._read_kpm_asm() if self.kind == "asm" else None
        if kpm is not None:
            kpm_modes, kpm_state, kpm_power = kpm
        else:
            kpm_modes, kpm_state, kpm_power = {}, {}, None
        ue_rows = self._read_latest_ue_rows()
        occ = self._path_occupancy(ue_rows)
        all_sats = [self._row_float(item, "sat") for item in ue_rows]
        sat_mean = float(np.mean(all_sats)) if all_sats else 0.0
        sat_p5 = float(np.clip(np.percentile(all_sats, 5), 0.0, 1.0)) if all_sats else 0.0
        app_kbps = self._row_float(row, "appKbps")
        lte_kbps = self._row_float(row, "lteKbps")
        mmw_kbps = self._row_float(row, "mmwKbps")
        lte_frac = self._row_float(row, "lteFrac")
        lte_duty = float(np.clip(lte_kbps / max(self.lte_cap_kbps, 1e-6), 0.0, 1.0))
        app_norm = float(np.clip(app_kbps / max(self.offered_kbps, 1e-6), 0.0, 1.0))
        n_total = max(self.n_ues, 1)

        values = []
        for cell_id in self.mmwave_cell_ids:
            if cell_id in kpm_modes:
                mode = int(kpm_modes[cell_id])
            else:
                mode = int(applied.get(cell_id, tput_modes.get(cell_id, 1)))
            one_hot = [0.0] * 6
            if 1 <= mode <= 6:
                one_hot[mode - 1] = 1.0
            if cell_id in kpm_state:
                phase, target = kpm_state[cell_id]
            else:
                phase, target = asm_state.get(cell_id, (0, mode))
            phase_hot = [0.0, 0.0, 0.0]
            if 0 <= int(phase) <= 2:
                phase_hot[int(phase)] = 1.0
            target_hot = [0.0] * 6
            if 1 <= int(target) <= 6:
                target_hot[int(target) - 1] = 1.0
            n_ue = occ["nUe"].get(cell_id, 0)
            cell_sats = occ["sats"].get(cell_id, [])
            sat_cell = float(np.mean(cell_sats)) if cell_sats else 1.0
            rlc_frac, _prb_frac = self._du_occupancy(cell_id)
            values.extend(one_hot)
            if self.kind == "asm":
                values.extend(phase_hot)
                values.extend(target_hot)
            values.extend((
                float(np.clip(n_ue / n_total, 0.0, 1.0)),
                float(np.clip(occ["offered"].get(cell_id, 0.0) / max(self.offered_kbps, 1e-6), 0.0, 1.0)),
                float(np.clip(sat_cell, 0.0, 1.0)),
                rlc_frac,
            ))
        lte_rlc, _lte_prb = self._du_occupancy(1)
        values.extend((
            float(np.clip(occ["nUeLte"] / n_total, 0.0, 1.0)),
            float(np.clip(lte_frac, 0.0, 1.0)),
            lte_duty,
            app_norm,
            float(np.clip(sat_mean, 0.0, 1.0)),
            sat_p5,
            lte_rlc,
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
            "asmPowerW": kpm_power if kpm_power is not None else self._row_float(row, "asmPowerW"),
            "asmState": asm_state,
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

    def _mode_transition_cost(self, prev: int, new: int) -> float:
        prev = int(prev)
        new = int(new)
        if prev == new:
            return 0.0
        if self.kind == "on_off" or not self.sequential:
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
        if not costs:
            return 0.0, costs
        if self.kind == "on_off":
            return float(np.max(list(costs.values()))), costs
        return float(np.mean(list(costs.values()))), costs

    def _compute_reward(self) -> float:
        modes = self._current_modes()
        if self.kind == "asm":
            p_mmw = float(self.last_metrics.get("asmPowerW", 0.0) or 0.0)
        else:
            p_mmw = sum(self._cell_power(modes[cid]) for cid in self.mmwave_cell_ids)
        lte_kbps = float(self.last_metrics.get("lteKbps", 0.0) or 0.0)
        p_lte, lte_duty = self._lte_power(lte_kbps)
        power = p_mmw + p_lte
        app_kbps = float(self.last_metrics.get("appKbps", 0.0) or 0.0)
        ee = app_kbps / max(power, 1e-6)
        # all_on already saturates offered/p_ref; allow EE>1 so sleep/dim can win.
        ee_norm = float(np.clip(ee / max(self.ee_ref, 1e-6), 0.0, 4.0))
        sats = self._read_latest_sats()
        sat_p5 = float(np.clip(np.percentile(sats, 5), 0.0, 1.0)) if sats else 0.0
        trans, trans_cell = self._transition_cost(modes)
        lte_frac = float(np.clip(self.last_metrics.get("lteFrac", 0.0) or 0.0, 0.0, 1.0))
        n_ue_lte = float(np.clip(self.last_metrics.get("nUeFracLte", 0.0) or 0.0, 0.0, 1.0))
        if n_ue_lte <= 0.0:
            n_ue_lte = float(self.last_metrics.get("nUeLte", 0) or 0) / max(self.n_ues, 1)
        lte_use = max(lte_frac, n_ue_lte)
        n_off = sum(1 for cid in self.mmwave_cell_ids if int(modes.get(cid, 1)) == 6)
        both_off = n_off >= self.n_gnbs
        quality = (1.0 - self.omega) * ee_norm + self.omega * sat_p5
        reward = float(quality * (1.0 - trans) * (1.0 - self.beta * lte_use))
        if both_off:
            reward *= (1.0 - self.both_off_penalty)
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
            "ltePenalty": self.beta * lte_use,
            "lteUse": lte_use,
            "nOff": n_off,
            "bothOff": both_off,
            "omega": self.omega,
            "beta": self.beta,
            "nSatSamples": len(sats),
            "reward": reward,
        })
        if self.left_on_bonus and int(modes.get(self.mmwave_cell_ids[0], 1)) != 6:
            return float(reward + self.left_on_bonus)
        return reward
