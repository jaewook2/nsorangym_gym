"""
Gymnasium env for 5G LENA (NR) Traffic Steering ↔ ns-o-ran-gym.

Control path matches wineslab (ts_actions_for_ns3.csv + POSIX semaphores).
HO application inside LENA is still parse/log-only.
"""

from copy import deepcopy

from nsoran.ns_env import NsOranEnv
from gymnasium import spaces
import numpy as np


def _first(value):
    return value[0] if isinstance(value, list) else value


class LenaTrafficSteeringEnv(NsOranEnv):
    """TS env for ns-3-dev + contrib/nr (LENA)."""

    def __init__(
        self,
        ns3_path: str,
        scenario_configuration: dict,
        output_folder: str,
        optimized: bool = False,
        scenario: str = "lena-oran-test",
        n_gnbs: int = 1,
        verbose: bool = False,
        skip_configuration: bool = True,
    ):
        raw = deepcopy(scenario_configuration)
        n_gnbs = int(_first(raw.get("gNbNum", n_gnbs)))
        if "ues" in raw:
            n_ues = int(_first(raw["ues"]))
        else:
            n_ues = int(_first(raw.get("ueNumPergNb", 1))) * n_gnbs
            raw["ues"] = [n_ues]

        # Keep `ues` for Datalake; ns-3 accepts it as a CLI alias now.
        super().__init__(
            ns3_path=ns3_path,
            scenario=scenario,
            scenario_configuration=raw,
            output_folder=output_folder,
            optimized=optimized,
            skip_configuration=skip_configuration,
            control_header=["timestamp", "ueId", "nrCellId"],
            log_file="TsActions.txt",
            control_file="ts_actions_for_ns3.csv",
        )
        self.verbose = verbose
        self.n_gnbs = n_gnbs
        self.n_ues = n_ues
        self.columns_state = [
            "RRU.PrbUsedDl",
            "DRB.MeanActiveUeDl",
            "TB.TotNbrDlInitial",
        ]
        self.columns_reward = ["DRB.UEThpDl.UEID", "nrCellId"]

        self.observation_space = spaces.Box(
            shape=(max(n_ues * n_gnbs, 1), len(self.columns_state) + 1),
            low=-np.inf,
            high=np.inf,
            dtype=np.float64,
        )
        self.action_space = spaces.MultiDiscrete([n_gnbs + 1] * max(n_ues * n_gnbs, 1))

    def _compute_action(self, action) -> list[tuple]:
        action_list = []
        for ue_id, target_cell in enumerate(action):
            if target_cell != 0:
                action_list.append((ue_id + 1, int(target_cell)))
        if self.verbose:
            print(f"[LenaTS] actions={action_list}")
        return action_list

    def _fill_datalake_usecase(self):
        pass

    def _get_obs(self):
        try:
            ue_kpms = self.datalake.read_kpms(self.last_timestamp, self.columns_state)
            self.observations = np.array(ue_kpms, dtype=np.float64)
            if self.observations.size == 0:
                self.observations = np.zeros(self.observation_space.shape, dtype=np.float64)
        except Exception:
            self.observations = np.zeros(self.observation_space.shape, dtype=np.float64)
        return self.observations

    def _compute_reward(self) -> float:
        return 0.0
