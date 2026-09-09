"""
Stub Gymnasium env for 5G LENA (NR) Traffic Steering.

Mirrors TrafficSteeringEnv control-file interface so ns-3
`NrGnbNetDevice::ReadControlFile` can ingest actions.
HO application inside LENA is not enabled yet (parse/log only on ns-3 side).
"""

from nsoran.ns_env import NsOranEnv
from gymnasium import spaces
import numpy as np


class LenaTrafficSteeringEnv(NsOranEnv):
    """Minimal TS env aimed at ns-3-dev + contrib/nr (LENA)."""

    def __init__(
        self,
        ns3_path: str,
        scenario_configuration: dict,
        output_folder: str,
        optimized: bool = False,
        scenario: str = "lena-oran-test",
        n_gnbs: int = 1,
        verbose: bool = False,
    ):
        super().__init__(
            ns3_path=ns3_path,
            scenario=scenario,
            scenario_configuration=scenario_configuration,
            output_folder=output_folder,
            optimized=optimized,
            control_header=["timestamp", "ueId", "nrCellId"],
            log_file="TsActions.txt",
            control_file="ts_actions_for_ns3.csv",
        )
        self.verbose = verbose
        self.n_gnbs = n_gnbs
        # Columns to be aligned with LENA uemeas/cellmeas (or wineslab-compatible exports) later
        self.columns_state = [
            "RRU.PrbUsedDl",
            "DRB.MeanActiveUeDl",
            "TB.TotNbrDlInitial",
        ]
        self.columns_reward = ["DRB.UEThpDl.UEID", "nrCellId"]

        n_ues = int(self.scenario_configuration.get("ues", 1))
        # 0 = no action; 1..n_gnbs = target cell id (gym encoding, mapped in _compute_action)
        self.observation_space = spaces.Box(
            shape=(max(n_ues * n_gnbs, 1), len(self.columns_state) + 1),
            low=-np.inf,
            high=np.inf,
            dtype=np.float64,
        )
        self.action_space = spaces.MultiDiscrete([n_gnbs + 1] * max(n_ues * n_gnbs, 1))

    def _compute_action(self, action) -> list[tuple]:
        """Gym action → list of (ueId, targetCellId) for ts_actions_for_ns3.csv."""
        action_list = []
        for ue_id, target_cell in enumerate(action):
            if target_cell != 0:
                # Match current LENA/TS CSV: ueId starting at 1, cellId as written by agent
                action_list.append((ue_id + 1, int(target_cell)))
        if self.verbose:
            print(f"[LenaTS] actions={action_list}")
        return action_list

    def _fill_datalake_usecase(self):
        pass

    def _get_obs(self):
        # Placeholder until LENA KPM files are wired into Datalake
        try:
            ue_kpms = self.datalake.read_kpms(self.last_timestamp, self.columns_state)
            self.observations = np.array(ue_kpms, dtype=np.float64)
            if self.observations.size == 0:
                self.observations = np.zeros(self.observation_space.shape, dtype=np.float64)
        except Exception:
            self.observations = np.zeros(self.observation_space.shape, dtype=np.float64)
        return self.observations

    def _compute_reward(self) -> float:
        # Stub reward; replace with LENA throughput / HO cost later
        return 0.0
